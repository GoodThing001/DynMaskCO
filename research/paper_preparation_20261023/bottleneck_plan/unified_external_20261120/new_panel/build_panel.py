"""Select low-overlap customer panels and independently audit their common feasible starts."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

from fetch_panel_sources import XTW, official_sources


HERE = Path(__file__).resolve().parent
STUDIES = HERE.parents[1]
PREVIOUS = STUDIES / "maskco_perishable_external_20261119"
sys.path.insert(0, str(PREVIOUS))
import prepare_panel as panel
from freight_admission import DAYS, arc, delivery_day


URLS = dict(official_sources())
GROUPS = tuple(panel.archives.GROUPS)


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def xy(rows):
    return Counter(tuple(r[1:3]) for r in rows)


def bag(counter):
    return tuple(sorted(counter.items()))


def overlap(current, prior):
    if not prior:
        return {"count": 0, "case": None}
    size, case = max((sum((current & other).values()), name) for name, other in prior.items())
    return {"count": size, "case": case}


def projection(case, source, cap, depot, clients, url):
    scale = 1440 / depot[5]
    rows = [depot, *clients]
    nodes = [[i, r[1]*scale, r[2]*scale, 20000*r[3]/cap, r[4]*scale, r[5]*scale, r[6]*scale]
             for i, r in enumerate(rows)]
    return case, source, nodes, panel.archives.coordinates(rows), url


def xtw_cases(entries, meta):
    known = {}
    for path in STUDIES.rglob("XTW-n*.vrp"):
        if path.is_relative_to(HERE.parent) or path.stem in known:
            continue
        raw = panel.archives.xtw(path.read_bytes())[2]
        known[path.stem] = xy(raw[1:101])
    history = dict(known)
    trace = []
    count = 0
    for n in XTW:
        title = f"XTW-n{n}"
        fn = title + ".vrp"
        name, cap, rows = panel.archives.xtw((HERE/"raw"/fn).read_bytes())
        if name != title or [r[0] for r in rows] != list(range(n)):
            raise ValueError(title)
        c = xy(rows[1:101])
        closest = overlap(c, known)
        include = closest["count"] <= 10 and title not in history
        trace.append(dict(raw=title, historical=overlap(c, history), prior_or_selected=closest, selected=include))
        if include:
            case = f"xtw_n{n}_p100"
            entries.append(projection(case, "XTW-p100", cap, rows[0], rows[1:101], URLS[fn]))
            meta[case] = dict(evidence_role="new_low_overlap_geometry", original=title, source_ids=[1, 100])
            known[title] = c
            count += 1
            if count == 8:
                break
    return dict(historical_originals=len(history), available_order=list(XTW), selected=count, decisions=trace)


def gh_cases(entries, meta):
    previous = {}
    local = panel.archives.OLD/"sources/homberger200.zip"
    with ZipFile(local) as zipped:
        for name in zipped.namelist():
            title = Path(name).stem.upper()
            if name.upper().endswith(".TXT") and title.split("_")[-1] in ("1", "2"):
                previous[title] = panel.archives.gh(zipped.read(name))[1]
    for folder in (panel.PREV/"validation_sources", PREVIOUS/"source_external",
                   PREVIOUS/"reserved_validation/raw", PREVIOUS/"future_validation/raw"):
        for p in folder.glob("*.TXT"):
            if p.stem.upper().split("_")[0] in GROUPS:
                previous[p.stem.upper()] = panel.archives.gh(p.read_bytes())[1]
    old_full = {bag(xy(rows[1:])) for rows in previous.values()}
    old_prefix = {n: xy(rows[1:101]) for n, rows in previous.items()}
    eligible = {}
    fallback = {}
    traces = []
    urls_file = HERE/"raw/gh_source_urls.json"
    original_urls = json.loads(urls_file.read_text(encoding="utf8")) if urls_file.is_file() else {}
    for n in (200, 400, 600, 800, 1000):
        path = local if n == 200 else HERE/"raw"/f"homberger{n}.zip"
        url = f"https://www.sintef.no/projectweb/top/vrptw/{n}-customers/"
        with ZipFile(path) as z:
            names = {Path(f).stem.upper(): f for f in z.namelist() if f.upper().endswith(".TXT")}
            for family in GROUPS:
                for i in range(1, 11):
                    title = f"{family}_{n//100}_{i}"
                    cap, rows = panel.archives.gh(z.read(names[title]))
                    if [r[0] for r in rows] != list(range(n+1)):
                        raise ValueError(title)
                    previously_used = bag(xy(rows[1:])) in old_full
                    closest = overlap(xy(rows[1:101]), old_prefix)
                    valid = not previously_used and closest["count"] <= 10
                    traces.append(dict(raw=title, prior_mother=previously_used, prior_prefix=closest, eligible=valid))
                    if valid and family not in eligible:
                        eligible[family] = (title, cap, rows, original_urls.get(title,url), 1)
                    if n == 200 and i == 3:
                        fallback[family] = (title, cap, rows, original_urls.get(title,url), 101)
    groups = {}
    included = []
    for family in GROUPS:
        title, cap, rows, url, first = eligible.get(family, fallback[family])
        primary = first == 1
        source = "GH-p100" if primary else "GH-p100-block2"
        case = "gh_" + title.lower() + ("_p100" if primary else "_c101_200")
        clients = rows[first:first+100]
        group = groups.setdefault(bag(xy(clients)), "GH:" + title + ("/block1" if primary else "/block2"))
        entries.append(projection(case, source, cap, rows[0], clients, url + "#" + title))
        meta[case] = dict(evidence_role="new_low_overlap_geometry" if primary else "within_mother_customer_block",
                          geometry=group, original=title, source_ids=[first,first+99])
        included.append(dict(case=case, mother_previously_used=bag(xy(rows[1:])) in old_full,
                             nearest_previously_scored_or_sealed_prefix=overlap(xy(clients), old_prefix)))
    return dict(originals_screened=len(traces), historical_and_sealed_time_window_variants=len(previous),
                historical_and_sealed_mothers=len(old_full), eligible_first_block_originals=sum(t["eligible"] for t in traces),
                selected=included, decisions=traces)


def freight_cases(entries, meta):
    authors = json.loads((PREVIOUS/"freight_source_inventory.json").read_text(encoding="utf8"))
    known = {}
    for country in ("BE","NL"):
        for day in DAYS:
            old = f"{country}_{day}_0_F"
            known[old] = Counter(delivery_day(old)[2][1:])
    history = dict(known)
    trace = []
    for country in ("BE","NL"):
        for day in DAYS:
            name = f"{country}_{day}_1_F"
            nodes, ids, ll, total = delivery_day(name)
            loc = Counter(ll[1:])
            err = max(abs(math.dist(nodes[i][1:3],nodes[j][1:3])/arc(ll[i],ll[j])[0]-1)
                      for i,j in combinations(range(101),2) if ll[i]!=ll[j])
            url = f"https://github.com/siemterbraake/vrppdl-algorithms-instances/blob/{authors['archive_commit']}/instances/large/{name}.txt"
            case = "vrppdl_" + name.lower() + "_p100"
            entries.append((case, "VRPPDL-p100", nodes, tuple(sorted(ll)), url))
            meta[case] = dict(country=country, source_day=name, evidence_role="new_order_day_same_provider")
            trace.append(dict(case=case, available_orders=total, source_node_ids=ids,
                              max_projection_relative_error=err, previous_days=overlap(loc,history),
                              previous_and_selected_days=overlap(loc, known)))
            known[name] = loc
    return dict(original_files=len(authors["instance_files"]), original_days_reserved=len(history), selected=trace)


def build():
    entries = []
    meta = {}
    audit = {}
    audit["freight"] = freight_cases(entries, meta)
    audit["xtw"] = xtw_cases(entries, meta)
    audit["gh"] = gh_cases(entries, meta)
    panel.HERE = HERE
    catalog, checks, _ = panel.keep(entries, set())
    for case, info in meta.items():
        catalog["cases"][case].update(info)
    old = json.loads((PREVIOUS/"future_validation/heldout_sources.json").read_text(encoding="utf8"))
    primary = json.loads((PREVIOUS/"reserved_validation/heldout_sources.json").read_text(encoding="utf8"))
    if set(meta) & (set(old["cases"]) | set(primary["cases"])):
        raise ValueError("A previously used or registered case was selected")
    cases = catalog["cases"]
    by_source = {source:dict(cases=len(rows),groups=len({row["geometry"] for row in rows}))
                 for source in sorted({r["dataset"] for r in cases.values()})
                 for rows in [[r for r in cases.values() if r["dataset"]==source]]}
    good = (all(not v["infeasible_singletons"] and v.get("initializer_routes",0)>0 for v in checks.values())
            and all(f["max_projection_relative_error"] <= .001 for f in audit["freight"]["selected"]))
    status = "PASS" if good else "STOP_ADMISSION"
    write("heldout_sources.json",catalog)
    write("heldout_admission.json",dict(status=status,cases=checks,by_source=by_source))
    write("selection.json",dict(status=status,solver_endpoints_at_admission=0,by_source=by_source,**audit))
    xtw = audit["xtw"]
    gh = audit["gh"]
    freight = audit["freight"]["selected"]
    lines = ["# 新配送日、低重叠实例与 GH 跨客户块复核", "",
             f"准入结果为 **{status}**。共同起点由独立评价器检查；本阶段方法求解终点为 0。", "",
             "| 来源 | 案例 | 几何或订单日 | 证据范围 |", "|---|---:|---:|---|"]
    labels = {"VRPPDL-p100":"同一运营商的新配送日，两国单列", "XTW-p100":"客户位置低重叠的新几何",
              "GH-p100":"历史与封存面板之外的新母几何", "GH-p100-block2":"相同母几何内第 101–200 客户；压力测试单列"}
    for group,counts in by_source.items():
        lines.append(f"| {group} | {counts['cases']} | {counts['groups']} | {labels[group]} |")
    lines += ["", f"XTW 逐一检查 n701 之后的官方编号。与 {xtw['historical_originals']} 个既有或封存原件及此前入选新例比较，共选出 "
              + "、".join(t["raw"] for t in xtw["decisions"] if t["selected"]) + "。", "",
              f"荷比十个编号 1 的配送日有 {min(t['available_orders'] for t in freight)}–{max(t['available_orders'] for t in freight)} 条符合预定字段的订单，"
              f"按原行序取首 100 条。与封存编号 0 的十个配送日逐对比较，精确客户坐标最大重叠为 {max(t['previous_days']['count'] for t in freight)}，"
              f"公里距离的最大投影相对误差为 {100*max(t['max_projection_relative_error'] for t in freight):.5f}%。配送日来自一般零担业务，原作者对客户位置作过匿名扰动。", "",
              f"GH 官方五个规模合计 {gh['originals_screened']} 个原件。历史及封存输入含 {gh['historical_and_sealed_mothers']} 个母客户集合，"
              f"按事先登记的母几何与客户重叠双重标准，有 {gh['eligible_first_block_originals']} 个标准首百客户变体满足新母几何条件。", "",
              "[输入选择预登记](输入选择预登记.md) · [原件逐项审查](selection.json) · [逐例可行性](heldout_admission.json) · [原始来源和投影索引](heldout_sources.json)"]
    (HERE/"准入.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(status, len(cases), by_source, flush=True)
    if not good:
        raise ValueError("One or more prespecified inputs failed admission; see heldout_admission.json")


if __name__=="__main__":
    build()
