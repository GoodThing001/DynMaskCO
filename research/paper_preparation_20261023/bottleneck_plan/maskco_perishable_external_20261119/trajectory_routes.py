"""Recover complete routes from paired development trajectories and reassemble their fleets."""
from collections import defaultdict
from functools import lru_cache
from statistics import mean
import json
import math
import time

from prepare_panel import HERE, core, independent_audit
from whole_group import priced_cover

OUT = HERE / "results/trajectory_routes01"
ARMS = ("distance", "original", "eight", "shuffled")
SCOPES = {"terminal": ("256",), "history": ("0", "64", "256")}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def assemble(checkpoints, model, value):
    base = max(checkpoints, key=lambda fleet: fleet["U"])
    routes = list(map(tuple, base["routes"]))
    if abs(sum(value(r) for r in routes) - base["U"]) > 1.e-6:
        raise ValueError("Archived fleet value differs")
    columns = {}
    for fleet in checkpoints:
        for items in fleet["routes"]:
            r = tuple(items)
            key = tuple(sorted(r))
            if key not in columns or value(r) > columns[key][0]:
                columns[key] = value(r), r
    answer = priced_cover(routes, columns, value)
    score = independent_audit(model, [model.route(r) for r in answer["routes"]], True)
    if abs(score - base["U"] - answer["gain"]) > 1.e-6:
        raise ValueError("Mixed fleet audit differs")
    return dict(U=score, direct_U=base["U"], columns=len(columns), optimal=answer["optimal"],
                gap_U=answer["gap_U"], routes=answer["routes"])


def measure(block, receipt, catalog):
    case = block["case"]
    if receipt["case"] != case or [r["seed"] for r in receipt["seeds"]] != [51291, 51292]:
        raise ValueError("Carryover source or seeds differ")
    nodes = read(HERE / catalog["derived_instances"][case]["file"])["nodes"]
    distances = [[math.dist(a[1:3], b[1:3]) for b in nodes] for a in nodes]
    model = core.PublicationModel(dict(name=case, nodes=nodes, distance=distances),
                                  core.load_config(), "main_Product1", .8)

    @lru_cache(maxsize=None)
    def value(route):
        row = model.route(route)
        if not row.feasible:
            raise ValueError((case, route))
        return row.net_benefit

    rows = []
    source = receipt["seeds"]
    for name, steps in SCOPES.items():
        # Reversing the two pure-distance parents gives the same route pool.
        reference = assemble([s["arms"]["distance"]["checkpoints"][t] for s in source for t in steps], model, value)
        for donor_index, donor in enumerate(source):
            anchor = source[1-donor_index]["arms"]["distance"]["checkpoints"]
            arms = {"distance": reference}
            for arm in ARMS[1:]:
                ck = donor["arms"][arm]["checkpoints"]
                arms[arm] = assemble([a[t] for a in (anchor, ck) for t in steps], model, value)
            rows.append(dict(case=case, **catalog["cases"][case], scope=name, donor_seed=donor["seed"], arms=arms))
    return rows


def summarize(rows):
    result = {}
    passed = True
    for dataset in sorted({r["dataset"] for r in rows}):
        section = [r for r in rows if r["dataset"] == dataset]
        def contrast(group, arm, metric):
            geometries = defaultdict(list)
            for row in group:
                methods = row["arms"]
                geometries[row["geometry"]].append(methods[arm][metric] - methods["distance"][metric])
            return {g:mean(v) for g,v in geometries.items()}
        history = [r for r in section if r["scope"] == "history"]
        terminal = [r for r in section if r["scope"] == "terminal"]
        geo = contrast(history, "eight", "U")
        def effect(part, arm="eight", metric="U"):
            return mean(contrast(part,arm,metric).values())
        by_seed = {s:effect([r for r in history if r["donor_seed"] == s]) for s in (51291,51292)}
        wins = sum(x > 1.e-6 for x in geo.values())
        result[dataset] = dict(geometries=geo, mean=mean(geo.values()), wins=wins, donor_seeds=by_seed,
            direct=effect(history,metric="direct_U"), terminal=effect(terminal),
            original=effect(history,arm="original"), shuffled=effect(history,arm="shuffled"))
        passed = passed and min(by_seed.values()) > 1.e-6 and 2*wins > len(geo) and mean(geo.values()) > 1.e-6
    passed = passed and all(mean(s["mean"]-s[arm] for s in result.values()) > 1.e-6 for arm in ("original","shuffled","direct"))
    return result, passed


def main():
    plan = read(HERE/"carryprobe01.json")
    catalog = read(HERE/"heldout_sources.json")
    if len(plan["blocks"]) != 27 or not (HERE/"results/carryprobe01/analysis.json").exists():
        raise ValueError("Complete trajectories required")
    OUT.mkdir(exist_ok=True)
    rows = []
    cpu, wall = time.process_time(), time.perf_counter()
    for i, block in enumerate(plan["blocks"]):
        rows.extend(measure(block, read(HERE/f"results/carryprobe01/out/{i}.json"), catalog))
    scores, passed = summarize(rows)
    unique = [a for r in rows for k, a in r["arms"].items() if k != "distance" or r["donor_seed"] == 51291]
    receipt = dict(sources=scores,passed=passed,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,
                   audited=len(unique),optimal=sum(r["optimal"] for r in unique),max_gap_U=max(r["gap_U"] for r in unique))
    (OUT/"fleets.json").write_text(json.dumps(rows,allow_nan=False),encoding="utf8")
    (OUT/"analysis.json").write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding="utf8")
    lines=["# 两条轨迹的路线重组", "", "各列为八视图起点供体减纯距离供体的几何等权净效益 U。共同距离轨迹与供体使用不同种子。","",
        "| 来源 | 整车队择优 | 仅终点路线重组 | 三个时点路线重组 | 改善几何 | 供体 51291 | 供体 51292 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("PDPCDPG-D100","XTW-p100","GH-p100"):
        s=scores[name]
        lines.append(f"| {name} | {s['direct']:+.2f} | {s['terminal']:+.2f} | {s['mean']:+.2f} | {s['wins']}/{len(s['geometries'])} | "
                     +" | ".join(f"{x:+.2f}" for x in s["donor_seeds"].values())+" |")
    decision = "达到" if passed else "未达到"
    lines.extend(["",f"{len(unique)} 个组合车队经过独立复算，{receipt['optimal']} 个达到路线池最优，最大剩余界差 {receipt['max_gap_U']:.4f} U。"
        f"新增 CPU {receipt['cpu_s']:.1f} 秒，墙钟 {receipt['wall_s']:.1f} 秒。{decision}单核完整计时开发门槛。",
        "","[方案与原门槛](../../轨迹中的整条路线复用.md) · [几何与置乱对照](analysis.json) · [逐车队结果](fleets.json)"])
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
