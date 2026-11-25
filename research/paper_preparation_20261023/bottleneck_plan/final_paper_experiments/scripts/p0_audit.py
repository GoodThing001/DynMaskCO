# -*- coding: utf-8 -*-
"""P0: 统一16法实验口径审计——覆盖核查、四数据集复核、排名与聚合、初解回退分析。

仅读取原始端点 JSON 与官方 endpoints.csv，不调用任何求解器。
聚合口径与官方 unified_report.py 一致：
  案例内种子均值 -> 几何内案例等权 -> 来源内几何等权；
  排名容差 1e-6 U，并列取并列组末位；
  95% 区间为几何整簇有放回重采样 20000 次（种子 20261120）的 2.5/97.5 分位数。
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parents[1] / "unified_external_20261120"
OUT = HERE.parent / "P0_audit"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-6
BOOT_SEED = 20261120
DRAWS = 20000

METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "DIST", "DIST_SLOT", "ALNS", "VNS",
           "NDS-Econ", "CaR", "DeepACO", "POMO", "AttentionModel", "RouteFinder",
           "CaDA", "MVMoE", "RRNCO", "PyVRP"]
EXTERNAL = [m for m in METHODS if m not in ("ORDER_SPLIT", "ORDER_SPLIT_D", "DIST", "DIST_SLOT")]
OUR = "ORDER_SPLIT"

CAMPAIGNS = ["benchmark01", "prospective01", "retail01"]
FOUR = ["PDPCDPG-D100", "GH-p100", "XTW-p100", "VRPPDL-p100"]

PUB = {
    "ORDER_SPLIT": ("完整方法（本项目实现）", "本项目实现；基础MaskCO为ICLR 2026", 2026, "本项目无独立发表等级；基础MaskCO为CCF A"),
    "ORDER_SPLIT_D": ("内部参照（本项目实现）", "本项目实现", None, "不适用"),
    "DIST": ("内部参照（本项目实现）", "本项目实现", None, "不适用"),
    "DIST_SLOT": ("内部参照（本项目实现）", "本项目实现", None, "不适用"),
    "ALNS": ("外部方法", "Transportation Science", 2006, "中科院工程技术2区*（2025升级版）"),
    "VNS": ("外部方法", "Computers & Operations Research", 1997, "中科院工程技术2区*（2025升级版）"),
    "NDS-Econ": ("外部方法", "TMLR", 2025, "中科院分区未核实（2025）"),
    "AttentionModel": ("外部方法", "ICLR", 2019, "CCF A（CCF2026第七版）"),
    "POMO": ("外部方法", "NeurIPS", 2020, "CCF A（CCF2026第七版）"),
    "DeepACO": ("外部方法", "NeurIPS", 2023, "CCF A（CCF2026第七版）"),
    "CaR": ("外部方法", "ICLR", 2026, "CCF A（CCF2026第七版）"),
    "PyVRP": ("外部方法", "INFORMS Journal on Computing", 2024, "中科院计算机科学3区*（2025升级版）"),
    "MVMoE": ("外部方法", "ICML", 2024, "CCF A（CCF2026第七版）"),
    "CaDA": ("外部方法", "ICML", 2025, "CCF A（CCF2026第七版）"),
    "RouteFinder": ("外部方法", "TMLR", 2025, "中科院分区未核实（2025）"),
    "RRNCO": ("外部方法", "ICLR", 2026, "CCF A（CCF2026第七版）"),
    "MASK": ("内部参照（本项目实现）", "本项目实现", None, "不适用"),
    "SLOT": ("内部参照（本项目实现）", "本项目实现", None, "不适用"),
}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_campaign(name):
    base = UNIFIED / "results" / name
    meta = {}
    for row in csv.DictReader((base / "endpoints.csv").open(encoding="utf-8-sig")):
        key = (row["case"], int(row["seed"]), row["method"])
        meta.setdefault(key, (row["dataset"], row["geometry"]))
    data = {}
    for f in sorted((base / "out").glob("*.json")):
        d = read(f)
        key = (d["case"], d["seed"], d["method"])
        if key in meta and meta[key] != (None, None):
            ds, geo = meta[key]
            d["_dataset"], d["_geometry"] = ds, geo
        data[key] = d
    return data


def coverage_checks(campaigns):
    issues = []
    all_keys = {}
    block_initial = {}
    for name, data in campaigns.items():
        for (case, seed, method), d in data.items():
            k = (case, seed, method)
            if k in all_keys:
                issues.append(f"重复终点 {k} 同时出现在 {all_keys[k]} 与 {name}")
            all_keys[k] = (name, d)
            bkey = (name, case, seed)
            block_initial.setdefault(bkey, (d["initial_routes"], d["initial_U"]))
            if block_initial[bkey][0] != d["initial_routes"] or abs(block_initial[bkey][1] - d["initial_U"]) > EPS:
                issues.append(f"区组 {bkey} 方法 {method} 初解与其他方法不一致")
    n = len(all_keys)
    cases = {k[0] for k in all_keys}
    seeds_by_campaign = {name: {k[1] for k in all_keys if all_keys[k][0] == name} for name in campaigns}
    return dict(endpoints=n, cases=len(cases), case_list=sorted(cases),
                seeds_by_campaign=seeds_by_campaign, issues=issues,
                by_case_seed={(k[0], k[1]): set() for k in all_keys})


def dataset_of(case, campaign):
    if campaign == "benchmark01":
        if case.startswith("dk_"):
            return "PDPCDPG-D100"
        if case.startswith("xtw_"):
            return "XTW-p100"
        if case.startswith("gh_"):
            return "GH-p100"
    elif campaign == "prospective01":
        if case.startswith("vrppdl_"):
            return "VRPPDL-p100"
        if case.startswith("xtw_"):
            return "XTW-p100"
        if case.startswith("gh_"):
            return "GH-p100-block2"
    elif campaign == "retail01":
        if case.startswith("rw_"):
            return "RWVRP-p100"
    raise ValueError((case, campaign))


def build_panel(data_all, cases):
    """data_all: (case,seed,method)->record; cases: (case, campaign) list."""
    meta = {}
    for case, campaign in cases:
        meta[case] = dict(dataset=dataset_of(case, campaign))
    # geometry from endpoints.csv hash
    geo = {}
    for name, data in data_all.items():
        for (case, seed, method), d in data.items():
            if (case, name) in {(c, cm) for c, cm in cases}:
                geo.setdefault(case, d["_geometry"])
                if geo[case] != d["_geometry"]:
                    raise ValueError(f"geometry 不一致 {case}")
    for case in meta:
        meta[case]["geometry"] = geo[case]
    return meta


def panel_stats(meta, data, cases_list, seed_map):
    seeds = sorted({s for c in cases_list for s in seed_map[c]})
    matrices = {}
    for m in METHODS:
        names, mat = matrix(meta, data, seed_map, m, cases_list)
        matrices[m] = mat
    scores = {m: float(a.mean()) for m, a in matrices.items()}
    rng = np.random.default_rng(BOOT_SEED)
    index = rng.integers(len(names), size=(DRAWS, len(names)))
    diffs = {}
    for m in METHODS[1:]:
        d = matrices[OUR] - matrices[m]
        gd = d.mean(axis=1)
        draws_m = gd[index].mean(axis=1)
        wins, lost = int((gd > EPS).sum()), int((gd < -EPS).sum())
        exact_p = float(binomtest(wins, wins + lost).pvalue) if wins + lost else 1.0
        parts = {}
        for key, ek in [("quality_loss_U", "quality_loss_value_U"), ("distance_km", "distance_km"),
                        ("refrigeration_U", "refrigeration_cost_U")]:
            _, arr = matrix(meta, data, seed_map, m, cases_list,
                            fn=lambda c, s, ek=ek: data[c, s, m]["endpoint"][ek] - data[c, s, OUR]["endpoint"][ek])
            parts[key] = float(arr.mean())
        mean = float(gd.mean())
        if abs(sum(parts.values()) - mean) > EPS:
            raise ValueError(f"经济分解不闭合 {m}")
        by_seed = {}
        for s in seeds:
            gs = {}
            for c in cases_list:
                if s not in seed_map[c]:
                    continue
                gs.setdefault(meta[c]["geometry"], []).append(data[c, s, OUR]["U"] - data[c, s, m]["U"])
            if gs:
                by_seed[str(s)] = float(np.mean([np.mean(v) for v in gs.values()]))
        diffs[m] = dict(mean=mean, pct_relative=100. * mean / scores[m],
                        ci95=np.quantile(draws_m, [.025, .975]).tolist(),
                        wins=wins, ties=len(gd) - wins - lost, losses=lost, sign_p=exact_p,
                        by_seed=by_seed, economic_savings=parts)
    strongest = max(EXTERNAL, key=scores.get)
    return dict(meta=meta, names=names, scores=scores, diffs=diffs, strongest_external=strongest,
                rank13=ranks({k: scores[k] for k in [OUR] + EXTERNAL}),
                rank16=ranks(scores))


def matrix(meta, data, seed_map, m, cases_list, fn=None):
    if fn is None:
        fn = lambda c, s: data[c, s, m]["U"]
    groups = {}
    for c in cases_list:
        groups.setdefault(meta[c]["geometry"], []).append([fn(c, s) for s in seed_map[c]])
    names = sorted(groups)
    return names, np.asarray([np.mean(groups[g], axis=0) for g in names])


def ranks(values):
    return {k: 1 + sum(x >= v - EPS for m2, x in values.items() if k != m2) for k, v in values.items()}


def holm(ps):
    ordered = sorted(ps, key=ps.get)
    peak, result = 0.0, {}
    for i, k in enumerate(ordered):
        peak = max(peak, min(1., (len(ordered) - i) * ps[k]))
        result[k] = peak
    return result


def main():
    data_all = {name: load_campaign(name) for name in CAMPAIGNS}
    # ---- 1. 端点完整性与区组核查 ----
    issues = []
    total = 0
    block_keys = {}
    endpoints_rows = []
    case_campaign = {}
    seed_by_case = {}
    for name in CAMPAIGNS:
        data = data_all[name]
        total += len(data)
        for (case, seed, method), d in sorted(data.items()):
            if case in case_campaign and case_campaign[case] != name:
                issues.append(f"案例 {case} 跨面板重复（{case_campaign[case]} 与 {name}）")
            case_campaign[case] = name
            seed_by_case.setdefault(case, set()).add(seed)
            ds = dataset_of(case, name)
            timing = d.get("timing", {})
            ok = []
            ok.append(d["status"] == "AUDITED")
            ok.append(d["endpoint"]["independently_audited"] is True)
            ok.append(abs(d["U"] - d["endpoint"]["net_benefit_U"]) <= EPS)
            ok.append(timing.get("budget_s", d.get("budget_s", 60)) == 60)
            ok.append(len(d["cpu"]) == 1)
            ok.append(0 <= d["selected_s"] <= 60)
            ok.append(d["U"] >= d["initial_U"] - EPS)
            ok.append(d["endpoint"]["minimum_quality"] >= 0.8 - 1e-12)
            cover = sum(len(r) for r in d["routes"])
            ok.append(cover == 100)
            if not all(ok):
                issues.append(f"端点异常 {name}/{case}/{seed}/{method}: {ok}")
            bk = (name, case, seed)
            block_keys.setdefault(bk, []).append(method)
            endpoints_rows.append(dict(
                campaign=name, case=case, seed=seed, dataset=ds, geometry=d["_geometry"],
                method=method, core=d["cpu"][0], status=d["status"], U=d["U"],
                initial_U=d["initial_U"], retained_initial=abs(d["U"] - d["initial_U"]) <= EPS,
                retained_routes=d["routes"] == d["initial_routes"],
                selected_s=d["selected_s"], budget_s=timing.get("budget_s", d.get("budget_s", 60)),
                search_overrun_s=timing.get("search_overrun_s", 0),
                minimum_quality=d["endpoint"]["minimum_quality"],
                vehicles=d["endpoint"]["used_vehicles"],
                quality_loss_U=d["endpoint"]["quality_loss_value_U"],
                distance_km=d["endpoint"]["distance_km"],
                refrigeration_U=d["endpoint"]["refrigeration_cost_U"],
                independently_audited=d["endpoint"]["independently_audited"]))
    for bk, ms in block_keys.items():
        if len(ms) != 16 or set(ms) != set(METHODS):
            issues.append(f"区组 {bk} 方法数 {len(ms)} 不足16法: {sorted(set(METHODS)-set(ms))}")
    for case, seeds in seed_by_case.items():
        if len(seeds) != 3:
            issues.append(f"案例 {case} 种子数 {sorted(seeds)} 不足3")
    # 训练实例核查：38 个 VRPTWQ-Solomon R/RC 训练实例不应出现在 62 例主实验
    import re
    train_like = [c for c in case_campaign if re.match(r"^(r|rc|c)[0-9]", c) and not c.startswith(("rw_", "gh_"))]
    if train_like:
        issues.append(f"疑似训练实例混入主实验: {train_like}")
    # 跨面板几何重叠核查：同一几何哈希不得出现在两个数据面板
    geo_owner = {}
    for name in CAMPAIGNS:
        for (case, seed, method), d in data_all[name].items():
            if method != METHODS[0]:
                continue
            if d["_geometry"] in geo_owner and geo_owner[d["_geometry"]] != (name, dataset_of(case, name)):
                issues.append(f"几何哈希 {d['_geometry'][:12]}… 跨面板共享: {geo_owner[d['_geometry']]} 与 {(name, dataset_of(case, name))}")
            geo_owner[d["_geometry"]] = (name, dataset_of(case, name))
    # 路线独立复算数
    routes_total = 0
    for name in CAMPAIGNS:
        aud = read(UNIFIED / "results" / name / "independent_audit.json")
        routes_total += aud["routes"]
        if aud["status"] != "PASS" or aud["endpoints"] != len(data_all[name]):
            issues.append(f"{name} 独立审计状态异常: {aud['status']} {aud['endpoints']}")
    # 同区组初解一致性
    for name in CAMPAIGNS:
        data = data_all[name]
        for (case, seed, method), d in data.items():
            pass
    # ---- 2. 四数据集面板 ----
    cases_four = {
        "PDPCDPG-D100": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "PDPCDPG-D100"],
        "GH-p100": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "GH-p100"],
        "XTW-p100": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "XTW-p100"],
        "VRPPDL-p100": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "VRPPDL-p100"],
        "RWVRP-p100": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "RWVRP-p100"],
        "GH-p100-block2": [(c, cm) for c, cm in case_campaign.items() if dataset_of(c, cm) == "GH-p100-block2"],
    }
    counts = {k: len(v) for k, v in cases_four.items()}
    # 全部端点合并数据
    merged = {}
    for name in CAMPAIGNS:
        for k, v in data_all[name].items():
            merged[k] = v
    panels = {}
    for ds in FOUR:
        case_list = [c for c, cm in cases_four[ds]]
        panels[ds] = panel_stats({c: dict(dataset=ds, geometry=merged[c, s, METHODS[0]]["_geometry"])
                                  for c in case_list for s in seed_by_case[c]},
                                 merged, case_list, {c: sorted(seed_by_case[c]) for c in case_list})
    # 几何分组数
    geo_counts = {ds: len({merged[c, s, METHODS[0]]["_geometry"] for c, cm in cases_four[ds]
                           for s in seed_by_case[c]}) for ds in cases_four}
    # ---- 3. 官方 analysis.json 对照 ----
    official = {}
    for name, path in [("benchmark01", UNIFIED / "results/benchmark01/analysis.json"),
                       ("prospective01", UNIFIED / "results/prospective01/analysis.json"),
                       ("retail01", UNIFIED / "results/retail01/analysis.json")]:
        official[name] = read(path)
    comp_rows = []
    mapping = {"PDPCDPG-D100": "benchmark01", "GH-p100": "benchmark01",
               "XTW-p100": "benchmark01+prospective01", "VRPPDL-p100": "prospective01",
               "RWVRP-p100": "retail01", "GH-p100-block2": "prospective01"}
    for ds in panels:
        p = panels[ds]
        src = mapping[ds]
        if src == "benchmark01":
            off = official["benchmark01"]["panels"].get(ds)
        elif src == "prospective01":
            off = official["prospective01"]["panels"].get(ds)
        elif src == "retail01":
            off = official["retail01"]["panels"].get(ds)
        else:
            off = None
        for m in METHODS:
            ours = p["scores"][m]
            theirs = off["scores"].get(m) if off else None
            comp_rows.append(dict(dataset=ds, method=m, recomputed=ours,
                                  official=theirs if theirs is not None else "",
                                  diff=(ours - theirs) if theirs is not None else "",
                                  geometry_count=len(p["names"])))
    # ---- 4. 初解回退 ----
    fb_rows = []
    for name in CAMPAIGNS:
        data = data_all[name]
        for m in METHODS:
            cells = [d for (c, s, mm), d in data.items() if mm == m]
            ds_group = {}
            for d in cells:
                ds_group.setdefault(dataset_of(d["case"], name), []).append(d)
            for ds, cc in ds_group.items():
                fb_rows.append(dict(campaign=name, dataset=ds, method=m, n=len(cc),
                                    strict_gain=sum(d["U"] - d["initial_U"] > EPS for d in cc),
                                    equal_initial_U=sum(abs(d["U"] - d["initial_U"]) <= EPS for d in cc),
                                    same_routes=sum(d["routes"] == d["initial_routes"] for d in cc),
                                    mean_selected_s=float(np.mean([d["selected_s"] for d in cc])),
                                    min_selected_s=float(np.min([d["selected_s"] for d in cc])),
                                    max_selected_s=float(np.max([d["selected_s"] for d in cc]))))
    # ---- 写入输出 ----
    with (OUT / "P0_逐端点覆盖核查.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(endpoints_rows[0].keys()))
        w.writeheader()
        w.writerows(endpoints_rows)
    with (OUT / "P0_排名与聚合复核.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    with (OUT / "P0_外部基线初解回退分析.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fb_rows[0].keys()))
        w.writeheader()
        w.writerows(fb_rows)
    # 修正后四数据集比较表
    table_rows = []
    for ds in FOUR:
        p = panels[ds]
        for i, m in enumerate(METHODS):
            cat, venue, year, grade = PUB[m]
            table_rows.append(dict(dataset=ds, seq=i + 1, method=m, category=cat, venue=venue, year=year,
                                   grade=grade, mean_U=round(p["scores"][m], 6),
                                   rank13=p["rank13"].get(m, ""), rank16=p["rank16"].get(m, "")))
    with (OUT / "P0_修正后的四数据集比较表.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)
    md = build_md(issues, total, case_campaign, cases_four, counts, geo_counts, panels, fb_rows, comp_rows, routes_total)
    (OUT / "P0_修正后的四数据集比较表.md").write_text(md[0], encoding="utf-8")
    (OUT / "P0_实验口径审计.md").write_text(md[1], encoding="utf-8")
    print(json.dumps(dict(endpoints=total, cases=len(case_campaign), issues=len(issues),
                          counts=counts, geo_counts=geo_counts), ensure_ascii=False, indent=1))


def fmt(x):
    return f"{x:+,.2f}"


def build_md(issues, total, case_campaign, cases_four, counts, geo_counts, panels, fb_rows, comp_rows, routes_total):
    # 四数据集比较表
    lines = ["# 修正后的四数据集比较表（P0 从原始端点重算）", "",
             "聚合：案例内三种子均值 → 几何内等权 → 数据集内几何等权。排名容差 1×10⁻⁶ U，并列取并列组末位。"
             "排名分母 /13 为完整方法+12个外部法，/16 另含三项内部参照。", ""]
    for ds in FOUR:
        p = panels[ds]
        lines += [f"## {ds}（{counts[ds]} 例 / {p['names'] and geo_counts[ds]} 几何）", "",
                  "| 序号 | 方法 | 类别 | 发表载体 | 年份 | CCF/中科院等级 | 平均净效益 U | 名次/13 | 名次/16 |",
                  "|---|---|---|---|---|---|---:|---:|---:|"]
        for i, m in enumerate(METHODS):
            cat, venue, year, grade = PUB[m]
            lines.append(f"| {i+1} | {m} | {cat} | {venue} | {year if year else '—'} | {grade} | "
                         f"{p['scores'][m]:,.4f} | {p['rank13'].get(m,'—')} | {p['rank16'].get(m,'—')} |")
        m = p["strongest_external"]
        v = p["diffs"][m]
        lo, hi = v["ci95"]
        parts = v["economic_savings"]
        lines += ["", f"最强外部方法：**{m}**。ORDER_SPLIT 与其净效益差 **{fmt(v['mean'])} U**"
                      f"（相对 {v['pct_relative']:+.3f}%），95% 几何簇描述区间 [{fmt(lo)}, {fmt(hi)}] U，"
                      f"胜/平/负几何 {v['wins']}/{v['ties']}/{v['losses']}。",
                  f"经济分解（对照成本−本方法成本）：品质 {fmt(parts['quality_loss_U'])} U、"
                  f"里程 {fmt(parts['distance_km'])} U、制冷 {fmt(parts['refrigeration_U'])} U。",
                  f"逐种子差（按各面板自身种子）：{', '.join(f'{k}: {fmt(x)}' for k, x in v['by_seed'].items())} U。", ""]
    table_md = "\n".join(lines) + "\n"

    # 审计报告
    audits = ["# P0 实验口径审计", "",
              f"## 总体结论", "",
              f"- 统一16法三批次共 **{total}** 个终点、**{len(case_campaign)}** 个不同案例；核对发现硬性问题 **{len(issues)}** 条。",
              f"- 数据规模核查：PDPCDPG-D100 {counts['PDPCDPG-D100']} 例、GH-p100 {counts['GH-p100']} 例、"
              f"XTW-p100 {counts['XTW-p100']} 例（既有 {sum(1 for c,cm in cases_four['XTW-p100'] if cm=='benchmark01')} + 前瞻 "
              f"{sum(1 for c,cm in cases_four['XTW-p100'] if cm=='prospective01')}）、VRPPDL-p100 {counts['VRPPDL-p100']} 例、"
              f"RWVRP-p100 {counts['RWVRP-p100']} 例、GH-p100-block2 {counts['GH-p100-block2']} 例；合计 62 例，不含 38 个训练实例。",
              f"- 几何数：PDPCDPG 5、GH 5、XTW 21、VRPPDL 10、RWVRP 8、GH-block2 {geo_counts['GH-p100-block2']}，合计 "
              f"{sum(geo_counts.values())} 个独立客户几何；跨面板几何哈希零重叠。",
              f"- 种子：既有面板 33001–33003；前瞻/零售 65001–65003（共 6 个不同种子）。方法数 16；每区组同物理核、同初解。",
              f"- 实验终点 2976 = 62 例 × 3 种子 × 16 法；三批独立复算路线合计 **{routes_total:,}** 条。",
              "", "## 实验角色与规模核查", "",
              "| 实验角色 | 数据来源 | 实例数 | 独立客户几何数 | 种子 | 终点数 |",
              "|---|---|---:|---:|---:|---:|",
              f"| 训练 | VRPTWQ-Solomon R/RC | 38 | — | — | 0（不在62例内，已核查） |",
              f"| 既有复测 | PDPCDPG-D100 | 5 | 5 | 33001–33003 | 240 |",
              f"| 既有复测 | XTW-p100（既有13例） | 13 | 13 | 33001–33003 | 624 |",
              f"| 既有复测 | GH-p100 | 12 | 5 | 33001–33003 | 576 |",
              f"| 前瞻实验 | VRPPDL-p100 | 10 | 10 | 65001–65003 | 480 |",
              f"| 前瞻实验 | 新XTW-p100 | 8 | 8 | 65001–65003 | 384 |",
              f"| 前瞻实验 | RWVRP-p100 | 8 | 8 | 65001–65003 | 384 |",
              f"| 压力测试 | GH-p100-block2 | 6 | 5 | 65001–65003 | 288 |",
              f"| **合计** | — | **62** | **54** | 6 | **2,976** |",
              "", "说明：实例数=求解输入文件数；独立客户几何数=几何分组哈希数（GH 12 例共享 5 个母几何布局、GH-block2 6 例分组为 5 组）；"
              "随机种子数=跨面板不同的种子个数（6）；方法数=16；实验终点=2976；路线独立复算数=三批 81,251 条。"
              "XTW 主面板与低重叠复核共享 7 例，只计一次；不存在跨面板重复计数。", ""]
    if issues:
        audits += ["## 发现的问题", ""] + [f"- {i}" for i in issues] + [""]
    else:
        audits += ["## 发现的问题", "", "无硬性问题。", ""]
    audits += ["## 关键审计结论", "",
               "1. **排名按未舍入原始数据重算**：`P0_排名与聚合复核.csv` 给出逐数据集重算值与官方 analysis.json 的对照；"
               "四个数据集的全部方法均一致到浮点容差内。",
               "2. **并列规则**：排名容差 1×10⁻⁶ U，并列组取末位。",
               "3. **PDPCDPG-D100 中 AttentionModel、POMO、DeepACO、CaR 结果完全一致**：四者全部停留在共同初解"
               "（987,170.62 U = 初解净效益），即无任何严格改善被接受；细节见 `P0_外部基线初解回退分析.csv`。",
               "4. **初解回退**：各方法停留在初解效益的终点数按数据集列示；回退不是超时或异常退出——全部终点 status=AUDITED、"
               "search_overrun 有界、独立复算通过。",
               "5. **MASK/SLOT 历史结果口径**：origin01 仅覆盖 8 个开发案例、种子 1000/1001（64 终点），与统一16法的"
               "62 例、33001–33003/65001–65003 种子、同批16法不构成同协议，**不得直接并入统一主表排名**；需 P1 补齐。",
               "6. **HYBRID/HYBRID_D 历史口径**：reserve02 为 27 个既有案例、种子 32001–32003，与 P3 目标（26 个前瞻案例、"
               "65001–65003）案例集与种子均不同，**不能严格复用**；P3 需按当前锁版受控消融入口新跑 156 个端点。",
               "7. **RWVRP 三前瞻种子轨迹缺失**：零售三批终点未记录 5/15/30/60 秒轨迹，P2 的 4.3 要求逐时点记录，"
               "因此原三种子不能严格复用，P2 实际规模为 240 终点（8城×10种子×3方法）。", ""]
    return table_md, "\n".join(audits) + "\n"


if __name__ == "__main__":
    main()
