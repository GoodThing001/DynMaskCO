# -*- coding: utf-8 -*-
"""统一18法比较与配对增量分析（论文表3/4/5/6）。

聚合口径：案例内种子均值 → 几何内等权 → 来源内几何等权；
排名容差 1e-6 U，并列取并列组末位；95% 区间为几何整簇有放回重采样 20000 次（种子 20261120）。
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
FP = HERE.parent
UNIFIED = FP.parent / "unified_external_20261120"
OUT = FP / "tables"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-6
BOOT_SEED = 20261120
DRAWS = 20000
OUR = "ORDER_SPLIT"

METHODS16 = ["ORDER_SPLIT", "ORDER_SPLIT_D", "DIST", "DIST_SLOT", "ALNS", "VNS",
             "NDS-Econ", "CaR", "DeepACO", "POMO", "AttentionModel", "RouteFinder",
             "CaDA", "MVMoE", "RRNCO", "PyVRP"]
METHODS18 = METHODS16 + ["MASK", "SLOT"]
EXTERNAL = [m for m in METHODS16 if m not in ("ORDER_SPLIT", "ORDER_SPLIT_D", "DIST", "DIST_SLOT")]
INTERNAL = ["ORDER_SPLIT_D", "DIST", "DIST_SLOT", "MASK", "SLOT"]

PUB = {
    "ORDER_SPLIT": ("完整方法（本项目实现）", "本项目实现；基础MaskCO为ICLR 2026", "2026", "本项目无独立发表等级；基础MaskCO为CCF A"),
    "ORDER_SPLIT_D": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "DIST": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "DIST_SLOT": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "MASK": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "SLOT": ("内部参照（本项目实现）", "本项目实现", "—", "不适用"),
    "ALNS": ("外部方法", "Transportation Science", "2006", "中科院工程技术2区*（2025升级版）"),
    "VNS": ("外部方法", "Computers & Operations Research", "1997", "中科院工程技术2区*（2025升级版）"),
    "NDS-Econ": ("外部方法", "TMLR", "2025", "中科院分区未核实（2025）"),
    "AttentionModel": ("外部方法", "ICLR", "2019", "CCF A（CCF2026第七版）"),
    "POMO": ("外部方法", "NeurIPS", "2020", "CCF A（CCF2026第七版）"),
    "DeepACO": ("外部方法", "NeurIPS", "2023", "CCF A（CCF2026第七版）"),
    "CaR": ("外部方法", "ICLR", "2026", "CCF A（CCF2026第七版）"),
    "PyVRP": ("外部方法", "INFORMS Journal on Computing", "2024", "中科院计算机科学3区*（2025升级版）"),
    "MVMoE": ("外部方法", "ICML", "2024", "CCF A（CCF2026第七版）"),
    "CaDA": ("外部方法", "ICML", "2025", "CCF A（CCF2026第七版）"),
    "RouteFinder": ("外部方法", "TMLR", "2025", "中科院分区未核实（2025）"),
    "RRNCO": ("外部方法", "ICLR", "2026", "CCF A（CCF2026第七版）"),
}

PANEL_DEFS = None  # 已由 main 内 spec_map 取代


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_all():
    data = {}
    geo = {}
    seeds = {}
    for name in ("benchmark01", "prospective01", "retail01"):
        base = UNIFIED / "results" / name
        meta = {}
        for row in csv.DictReader((base / "endpoints.csv").open(encoding="utf-8-sig")):
            meta[(row["case"], int(row["seed"]), row["method"])] = (row["dataset"], row["geometry"])
        for f in sorted((base / "out").glob("*.json")):
            d = read(f)
            key = (d["case"], d["seed"], d["method"])
            ds, g = meta[key]
            d["_dataset"], d["_geometry"] = ds, g
            data[key] = d
            geo.setdefault(d["case"], set()).add(g)
            seeds.setdefault(d["case"], set()).add(d["seed"])
    p1 = FP / "P1_original_baselines" / "results" / "out"
    p1_present = p1.is_dir() and list(p1.glob("*.json"))
    if p1_present:
        p1geo = {}
        for name in ("benchmark01", "prospective01", "retail01"):
            base = UNIFIED / "results" / name
            for row in csv.DictReader((base / "endpoints.csv").open(encoding="utf-8-sig")):
                p1geo[row["case"]] = (row["dataset"], row["geometry"])
        for f in sorted(p1.glob("*.json")):
            d = read(f)
            key = (d["case"], d["seed"], d["method"])
            ds, g = p1geo[d["case"]]
            d["_dataset"], d["_geometry"] = ds, g
            data[key] = d
    for c in geo:
        if len(geo[c]) != 1:
            raise ValueError("geometry inconsistent for " + c)
        geo[c] = next(iter(geo[c]))
    for c in seeds:
        if len(seeds[c]) != 3:
            raise ValueError("seed count != 3 for " + c)
        seeds[c] = sorted(seeds[c])
    return data, geo, seeds, p1_present


def rank_of(values):
    return {k: 1 + sum(x >= v - EPS for m2, x in values.items() if k != m2) for k, v in values.items()}


def agg_scores(data, geo, seeds, cases, methods):
    """案例内种子均值 → 几何内等权 → 来源内几何等权。"""
    out = {}
    for m in methods:
        gm = {}
        for c in cases:
            vals = [data[c, s, m]["U"] for s in seeds[c]]
            gm.setdefault(geo[c], []).append(np.mean(vals))
        out[m] = float(np.mean([np.mean(v) for v in gm.values()]))
    return out


def paired_analysis(data, geo, seeds, cases, a, b):
    """a − b：几何均值差、95% 区间、胜平负几何、逐种子、经济分解、逐案例逐种子。"""
    rng = np.random.default_rng(BOOT_SEED)
    gd = {}
    for c in cases:
        gd.setdefault(geo[c], []).append([data[c, s, a]["U"] - data[c, s, b]["U"] for s in seeds[c]])
    names = sorted(gd)
    mat = np.asarray([np.mean(gd[g], axis=0) for g in names])
    gmean = mat.mean(axis=1)
    draws = gmean[rng.integers(len(names), size=(DRAWS, len(names)))].mean(axis=1)
    wins, lost = int((gmean > EPS).sum()), int((gmean < -EPS).sum())
    p = float(binomtest(wins, wins + lost).pvalue) if wins + lost else 1.0
    parts = {}
    for key, ek in [("quality", "quality_loss_value_U"), ("distance", "distance_km"), ("refrigeration", "refrigeration_cost_U")]:
        pm = {}
        for c in cases:
            vals = [data[c, s, b]["endpoint"][ek] - data[c, s, a]["endpoint"][ek] for s in seeds[c]]
            pm.setdefault(geo[c], []).append(np.mean(vals))
        parts[key] = float(np.mean([np.mean(v) for v in pm.values()]))
    by_seed = {}
    for s in sorted({x for c in cases for x in seeds[c]}):
        gs = {}
        for c in cases:
            if s not in seeds[c]:
                continue
            gs.setdefault(geo[c], []).append(data[c, s, a]["U"] - data[c, s, b]["U"])
        if gs:
            by_seed[str(s)] = float(np.mean([np.mean(v) for v in gs.values()]))
    loss_a = {c: np.mean([data[c, s, a]["endpoint"]["equivalent_loss_cost_U"] for s in seeds[c]]) for c in cases}
    loss_b = {c: np.mean([data[c, s, b]["endpoint"]["equivalent_loss_cost_U"] for s in seeds[c]]) for c in cases}
    rows = []
    for c in sorted(cases):
        for s in seeds[c]:
            d1, d2 = data[c, s, a], data[c, s, b]
            rows.append(dict(case=c, seed=s, geometry=geo[c][:12], U_a=d1["U"], U_b=d2["U"], delta=d1["U"] - d2["U"]))
    improved_instances = sum(1 for c in cases if loss_a[c] < loss_b[c])
    # 经济损失降低率：逐实例 (L_b − L_a)/L_b ×100%，分母过小标记不可用
    loss_reds = []
    for c in cases:
        if loss_b[c] < 0.5:
            loss_reds.append(None)
        else:
            loss_reds.append(100.0 * (loss_b[c] - loss_a[c]) / loss_b[c])
    usable = [x for x in loss_reds if x is not None]
    # 改善几何数：几何均值 ΔU > EPS 的几何
    improved_geometries = int((gmean > EPS).sum())
    return dict(mean=float(gmean.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                wins=wins, ties=int((np.abs(gmean) <= EPS).sum()), losses=lost, sign_p=p,
                by_seed=by_seed, parts=parts, rows=rows, improved_instances=improved_instances,
                improved_geometries=improved_geometries, total_geometries=len(names),
                loss_reduction_pct=float(np.mean(usable)) if usable else None,
                loss_reduction_unusable=sum(1 for x in loss_reds if x is None))


def main():
    data, geo, seeds, p1_present = load_all()
    print("endpoints loaded:", len(data), "P1 present:", p1_present)
    methods = METHODS18 if p1_present else METHODS16
    panels = ["PDPCDPG-D100", "GH-p100", "XTW-p100(21)", "VRPPDL-p100", "XTW-p100(既有13)",
              "XTW-p100(新8)", "RWVRP-p100", "GH-p100-block2"]
    batch_of = {}
    for name in ("benchmark01", "prospective01", "retail01"):
        base = UNIFIED / "results" / name
        for row in csv.DictReader((base / "endpoints.csv").open(encoding="utf-8-sig")):
            batch_of[row["case"]] = name
    spec_map = {
        "PDPCDPG-D100": ("benchmark01", lambda c: c.startswith("dk_")),
        "GH-p100": ("benchmark01", lambda c: c.startswith("gh_") and not c.endswith("_c101_200")),
        "XTW-p100(21)": (("benchmark01", "prospective01"), lambda c: c.startswith("xtw_")),
        "XTW-p100(既有13)": ("benchmark01", lambda c: c.startswith("xtw_")),
        "XTW-p100(新8)": ("prospective01", lambda c: c.startswith("xtw_")),
        "VRPPDL-p100": ("prospective01", lambda c: c.startswith("vrppdl_")),
        "RWVRP-p100": ("retail01", lambda c: c.startswith("rw_")),
        "GH-p100-block2": ("prospective01", lambda c: c.endswith("_c101_200")),
    }
    scores = {}
    rows = []
    for panel in panels:
        batches, keep = spec_map[panel]
        batches = (batches,) if isinstance(batches, str) else batches
        cases = sorted(c for c in geo if keep(c) and batch_of[c] in batches)
        assert cases, panel
        sc = agg_scores(data, geo, seeds, cases, methods)
        scores[panel] = sc
        rank13 = rank_of({k: sc[k] for k in [OUR] + EXTERNAL})
        rank18 = rank_of(sc)
        for m in methods:
            cat, venue, year, grade = PUB[m]
            rows.append(dict(panel=panel, method=m, category=cat, venue=venue, year=year, grade=grade,
                             mean_U=round(sc[m], 6), rank13=rank13.get(m, ""), rank18=rank18.get(m, "")))
    with (OUT / "表3_四数据集统一18法比较表.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 配对增量：完整方法 vs MASK、SLOT、ORDER_SPLIT_D（若 MASK/SLOT 存在）
    if p1_present:
        inc_rows = []
        for panel in ["PDPCDPG-D100", "GH-p100", "XTW-p100(21)", "VRPPDL-p100", "RWVRP-p100", "GH-p100-block2"]:
            batches, keep = spec_map[panel]
            batches = (batches,) if isinstance(batches, str) else batches
            cases = sorted(c for c in geo if keep(c) and batch_of[c] in batches)
            for b in ("MASK", "SLOT", "ORDER_SPLIT_D"):
                r = paired_analysis(data, geo, seeds, cases, OUR, b)
                lo, hi = r["ci95"]
                inc_rows.append(dict(panel=panel, comparator=b, delta_U=round(r["mean"], 6),
                                     ci95_lo=round(lo, 6), ci95_hi=round(hi, 6),
                                     wins=r["wins"], ties=r["ties"], losses=r["losses"],
                                     improved_instances=r["improved_instances"], total_instances=len(cases),
                                     improved_geometries=r["improved_geometries"], total_geometries=r["total_geometries"],
                                     loss_reduction_pct=r["loss_reduction_pct"],
                                     loss_reduction_unusable=r["loss_reduction_unusable"],
                                     by_seed=";".join(f"{k}:{v:.4f}" for k, v in r["by_seed"].items()),
                                     quality=round(r["parts"]["quality"], 6), distance=round(r["parts"]["distance"], 6),
                                     refrigeration=round(r["parts"]["refrigeration"], 6), sign_p=r["sign_p"]))
        with (OUT / "表6b_相对MASK与SLOT配对增量表.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inc_rows[0].keys()))
            w.writeheader()
            w.writerows(inc_rows)
        # 逐案例逐种子端点表
        ep = []
        for key in sorted(data):
            c, s, m = key
            d = data[key]
            ep.append(dict(case=c, seed=s, method=m, dataset=d["_dataset"], geometry=d["_geometry"],
                           U=d["U"], initial_U=d["initial_U"], selected_s=d["selected_s"],
                           vehicles=d["endpoint"]["used_vehicles"], distance_km=d["endpoint"]["distance_km"],
                           quality_loss_U=d["endpoint"]["quality_loss_value_U"],
                           refrigeration_U=d["endpoint"]["refrigeration_cost_U"],
                           minimum_quality=d["endpoint"]["minimum_quality"], status=d["status"]))
        with (OUT / "逐案例逐种子完整端点表.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ep[0].keys()))
            w.writeheader()
            w.writerows(ep)
        print("paired increment table + endpoint table written")
    print("panels done:", {p: len(scores[p]) for p in panels})


if __name__ == "__main__":
    main()
