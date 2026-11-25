# -*- coding: utf-8 -*-
"""P3 双因素消融分析：A/B/C/D 四类配对收益与交互项。

A = HYBRID − HYBRID_D          普通经济重插中的学习引导增益
B = ORDER_SPLIT − ORDER_SPLIT_D 完整框架中的学习引导增益
C = ORDER_SPLIT − HYBRID        学习引导下的跨车分车增益
D = ORDER_SPLIT_D − HYBRID_D    纯距离下的跨车分车增益
交互 = B − A = C − D

统计全部从逐 (case, seed) 配对差聚合：案例内均种子 → 来源内几何等权 → 三来源等权；
95% 区间为几何整簇有放回重采样 20,000 次（种子 20261127）。
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P3 = FP / "P3_factorial_ablation"
UNI = FP.parent / "unified_external_20261120"
EPS = 1e-6
BOOT_SEED = 20261127
DRAWS = 20000

SOURCES = {
    "VRPPDL-p100": lambda c: c.startswith("vrppdl_"),
    "XTW-p100(新8)": lambda c: c.startswith("xtw_"),
    "RWVRP-p100": lambda c: c.startswith("rw_"),
}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_unified():
    data, geo, seeds = {}, {}, {}
    for name in ("prospective01", "retail01"):
        base = UNI / "results" / name
        meta = {}
        for row in csv.DictReader((base / "endpoints.csv").open(encoding="utf-8-sig")):
            meta[(row["case"], int(row["seed"]), row["method"])] = row["geometry"]
        for f in sorted((base / "out").glob("*.json")):
            d = read(f)
            key = (d["case"], d["seed"], d["method"])
            if d["method"] not in ("ORDER_SPLIT", "ORDER_SPLIT_D"):
                continue
            d["_geometry"] = meta[key]
            data[key] = d
            geo[d["case"]] = meta[key]
            seeds.setdefault(d["case"], set()).add(d["seed"])
    return data, geo, {c: sorted(v) for c, v in seeds.items()}


def load_p3():
    base = P3 / "results" / "out"
    data = {}
    for f in sorted(base.glob("*.json")):
        d = read(f)
        data[(d["case"], d["seed"], d["method"])] = d
    return data


def source_of(case):
    for name, keep in SOURCES.items():
        if keep(case):
            return name
    raise ValueError(case)


def agg_diff(data, geo, seeds, cases, a, b):
    rng = np.random.default_rng(BOOT_SEED)
    gd = {}
    for c in cases:
        gd.setdefault(geo[c], []).append([data[c, s, a]["U"] - data[c, s, b]["U"] for s in seeds[c]])
    names = sorted(gd)
    mat = np.asarray([np.mean(gd[g], axis=0) for g in names])
    gmean = mat.mean(axis=1)
    draws = gmean[rng.integers(len(names), size=(DRAWS, len(names)))].mean(axis=1)
    units_pos = sum(1 for c in cases for s in seeds[c] if data[c, s, a]["U"] - data[c, s, b]["U"] > EPS)
    units_total = sum(len(seeds[c]) for c in cases)
    wins = int((gmean > EPS).sum())
    lost = int((gmean < -EPS).sum())
    p = float(binomtest(wins, wins + lost).pvalue) if wins + lost else 1.0
    by_seed = {}
    for s in sorted({x for c in cases for x in seeds[c]}):
        gs = {}
        for c in cases:
            if s in seeds[c]:
                gs.setdefault(geo[c], []).append(data[c, s, a]["U"] - data[c, s, b]["U"])
        if gs:
            by_seed[str(s)] = float(np.mean([np.mean(v) for v in gs.values()]))
    return dict(mean=float(gmean.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                wins=wins, ties=int((np.abs(gmean) <= EPS).sum()), losses=lost, sign_p=p,
                units_pos=units_pos, units_total=units_total, by_seed=by_seed,
                by_geometry=dict(zip(names, gmean.tolist())))


def main():
    unified, geo, seeds = load_unified()
    p3 = load_p3()
    print("unified prospective endpoints:", len(unified), "P3 endpoints:", len(p3))
    if not p3:
        print("P3 not run yet.")
        return
    cases = sorted(c for c in geo if any(keep(c) for keep in SOURCES.values()))
    assert len(cases) == 26
    data = {**unified, **p3}
    rows = []
    for (c, s) in [(c, s) for c in cases for s in seeds[c]]:
        u = {m: data[c, s, m]["U"] for m in ("HYBRID", "HYBRID_D", "ORDER_SPLIT", "ORDER_SPLIT_D")}
        rows.append(dict(case=c, seed=s, source=source_of(c), geometry=geo[c][:12],
                         HYBRID=u["HYBRID"], HYBRID_D=u["HYBRID_D"],
                         ORDER_SPLIT=u["ORDER_SPLIT"], ORDER_SPLIT_D=u["ORDER_SPLIT_D"],
                         A=u["HYBRID"] - u["HYBRID_D"], B=u["ORDER_SPLIT"] - u["ORDER_SPLIT_D"],
                         C=u["ORDER_SPLIT"] - u["HYBRID"], D=u["ORDER_SPLIT_D"] - u["HYBRID_D"],
                         interaction=u["ORDER_SPLIT"] - u["ORDER_SPLIT_D"] - (u["HYBRID"] - u["HYBRID_D"])))
    with (P3 / "P3_逐案例配对结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 三来源等权
    summaries = []
    for src in SOURCES:
        sc = [c for c in cases if source_of(c) == src]
        for a, b, label in [("HYBRID", "HYBRID_D", "A_learning_without_split"),
                            ("ORDER_SPLIT", "ORDER_SPLIT_D", "B_learning_with_split"),
                            ("ORDER_SPLIT", "HYBRID", "C_split_with_learning"),
                            ("ORDER_SPLIT_D", "HYBRID_D", "D_split_without_learning")]:
            r = agg_diff(data, geo, seeds, sc, a, b)
            lo, hi = r["ci95"]
            summaries.append(dict(source=src, quantity=label, mean=round(r["mean"], 6),
                                  ci95_lo=round(float(lo), 6), ci95_hi=round(float(hi), 6),
                                  wins=r["wins"], ties=r["ties"], losses=r["losses"],
                                  units_pos=r["units_pos"], units_total=r["units_total"],
                                  by_seed=";".join(f"{k}:{v:.4f}" for k, v in r["by_seed"].items())))
    # 交互 = B − A（逐案例逐种子后聚合）
    for src in SOURCES:
        sc = [c for c in cases if source_of(c) == src]
        sub = [r for r in rows if r["source"] == src]
        rng = np.random.default_rng(BOOT_SEED)
        gd = {}
        for r in sub:
            gd.setdefault(r["geometry"], []).append(r["interaction"])
        names = sorted(gd)
        mat = np.asarray([np.mean(gd[g]) for g in names])
        draws = mat[rng.integers(len(names), size=(DRAWS, len(names)))].mean(axis=1)
        pos = sum(1 for r in sub if r["interaction"] > EPS)
        summaries.append(dict(source=src, quantity="interaction_B_minus_A",
                              mean=round(float(mat.mean()), 6),
                              ci95_lo=round(float(np.quantile(draws, .025)), 6),
                              ci95_hi=round(float(np.quantile(draws, .975)), 6),
                              wins=int((mat > EPS).sum()), ties=int((np.abs(mat) <= EPS).sum()),
                              losses=int((mat < -EPS).sum()), units_pos=pos, units_total=len(sub),
                              by_seed=""))
    # 三来源等权
    eq_rows = []
    for q in ("A_learning_without_split", "B_learning_with_split", "C_split_with_learning",
              "D_split_without_learning", "interaction_B_minus_A"):
        srcs = [r for r in summaries if r["quantity"] == q]
        m = float(np.mean([r["mean"] for r in srcs]))
        eq_rows.append(dict(quantity=q, equal_source_mean=round(m, 6),
                            source_means=";".join(f"{r['source']}:{r['mean']}" for r in srcs)))
    with (P3 / "P3_双因素消融汇总表.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    with (P3 / "P3_三来源等权汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq_rows[0].keys()))
        w.writeheader()
        w.writerows(eq_rows)
    md = build_md(summaries, eq_rows)
    (P3 / "P3_双因素消融汇总表.md").write_text(md, encoding="utf-8")
    print("P3 analysis written")


def build_md(summaries, eq_rows):
    lines = ["# P3 双因素消融汇总", "",
             "A = HYBRID − HYBRID_D；B = ORDER_SPLIT − ORDER_SPLIT_D；C = ORDER_SPLIT − HYBRID；D = ORDER_SPLIT_D − HYBRID_D；",
             "交互 = B − A（= C − D）。逐 (case, seed) 配对差聚合；95% 区间为几何整簇重采样 20,000 次（种子 20261127）。", "",
             "| 来源 | 量 | 均值 ΔU | 95% 区间 | 胜/平/负几何 | 改善单位 | 逐种子 |",
             "|---|---|---:|---|---|---|---|"]
    for r in summaries:
        lo, hi = r["ci95_lo"], r["ci95_hi"]
        lines.append(f"| {r['source']} | {r['quantity']} | {r['mean']:+,.4f} | [{lo:+,.4f}, {hi:+,.4f}] | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} | {r['units_pos']}/{r['units_total']} | {r['by_seed']} |")
    lines += ["", "| 量 | 三来源等权均值 | 来源分解 |", "|---|---|---|"]
    for r in eq_rows:
        lines.append(f"| {r['quantity']} | {r['equal_source_mean']:+,.4f} | {r['source_means']} |")
    lines += ["", "跨零的区间表示该量在当前来源方向不稳定；正均值不直接表述为统计显著。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
