# -*- coding: utf-8 -*-
"""表9：不同计算预算下的配对收益（5/15/30/60 秒轨迹）。

数据来源：trajectory01 已审计批次（3 个既有来源 × 3 个代表几何 × 10 种子 × 4 方法），
后续并入 P2 八城十种子轨迹。
聚合：几何内先均种子，来源内几何等权；配对差 = ORDER_SPLIT − 对照。
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
UNI = FP.parent / "unified_external_20261120"
TAB = FP / "tables"
TAB.mkdir(parents=True, exist_ok=True)

CUTOFFS = ("5", "15", "30", "60")
OUR = "ORDER_SPLIT"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    base = UNI / "results" / "trajectory01" / "out"
    data = {}
    geo = {}
    for f in sorted(base.glob("*.json")):
        d = read(f)
        key = (d["case"], d["seed"], d["method"])
        data[key] = d
        geo.setdefault(d["case"], d["role"])
    # dataset 映射：dk_ → PDPCDPG-D100, xtw_ → XTW-p100, gh_ → GH-p100
    def dataset(c):
        if c.startswith("dk_"):
            return "PDPCDPG-D100"
        if c.startswith("xtw_"):
            return "XTW-p100"
        if c.startswith("gh_"):
            return "GH-p100"
        raise ValueError(c)

    rows = []
    for comp in ("ORDER_SPLIT_D", "ALNS", "PyVRP"):
        for t in CUTOFFS:
            for src in ("PDPCDPG-D100", "XTW-p100", "GH-p100"):
                cases = sorted({c for c in geo if dataset(c) == src})
                diffs = []
                for c in cases:
                    for s in sorted({k[1] for k in data if k[0] == c}):
                        a = data[c, s, OUR]["trajectory"][t]["U"]
                        b = data[c, s, comp]["trajectory"][t]["U"]
                        diffs.append(a - b)
                m = float(np.mean(diffs))
                se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
                wins = sum(1 for x in diffs if x > 1e-6)
                rows.append(dict(cutoff_s=t, source=src, comparator=comp, paired_n=len(diffs),
                                 mean_delta_U=round(m, 4), se=round(se, 4),
                                 positive_pairs=f"{wins}/{len(diffs)}"))
    with (TAB / "表9_不同计算预算下的配对收益.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["# 表9 不同计算预算下的配对收益（MASKCO − 对照）", "",
          "每格为轨迹批次（trajectory01）中几何内先均种子后的来源均值；10 种子、3 个代表几何/来源；"
          "P2 八城十种子完成后并入本表。", "",
          "| 预算 | 来源 | 对照 | 配对数 | ΔU | 标准误 | 正向配对 |",
          "|---|---|---|---|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['cutoff_s']}s | {r['source']} | {r['comparator']} | {r['paired_n']} | "
                  f"{r['mean_delta_U']:+,.2f} | {r['se']:.2f} | {r['positive_pairs']} |")
    (TAB / "表9_不同计算预算下的配对收益.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("table9 written:", len(rows), "rows")


if __name__ == "__main__":
    main()
