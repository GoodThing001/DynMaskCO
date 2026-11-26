# -*- coding: utf-8 -*-
"""固定候选数量诊断分析：4 种排序的插位质量对照（表 V2-diag + 图 F12）。

输入：<diag_dir>/out/*_DIAG.json
输出：tables/V2-diag_fixed_candidates.csv、tables/analysis_diag.json
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
VARIANTS = ["distance", "heat_raw", "heat_calibrated", "heat_norm"]


def main(diag_dir=None):
    root = Path(diag_dir) if diag_dir else V2 / "diagnostics"
    files = sorted(root.rglob("*_DIAG.json"))
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if not rows:
        raise SystemExit("no diagnostic rows in " + str(root))
    per = {}
    for r in rows:
        key = (r["case"], r["seed"])
        per.setdefault(key, {})[r["variants"]["distance"]["final"]["distance"]
                               if "variants" in r else 0] = r
    table = []
    for r in rows:
        v = r["variants"]
        entry = dict(case=r["case"], seed=r["seed"], objective=r["objective"], n=r["dimension"],
                     forward_fix_s=round(r["forward_s"]["v2_fix"], 4),
                     forward_norm_s=round(r["forward_s"]["v2_norm"], 4),
                     initial_distance=r["initial"]["distance"],
                     distance_after_removal=r["after_removal"]["distance"])
        for k in VARIANTS:
            d = v[k]
            entry[k + "_mean_regret"] = round(d["mean_regret"], 6)
            entry[k + "_recall1"] = round(d["recall_at_1"], 4)
            entry[k + "_recall3"] = round(d["recall_at_3"], 4)
            entry[k + "_final"] = round(d["final"]["distance"], 4)
            entry[k + "_final_ok"] = d["final"]["feasible"]
        table.append(entry)
    dest = V2 / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "V2-diag_fixed_candidates.csv").open("w", newline="",
                                                      encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    agg = {}
    for k in VARIANTS:
        agg[k] = dict(
            n=len(table),
            mean_regret=round(st.mean([t[k + "_mean_regret"] for t in table]), 6),
            median_regret=round(st.median([t[k + "_mean_regret"] for t in table]), 6),
            mean_recall1=round(st.mean([t[k + "_recall1"] for t in table]), 4),
            mean_recall3=round(st.mean([t[k + "_recall3"] for t in table]), 4),
            mean_final=round(st.mean([t[k + "_final"] for t in table]), 4),
            better_than_distance=sum(1 for t in table
                                     if t[k + "_final"] < t["distance_final"] - 1e-9),
            worse_than_distance=sum(1 for t in table
                                    if t[k + "_final"] > t["distance_final"] + 1e-9))
    # 配对：热图类 vs 距离
    pairs = {}
    for k in VARIANTS[1:]:
        diffs = [100.0 * (t[k + "_final"] - t["distance_final"]) / t["distance_final"]
                 for t in table if t["distance_final"]]
        pairs["%s_vs_distance" % k] = dict(
            n=len(diffs), mean_delta_pct=round(st.mean(diffs), 4),
            median_delta_pct=round(st.median(diffs), 4),
            better=sum(1 for d in diffs if d < 0), worse=sum(1 for d in diffs if d > 0))
    payload = dict(rows=table, per_variant=agg, pairs=pairs,
                   registration=dict(mask_size=40, order="fixed shuffle",
                                     candidate_set="all feasible slots + new route",
                                     variants=VARIANTS))
    (dest / "analysis_diag.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(dict(endpoints=len(table), per_variant=agg, pairs=pairs),
                     ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
