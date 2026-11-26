# -*- coding: utf-8 -*-
"""外部基线（PyVRP VRPTW 168 端点）对照表：与官方 BKS 及本方法同实例结果比较。

输出 tables/V2-ext_pyvrp_vrptw.csv 与 tables/analysis_external.json。
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent


def main(out_dir=None, formal_dir=None):
    ext_root = Path(out_dir) if out_dir else V2 / "external_results/pyvrp_vrptw"
    rows = []
    for f in sorted(ext_root.rglob("*.json")):
        if "phase" in f.name or "status" in f.name or "audit" in f.name:
            continue
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    bks = {r["instance"]: r for r in
           json.loads((V2 / "data/solomon_bks_derived.json").read_text(encoding="utf-8"))["rows"]}
    per = {}
    for r in rows:
        per.setdefault((r["case"], r["seed"]), []).append(r)
    table = []
    for (case, seed), items in sorted(per.items()):
        r = items[0]
        b = bks.get(case, {})
        table.append(dict(instance=case, seed=seed, pyvrp_K=r["vehicles"],
                          pyvrp_D=round(r["distance"], 4), bks_K=b.get("k_bks"),
                          bks_D=b.get("distance_double"),
                          K_diff=r["vehicles"] - (b.get("k_bks") or 0),
                          D_delta=round(r["distance"] - b["distance_double"], 4)
                          if b.get("distance_double") else None,
                          online_s=round(r["online_s"], 2), status=r["status"]))
    dest = V2 / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "V2-ext_pyvrp_vrptw.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    kd = [t["K_diff"] for t in table]
    same = [t["D_delta"] for t in table if t["K_diff"] == 0]
    summary = dict(endpoints=len(table), cells=len(per),
                   K_better=sum(1 for x in kd if x < 0), K_tie=sum(1 for x in kd if x == 0),
                   K_worse=sum(1 for x in kd if x > 0),
                   mean_D_delta_sameK=round(st.mean(same), 4) if same else None,
                   median_D_delta_sameK=round(st.median(same), 4) if same else None,
                   mean_online_s=round(st.mean([t["online_s"] for t in table]), 3),
                   max_online_s=max(t["online_s"] for t in table),
                   statuses={s: sum(1 for t in table if t["status"] == s)
                             for s in {t["status"] for t in table}})
    (dest / "analysis_external.json").write_text(
        json.dumps(dict(summary=summary, rows=table), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
