# -*- coding: utf-8 -*-
"""P2 车队结构统计（本地严格评价器重放：车辆运行时间、等待时间、装载率）。"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P2 = FP / "P2_retail_analysis"
RECENT = FP.parent / "recent5_20261120"
EXT = FP.parent.parent.parent / "C-VRP_Cold-chainVehicleRoutingProblem"
sys.path.insert(0, str(EXT / "tools"))
sys.path.insert(0, str(RECENT))
from recent_independent import original_model  # noqa: E402

CITIES = ["rw_zg170_p100", "rw_br100_p100", "rw_cp150_p100", "rw_ru120_p100",
          "rw_sg200_p100", "rw_me340_p100", "rw_ca240_p100", "rw_sf300_p100"]
NAMES = {"rw_zg170_p100": "Zagreb", "rw_br100_p100": "Brasilia", "rw_cp150_p100": "Copenhagen",
         "rw_ru120_p100": "Ruhr", "rw_sg200_p100": "Singapore", "rw_me340_p100": "Melbourne",
         "rw_ca240_p100": "Cairo", "rw_sf300_p100": "SF Bay Area"}
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS"]
SEEDS = [65001, 65002, 65003, 65004, 65005, 65006, 65007, 65008, 65009, 65010]
CAPACITY_T = 20.0


def main():
    folder = P2 / "results" / "out"
    models = {}
    for c in CITIES:
        m, _, _ = original_model(P2 / "inputs" / (c + ".json"), c)
        models[c] = m
        demand_t = sum(n[3] for n in m.ins["nodes"]) / 1000.0
        models[c].demand_t = demand_t
    rows = []
    for c in CITIES:
        for meth in METHODS:
            for s in SEEDS:
                p = folder / f"{c}_{s}_{meth}.json"
                if not p.is_file():
                    # 文件名为 idx_method.json
                    continue
    # 端点文件名是 idx_method.json，按内容索引
    idx = {}
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        idx[(d["case"], d["seed"], d["method"])] = d
    for c in CITIES:
        for meth in METHODS:
            recs = []
            for s in SEEDS:
                d = idx[(c, s, meth)]
                scores = models[c].solution(d["routes"])
                k = len(d["routes"])
                run_min = sum(r.returned - r.departure for r in scores)
                wait_min = sum(st["wait"] for r in scores for st in r.stops)
                recs.append(dict(k=k, run_min=run_min, wait_min=wait_min,
                                 load_rate=models[c].demand_t / (k * CAPACITY_T),
                                 customers_per_vehicle=100.0 / k,
                                 distance_km=sum(r.distance for r in scores)))
            rows.append(dict(city=NAMES[c], case=c, method=meth,
                             initial_vehicles=len(idx[(c, SEEDS[0], meth)]["initial_routes"]),
                             final_vehicles=round(float(np.mean([r["k"] for r in recs])), 2),
                             load_rate=round(float(np.mean([r["load_rate"] for r in recs])), 4),
                             customers_per_vehicle=round(float(np.mean([r["customers_per_vehicle"] for r in recs])), 2),
                             total_distance_km=round(float(np.mean([r["distance_km"] for r in recs])), 2),
                             total_run_min=round(float(np.mean([r["run_min"] for r in recs])), 2),
                             total_wait_min=round(float(np.mean([r["wait_min"] for r in recs])), 2)))
    with (P2 / "P2_车队结构统计.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("P2 fleet structure written:", len(rows), "rows")


if __name__ == "__main__":
    main()
