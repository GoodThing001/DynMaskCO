# -*- coding: utf-8 -*-
"""P5-B 全量前预检：56 例 Solomon 的共同初解（nn_witness）可行性。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import VRPTWInstance, evaluate_vrptw
from standard_vrp.solver.standard_pool_search import nn_witness

bad = []
rows = []
d = HERE / "data/solomon/instances"
for f in sorted(d.glob("*.vrp")):
    inst = VRPTWInstance.parse_solomon(f)
    w = nn_witness(inst)
    ok, k, dist, _, _ = evaluate_vrptw(inst, w)
    rows.append((f.stem, ok, k, dist))
    if not ok:
        bad.append(f.stem)
for name, ok, k, dist in rows:
    print("%-6s witness feasible=%s K=%d D=%.2f (limit=%s)" % (name, ok, k, dist, inst.max_vehicles if False else "25"))
print("INFEASIBLE:", bad)
