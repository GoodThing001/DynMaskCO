# -*- coding: utf-8 -*-
"""车队约束（Uchoa X 官方 k）修复验证：100 例 witness 可行性 + X-n101-k25 60s 搜索。"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp
from standard_vrp.solver.standard_pool_search import standard_search, nn_witness

d = HERE / "data/cvrplib/instances"
bad = []
for f in sorted(d.glob("X-n*.vrp"), key=lambda p: int(p.stem.split("-n")[1].split("-")[0])):
    inst = CVRPInstance.parse_vrp(f.read_text(encoding="utf-8"))
    w = nn_witness(inst)
    ok, dist, loads, k = evaluate_cvrp(inst, w, rounded=True)
    if not ok:
        bad.append((f.stem, k, inst.max_vehicles))
print("witness infeasible:", bad[:10], "count", len(bad))

inst = CVRPInstance.parse_vrp((d / "X-n101-k25.vrp").read_text(encoding="utf-8"))
print("X-n101-k25 fleet:", inst.max_vehicles)
t0 = time.perf_counter()
out = standard_search(inst, seconds=60, seed=68001, objective="cvrp", split_enabled=False, slot=True)
el = time.perf_counter() - t0
ok, dist, loads, k = evaluate_cvrp(inst, out["routes"], rounded=True)
print("60s search: ok=%s K=%d D=%.1f elapsed=%.1f (BKS 27591)" % (ok, k, dist, el))
