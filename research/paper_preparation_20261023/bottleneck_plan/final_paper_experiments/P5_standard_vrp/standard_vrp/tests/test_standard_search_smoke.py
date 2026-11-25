# -*- coding: utf-8 -*-
"""标准求解器冒烟：小实例（无神经前向）距离目标与字典序目标搜索正确性。"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw)
from standard_vrp.solver.standard_pool_search import standard_search, nn_witness


def test_cvrp_search():
    coords = {0: (0, 0), 1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 6: (5, 30), 7: (-8, 15)}
    demands = {0: 0, 1: 10, 2: 12, 3: 9, 4: 15, 5: 14, 6: 8, 7: 11}
    inst = CVRPInstance("smoke_cvrp", 7, 30, coords, demands)
    w = nn_witness(inst)
    ok0, d0, _, _ = evaluate_cvrp(inst, w, rounded=True)
    assert ok0
    t0 = time.perf_counter()
    out = standard_search(inst, seconds=3, seed=1, objective="cvrp")
    elapsed = time.perf_counter() - t0
    assert out["distance"] <= d0 + 1e-9, (out, d0)
    assert elapsed < 3.5, "search overshot budget: %.2fs" % elapsed
    print("CVRP search: witness %.2f -> %.2f (epochs %d, %.1fs)" % (d0, out["distance"], out["epochs"], elapsed))


def test_vrptw_search():
    coords = {0: (0, 0), 1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 6: (5, 30), 7: (-8, 15)}
    inst = VRPTWInstance("smoke_vrptw", 7, 30, coords,
                         {i: 10 for i in range(1, 8)},
                         {1: (0, 30), 2: (20, 60), 3: (0, 30), 4: (0, 60), 5: (30, 90), 6: (0, 100), 7: (0, 100)},
                         {i: 5 for i in range(1, 8)}, (0, 200), None)
    w = nn_witness(inst)
    ok0, k0, d0, _, _ = evaluate_vrptw(inst, w)
    assert ok0
    t0 = time.perf_counter()
    out = standard_search(inst, seconds=3, seed=1, objective="vrptw")
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.5, "search overshot budget: %.2fs" % elapsed
    ok, k, d, _, _ = evaluate_vrptw(inst, out["routes"])
    assert ok
    from standard_vrp.core.standard_evaluator import better_lex
    assert better_lex(k, d, k0, d0) or (k == k0 and abs(d - d0) < 1e-9), (k, d, k0, d0)
    print("VRPTW search: witness (K=%d,D=%.2f) -> (K=%d,D=%.2f) (epochs %d, %.1fs)"
          % (k0, d0, k, d, out["epochs"], elapsed))


if __name__ == "__main__":
    test_cvrp_search()
    test_vrptw_search()
    print("STANDARD SOLVER SMOKE PASSED")
