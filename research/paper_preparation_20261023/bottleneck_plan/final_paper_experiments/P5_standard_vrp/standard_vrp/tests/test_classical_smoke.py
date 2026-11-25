# -*- coding: utf-8 -*-
"""标准 ALNS/VNS 冒烟测试（小实例，不退化且能改善）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw, better_lex)
from standard_vrp.solver.standard_classical import alns_search, vns_search, nn_witness_std


def test_cvrp():
    coords = {0: (0, 0), 1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 6: (5, 30), 7: (-8, 15)}
    demands = {0: 0, 1: 10, 2: 12, 3: 9, 4: 15, 5: 14, 6: 8, 7: 11}
    inst = CVRPInstance("alns_cvrp", 7, 30, coords, demands)
    w = nn_witness_std(inst)
    ok0, d0, _, _ = evaluate_cvrp(inst, w, rounded=True)
    for name, fn in (("ALNS", alns_search), ("VNS", vns_search)):
        out = fn(inst, seconds=3, seed=1, objective="cvrp")
        ok, d, _, _ = evaluate_cvrp(inst, out["routes"], rounded=True)
        assert ok and d <= d0 + 1e-9, (name, d, d0)
        print(f"CVRP {name}: {d0:.2f} -> {d:.2f}")


def test_vrptw():
    coords = {0: (0, 0), 1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 6: (5, 30), 7: (-8, 15)}
    inst = VRPTWInstance("alns_tw", 7, 30, coords,
                         {i: 10 for i in range(1, 8)},
                         {1: (0, 30), 2: (20, 60), 3: (0, 30), 4: (0, 60), 5: (30, 90), 6: (0, 100), 7: (0, 100)},
                         {i: 5 for i in range(1, 8)}, (0, 200), None)
    w = nn_witness_std(inst)
    ok0, k0, d0, _, _ = evaluate_vrptw(inst, w)
    for name, fn in (("ALNS", alns_search), ("VNS", vns_search)):
        out = fn(inst, seconds=3, seed=1, objective="vrptw")
        ok, k, d, _, _ = evaluate_vrptw(inst, out["routes"])
        assert ok and better_lex(k, d, k0, d0) or (k == k0 and abs(d - d0) < 1e-9), (name, k, d, k0, d0)
        print(f"VRPTW {name}: (K={k0},D={d0:.2f}) -> (K={k},D={d:.2f})")


if __name__ == "__main__":
    test_cvrp()
    test_vrptw()
    print("STANDARD CLASSICAL SMOKE PASSED")
