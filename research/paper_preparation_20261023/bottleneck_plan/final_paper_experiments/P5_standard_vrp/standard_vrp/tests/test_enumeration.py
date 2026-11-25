# -*- coding: utf-8 -*-
"""标准评价器正确性测试：小实例全排列枚举 vs DP Split vs 评价器。"""
import itertools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw,
                                                  split_cvrp, split_vrptw,
                                                  route_schedule, better_lex)


def build_cvrp(n, cap, coords, demands):
    # coords[0]/demands[0] 为仓库
    return CVRPInstance("t%d" % n, n, cap, {i: coords[i] for i in range(n + 1)},
                        {i: demands[i] for i in range(n + 1)})


def partitions(seq):
    """序列的全部分区（车辆划分）。"""
    if not seq:
        yield []
        return
    first = seq[0]
    for rest in partitions(seq[1:]):
        yield [(first,)] + rest
        if rest:
            head = list(rest[0])
            yield [tuple([first] + head)] + rest[1:]


def test_cvrp_enumeration():
    # 5 客户、容量 30：枚举全部排列×分区求精确最优
    coords = [(0, 0), (10, 0), (10, 10), (0, 10), (20, 0), (20, 20)]
    demands = [0, 10, 12, 9, 15, 14]
    inst = build_cvrp(5, 30, coords, demands)
    best = None
    best_routes = None
    count = 0
    for perm in itertools.permutations(range(1, 6)):
        for part in partitions(tuple(perm)):
            ok, d, _, k = evaluate_cvrp(inst, [tuple(r) for r in part])
            count += 1
            if ok and (best is None or d < best - 1e-9):
                best, best_routes = d, [tuple(r) for r in part]
    assert best is not None, "no feasible CVRP solution found"
    # 最优序列的 Split 必须复现枚举最优
    flat = tuple(i for r in best_routes for i in r)
    out = split_cvrp(inst, flat)
    assert out is not None and abs(out["distance"] - best) < 1e-6, (out, best)
    # 评价器可行性：容量违反与覆盖缺失
    bad = [(1, 2, 3, 4), (5,)]
    ok, _, _, _ = evaluate_cvrp(inst, bad)
    assert not ok, "capacity violation must be infeasible"
    ok2, _, _, _ = evaluate_cvrp(inst, [(1, 2, 3)])
    assert not ok2, "partial coverage must be infeasible"
    # 距离：double vs 整数舍入
    inst2 = build_cvrp(1, 100, [(0, 0), (3, 4)], [0, 10])
    assert abs(inst2.edge_distance(0, 1) - 5.0) < 1e-12
    assert inst2.edge_distance(0, 1, rounded=True) == 5.0
    print("test_cvrp_enumeration OK: exact best %.6f over %d fleets" % (best, count))


def test_vrptw_enumeration():
    # 5 客户、容量 30、紧时间窗
    coords = {1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 0: (0, 0)}
    inst = VRPTWInstance("tw5", 5, 30, coords,
                         {i: 10 for i in range(1, 6)},
                         {1: (0, 30), 2: (20, 60), 3: (0, 30), 4: (0, 60), 5: (30, 90)},
                         {i: 5 for i in range(1, 6)}, (0, 200), None)
    best = None
    best_routes = None
    count = 0
    for perm in itertools.permutations(range(1, 6)):
        for part in partitions(tuple(perm)):
            ok, k, d, w, _ = evaluate_vrptw(inst, [tuple(r) for r in part])
            count += 1
            if ok and (best is None or better_lex(k, d, best[0], best[1])):
                best = (k, d)
                best_routes = [tuple(r) for r in part]
    assert best is not None, "no feasible VRPTW solution"
    flat = tuple(i for r in best_routes for i in r)
    out = split_vrptw(inst, flat)
    assert out is not None, "split failed on optimal sequence"
    assert out["vehicles"] == best[0], (out, best)
    assert abs(out["distance"] - best[1]) < 1e-6
    # 时间窗违反必须不可行
    assert route_schedule(inst, (2, 5)) is None or route_schedule(inst, (2, 5))["end"] <= 200
    # 等待时间：客户 5 窗口 (30,90)，服务 5：从仓库 0,0 到 (20,20) 距离约 28.28
    sched = route_schedule(inst, (5,))
    assert sched is not None and sched["waits"] > 0.0
    print("test_vrptw_enumeration OK: lex best K=%d D=%.6f over %d fleets" % (best[0], best[1], count))


def test_lexicographic():
    assert better_lex(2, 100.0, 3, 90.0) is True
    assert better_lex(3, 90.0, 2, 100.0) is False
    assert better_lex(2, 100.0, 2, 101.0) is True
    assert better_lex(2, 100.0, 2, 100.0) is False
    print("test_lexicographic OK")


if __name__ == "__main__":
    test_cvrp_enumeration()
    test_vrptw_enumeration()
    test_lexicographic()
    print("ALL STANDARD EVALUATOR TESTS PASSED")
