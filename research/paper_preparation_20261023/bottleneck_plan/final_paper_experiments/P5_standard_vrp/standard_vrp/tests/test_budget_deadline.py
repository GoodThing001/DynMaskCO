# -*- coding: utf-8 -*-
"""60 秒预算回归：X-n101-k25 DIST_SLOT 语义（无热图全位置评价）不得超时。

对应远端 P5-A smoke 的失败单元（online_s=62.99 > 60）：
轮体截止检查只在轮首，最后一轮 3.8s 的 body 溢出预算。修复后要求
elapsed <= seconds + 0.3 且结果不差于最近邻见证。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from standard_vrp.core.standard_evaluator import CVRPInstance, evaluate_cvrp
from standard_vrp.solver.standard_pool_search import standard_search, nn_witness
from standard_vrp.solver.standard_classical import vns_search, alns_search


def test_vns_budget():
    """VNS 大实例 30 秒预算：邻域扫描 O(n^3) 需轮内截止，不得溢出。"""
    candidates = [HERE.parent.parent / "data/cvrplib/instances/X-n502-k39.vrp",
                  HERE.parent.parent / "cvrplib/instances/X-n502-k39.vrp"]
    vrp = next(p for p in candidates if p.exists())
    inst = CVRPInstance.parse_vrp(vrp.read_text(encoding="utf-8"))
    w = nn_witness(inst)
    ok0, d0, _, _ = evaluate_cvrp(inst, w, rounded=True)
    assert ok0
    t0 = time.perf_counter()
    out = vns_search(inst, seconds=30, seed=68001, objective="cvrp")
    elapsed = time.perf_counter() - t0
    ok, d, _, k = evaluate_cvrp(inst, out["routes"], rounded=True)
    assert ok and d <= d0 + 1e-9, (ok, d, d0)
    assert elapsed <= 30.3, "VNS budget overshoot: %.2fs" % elapsed
    print("X-n502-k39 VNS 30s: witness %.2f -> %.2f (K=%d, iters %d, %.2fs <= 30.3)"
          % (d0, d, k, out["iterations"], elapsed))


def test_alns_budget():
    """ALNS 大实例 30 秒预算：修复轮内截止（n=247 单轮曾溢出 0.02s）。"""
    candidates = [HERE.parent.parent / "data/cvrplib/instances/X-n502-k39.vrp",
                  HERE.parent.parent / "cvrplib/instances/X-n502-k39.vrp"]
    vrp = next(p for p in candidates if p.exists())
    inst = CVRPInstance.parse_vrp(vrp.read_text(encoding="utf-8"))
    w = nn_witness(inst)
    ok0, d0, _, _ = evaluate_cvrp(inst, w, rounded=True)
    assert ok0
    t0 = time.perf_counter()
    out = alns_search(inst, seconds=30, seed=68001, objective="cvrp")
    elapsed = time.perf_counter() - t0
    ok, d, _, k = evaluate_cvrp(inst, out["routes"], rounded=True)
    assert ok and d <= d0 + 1e-9, (ok, d, d0)
    assert elapsed <= 30.3, "ALNS budget overshoot: %.2fs" % elapsed
    print("X-n502-k39 ALNS 30s: witness %.2f -> %.2f (K=%d, iters %d, %.2fs <= 30.3)"
          % (d0, d, k, out["iterations"], elapsed))


def main():
    test_vns_budget()
    test_alns_budget()
    candidates = [HERE.parent.parent / "data/cvrplib/instances/X-n101-k25.vrp",
                  HERE.parent.parent / "cvrplib/instances/X-n101-k25.vrp"]
    vrp = next(p for p in candidates if p.exists())
    inst = CVRPInstance.parse_vrp(vrp.read_text(encoding="utf-8"))
    w = nn_witness(inst)
    ok0, d0, _, _ = evaluate_cvrp(inst, w, rounded=True)
    assert ok0
    t0 = time.perf_counter()
    out = standard_search(inst, seconds=60, seed=68001, objective="cvrp",
                          heatmap_fn=None, split_enabled=False, slot=True)
    elapsed = time.perf_counter() - t0
    ok, d, _, k = evaluate_cvrp(inst, out["routes"], rounded=True)
    assert ok and d < d0 - 1e-9, (ok, d, d0)
    assert elapsed <= 60.3, "budget overshoot: %.2fs" % elapsed
    print("X-n101-k25 DIST_SLOT 60s: witness %.2f -> %.2f (K=%d, epochs %d, %.2fs <= 60.3)"
          % (d0, d, k, out["epochs"], elapsed))
    print("BUDGET DEADLINE REGRESSION PASSED")


if __name__ == "__main__":
    main()
