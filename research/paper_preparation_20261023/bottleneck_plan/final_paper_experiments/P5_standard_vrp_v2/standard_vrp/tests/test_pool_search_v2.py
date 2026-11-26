# -*- coding: utf-8 -*-
"""V2 池搜索冒烟测试：机制计数（epoch/前向/跨车/试选）+ 目标改善 + 可行性。"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance, evaluate_cvrp, evaluate_vrptw
from standard_vrp.adapters.objective import CvrpAdapter, VrptwAdapter
from standard_vrp.solver.pool_search import standard_search, nn_witness, bundle, propose_orders, conflict_pool

DATA = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp\data")


def cvrp(name="X-n101-k25"):
    return CVRPInstance.parse_vrp((DATA / "cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))


def vrptw(name="C101"):
    return VRPTWInstance.parse_solomon(DATA / "solomon/instances" / (name + ".vrp"))


def fake_heat(inst):
    """确定性伪热图（-距离/尺度），用于测试搜索机制；真实实验用冻结网络。"""
    n = inst.dimension
    scale = max(1e-9, float(np.median([inst.dist[i][j] for i in range(1, n + 1) for j in range(1, n + 1)])))
    h = np.full((n + 1, n + 1), -np.inf, np.float64)
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            if i != j:
                h[i][j] = -inst.dist[i][j] / scale
    return h


def test_cvrp_search():
    inst = cvrp()
    ad = CvrpAdapter(inst)
    w = nn_witness(inst)
    ok0, d0, k0 = ad.evaluate([tuple(r) for r in w])
    t0 = time.perf_counter()
    out = standard_search(ad, seconds=10, seed=68001, heatmap_fn=lambda *a: fake_heat(inst),
                          split_enabled=True, mode_name="hybrid")
    el = time.perf_counter() - t0
    assert out["ok"], out
    assert out["epochs"] >= 3, out["epochs"]
    assert out["predictor"]["success"] >= 1
    assert out["split_blocks"] >= 1, "split_enabled 时应执行跨车"
    assert sum(out["ranking_counts"]) >= 1, out["ranking_counts"]
    assert out["distance"] <= d0 + 1e-9, (out["distance"], d0)
    assert el <= 10.5, el
    print("CVRP 池搜索 PASS：epochs=%d 前向=%d 跨车块=%d 单=%d 试选占比=%s D %.1f→%.1f (%.1fs)"
          % (out["epochs"], out["predictor"]["success"], out["split_blocks"], out["split_orders"],
             out["ranking_counts"], d0, out["distance"], el))


def test_vrptw_search():
    inst = vrptw()
    ad = VrptwAdapter(inst)
    w = nn_witness(inst)
    ok0, d0, k0 = ad.evaluate([tuple(r) for r in w])
    t0 = time.perf_counter()
    out = standard_search(ad, seconds=10, seed=69001, heatmap_fn=lambda *a: fake_heat(inst),
                          split_enabled=True, mode_name="hybrid")
    el = time.perf_counter() - t0
    assert out["ok"], out
    assert out["epochs"] >= 3
    assert el <= 10.5, el
    (k1, d1) = (out["vehicles"], out["distance"])
    assert (k1, d1) <= (k0, d0), ((k1, d1), (k0, d0))
    print("VRPTW 池搜索 PASS：epochs=%d 前向=%d 跨车块=%d (K,D) (%d,%.1f)→(%d,%.1f) (%.1fs)"
          % (out["epochs"], out["predictor"]["success"], out["split_blocks"],
             k0, d0, k1, d1, el))


def test_distance_only_mode():
    """mode='pool'（无神经、无校准）仍应使用相同机制（距离排序）。"""
    inst = cvrp()
    ad = CvrpAdapter(inst)
    out = standard_search(ad, seconds=6, seed=68002, heatmap_fn=None, split_enabled=True, mode_name="pool")
    assert out["ok"]
    assert out["predictor"]["attempts"] == 0, "无 heatmap 时不应有前向尝试"
    print("距离模式 PASS：epochs=%d D=%.1f" % (out["epochs"], out["distance"]))


def test_bundle_and_orders():
    inst = cvrp()
    w = [list(r) for r in nn_witness(inst)]
    sel = bundle(w, inst.dist, 0)
    assert len(sel) >= 2 and sum(len(w[k]) for k in sel) <= 24, sel
    parts = [tuple(w[k]) for k in sel]
    orders = list(propose_orders(parts, fake_heat(inst), 1, 5))
    assert len(orders) == 5
    assert all(sorted(o) == sorted(i for r in parts for i in r) for o in orders)
    print("bundle/proposals PASS：%s，%d 条提议，客户覆盖正确" % (str(sel), len(orders)))


def test_conflict_pool():
    inst = cvrp()
    w = [tuple(r) for r in nn_witness(inst)]
    removed = {i for r in w[:2] for i in r}
    cp = conflict_pool(w, removed, fake_heat(inst))
    assert cp and set(cp) <= removed
    print("冲突池 PASS：%d 个客户" % len(cp))


def main():
    test_bundle_and_orders()
    test_conflict_pool()
    test_distance_only_mode()
    test_cvrp_search()
    test_vrptw_search()
    print("ALL P5-V2 POOL SEARCH SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
