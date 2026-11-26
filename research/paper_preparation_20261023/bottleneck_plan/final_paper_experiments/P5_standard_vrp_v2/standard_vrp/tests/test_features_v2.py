# -*- coding: utf-8 -*-
"""V2 特征构造单元测试：冻结通道语义（ch10 仓库/可见、ch11-15 全零、timestep 公式）、
窗口与完整一致、遮蔽邻接无泄漏。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance, route_schedule
from standard_vrp.core.standard_features import (StandardFeatureBuilder, WindowedFeatureBuilder,
                                                 window_indices, FROZEN, NORM_DEMAND)

DATA = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp\data")


def cvrp(name="X-n101-k25"):
    return CVRPInstance.parse_vrp((DATA / "cvrplib/instances" / (name + ".vrp")).read_text(encoding="utf-8"))


def vrptw(name="C101"):
    return VRPTWInstance.parse_solomon(DATA / "solomon/instances" / (name + ".vrp"))


def witness(inst):
    n = inst.dimension
    remaining = set(range(1, n + 1))
    is_tw = isinstance(inst, VRPTWInstance)
    routes = []
    while remaining:
        route, cur = [], 0
        while remaining:
            nxt = min(remaining, key=lambda i: inst.dist[cur][i])
            trial = route + [nxt]
            if is_tw:
                if route_schedule(inst, tuple(trial)) is None:
                    break
            else:
                if sum(inst.demands[i] for i in trial) > inst.capacity + 1e-9:
                    break
            route.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        routes.append(route)
    return routes


def test_ch10_visibility():
    inst = cvrp()
    b = StandardFeatureBuilder(inst, "cvrp", mode="v2_fix")
    routes = witness(inst)
    removed = {routes[0][0], routes[1][0], 5}
    batch = b.build(routes, removed)
    f = batch["features"][0]
    assert f[0, 10] == 1.0, "仓库 ch10 必须为 1（冻结语义 i==0 或未遮）"
    for i in range(1, inst.dimension + 1):
        expected = 0.0 if i in removed else 1.0
        assert f[i, 10] == expected, (i, f[i, 10], expected)
    print("ch10 PASS：仓库=1、可见=1、被遮=0")


def test_dynamic_channels_zero():
    inst = vrptw()
    b = StandardFeatureBuilder(inst, "vrptw", mode="v2_fix")
    routes = witness(inst)
    removed = {routes[0][0]}
    f = b.build(routes, removed)["features"][0]
    assert np.allclose(f[:, 11:16], 0.0), "maskco_base 变体下 ch11-15 必须全零"
    print("ch11-15 PASS：全零（冻结 base 语义）")


def test_timestep_formula():
    inst = cvrp()
    b = StandardFeatureBuilder(inst, "cvrp", mode="v2_fix")
    routes = witness(inst)
    batch = b.build(routes, set())
    adj = batch["adjacency"][0]
    expected = float(adj[1:].sum()) / (inst.dimension * 2)
    assert abs(float(batch["timestep"][0]) - expected) < 1e-6, (batch["timestep"], expected)
    print("timestep PASS：分母 = n*2 (n=%d)" % inst.dimension)


def test_adjacency_no_leak():
    inst = cvrp()
    b = StandardFeatureBuilder(inst, "cvrp", mode="v2_fix")
    routes = witness(inst)
    removed = {routes[0][0], routes[1][0]}
    adj = b.build(routes, removed)["adjacency"][0]
    idx = sorted(removed)
    assert adj[idx, :].sum() == 0 and adj[:, idx].sum() == 0, "被遮节点邻接必须清零"
    assert (adj == adj.T).all(), "undirected 邻接必须对称"
    print("邻接 PASS：无被遮节点边、对称")


def test_window_full_consistency():
    """n ≤ 100 时窗口覆盖全部节点：窗口 batch 与完整 batch 完全一致。"""
    for name, objective, inst in (("X-n101-k25", "cvrp", cvrp()), ("C101", "vrptw", vrptw())):
        b = StandardFeatureBuilder(inst, objective, mode="v2_fix")
        wb = WindowedFeatureBuilder(inst, objective, mode="v2_fix")
        routes = witness(inst)
        removed = {routes[0][0], routes[0][-1]}
        full = b.build(routes, removed)
        window, win, pos_of = wb.build_windowed(routes, removed)
        assert window == list(range(0, inst.dimension + 1)), "n<=100 时窗口应覆盖全部节点"
        assert np.allclose(full["features"], win["features"]), name + " 特征不一致"
        assert np.allclose(full["adjacency"], win["adjacency"]), name + " 邻接不一致"
        assert abs(float(full["timestep"][0]) - float(win["timestep"][0])) < 1e-6, name + " timestep 不一致"
        print("窗口一致性 PASS：%s（n=%d）特征/邻接/timestep 完全一致" % (name, inst.dimension))


def test_window_subset_consistency():
    """n > 100：窗口特征必须是完整特征的对应子阵（含 ch10、邻接子阵）。"""
    inst = cvrp("X-n204-k19")
    b = StandardFeatureBuilder(inst, "cvrp", mode="v2_fix")
    wb = WindowedFeatureBuilder(inst, "cvrp", mode="v2_fix")
    routes = witness(inst)
    removed = {routes[0][0], routes[1][0], routes[2][0]}
    full = b.build(routes, removed)
    window, win, pos_of = wb.build_windowed(routes, removed)
    assert 1 + len(removed) <= len(window) <= 101 + 1, len(window)
    for p, i in enumerate(window):
        assert np.allclose(win["features"][0][p], full["features"][0][i]), ("feature", i)
        for q, j in enumerate(window):
            assert win["adjacency"][0][p, q] == full["adjacency"][0][i, j], ("adj", i, j)
    print("窗口子阵 PASS：X-n204-k19 窗口 %d 节点，特征/邻接与完整构建一致" % len(window))


def test_v1_mode_reproduces_deviation():
    """mode='v1' 应复现 P5-V1 的偏离（ch12=1、动态通道非零），用于 STD_OLD 参照。"""
    inst = vrptw()
    b = StandardFeatureBuilder(inst, "vrptw", mode="v1")
    routes = witness(inst)
    f = b.build(routes, {routes[0][0]})["features"][0]
    assert f[:, 12].min() == 1.0, "V1 编码 ch12=1"
    assert np.abs(f[:, 11:16]).sum() > 0, "V1 编码动态通道非零"
    print("V1 复现 PASS：ch12=1、动态通道非零（与 v2_fix 全零形成对照）")


def main():
    test_ch10_visibility()
    test_dynamic_channels_zero()
    test_timestep_formula()
    test_adjacency_no_leak()
    test_window_full_consistency()
    test_window_subset_consistency()
    test_v1_mode_reproduces_deviation()
    print("ALL P5-V2 FEATURE TESTS PASSED")


if __name__ == "__main__":
    main()
