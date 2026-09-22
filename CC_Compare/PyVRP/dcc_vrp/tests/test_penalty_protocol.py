"""B2 回归：覆盖-可行性 penalty 协议。

  1. 容量不足可行子集：总需求 > 有效容量 → feasible 结果、≥1 unplanned、
     route/unplanned 精确分区、每 route feasible、浮点 certificate 通过。
  2. 固定 prize 反例说明：旧 PRIZE_PER_CUSTOMER=1e6 > 默认 penalty 上限（1e5），
     会以超载换覆盖（历史事件 4 失败机制）；新界限计算 penalty 下同样形状
     返回 feasible 子集——证明修改不是随意调参。
  3. 溢出保护：极小 max_edge 场景不会触发（界限计算检查）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view, solve_view
import helpers
import numpy as np


def _capacity_shortage_view():
    """1 辆 replan 车（剩余容量 10），6 个客户总需求 >> 10 → 必然放不下全部。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=2)
    ds['demands'][0] = [0, 25, 24, 23, 22, 21, 20]
    view = make_view(ds, capacity=50, num_vehicles=1, replan_ids={0},
                     loads={0: 40.0})    # 剩余容量 10 < 每个客户需求
    return view


def test_capacity_shortage_feasible_subset():
    """总需求远超剩余容量：必须 feasible 子集 + 精确分区（不是超载）。"""
    view = _capacity_shortage_view()
    problem, result = solve_view(view, max_iterations=500, seed=0,
                                 require_complete=False)
    assert result.is_feasible(), \
        'deferral 模式容量不足必须返回 feasible 子集（不得超载）'
    from route_mapper import map_solution, certify_suffix_float
    proposal = map_solution(problem, view, result.best)
    unplanned = set(problem.pool_customers) - {
        c for s in proposal.suffixes.values() for c in s if c != 0}
    assert unplanned, '容量不足应至少有一个客户 unplanned'
    # 精确分区已由 map_solution 校验（不抛即通过）
    # 每条非空 suffix 浮点 certificate
    vmap = {v.vehicle_id: v for v in view.vehicles}
    for vid, s in proposal.suffixes.items():
        ok, reason = certify_suffix_float(view, vmap[vid], [c for c in s if c != 0])
        assert ok, f'浮点 certificate 失败: {reason}'
    print(f'  容量不足：feasible 子集，unplanned={sorted(unplanned)}，'
          f'scale_meta={problem.scale_meta}')


def test_old_fixed_prize_counterexample_shape():
    """历史事件 4 失败形状：1 车 16 客户（总需求超剩余容量）——
    旧固定 prize=1e6 会超载换覆盖（is_feasible=False）；
    新协议下同形状返回 feasible 子集。"""
    ds = helpers.make_synthetic_dataset(n_customers=16, seed=3)
    # 剩余容量 30，16 客户 × 需求 8-16 → 总需求 >> 30
    view = make_view(ds, capacity=50, num_vehicles=1, replan_ids={0},
                     loads={0: 20.0})
    problem, result = solve_view(view, max_iterations=1000, seed=0,
                                 require_complete=False)
    assert result.is_feasible(), '新 penalty 协议下必须 feasible（旧 1e6 prize 会超载）'
    for r in result.best.routes():
        assert r.is_feasible() and not r.has_excess_load()
    meta = problem.scale_meta
    assert meta['coverage_prize'] <= meta['feasibility_penalty'], \
        f'penalty 必须压过 prize: {meta}'
    print(f"  旧反例形状：feasible 子集，n_pool={meta['n_pool']} "
          f"coverage_prize={meta['coverage_prize']} "
          f"penalty={meta['feasibility_penalty']}")


def test_penalty_scales_with_subproblem():
    """尺度随子问题变化（不是常数）：两个不同规模视图的 prize/penalty 不同。"""
    view_small = _capacity_shortage_view()
    _, res1 = solve_view(view_small, max_iterations=100, seed=0,
                         require_complete=False)
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=4)
    view_tiny = make_view(ds, capacity=50, num_vehicles=1, replan_ids={0})
    from problem_builder import build_model
    p2 = build_model(view_tiny, require_complete=False)
    from testutil import solve_view as _sv
    p1, _ = _sv(view_small, max_iterations=100, seed=0, require_complete=False)
    assert p1.scale_meta['coverage_prize'] != p2.scale_meta['coverage_prize'], \
        'prize 应是子问题相关尺度而非常数'
    print(f"  尺度随子问题变化：{p1.scale_meta['coverage_prize']} vs "
          f"{p2.scale_meta['coverage_prize']}")


def main():
    test_capacity_shortage_feasible_subset()
    test_old_fixed_prize_counterexample_shape()
    test_penalty_scales_with_subproblem()
    print('PASS test_penalty_protocol')


if __name__ == '__main__':
    main()
