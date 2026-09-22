"""B2 步骤 2-6：单事件静态映射——单车辆 pickup+TW+返仓、多车辆映射、
车辆已有 cargo 的剩余容量、非 depot anchor 起点、空路线。

所有断言都在「构建 Model → 求解 → Solution 检查」层验证建模正确性。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view, solve_view
import helpers


def route_visits(problem, route):
    """route → (vid, [customer_id...])（vt 索引身份映射 + Activity 迭代）。"""
    vid = problem.vid_by_vt_idx[route.vehicle_type()]
    visits = [problem.customer_id_by_client_idx[a.idx]
              for a in route if a.is_client()]
    return vid, visits


def test_single_vehicle_pickup_tw_return():
    """步骤 2：单车辆，pickup 需求 + TW + 返仓。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=1)
    view = make_view(ds, capacity=100, num_vehicles=1, replan_ids={0})
    assert sum(view.pool_mask) == 3
    problem, result = solve_view(view)
    assert result.is_feasible(), '单车辆宽松实例应可行'
    sol = result.best
    assert sol.is_complete() and sol.num_missing_clients() == 0
    routes = list(sol.routes())
    assert len(routes) == 1
    vid, visits = route_visits(problem, routes[0])
    assert vid == 0
    assert sorted(visits) == [1, 2, 3], f'客户覆盖不全: {visits}'
    print(f'  单车辆：route 客户 {visits}，cost={sol.distance_cost()}')


def test_multi_vehicle_exact_mapping():
    """步骤 3：多车辆 + 精确 vehicle-ID 映射（按 vehicle type 索引，非 route 顺序）。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=2)
    view = make_view(ds, capacity=50, num_vehicles=4, replan_ids={0, 1, 2, 3})
    problem, result = solve_view(view)
    assert result.is_feasible() and result.best.is_complete()
    seen = {}
    for route in result.best.routes():
        vid, visits = route_visits(problem, route)
        assert vid is not None and vid not in seen, \
            f'vehicle type 索引身份映射错误或重复: vid={vid}'
        seen[vid] = visits
    all_customers = sorted(c for r in seen.values() for c in r)
    assert all_customers == sorted(problem.pool_customers), \
        f'客户覆盖: {all_customers} vs pool {sorted(problem.pool_customers)}'
    print(f'  多车辆映射: { {k: v for k, v in sorted(seen.items())} }')


def test_cargo_effective_capacity():
    """步骤 4：车辆已有 cargo → effective_capacity = capacity − load；
    后续 pickup 不得超过剩余容量；当前货物不进入 PyVRP 建模（不释放）。"""
    ds = helpers.make_synthetic_dataset(capacity_tight=True, seed=3)
    # 手工可装箱需求：车 0（载重 30 → 剩余 20）最多拿一个 18；
    # 其余 {22,18,22,18,22} 由 3 辆空载车（各 50）覆盖
    ds['demands'][0] = [0, 18, 22, 18, 22, 18, 22]
    view = make_view(ds, capacity=50, num_vehicles=4, replan_ids={0, 1, 2, 3},
                     loads={0: 30.0})
    problem, result = solve_view(view)
    assert result.is_feasible(), '4 车应能覆盖全部客户'
    idx = {n: i for i, n in enumerate(view.node_ids)}
    for route in result.best.routes():
        vid, visits = route_visits(problem, route)
        pickup = sum(float(view.demands[idx[c]]) for c in visits)
        if vid == 0:
            assert pickup <= 20, f'车辆 0 后续 pickup {pickup} 超过剩余容量 20'
    print('  cargo 剩余容量：车 0（载重 30）后续 pickup ≤ 20，'
          '当前货物不进入模型')


def test_non_depot_anchor_start():
    """步骤 5：当前 anchor 不在 depot 的起点建模（start depot = anchor 坐标，
    tw_early = ready_time）。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=4)
    env = helpers.build_env(ds, 100, 1, None)
    from strict_online_env import VehicleState
    import numpy as np
    vehicles = [VehicleState(vehicle_id=0)]
    vehicles[0].status = 'ready'
    vehicles[0].current_node = 1
    vehicles[0].ready_time = 5.0
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = served[1] = True
    visible = [c for c in range(1, ds['coords'].shape[1])
               if ds['demands'][0, c] > 0 and not served[c]]
    from method_adapter import build_decision_view
    view = build_decision_view(env, 0, 5.0, vehicles, served, visible, {0})
    assert 1 in view.node_ids and not view.pool_mask[view.node_ids.index(1)], \
        'anchor 客户 1 应是公开节点但不在可变池'
    v = view.vehicles[0]
    assert v.anchor_node_id == 1 and abs(v.ready_time - 5.0) < 1e-9
    problem, result = solve_view(view)
    assert result.is_feasible()
    route = next(r for r in result.best.routes()
                 if problem.vid_by_vt_idx[r.vehicle_type()] == 0)
    assert route.start_depot() is not None
    vid, visits = route_visits(problem, route)
    print(f'  非 depot 起点：anchor=1@t=5.0，route 客户 {visits}')


def test_unused_vehicle_empty_route():
    """步骤 6：一辆车空路线、其他车有路线（proposal 含全部 replan ids）。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=5)
    view = make_view(ds, capacity=100, num_vehicles=3, replan_ids={0, 1, 2})
    from route_mapper import map_solution
    problem, result = solve_view(view)
    proposal = map_solution(problem, view, result.best, runtime_s=0.0)
    assert set(proposal.suffixes.keys()) == {0, 1, 2}, \
        f'proposal 键 {set(proposal.suffixes)} != replan_ids'
    used = {vid: s for vid, s in proposal.suffixes.items() if s != () and s != (0,)}
    assert used, '至少一辆车应有路线'
    for vid, s in proposal.suffixes.items():
        assert s == () or (len(s) >= 1 and s[-1] == 0), f'车辆 {vid} suffix 非法: {s}'
    print(f'  空路线: suffixes={proposal.suffixes}')


def test_vt_mapping_order_independent():
    """vt 索引映射加固：replan_ids 乱序 + route 顺序 ≠ 车辆编号顺序 +
    一辆车未使用 → 仍精确映射到真实 vid。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=6)
    view = make_view(ds, capacity=100, num_vehicles=20, replan_ids={7, 19})
    problem, result = solve_view(view, max_iterations=200, seed=0)
    assert result.is_feasible()
    # sorted(replan_ids) = [7, 19] → vt idx 0 → vid 7，idx 1 → vid 19
    assert problem.vid_by_vt_idx == {0: 7, 1: 19}, \
        f'vt 索引映射错误: {problem.vid_by_vt_idx}'
    from route_mapper import map_solution
    proposal = map_solution(problem, view, result.best, runtime_s=0.0)
    assert set(proposal.suffixes.keys()) == {7, 19}
    # 恰好一辆车（或两辆）覆盖 2 个客户；未使用车 = 空 suffix
    covered = [vid for vid, s in proposal.suffixes.items()
               if any(c != 0 for c in s)]
    assert covered, '至少一辆 replan 车应覆盖客户'
    print(f'  乱序 replan_ids={sorted(view.replan_ids)}：vt 映射 '
          f'{problem.vid_by_vt_idx}，suffixes={proposal.suffixes}')


def main():
    test_single_vehicle_pickup_tw_return()
    test_multi_vehicle_exact_mapping()
    test_cargo_effective_capacity()
    test_non_depot_anchor_start()
    test_unused_vehicle_empty_route()
    test_vt_mapping_order_independent()
    print('PASS test_problem_builder')


if __name__ == '__main__':
    main()
