"""OR7-2/3/4/5/6：mapper 精确分区 / 整数边界 / 可选客户 / 空 pool / 失败记录。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view, solve_view, status_ok
import helpers
import numpy as np
from problem_builder import build_model
from route_mapper import map_solution, MappingError, wait_or_close, certify_suffix_float


def _map(problem, solution, view, dropped=()):
    return map_solution(problem, view, solution, runtime_s=0.0,
                        solver_status='SOLVED', dropped_customers=dropped)


def _dropped(problem, solution):
    out = []
    for node_number in problem.customer_id_by_node_index:
        idx = problem.manager.NodeToIndex(node_number)
        if solution.Value(problem.routing.NextVar(idx)) == idx:
            out.append(problem.customer_id_by_node_index[node_number])
    return out


def test_mapper_exact_partition():
    """planned ∪ dropped == pool，无交叉，proposal 键完整。"""
    ds = helpers.make_synthetic_dataset(n_customers=4, seed=1)
    view = make_view(ds, capacity=100, num_vehicles=2)
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    proposal = _map(problem, solution, view, dropped=dropped)
    assert set(proposal.suffixes.keys()) == set(view.replan_ids)
    planned = [c for s in proposal.suffixes.values() for c in s if c != 0]
    assert sorted(set(planned) | set(dropped)) == sorted(view.pool_customer_ids)
    assert set(planned) & set(dropped) == set()
    assert not any(c not in view.pool_customer_ids for c in planned)
    print(f'  mapper 精确分区：planned={len(planned)} dropped={len(dropped)}')




def test_tw_exact_boundary():
    """arrival == tw_end 恰好边界：整数保守转换后仍可行。"""
    ds = helpers.make_synthetic_dataset(n_customers=1, seed=2)
    ds['coords'][0, 0] = [0.5, 0.5]
    ds['coords'][0, 1] = [1.0, 0.5]        # dist 0.5
    ds['tw_start'][0, 1] = 0.0
    ds['tw_end'][0, 1] = 0.5               # arrival == tw_end
    ds['demands'][0, 1] = 10
    view = make_view(ds, capacity=50, num_vehicles=1)
    problem, solution, status = solve_view(view)
    assert status_ok(status), f'status={status}'
    proposal = _map(problem, solution, view, dropped=_dropped(problem, solution))
    planned = [c for s in proposal.suffixes.values() for c in s if c != 0]
    assert planned == [1], f'边界可行实例应覆盖客户，planned={planned}'
    print('  TW 恰好边界可行')


def test_capacity_exact_limit():
    """容量恰好 50/50 可行；51 必须 drop。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=3)
    ds['demands'][0] = [0, 25, 25]
    view = make_view(ds, capacity=50, num_vehicles=1)
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    assert dropped == [], f'25+25=50 应全覆盖，dropped={dropped}'

    ds2 = helpers.make_synthetic_dataset(n_customers=2, seed=3)
    ds2['demands'][0] = [0, 26, 25]
    view2 = make_view(ds2, capacity=50, num_vehicles=1)
    problem2, solution2, status2 = solve_view(view2)
    assert status_ok(status2)
    dropped2 = _dropped(problem2, solution2)
    assert len(dropped2) == 1, f'26+25=51 必须 drop 1 个，dropped={dropped2}'
    print('  容量恰好 50 可行；51 必须 drop 1 个（精确分区由 mapper 复核）')


def test_capacity_shortage_feasible_subset():
    """容量不足：feasible 子集 + 剩余 DEFER（drop 只延后不是服务成功）。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=4)
    ds['demands'][0] = [0, 25, 24, 23, 22, 21, 20]
    view = make_view(ds, capacity=50, num_vehicles=1, loads={0: 40.0})
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    assert dropped, '剩余容量 10 < 每个客户需求，必须 drop'
    proposal = _map(problem, solution, view, dropped=dropped)
    planned = [c for s in proposal.suffixes.values() for c in s if c != 0]
    assert sorted(set(planned) | set(dropped)) == sorted(view.pool_customer_ids)
    v = view.vehicles[0]
    ok, reason = certify_suffix_float(view, v, planned)
    assert ok, f'浮点 certificate 失败: {reason}'
    print(f'  容量不足：drop {len(dropped)} 个（DEFER），浮点 certificate 通过')


def test_empty_pool_fast_path():
    """空 pool：不调用 OR-Tools（build_model 不被调用）+ WAIT/CLOSE 规则。"""
    ds = helpers.make_synthetic_dataset(reveal_spec={5: 5.0, 6: 5.0}, seed=5)
    env = helpers.build_env(ds, 50, 2, None)
    from strict_online_env import VehicleState
    from method_adapter import build_decision_view
    vehicles = [VehicleState(vehicle_id=i) for i in range(2)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    for c in range(1, 5):
        served[c] = True          # 已 reveal 全服务 → pool 空
    view = build_decision_view(env, 0, 2.0, vehicles, served, [], {0, 1})

    from ortools_adapter import ORToolsRHDAdapter
    adapter = ORToolsRHDAdapter(check_env=False)
    import ortools_adapter as oa
    calls = []
    orig = oa.build_model

    def counting(v, **kw):
        calls.append(1)
        return orig(v, **kw)

    oa.build_model = counting
    try:
        proposal = adapter.propose(view)
    finally:
        oa.build_model = orig
    assert not calls, '空 pool 仍调用了 OR-Tools'
    assert set(proposal.suffixes.keys()) == {0, 1}
    assert proposal.solve_meta['solver_status'] == 'EMPTY_POOL_FAST_PATH'
    # 有 future + depot 等待 → WAIT
    assert proposal.suffixes[0] == () and proposal.suffixes[1] == ()
    print('  空 pool 快速路径：不调用 OR-Tools，depot+future → WAIT')


class _TimeoutProblem:
    """模拟 status=4（FAIL_TIMEOUT）+ 无解的 stub（确定性，不依赖真实计时）。"""
    class _Routing:
        def SolveWithParameters(self, *a, **k):
            return None
        def status(self):
            return 4
    routing = _Routing()
    scale_meta = {'n_pool': 2, 'n_vehicles': 2}
    forced_dropped = []


def test_solver_failure_recorded():
    """TIME_LIMIT（确定性 stub）：fallback_triggered=True + solver_status
    记录 + failure_stage=SOLVE + 安全 suffix（不伪装空解）。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=6)
    view = make_view(ds, capacity=50, num_vehicles=2)
    from ortools_adapter import ORToolsRHDAdapter
    adapter = ORToolsRHDAdapter(solution_limit=10, check_env=False)
    import ortools_adapter as oa
    orig_build = oa.build_model
    oa.build_model = lambda v, **kw: _TimeoutProblem()
    try:
        proposal = adapter.propose(view)
    finally:
        oa.build_model = orig_build
    assert proposal.fallback_triggered is True, '超时必须标记 fallback'
    assert proposal.solve_meta['solver_status'] == 'TIME_LIMIT'
    assert proposal.solve_meta['failure_stage'] == 'SOLVE'
    assert proposal.solve_meta['solver_status_code'] == 4
    assert set(proposal.suffixes.keys()) == set(view.replan_ids)
    for vid, s in proposal.suffixes.items():
        assert s == () or s[-1] == 0, f'安全 suffix 非法: {s}'
    print('  失败记录（确定性 stub）：TIME_LIMIT + stage=SOLVE + code=4 '
          '+ fallback=True + 安全 suffix')


def main():
    test_mapper_exact_partition()
    test_tw_exact_boundary()
    test_capacity_exact_limit()
    test_capacity_shortage_feasible_subset()
    test_empty_pool_fast_path()
    test_solver_failure_recorded()
    print('PASS test_ortools_unit')


if __name__ == '__main__':
    main()
