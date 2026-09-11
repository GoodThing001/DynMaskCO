"""OR4.1 加固测试：dummy 终端 / 结构化失败 / 亚秒毫秒 / forced-drop /
无 SKIP 交叉负例。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view, solve_view, status_ok
import helpers
import numpy as np
from problem_builder import build_model, BuildError
from route_mapper import map_solution, MappingError
from ortools_adapter import ORToolsRHDAdapter


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


def test_dummy_terminal_and_objective_consistency():
    """dummy 只能出现在车辆 0 的终端；原始整数目标 == 映射路线重算目标。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=1)
    view = make_view(ds, capacity=50, num_vehicles=2)
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    # dummy 终端约束由模型强制 + mapper 审计（不抛即通过）
    dropped = _dropped(problem, solution)
    proposal = _map(problem, solution, view, dropped=dropped)
    # 目标一致性已由 mapper 内置断言（OBJECTIVE_MISMATCH 会抛）
    assert proposal.solve_meta['n_dummy'] == len(view.replan_ids)
    print('  dummy 终端约束 + 目标一致性通过（mapper 内置审计不抛）')


def test_all_drop_still_solvable():
    """全部客户不可服务（剩余容量 < 每个需求）时 dummy 兜底仍可解。"""
    ds = helpers.make_synthetic_dataset(n_customers=4, seed=2)
    ds['demands'][0] = [0, 25, 25, 25, 25]
    view = make_view(ds, capacity=50, num_vehicles=1, loads={0: 40.0})
    problem, solution, status = solve_view(view)
    assert status_ok(status), f'全 drop 场景应可解（dummy 兜底），status={status}'
    dropped = _dropped(problem, solution)
    assert sorted(dropped) == sorted(view.pool_customer_ids)
    proposal = _map(problem, solution, view, dropped=dropped)
    assert proposal.solve_meta['n_dropped'] == 4
    print('  全 drop 场景：dummy 兜底可解，4/4 dropped')


def test_intersection_rejected_no_skip():
    """确定性交叉负例（无 SKIP）：必然 1 planned + 1 dropped 的 fixture，
    人为把 planned 客户放进 dropped → mapper 必须抛。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=3)
    ds['demands'][0] = [0, 30, 30]        # cap 50：恰好服务一个
    view = make_view(ds, capacity=50, num_vehicles=1)
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    assert len(dropped) == 1, f'fixture 应恰好 drop 1 个，实际 {dropped}'
    planned = [c for s in _map(problem, solution, view, dropped=dropped)
               .suffixes.values() for c in s if c != 0]
    assert len(planned) == 1
    try:
        _map(problem, solution, view, dropped=[planned[0]])
        raise AssertionError('planned/dropped 交叉未被拒绝')
    except MappingError as e:
        assert '交叉' in str(e) or '同时' in str(e)
    print('  交叉负例（无 SKIP）：planned/dropped 交叉被拒绝')


def test_integerized_tw_empty_forced_drop():
    """ceil(tw_start) > floor(tw_end) 的客户强制 drop（INTEGERIZED_TW_EMPTY），
    不建节点、不让求解器抛模糊错误。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=4)
    ds['tw_start'][0, 1] = 0.5006
    ds['tw_end'][0, 1] = 0.5004       # 整数化后 TW 空
    view = make_view(ds, capacity=50, num_vehicles=1)
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    assert problem.forced_dropped == [1]
    assert problem.scale_meta['n_model_pool'] == 1
    proposal = _map(problem, solution, view,
                    dropped=_dropped(problem, solution) + problem.forced_dropped)
    assert proposal.solve_meta['n_dropped'] == 1
    print('  INTEGERIZED_TW_EMPTY：客户 1 强制 drop，模型只含 1 个客户')


def test_input_boundary_validation():
    """replan_ids 为空 → BuildError；solution_limit/time_limit 非法 → ValueError。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=5)
    env = helpers.build_env(ds, 50, 1, None)
    from strict_online_env import VehicleState
    from method_adapter import build_decision_view
    vehicles = [VehicleState(vehicle_id=0)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = True
    view = build_decision_view(env, 0, 0.0, vehicles, served, [1, 2], set())
    try:
        build_model(view)
        raise AssertionError('空 replan_ids 未被拒绝')
    except BuildError:
        pass
    try:
        ORToolsRHDAdapter(solution_limit=0, check_env=False)
        raise AssertionError('solution_limit=0 未被拒绝')
    except ValueError:
        pass
    try:
        ORToolsRHDAdapter(time_limit_s=0, check_env=False)
        raise AssertionError('time_limit_s=0 未被拒绝')
    except ValueError:
        pass
    print('  输入边界：空 replan_ids / 非法预算参数全部拒绝')


def test_subsecond_time_limit_ms():
    """亚秒 time limit：0.0001s → 1ms（FromMilliseconds），不再退化为 0 秒。"""
    adapter = ORToolsRHDAdapter(solution_limit=10, time_limit_s=0.0001,
                                check_env=False)
    sp = adapter._search_params()
    assert sp.time_limit.ToMilliseconds() == 1, \
        f'0.0001s 应转 1ms，实际 {sp.time_limit.ToMilliseconds()}ms'
    adapter2 = ORToolsRHDAdapter(solution_limit=10, time_limit_s=1.5,
                                 check_env=False)
    sp2 = adapter2._search_params()
    assert sp2.time_limit.ToMilliseconds() == 1500
    print('  亚秒 time limit：0.0001s→1ms、1.5s→1500ms（不再退化为 0）')


def test_structured_failure_records():
    """求解器异常/映射异常 → 结构化失败记录（阶段/状态码/原因），不中断。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=6)
    view = make_view(ds, capacity=50, num_vehicles=1)

    # (a) SOLVER_EXCEPTION：monkeypatch SolveWithParameters 抛异常
    adapter = ORToolsRHDAdapter(solution_limit=10, check_env=False)
    import ortools_adapter as oa
    orig_build = oa.build_model

    class Boom:
        routing = None

        def __getattr__(self, name):
            if name == 'routing':
                raise RuntimeError('solver exploded')
            raise AttributeError(name)

    oa.build_model = lambda v, **kw: _boom_problem()
    try:
        proposal = adapter.propose(view)
    finally:
        oa.build_model = orig_build
    assert proposal.fallback_triggered is True
    assert proposal.solve_meta['failure_stage'] == 'SOLVE'
    assert proposal.solve_meta['solver_status'] == 'SOLVER_EXCEPTION'
    assert set(proposal.suffixes.keys()) == set(view.replan_ids)

    # (b) MAPPING_ERROR：monkeypatch map_solution 抛 FLOAT_CERTIFICATE_FAIL
    adapter2 = ORToolsRHDAdapter(solution_limit=10, check_env=False)
    orig_map = oa.map_solution
    def bad_map(*a, **kw):
        raise MappingError('伪造证书失败', code='FLOAT_CERTIFICATE_FAIL')
    oa.map_solution = bad_map
    try:
        proposal2 = adapter2.propose(view)
    finally:
        oa.map_solution = orig_map
    assert proposal2.fallback_triggered is True
    assert proposal2.solve_meta['failure_stage'] == 'MAP'
    assert proposal2.solve_meta['solver_status'] == 'FLOAT_CERTIFICATE_FAIL'
    print('  结构化失败：SOLVER_EXCEPTION / FLOAT_CERTIFICATE_FAIL 全部落盘'
          '（阶段/状态码/原因 + 安全 suffix），不中断 runner')


class _boom_problem:
    class _Routing:
        def SolveWithParameters(self, *a, **k):
            raise RuntimeError('solver exploded')
    routing = _Routing()
    scale_meta = {}
    manager = None


def test_wait_empty_route_zero_objective():
    """OR4.2 P0-1：有 future + 空路线 → 写回 ()，目标贡献 0
    （求解目标与写回 WAIT 动作一致，一致性断言不抛即证明）。"""
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=7,
                                        reveal_spec={2: 5.0})
    ds['demands'][0] = [0, 25, 25]
    view = make_view(ds, capacity=50, num_vehicles=1, loads={0: 40.0})
    assert view.has_future_reveal
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    assert sorted(dropped) == sorted(view.pool_customer_ids)  # 全 drop → 空路线
    proposal = _map(problem, solution, view, dropped=dropped)
    assert proposal.suffixes[0] == (), '有 future 应 WAIT'
    # WAIT：模型空弧已置 0；mapper 目标重算 () 计 0——一致性断言（不抛即证明）
    print('  WAIT 空路线：suffix=()，目标贡献 0（模型空弧置 0 + 一致性断言通过）')


def test_close_empty_route_real_cost():
    """OR4.2 P0-1：无 future → 未分配车辆写回 (0,)，目标必须含真实返仓距离。"""
    ds = helpers.make_synthetic_dataset(n_customers=1, seed=8)
    ds['demands'][0] = [0, 25]
    view = make_view(ds, capacity=50, num_vehicles=2, loads={0: 40.0})
    assert not view.has_future_reveal
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    dropped = _dropped(problem, solution)
    proposal = _map(problem, solution, view, dropped=dropped)
    # 车 0（载重 40）装不下 25 → 空路线 CLOSE；车 1 服务客户 1
    assert proposal.suffixes[0] == (0,)
    # 目标一致性由 mapper 断言（CLOSE 计真实返仓弧，不抛即证明）
    print('  CLOSE 空路线：suffix=(0,)，目标含真实返仓距离（一致性断言通过）')


def test_wrapper_native_hash_distinct():
    """OR4.2 P0-2：wrapper 与 native 文件路径/hash 明确不同；wheel 磁盘实算。"""
    import identity as ortools_identity
    ident = ortools_identity.check_environment()
    assert ident['wrapper_file'].endswith('pywrapcp.py')
    assert not ident['native_file'].endswith('.py'),         f'native 应是二进制扩展: {ident["native_file"]}'
    assert ident['wrapper_sha256'] != ident['native_extension_sha256']
    assert ident['wheel_sha256'] == ortools_identity.WHEEL_SHA256
    print(f"  wrapper={ident['wrapper_sha256'][:16]} "
          f"native={ident['native_extension_sha256'][:16]} 明确不同；"
          f'wheel 磁盘实算比对通过')


def test_adapter_real_environment_initialization():
    """OR4.2 收口：check_env=True 真实环境初始化（旧字段名 bug 的回归测试）。"""
    adapter = ORToolsRHDAdapter(solution_limit=30, time_limit_s=30.0,
                                check_env=True)
    ident = adapter.env_identity
    for k in ('wrapper_sha256', 'native_extension_sha256', 'wheel_sha256'):
        assert len(ident[k]) == 64, f'{k} 应为完整 64 位 hash'
    assert ident['wrapper_sha256'] != ident['native_extension_sha256']
    assert ident['native_extension_sha256'][:16] in adapter.checkpoint_hash
    # 完整 hash 通过真实运行落盘（identity 记录，非仅截断 checkpoint）
    ds = helpers.make_synthetic_dataset(n_customers=2, seed=10)
    from strict_online_runner import run_instance
    from coldchain_contract import default_pilot_profile
    rec = run_instance(ds, 50, 2,
                       adapter_factory=lambda: ORToolsRHDAdapter(
                           solution_limit=30, check_env=True),
                       inst_idx=0, objective='coldchain',
                       profile=default_pilot_profile(), seed=0,
                       data_sha256='t', instance_seed=0)
    assert rec['checkpoint_hash'].startswith('ortools=9.11.4210_native=')
    print('  真实环境初始化：check_env=True 成功；wrapper/native/wheel 全 64 位；'
          'native 前 16 位进 checkpoint；完整身份经真实运行落盘')


def test_unexpected_exceptions_recorded():
    """OR4.2 P1：BUILD_EXCEPTION / MAPPING_EXCEPTION 意外异常结构化落盘。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=9)
    view = make_view(ds, capacity=50, num_vehicles=1)
    from ortools_adapter import ORToolsRHDAdapter
    import ortools_adapter as oa

    # (a) BUILD_EXCEPTION：build_model 抛 ValueError
    adapter = ORToolsRHDAdapter(solution_limit=10, check_env=False)
    orig_build = oa.build_model
    def bad_build(v, **kw):
        raise ValueError('建模参数损坏')
    oa.build_model = bad_build
    try:
        p1 = adapter.propose(view)
    finally:
        oa.build_model = orig_build
    assert p1.fallback_triggered is True
    assert p1.solve_meta['failure_stage'] == 'BUILD'
    assert p1.solve_meta['solver_status'] == 'BUILD_EXCEPTION'
    assert 'ValueError' in p1.solve_meta['failure_reason']

    # (b) MAPPING_EXCEPTION：map_solution 抛 RuntimeError
    adapter2 = ORToolsRHDAdapter(solution_limit=10, check_env=False)
    orig_map = oa.map_solution
    def bad_map(*a, **kw):
        raise RuntimeError('记录字段损坏')
    oa.map_solution = bad_map
    try:
        p2 = adapter2.propose(view)
    finally:
        oa.map_solution = orig_map
    assert p2.fallback_triggered is True
    assert p2.solve_meta['failure_stage'] == 'MAP'
    assert p2.solve_meta['solver_status'] == 'MAPPING_EXCEPTION'
    assert 'RuntimeError' in p2.solve_meta['failure_reason']
    print('  意外异常：BUILD_EXCEPTION / MAPPING_EXCEPTION 结构化落盘'
          '（异常类 + repr + 阶段），不中断 runner')


def main():
    test_dummy_terminal_and_objective_consistency()
    test_all_drop_still_solvable()
    test_intersection_rejected_no_skip()
    test_integerized_tw_empty_forced_drop()
    test_input_boundary_validation()
    test_subsecond_time_limit_ms()
    test_structured_failure_records()
    test_wait_empty_route_zero_objective()
    test_close_empty_route_real_cost()
    test_wrapper_native_hash_distinct()
    test_adapter_real_environment_initialization()
    test_unexpected_exceptions_recorded()
    print('PASS test_ortools_hardening')


if __name__ == '__main__':
    main()
