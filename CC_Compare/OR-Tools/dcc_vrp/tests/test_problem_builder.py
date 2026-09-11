"""OR7-1：problem_builder——virtual start / 容量 / 时间 / 非连续 vehicle ID。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil
from testutil import make_view, solve_view, status_ok
import helpers
import numpy as np
from problem_builder import build_model


def route_visits(problem, solution, v_idx):
    start = problem.routing.Start(v_idx)
    node = int(solution.Value(problem.routing.NextVar(start)))
    visits = []
    while not problem.routing.IsEnd(node):
        node_number = problem.manager.IndexToNode(node)
        cust = problem.customer_id_by_node_index.get(node_number)
        if cust != -1:     # dummy sentinel 过滤
            visits.append(cust)
        node = int(solution.Value(problem.routing.NextVar(node)))
    return visits


def test_virtual_start_basic():
    """单车从客户 anchor 出发：virtual start 不入 route、客户全覆盖。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=1)
    env = helpers.build_env(ds, 100, 1, None)
    from strict_online_env import VehicleState
    vehicles = [VehicleState(vehicle_id=0)]
    vehicles[0].status = 'ready'
    vehicles[0].current_node = 1
    vehicles[0].ready_time = 5.0
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = served[1] = True
    visible = [2, 3]
    from method_adapter import build_decision_view
    view = build_decision_view(env, 0, 5.0, vehicles, served, visible, {0})
    problem, solution, status = solve_view(view)
    assert status_ok(status) and solution is not None
    visits = route_visits(problem, solution, 0)
    assert sorted(visits) == [2, 3], f'客户覆盖: {visits}'
    assert 1 not in visits, 'anchor 客户（已服务）不得出现在 route'
    print('  单车 customer-anchor：virtual start 正确，客户 {visits}')


def test_multi_vehicle_distinct_anchors_and_noncontiguous_ids():
    """多车不同 anchor + 非连续 vehicle ID（replan_ids={7,19}）。"""
    ds = helpers.make_synthetic_dataset(n_customers=4, seed=2)
    env = helpers.build_env(ds, 100, 20, None)
    from strict_online_env import VehicleState
    vehicles = [VehicleState(vehicle_id=i) for i in range(20)]
    vehicles[7].status = 'ready'
    vehicles[7].current_node = 1
    vehicles[7].ready_time = 3.0
    vehicles[19].status = 'ready'
    vehicles[19].current_node = 2
    vehicles[19].ready_time = 4.0
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = served[1] = served[2] = True
    visible = [3, 4]
    from method_adapter import build_decision_view
    view = build_decision_view(env, 0, 4.0, vehicles, served, visible, {7, 19})
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    v7 = route_visits(problem, solution, 0)
    v19 = route_visits(problem, solution, 1)
    assert sorted(v7 + v19) == [3, 4], f'覆盖: {v7} + {v19}'
    assert problem.vid_by_vehicle_index == {0: 7, 1: 19}
    print(f'  多车非连续 ID：vid 映射 {problem.vid_by_vehicle_index}，'
          f'route {v7} + {v19}')


def test_same_physical_anchor():
    """多车同物理 anchor：各自独立 virtual start，不冲突。"""
    ds = helpers.make_synthetic_dataset(n_customers=4, seed=3)
    env = helpers.build_env(ds, 100, 2, None)
    from strict_online_env import VehicleState
    vehicles = [VehicleState(vehicle_id=i) for i in range(2)]
    for v in vehicles:
        v.status = 'ready'
        v.current_node = 1
        v.ready_time = 2.0
    served = np.zeros(ds['coords'].shape[1], dtype=bool)
    served[0] = served[1] = True
    visible = [2, 3, 4]
    from method_adapter import build_decision_view
    view = build_decision_view(env, 0, 2.0, vehicles, served, visible, {0, 1})
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    v0 = route_visits(problem, solution, 0)
    v1 = route_visits(problem, solution, 1)
    assert sorted(v0 + v1) == [2, 3, 4]
    print(f'  同 anchor 两车：{v0} + {v1} 无冲突')


def test_initial_load_capacity():
    """初始载重 30（cap 50）：virtual start demand=30，pickup ≤ 20。"""
    ds = helpers.make_synthetic_dataset(n_customers=6, seed=4)
    ds['demands'][0] = [0, 18, 22, 18, 22, 18, 22]
    view = make_view(ds, capacity=50, num_vehicles=4, loads={0: 30.0})
    problem = build_model(view)
    assert problem.scale_meta['initial_loads'][0] == 30, \
        '标准建模：start cumul SetValue(当前载重)'
    problem2, solution, status = solve_view(view)
    assert status_ok(status)
    v0 = route_visits(problem2, solution, 0)
    idx = {n: i for i, n in enumerate(view.node_ids)}
    pickup = sum(float(view.demands[idx[c]]) for c in v0)
    assert pickup <= 20, f'车 0（载重 30）pickup {pickup} 超过剩余容量 20'
    print(f'  初始载重：virtual start demand=30，车 0 pickup={pickup} ≤ 20')


def test_near_capacity_limit():
    """初始载重接近上限：剩余容量小，只能装少量客户。"""
    ds = helpers.make_synthetic_dataset(n_customers=3, seed=5)
    ds['demands'][0] = [0, 10, 10, 10]
    view = make_view(ds, capacity=50, num_vehicles=2, loads={0: 45.0})
    problem, solution, status = solve_view(view)
    assert status_ok(status)
    v0 = route_visits(problem, solution, 0)
    v1 = route_visits(problem, solution, 1)
    idx = {n: i for i, n in enumerate(view.node_ids)}
    p0 = sum(float(view.demands[idx[c]]) for c in v0)
    assert p0 <= 5, f'车 0（载重 45）pickup {p0} 超过剩余容量 5'
    assert sorted(v0 + v1) == [1, 2, 3], '全部客户必须被覆盖（两车合力）'
    print(f'  接近容量上限：车 0 pickup={p0} ≤ 5，其余由车 1 覆盖')


def main():
    test_virtual_start_basic()
    test_multi_vehicle_distinct_anchors_and_noncontiguous_ids()
    test_same_physical_anchor()
    test_initial_load_capacity()
    test_near_capacity_limit()
    print('PASS test_problem_builder')


if __name__ == '__main__':
    main()
