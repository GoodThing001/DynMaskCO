"""B2 tests 公共：路径引导 + solve_view 工具。"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_DCC_VRP = os.path.dirname(_TESTS)
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
_COMMON_TESTS = os.path.join(_COMMON, 'tests')
for p in (_TESTS, _DCC_VRP, _COMMON, _COMMON_TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: F401

import numpy as np

import helpers  # noqa: F401  (common/tests/helpers: 合成数据集)
from method_adapter import build_decision_view
from strict_online_env import VehicleState


def make_view(ds, capacity=50, num_vehicles=4, clock=0.0, reveal_spec=None,
              served=None, ready_times=None, loads=None, statuses=None,
              committed_next=None, replan_ids=None):
    """构造 DecisionView（未运行的 env + 手工车辆状态）。"""
    env = helpers.build_env(ds, capacity, num_vehicles, None)
    vehicles = [VehicleState(vehicle_id=i) for i in range(num_vehicles)]
    served = np.zeros(ds['coords'].shape[1], dtype=bool) if served is None else served
    served[0] = True
    visible = [c for c in range(1, ds['coords'].shape[1])
               if float(ds['reveal_time'][0, c]) <= clock + 1e-6
               and ds['demands'][0, c] > 0]
    if ready_times:
        for vid, t in ready_times.items():
            vehicles[vid].ready_time = float(t)
    if loads:
        for vid, l in loads.items():
            vehicles[vid].current_load = float(l)
    if statuses:
        for vid, s in statuses.items():
            vehicles[vid].status = s
            if s == 'ready':
                vehicles[vid].current_node = vehicles[vid].current_node or 0
    if committed_next:
        for vid, n in committed_next.items():
            vehicles[vid].status = 'committed'
            vehicles[vid].committed_next = n
            vehicles[vid].committed_finish = vehicles[vid].ready_time
    if replan_ids is None:
        replan_ids = set(range(num_vehicles))
    return build_decision_view(env, 0, clock, vehicles, served, visible, replan_ids)


def solve_view(view, max_iterations=200, seed=0, require_complete=True):
    """build model → solve → (problem, result)。

    require_complete=True（测试默认）：required 客户严格覆盖；
    集成测试用 adapter 的 deferral 模式（require_complete=False）。
    """
    from problem_builder import build_model
    from pyvrp.stop import MaxIterations
    problem = build_model(view, require_complete=require_complete)
    result = problem.model.solve(stop=MaxIterations(max_iterations), seed=seed,
                                 display=False)
    return problem, result
