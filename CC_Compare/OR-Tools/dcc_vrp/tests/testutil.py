"""OR-Tools tests 公共：路径引导 + 求解工具。"""
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

import helpers  # noqa: F401  (common/tests 合成数据)
from method_adapter import build_decision_view
from strict_online_env import VehicleState


def make_view(ds, capacity=50, num_vehicles=4, clock=0.0, served=None,
              ready_times=None, loads=None, statuses=None, replan_ids=None):
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
    if replan_ids is None:
        replan_ids = set(range(num_vehicles))
    return build_decision_view(env, 0, clock, vehicles, served, visible, replan_ids)


def solve_view(view, solution_limit=30, time_limit_s=30.0):
    """build model → solve → (problem, solution, routing.status())。"""
    from problem_builder import build_model
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    problem = build_model(view)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    sp.solution_limit = solution_limit
    sp.time_limit.seconds = int(time_limit_s)
    sp.log_search = False
    solution = problem.routing.SolveWithParameters(sp)
    return problem, solution, problem.routing.status()


def status_ok(status):
    """成功状态：1 SUCCESS / 2 PARTIAL / 7 OPTIMAL（9.11 中 7=证明最优）。"""
    return status in (1, 2, 7)
