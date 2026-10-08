# -*- coding: utf-8 -*-
"""Minimal OR-Tools capacity-model repro (A-05d debugging).

Reproduces the exact modeling of SolverAcceptReplanner._solve_ortools:
1 vehicle, capacity 49999 (int units), start cumul pinned 32159,
9 clients with demands 2761/1712/1782/1548/2380/2988/1556/1107/2850 (sum 18684),
start pin + demands = 50843 > 49999 -> the model must drop >=1 client or be infeasible.
If OR-Tools returns a route covering all 9 -> the model is not enforcing capacity.
"""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

INT_SCALE = 1000
capacity_int = int((50.0 - 1e-4) * INT_SCALE)  # 49999

V, C = 1, 9
n_nodes = 1 + V + C + V
demand = np.zeros(n_nodes, dtype=np.int64)
demands = [2761, 1712, 1782, 1548, 2380, 2988, 1556, 1107, 2850]
for k, d in enumerate(demands):
    demand[1 + V + k] = d
start_load = 32159

manager = pywrapcp.RoutingIndexManager(n_nodes, V, [1], [0])
routing = pywrapcp.RoutingModel(manager)


def demand_cb(i, j):
    return int(demand[manager.IndexToNode(j)])


def dist_cb(i, j):
    return 1


routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterTransitCallback(demand_cb), 0,
    [capacity_int] * V, False, 'Capacity')
capacity_dim = routing.GetDimensionOrDie('Capacity')
capacity_dim.CumulVar(routing.Start(0)).SetValue(
    min(start_load, capacity_int))
print('pinned start cumul =', capacity_dim.CumulVar(routing.Start(0)).Value()
      if capacity_dim.CumulVar(routing.Start(0)).Bound() else 'unbound')

max_arc = 1
drop_penalty = (C + V) * max_arc + 1
for k in range(C):
    routing.AddDisjunction([manager.NodeToIndex(1 + V + k)], int(drop_penalty))

sp = pywrapcp.DefaultRoutingSearchParameters()
sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
sp.solution_limit = 30
solution = routing.SolveWithParameters(sp)
print('status =', routing.status(), 'solution =', solution is not None)
if solution is not None:
    idx = routing.Start(0)
    route, cumuls = [], []
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        route.append(node)
        cumuls.append(solution.Value(capacity_dim.CumulVar(idx)))
        idx = solution.Value(routing.NextVar(idx))
    dropped = [1 + V + k for k in range(C)
               if solution.Value(routing.NextVar(manager.NodeToIndex(1 + V + k)))
               == manager.NodeToIndex(1 + V + k)]
    print('route nodes =', route)
    print('cumuls =', cumuls)
    print('dropped =', dropped)
    print('max cumul =', max(cumuls) if cumuls else None)
