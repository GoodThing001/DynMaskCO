# -*- coding: utf-8 -*-
"""Minimal OR-Tools capacity-model repro #2: full model shape (A-05d debugging).

Mirrors SolverAcceptReplanner._solve_ortools exactly with V=2 vehicles:
v0 idle (start load 0), v14 committed (start load pinned 32159), 9 clients,
dummy nodes + NextVar(dummy)==End + SetAllowedVehiclesForIndex + time dimension.
Demands sum 18684; v14 pin 32159; 32159+18684=50843 > 49999 -> must drop >=1
(or move to v0). Checks whether the full model shape still enforces capacity.
"""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

INT_SCALE = 1000
capacity_int = int((50.0 - 1e-4) * INT_SCALE)  # 49999

V, C = 2, 9
n_nodes = 1 + V + C + V
demand = np.zeros(n_nodes, dtype=np.int64)
demands = [2761, 1712, 1782, 1548, 2380, 2988, 1556, 1107, 2850]
for k, d in enumerate(demands):
    demand[1 + V + k] = d

manager = pywrapcp.RoutingIndexManager(n_nodes, V, [1, 2], [0, 0])
routing = pywrapcp.RoutingModel(manager)


def demand_cb(i, j):
    return int(demand[manager.IndexToNode(j)])


def dist_cb(i, j):
    return 1


def transit_cb(i, j):
    return 1


routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterTransitCallback(demand_cb), 0,
    [capacity_int] * V, False, 'Capacity')
capacity_dim = routing.GetDimensionOrDie('Capacity')
capacity_dim.CumulVar(routing.Start(0)).SetValue(0)
capacity_dim.CumulVar(routing.Start(1)).SetValue(32159)

# time dimension (horizon large so it doesn't bind)
routing.AddDimension(routing.RegisterTransitCallback(transit_cb), 1000000,
                     1000000, False, 'Time')

# dummy nodes: per-vehicle, NextVar(dummy)==End
for k in range(V):
    dummy_idx = manager.NodeToIndex(1 + V + C + k)
    routing.SetAllowedVehiclesForIndex([k], dummy_idx)
    routing.solver().Add(routing.NextVar(dummy_idx) == routing.End(k))

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
    for k in range(V):
        idx = routing.Start(k)
        seq, cumuls = [], []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            seq.append(node)
            cumuls.append(solution.Value(capacity_dim.CumulVar(idx)))
            idx = solution.Value(routing.NextVar(idx))
        print(f'vehicle {k} route nodes =', seq, 'cumuls =', cumuls)
    dropped = [1 + V + k for k in range(C)
               if solution.Value(routing.NextVar(manager.NodeToIndex(1 + V + k)))
               == manager.NodeToIndex(1 + V + k)]
    print('dropped =', dropped)
