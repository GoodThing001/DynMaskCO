# -*- coding: utf-8 -*-
"""独立重放 OR-Tools 接单求解（A-05d 调试）：加载 diag_solve_inputs.json，
用与 _solve_ortools 完全一致的建模重建并求解，审计返回计划的容量维度。"""
import json
import math
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from solver_accept_replanner import (INT_SCALE, ceil_int, floor_int, round_int,
                                     _demand_int, _OT_SUCCESS_STATUSES)

CAP = 50.0
TIME_LIMIT = 10.0
SOLUTION_LIMIT = 30


def main():
    dump_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'results', 'diag_solve_inputs.json')
    with open(dump_path, encoding='utf-8') as f:
        d = json.load(f)
    sts = {int(k): v for k, v in d['sts'].items()}
    vids = sorted(sts)
    clients = [int(c) for c in d['clients']]
    depot_deadline = float(d['depot_deadline'])
    D = np.array(d['dist'], float)
    tws = np.array(d['tw_start'], float)
    twe = np.array(d['tw_end'], float)
    st_ = np.array(d['service'], float)
    dem = np.array(d['demands'], float)
    inv_speed = 1.0 / float(d['tw_speed'])

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    V, C = len(vids), len(clients)
    n_nodes = 1 + V + C + V
    coord_idx = {0: 0}
    for k, vid in enumerate(vids):
        coord_idx[1 + k] = sts[vid]['anchor']
    client_node = {1 + V + k: o for k, o in enumerate(clients)}
    for k, o in enumerate(clients):
        coord_idx[1 + V + k] = o
    for k in range(V):
        coord_idx[1 + V + C + k] = 0

    dist = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    trav = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dd = float(D[coord_idx[i], coord_idx[j]])
            dist[i, j] = round_int(dd)
            trav[i, j] = ceil_int(dd * inv_speed)
    service = np.zeros(n_nodes, dtype=np.int64)
    for k, o in enumerate(clients):
        service[1 + V + k] = ceil_int(st_[o])
    transit = trav + service[:, None]
    demand = np.zeros(n_nodes, dtype=np.int64)
    for k, o in enumerate(clients):
        demand[1 + V + k] = _demand_int(dem[o])

    manager = pywrapcp.RoutingIndexManager(n_nodes, V,
                                          list(range(1, 1 + V)), [0] * V)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

    def transit_cb(i, j):
        return int(transit[manager.IndexToNode(i)][manager.IndexToNode(j)])

    def demand_cb(i, j):
        return int(demand[manager.IndexToNode(j)])

    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(dist_cb))

    capacity_int = int(math.floor((CAP - 1e-4) * INT_SCALE))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterTransitCallback(demand_cb), 0,
        [capacity_int] * V, False, 'Capacity')
    capacity_dim = routing.GetDimensionOrDie('Capacity')
    for k, vid in enumerate(vids):
        capacity_dim.CumulVar(routing.Start(k)).SetValue(
            min(ceil_int(sts[vid]['load']), capacity_int))

    horizon = floor_int(depot_deadline)
    routing.AddDimension(routing.RegisterTransitCallback(transit_cb),
                         horizon, horizon, False, 'Time')
    td = routing.GetDimensionOrDie('Time')
    for k, vid in enumerate(vids):
        td.CumulVar(manager.NodeToIndex(1 + k)).SetRange(
            ceil_int(sts[vid]['anchor_time']), ceil_int(sts[vid]['anchor_time']))
    for k, o in enumerate(clients):
        td.CumulVar(manager.NodeToIndex(1 + V + k)).SetRange(
            ceil_int(tws[o]), floor_int(twe[o]))
    for k in range(V):
        td.CumulVar(manager.NodeToIndex(1 + V + C + k)).SetRange(0, horizon)

    for k in range(V):
        dummy_idx = manager.NodeToIndex(1 + V + C + k)
        routing.SetAllowedVehiclesForIndex([k], dummy_idx)
        routing.solver().Add(routing.NextVar(dummy_idx) == routing.End(k))

    max_arc = max(int(dist.max()), 1)
    drop_penalty = (C + V) * max_arc + 1
    for k in range(C):
        routing.AddDisjunction([manager.NodeToIndex(1 + V + k)],
                               int(drop_penalty))

    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    sp.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    sp.solution_limit = SOLUTION_LIMIT
    sp.time_limit.FromMilliseconds(int(TIME_LIMIT * 1000))
    sp.log_search = False
    solution = routing.SolveWithParameters(sp)
    status = routing.status()
    print('V=', V, 'C=', C, 'status =', status, 'solution =', solution is not None)
    print('vids =', vids)
    print('sts =', sts)
    print('clients =', clients)
    if solution is None:
        return
    for k, vid in enumerate(vids):
        idx = routing.Start(k)
        seq, cumuls = [], []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node in client_node:
                seq.append(client_node[node])
            cumuls.append((node, solution.Value(capacity_dim.CumulVar(idx))))
            idx = solution.Value(routing.NextVar(idx))
        print(f'vehicle k={k} vid={vid} seq={seq}')
        print('   cumuls(node, cap) =', cumuls)
    dropped = [client_node[1 + V + k] for k in range(C)
               if solution.Value(routing.NextVar(
                   manager.NodeToIndex(1 + V + k))) == manager.NodeToIndex(1 + V + k)]
    print('dropped clients =', dropped)
    for k, vid in enumerate(vids):
        idx = routing.Start(k)
        seq = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node in client_node:
                seq.append(client_node[node])
            idx = solution.Value(routing.NextVar(idx))
        pin = min(ceil_int(sts[vid]['load']), capacity_int)
        seen = pin + sum(_demand_int(dem[o]) for o in seq)
        print(f'audit vid={vid}: pin={pin} seq={seq} seen={seen} cap={capacity_int} '
              f'{"OVER" if seen > capacity_int else "ok"}')


if __name__ == '__main__':
    main()
