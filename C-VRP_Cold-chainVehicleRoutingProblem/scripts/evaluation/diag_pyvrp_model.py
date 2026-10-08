# -*- coding: utf-8 -*-
"""PyVRP 建模烟测（A-05e）：加载 diag_solve_inputs.json，用 0.11.3 API 重建模型求解。"""
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
                                     _demand_int, _start_pin_int)

CAP = 50.0


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
    X = None  # 坐标未落盘，用 depot/anchor 近似构造（边已显式给出，坐标仅元数据）
    # 从 dist 第一行恢复不了坐标：改为直接用 add_client 的坐标传 0（0.11 坐标仅用于输出元数据）。
    inv_speed = 1.0 / float(d['tw_speed'])
    tws = np.array(d['tw_start'], float)
    twe = np.array(d['tw_end'], float)
    st_ = np.array(d['service'], float)
    dem = np.array(d['demands'], float)

    from pyvrp import Model
    from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria

    capacity_int = int(math.floor((CAP - 1e-4) * INT_SCALE))

    m = Model()
    end_depot = m.add_depot(0, 0, tw_early=0, tw_late=floor_int(depot_deadline))
    client_idx_of = {}
    client_locs = []
    for k, o in enumerate(clients):
        cl = m.add_client(0, 0, pickup=_demand_int(dem[o]),
                          service_duration=ceil_int(st_[o]),
                          tw_early=ceil_int(tws[o]),
                          tw_late=floor_int(twe[o]),
                          required=True, prize=0)
        client_idx_of[(1 + len(vids)) + k] = o
        client_locs.append(cl)

    vid_by_vt = {}
    start_locs = []
    for k, vid in enumerate(vids):
        s = sts[vid]
        start_depot = m.add_depot(0, 0, tw_early=ceil_int(s['anchor_time']),
                                  tw_late=9223372036854775807)
        vt_idx = len(m.vehicle_types)
        m.add_vehicle_type(1, capacity=capacity_int, start_depot=start_depot,
                           end_depot=end_depot, fixed_cost=0, tw_early=0,
                           tw_late=floor_int(depot_deadline),
                           unit_distance_cost=1, unit_duration_cost=0,
                           initial_load=_start_pin_int(s['load'], capacity_int),
                           name='vehicle_%d' % vid)
        vid_by_vt[vt_idx] = vid
        start_locs.append(start_depot)

    coord_of = [0] + [int(o) for o in clients] + [int(sts[vid]['anchor'])
                                                   for vid in vids]
    all_locs = [end_depot] + client_locs + start_locs
    for a_i, a in enumerate(all_locs):
        for b_i, b in enumerate(all_locs):
            if a_i == b_i:
                m.add_edge(a, b, 0, 0)
                continue
            dd = float(D[coord_of[a_i], coord_of[b_i]])
            m.add_edge(a, b, round_int(dd), ceil_int(dd * inv_speed))

    stop = MultipleCriteria([MaxIterations(1000), MaxRuntime(5.0)])
    res = m.solve(stop=stop, seed=0)
    print('solve result:', None if res is None else 'ok')
    if res is None or res.best is None:
        return
    print('is_feasible =', res.best.is_feasible())
    plan = {}
    covered = []
    from pyvrp import Model as _M
    _data = m.data()
    print('num_depots/num_clients/num_locations:',
          _data.num_depots, _data.num_clients, _data.num_locations)
    print('data clients identity vs created:',
          [any(cl is c for c in _data.clients()) for cl in client_locs])
    print('route dir:', [a for a in dir(res.best.routes()[0]) if not a.startswith('_')])
    for route in res.best.routes():
        vt = route.vehicle_type()
        vid = vid_by_vt.get(vt)
        raw = list(route)
        print(f'vid={vid} vt={vt} raw={raw}')
        seq = [client_idx_of[a] for a in route if a in client_idx_of]
        if seq:
            plan[vid] = seq
        covered.extend(seq)
        loads = [dem[o] for o in seq]
        print(f'vid={vid} seq={seq} real_loads={loads}')
    print('covered == clients:', sorted(covered) == sorted(clients))
    for vid in vids:
        seq = plan.get(vid, [])
        pin = _start_pin_int(sts[vid]['load'], capacity_int)
        seen = pin + sum(_demand_int(dem[o]) for o in seq)
        print(f'audit vid={vid}: pin={pin} seq={seq} seen={seen} cap={capacity_int} '
              f'{"OVER" if seen > capacity_int else "ok"}')


if __name__ == '__main__':
    main()
