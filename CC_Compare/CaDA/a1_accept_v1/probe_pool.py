"""CaDA provider 大池耗时探针（pool 90/120，评估 10s 决策预算风险）。

用法（服务器）：OMP_NUM_THREADS=16 /home/hzeng/envs/cc_compare/bin/python \
    CC_Compare/CaDA/a1_accept_v1/probe_pool.py
"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np


def make_sp(coords, demands, capacity=50.0):
    n = len(coords)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    sp = SimpleNamespace(
        vehicle_id=1, anchor_node_id=0, anchor_idx=0, ready_time=0.0,
        current_load=0.0,
        pool_customer_ids=tuple(range(1, n)),
        node_ids=tuple(range(n)),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(0.0 for _ in range(n)),
        tw_end=tuple(22.0 for _ in range(n)),
        service_time=tuple(0.05 for _ in range(n)),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity), depot_tw_end=22.0,
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def main():
    from cada_provider import CaDAProvider
    ckpt = os.path.normpath(os.path.join(
        _HERE, '..', '100', 'result', '2024-1121-1355', 'checkpoint-300.pt'))
    p = CaDAProvider(ckpt, device='cpu', num_loc=100)
    p._load()
    rng = np.random.default_rng(1)
    for n_pool in (90, 120):
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
        demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
        sp = make_sp(coords, demands)
        t0 = time.perf_counter()
        order = p.order(sp)
        dt = time.perf_counter() - t0
        assert len(order) == n_pool, 'coverage fail %d != %d' % (len(order), n_pool)
        print('pool=%3d  time=%6.2fs  x15veh=%6.1fs' % (n_pool, dt, 15 * dt), flush=True)


if __name__ == '__main__':
    main()
