"""DeepACO provider 服务器烟测（A-v1 验收前置，CPU）。

验证：真实 pretrained/cvrp/cvrp100.pt 加载 → GNN 启发式 → ACO(n_ants=20, T=50)
构造 → 排序覆盖全池且输出为**真实客户 id**（mock 用非连续 id，防 index/id 混淆）。
用法（服务器，仓库根）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/DeepACO/a1_accept_v1/smoke_deepaco.py
"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


def make_sp(cust_ids, coords, demands, tws, twe, service, capacity=50.0,
            depot_tw_end=22.0):
    n = len(coords)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    sp = SimpleNamespace(
        pool_customer_ids=tuple(int(c) for c in cust_ids),
        node_ids=tuple([0] + [int(c) for c in cust_ids]),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(float(t) for t in tws),
        tw_end=tuple(float(t) for t in twe),
        service_time=tuple(float(s) for s in service),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def main():
    from deepaco_provider import DeepACOProvider
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'pretrained', 'cvrp',
                                         'cvrp100.pt'))
    provider = DeepACOProvider(ckpt, device='cpu', num_loc=100)
    print('loading model...', flush=True)
    provider._load()
    print('model loaded (n_ants=%d T=%d)' % (provider.n_ants, provider.t_iter),
          flush=True)
    rng = np.random.default_rng(0)
    for n_pool in (5, 25, 60):
        cust_ids = sorted(int(x) for x in
                          rng.choice(np.arange(101, 300), n_pool, replace=False))
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
        demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
        tws = np.zeros(n_pool + 1)
        twe = np.full(n_pool + 1, 22.0)
        twe[1:] = rng.uniform(2, 20, n_pool)
        service = np.full(n_pool + 1, 0.05)
        service[0] = 0.0
        sp = make_sp(cust_ids, coords, demands, tws, twe, service)
        t0 = time.time()
        order = provider.order(sp)
        dt = time.time() - t0
        assert sorted(order) == sorted(cust_ids), \
            '排序必须精确覆盖全池（真实 id 集合相等）'
        assert len(order) == len(set(order)), '排序含重复 id'
        t1 = time.time()
        order2 = provider.order(sp)
        dt2 = time.time() - t1
        assert order2 == order
        print('pool=%3d ids=[%d..%d] ok  solve=%.3fs  memo-hit=%.6fs  head=%s'
              % (n_pool, cust_ids[0], cust_ids[-1], dt, dt2, list(order)[:6]),
              flush=True)
    print('SMOKE OK: calls=%d cache_hits=%d edd=%d'
          % (provider.n_calls, provider.n_cache_hits, provider.n_edd_fallback))


if __name__ == '__main__':
    main()
