"""LIH provider 服务器烟测（A-v1 验收前置，CPU）。

验证：真实 outputs/cvrp_50/run/epoch-199.pt 加载 → 官方 validate() 流水线
（100-token 循环序列 + NeuRewriter 改进步）→ 排序覆盖全池且输出为**真实客户 id**
（mock 用非连续 id，防 index/id 混淆）；并测 steps=1000 的 n=50 单解耗时。
用法（服务器，仓库根）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/Learn-Improvement-Heuristics/a1_accept_v1/smoke_lih.py
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
    from lih_provider import LIHProvider
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'CVRP', 'CVRP50',
                                         'outputs', 'cvrp_50', 'run',
                                         'epoch-199.pt'))
    provider = LIHProvider(ckpt, device='cpu', num_loc=100)
    print('loading model...', flush=True)
    provider._load()
    print('model loaded (steps=%d)' % provider.steps, flush=True)
    rng = np.random.default_rng(0)
    for n_pool in (5, 25, 60):          # 60 → 分块 50+10（官方模型硬编码 50 客户槽）
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
    # 官方 test.py 步数（1000）单解耗时测量（n=50，只计时，不用于臂）
    provider.steps = 1000
    n_pool = 50
    cust_ids = sorted(int(x) for x in
                      rng.choice(np.arange(101, 300), n_pool, replace=False))
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
    twe = np.full(n_pool + 1, 22.0)
    twe[1:] = rng.uniform(2, 20, n_pool)
    sp = make_sp(cust_ids, coords, demands, np.zeros(n_pool + 1), twe,
                 np.concatenate([[0.0], np.full(n_pool, 0.05)]))
    provider._cache.clear()
    t0 = time.time()
    provider.order(sp)
    print('steps=1000 n=50 single-solve=%.1fs (官方 test.py 硬编码步数, 仅测量)'
          % (time.time() - t0), flush=True)
    print('SMOKE OK: calls=%d cache_hits=%d'
          % (provider.n_calls, provider.n_cache_hits))


if __name__ == '__main__':
    main()
