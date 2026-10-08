"""RouteFinder provider 服务器烟测（P2 批次验收前置）。

验证：真实 rf-transformer-100 checkpoint 加载 → vrptw TD 构造 → 贪心解码 → 排序输出。
用法（服务器，cc_compare env，GPU）：
    cd /home/hzeng/project/MASKCO-Main
    CUDA_VISIBLE_DEVICES=1 /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/RouteFinder/a1_accept_v1/smoke_routefinder.py
"""
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(os.path.dirname(_HERE), '..', 'RRNCO', 'dcc_rh_v4')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


def make_sp(coords, demands, tws, twe, service, capacity=50.0, depot_tw_end=22.0):
    """构造与 dcc_rh_v4 SubProblem 同接口的 mock 子问题。"""
    n = len(coords)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    sp = SimpleNamespace(
        pool_customer_ids=tuple(range(1, n)),
        node_ids=tuple(range(n)),
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
    from routefinder_provider import RouteFinderProvider
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints', '100',
                                         'rf-transformer.ckpt'))
    provider = RouteFinderProvider(ckpt, device='cuda', num_loc=100)
    print('loading model...', flush=True)
    provider._load()
    print('model loaded', flush=True)
    rng = np.random.default_rng(0)
    for n_pool in (5, 25, 60):
        print('building td for pool=%d...' % n_pool, flush=True)
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
        demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
        tws = np.zeros(n_pool + 1)
        twe = np.full(n_pool + 1, 22.0)
        twe[1:] = rng.uniform(2, 20, n_pool)
        service = np.full(n_pool + 1, 0.05)
        service[0] = 0.0
        sp = make_sp(coords, demands, tws, twe, service)
        print('decoding...', flush=True)
        order = provider.order(sp)
        assert sorted(order) == sorted(range(1, n_pool + 1)), '排序未覆盖全池'
        print('pool=%3d ordering=%s' % (n_pool, list(order)[:10]), flush=True)
    print('SMOKE OK: calls=%d edd_fallback=%d' % (provider.n_calls,
                                                  provider.n_edd_fallback))


if __name__ == '__main__':
    main()
