"""MVMoE provider n=90 无竞争单决策耗时探针（正式批次定档依据）。

构造 mock sp（池 90、真实风格 id 101..190），在无其他臂争抢 CPU 的窗口对
provider.order 测 3 次墙钟，输出每次耗时与中位数（当前选档 aug1+multistart）；
同时实测官方空间内更轻档（aug1+single / aug8+single）供 >7s 时的替代。
用法（服务器，cc_compare env，CPU）：
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/MVMoE/a1_accept_v1/probe_n90.py
"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402


def make_sp(coords, demands, tws, twe, service, pool_ids, capacity=50.0,
            depot_tw_end=22.0):
    n = len(coords)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    sp = SimpleNamespace(
        pool_customer_ids=tuple(int(c) for c in pool_ids),
        node_ids=tuple([0] + [int(c) for c in pool_ids]),
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
    from mvmoe_provider import MVMoEProvider
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'pretrained',
                                         'pomo_vrptw_n100', 'epoch-5000.pt'))
    n_pool = 90
    rng = np.random.default_rng(42)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 5, n_pool)])
    tws = np.zeros(n_pool + 1)
    twe = np.full(n_pool + 1, 22.0)
    twe[1:] = rng.uniform(2, 20, n_pool)
    service = np.full(n_pool + 1, 0.1)
    service[0] = 0.0
    sp = make_sp(coords, demands, tws, twe, service,
                 [101 + k for k in range(n_pool)])

    for kwargs, label in ((dict(aug_factor=1, multistart=True),
                           'current(aug1+multistart)'),
                          (dict(aug_factor=1, multistart=False), 'lighter-aug1-single'),
                          (dict(aug_factor=8, multistart=False), 'lighter-aug8-single')):
        p = MVMoEProvider(ckpt, device='cpu', num_loc=100, **kwargs)
        p._load()
        p.order(sp)   # 预热（不计数）
        times = []
        for i in range(3):
            p._cache.clear()   # 每轮强制真解码（否则命中跨车同池缓存，计时失真）
            t0 = time.perf_counter()
            order = p.order(sp)
            times.append(time.perf_counter() - t0)
            assert sorted(order) == sorted(sp.pool_customer_ids)
        times.sort()
        print('[%s] n=%d runs=%.2f/%.2f/%.2f median=%.2fs' % (
            label, n_pool, times[0], times[1], times[2], times[1]), flush=True)
    print('PROBE_N90_DONE', flush=True)


if __name__ == '__main__':
    main()
