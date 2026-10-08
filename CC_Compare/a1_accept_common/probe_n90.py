"""n=90 最大池耗时探针（正式批次定档依据，2026-09-30）。

对指定方法测 3 次单决策 provider.order(n=90 真实风格 id) 墙钟，输出中位数。
每个方法独立进程运行（避免跨方法 utils/problems 命名空间冲突）：
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/a1_accept_common/probe_n90.py <am|deepaco|omnivrp|lih>
- am: 官方 greedy 档
- deepaco: n_ants=20，官方 t_aco 网格 T ∈ {10,20,30,40,50,100} 全扫描（定 ≤7s 档）
- omnivrp: 官方评测档（pomo_size=problem_size，aug×8）
- lih: 官方最小改进步数档 steps=100（n=90 → 2 块 50+40）；另附 steps=1000 参考
"""
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

_ROOT = '/home/hzeng/project/MASKCO-Main/CC_Compare'
_METHOD = sys.argv[1] if len(sys.argv) > 1 else 'am'
sys.path.insert(0, os.path.join(_ROOT, {
    'am': 'AttentionModel', 'deepaco': 'DeepACO',
    'omnivrp': 'Omni-VRP', 'lih': 'Learn-Improvement-Heuristics',
}[_METHOD] + '/a1_accept_v1'))


def make_sp(n_pool, seed=7):
    rng = np.random.default_rng(seed)
    cust_ids = sorted(int(x) for x in
                      rng.choice(np.arange(101, 300), n_pool, replace=False))
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
    tws = np.zeros(n_pool + 1)
    twe = np.full(n_pool + 1, 22.0)
    twe[1:] = rng.uniform(2, 20, n_pool)
    service = np.full(n_pool + 1, 0.05)
    service[0] = 0.0
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
        capacity=50.0,
        depot_tw_end=22.0,
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def probe(provider, sp, n_runs=3):
    times = []
    for _ in range(n_runs):
        provider._cache.clear()
        t0 = time.time()
        order = provider.order(sp)
        times.append(time.time() - t0)
        assert sorted(order) == sorted(list(sp.pool_customer_ids)), '覆盖失败'
    return times


def main():
    sp = make_sp(90)
    if _METHOD == 'am':
        from am_provider import AMProvider
        p = AMProvider(_ROOT + '/AttentionModel/pretrained/cvrp_100/epoch-99.pt',
                       device='cpu')
        p._load()
        t = probe(p, sp)
        print('AM greedy n=90 median=%.3fs all=%s'
              % (statistics.median(t), ['%.3f' % x for x in t]), flush=True)
    elif _METHOD == 'deepaco':
        from deepaco_provider import DeepACOProvider
        p = DeepACOProvider(_ROOT + '/DeepACO/pretrained/cvrp/cvrp100.pt',
                            device='cpu')
        p._load()
        for T in (100, 50, 40, 30, 20, 10):
            p.t_iter = T
            t = probe(p, sp)
            print('DeepACO n_ants=20 T=%3d n=90 median=%.3fs all=%s'
                  % (T, statistics.median(t), ['%.3f' % x for x in t]),
                  flush=True)
    elif _METHOD == 'omnivrp':
        from omnivrp_provider import OmniVRPProvider
        p = OmniVRPProvider(
            _ROOT + '/Omni-VRP/pretrained/POMO-CVRP/uniform/'
            'checkpoint-30500-cvrp100-instance-norm.pt', device='cpu')
        p._load()
        t = probe(p, sp)
        print('OmniVRP aug8 n=90 median=%.3fs all=%s'
              % (statistics.median(t), ['%.3f' % x for x in t]), flush=True)
    elif _METHOD == 'lih':
        from lih_provider import LIHProvider
        p = LIHProvider(
            _ROOT + '/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/'
            'cvrp_50/run/epoch-199.pt', device='cpu')
        p._load()
        p.steps = 100
        t = probe(p, sp)
        print('LIH steps=100 (2块) n=90 median=%.3fs all=%s'
              % (statistics.median(t), ['%.3f' % x for x in t]), flush=True)
        p.steps = 1000
        t = probe(p, sp)
        print('LIH steps=1000(参考) n=90 median=%.3fs all=%s'
              % (statistics.median(t), ['%.3f' % x for x in t]), flush=True)
    print('PROBE %s DONE' % _METHOD, flush=True)


if __name__ == '__main__':
    main()
