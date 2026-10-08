"""SGBS provider 服务器烟测（P3 批次验收前置，CPU）。

验证：真实 checkpoint-30500.pt 加载 → 官方 SGBS 流程（POMO 起点 + 仿真引导束搜索）
→ 全池排序输出；断言覆盖全池 + 确定性 + 空池/超域防御；打印各池耗时。

用法（服务器，cc_compare env，CPU）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/SGBS/a1_accept_v1/smoke_sgbs.py [--beta K] [--gamma 4] [--num-aug 1|8] [--threads 16]

档位选择：官方论文 CVRP100 beam=1280（CPU 不可行）→ 在官方参数空间
（--beta/--gamma/-disable_aug 均官方 test.py 已存在）取 pool=60 单决策 ≤ ~10s 的
最大 beam（候选 100/50/10，仍超限取 repo 默认 beta=4），落 sgbs_provider.py 的
_CONFIG 并写进 RESULTS.md。
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


def make_sp(coords, demands, pool, capacity=50.0, depot_tw_end=22.0):
    ids = sorted(set([0]) | set(pool))
    idx = {n: i for i, n in enumerate(ids)}
    dist = [[np.hypot(coords[idx[a]][0] - coords[idx[b]][0],
                      coords[idx[a]][1] - coords[idx[b]][1])
             for b in ids] for a in ids]
    sp = SimpleNamespace(
        pool_customer_ids=tuple(sorted(pool)),
        node_ids=tuple(ids),
        coords=tuple(tuple(float(x) for x in coords[idx[c]]) for c in ids),
        demands=tuple(float(demands[idx[c]]) for c in ids),
        tw_start=tuple(0.0 for _ in ids),
        tw_end=tuple(float(depot_tw_end) for _ in ids),
        service_time=tuple(0.0 for _ in ids),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=int, default=None)
    ap.add_argument('--gamma', type=int, default=None)
    ap.add_argument('--num-aug', type=int, default=None)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--probe90', action='store_true',
                    help='仅做 n=90 无竞争耗时探针：3 次单决策求解墙钟（绕过 memo）')
    args = ap.parse_args()

    import sgbs_provider as P
    if args.threads:
        P._CONFIG['num_threads'] = args.threads
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'CVRP',
                                         '1_pre_trained_model',
                                         'Saved_CVRP100_Model',
                                         'checkpoint-30500.pt'))
    provider = P.SGBSProvider(ckpt, device='cpu', num_loc=100,
                              beta=args.beta, gamma=args.gamma,
                              num_aug=args.num_aug)
    print('config: beta=%d gamma=%d num_aug=%d threads=%d' % (
        provider.beta, provider.gamma, provider.num_aug,
        P._CONFIG['num_threads']))
    print('loading model...', flush=True)
    provider._load()
    print('model loaded', flush=True)

    if args.probe90:
        rng = np.random.default_rng(7)
        pool_ids = [100 + i * 2 for i in range(90)]
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (max(pool_ids), 2))])
        demands = np.zeros(max(pool_ids) + 1)
        demands[pool_ids] = rng.uniform(1, 3, len(pool_ids))
        sp = make_sp(coords, demands, pool_ids)
        provider._solve(sp, pool_ids)      # 预热（不计时）
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            provider._solve(sp, pool_ids)
            times.append(time.perf_counter() - t0)
            print('probe90 run%d t=%.2fs' % (i + 1, times[-1]), flush=True)
        times.sort()
        print('PROBE90 median=%.2fs (n=90, no-contention)' % times[1])
        return

    rng = np.random.default_rng(0)
    # 非连续真实风格客户 id（A-v1 真实 id 与数组下标不同，防 index/id 混淆）
    for n_pool, pool_ids in ((5, [101, 105, 120, 133, 141]),
                             (25, [200 + i * 3 for i in range(25)]),
                             (60, [300 + i for i in range(60)])):
        n_nodes = max(pool_ids) + 1
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_nodes - 1, 2))])
        demands = np.zeros(n_nodes)
        demands[pool_ids] = rng.uniform(1, 3, len(pool_ids))
        sp = make_sp(coords, demands, pool_ids)
        t0 = time.perf_counter()
        order = provider.order(sp)
        dt = time.perf_counter() - t0
        assert sorted(order) == sorted(pool_ids), '排序未覆盖全池（真实 id 集合）'
        assert len(set(order)) == len(order), '排序含重复'
        assert all(isinstance(x, int) for x in order)
        order2 = provider.order(sp)
        assert order == order2, '两次调用结果不一致（确定性破坏）'
        print('pool=%3d t=%.2fs n_solves=%d calls=%d head=%s' % (
            n_pool, dt, provider.n_solves, provider.n_calls,
            list(order)[:8]), flush=True)

    assert provider.order(make_sp(np.array([[0.5, 0.5]]), np.array([0.0]),
                                  [])) == (), '空池应返回 ()'
    print('empty pool OK', flush=True)

    big = [400 + i for i in range(101)]
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (max(big), 2))])
    demands = np.zeros(max(big) + 1)
    demands[big] = rng.uniform(1, 3, len(big))
    spb = make_sp(coords, demands, big)
    ob = provider.order(spb)
    assert sorted(ob) == sorted(big) and provider.n_edd_fallback == 1
    print('big-pool EDD fallback OK', flush=True)

    print('SMOKE OK: calls=%d solves=%d edd_fallback=%d solve_time=%.2fs'
          % (provider.n_calls, provider.n_solves, provider.n_edd_fallback,
             provider.solve_time_s))


if __name__ == '__main__':
    main()
