"""MVMoE provider 服务器烟测（A-v1 臂验收前置）。

验证：真实 pomo_vrptw_n100/epoch-5000.pt checkpoint 加载 → 官方 VRPTWEnv 缩放
构造 → POMO 解码（argmax）→ 排序输出。断言（pool 用真实风格客户 id ≠ 数组下标，
回归 provider 初版 local-index/真实-id 混淆 bug）：
  - pool 5/25/60：排序精确覆盖全池（无缺失/重复/外部）；
  - 空池 → ()；
  - 紧 TW 恶意客户（官方 mask 下永久不可达）：解码在步数上限内终止 + EDD 补尾
    仍精确覆盖全池（防死循环）；
  - 跨车同池缓存：同池两次调用结果一致且第二次命中缓存；
  - 计时：multistart(aug=1) / single(aug=8) / single(aug=1) 三档 pool=60 解码
    耗时 → 供 10s/决策预算下选档（官方评测空间内）。

用法（服务器，cc_compare env，CPU）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/MVMoE/a1_accept_v1/smoke_mvmoe.py
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
    """构造与 dcc_rh_v4 SubProblem 同接口的 mock 子问题。

    pool_ids 用真实风格客户 id（≠ 数组下标，如 101..）——回归「local index 与
    真实 id 混淆」类 bug（provider 初版 99.5% EDD fallback 的根因）。
    node_ids = [0] + pool_ids；node_index 按 node_ids 查找。
    """
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


def mock_pool(n_pool, seed=0, tight_one=False, id_off=101):
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 5, n_pool)])
    tws = np.zeros(n_pool + 1)
    twe = np.full(n_pool + 1, 22.0)
    twe[1:] = rng.uniform(2, 20, n_pool)
    service = np.full(n_pool + 1, 0.1)
    service[0] = 0.0
    if tight_one:
        twe[1] = 0.05   # 官方 mask 下 depot 起永久不可达（arrival > tw_end）
    pool_ids = [id_off + k for k in range(1, n_pool + 1)]
    return make_sp(coords, demands, tws, twe, service, pool_ids)


def time_order(provider, sp, label, expect_ok=True):
    t0 = time.perf_counter()
    order = provider.order(sp)
    dt = time.perf_counter() - t0
    pool = sorted(sp.pool_customer_ids)
    if expect_ok:
        assert sorted(order) == pool, '%s: 排序未覆盖全池 %s vs %s' % (
            label, sorted(order), pool)
        assert len(order) == len(set(order)), '%s: 排序含重复' % label
    print('[%s] pool=%-3d decode=%.2fs first10=%s' % (
        label, len(pool), dt, list(order)[:10]), flush=True)
    return order, dt


def main():
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'pretrained',
                                         'pomo_vrptw_n100', 'epoch-5000.pt'))
    assert os.path.exists(ckpt), 'checkpoint 缺失: %s' % ckpt
    provider = None
    try:
        from mvmoe_provider import MVMoEProvider
        provider = MVMoEProvider(ckpt, device='cpu', num_loc=100,
                                 aug_factor=1, multistart=True)
        t0 = time.perf_counter()
        provider._load()
        print('>> loaded ckpt problem=%s epoch=%s (%.1fs)' % (
            provider._ckpt_problem, provider._ckpt_epoch,
            time.perf_counter() - t0), flush=True)
        assert provider._ckpt_problem == 'VRPTW', (
            'checkpoint problem=%r != VRPTW' % provider._ckpt_problem)

        # 1) 标准池 5/25/60（多起点 aug=1，默认档）
        for n_pool in (5, 25, 60):
            time_order(provider, mock_pool(n_pool), 'multistart-aug1')

        # 2) 空池
        sp0 = mock_pool(3)
        sp0.pool_customer_ids = tuple()
        assert provider.order(sp0) == (), '空池必须返回 ()'
        print('[empty] -> ()', flush=True)

        # 3) 紧 TW 恶意客户：终止性 + 覆盖（EDD 补尾）
        sp_tight = mock_pool(20, seed=7, tight_one=True)
        time_order(provider, sp_tight, 'tight-tw')

        # 4) 跨车同池缓存（同池不同 anchor 状态 → 相同排序 + 缓存命中）
        sp_a = mock_pool(25, seed=3)
        sp_b = mock_pool(25, seed=3)
        o1, _ = time_order(provider, sp_a, 'cache-1st')
        o2, _ = time_order(provider, sp_b, 'cache-2nd')
        assert o1 == o2, '同池两次排序不一致（确定性破坏）'
        assert provider.n_cache_hits >= 1, '跨车缓存未命中'
        print('[cache] hit_count=%d' % provider.n_cache_hits, flush=True)

        # 5) 计时对比：single(aug=8) / single(aug=1)（pool=60 最重档）
        sp60 = mock_pool(60)
        for kwargs, label in ((dict(aug_factor=8, multistart=False),
                               'single-aug8'),
                              (dict(aug_factor=1, multistart=False),
                               'single-aug1')):
            p2 = MVMoEProvider(ckpt, device='cpu', num_loc=100, **kwargs)
            p2._load()
            time_order(p2, sp60, label)

        print('SMOKE OK: calls=%d cache_hits=%d edd_fallback=%d '
              'total_decode_s=%.1f' % (provider.n_calls, provider.n_cache_hits,
                                       provider.n_edd_fallback,
                                       provider.total_decode_s), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
