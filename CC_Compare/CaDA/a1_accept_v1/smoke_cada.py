"""CaDA provider 服务器烟测 v2（P2 批次验收前置，CPU）。

v2 变更（MVMoE 教训）：mock sp 使用**非连续真实风格客户 id**（101/105/120…，与
数组下标不同），断言 order() 返回值 == 真实 id 集合（sorted(order) ==
sorted(pool_customer_ids)），杜绝「local index 冒充真实 id」上线事故。

验证：官方 CVRP100 checkpoint-300.pt 加载 → sp→CVRP TensorDict 构造 → 官方贪心
multi-start 解码 → 全池排序输出、覆盖断言、解码终止、耗时（对照 10s 决策预算）。

用法（服务器，仓库根，cc_compare env，CPU）：
    OMP_NUM_THREADS=16 /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/CaDA/a1_accept_v1/smoke_cada.py
"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


def make_sp(node_ids, coords, demands, capacity=50.0, depot_tw_end=22.0,
            anchor_id=None, current_load=0.0):
    """构造与 dcc_rh_v4 SubProblem 同接口的 mock 子问题（node_ids = 真实 id）。

    coords/demands 按 node_ids 顺序对齐（数组下标 i ↔ node_ids[i]）。
    """
    n = len(node_ids)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    pool = [i for i in node_ids if i != 0 and i != anchor_id]
    anchor_idx = (0 if anchor_id is None else node_ids.index(anchor_id))
    sp = SimpleNamespace(
        vehicle_id=1,
        anchor_node_id=(0 if anchor_id is None else anchor_id),
        anchor_idx=anchor_idx,
        ready_time=0.0,
        current_load=float(current_load),
        pool_customer_ids=tuple(sorted(pool)),
        node_ids=tuple(node_ids),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(0.0 for _ in range(n)),
        tw_end=tuple(float(depot_tw_end) for _ in range(n)),
        service_time=tuple(0.05 for _ in range(n)),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def main():
    from cada_provider import CaDAProvider
    ckpt = os.path.normpath(os.path.join(
        _HERE, '..', '100', 'result', '2024-1121-1355', 'checkpoint-300.pt'))
    provider = CaDAProvider(ckpt, device='cpu', num_loc=100)
    t0 = time.perf_counter()
    print('loading model...', flush=True)
    provider._load()
    print('model loaded in %.1fs' % (time.perf_counter() - t0), flush=True)

    rng = np.random.default_rng(0)
    # 1) 空池
    sp_empty = make_sp([0], np.array([[0.5, 0.5]]), np.array([0.0]))
    assert provider.order(sp_empty) == (), '空池应返回 ()'
    print('empty pool -> () OK', flush=True)

    # 2) 无 anchor：pool 1 / 2 / 5 / 25 / 60，非连续真实风格 id（101/105/120…）
    for n_pool in (1, 2, 5, 25, 60):
        ids = [0] + [100 + 3 * k + rng.integers(0, 3) for k in range(n_pool)]
        ids = [ids[0]] + sorted(set(ids[1:]))          # 去重保证唯一 id
        coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
        demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
        sp = make_sp(ids, coords, demands)
        t0 = time.perf_counter()
        order = provider.order(sp)
        dt = time.perf_counter() - t0
        assert sorted(order) == sorted(sp.pool_customer_ids), \
            '排序集合 != 真实 id 池（id/index 混淆）'
        assert len(order) == len(set(order)) == len(sp.pool_customer_ids)
        assert all(isinstance(x, int) for x in order)
        print('pool=%3d  time=%6.2fs  head=%s' % (n_pool, dt, list(order)[:8]),
              flush=True)

    # 3) 带 anchor + 载重注入：pool 25，anchor id=999（池外），current_load=12.5
    n_pool = 25
    ids = [0] + [100 + 3 * k for k in range(n_pool)] + [999]
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    coords = np.vstack([coords, [[0.3, 0.7]]])          # 末行 = anchor
    demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool), [2.0]])
    sp_a = make_sp(ids, coords, demands, anchor_id=999, current_load=12.5)
    t0 = time.perf_counter()
    order_a = provider.order(sp_a)
    dt = time.perf_counter() - t0
    assert sorted(order_a) == sorted(sp_a.pool_customer_ids), 'anchor 变体 id 集合不符'
    assert 999 not in order_a, 'anchor 不应出现在排序中'
    print('pool=%3d(anchor) time=%6.2fs head=%s' % (n_pool, dt, list(order_a)[:8]),
          flush=True)

    print('SMOKE OK: calls=%d singleton=%d max_pool=%d total_runtime=%.1fs' % (
        provider.n_calls, provider.n_singleton, provider.max_pool_seen,
        provider.total_runtime_s))


if __name__ == '__main__':
    main()
