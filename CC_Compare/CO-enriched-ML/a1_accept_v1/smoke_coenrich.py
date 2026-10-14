"""CO-enriched-ML A-v1 provider 烟测（CPU / 纯 numpy）。

验收内容（对齐工作包 §4.3「每个 learned 适配最小验收」）：
  1. 真实 checkpoint 推理被调用（官方 ML-CO 权重经 NumPy 复刻前向；对账 max|Δ|）；
  2. `order(sp)` 输出 = 可见池的一个排列（覆盖 + 无重复）；
  3. 偏好**确实改变排序**（与 EDD 基线比较：不同位置数 + 秩相关）；
  4. 确定性（同 sp 两次 → 同排序）；
  5. 未来信息不变性（只扰动**不可见**节点字段 → sp canonical_hash 与排序都不变）；
  6. 边界：空池 / 单客户池 / 大池；
  7. 逐池耗时（10s 预算证据）；
  8. PC-HGS（原方法 CO 层）可选：给出 exe 时验证「服务前缀 + 补全」仍为合法排列。

用法（服务器，cc_compare env；或本地 cc_ortools，需先下载 artifacts）：
    python smoke_coenrich.py --artifacts <a1_accept_v1/artifacts> [--pchgs-exe <PCHGS>]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CC_COMPARE = os.path.dirname(os.path.dirname(_HERE))          # CC_Compare/
for _p in (_HERE, os.path.join(_CC_COMPARE, 'RRNCO', 'dcc_rh_v4')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coenrich_model import NumpyMLP                     # noqa: E402
from coenrich_provider import COEnrichedMLProvider      # noqa: E402
from subproblem import build_subproblem                 # noqa: E402


# --------------------------------------------------------------------------- #
# mock view（自包含；节点 id 可为非连续真实风格 id）
# --------------------------------------------------------------------------- #
def _euclid(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def make_view(node_ids, coords, demands, tw_start, tw_end, service, pool,
              capacity=50.0, depot_tw_end=22.0, anchor=0, ready_time=0.0,
              load=0.0, speed=1.5, replan_ids=(0,)):
    n = len(node_ids)
    dist = [[_euclid(coords[i], coords[j]) for j in range(n)] for i in range(n)]
    travel = [[dist[i][j] / speed for j in range(n)] for i in range(n)]
    view = SimpleNamespace(
        node_ids=tuple(int(x) for x in node_ids),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(float(t) for t in tw_start),
        tw_end=tuple(float(t) for t in tw_end),
        service_time=tuple(float(s) for s in service),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in travel),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
        pool_customer_ids=tuple(sorted(int(c) for c in pool)),
        vehicles=(),
        replan_ids=tuple(sorted(int(x) for x in replan_ids)),
        has_future_reveal=True,
    )
    view.vehicles = (SimpleNamespace(
        vehicle_id=0, anchor_node_id=int(anchor),
        anchor_idx=view.node_ids.index(int(anchor)),
        ready_time=float(ready_time), load=float(load), status='ready',
        mutable_suffix=()),)
    return view


def scenario(pool_ids, n_hidden, seed=7, hidden_jitter=0.0):
    """构造「depot + pool + n_hidden 未揭示节点」的 view；hidden_jitter 只扰动隐藏节点。"""
    rng = np.random.default_rng(seed)
    node_ids = [0] + sorted(int(c) for c in pool_ids) + \
        [1000 + i for i in range(n_hidden)]
    n = len(node_ids)
    coords = rng.uniform(0, 20, size=(n, 2))
    demands = rng.integers(1, 6, size=n).astype(float)
    demands[0] = 0.0
    tw_start = np.zeros(n)
    tw_end = rng.uniform(8.0, 22.0, size=n)
    tw_end[0] = 22.0
    service = np.full(n, 0.05)
    service[0] = 0.0
    if hidden_jitter:                       # 只改隐藏（非 pool、非 depot）节点
        hid = [i for i, nid in enumerate(node_ids) if nid >= 1000]
        for i in hid:
            coords[i] = coords[i] + hidden_jitter
            tw_end[i] = tw_end[i] + hidden_jitter
            demands[i] = demands[i] + hidden_jitter
    view = make_view(node_ids, coords, demands, tw_start, tw_end, service,
                     pool=pool_ids)
    return view, node_ids


def spearman(a, b):
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    x = np.array([ra[v] for v in a], dtype=float)
    y = np.array([rb[v] for v in a], dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default=os.path.join(_HERE, 'artifacts'))
    ap.add_argument('--pchgs-exe', default=None)
    ap.add_argument('--pchgs-seconds', type=int, default=1)
    args = ap.parse_args()

    ok_all = True

    def check(name, cond, detail=''):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))

    # ---- 0. 权重复刻对账 ----
    def _pick(d, names):
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
        return os.path.join(d, names[0])

    npz = _pick(args.artifacts, ['official_weights.npz', 'a1_weights.npz'])
    struct = _pick(args.artifacts, ['official_model_struct.json', 'a1_model_struct.json'])
    chk = os.path.join(args.artifacts, 'official_forward_check.npz')
    print('weights=%s' % npz)
    m = NumpyMLP(npz, struct)
    print('模型层数=%d 输入维=%d 层结构=%s'
          % (len(m.layers), m.input_dim,
             [(l['name'], l['W'].shape, l['activation']) for l in m.layers]))
    if os.path.isfile(chk):
        dmax = m.verify_against_official(chk)
        check('官方权重加载 + NumPy 前向复刻对账 (max|Δ| < 1e-4)', dmax < 1e-4,
              'max|Δ|=%.3e' % dmax)
    else:
        print('[SKIP] 无官方 TF 前向对账文件（%s）——本 artifacts 为 A-v1 重训权重，'
              '其架构与官方一致（见 a1_model_struct.json）' % chk)

    # ---- 1. 边界 ----
    p = COEnrichedMLProvider(args.artifacts, pchgs_exe=args.pchgs_exe,
                             pchgs_seconds=args.pchgs_seconds)
    v0, _ = scenario([], 5)
    sp0 = build_subproblem(v0, v0.vehicles[0])
    check('空池 → ()', p.order(sp0) == ())

    v1, ids1 = scenario([101], 3)
    sp1 = build_subproblem(v1, v1.vehicles[0])
    check('单客户池 → (101,)', p.order(sp1) == (101,), repr(p.order(sp1)))

    # ---- 2. 主池（真实风格非连续 id，n=60）----
    pool60 = sorted(int(x) for x in
                    (np.random.default_rng(1).choice(200, size=60, replace=False) + 100))
    v, _ = scenario(pool60, 20)
    sp = build_subproblem(v, v.vehicles[0])
    t0 = time.perf_counter()
    o1 = p.order(sp)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    o2 = p.order(sp)                  # 确定性 + 缓存
    t2 = time.perf_counter() - t0
    check('order 覆盖全池且无重复', sorted(o1) == sorted(pool60),
          'n=%d uniq=%d' % (len(o1), len(set(o1))))
    check('确定性（两次调用一致）', o1 == o2)

    edd = tuple(sorted(pool60, key=lambda c: float(sp.tw_end[sp.node_index(c)])))
    diff_pos = sum(1 for a, b in zip(o1, edd) if a != b)
    rho = spearman(list(o1), list(edd))
    check('偏好改变排序（与 EDD 不同）', diff_pos >= max(3, len(pool60) // 10),
          '不同位置 %d/%d, Spearman=%.3f' % (diff_pos, len(pool60), rho))

    # ---- 3. 未来信息不变性 ----
    v_h1, _ = scenario(pool60, 20, hidden_jitter=0.0)
    v_h2, _ = scenario(pool60, 20, hidden_jitter=3.0)
    sp_h1 = build_subproblem(v_h1, v_h1.vehicles[0])
    sp_h2 = build_subproblem(v_h2, v_h2.vehicles[0])
    p2 = COEnrichedMLProvider(args.artifacts, pchgs_exe=args.pchgs_exe,
                              pchgs_seconds=args.pchgs_seconds)
    oh1, oh2 = p2.order(sp_h1), p2.order(sp_h2)
    check('隐藏节点扰动不改排序（未来信息不变性）', oh1 == oh2,
          'hash_eq=%s' % (sp_h1.canonical_hash() == sp_h2.canonical_hash()))

    # ---- 4. 逐池耗时 ----
    for npool in (5, 25, 60):
        ids = pool60[:npool]
        vv, _ = scenario(ids, 5, seed=11)
        spp = build_subproblem(vv, vv.vehicles[0])
        pp = COEnrichedMLProvider(args.artifacts, pchgs_exe=args.pchgs_exe,
                                  pchgs_seconds=args.pchgs_seconds)
        t = time.perf_counter()
        oo = pp.order(spp)
        dt = time.perf_counter() - t
        print('    pool=%-3d order_len=%-3d  首次调用 %.3fs' % (npool, len(oo), dt))

    # ---- 5. 可选：PC-HGS（原方法 CO 层）----
    if args.pchgs_exe:
        vp, _ = scenario(pool60, 5, seed=13)
        spp = build_subproblem(vp, vp.vehicles[0])
        pc = COEnrichedMLProvider(args.artifacts, pchgs_exe=args.pchgs_exe,
                                  pchgs_seconds=args.pchgs_seconds)
        t = time.perf_counter()
        op = pc.order(spp)
        dt = time.perf_counter() - t
        print('    PC-HGS: n_calls=%d n_pchgs=%d n_err=%d %.3fs'
              % (pc.n_calls, pc.n_pchgs_calls, pc.n_pchgs_error, dt))
        check('PC-HGS 模式排序仍为合法排列', sorted(op) == sorted(pool60),
              'pchgs_calls=%d errors=%d' % (pc.n_pchgs_calls, pc.n_pchgs_error))

    print('\nstats:', p.stats())
    print('\nSMOKE_%s' % ('OK' if ok_all else 'FAILED'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
