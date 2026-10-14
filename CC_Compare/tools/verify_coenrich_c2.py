"""CO-enriched-ML 的 **C2 权重因果对照**（2026-10-14）。

补的缺口：CO-enriched-ML 是 13 个学习型外部方法里**唯一**在
`CC_Compare/results/a1_compare_table/participation/` 下没有 C2 证据的那个。

原因有二，且都已定位：
  ① `verify_provider_participation.py` 的 `probe_provider()` 会调 `provider._load()`，
     而 `COEnrichedMLProvider` **在 `__init__` 里就加载完**、没有 `_load`
     ⇒ 直接 `AttributeError`（已在 harness 侧改为「有 `_load` 才调」）；
  ② 即使跑通，通用 C2 用的是 `randomize_weights()`（**torch 模块**扰动），
     而 CO-enriched 是**纯 NumPy MLP**（9 个数组、550 个参数，见 `a1_weights.npz`）
     ⇒ 通用 C2 只能是 `NOT_SUPPORTED`。

对 NumPy 实现，**更强也更贴题的对照是「换权重文件」**：
  官方导出权重 `artifacts/official_weights.npz`  vs  A-v1 重训权重 `artifacts_a1/a1_weights.npz`
（台账 `模型参与证据.md` 已把这条写成 CO-enriched 的 C2；本脚本把它**机器化落盘**，
使它从「散文断言」升级为与其余 12 个 provider 同级的证据文件。）

纪律（与 `verify_rrnco_c2.py` 一致）：
- **只读**：不动 provider / 上游 CO-enriched-ML 代码；
- 单线程（`OMP/MKL/OPENBLAS/NUMEXPR=1`）；
- 两个 provider **各自独立构造**（模型在 `__init__` 定死，不能就地改权重）；
- 每个 provider 都做**清缓存两次**的 B3 确定性检查。

用法（1003，cc_compare env；本地无官方工具链，必须远端跑）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/tools/verify_coenrich_c2.py \
        --dir CC_Compare/CO-enriched-ML/a1_accept_v1 \
        --outdir CC_Compare/results/a1_compare_table/participation
"""
import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import verify_provider_participation as vp  # noqa: E402

ROOT = vp._ROOT


def build(dir_abs, weights_subdir, cls_name='COEnrichedMLProvider',
          module='coenrich_provider'):
    """按 registry 的同一条路径构造 provider（provider 目录插 sys.path → import → getattr）。"""
    import importlib
    if dir_abs not in sys.path:
        sys.path.insert(0, dir_abs)
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    ckpt = os.path.join(dir_abs, weights_subdir)
    return cls(ckpt, device='cpu')


def probe(provider, sp, sp2, label):
    """单个权重档的 C1 + B3 + B6。"""
    out = {'label': label}
    o, t, err = vp.timed_order(provider, sp)
    out['order'] = list(o) if o else None
    out['seconds'] = t
    out['error'] = err
    out['covered'] = bool(o) and sorted(o) == sorted(sp.pool_customer_ids)
    o_a, _, e1 = vp.timed_order(provider, sp)
    o_b, _, e2 = vp.timed_order(provider, sp)
    out['B3_determinism'] = {'same': (o_a is not None and o_a == o_b),
                             'errors': [e1, e2]}
    o_p, _, e3 = vp.timed_order(provider, sp2)
    out['B6_future_perturbation'] = {
        'ordering_same': (o_p is not None and o is not None and o_p == o),
        'error': e3,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='CC_Compare/CO-enriched-ML/a1_accept_v1')
    ap.add_argument('--official-subdir', default='artifacts')
    ap.add_argument('--retrained-subdir', default='artifacts_a1')
    ap.add_argument('--pool', type=int, default=40)
    ap.add_argument('--hidden', type=int, default=60)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    d = os.path.normpath(os.path.join(ROOT, *args.dir.split('/'))) \
        if not os.path.isabs(args.dir) else args.dir
    if not os.path.isdir(d):
        raise SystemExit('provider dir missing: %s' % d)

    inst = vp.build_instance(n_pool=args.pool, n_hidden=args.hidden)
    sp = vp.make_sp(inst['node_ids_all'], inst['coords'], inst['demands'],
                    inst['tw_end'], inst['pool'], capacity=args.capacity,
                    depot_tw_end=22.0)
    # B6：只扰动不可见（非 pool）节点 → 子问题哈希必须不变
    import numpy as np
    coords2 = inst['coords'].copy()
    dem2 = inst['demands'].copy()
    tw2 = inst['tw_end'].copy()
    hpos = [inst['node_ids_all'].index(h) for h in inst['hidden']]
    rng = np.random.default_rng(1234)
    coords2[hpos] = rng.uniform(0.0, 1.0, (len(hpos), 2))
    dem2[hpos] = rng.uniform(1.0, 3.0, len(hpos))
    tw2[hpos] = 6.0 + rng.uniform(0.0, 8.0, len(hpos))
    sp2 = vp.make_sp(inst['node_ids_all'], coords2, dem2, tw2, inst['pool'],
                     capacity=args.capacity, depot_tw_end=22.0)

    res = {
        'provider': 'coenrich',
        'dir': args.dir,
        'device': 'cpu',
        'n_pool': len(inst['pool']),
        'n_hidden': len(inst['hidden']),
        'subproblem_fields': list(vp.SP_FIELDS),
        'subproblem_hash': sp.canonical_hash(),
        'subproblem_hash_perturbed': sp2.canonical_hash(),
        'C2_control': 'weight-file swap: official export vs A-v1 retrained '
                      '(NOT torch randomize — this provider is pure NumPy)',
    }

    t0 = time.perf_counter()
    p_off = build(d, args.official_subdir)
    res['load_seconds_official'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_a1 = build(d, args.retrained_subdir)
    res['load_seconds_retrained'] = time.perf_counter() - t0

    res['official'] = probe(p_off, sp, sp2, 'official_weights')
    res['retrained'] = probe(p_a1, sp, sp2, 'a1_weights')

    o_real = tuple(res['retrained']['order'] or ())
    o_rand = tuple(res['official']['order'] or ())
    o_edd = vp.edd_order(sp)
    res['C3_edd_order'] = list(o_edd)
    if o_real and o_rand:
        res['metrics'] = {
            'order_changed_by_weights': bool(o_real != o_rand),
            'set_equal': sorted(o_real) == sorted(o_rand),
            'diff_positions_real_vs_rand': vp.diff_positions(o_real, o_rand),
            'spearman_real_rand': vp.spearman(o_real, o_rand),
            'first12_overlap_real_rand': vp.first_k_overlap(o_real, o_rand, 12),
            'diff_positions_real_vs_edd': vp.diff_positions(o_real, o_edd),
            'spearman_real_edd': vp.spearman(o_real, o_edd),
        }
    res['calls'] = {k: getattr(p_a1, k, None) for k in
                    ('n_calls', 'n_pchgs_calls', 'n_pchgs_error',
                     'n_edd_fallback', 'n_bad_dim')}

    os.makedirs(args.outdir, exist_ok=True)
    jp = os.path.join(args.outdir, 'coenrich_c2_swap.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    m = res.get('metrics') or {}
    print('order_changed_by_weights = %s' % m.get('order_changed_by_weights'))
    print('diff_positions(real vs official) = %s / %d'
          % (m.get('diff_positions_real_vs_rand'), res['n_pool']))
    print('spearman(real, official) = %s' % m.get('spearman_real_rand'))
    print('first12_overlap = %s' % m.get('first12_overlap_real_rand'))
    print('set_equal = %s' % m.get('set_equal'))
    print('B3 determinism: official=%s retrained=%s'
          % (res['official']['B3_determinism']['same'],
             res['retrained']['B3_determinism']['same']))
    print('B6 future-shift ordering_same: official=%s retrained=%s'
          % (res['official']['B6_future_perturbation']['ordering_same'],
             res['retrained']['B6_future_perturbation']['ordering_same']))
    print('wrote %s' % jp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
