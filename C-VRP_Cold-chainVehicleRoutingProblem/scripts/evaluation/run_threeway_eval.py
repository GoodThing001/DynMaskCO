"""M1 三组同口径离线评价：feature_local / encoder_real(H-only) / m1(H+Z)。

对每组 checkpoint，在同一批 CAL/DEV context 上打分，报告：
  - 实例等权净收益（τ=0 与 CAL 选出的 τ*）、接受覆盖率、益/平/害计数、服务失败/标签缺失；
  - regret（相对已标注候选最优单步选择 g*）；
  - Spearman / MAE（辅助，离线排序口径）。

标签张量（legal/supervision/delta/service_ok/is_pseudo/instance_ids）是数据集层，三组共享；
仅打分输入随模型不同。固定 CAL 选择规则复用 run_margin_scan（实例等权 + 排除无效配置）。

用法（服务器）：
  # 组① feature-only-local（去编号）
  python scripts/evaluation/run_threeway_eval.py --model feature_local \
    --ckpt results/m0_scale/probe_local_n64_deindex_s42/local.ckpt --deindex \
    --cal-teacher-dir results/m0_scale/cal --cal-data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
    --dev-teacher-dir results/m0_scale/dev_check --dev-data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --out results/m0_scale/threeway/feature_local_s42
  # 组② H-only（冻结 encoder H + utility head，去编号）
  python scripts/evaluation/run_threeway_eval.py --model encoder_real \
    --ckpt results/m0_scale/probe_enc_real_deindex_s42/probe.ckpt --deindex \
    --encoder-ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
    --cal-teacher-dir ... --dev-teacher-dir ... --out ...
  # 组③ 完整 M1（H + masked decoder Z + utility head，去编号）
  python scripts/evaluation/run_threeway_eval.py --model m1 \
    --ckpt results/m0_scale/m1_deindex_s42/m1.ckpt --deindex \
    --encoder-ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
    --cal-teacher-dir ... --dev-teacher-dir ... --out ...
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert',
          'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax.numpy as jnp
from flax import nnx

from coldchain_teacher_dataset import load_teacher_dataset
from run_margin_scan import (_scan, select_tau_cal, _per_instance_json, TAU_GRID,
                             CAL_SELECTION_RULE)


def build_label_tensors(ds):
    """数据集层标签张量（三组共享）：[C, M] padded。"""
    contexts = ds.contexts
    cands_by = ds.candidates_by_context
    C = len(contexts)
    M = max(len(cands_by[c['context_id']]) for c in contexts)
    legal = np.zeros((C, M), bool)
    supervision = np.zeros((C, M), bool)
    service_ok = np.zeros((C, M), bool)
    is_pseudo = np.zeros((C, M), bool)
    selected = np.zeros((C, M), bool)
    delta = np.zeros((C, M), np.float32)
    instance_ids = np.zeros(C, np.int64)
    for i, ctx in enumerate(contexts):
        instance_ids[i] = int(ctx['inst_idx'])
        cands = cands_by[ctx['context_id']]
        lm = ds.legal_mask(cands)
        sm = ds.supervision_mask(cands)
        for j, c in enumerate(cands):
            legal[i, j] = bool(lm[j])
            supervision[i, j] = bool(sm[j])
            service_ok[i, j] = bool(c.get('service_ok', False))
            is_pseudo[i, j] = bool(c.get('is_pseudo', False))
            selected[i, j] = bool(c.get('selected_by_teacher', False))
            d = c.get('delta_vs_keep')
            delta[i, j] = float(d) if d is not None else 0.0
    return {'legal': legal, 'supervision': supervision, 'service_ok': service_ok,
            'is_pseudo': is_pseudo, 'selected': selected, 'delta': delta,
            'instance_ids': instance_ids}


def _avg_rank(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for u in np.flatnonzero(counts > 1):
        m = inv == u
        ranks[m] = ranks[m].mean()
    return ranks


def _spearman(a, b):
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def compute_spearman_mae(scores, t, s_scale):
    sup = np.asarray(t['supervision'])
    delta = np.asarray(t['delta'])
    y = np.where(sup, -delta / s_scale, 0.0)
    C = scores.shape[0]
    mae = rmse = float('nan')
    if sup.sum() > 0:
        err = scores[sup] - y[sup]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
    rho = []
    for i in range(C):
        m = sup[i]
        if m.sum() >= 2 and np.ptp(y[i][m]) > 1e-12:
            rho.append(_spearman(scores[i][m], y[i][m]))
    return {'utility_mae': mae, 'utility_rmse': rmse,
            'rank_corr_spearman': (float(np.mean(rho)) if rho else float('nan')),
            'n_ranked_contexts': len(rho)}


# --------------------------------------------------------------------------- #
# 模型打分
# --------------------------------------------------------------------------- #
def _score_feature_local(ckpt_path, ds, data_npz, capacity, deindex, seed):
    import pickle
    from train_feature_only_local import LocalHead, IN_DIM, build_dataset
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    head = LocalHead(IN_DIM, rngs=seed)
    graphdef, _ = nnx.split(head)
    params = ck['params']
    t = build_dataset(ds, data_npz, capacity, deindex=deindex)
    h = nnx.merge(graphdef, params)
    out = []
    for i in range(t['context_vecs'].shape[0]):
        s = h(jnp.asarray(t['context_vecs'][i:i+1]), jnp.asarray(t['action_vals'][i:i+1]),
              jnp.asarray(t['action_valid'][i:i+1]), jnp.asarray(t['local_vals'][i:i+1]),
              jnp.asarray(t['local_valid'][i:i+1]))
        out.append(np.asarray(s)[0])
    M = max(x.shape[0] for x in out)
    scores = np.stack([np.pad(x, (0, M - x.shape[0])) for x in out]).astype(np.float32)
    return t, scores, float(ck['s'])


def _score_encoder_real(ckpt_path, encoder_ckpt, ds, data_npz, capacity, deindex, seed):
    from train_coldchain_utility_probe import (build_dataset, build_encoder_contexts,
                                               build_head, load_checkpoint)
    ck = load_checkpoint(ckpt_path)
    config = ck['config']
    params = ck['params']
    t = build_dataset(ds, data_npz, deindex=deindex)
    from train_fleet_head import load_base_model
    encoder = load_base_model(encoder_ckpt)
    enc = config.get('encoder') or {}
    tw_max = enc.get('tw_max', float(data_npz['tw_end'][:, 0].max()))
    ctx, _H = build_encoder_contexts(ds, data_npz, encoder, capacity=enc.get('capacity', capacity),
                                     tw_max=tw_max)
    t['context_vecs'] = ctx
    graphdef = build_head(config)
    h = nnx.merge(graphdef, params)
    out = []
    for i in range(ctx.shape[0]):
        s = h(jnp.asarray(ctx[i:i+1]), jnp.asarray(t['action_vals'][i:i+1]),
              jnp.asarray(t['action_valid'][i:i+1]))
        out.append(np.asarray(s)[0])
    M = max(x.shape[0] for x in out)
    scores = np.stack([np.pad(x, (0, M - x.shape[0])) for x in out]).astype(np.float32)
    return t, scores, float(config['s'])


def _score_m1(ckpt_path, encoder_ckpt, ds, data_npz, capacity, deindex, seed, tw_max):
    from train_dynmaskco_cc import build_training_tensors
    from train_fleet_head import load_base_model
    from dynmaskco_cc import load_m1_model, M1Scorer
    base_model = load_base_model(encoder_ckpt)
    mt = build_training_tensors(ds, data_npz, base_model, capacity, tw_max, deindex=deindex,
                                keep_all=True)
    model = load_m1_model(ckpt_path, base_model)
    scorer = M1Scorer(model)
    n = mt['n_ctx']
    # 全局 padding 到固定形状 → 单次 JIT（避免 per-context 重编译）。
    Nv_max = max(mt['Hv'][i].shape[0] for i in range(n))
    M_max = max(mt['endpoints'][i].shape[0] for i in range(n))
    d = mt['Hv'][0].shape[1]
    E = mt['explicit'][0].shape[1]
    Hv = np.zeros((n, Nv_max, d), np.float32)
    A_in = np.zeros((n, Nv_max, Nv_max), np.float32)
    endpoints = np.zeros((n, M_max, 4), np.int32)
    valid = np.zeros((n, M_max, 4), bool)
    explicit = np.zeros((n, M_max, E), np.float32)
    ts = np.zeros(n, np.float32)
    for i in range(n):
        ni = mt['Hv'][i].shape[0]
        mi = mt['endpoints'][i].shape[0]
        Hv[i, :ni] = mt['Hv'][i]
        A_in[i, :ni, :ni] = mt['A_in'][i]
        endpoints[i, :mi] = mt['endpoints'][i]
        valid[i, :mi] = mt['valid'][i]
        explicit[i, :mi] = mt['explicit'][i]
        ts[i] = mt['timestep'][i]
    scores = scorer.score(jnp.asarray(Hv), jnp.asarray(ts), jnp.asarray(A_in),
                          jnp.asarray(endpoints).astype(jnp.int32), jnp.asarray(valid),
                          jnp.asarray(explicit))
    scores = np.asarray(scores).astype(np.float32)   # [n, M_max]
    t = build_label_tensors(ds)
    assert n == t['instance_ids'].shape[0], f"m1 n_ctx={n} != label C={t['instance_ids'].shape[0]}"
    import pickle
    with open(ckpt_path, 'rb') as f:
        s_scale = float(pickle.load(f)['s'])
    return t, scores, s_scale


def _score_compact(ckpt_path, encoder_ckpt, ds, data_npz, capacity, deindex, seed, tw_max):
    import pickle
    from train_m1_compact import build_explicit_and_labels, build_encoder_inputs, EXPLICIT_DIM
    from dynmaskco_cc_compact import gather_endpoints, CompactDecoder, ScoringMLP
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    use_H = ck['use_H']
    use_Z = ck['use_Z']
    d = ck['d_compact']
    params = ck['params']
    explicit, t = build_explicit_and_labels(ds, data_npz, capacity, deindex)
    C = explicit.shape[0]

    proj_gd = dec_gd = None
    if use_H:
        proj_gd, _ = nnx.split(nnx.Linear(256, d, use_bias=False, rngs=nnx.Rngs(0)))
    if use_Z:
        dec_gd, _ = nnx.split(CompactDecoder(d, rngs=0))
    head_in = EXPLICIT_DIM + (4 * d if use_H else 0) + (4 * d if use_Z else 0)
    head_gd, _ = nnx.split(ScoringMLP(head_in, (128, 64), rngs=0))

    Hv = A_in = nv = e = v = ts = None
    if use_H:
        from train_fleet_head import load_base_model
        base_model = load_base_model(encoder_ckpt)
        enc = build_encoder_inputs(ds, data_npz, base_model, capacity, tw_max, deindex)
        Hv, A_in, nv, e, v, ts = enc['Hv'], enc['A_in'], enc['node_valid'], enc['endpoints'], \
                                 enc['valid'], enc['ts']
        assert Hv.shape[0] == C

    def fwd(ex, Hv_b, A_in_b, nv_b, e_b, v_b, ts_b):
        parts = [ex]
        h = None
        if use_H:
            h = nnx.merge(proj_gd, params['proj'])(Hv_b)
            parts.append(gather_endpoints(h, e_b, v_b))
        if use_Z:
            z = nnx.merge(dec_gd, params['dec'])(h, ts_b, A_in_b, nv_b)
            parts.append(gather_endpoints(z, e_b, v_b))
        x = jnp.concatenate(parts, axis=-1)
        return nnx.merge(head_gd, params['head'])(x)

    if use_H:
        scores = fwd(jnp.asarray(explicit), jnp.asarray(Hv), jnp.asarray(A_in),
                     jnp.asarray(nv), jnp.asarray(e).astype(jnp.int32), jnp.asarray(v),
                     jnp.asarray(ts))
    else:
        scores = fwd(jnp.asarray(explicit), None, None, None, None, None, None)
    scores = np.asarray(scores).astype(np.float32)
    return t, scores, float(ck['s'])


def _score_state_2x2(ckpt_path, ds, data_npz, capacity, deindex, tw_max):
    import pickle
    from dynmaskco_cc_compact import ScoringMLP
    from train_m1_compact import build_explicit_and_labels
    from train_state_2x2 import build_x_state
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    in_dim = ck['in_dim']
    use_state = ck['use_state']
    params = ck['params']
    explicit, t = build_explicit_and_labels(ds, data_npz, capacity, deindex)
    X = explicit
    if use_state:
        xs = build_x_state(ds, data_npz, capacity, tw_max)
        X = np.concatenate([explicit, xs], axis=-1)
    head_gd, _ = nnx.split(ScoringMLP(in_dim, (128, 64), rngs=0))
    scores = np.asarray(nnx.merge(head_gd, params)(jnp.asarray(X))).astype(np.float32)
    return t, scores, float(ck['s'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['feature_local', 'encoder_real', 'm1', 'compact', 'state_2x2'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--encoder-ckpt', default=None)
    ap.add_argument('--cal-teacher-dir', required=True)
    ap.add_argument('--cal-data', required=True)
    ap.add_argument('--dev-teacher-dir', required=True)
    ap.add_argument('--dev-data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--tw-max', type=float, default=None)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.model in ('encoder_real', 'm1') and args.encoder_ckpt is None:
        raise SystemExit(f'--model={args.model} 需要 --encoder-ckpt')

    result = {}
    for name, td, dpath in [('cal', args.cal_teacher_dir, args.cal_data),
                            ('dev', args.dev_teacher_dir, args.dev_data)]:
        ds = load_teacher_dataset(td, data_path=dpath)
        data_npz = dict(np.load(dpath))
        tw_max = args.tw_max if args.tw_max is not None else float(data_npz['tw_end'][:, 0].max())
        if args.model == 'feature_local':
            t, scores, s = _score_feature_local(args.ckpt, ds, data_npz, args.capacity,
                                                args.deindex, args.seed)
        elif args.model == 'encoder_real':
            t, scores, s = _score_encoder_real(args.ckpt, args.encoder_ckpt, ds, data_npz,
                                               args.capacity, args.deindex, args.seed)
        elif args.model == 'm1':
            t, scores, s = _score_m1(args.ckpt, args.encoder_ckpt, ds, data_npz, args.capacity,
                                     args.deindex, args.seed, tw_max)
        elif args.model == 'compact':
            t, scores, s = _score_compact(args.ckpt, args.encoder_ckpt, ds, data_npz,
                                          args.capacity, args.deindex, args.seed, tw_max)
        else:
            t, scores, s = _score_state_2x2(args.ckpt, ds, data_npz, args.capacity,
                                            args.deindex, tw_max)
        rows, per_inst = _scan(t, scores, s)
        sm = compute_spearman_mae(scores, t, s)
        result[name] = {'rows': rows, 'per_instance': per_inst, 'spearman_mae': sm,
                        's_scale': s, 'n_contexts': int(t['instance_ids'].shape[0])}

    cal_rows = result['cal']['rows']
    tau_star = select_tau_cal(cal_rows)
    keep_baseline = (tau_star is not None and float(tau_star) == float('inf'))

    report = {
        'model': args.model, 'ckpt': args.ckpt, 'deindex': args.deindex,
        'cal_selection_rule': CAL_SELECTION_RULE,
        'cal_selected_tau': tau_star, 'cal_selected_tau_is_keep_baseline': keep_baseline,
        'cal': {'rows': cal_rows, 'per_instance': _per_instance_json(result['cal']['per_instance']),
                'spearman_mae': result['cal']['spearman_mae'], 's_scale': result['cal']['s_scale'],
                'n_contexts': result['cal']['n_contexts']},
        'dev': {'rows': result['dev']['rows'],
                'per_instance': _per_instance_json(result['dev']['per_instance']),
                'spearman_mae': result['dev']['spearman_mae'], 's_scale': result['dev']['s_scale'],
                'n_contexts': result['dev']['n_contexts']},
    }
    with open(os.path.join(args.out, 'threeway.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # 打印摘要
    for name in ('cal', 'dev'):
        r = result[name]
        sm = r['spearman_mae']
        print(f"\n[{name}] {args.model} s={r['s_scale']:.4f} contexts={r['n_contexts']} "
              f"MAE={sm['utility_mae']:.4f} spearman={sm['rank_corr_spearman']:.4f} "
              f"(n={sm['n_ranked_contexts']})")
        print(f"  {'tau':>7s} | {'gain(inst)':>10s} {'cov':>6s} {'ben/neu/harm':>14s} "
              f"{'srv/label':>9s} {'regret':>8s}")
        for tau in TAU_GRID:
            row = r['rows'][str(tau)]
            print(f"  {tau:7.4g} | {row['net_gain_instance_mean']:10.4f} "
                  f"{row['accept_coverage']:6.3f} "
                  f"{row['n_accept_beneficial']}/{row['n_accept_neutral']}/{row['n_accept_harmful']:>3d} "
                  f"  {row['n_accept_service_fail']}/{row['n_accept_label_missing']:>3d} "
                  f"{_fmt(row['regret_instance_mean']):>8s}")
    d0 = result['dev']['rows']['0.0']
    print(f"\n  DEV@tau0: gain(inst)={d0['net_gain_instance_mean']:+.4f} "
          f"cov={d0['accept_coverage']:.3f} harm={_fmt(d0['harmful_accept_ratio'])} "
          f"regret={_fmt(d0['regret_instance_mean'])}")
    print(f"  CAL tau*={tau_star}{' (keep baseline)' if keep_baseline else ''}")
    print(f"saved: {args.out}/threeway.json")


def _fmt(x):
    return 'nan' if x is None else f'{x:.4f}'


if __name__ == '__main__':
    main()
