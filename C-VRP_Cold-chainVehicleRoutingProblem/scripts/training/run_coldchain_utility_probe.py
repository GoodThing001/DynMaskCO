"""M0 utility probe 独立评价入口（工作包 B）。

加载 checkpoint（params + 固定尺度 s + config）→ 重建 head → 读 teacher 数据集 + NPZ →
构建 feature-only 特征 → 打分 → 输出评价指标。

区分两类口径：
  - 离线排序（有效标签集合内）：utility_error（MAE/RMSE）与 rank_corr（每 context 的 Spearman，
    只在 supervision 候选 ≥2 的 context 上）。
  - 在线选择（全部 legal 候选上）：top1_agreement（与 teacher 选择一致率）、keep_decision_error
    （KEEP/DEFER 决策是否与 teacher 相反）、failure_selected_rate（选中 service_ok=False 候选
    的比例）、tie_fraction（top1 与 top2 分数差 < eps 的 context 占比）。

用法（服务器）：
    python scripts/training/run_coldchain_utility_probe.py \
        --ckpt <dir>/probe.ckpt --teacher-dir <teacher_dataset_dir> --data <npz> --out <dir>
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

from coldchain_teacher_dataset import load_teacher_dataset
from train_coldchain_utility_probe import (build_dataset, drop_unsupervised, apply_scale,
                                           build_head, score_batch, load_checkpoint,
                                           build_encoder_contexts)

TIE_EPS = 1e-6


def _avg_rank(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
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


def compute_metrics(scores, tensors):
    """scores [C, M] → 指标 dict。"""
    C, M = scores.shape
    legal = tensors['legal']
    sup = tensors['supervision']
    y = tensors['labels']
    service_ok = tensors['service_ok']
    is_pseudo = tensors['is_pseudo']
    selected = tensors['selected']

    # 离线：效用误差（supervision 集合）
    sm = sup
    if sm.sum() > 0:
        err = scores[sm] - y[sm]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
    else:
        mae = rmse = float('nan')

    # 离线：排序相关（每 context 内 supervision ≥2 且标签非全平局，才有有效排序）
    rho = []
    for i in range(C):
        m = sup[i]
        if m.sum() >= 2 and np.ptp(y[i][m]) > 1e-12:
            rho.append(_spearman(scores[i][m], y[i][m]))
    rank_corr = float(np.mean(rho)) if rho else float('nan')

    # 在线：legal 上的实际选择
    n_ctx = 0
    n_sel = 0
    n_agree = 0
    n_keep_err = 0
    n_fail = 0
    n_tie = 0
    n_tie_denom = 0
    for i in range(C):
        lm = legal[i]
        if lm.sum() == 0:
            continue
        n_ctx += 1
        idx = np.flatnonzero(lm)
        s_i = scores[i][idx]
        order = np.argsort(-s_i)
        top = idx[order[0]]
        sel_idx = np.flatnonzero(selected[i])
        if sel_idx.size > 0:
            n_sel += 1
            if sel_idx[0] == top:
                n_agree += 1
            teach_keep = bool(is_pseudo[i][sel_idx[0]])
            model_keep = bool(is_pseudo[i][top])
            if teach_keep != model_keep:
                n_keep_err += 1
        if not service_ok[i][top]:
            n_fail += 1
        if len(order) >= 2:
            n_tie_denom += 1
            if abs(s_i[0] - s_i[1]) < TIE_EPS:
                n_tie += 1

    return {
        'n_contexts': C,
        'n_online_contexts': n_ctx,
        'offline': {
            'n_supervision': int(sm.sum()),
            'utility_mae': mae,
            'utility_rmse': rmse,
            'rank_corr_spearman': rank_corr,
            'n_ranked_contexts': len(rho),
        },
        'online': {
            'n_with_teacher_selection': n_sel,
            'top1_agreement': (n_agree / n_sel) if n_sel else float('nan'),
            'keep_decision_error': (n_keep_err / n_sel) if n_sel else float('nan'),
            'failure_selected_rate': (n_fail / n_ctx) if n_ctx else float('nan'),
            'tie_fraction': (n_tie / n_tie_denom) if n_tie_denom else float('nan'),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt)
    config = ckpt['config']
    params = ckpt['params']
    graphdef = build_head(config)

    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    tensors = build_dataset(ds, data_npz)
    tensors, _keep = drop_unsupervised(tensors)
    tensors['labels'] = apply_scale(tensors, config['s'])   # 用 checkpoint 固定的 s

    # encoder 模式：按 config 记录的冻结 encoder 重建 context 向量（与训练同一来源）。
    mode = config.get('mode', 'feature_only')
    if mode in ('encoder_real', 'encoder_shuffle'):
        from train_fleet_head import load_base_model
        enc = config.get('encoder') or {}
        encoder = load_base_model(enc['ckpt'])
        ctx_vecs, _H = build_encoder_contexts(
            ds, data_npz, encoder, capacity=enc.get('capacity', 50.0),
            tw_max=enc.get('tw_max', 24.0), shuffle=(mode == 'encoder_shuffle'),
            shuffle_seed=enc.get('shuffle_seed', 0))
        tensors['context_vecs'] = ctx_vecs

    scores = score_batch(graphdef, params, tensors['context_vecs'],
                         tensors['action_vals'], tensors['action_valid'])
    metrics = compute_metrics(scores, tensors)

    report = {
        'ckpt': args.ckpt,
        'mode': config.get('mode', 'feature_only'),
        'scale_s': config.get('s'),
        'metrics': metrics,
    }
    with open(os.path.join(args.out, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("=== M0 utility probe eval ===")
    print(f"  mode={report['mode']}  s={report['scale_s']}")
    print(f"  contexts={metrics['n_contexts']}  online={metrics['n_online_contexts']}")
    print(f"  offline: MAE={metrics['offline']['utility_mae']:.4f} "
          f"RMSE={metrics['offline']['utility_rmse']:.4f} "
          f"spearman={metrics['offline']['rank_corr_spearman']:.4f} "
          f"(n={metrics['offline']['n_ranked_contexts']})")
    print(f"  online:  top1_agree={metrics['online']['top1_agreement']:.3f} "
          f"keep_err={metrics['online']['keep_decision_error']:.3f} "
          f"fail_rate={metrics['online']['failure_selected_rate']:.3f} "
          f"tie={metrics['online']['tie_fraction']:.3f}")
    print(f"  saved: {args.out}/eval_report.json")


if __name__ == '__main__':
    main()
