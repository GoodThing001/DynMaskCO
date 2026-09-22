"""分层复评（方案 2 扩大版）：不训练，对已有 checkpoint 重算分层指标。

对 feature-only(-earliest/spread) 与 feature-only-local 分别统计：
  TRAIN-fit（真正参与梯度更新的实例）/ internal-dev（TRAIN 文件内留出）/ external-DEV；
  每层再按 (instance_id, event_id) 分初始(event_id==0) vs 非初始，非初始按 clock 三分（早/中/晚）。

每行报告：实例数/context 数/有效排序 context 数/监督候选数/KEEP-DEFER 占比/
  context 内 Spearman/top1 一致率/永远 KEEP-DEFER 的 top1/模型预测接受数(有益·有害)。

Spearman 只在 supervision≥2 且标签非全平局的 context 上算，保留 n_ranked。

用法（服务器）：
    python scripts/evaluation/run_stratified_eval.py \
        --ckpt results/m0dev/probe_fo_spread/probe.ckpt \
        --teacher-dir results/m0dev/train_spread --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
        --dev-teacher-dir results/m0dev/dev_spread --dev-data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
        --out results/m0dev/strat_fo_spread
    # feature-only-local 加 --local
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

import jax.numpy as jnp
from flax import nnx

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        extract_candidate_local_features,
                                        ACTION_FEAT_DIM, LOCAL_VAL_DIM, LOCAL_VALID_DIM)
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM, UtilityHead
from train_coldchain_utility_probe import load_checkpoint, build_head, split_train_dev


def _ctx_vec(data_npz, inst_idx, snapshot):
    feats = extract_context_features(data_npz, inst_idx, snapshot)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def build_tensors(ds, data_npz, local, capacity=50.0):
    contexts = ds.contexts
    cands_by = ds.candidates_by_context
    C = len(contexts)
    M = max(len(cands_by[c['context_id']]) for c in contexts)
    context_vecs = np.zeros((C, FEATURE_CONTEXT_DIM), np.float32)
    action_vals = np.zeros((C, M, ACTION_FEAT_DIM), np.float32)
    action_valid = np.zeros((C, M, ACTION_FEAT_DIM), np.float32)
    local_vals = np.zeros((C, M, LOCAL_VAL_DIM), np.float32)
    local_valid = np.zeros((C, M, LOCAL_VALID_DIM), np.float32)
    cand_valid = np.zeros((C, M), bool)
    legal = np.zeros((C, M), bool)
    supervision = np.zeros((C, M), bool)
    service_ok = np.zeros((C, M), bool)
    is_pseudo = np.zeros((C, M), bool)
    selected = np.zeros((C, M), bool)
    delta = np.zeros((C, M), np.float32)
    instance_ids = np.zeros(C, np.int64)
    event_ids = np.zeros(C, np.int64)
    clock = np.zeros(C, np.float32)

    for i, ctx in enumerate(contexts):
        ctx_id = ctx['context_id']
        instance_ids[i] = int(ctx['inst_idx'])
        event_ids[i] = int(ctx['event_id'])
        clock[i] = float(ctx['clock'])
        context_vecs[i] = _ctx_vec(data_npz, int(ctx['inst_idx']), ctx['snapshot'])
        cands = cands_by[ctx_id]
        for j, c in enumerate(cands):
            av, avd = extract_action_features(c)
            action_vals[i, j] = av
            action_valid[i, j] = avd
            if local:
                lv, lvd = extract_candidate_local_features(
                    data_npz, int(ctx['inst_idx']), c.get('action'),
                    c.get('certificate', {}).get('incremental_distance'), capacity)
                local_vals[i, j] = lv
                local_valid[i, j] = lvd
            cand_valid[i, j] = True
            legal[i, j] = bool(ds.legal_mask([c])[0])
            supervision[i, j] = bool(ds.supervision_mask([c])[0])
            service_ok[i, j] = bool(c.get('service_ok', False))
            is_pseudo[i, j] = bool(c.get('is_pseudo', False))
            selected[i, j] = bool(c.get('selected_by_teacher', False))
            d = c.get('delta_vs_keep')
            delta[i, j] = float(d) if d is not None else 0.0

    return {'context_vecs': context_vecs, 'action_vals': action_vals,
            'action_valid': action_valid, 'local_vals': local_vals,
            'local_valid': local_valid, 'cand_valid': cand_valid, 'legal': legal,
            'supervision': supervision, 'service_ok': service_ok, 'is_pseudo': is_pseudo,
            'selected': selected, 'delta': delta, 'instance_ids': instance_ids,
            'event_ids': event_ids, 'clock': clock}


def _score(graphdef, params, local, ctx, av, avd, lv, lvd):
    h = nnx.merge(graphdef, params)
    if local:
        return h(jnp.asarray(ctx), jnp.asarray(av), jnp.asarray(avd),
                 jnp.asarray(lv), jnp.asarray(lvd))
    return h(jnp.asarray(ctx), jnp.asarray(av), jnp.asarray(avd))


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


def _stratum_metrics(t, scores, s_scale, idxs):
    if len(idxs) == 0:
        return None
    legal = t['legal'][idxs]
    sup = t['supervision'][idxs]
    selected = t['selected'][idxs]
    is_pseudo = t['is_pseudo'][idxs]
    delta = t['delta'][idxs]
    s = scores[idxs]

    n_instances = len(set(int(t['instance_ids'][i]) for i in idxs))
    n_contexts = len(idxs)
    n_supervision = int(sup.sum())

    # 有效排序 context（supervision≥2 且标签非全平局）
    ranked = []
    for r, i in enumerate(idxs):
        m = sup[r]
        if m.sum() >= 2 and np.ptp(delta[r][m]) > 1e-12:
            ranked.append(_spearman(s[r][m], -delta[r][m] / s_scale))
    n_ranked = len(ranked)
    spearman = float(np.mean(ranked)) if ranked else float('nan')

    # top1（在线合法候选）：模型 vs teacher vs 永远 KEEP-DEFER
    top1_agree = 0
    keep_agree = 0
    accept = 0
    beneficial = 0
    harmful = 0
    for r, i in enumerate(idxs):
        lm = legal[r]
        if lm.sum() == 0:
            continue
        idx = np.flatnonzero(lm)
        s_i = s[r][idx]
        top = idx[int(np.argmax(s_i))]
        sel_idx = np.flatnonzero(selected[r])
        has_teacher = sel_idx.size > 0
        if has_teacher:
            # 模型 top1 是否 = teacher 选择
            if sel_idx[0] == top:
                top1_agree += 1
            # 永远 KEEP-DEFER：teacher 选的是 pseudo 则算对
            if is_pseudo[r][sel_idx[0]]:
                keep_agree += 1
        # 模型预测接受：top1 非 pseudo
        if not is_pseudo[r][top]:
            accept += 1
            d = delta[r][top]
            if d < -1e-9:
                beneficial += 1
            elif d > 1e-9:
                harmful += 1

    # KEEP-DEFER 占比（teacher 选择中 pseudo 的比例）
    n_teacher = sum(1 for r in range(len(idxs)) if selected[r].sum() > 0)
    keep_def_ratio = (sum(1 for r in range(len(idxs))
                          if selected[r].sum() > 0 and is_pseudo[r][selected[r]].any())
                      / n_teacher) if n_teacher else float('nan')

    return {
        'n_instances': n_instances, 'n_contexts': n_contexts,
        'n_supervision': n_supervision, 'n_ranked': n_ranked,
        'spearman': spearman,
        'top1_agree': (top1_agree / n_teacher) if n_teacher else float('nan'),
        'keep_baseline_top1': (keep_agree / n_teacher) if n_teacher else float('nan'),
        'keep_defer_ratio': keep_def_ratio,
        'model_accept': accept, 'accept_beneficial': beneficial, 'accept_harmful': harmful,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--dev-teacher-dir', required=True)
    ap.add_argument('--dev-data', required=True)
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.local:
        from train_feature_only_local import LocalHead, IN_DIM
        import pickle
        with open(args.ckpt, 'rb') as f:
            ck = pickle.load(f)
        head = LocalHead(IN_DIM, rngs=args.seed)
        graphdef, _ = nnx.split(head)
        params = ck['params']
        s_scale = ck['s']
    else:
        ck = load_checkpoint(args.ckpt)
        graphdef = build_head(ck['config'])
        params = ck['params']
        s_scale = ck['config']['s']

    train_ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    train_npz = dict(np.load(args.data))
    train_t = build_tensors(train_ds, train_npz, args.local, args.capacity)
    dev_ds = load_teacher_dataset(args.dev_teacher_dir, data_path=args.dev_data)
    dev_npz = dict(np.load(args.dev_data))
    dev_t = build_tensors(dev_ds, dev_npz, args.local, args.capacity)

    # TRAIN 文件内部按实例切 train-fit / internal-dev（与训练一致）
    train_idx, dev_idx = split_train_dev(train_t['instance_ids'], args.seed)

    def _score_all(t):
        C = t['context_vecs'].shape[0]
        out = []
        for i in range(C):
            s = _score(graphdef, params, args.local,
                       t['context_vecs'][i:i+1], t['action_vals'][i:i+1],
                       t['action_valid'][i:i+1], t['local_vals'][i:i+1],
                       t['local_valid'][i:i+1])
            out.append(np.asarray(s)[0])
        return np.stack(out)

    train_scores = _score_all(train_t)
    dev_scores = _score_all(dev_t)

    # 分层
    rows = []
    def _event_stratum(event_ids, clock, init):
        if init:
            return 'initial'
        # 非初始按 clock 三分
        t = np.percentile(clock[~init], [33, 67])
        stage = np.where(clock < t[0], 'early',
                         np.where(clock < t[1], 'mid', 'late'))
        return stage

    # TRAIN-fit
    ti = train_idx
    init_mask = (train_t['event_ids'][ti] == 0)
    # 全局非初始 clock 三分位（跨 TRAIN+DEV 统一，避免层内空）
    all_noninit_clock = np.concatenate([
        train_t['clock'][train_t['event_ids'] != 0],
        dev_t['clock'][dev_t['event_ids'] != 0]]) if (train_t['event_ids'] != 0).any() else np.array([0.0])
    qs = np.percentile(all_noninit_clock, [33, 67]) if len(all_noninit_clock) > 0 else [0, 0]

    def _stage(clock, event_id):
        if event_id == 0:
            return 'initial'
        if clock < qs[0]:
            return 'early'
        if clock < qs[1]:
            return 'mid'
        return 'late'

    for name, t, scores, idxs in [('TRAIN-fit', train_t, train_scores, train_idx),
                                  ('internal-dev', train_t, train_scores, dev_idx),
                                  ('external-DEV', dev_t, dev_scores,
                                   np.arange(len(dev_t['instance_ids'])))]:
        if len(idxs) == 0:
            continue
        # 全层
        rows.append({'set': name, 'stage': 'all',
                     **_stratum_metrics(t, scores, s_scale, idxs)})
        # 分层
        for stage in ['initial', 'early', 'mid', 'late']:
            sub = np.array([i for i in idxs
                            if _stage(t['clock'][i], t['event_ids'][i]) == stage])
            m = _stratum_metrics(t, scores, s_scale, sub)
            if m is not None:
                rows.append({'set': name, 'stage': stage, **m})

    with open(os.path.join(args.out, 'stratified.json'), 'w') as f:
        json.dump({'s_scale': s_scale, 'rows': rows}, f, indent=2)

    print(f"\n=== stratified eval (s={s_scale:.4f}) ===")
    print(f"{'set':14s} {'stage':9s} {'inst':>4s} {'ctx':>4s} {'rank':>4s} {'sup':>5s} "
          f"{'spm':>7s} {'top1':>6s} {'keep_top1':>9s} {'keep%':>6s} {'acc':>4s} "
          f"{'ben':>4s} {'harm':>4s}")
    for r in rows:
        print(f"{r['set']:14s} {r['stage']:9s} {r['n_instances']:4d} {r['n_contexts']:4d} "
              f"{r['n_ranked']:4d} {r['n_supervision']:5d} "
              f"{r['spearman']:7.3f} {r['top1_agree']:6.3f} {r['keep_baseline_top1']:9.3f} "
              f"{r['keep_defer_ratio']:6.3f} {r['model_accept']:4d} "
              f"{r['accept_beneficial']:4d} {r['accept_harmful']:4d}")
    print(f"saved: {args.out}/stratified.json")


if __name__ == '__main__':
    main()
