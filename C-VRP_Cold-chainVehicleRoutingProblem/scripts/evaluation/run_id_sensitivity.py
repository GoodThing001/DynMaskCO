"""编号敏感性检查（去编号消融的前置/收口证据，不训练）。

对给定 feature-only-local checkpoint 选若干 context，固定 depot，对非 depot 节点做一致重编号
（同步重映射节点属性、snapshot 引用与动作端点；保持物理状态/路线顺序/动作语义不变），
再对每个候选重算分数，比较「同一个物理决策，仅换客户编号」下分数/排序/接受决定是否变化。

关键：不重跑 baseline 构造对照状态（编号变化不会经 tie-breaking 改变 baseline 计划，混入他因）。
renumber 只改变 6 个编号通道（动作 customer/slot_anchor/predecessor/successor + 车队
vehicle_node/committed_next），其余（局部几何、订单特征、有效位、插入位置、KEEP/DEFER）
严格不变——脚本内做不变性自检，确保任何分数差异只来自编号通道。

用法（服务器，索引 checkpoint 的敏感性）：
    python scripts/evaluation/run_id_sensitivity.py \
        --ckpt results/m0_scale/probe_local_n64_s42/local.ckpt \
        --teacher-dir results/m0_scale/train --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --max-contexts 8 --num-renumbers 5 --out results/m0_scale/id_sens_n64_s42

去编号 checkpoint 的不变性检查（加 --deindex）：
    python scripts/evaluation/run_id_sensitivity.py \
        --ckpt results/m0_scale/probe_local_n64_deindex_s42/local.ckpt \
        --deindex --teacher-dir ... --out results/m0_scale/id_sens_n64_deindex_s42
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
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM
from train_feature_only_local import LocalHead, IN_DIM

# 重编号会改变的编号通道（与 coldchain_visible_features 的 ACTION/FLEET_ID_CHANNELS 一致）。
_NODE_AXIS_KEYS = ('coords', 'demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
                   'initial_quality', 'reveal_time', 'quality_loss', 'visible_mask')


def renumber_context(dataset, snapshot, cands, rng):
    """对非 depot 节点做一致重编号。

    perm[old_id] = new_id，perm[0]=0（depot 固定）。返回 (renamed_dataset, renamed_snapshot,
    renamed_cands, perm)。per-node 数组按 node 轴用 inv_perm 重排；node 引用用 perm 重映射；
    slot_anchor 仅在 >0（真实节点锚）时重映射，<=0（depot/车辆编码）保持不变。
    """
    N = int(dataset['coords'].shape[1])
    perm = np.arange(N, dtype=np.int64)
    perm[1:] = (rng.permutation(N - 1) + 1).astype(np.int64)
    inv_perm = np.empty(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N, dtype=np.int64)

    renamed_ds = dict(dataset)
    for k in _NODE_AXIS_KEYS:
        if k in dataset and np.asarray(dataset[k]).ndim >= 2:
            renamed_ds[k] = np.asarray(dataset[k])[:, inv_perm]
    for k in ('dist_mat', 'energy_mat'):
        if k in dataset and np.asarray(dataset[k]).ndim == 3:
            renamed_ds[k] = np.asarray(dataset[k])[:, inv_perm][:, :, inv_perm]

    renamed_snap = dict(snapshot)
    if 'vehicle_node' in snapshot:
        renamed_snap['vehicle_node'] = perm[np.asarray(snapshot['vehicle_node'], np.int64)].tolist()
    if 'committed_next' in snapshot:
        cn = np.asarray(snapshot['committed_next'], np.int64)
        renamed_snap['committed_next'] = np.where(cn >= 0, perm[np.clip(cn, 0, N - 1)], cn).tolist()
    if 'visible_mask' in snapshot:
        renamed_snap['visible_mask'] = np.asarray(snapshot['visible_mask'])[inv_perm].tolist()

    renamed_cands = []
    for c in cands:
        rc = dict(c)
        a = c.get('action')
        if isinstance(a, dict):
            ra = dict(a)
            for key in ('customer', 'predecessor', 'successor'):
                if ra.get(key) is not None:
                    ra[key] = int(perm[int(ra[key])])
            if ra.get('slot_anchor') is not None and int(ra['slot_anchor']) > 0:
                ra['slot_anchor'] = int(perm[int(ra['slot_anchor'])])
            rc['action'] = ra
        renamed_cands.append(rc)
    return renamed_ds, renamed_snap, renamed_cands, perm


def _context_vec(data_npz, inst_idx, snapshot, deindex=False):
    feats = extract_context_features(data_npz, inst_idx, snapshot, deindex=deindex)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def build_context_tensors(data_npz, inst_idx, snapshot, cands, capacity, deindex=False):
    ctx_vec = _context_vec(data_npz, inst_idx, snapshot, deindex=deindex)
    M = len(cands)
    av = np.zeros((M, ACTION_FEAT_DIM), np.float32)
    avd = np.zeros((M, ACTION_FEAT_DIM), np.float32)
    lv = np.zeros((M, LOCAL_VAL_DIM), np.float32)
    lvd = np.zeros((M, LOCAL_VALID_DIM), np.float32)
    for j, c in enumerate(cands):
        a, ad = extract_action_features(c, deindex=deindex)
        av[j], avd[j] = a, ad
        l, ld = extract_candidate_local_features(
            data_npz, inst_idx, c.get('action'),
            c.get('certificate', {}).get('incremental_distance'), capacity)
        lv[j], lvd[j] = l, ld
    return ctx_vec, av, avd, lv, lvd


def _score(graphdef, params, ctx_vec, av, avd, lv, lvd):
    h = nnx.merge(graphdef, params)
    return np.asarray(h(jnp.asarray(ctx_vec[None]), jnp.asarray(av[None]),
                        jnp.asarray(avd[None]), jnp.asarray(lv[None]),
                        jnp.asarray(lvd[None])))[0]


def _decision(scores, legal, is_pseudo, s_scale):
    keep_idx = np.flatnonzero(np.asarray(legal) & np.asarray(is_pseudo))
    if keep_idx.size > 0:
        keep_score = float(scores[keep_idx[0]])
    else:
        keep_score = float(scores[np.flatnonzero(np.asarray(legal))[0]])
    mod_idx = np.flatnonzero(np.asarray(legal) & ~np.asarray(is_pseudo))
    if mod_idx.size == 0:
        return None, None, False
    best = mod_idx[int(np.argmax(scores[mod_idx]))]
    g_hat = s_scale * (float(scores[best]) - keep_score)
    return best, g_hat, g_hat > 0.0


def _verify_renumber_invariant(t_orig, t_ren, tol=1e-6):
    """自检：renumber 只改编号通道，局部几何/订单特征/有效位/插入位置不变。"""
    ctx_o, av_o, avd_o, lv_o, lvd_o = t_orig
    ctx_r, av_r, avd_r, lv_r, lvd_r = t_ren
    # 局部特征、局部有效位、动作有效位严格不变。
    ok = np.allclose(lv_o, lv_r, atol=tol) and np.array_equal(lvd_o, lvd_r)
    ok = ok and np.array_equal(avd_o, avd_r)
    # 动作非编号通道（kind=1, position=3, incumbent=6, is_pseudo=7）不变。
    keep_cols = [1, 3, 6, 7]
    ok = ok and np.allclose(av_o[:, keep_cols], av_r[:, keep_cols], atol=tol)
    # 订单特征（masked mean）不变。
    ok = ok and np.allclose(ctx_o[:FEATURE_CONTEXT_DIM - 16], ctx_r[:FEATURE_CONTEXT_DIM - 16],
                            atol=tol)
    # 车队非编号通道不变（编号通道 0,4 之外）。
    f_o = ctx_o[FEATURE_CONTEXT_DIM - 16:]
    f_r = ctx_r[FEATURE_CONTEXT_DIM - 16:]
    keep = [i for i in range(16) if i not in (0, 4)]
    ok = ok and np.allclose(f_o[keep], f_r[keep], atol=tol)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-contexts', type=int, default=8)
    ap.add_argument('--num-renumbers', type=int, default=5)
    ap.add_argument('--deindex', action='store_true',
                    help='checkpoint 为去编号训练，特征也按 deindex 提取（用于不变性检查）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import pickle
    with open(args.ckpt, 'rb') as f:
        ck = pickle.load(f)
    head = LocalHead(IN_DIM, rngs=args.seed)
    graphdef, _ = nnx.split(head)
    params = ck['params']
    s_scale = ck['s']

    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    contexts = ds.contexts[:args.max_contexts]

    rng = np.random.default_rng(args.seed)
    per_context = []
    n_score_changed = 0
    n_rank_changed = 0
    n_accept_changed = 0
    n_invariant_violated = 0
    max_diffs = []

    for ctx in contexts:
        inst_idx = int(ctx['inst_idx'])
        snapshot = ctx['snapshot']
        cands = ds.candidates_by_context[ctx['context_id']]
        legal = ds.legal_mask(cands)
        is_pseudo = [bool(c.get('is_pseudo', False)) for c in cands]

        t_orig = build_context_tensors(data_npz, inst_idx, snapshot, cands, args.capacity,
                                       deindex=args.deindex)
        scores_orig = _score(graphdef, params, *t_orig)

        ctx_max_diff = 0.0
        ctx_rank_changed = False
        ctx_accept_changed = False
        for _ in range(args.num_renumbers):
            rd, rs, rcands, _ = renumber_context(data_npz, snapshot, cands, rng)
            t_ren = build_context_tensors(rd, inst_idx, rs, rcands, args.capacity,
                                          deindex=args.deindex)
            if not _verify_renumber_invariant(t_orig, t_ren):
                n_invariant_violated += 1
            scores_ren = _score(graphdef, params, *t_ren)
            d = float(np.max(np.abs(scores_orig - scores_ren)))
            ctx_max_diff = max(ctx_max_diff, d)
            bo, _, ao = _decision(scores_orig, legal, is_pseudo, s_scale)
            br, _, ar = _decision(scores_ren, legal, is_pseudo, s_scale)
            if bo != br:
                ctx_rank_changed = True
            if ao != ar:
                ctx_accept_changed = True

        max_diffs.append(ctx_max_diff)
        if ctx_max_diff > 1e-6:
            n_score_changed += 1
        if ctx_rank_changed:
            n_rank_changed += 1
        if ctx_accept_changed:
            n_accept_changed += 1
        per_context.append({
            'context_id': ctx['context_id'],
            'inst_idx': inst_idx,
            'event_id': int(ctx['event_id']),
            'n_candidates': len(cands),
            'max_abs_score_diff': ctx_max_diff,
            'rank_changed': bool(ctx_rank_changed),
            'accept_changed': bool(ctx_accept_changed),
        })

    summary = {
        'ckpt': args.ckpt, 'deindex': args.deindex, 's_scale': s_scale,
        'n_contexts': len(contexts), 'num_renumbers': args.num_renumbers,
        'n_score_changed': n_score_changed,
        'n_rank_changed': n_rank_changed,
        'n_accept_changed': n_accept_changed,
        'n_invariant_violated': n_invariant_violated,
        'max_abs_score_diff': float(np.max(max_diffs)) if max_diffs else 0.0,
        'mean_max_abs_score_diff': float(np.mean(max_diffs)) if max_diffs else 0.0,
        'invariant': n_score_changed == 0 and n_invariant_violated == 0,
    }
    with open(os.path.join(args.out, 'id_sensitivity.json'), 'w') as f:
        json.dump({'summary': summary, 'per_context': per_context}, f, indent=2)

    print(f"\n=== ID sensitivity (deindex={args.deindex}, contexts={len(contexts)}, "
          f"renumbers={args.num_renumbers}) ===")
    print(f"  score changed:   {n_score_changed}/{len(contexts)}")
    print(f"  rank changed:    {n_rank_changed}/{len(contexts)}")
    print(f"  accept changed:  {n_accept_changed}/{len(contexts)}")
    print(f"  invariant violated: {n_invariant_violated}")
    print(f"  max abs score diff: {summary['max_abs_score_diff']:.6f}")
    print(f"  mean max abs score diff: {summary['mean_max_abs_score_diff']:.6f}")
    print(f"  invariant: {summary['invariant']}")
    print(f"saved: {args.out}/id_sensitivity.json")


if __name__ == '__main__':
    main()
