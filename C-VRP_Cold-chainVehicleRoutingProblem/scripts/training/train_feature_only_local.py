"""Feature-only-local：在 feature-only 基础上加入候选局部特征。

输入 = feature_only_context(25) + 动作(8 值 + 8 有效) + 候选局部(25 值 + 4 有效) = 70 维。
候选局部 = customer/anchor/pred/succ 的坐标/需求/时间窗/温区 + 插入增量。这是把「节点编号」
换成「节点几何与冷链属性」的对照，用于判断局部信息能否提升效用排序。

与 feature-only 共用同一训练协议（Huber y=-delta/s，s 只在 TRAIN 估计，context 内平均），
只是输入维度不同。训练 + 评价一体（打印 TRAIN/DEV 的 MAE/spearman/top1_agree）。

用法（服务器）：
    python scripts/training/train_feature_only_local.py \
        --teacher-dir results/m0dev/train_spread --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
        --dev-teacher-dir results/m0dev/dev_spread --dev-data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
        --num-steps 2000 --batch-size 8 --seed 42 --out results/m0dev/probe_fo_local
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        extract_candidate_local_features,
                                        ACTION_FEAT_DIM, LOCAL_VAL_DIM, LOCAL_VALID_DIM)
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM
from run_coldchain_utility_probe import compute_metrics

IN_DIM = FEATURE_CONTEXT_DIM + 2 * ACTION_FEAT_DIM + LOCAL_VAL_DIM + LOCAL_VALID_DIM  # 70


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


class LocalHead(nnx.Module):
    def __init__(self, in_dim=IN_DIM, hidden=(128, 64), rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.l1 = nnx.Linear(in_dim, hidden[0], rngs=rngs)
        self.l2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
        self.l3 = nnx.Linear(hidden[1], 1, rngs=rngs)

    def __call__(self, context, action_vals, action_valid, local_vals, local_valid):
        a = jnp.concatenate([action_vals, action_valid.astype(action_vals.dtype)], axis=-1)
        l = jnp.concatenate([local_vals, local_valid.astype(local_vals.dtype)], axis=-1)
        # context 广播到候选维
        context = jnp.broadcast_to(jnp.expand_dims(context, axis=-2),
                                   a.shape[:-1] + (context.shape[-1],))
        x = jnp.concatenate([context, a, l], axis=-1)
        x = jax.nn.gelu(self.l1(x))
        x = jax.nn.gelu(self.l2(x))
        return self.l3(x)[..., 0]


def _context_vec(data_npz, inst_idx, snapshot, deindex=False):
    feats = extract_context_features(data_npz, inst_idx, snapshot, deindex=deindex)
    return np.asarray(feature_only_context(
        jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
        jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))


def build_dataset(ds, data_npz, capacity=50.0, deindex=False):
    contexts = ds.contexts
    cands_by = ds.candidates_by_context
    C = len(contexts)
    M = max(len(cands_by[c['context_id']]) for c in contexts)
    A = ACTION_FEAT_DIM
    context_vecs = np.zeros((C, FEATURE_CONTEXT_DIM), np.float32)
    action_vals = np.zeros((C, M, A), np.float32)
    action_valid = np.zeros((C, M, A), np.float32)
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

    for i, ctx in enumerate(contexts):
        ctx_id = ctx['context_id']
        instance_ids[i] = int(ctx['inst_idx'])
        context_vecs[i] = _context_vec(data_npz, int(ctx['inst_idx']), ctx['snapshot'],
                                       deindex=deindex)
        cands = cands_by[ctx_id]
        for j, c in enumerate(cands):
            av, avd = extract_action_features(c, deindex=deindex)
            action_vals[i, j] = av
            action_valid[i, j] = avd
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

    return {'context_ids': [c['context_id'] for c in contexts],
            'instance_ids': instance_ids, 'context_vecs': context_vecs,
            'action_vals': action_vals, 'action_valid': action_valid,
            'local_vals': local_vals, 'local_valid': local_valid,
            'cand_valid': cand_valid, 'legal': legal, 'supervision': supervision,
            'service_ok': service_ok, 'is_pseudo': is_pseudo, 'selected': selected,
            'delta': delta}


def huber_loss(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def split_train_dev(instance_ids, seed=42, dev_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.unique(instance_ids)
    n_dev = max(1, int(round(len(uniq) * dev_frac)))
    n_dev = min(n_dev, len(uniq) - 1) if len(uniq) > 1 else 0
    dev_inst = set(rng.permutation(uniq)[:n_dev].tolist())
    dev_idx = np.array([i for i in range(len(instance_ids)) if instance_ids[i] in dev_inst],
                       dtype=np.int64)
    train_idx = np.array([i for i in range(len(instance_ids)) if instance_ids[i] not in dev_inst],
                         dtype=np.int64)
    return train_idx, dev_idx


def drop_unsupervised(tensors):
    keep = np.asarray(tensors['supervision']).any(axis=1)
    out = {}
    for k, v in tensors.items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(keep):
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == len(keep):
            out[k] = [x for x, kk in zip(v, keep) if kk]
        else:
            out[k] = v
    return out, keep


def score_batch(graphdef, params, tensors, idxs, chunk=256):
    h = nnx.merge(graphdef, params)
    out = []
    for lo in range(0, len(idxs), chunk):
        b = idxs[lo:lo + chunk]
        s = h(jnp.asarray(tensors['context_vecs'][b]),
              jnp.asarray(tensors['action_vals'][b]),
              jnp.asarray(tensors['action_valid'][b]),
              jnp.asarray(tensors['local_vals'][b]),
              jnp.asarray(tensors['local_valid'][b]))
        out.append(np.asarray(s))
    return np.concatenate(out, axis=0) if out else np.zeros((0,), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--dev-teacher-dir', default=None)
    ap.add_argument('--dev-data', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-train-instances', type=int, default=None,
                    help='只用前 N 个实例（按 instance_id 升序）训练；None=全部实例')
    ap.add_argument('--deindex', action='store_true',
                    help='去编号消融：屏蔽动作/车队的原始编号通道，保持 70 维')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    tensors = build_dataset(ds, data_npz, args.capacity, deindex=args.deindex)
    tensors, _ = drop_unsupervised(tensors)
    # 选前 N 个实例（TRAIN-8 ⊂ TRAIN-64，按 instance_id 升序，确定性）
    if args.max_train_instances is not None:
        uniq = sorted(set(int(x) for x in tensors['instance_ids']))[:args.max_train_instances]
        uniq_set = set(uniq)
        keep = np.array([int(tensors['instance_ids'][i]) in uniq_set
                         for i in range(len(tensors['instance_ids']))])
        for k in list(tensors.keys()):
            v = tensors[k]
            if isinstance(v, np.ndarray) and v.shape[0] == len(keep):
                tensors[k] = v[keep]
            elif isinstance(v, list) and len(v) == len(keep):
                tensors[k] = [x for x, kk in zip(v, keep) if kk]
    # 全部实例参与梯度更新（CAL/DEV 已独立，不再内部切 20%）
    train_idx = np.arange(len(tensors['instance_ids']), dtype=np.int64)
    s = max(float(np.std(np.asarray(tensors['delta'])[np.asarray(tensors['supervision'])])),
            1e-6)
    labels = np.where(tensors['supervision'], -np.asarray(tensors['delta']) / s, 0.0).astype(np.float32)
    tensors['labels'] = labels
    print(f"  contexts={len(tensors['context_ids'])} train={len(train_idx)} s={s:.4f} "
          f"n_instances={len(set(int(x) for x in tensors['instance_ids']))}", flush=True)

    head = LocalHead(IN_DIM, rngs=args.seed)
    graphdef, params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    opt_state = tx.init(params)

    ctx = jnp.asarray(tensors['context_vecs'])
    av = jnp.asarray(tensors['action_vals'])
    avd = jnp.asarray(tensors['action_valid'])
    lv = jnp.asarray(tensors['local_vals'])
    lvd = jnp.asarray(tensors['local_valid'])
    sup = jnp.asarray(tensors['supervision'])
    y = jnp.asarray(tensors['labels'])
    train_idx = np.asarray(train_idx, dtype=np.int64)

    def loss_fn(params_, ctx_b, av_b, avd_b, lv_b, lvd_b, sup_b, y_b):
        h = nnx.merge(graphdef, params_)
        score = h(ctx_b, av_b, avd_b, lv_b, lvd_b)   # [B, M]
        err = score - y_b
        per = jnp.where(sup_b, huber_loss(err), 0.0)
        return (per.sum(axis=-1) / jnp.maximum(sup_b.sum(axis=-1), 1.0)).mean()

    @jax.jit
    def train_step(params_, opt_state_, ctx_b, av_b, avd_b, lv_b, lvd_b, sup_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(params_, ctx_b, av_b, avd_b, lv_b, lvd_b,
                                                  sup_b, y_b)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, optax.apply_updates(params_, updates), new_opt

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    losses = []
    for step in range(args.num_steps):
        idx = rng.choice(train_idx, size=args.batch_size, replace=True)
        loss, params, opt_state = train_step(params, opt_state, ctx[idx], av[idx], avd[idx],
                                             lv[idx], lvd[idx], sup[idx], y[idx])
        losses.append(float(loss))
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{args.num_steps} loss={float(loss):.4f} | "
                  f"{time.time()-t0:.0f}s", flush=True)

    # 评价（TRAIN + DEV）
    def _eval(tensors_):
        scores = score_batch(graphdef, params, tensors_, np.arange(len(tensors_['context_ids'])))
        m = compute_metrics(scores, tensors_)
        return m

    train_m = _eval(tensors)
    print(f"  [TRAIN] MAE={train_m['offline']['utility_mae']:.4f} "
          f"spearman={train_m['offline']['rank_corr_spearman']:.4f} "
          f"top1={train_m['online']['top1_agreement']:.4f}", flush=True)

    dev_m = None
    if args.dev_teacher_dir:
        dds = load_teacher_dataset(args.dev_teacher_dir, data_path=args.dev_data)
        dtensors = build_dataset(dds, dict(np.load(args.dev_data)), args.capacity,
                                 deindex=args.deindex)
        dtensors, _ = drop_unsupervised(dtensors)
        dtensors['labels'] = np.where(dtensors['supervision'],
                                      -np.asarray(dtensors['delta']) / s, 0.0).astype(np.float32)
        dev_m = _eval(dtensors)
        print(f"  [DEV]   MAE={dev_m['offline']['utility_mae']:.4f} "
              f"spearman={dev_m['offline']['rank_corr_spearman']:.4f} "
              f"top1={dev_m['online']['top1_agreement']:.4f}", flush=True)

    with open(os.path.join(args.out, 'local.ckpt'), 'wb') as f:
        pickle.dump({'params': params, 's': s, 'in_dim': IN_DIM}, f)
    json.dump({'in_dim': IN_DIM, 's': s, 'num_steps': args.num_steps,
               'deindex': args.deindex,
               'final_loss': float(loss),
               'train': {'mae': train_m['offline']['utility_mae'],
                         'spearman': train_m['offline']['rank_corr_spearman'],
                         'top1': train_m['online']['top1_agreement']},
               'dev': ({'mae': dev_m['offline']['utility_mae'],
                        'spearman': dev_m['offline']['rank_corr_spearman'],
                        'top1': dev_m['online']['top1_agreement']} if dev_m else None)},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    with open(os.path.join(args.out, 'losses.json'), 'w') as f:
        json.dump({'num_steps': args.num_steps, 'loss': losses}, f)
    print(f"saved: {args.out}/local.ckpt")


if __name__ == '__main__':
    main()
