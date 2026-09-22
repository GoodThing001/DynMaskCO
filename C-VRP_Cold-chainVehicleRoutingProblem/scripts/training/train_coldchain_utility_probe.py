"""M0 feature-only utility probe 训练器（工作包 B）。

冻结特征提取后训练 `UtilityHead`：Huber 回归 y_a = -(J_a - J_keep)/s。尺度 s 只在 TRAIN
有效标签（supervision_mask）上估计并随 checkpoint 保存，DEV 固定使用。损失先按 context 内
平均（避免候选多的事件占过大权重），再按 context 平均；采样先按实例再抽 context（同实例的
context 不跨 train/dev，通过 instance 级切分保证）。

本模块导出可复用函数：`build_dataset` / `estimate_scale` / `split_train_dev` / `train_probe`
/ `build_head` / `save_checkpoint` / `load_checkpoint`。`run_coldchain_utility_probe.py`（评价
入口）与后续 encoder 模式共用 `train_probe` 训练循环，只替换 context 向量的来源，不复制训练
流程。

用法（服务器）：
    python scripts/training/train_coldchain_utility_probe.py \
        --teacher-dir <teacher_dataset_dir> --data <npz> \
        --num-steps 2000 --batch-size 8 --lr 1e-3 --seed 42 --out <dir>
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))          # scripts/training
_SCRIPTS = os.path.dirname(_BASE)                            # scripts
_CVRPTW = os.path.dirname(_SCRIPTS)                          # extension root
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        ACTION_FEAT_DIM)
from coldchain_utility_head import (UtilityHead, feature_only_context,
                                    encoder_context, shuffle_node_embeddings,
                                    FEATURE_CONTEXT_DIM)
from cvrptw_utils import coord_normalize_visible

DEFAULT_HIDDEN = (128, 64)
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 8
ENCODER_EMBED_DIM = 256


def huber_loss(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def build_dataset(ds, data_npz, deindex=False):
    """从 TeacherDataset + NPZ 构造训练张量（feature-only）。

    deindex=True 时屏蔽动作/车队的编号通道（与 feature-only-local 去编号消融同口径）。
    encoder 模式的 context 向量由 encoder 表征聚合得到（本身无编号通道），但动作特征仍走
    本函数，故 deindex 对 encoder 模式同样生效（只影响动作显式特征）。

    返回 dict（每项 [C, ...]，C=context 数，M=最大候选数）：
      context_ids [C], instance_ids [C], context_vecs [C, 25],
      action_vals [C, M, 8], action_valid [C, M, 8], cand_valid [C, M],
      legal [C, M], supervision [C, M], delta [C, M], service_ok [C, M],
      selected [C, M], is_pseudo [C, M]
    """
    contexts = ds.contexts
    cands_by = ds.candidates_by_context
    C = len(contexts)
    M = max(len(cands_by[c['context_id']]) for c in contexts)
    A = ACTION_FEAT_DIM

    context_vecs = np.zeros((C, FEATURE_CONTEXT_DIM), np.float32)
    action_vals = np.zeros((C, M, A), np.float32)
    action_valid = np.zeros((C, M, A), np.float32)
    cand_valid = np.zeros((C, M), bool)
    legal = np.zeros((C, M), bool)
    supervision = np.zeros((C, M), bool)
    delta = np.zeros((C, M), np.float32)
    service_ok = np.zeros((C, M), bool)
    selected = np.zeros((C, M), bool)
    is_pseudo = np.zeros((C, M), bool)
    instance_ids = np.zeros(C, np.int64)
    context_ids = []

    for i, ctx in enumerate(contexts):
        ctx_id = ctx['context_id']
        context_ids.append(ctx_id)
        instance_ids[i] = int(ctx['inst_idx'])
        feats = extract_context_features(data_npz, int(ctx['inst_idx']), ctx['snapshot'],
                                         deindex=deindex)
        context_vecs[i] = np.asarray(feature_only_context(
            jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
            jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))
        cands = cands_by[ctx_id]
        for j, c in enumerate(cands):
            av, avd = extract_action_features(c, deindex=deindex)
            action_vals[i, j] = av
            action_valid[i, j] = avd
            cand_valid[i, j] = True
            legal[i, j] = bool(ds.legal_mask([c])[0])
            supervision[i, j] = bool(ds.supervision_mask([c])[0])
            d = c.get('delta_vs_keep')
            delta[i, j] = float(d) if d is not None else 0.0
            service_ok[i, j] = bool(c.get('service_ok', False))
            selected[i, j] = bool(c.get('selected_by_teacher', False))
            is_pseudo[i, j] = bool(c.get('is_pseudo', False))

    return {
        'context_ids': context_ids,
        'instance_ids': instance_ids,
        'context_vecs': context_vecs,
        'action_vals': action_vals,
        'action_valid': action_valid,
        'cand_valid': cand_valid,
        'legal': legal,
        'supervision': supervision,
        'delta': delta,
        'service_ok': service_ok,
        'selected': selected,
        'is_pseudo': is_pseudo,
    }


def estimate_scale(delta, supervision):
    """在 TRAIN 有效标签上估计尺度 s = std(delta_vs_keep)，floor 防除零。"""
    d = np.asarray(delta)[np.asarray(supervision)]
    if d.size == 0:
        return 1.0
    return max(float(np.std(d)), 1e-6)


def apply_scale(tensors, s):
    """按已固定的 s 构造标签 y = -delta/s（仅 supervision 处有效，其余置 0）。"""
    y = np.where(tensors['supervision'],
                 -np.asarray(tensors['delta']) / s, 0.0).astype(np.float32)
    return y


def build_encoder_contexts(ds, data_npz, base_model, capacity=50, tw_max=24.0,
                           shuffle=False, shuffle_seed=0, chunk=8):
    """冻结 encoder → 每 context 节点表征 H → 可见掩码聚合为 context 向量。

    7D 输入与 train_fleet_head.build_H_batch 完全一致（coord_normalize_visible + 隐藏节点
    特征置 0、坐标置 0.5）；edge_feat=None（「预训练 encoder、无边特征输入」模式）。时间尺度
    tw_max 用任务固定 horizon（R1=24.0，来自恒可见 depot，非隐藏订单统计）。

    返回 (context_vecs [C, D], H [C, N, D])。shuffle=True 时先 shuffle_node_embeddings 再聚合
    （破坏节点身份对应，作为表征控制组）。
    """
    from train_fleet_head import load_base_model  # lazy，避免模块级重依赖

    contexts = ds.contexts
    C = len(contexts)
    N = int(data_npz['coords'].shape[1])
    inst_ids = np.array([int(c['inst_idx']) for c in contexts], np.int64)
    visible = np.stack([np.asarray(c['snapshot']['visible_mask'], bool) for c in contexts])

    coords = data_npz['coords'][inst_ids].astype(np.float32)
    demands = data_npz['demands'][inst_ids].astype(np.float32)
    tw_start = data_npz['tw_start'][inst_ids].astype(np.float32)
    tw_end = data_npz['tw_end'][inst_ids].astype(np.float32)
    temp_class = data_npz['temp_class'][inst_ids].astype(np.float32)
    reveal_time = data_npz['reveal_time'][inst_ids].astype(np.float32)

    raw = np.concatenate([
        coords, (demands / capacity)[..., None], (tw_start / tw_max)[..., None],
        (tw_end / tw_max)[..., None], (temp_class / 2.0)[..., None],
        (reveal_time / tw_max)[..., None],
    ], axis=-1).astype(np.float32)
    vis = visible[..., None]
    raw[..., 2:] = raw[..., 2:] * vis
    raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5

    Hs = []
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        raw_j = jnp.array(raw[lo:hi])
        raw_j = raw_j.at[..., :2].set(
            coord_normalize_visible(raw_j[..., :2], jnp.array(visible[lo:hi])))
        H = base_model.encode(raw_j, visible_mask=jnp.array(visible[lo:hi]))  # edge_feat=None
        Hs.append(np.asarray(H))
    H = np.concatenate(Hs, axis=0).astype(np.float32)   # [C, N, D]

    if shuffle:
        H = shuffle_node_embeddings(H, np.random.default_rng(shuffle_seed))
    ctx = np.asarray(encoder_context(jnp.asarray(H), jnp.asarray(visible)))
    return ctx, H


def _params_checksum(params):
    """对 nnx.State 全部叶子 .value 做 sha256，用于验证冻结参数不变。"""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        arr = np.asarray(getattr(leaf, 'value', leaf))
        h.update(arr.tobytes())
    return h.hexdigest()


def split_train_dev(instance_ids, seed=42, dev_frac=0.2):
    """按实例切分 train/dev（同实例 context 不跨切分）；至少保留 1 个 train 实例。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(instance_ids)
    n_inst = len(uniq)
    n_dev = min(max(1, int(round(n_inst * dev_frac))), n_inst - 1) if n_inst > 1 else 0
    dev_inst = set(rng.permutation(uniq)[:n_dev].tolist())
    dev_idx = np.array([i for i in range(len(instance_ids)) if instance_ids[i] in dev_inst],
                       dtype=np.int64)
    train_idx = np.array([i for i in range(len(instance_ids)) if instance_ids[i] not in dev_inst],
                         dtype=np.int64)
    return train_idx, dev_idx


def drop_unsupervised(tensors):
    """去掉无任何有效监督的 context（读器已记录，训练/评价均跳过）。"""
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


def build_head(config):
    """从 config 重建 graphdef（结构确定；params 由 checkpoint 提供）。"""
    head = UtilityHead(config['context_dim'], config.get('action_dim', ACTION_FEAT_DIM),
                       tuple(config.get('hidden', DEFAULT_HIDDEN)), rngs=0)
    return nnx.split(head)[0]


def train_probe(graphdef, params, tx, opt_state, tensors, train_idx, num_steps, batch_size,
                seed, verbose=True):
    """通用训练循环（feature-only / encoder 共用）：Huber 回归，context 内平均再 context 平均。"""
    ctx = jnp.asarray(tensors['context_vecs'])
    av = jnp.asarray(tensors['action_vals'])
    avd = jnp.asarray(tensors['action_valid'])
    sup = jnp.asarray(tensors['supervision'])
    y = jnp.asarray(tensors['labels'])
    train_idx = np.asarray(train_idx, dtype=np.int64)

    def loss_fn(params_, ctx_b, av_b, avd_b, sup_b, y_b):
        h = nnx.merge(graphdef, params_)
        score = h(ctx_b, av_b, avd_b)                      # [B, M]
        err = score - y_b
        per = jnp.where(sup_b, huber_loss(err), 0.0)
        n_sup = jnp.maximum(sup_b.sum(axis=-1), 1.0)
        return (per.sum(axis=-1) / n_sup).mean()

    @jax.jit
    def train_step(params_, opt_state_, ctx_b, av_b, avd_b, sup_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(params_, ctx_b, av_b, avd_b, sup_b, y_b)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, optax.apply_updates(params_, updates), new_opt

    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.time()
    for step in range(num_steps):
        idx = rng.choice(train_idx, size=batch_size, replace=True)
        loss, params, opt_state = train_step(
            params, opt_state, ctx[idx], av[idx], avd[idx], sup[idx], y[idx])
        losses.append(float(loss))
        if verbose and (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{num_steps} | loss={float(loss):.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    return params, opt_state, losses


def score_batch(graphdef, params, context_vecs, action_vals, action_valid, chunk=256):
    """分批打分，返回 NumPy [C, M]。"""
    h = nnx.merge(graphdef, params)
    out = []
    for lo in range(0, context_vecs.shape[0], chunk):
        hi = min(lo + chunk, context_vecs.shape[0])
        s = h(jnp.asarray(context_vecs[lo:hi]), jnp.asarray(action_vals[lo:hi]),
              jnp.asarray(action_valid[lo:hi]))
        out.append(np.asarray(s))
    return np.concatenate(out, axis=0) if out else np.zeros((0, action_vals.shape[1]), np.float32)


def save_checkpoint(out_dir, params, config):
    """Flax 0.10.4 无 nnx.save：pickle 存 nnx.State + 配置（沿用 train_fleet_head 约定）。"""
    path = os.path.join(out_dir, 'probe.ckpt')
    with open(path, 'wb') as f:
        pickle.dump({'params': params, 'config': config}, f)
    return path


def load_checkpoint(ckpt_path):
    with open(ckpt_path, 'rb') as f:
        return pickle.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True, help='基础实例 NPZ（与 manifest.dataset_hash 一致）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH)
    ap.add_argument('--lr', type=float, default=DEFAULT_LR)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dev-frac', type=float, default=0.2)
    ap.add_argument('--max-contexts', type=int, default=None, help='smoke：截断 context 数')
    ap.add_argument('--context-mode', choices=['feature_only', 'encoder_real', 'encoder_shuffle'],
                    default='feature_only')
    ap.add_argument('--ckpt', default=None, help='冻结 MaskCO encoder（encoder 模式必填）')
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--tw-max', type=float, default=None,
                    help='encoder 时间尺度；默认取恒可见 depot tw_end（R1=24.0）')
    ap.add_argument('--shuffle-seed', type=int, default=0)
    ap.add_argument('--deindex', action='store_true',
                    help='去编号消融：屏蔽动作/车队的编号通道（encoder 模式影响动作显式特征）')
    ap.add_argument('--train-all', action='store_true',
                    help='所有实例参与梯度更新（不内部切 dev；与 feature-local / M1 同口径，'
                         'CAL/DEV 已独立）')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.context_mode != 'feature_only' and args.ckpt is None:
        raise SystemExit(f'--context-mode={args.context_mode} 需要 --ckpt（冻结 encoder）')

    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    tensors = build_dataset(ds, data_npz, deindex=args.deindex)
    if args.max_contexts is not None:
        for k in list(tensors.keys()):
            v = tensors[k]
            if isinstance(v, np.ndarray) and v.shape[0] > args.max_contexts:
                tensors[k] = v[:args.max_contexts]
            elif isinstance(v, list) and len(v) > args.max_contexts:
                tensors[k] = v[:args.max_contexts]

    tensors, _keep = drop_unsupervised(tensors)
    if args.train_all:
        train_idx = np.arange(len(tensors['instance_ids']), dtype=np.int64)
        dev_idx = np.array([], dtype=np.int64)
    else:
        train_idx, dev_idx = split_train_dev(tensors['instance_ids'], args.seed, args.dev_frac)
    # s 只在 TRAIN 有效标签上估计，DEV 固定使用同一 s。
    s = estimate_scale(tensors['delta'][train_idx], tensors['supervision'][train_idx])
    tensors['labels'] = apply_scale(tensors, s)
    print(f"  contexts={len(tensors['context_ids'])} train={len(train_idx)} dev={len(dev_idx)} "
          f"scale_s={s:.4f} (estimated on TRAIN supervision)", flush=True)

    context_dim = FEATURE_CONTEXT_DIM
    encoder_info = None
    encoder_checksum_before = None
    encoder = None
    if args.context_mode != 'feature_only':
        from train_fleet_head import load_base_model
        encoder = load_base_model(args.ckpt)
        tw_max = args.tw_max if args.tw_max is not None else float(data_npz['tw_end'][:, 0].max())
        ctx_vecs, _H = build_encoder_contexts(
            ds, data_npz, encoder, capacity=args.capacity, tw_max=tw_max,
            shuffle=(args.context_mode == 'encoder_shuffle'), shuffle_seed=args.shuffle_seed)
        tensors['context_vecs'] = ctx_vecs
        context_dim = ENCODER_EMBED_DIM
        encoder_params = nnx.split(encoder)[1]
        encoder_checksum_before = _params_checksum(encoder_params)
        encoder_info = {
            'ckpt': args.ckpt,
            'ckpt_sha256': _sha256(args.ckpt),
            'model_type': type(encoder).__name__,
            'encoder_input_dim': 7,
            'edge_feat': None,
            'tw_max': tw_max,
            'capacity': args.capacity,
            'shuffle_seed': args.shuffle_seed,
        }
        print(f"  encoder: {encoder_info['model_type']} "
              f"sha={encoder_info['ckpt_sha256'][:12]} edge_feat=None "
              f"tw_max={tw_max} checksum_before={encoder_checksum_before[:12]}", flush=True)

    config = {
        'context_dim': context_dim,
        'action_dim': ACTION_FEAT_DIM,
        'hidden': list(DEFAULT_HIDDEN),
        's': s,
        'seed': args.seed,
        'mode': args.context_mode,
        'encoder': encoder_info,
    }
    graphdef = build_head(config)
    head = UtilityHead(config['context_dim'], config['action_dim'],
                       tuple(config['hidden']), rngs=args.seed)
    _, params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = tx.init(params)

    params, opt_state, losses = train_probe(
        graphdef, params, tx, opt_state, tensors, train_idx, args.num_steps,
        args.batch_size, args.seed)

    # 冻结检查：encoder 参数在训练前后不变（head 参数已在 train_probe 内更新）。
    freeze_ok = None
    if encoder is not None:
        encoder_checksum_after = _params_checksum(nnx.split(encoder)[1])
        freeze_ok = (encoder_checksum_before == encoder_checksum_after)
        print(f"  freeze check: encoder {'UNCHANGED' if freeze_ok else 'CHANGED'} "
              f"({encoder_checksum_after[:12]})", flush=True)

    save_checkpoint(args.out, params, config)
    manifest = {
        'mode': args.context_mode,
        'deindex': args.deindex,
        'train_all': args.train_all,
        'teacher_dir': args.teacher_dir,
        'data': args.data,
        'scale_s': s,
        'n_contexts': len(tensors['context_ids']),
        'n_train': int(len(train_idx)),
        'n_dev': int(len(dev_idx)),
        'num_steps': args.num_steps,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'encoder_frozen': freeze_ok,
        'final_loss': float(losses[-1]) if losses else None,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved: {args.out}/probe.ckpt + manifest.json (final loss={losses[-1]:.4f})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    main()
