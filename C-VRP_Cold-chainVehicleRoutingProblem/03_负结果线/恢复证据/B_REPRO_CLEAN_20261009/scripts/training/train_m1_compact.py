"""A/B/C/D 紧凑模型统一训练（去编号、候选可区分、节点有效 mask、一致 utility 加权）。

组别：
  A：纯显式特征（context 25 + action 16 + local 29 = 70，去编号），无 encoder；
  B：A + 紧凑 H 端点表示（4 端点 × d_compact）；
  C：B + 紧凑 Z 端点表示（masked decoder，recon 权重 = 0）；
  D：C + 固定权重重构损失。

共享：同一 70 维显式特征、同一训练 context（全 64 实例，含无重构变化的 context）、
同一 scale s（TRAIN 有效标签 std(delta)）、同一 utility 损失（context 内平均再 context 平均）、
同一评价规则。encoder 冻结。

用法（服务器 GPU）：
    python scripts/training/train_m1_compact.py --group B \
        --teacher-dir results/m0_scale/train --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --num-steps 2000 --batch-size 8 --seed 42 --deindex \
        --out results/m0_scale/compact_B_s42
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
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from coldchain_teacher_dataset import load_teacher_dataset
from dynmaskco_cc_compact import gather_endpoints, CompactDecoder, ScoringMLP

EXPLICIT_DIM = 70  # context 25 + action 8+8 + local 25+4


def huber(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def build_explicit_and_labels(ds, npz, capacity, deindex):
    from train_feature_only_local import build_dataset as build_feat
    t = build_feat(ds, npz, capacity, deindex=deindex)
    C, M = t['action_vals'].shape[0], t['action_vals'].shape[1]
    ctx = np.broadcast_to(t['context_vecs'][:, None, :], (C, M, t['context_vecs'].shape[1]))
    explicit = np.concatenate([
        ctx, t['action_vals'], t['action_valid'].astype(np.float32),
        t['local_vals'], t['local_valid'].astype(np.float32)], axis=-1).astype(np.float32)
    return explicit, t


def build_encoder_inputs(ds, npz, base_model, capacity, tw_max, deindex):
    from train_dynmaskco_cc import build_training_tensors
    mt = build_training_tensors(ds, npz, base_model, capacity, tw_max, deindex=deindex,
                                keep_all=True)
    n = mt['n_ctx']
    Nv_max = max(mt['Hv'][i].shape[0] for i in range(n))
    M_max = max(mt['endpoints'][i].shape[0] for i in range(n))
    d = mt['Hv'][0].shape[1]
    Hv = np.zeros((n, Nv_max, d), np.float32)
    A_in = np.zeros((n, Nv_max, Nv_max), np.float32)
    M = np.zeros((n, Nv_max, Nv_max), np.float32)
    A_teacher = np.zeros((n, Nv_max, Nv_max), np.float32)
    node_valid = np.zeros((n, Nv_max), bool)
    endpoints = np.zeros((n, M_max, 4), np.int32)
    valid = np.zeros((n, M_max, 4), bool)
    ts = np.zeros(n, np.float32)
    for i in range(n):
        ni = mt['Hv'][i].shape[0]
        mi = mt['endpoints'][i].shape[0]
        Hv[i, :ni] = mt['Hv'][i]
        A_in[i, :ni, :ni] = mt['A_in'][i]
        M[i, :ni, :ni] = mt['M'][i]
        A_teacher[i, :ni, :ni] = mt['A_teacher'][i]
        node_valid[i, :ni] = True
        endpoints[i, :mi] = mt['endpoints'][i]
        valid[i, :mi] = mt['valid'][i]
        ts[i] = mt['timestep'][i]
    return dict(Hv=Hv, A_in=A_in, M=M, A_teacher=A_teacher, node_valid=node_valid,
                endpoints=endpoints, valid=valid, ts=ts)


def _recon_loss(logits, M, A_teacher):
    tri = jnp.triu(jnp.ones_like(M), k=1)
    region = (M * tri) > 0.5
    target = (A_teacher * tri) > 0.5
    bce = -jax.nn.log_sigmoid(logits) * target - jax.nn.log_sigmoid(-logits) * (1.0 - target)
    pos = region & target
    neg = region & (~target)
    pos_loss = (bce * pos).sum() / jnp.maximum(pos.sum(), 1.0)
    neg_loss = (bce * neg).sum() / jnp.maximum(neg.sum(), 1.0)
    return 0.5 * pos_loss + 0.5 * neg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=['A', 'B', 'C', 'D'])
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', default=None, help='冻结 encoder（B/C/D 必填）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--tw-max', type=float, default=None)
    ap.add_argument('--d-compact', type=int, default=32)
    ap.add_argument('--recon-weight', type=float, default=1.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--fit-check', action='store_true',
                    help='只用前几个 context 过拟合检查（记 loss 与梯度范数）')
    ap.add_argument('--fit-n', type=int, default=4)
    args = ap.parse_args()

    use_H = args.group in ('B', 'C', 'D')
    use_Z = args.group in ('C', 'D')
    recon_weight = args.recon_weight if args.group == 'D' else 0.0

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    npz = dict(np.load(args.data))
    tw_max = args.tw_max if args.tw_max is not None else float(npz['tw_end'][:, 0].max())

    explicit, t = build_explicit_and_labels(ds, npz, args.capacity, args.deindex)
    delta = np.asarray(t['delta']).astype(np.float32)
    sup = np.asarray(t['supervision']).astype(bool)
    s = max(float(np.std(delta[sup])), 1e-6)
    y = np.where(sup, -delta / s, 0.0).astype(np.float32)

    if args.fit_check:
        explicit = explicit[:args.fit_n]
        delta = delta[:args.fit_n]
        sup = sup[:args.fit_n]
        y = y[:args.fit_n]

    base_model = None
    enc = None
    if use_H:
        from train_fleet_head import load_base_model
        base_model = load_base_model(args.ckpt)
        enc = build_encoder_inputs(ds, npz, base_model, args.capacity, tw_max, args.deindex)
        if args.fit_check:
            for k in enc:
                enc[k] = enc[k][:args.fit_n]

    d = args.d_compact
    proj_gd = proj_params = None
    dec_gd = dec_params = None
    if use_H:
        proj = nnx.Linear(256, d, use_bias=False, rngs=nnx.Rngs(args.seed))
        proj_gd, proj_params = nnx.split(proj)
    if use_Z:
        dec = CompactDecoder(d, num_layers=6, num_heads=4, rngs=args.seed)
        dec_gd, dec_params = nnx.split(dec)
    head_in = EXPLICIT_DIM + (4 * d if use_H else 0) + (4 * d if use_Z else 0)
    head = ScoringMLP(head_in, (128, 64), rngs=args.seed)
    head_gd, head_params = nnx.split(head)

    params = {}
    if use_H:
        params['proj'] = proj_params
    if use_Z:
        params['dec'] = dec_params
    params['head'] = head_params
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    opt_state = tx.init(params)

    C = explicit.shape[0]
    M = explicit.shape[1]

    def loss_fn(params, explicit_b, Hv_b, A_in_b, nv_b, e_b, v_b, ts_b, M_b, A_teacher_b, y_b,
                sup_b):
        parts = [explicit_b]
        h = None
        if use_H:
            h = nnx.merge(proj_gd, params['proj'])(Hv_b)
            parts.append(gather_endpoints(h, e_b, v_b))
        z = None
        if use_Z:
            z = nnx.merge(dec_gd, params['dec'])(h, ts_b, A_in_b, nv_b)
            parts.append(gather_endpoints(z, e_b, v_b))
        x = jnp.concatenate(parts, axis=-1)
        scores = nnx.merge(head_gd, params['head'])(x)
        err = scores - y_b
        per = jnp.where(sup_b, huber(err), 0.0)
        util = (per.sum(axis=-1) / jnp.maximum(sup_b.sum(axis=-1), 1.0)).mean()
        recon = 0.0
        if recon_weight > 0.0 and use_Z:
            logits = jnp.matmul(z, jnp.transpose(z, (0, 2, 1)))
            recon = _recon_loss(logits, M_b, A_teacher_b)
        return util + recon_weight * recon, (util, recon_weight * recon)

    @jax.jit
    def train_step(params, opt_state, explicit_b, Hv_b, A_in_b, nv_b, e_b, v_b, ts_b, M_b,
                   A_teacher_b, y_b, sup_b):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, explicit_b, Hv_b, A_in_b, nv_b, e_b, v_b, ts_b, M_b, A_teacher_b, y_b, sup_b)
        updates, new_opt = tx.update(grads, opt_state, params)
        return loss, aux, optax.apply_updates(params, updates), new_opt, grads

    def _grad_norm(g):
        return float(jnp.sqrt(sum((x ** 2).sum() for x in jax.tree_util.tree_leaves(g))))

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(C)
    t0 = time.time()
    losses = []
    for step in range(args.num_steps):
        batch = rng.choice(idxs, size=min(args.batch_size, C), replace=True)
        eb = jnp.asarray(explicit[batch])
        Hv_b = jnp.asarray(enc['Hv'][batch]) if use_H else jnp.zeros((len(batch), 1, 1))
        A_in_b = jnp.asarray(enc['A_in'][batch]) if use_H else jnp.zeros((len(batch), 1, 1))
        nv_b = jnp.asarray(enc['node_valid'][batch]) if use_H else jnp.ones((len(batch), 1), bool)
        e_b = jnp.asarray(enc['endpoints'][batch]).astype(jnp.int32) if use_H else jnp.zeros((len(batch), M, 4), jnp.int32)
        v_b = jnp.asarray(enc['valid'][batch]).astype(bool) if use_H else jnp.zeros((len(batch), M, 4), bool)
        ts_b = jnp.asarray(enc['ts'][batch]) if use_H else jnp.zeros(len(batch))
        M_b = jnp.asarray(enc['M'][batch]) if (use_Z and recon_weight > 0.0) else jnp.zeros((len(batch), 1, 1))
        A_teacher_b = jnp.asarray(enc['A_teacher'][batch]) if (use_Z and recon_weight > 0.0) else jnp.zeros((len(batch), 1, 1))
        yb = jnp.asarray(y[batch])
        sup_b = jnp.asarray(sup[batch])
        loss, aux, params, opt_state, grads = train_step(
            params, opt_state, eb, Hv_b, A_in_b, nv_b, e_b, v_b, ts_b, M_b, A_teacher_b, yb, sup_b)
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{args.num_steps} loss={float(loss):.4f} util={float(aux[0]):.4f} "
                  f"recon={float(aux[1]):.4f} | {time.time()-t0:.0f}s", flush=True)

    # 梯度范数
    gn = {}
    for k, g in grads.items():
        gn[k] = _grad_norm(g)

    ck = {'params': params, 's': s, 'group': args.group, 'd_compact': d,
          'explicit_dim': EXPLICIT_DIM, 'use_H': use_H, 'use_Z': use_Z,
          'recon_weight': recon_weight}
    with open(os.path.join(args.out, 'compact.ckpt'), 'wb') as f:
        pickle.dump(ck, f)
    json.dump({'group': args.group, 's': s, 'd_compact': d, 'explicit_dim': EXPLICIT_DIM,
               'use_H': use_H, 'use_Z': use_Z, 'recon_weight': recon_weight,
               'deindex': args.deindex, 'n_contexts': int(C), 'num_steps': args.num_steps,
               'final_loss': float(losses[-1]) if losses else None,
               'grad_norms': gn},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/compact.ckpt (s={s:.4f}) grad_norms={ {k: round(v,3) for k,v in gn.items()} }")


if __name__ == '__main__':
    main()
