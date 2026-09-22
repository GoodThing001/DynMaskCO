"""候选冷链状态 × 决策损失 2×2（单 seed 冒烟）。

四格：{70 维显式 | +13 维候选冷链状态 X_state} × {Huber | Huber + KEEP 排序}。
小模型（无 encoder），回答「候选相关状态」和「KEEP 排序」是否有增量。B/C/D 的优化问题单独保留。

损失（固定版本）：
  基准 k = 当前 context 的 KEEP/DEFER；y_a = -δ_a/s，m_a = f(a) - f(k)。
  L_mag = mean_{a∈A_sup} Huber(f(a) - y_a)           （KEEP/DEFER 在内，目标 0）
  L_rank = mean_{a∈A_strict} softplus[-sign(y_a)·m_a/T]，T=1
  A_strict = 有效监督普通候选 且 |δ_a| > ε_J（ε_J=1e-6，原始 J 单位，仅数值平局）

部署规则 s·m_a > τ_J（τ_J 另记，不与 ε_J、s 混用）。
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
from coldchain_visible_features import (extract_vehicle_cargo_state, resolve_target_vid,
                                        X_STATE_DIM)
from dynmaskco_cc_compact import ScoringMLP

T = 1.0
EPS_J = 1e-6


def huber(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def compute_loss(scores, y, sup, pseudo, delta, use_rank, s, T=T, eps_J=EPS_J,
                 rank_mode='hard'):
    """返回 (loss, L_mag, L_rank)。scores [B, M]，y [B, M]（=-δ/s），sup/pseudo [B, M] bool，
    delta [B, M]（原始 J 单位，用于 ε_J 平局阈值）。

    rank_mode='hard'：旧硬符号 softplus[−sign(y)·m/T]（正确方向 margin 持续增大）；
    rank_mode='soft'：softplus(m/T) − σ(y/T)·(m/T)，对单个确定标签在 m=y 处取最小，
    与幅度项一致（不再要求互相冲突的分数幅度）。p=σ(y/T) 是收益标签的变换，不是校准概率。
    """
    err = scores - y
    per = jnp.where(sup, huber(err), 0.0)
    L_mag = (per.sum(axis=-1) / jnp.maximum(sup.sum(axis=-1), 1.0)).mean()
    if not use_rank:
        return L_mag, L_mag, 0.0
    k_idx = jnp.argmax(pseudo.astype(jnp.int32), axis=-1)
    B = scores.shape[0]
    k_score = scores[jnp.arange(B), k_idx]
    m_a = scores - k_score[:, None]
    strict = sup & (~pseudo) & (jnp.abs(delta) > eps_J)
    if rank_mode == 'soft':
        z_a = m_a / T
        p_a = jax.nn.sigmoid(y / T)
        r = jax.nn.softplus(z_a) - p_a * z_a
    else:
        y_sign = jnp.sign(y)
        r = jax.nn.softplus(-y_sign * m_a / T)
    r = jnp.where(strict, r, 0.0)
    L_rank = (r.sum(axis=-1) / jnp.maximum(strict.sum(axis=-1), 1.0)).mean()
    return L_mag + L_rank, L_mag, L_rank


def build_x_state(ds, npz, capacity, tw_max):
    """每个候选的 X_state [C, M, 13]。DEFER → 全零（无目标）；KEEP/普通动作 → 目标车状态。"""
    from dynmaskco_cc_context import compute_incumbent_plans
    from jf1h_repair import make_continuation
    from strict_online_env import StrictOnlineEnv
    contexts = ds.contexts
    cands_by = ds.candidates_by_context
    C = len(contexts)
    M = max(len(cands_by[c['context_id']]) for c in contexts)
    x = np.zeros((C, M, X_STATE_DIM), np.float32)
    cont = make_continuation()
    for i, ctx in enumerate(contexts):
        snapshot = ctx['snapshot']
        inst_idx = int(ctx['inst_idx'])
        env = StrictOnlineEnv(npz, capacity, 1.0, 25, replanner=None, coldchain_contract=None)
        P0, _, replan_ids = compute_incumbent_plans(env, snapshot, cont)
        for j, c in enumerate(cands_by[ctx['context_id']]):
            vid = resolve_target_vid(c.get('action'), P0, replan_ids)
            if vid is not None:
                x[i, j] = extract_vehicle_cargo_state(snapshot, vid, P0[vid], capacity, tw_max)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--use-state', action='store_true')
    ap.add_argument('--use-rank', action='store_true')
    ap.add_argument('--soft-rank', action='store_true',
                    help='软标签排序项 softplus(m/T)-σ(y/T)(m/T)，与幅度项一致；替代硬符号排序')
    ap.add_argument('--num-steps', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--tw-max', type=float, default=None)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--fit-n', type=int, default=None, help='冒烟：只用前 N 个 context')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    npz = dict(np.load(args.data))
    tw_max = args.tw_max if args.tw_max is not None else float(npz['tw_end'][:, 0].max())

    from train_m1_compact import build_explicit_and_labels
    explicit, t = build_explicit_and_labels(ds, npz, args.capacity, args.deindex)
    C, M = explicit.shape[0], explicit.shape[1]

    # s 在完整 TRAIN 有效标签上估计（四格一致），不受 --fit-n 截断影响
    delta_full = np.asarray(t['delta'], np.float32)
    sup_full = np.asarray(t['supervision'], bool)
    s = max(float(np.std(delta_full[sup_full])), 1e-6)

    # 轻量校验：每个可监督 context 恰有一个 KEEP/DEFER 参照，标签有效且 δ≈0
    n_bad_ref = 0
    for i in range(C):
        pseudo_row = np.asarray(t['is_pseudo'][i], bool)
        sup_row = np.asarray(t['supervision'][i], bool)
        if sup_row.any():
            ref = np.flatnonzero(pseudo_row & sup_row)
            if len(ref) != 1 or abs(float(t['delta'][i][ref[0]])) > 1e-6:
                n_bad_ref += 1

    if args.fit_n is not None:
        explicit = explicit[:args.fit_n]
        for k in list(t):
            if isinstance(t[k], np.ndarray) and t[k].shape[0] == C:
                t[k] = t[k][:args.fit_n]
        C = args.fit_n

    delta = np.asarray(t['delta'], np.float32)
    sup = np.asarray(t['supervision'], bool)
    pseudo = np.asarray(t['is_pseudo'], bool)
    y = np.where(sup, -delta / s, 0.0).astype(np.float32)

    X = explicit
    if args.use_state:
        xs = build_x_state(ds, npz, args.capacity, tw_max)
        if args.fit_n is not None:
            xs = xs[:args.fit_n]
        X = np.concatenate([explicit, xs], axis=-1)
    in_dim = X.shape[-1]

    head = ScoringMLP(in_dim, (128, 64), rngs=args.seed)
    head_gd, params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    opt_state = tx.init(params)

    rank_mode = 'soft' if args.soft_rank else ('hard' if args.use_rank else 'none')
    use_rank = rank_mode != 'none'

    def loss_fn(params_, x_b, y_b, sup_b, pseudo_b, delta_b):
        sc = nnx.merge(head_gd, params_)(x_b)
        return compute_loss(sc, y_b, sup_b, pseudo_b, delta_b, use_rank, s,
                            rank_mode=rank_mode)[0]

    @jax.jit
    def train_step(params_, opt_state_, x_b, y_b, sup_b, pseudo_b, delta_b):
        loss, grads = jax.value_and_grad(loss_fn)(params_, x_b, y_b, sup_b, pseudo_b, delta_b)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, optax.apply_updates(params_, updates), new_opt

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(C)

    def full_metrics():
        sc = np.asarray(nnx.merge(head_gd, params)(jnp.asarray(X)))
        _, lm, lr = compute_loss(jnp.asarray(sc), jnp.asarray(y), jnp.asarray(sup),
                                 jnp.asarray(pseudo), jnp.asarray(delta), use_rank, s,
                                 rank_mode=rank_mode)
        mae = float(np.mean(np.abs(sc[sup] - y[sup])))
        return float(lm), float(lr), mae

    print(f"config: use_state={args.use_state} rank_mode={rank_mode} in_dim={in_dim} "
          f"s={s:.4f} n_ctx={C}", flush=True)
    lm0, lr0, mae0 = full_metrics()
    print(f"  step 0: L_mag={lm0:.4f} L_rank={lr0:.4f} MAE={mae0:.4f}", flush=True)
    t0 = time.time()
    for step in range(args.num_steps):
        batch = rng.choice(idxs, size=min(args.batch_size, C), replace=True)
        loss, params, opt_state = train_step(
            params, opt_state, jnp.asarray(X[batch]), jnp.asarray(y[batch]),
            jnp.asarray(sup[batch]), jnp.asarray(pseudo[batch]), jnp.asarray(delta[batch]))
        if (step + 1) % 50 == 0:
            lm, lr, mae = full_metrics()
            print(f"  step {step+1}: L_mag={lm:.4f} L_rank={lr:.4f} MAE={mae:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(args.out, 'model.ckpt'), 'wb') as f:
        pickle.dump({'params': params, 's': s, 'in_dim': in_dim,
                     'use_state': args.use_state, 'use_rank': args.use_rank}, f)
    json.dump({'use_state': args.use_state, 'use_rank': args.use_rank, 'rank_mode': rank_mode,
               'in_dim': in_dim,
               's': s, 's_definition': 'std(delta_vs_keep) over full TRAIN supervision',
               'deindex': args.deindex, 'capacity': args.capacity, 'tw_max': tw_max,
               'lr': args.lr, 'weight_decay': 1e-2, 'batch_size': args.batch_size,
               'T': T, 'eps_J': EPS_J, 'rank_weight': 1.0,
               'checkpoint': 'last_step',
               'data_hash': ds.manifest.get('dataset_hash'),
               'n_ctx': C, 'num_steps': args.num_steps, 'seed': args.seed,
               'n_bad_ref_contexts': n_bad_ref},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/model.ckpt")


if __name__ == '__main__':
    main()
