"""event_plan_v2 E2：训练完整计划评价器（42/43/44 三 seed，A/B 两种表示）。

模型：小 MLP 输入 [context(25) + plan_feature(representation)] → 分数 f(z_e, P)。
损失：Huber((f(P) - f(P0)) - gB/s)，s 仅在共用有效 TRAIN 标签上估计。
gB_hat = s * (f(P) - f(P0))。

v2 修正：实例 → 事件 → 候选的平衡抽样（不再把候选展平后均匀抽样）。

表示：--representation A（旧 plan_delta 100 维）| B（可见执行后果 529 维）。

用法（本地/服务器）：
    python scripts/training/train_event_plan.py \
        --data results/m0_scale/event_plan_v2_export/data.json \
        --representation A --num-steps 2000 --batch-size 8 --seed 42 \
        --out results/m0_scale/event_plan_v2_A_s42
"""
import argparse
import json
import os
import pickle
import sys

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

from event_plan_features import plan_dim

CONTEXT_DIM = 25


def _huber(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


class PlanEvalMLP(nnx.Module):
    def __init__(self, in_dim, hidden=(128, 64), rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.l1 = nnx.Linear(in_dim, hidden[0], rngs=rngs)
        self.l2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
        self.l3 = nnx.Linear(hidden[1], 1, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.gelu(self.l1(x))
        x = jax.nn.gelu(self.l2(x))
        return self.l3(x)[..., 0]


def build_samples(data, representation):
    """按 实例→事件→候选 组织训练样本；返回 (samples, in_dim, n_inst, n_events, n_cands)。

    samples[i][e][c] = (ctx, feat, keep_feat, gB)（numpy 数组）；rows 按 inst 升序分组。
    空事件（无候选）不计入训练样本，但计入 n_events 覆盖统计（不造标签）。
    """
    feat_key = 'plan_delta' if representation == 'A' else 'consequence'
    keep_key = 'keep_plan_delta' if representation == 'A' else 'keep_consequence'
    samples = []
    n_inst = 0
    n_events = 0
    n_cands = 0
    in_dim = None
    cur_inst = None
    for row in data['rows']:
        if cur_inst != row['inst']:
            cur_inst = row['inst']
            samples.append([])
            n_inst += 1
        n_events += 1
        ctx = np.asarray(row['context'], np.float32)
        keep_feat = np.asarray(row[keep_key], np.float32)
        evt = []
        for c in row['cands']:
            feat = np.asarray(c[feat_key], np.float32)
            if in_dim is None:
                in_dim = CONTEXT_DIM + feat.shape[0]
            evt.append((ctx, feat, keep_feat, float(c['gB'])))
            n_cands += 1
        if evt:
            samples[-1].append(evt)
    # 去掉无有效事件的实例（防御：本数据不出现）
    samples = [evts for evts in samples if evts]
    n_inst = len(samples)
    return samples, in_dim, n_inst, n_events, n_cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--representation', choices=['A', 'B'], required=True)
    ap.add_argument('--num-steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        data = json.load(f)
    samples, in_dim, n_inst, n_events, n_cands = build_samples(data, args.representation)

    # 共用 TRAIN 标签尺度 s（全量 gB 的标准差，含零收益候选）
    all_gbs = np.asarray([g for inst_events in samples for evt in inst_events
                          for (_, _, _, g) in evt], np.float32)
    s = max(float(np.std(all_gbs)), 1e-6)
    print(f"representation={args.representation} in_dim={in_dim} n_inst={n_inst} "
          f"n_events={n_events} n_cands={n_cands} s={s:.4f}", flush=True)

    head = PlanEvalMLP(in_dim, rngs=args.seed)
    gd, params = nnx.split(head)
    tx = optax.adamw(1e-3, weight_decay=1e-2)
    opt_state = tx.init(params)

    def loss_fn(params_, Xb, XKb, yb):
        fP = nnx.merge(gd, params_)(Xb)
        fK = nnx.merge(gd, params_)(XKb)
        return _huber((fP - fK) - yb).mean()

    @jax.jit
    def train_step(params_, opt_state_, Xb, XKb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params_, Xb, XKb, yb)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, optax.apply_updates(params_, updates), new_opt

    rng = np.random.default_rng(args.seed)
    # 实例 → 事件 → 候选 平衡抽样
    for step in range(args.num_steps):
        ctxs, feats, keeps, ys = [], [], [], []
        for _ in range(args.batch_size):
            inst = int(rng.integers(0, n_inst))
            evt_idx = int(rng.integers(0, len(samples[inst])))
            cand_idx = int(rng.integers(0, len(samples[inst][evt_idx])))
            ctx, feat, keep, gb = samples[inst][evt_idx][cand_idx]
            ctxs.append(ctx)
            feats.append(feat)
            keeps.append(keep)
            ys.append(gb / s)
        X = jnp.asarray(np.concatenate([np.stack(ctxs), np.stack(feats)], axis=-1))
        XK = jnp.asarray(np.concatenate([np.stack(ctxs), np.stack(keeps)], axis=-1))
        yb = jnp.asarray(np.asarray(ys, np.float32))
        loss, params, opt_state = train_step(params, opt_state, X, XK, yb)
        if (step + 1) % 200 == 0:
            print(f"  step {step+1} loss={float(loss):.4f}", flush=True)

    with open(os.path.join(args.out, 'model.ckpt'), 'wb') as f:
        pickle.dump({'params': params, 's': s, 'in_dim': in_dim, 'seed': args.seed,
                     'representation': args.representation}, f)
    json.dump({'in_dim': in_dim, 's': s, 'n_candidates': int(n_cands), 'seed': args.seed,
               'representation': args.representation, 'n_instances': n_inst,
               'n_events': n_events, 'num_steps': args.num_steps,
               'batch_size': args.batch_size,
               'data': os.path.abspath(args.data)},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/model.ckpt")


if __name__ == '__main__':
    main()
