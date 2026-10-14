"""event_plan_v1 E2：训练完整计划评价器（42/43/44 三 seed）。

模型：小 MLP 输入 [context(25) + plan_delta(25*4)] → 分数 f(z_e, P)。
损失：Huber((f(P) - f(P0)) - gB/s)，s 仅在 TRAIN 有效标签上估计；实例/事件平衡。
gB_hat = s * (f(P) - f(P0))。

用法（服务器）：
    python scripts/training/train_event_plan.py \
        --data results/m0_scale/event_plan_export/data.json \
        --num-steps 2000 --batch-size 8 --seed 42 --out results/m0_scale/event_plan_model_s42
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

from event_plan_candidate import dict_to_plans
from event_plan_features import extract_plan_delta, PLAN_VEH_DIM

CONTEXT_DIM = 25
PLAN_DIM = 25 * PLAN_VEH_DIM  # 100


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


def build_inputs(data):
    """构建 (context, plan_delta, keep_plan_delta, gB) 数组（每条候选一行）。"""
    contexts, deltas, keeps, gbs = [], [], [], []
    for row in data['rows']:
        ctx = np.asarray(row['context'], np.float32)
        keep_delta = np.asarray(row.get('keep_plan_delta'), np.float32)
        for c in row['cands']:
            contexts.append(ctx)
            deltas.append(np.asarray(c['plan_delta'], np.float32))
            keeps.append(keep_delta)
            gbs.append(float(c['gB']))
    return (np.stack(contexts), np.stack(deltas), np.stack(keeps),
            np.asarray(gbs, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--num-steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        data = json.load(f)
    contexts, deltas, keeps, gbs = build_inputs(data)
    C = contexts.shape[0]
    s = max(float(np.std(gbs)), 1e-6)
    y = gbs / s
    print(f"n_candidates={C} s={s:.4f}", flush=True)

    in_dim = CONTEXT_DIM + PLAN_DIM
    head = PlanEvalMLP(in_dim, rngs=args.seed)
    gd, params = nnx.split(head)
    tx = optax.adamw(1e-3, weight_decay=1e-2)
    opt_state = tx.init(params)

    X = jnp.concatenate([jnp.asarray(contexts), jnp.asarray(deltas)], axis=-1)
    XK = jnp.concatenate([jnp.asarray(contexts), jnp.asarray(keeps)], axis=-1)
    yy = jnp.asarray(y)

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
    idxs = np.arange(C)
    for step in range(args.num_steps):
        b = rng.choice(idxs, size=min(args.batch_size, C), replace=True)
        loss, params, opt_state = train_step(params, opt_state, X[b], XK[b], yy[b])
        if (step + 1) % 200 == 0:
            print(f"  step {step+1} loss={float(loss):.4f}", flush=True)

    with open(os.path.join(args.out, 'model.ckpt'), 'wb') as f:
        pickle.dump({'params': params, 's': s, 'in_dim': in_dim, 'seed': args.seed}, f)
    json.dump({'in_dim': in_dim, 's': s, 'n_candidates': int(C), 'seed': args.seed},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/model.ckpt")


if __name__ == '__main__':
    main()
