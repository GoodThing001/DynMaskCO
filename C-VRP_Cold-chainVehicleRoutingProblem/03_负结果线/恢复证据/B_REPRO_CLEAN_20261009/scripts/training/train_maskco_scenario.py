"""A-v1 步骤 3：训练 MaskCO 条件场景生成器（掩码重建 CE，Flax NNX）。

数据 = 训练历史日（seed 20260925）切片：(可见 token 序列, 掩码未来 token 目标)。
监督 = 序列 [vis][MASK×m_max] 的 MASK 槽上 softmax CE（NULL 为合法输出，计数隐式建模）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_reserve import generate_dataset
from scenario_saa import build_history
from maskco_scenario import (MaskCOScenarioModel, make_training_slices, MASK_TOKEN,
                             NULL_TOKEN, VOCAB_SIZE, save_model)
TRAIN_SEED = 20260925


def make_batch(slices, idxs, m_max):
    vis_list, tgt_list = [], []
    for i in idxs:
        vis, tgt = slices[i]
        tgt = tgt[:m_max]
        vis_list.append(vis[:m_max])
        tgt_list.append(tgt)
    L = max(len(v) + m_max for v in vis_list)
    X = np.full((len(idxs), L), MASK_TOKEN, dtype=np.int32)
    Y = np.full((len(idxs), L), NULL_TOKEN, dtype=np.int32)  # 可见位置填 NULL（忽略）
    for b, (vis, tgt) in enumerate(zip(vis_list, tgt_list)):
        X[b, :len(vis)] = vis
        Y[b, len(vis):len(vis) + len(tgt)] = tgt
        Y[b, len(vis) + len(tgt):] = NULL_TOKEN
    return X, Y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=200)
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained"], default="random")
    ap.add_argument("--cvrp-ckpt", default=None,
                    help="预训练臂的 cvrp100.ckpt 路径（服务器）")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    hist = build_history(generate_dataset(args.train_instances, args.n_orders, TRAIN_SEED))
    slices = make_training_slices(hist, cuts_per_day=4, rng=np.random.default_rng(args.seed))
    print("slices:", len(slices), "arm:", args.arm)

    cvrp_model = None
    if args.arm == "pretrained":
        if not args.cvrp_ckpt:
            raise SystemExit("pretrained arm requires --cvrp-ckpt")
        from mpre import load_cvrp_model
        cvrp_model, _cfg, _step = load_cvrp_model(args.cvrp_ckpt)
        print("loaded cvrp ckpt")

    rngs = nnx.Rngs(args.seed)
    model = MaskCOScenarioModel(dim=args.dim, arm=args.arm, cvrp_model=cvrp_model, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(args.lr))

    def loss_fn(model, X, Y):
        logits = model(X)
        mask = X == MASK_TOKEN
        # 只监督 MASK 槽（Y 在 MASK 槽 = 目标 token / NULL 填充）
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
        n = jnp.maximum(mask.sum(), 1.0)
        return jnp.sum(jnp.where(mask, nll, 0.0)) / n

    @nnx.jit
    def step(model, optimizer, X, Y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, X, Y)
        optimizer.update(grads)
        return loss

    rng = np.random.default_rng(args.seed)
    losses = []
    for it in range(args.steps):
        idxs = rng.integers(0, len(slices), args.batch_size)
        X, Y = make_batch(slices, list(idxs), args.m_max)
        loss = step(model, optimizer, jnp.asarray(X), jnp.asarray(Y))
        losses.append(float(loss))
        if it % 200 == 0 or it == args.steps - 1:
            print(f"step {it}: loss={float(loss):.4f} (last50={np.mean(losses[-50:]):.4f})")

    os.makedirs(args.out, exist_ok=True)
    save_model(model, os.path.join(args.out, "model.bin"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "vocab_size": VOCAB_SIZE,
                   "final_loss": float(losses[-1]),
                   "loss_trace": [round(x, 4) for x in losses[::max(1, len(losses) // 20)]],
                   "train_seed": TRAIN_SEED}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
