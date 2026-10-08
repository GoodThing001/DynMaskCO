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
                             NULL_TOKEN, PAD_TOKEN, VOCAB_SIZE, save_model)
TRAIN_SEED = 20260925


def make_batch(slices, idxs, m_max, partial=False, p_max=0.8, rng=None):
    """P0 修复（2026-09-30）：批次补齐位用 PAD_TOKEN（与 MASK/NULL 分离）；
    损失只监督 X==MASK 位。I1：数量标签 N = min(len(tgt), m_max)；时钟桶 CB。
    I2（2026-09-30，partial=True）：部分掩码训练——每样本随机保留 n_keep∈[0,n-1]
    个未来 token 作为条件（可见、不监督），剩余 n−n_keep 个放 MASK 槽重建；
    MASK 槽数 = m_max − n_keep（可变，PAD 补齐），与部署迭代条件
    [vis + kept][MASK×rest] 一致；数量标签仍为完整未来数 n。"""
    rng = np.random.default_rng(0) if rng is None else rng
    vis_list, tgt_list, n_list, cb_list = [], [], [], []
    for i in idxs:
        vis, tgt, n, cb = slices[i]
        tgt = list(tgt[:m_max])
        vis_list.append(vis[:m_max])
        n_list.append(min(int(n), m_max))
        cb_list.append(int(cb))
        if partial and len(tgt) > 0:
            n_keep = int(rng.integers(0, max(int(len(tgt) * p_max), 1)))  # ≥0 保留、≥1 重建
            kept_idx = rng.choice(len(tgt), size=n_keep, replace=False)
            kept_set = set(int(x) for x in kept_idx)
            kept = [t for j, t in enumerate(tgt) if j in kept_set]
            regen = [t for j, t in enumerate(tgt) if j not in kept_set]
            tgt_list.append((kept, regen))
        else:
            tgt_list.append(([], tgt))
    # 固定批长（2026-09-30 修复编译风暴）：L 恒为 2*m_max（vis≤m_max 截断 + m_max MASK/保留槽），
    # jax/triton 每新形状重编译是 pretrained 臂 55s/步卡死的根因——恒定 L 只编译一次。
    L = 2 * m_max
    X = np.full((len(idxs), L), PAD_TOKEN, dtype=np.int32)
    Y = np.full((len(idxs), L), NULL_TOKEN, dtype=np.int32)
    for b, (vis, (kept, regen)) in enumerate(zip(vis_list, tgt_list)):
        X[b, :len(vis)] = vis
        X[b, len(vis):len(vis) + len(kept)] = kept        # I2：保留未来 = 条件（可见）
        n_mask = m_max - len(kept)
        X[b, len(vis) + len(kept):len(vis) + len(kept) + n_mask] = MASK_TOKEN
        s = len(vis) + len(kept)
        Y[b, s:s + len(regen)] = regen                    # 仅 MASK 槽有重建目标
    N = np.array(n_list, dtype=np.int32)
    CB = np.array(cb_list, dtype=np.int32)
    return X, Y, N, CB


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=200)
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    default="random",
                    help="random_cvrp=同 CVRP encoder 架构随机权重（同冻结策略，同架构消融）")
    ap.add_argument("--cvrp-ckpt", default=None,
                    help="预训练/同架构随机臂的 cvrp100.ckpt 路径（服务器）")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partial-mask", action="store_true",
                    help="I2：部分掩码训练（随机保留 n_keep∈[0,n-1] 未来 token 再重建其余）")
    ap.add_argument("--p-max", type=float, default=0.8,
                    help="I2：保留比例上限（保留数在 [0, floor(n*p_max)) 内均匀随机）")
    ap.add_argument("--save-every", type=int, default=0,
                    help="泛化诊断（2026-10-01）：每 N 步保存 ckpt_step_<it>.bin 到 out；0=关闭")
    ap.add_argument("--train-days", type=int, default=None,
                    help="只用前 train-days 个训练日（A3 内部 CV 切分用；默认=全部）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    hist = build_history(generate_dataset(args.train_instances, args.n_orders,
                                          TRAIN_SEED))
    if args.train_days is not None:
        hist = hist[:args.train_days]
    slices = make_training_slices(hist, cuts_per_day=4, rng=np.random.default_rng(args.seed))
    print("slices:", len(slices), "arm:", args.arm)

    cvrp_model = None
    if args.arm in ("pretrained", "random_cvrp"):
        if not args.cvrp_ckpt:
            raise SystemExit("%s arm requires --cvrp-ckpt" % args.arm)
        from mpre import load_cvrp_model
        cvrp_model, cfg, _step = load_cvrp_model(args.cvrp_ckpt)
        if args.arm == "random_cvrp":
            # 路线图 09-30：同架构随机初始化消融——同 CVRPModelConfig（含 512 维/层数/
            # dtype）随机权重、同 _cvrp 冻结谓词（训练期间不更新），只差权重来源。
            # 2026-09-30 修复：不可 `from CVRPModel import CVRPModel`（包相对导入失败）——
            # 经 config 自带 construct_model + dataclasses.replace 注入种子。
            from dataclasses import replace
            cvrp_model = replace(cfg, rngs=int(args.seed)).construct_model()
            print("random_cvrp: 同架构随机初始化（seed=%d，冻结同 pretrained）" % args.seed)
        else:
            print("loaded cvrp ckpt")

    rngs = nnx.Rngs(args.seed)
    model = MaskCOScenarioModel(dim=args.dim, arm=args.arm, cvrp_model=cvrp_model, rngs=rngs)

    def _flat_params():
        fs = nnx.state(model, nnx.Param).flat_state()
        return {k: v for k, v in zip(fs.paths, fs.leaves)}

    def _is_frozen(path):
        return any(str(x).startswith('_cvrp') for x in path)

    def _cvrp_hash():
        # 2026-09-30 修复：flat_state 叶为 VariableState——np.asarray(VariableState)
        # 得 0 维对象数组，tobytes()=不稳定 pickle → A-07 恒 False 伪告警；取 v.value
        # 真实数组哈希（逐叶值差实测 0.0 证实冻结本身正确）。trainable 仍须用
        # VariableState 原样（optax 树节点类型要求）。
        import hashlib
        h = hashlib.sha256()
        for k, v in _flat_params().items():
            if _is_frozen(k):
                h.update(np.asarray(v.value).tobytes())
        return h.hexdigest()

    cvrp_hash_before = _cvrp_hash()

    # A-07（2026-09-28）：优化器只更新非冻结（_cvrp 子树之外）参数——
    # 冻结分支不参与梯度更新也不参与 AdamW weight decay；start/end 保存其参数 hash。
    trainable = nnx.State.from_flat_path({
        k: v for k, v in _flat_params().items() if not _is_frozen(k)})
    tx = optax.adamw(args.lr)
    opt_state = tx.init(trainable)
    trainable_paths = sorted(str(k) for k in trainable.flat_state().paths)
    n_trainable = len(trainable_paths)

    def loss_fn(model, X, Y, N, CB):
        tok_logits, cnt_logits = model(X, clock_bin=CB)
        mask = X == MASK_TOKEN
        # 槽 token CE：只监督 MASK 槽（Y 在 MASK 槽 = 目标 token / NULL 填充）
        nll = optax.softmax_cross_entropy_with_integer_labels(tok_logits, Y)
        n = jnp.maximum(mask.sum(), 1.0)
        loss_tok = jnp.sum(jnp.where(mask, nll, 0.0)) / n
        # I1 数量头 CE：min(#未来, m_max)
        loss_cnt = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            cnt_logits, N.astype(jnp.int32)))
        return loss_tok + loss_cnt

    @nnx.jit(donate_argnums=())
    def step(model, opt_state, trainable, X, Y, N, CB):
        loss, grads = nnx.value_and_grad(loss_fn)(model, X, Y, N, CB)
        gfs = grads.flat_state()
        gd = {k: v for k, v in zip(gfs.paths, gfs.leaves)}
        tfs = trainable.flat_state()
        tk = {k for k in tfs.paths}
        grads = nnx.State.from_flat_path({k: v for k, v in gd.items() if k in tk})
        updates, opt_state = tx.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        # 2026-10-01 关键修复：不得在 jit 内 nnx.update(model, ...)——jit 会捐献模型
        # 参数，更新不传回外层 → 保存的是未训练权重（此前全部 checkpoint 均为 init）。
        return loss, opt_state, trainable

    rng = np.random.default_rng(args.seed)
    losses = []
    os.makedirs(args.out, exist_ok=True)   # 提前建目录（--save-every 中途写 checkpoint 依赖）
    for it in range(args.steps):
        idxs = rng.integers(0, len(slices), args.batch_size)
        X, Y, N, CB = make_batch(slices, list(idxs), args.m_max,
                                 partial=args.partial_mask, p_max=args.p_max, rng=rng)
        loss, opt_state, trainable = step(model, opt_state, trainable,
                                          jnp.asarray(X), jnp.asarray(Y),
                                          jnp.asarray(N), jnp.asarray(CB))
        nnx.update(model, trainable)   # 外层应用更新（模型不被捐献）
        losses.append(float(loss))
        if it % 200 == 0 or it == args.steps - 1:
            print(f"step {it}: loss={float(loss):.4f} (last50={np.mean(losses[-50:]):.4f})")
        if args.save_every > 0 and ((it + 1) % args.save_every == 0
                                    or it == args.steps - 1):
            save_model(model, os.path.join(args.out, "ckpt_step_%d.bin" % (it + 1)))

    cvrp_hash_after = _cvrp_hash()
    print(f"A-07: cvrp hash before={cvrp_hash_before[:16]}… after={cvrp_hash_after[:16]}… "
          f"unchanged={cvrp_hash_before == cvrp_hash_after}; "
          f"n_trainable={n_trainable}")

    os.makedirs(args.out, exist_ok=True)
    save_model(model, os.path.join(args.out, "model.bin"))
    # 模型代码世代封存（2026-09-30）：checkpoint 参数集与模型源码绑定——加载端必须
    # 校验 hash，防止旧权重静默加载进新模板（flax 缺键会报错，但这里显式防世代混用）。
    import hashlib
    src_hashes = {}
    for rel in ('scripts/models/maskco_scenario.py',
                'scripts/training/train_maskco_scenario.py'):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), *rel.split('/'))
        with open(p, 'rb') as fh:
            src_hashes[rel] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "vocab_size": VOCAB_SIZE,
                   "final_loss": float(losses[-1]),
                   "loss_trace": [round(x, 4) for x in losses[::max(1, len(losses) // 20)]],
                   "train_seed": TRAIN_SEED,
                   "frozen_cvrp_hash_before": cvrp_hash_before,
                   "frozen_cvrp_hash_after": cvrp_hash_after,
                   "frozen_unchanged": bool(cvrp_hash_before == cvrp_hash_after),
                   "n_trainable_params": int(n_trainable),
                   "model_source_sha256": src_hashes,
                   "trainable_paths": trainable_paths}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
