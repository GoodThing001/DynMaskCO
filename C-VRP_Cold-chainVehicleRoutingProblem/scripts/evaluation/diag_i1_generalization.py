# -*- coding: utf-8 -*-
"""A-v1 步骤 3 · 计数 NLL 泛化曲线（2026-10-01，定位留出泛化失利）。

同一口径（固定形状前向 + 相同切片分布）下，对每个中间/最终 checkpoint 计算：
  - 训练日切片（模型见过）的计数 NLL / 偏差 / Pearson；
  - 留出日切片（模型未见）的同三项；
  - 三非学习基线（未平滑边际 / α=1 平滑 / 时钟桶条件）供对照。
输出 {out}/generalization_curve.json：step -> 两侧指标。仅用训练历史日，不触碰开发/测试种子。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp
from flax import nnx

from run_exp_reserve import generate_dataset
from scenario_saa import build_history
from maskco_scenario import (MaskCOScenarioModel, load_model, check_model_generation,
                             MASK_TOKEN, PAD_TOKEN, M_MAX, order_to_token,
                             clock_bin_of, make_training_slices)

TRAIN_SEED = 20260925
_NLL_FLOOR = 1e-12


def _softmax(x):
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


def _slice_with_clocks(history_days, cuts_per_day, rng):
    out = []
    for day_idx, day in enumerate(history_days):
        if not day:
            continue
        day_sorted = sorted(day, key=lambda o: o.reveal)
        rev_times = [min(float(o.reveal), 15.99) for o in day_sorted]
        n_cuts = min(cuts_per_day, len(rev_times))
        cuts = sorted(rng.choice(rev_times, size=n_cuts, replace=False))
        for cut in cuts:
            vis = [order_to_token(o) for o in day_sorted if o.reveal <= cut + 1e-6]
            tgt = [order_to_token(o) for o in day_sorted if o.reveal > cut + 1e-6]
            if vis and tgt:
                out.append((int(day_idx), float(cut), vis, tgt, len(tgt),
                            clock_bin_of(float(cut))))
    return out


def _eval_slices(model, slices, m_max):
    actuals, cb_list = [], []
    nll_acc, bias_acc = [], []
    probs_all = []
    for day_idx, clock, vis, tgt, n, cb in slices:
        vis_c = vis[:m_max]
        toks = np.full(2 * m_max, PAD_TOKEN, dtype=np.int32)
        toks[:len(vis_c)] = vis_c
        toks[len(vis_c):len(vis_c) + m_max] = MASK_TOKEN
        tl, cl = model(jnp.asarray(toks)[None],
                       clock_bin=jnp.asarray([cb], dtype=jnp.int32))
        p = _softmax(np.asarray(cl[0][:m_max + 1]))
        a = min(int(n), m_max)
        nll_acc.append(-np.log(max(p[a], _NLL_FLOOR)))
        k = np.arange(m_max + 1, dtype=np.float64)
        bias_acc.append(float((p * k).sum() - a))
        actuals.append(a)
        cb_list.append(int(cb))
        probs_all.append(p)
    probs = np.stack(probs_all)
    mean_pred = (probs * np.arange(m_max + 1)).sum(1)
    pearson = (float(np.corrcoef(mean_pred, actuals)[0, 1])
               if len(actuals) > 1 and np.std(mean_pred) > 0 and np.std(actuals) > 0
               else None)
    return {"nll": float(np.mean(nll_acc)), "bias": float(np.mean(bias_acc)),
            "pearson": pearson}


def _baselines(train_slices, held_slices, m_max):
    marg = np.zeros(m_max + 1)
    per_cb = {}
    for _d, _c, _v, _t, n, cb in train_slices:
        a = min(int(n), m_max)
        marg[a] += 1.0
        per_cb.setdefault(int(cb), np.zeros(m_max + 1))[a] += 1.0
    p_raw = marg / max(marg.sum(), 1e-12)
    p_sm1 = (marg + 1.0) / (marg + 1.0).sum()
    p_cb = {cb: (h + 1.0) / (h + 1.0).sum() for cb, h in per_cb.items()}
    out = {}
    for name, slices in (('train', train_slices), ('held', held_slices)):
        nll_raw, nll_sm1, nll_cb = [], [], []
        for _d, _c, _v, _t, n, cb in slices:
            a = min(int(n), m_max)
            nll_raw.append(-np.log(max(p_raw[a], _NLL_FLOOR)))
            nll_sm1.append(-np.log(max(p_sm1[a], _NLL_FLOOR)))
            nll_cb.append(-np.log(max(p_cb.get(int(cb), p_sm1)[a], _NLL_FLOOR)))
        out[name] = {"nll_marginal_raw": float(np.mean(nll_raw)),
                     "nll_marginal_sm1": float(np.mean(nll_sm1)),
                     "nll_clock_cond_sm1": float(np.mean(nll_cb))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="训练输出目录（含 model.bin 与 ckpt_step_*.bin）")
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=None)
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--train-days", type=int, default=160)
    ap.add_argument("--history-days", type=int, default=200)
    ap.add_argument("--held-start", type=int, default=160)
    ap.add_argument("--cuts-per-day", type=int, default=4)
    ap.add_argument("--slice-seed", type=int, default=42)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--allow-code-mismatch", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m_max = args.m_max
    cfg_path = os.path.join(args.ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        m_max = int(cfg["args"].get("m_max", M_MAX))
        args.dim = int(cfg["args"].get("dim", args.dim))
        args.arm = cfg["args"].get("arm", args.arm)
    if m_max is None:
        m_max = M_MAX

    all_days = build_history(generate_dataset(args.history_days, args.n_orders,
                                              TRAIN_SEED))
    train_slices = _slice_with_clocks(all_days[:args.train_days], args.cuts_per_day,
                                      np.random.default_rng(args.slice_seed))
    held_slices = _slice_with_clocks(all_days[args.held_start:], args.cuts_per_day,
                                     np.random.default_rng(args.slice_seed + 1))
    print("train slices:", len(train_slices), "held slices:", len(held_slices))

    cvrp = None
    if args.arm in ("pretrained", "random_cvrp"):
        if not args.cvrp_ckpt:
            raise SystemExit("%s arm requires --cvrp-ckpt" % args.arm)
        from mpre import load_cvrp_model
        cvrp, cfg, _step = load_cvrp_model(args.cvrp_ckpt)
        if args.arm == "random_cvrp":
            from dataclasses import replace
            cvrp = replace(cfg, rngs=42).construct_model()

    def _fresh():
        return MaskCOScenarioModel(dim=args.dim, arm=args.arm, cvrp_model=cvrp,
                                   rngs=nnx.Rngs(0))

    bins = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step_*.bin")))
    bins.append(os.path.join(args.ckpt_dir, "model.bin"))
    curve = []
    for b in bins:
        step = (int(os.path.basename(b).split('_')[-1].split('.')[0])
                if 'ckpt_step_' in b else 'final')
        m = load_model(_fresh(), b)
        train_m = _eval_slices(m, train_slices, m_max)
        held_m = _eval_slices(m, held_slices, m_max)
        curve.append({"step": step, "train": train_m, "held": held_m})

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "n_train_slices": len(train_slices),
        "n_held_slices": len(held_slices),
        "baselines": _baselines(train_slices, held_slices, m_max),
        "curve": curve,
        "note": ("同一计数 NLL 口径（非训练 CE）在训练日 vs 留出日随步数变化；"
                 "train loss 为 token CE+计数 CE 之和，不可与计数 NLL 直接相减。"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generalization_curve.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
