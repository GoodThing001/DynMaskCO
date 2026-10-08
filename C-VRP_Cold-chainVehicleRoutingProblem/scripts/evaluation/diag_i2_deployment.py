# -*- coding: utf-8 -*-
"""A-v1 步骤 3 · I2 部署采样诊断（2026-10-01，目标②收尾）。

与 diag_i1_calibration 的离线头指标互补，本脚本测**部署口径**：
  1) 首轮全掩码采样：用 MaskCOScenarioSampler.sample() 生成 K 个完整场景，
     比较采样场景数量 vs 真实未来数（偏差/MAE/RMSE/均值，及与数量头 E[N] 的差）；
  2) 单步 vs 迭代：同一模型 --iterative 与单步的采样数量统计差异；
  3) 训练/部署条件分布差：第二（重掩码）轮条件里，部署保留的是**模型生成** token，
     训练保留的是**真实未来** token——对每个切片比较 [vis+真实保留] vs [vis+生成保留]
     条件下的槽分布（总变差），量化该 gap。
只读训练历史日切片（默认 [160,200) 留出），不触碰开发/论文测试种子。
"""
from __future__ import annotations

import argparse
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
from scenario_saa import build_history, VisibleSnapshot
from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools,
                             load_model, check_model_generation, M_MAX,
                             order_to_token, clock_bin_of, clock_support_mask,
                             keep_mask_by_prob, _cond_probs, _norm_probs)

TRAIN_SEED = 20260925


def _slice_with_orders(history_days, cuts_per_day, rng):
    """[(day_idx, clock, vis_tokens, tgt_tokens, n, cb, vis_orders)]。"""
    out = []
    for day_idx, day in enumerate(history_days):
        if not day:
            continue
        day_sorted = sorted(day, key=lambda o: o.reveal)
        rev_times = [min(float(o.reveal), 15.99) for o in day_sorted]
        n_cuts = min(cuts_per_day, len(rev_times))
        cuts = sorted(rng.choice(rev_times, size=n_cuts, replace=False))
        for cut in cuts:
            vis_o = [o for o in day_sorted if o.reveal <= cut + 1e-6]
            tgt = [order_to_token(o) for o in day_sorted if o.reveal > cut + 1e-6]
            vis = [order_to_token(o) for o in vis_o]
            if vis and tgt:
                out.append((int(day_idx), float(cut), vis, tgt, len(tgt),
                            clock_bin_of(float(cut)), vis_o))
    return out


def _snapshot(s, m_max):
    day_idx, clock, vis, tgt, n, cb, vis_o = s
    return VisibleSnapshot(clock=clock, orders=tuple(vis_o),
                           accepted=frozenset(), rejected=frozenset(),
                           vehicles=(), energy_used=0.0, budget=1e9,
                           booking_horizon=16.0, capacity=50.0)


def _gap_metrics(s, sampler, pools, m_max, rng, remask_frac=0.5):
    """真实 vs 生成保留条件的槽分布总变差（量化训练/部署条件 gap）。"""
    day_idx, clock, vis, tgt, n, cb, vis_o = s
    vis = vis[:m_max]
    support = clock_support_mask(float(clock))
    tl, cl = sampler._forward_masks(vis, clock_bin=cb)
    p1 = _cond_probs(tl, support)                        # (m_max, VOCAB)
    idx1 = rng.multinomial(1, p1).argmax(-1)             # 首轮每槽采样
    # 用真实未来数 n 决定保留槽数（两种口径都取 min(n, m_max)）
    n_use = min(int(n), m_max)
    probs1 = [float(p1[j, int(idx1[j])]) for j in range(n_use)]
    keep = keep_mask_by_prob(probs1, remask_frac, rng)
    kept_gen = [int(idx1[j]) for j in range(n_use) if keep[j]]
    kept_real = [int(tgt[j % len(tgt)]) for j in range(len(kept_gen))] if tgt else []
    n_remask = n_use - len(kept_gen)
    if n_remask <= 0:
        return None
    tl_g, _ = sampler._forward_masks(vis + kept_gen, clock_bin=cb,
                                     n_mask=m_max - len(kept_gen))
    tl_r, _ = sampler._forward_masks(vis + kept_real, clock_bin=cb,
                                     n_mask=m_max - len(kept_real))
    p_g = _cond_probs(tl_g[:n_remask], support)
    p_r = _cond_probs(tl_r[:n_remask], support)
    tv = float(np.abs(p_g - p_r).sum(axis=-1).mean()) / 2.0
    return {'n_kept': int(len(kept_gen)), 'n_remask': int(n_remask), 'tv_gap': tv}


def _softmax(x):
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", required=True)
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=None)
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--history-days", type=int, default=200)
    ap.add_argument("--day-start", type=int, default=160)
    ap.add_argument("--day-end", type=int, default=200)
    ap.add_argument("--cuts-per-day", type=int, default=4)
    ap.add_argument("--slice-seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterative", action="store_true")
    ap.add_argument("--remask-frac", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--allow-code-mismatch", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m_max = args.m_max
    gen = check_model_generation(args.model_bin, allow_mismatch=args.allow_code_mismatch)
    print("model generation check:", gen)
    if m_max is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.model_bin)),
                                "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            m_max = int(cfg["args"].get("m_max", M_MAX))
            args.dim = int(cfg["args"].get("dim", args.dim))
            args.arm = cfg["args"].get("arm", args.arm)

    all_days = build_history(generate_dataset(args.history_days, args.n_orders,
                                              TRAIN_SEED))
    rng_slices = np.random.default_rng(args.slice_seed)
    slices = _slice_with_orders(all_days[args.day_start:args.day_end],
                                args.cuts_per_day, rng_slices)
    print("slices:", len(slices))

    cvrp = None
    if args.arm in ("pretrained", "random_cvrp"):
        if not args.cvrp_ckpt:
            raise SystemExit("%s arm requires --cvrp-ckpt" % args.arm)
        from mpre import load_cvrp_model
        cvrp, cfg, _step = load_cvrp_model(args.cvrp_ckpt)
        if args.arm == "random_cvrp":
            from dataclasses import replace
            cvrp = replace(cfg, rngs=42).construct_model()
    model = MaskCOScenarioModel(dim=args.dim, arm=args.arm, cvrp_model=cvrp,
                                rngs=nnx.Rngs(0))
    model = load_model(model, args.model_bin)
    pools = HistoryPools(all_days[:args.day_start])
    sampler = MaskCOScenarioSampler(model, pools, m_max=m_max,
                                    iterative=args.iterative,
                                    remask_frac=args.remask_frac,
                                    rounds=args.rounds)

    rng = np.random.default_rng(args.slice_seed + 7)
    bias, mae, rmse = [], [], []
    cnt_mean = []          # 采样场景平均数量
    cnt_head_mean = []     # 数量头 E[N]
    for s in slices:
        snap = _snapshot(s, m_max)
        scens = sampler.sample(snap, rng, args.K)
        pred = np.mean([len(x) for x in scens])
        n = min(int(s[4]), m_max)
        bias.append(pred - n)
        mae.append(abs(pred - n))
        rmse.append((pred - n) ** 2)
        cnt_mean.append(pred)
        # 数量头 E[N]（与采样口径对照）
        vis = s[2][:m_max]
        tl, cl = sampler._forward_masks(vis, clock_bin=s[5])
        p_cnt = _softmax(cl)
        cnt_head_mean.append(float((p_cnt * np.arange(m_max + 1)).sum()))

    gap_metrics = []
    for s in slices:
        g = _gap_metrics(s, sampler, pools, m_max,
                         np.random.default_rng(args.slice_seed + 13))
        if g:
            gap_metrics.append(g)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "model_generation_check": gen,
        "n_slices": len(slices),
        "deployment_sampling_count": {
            "bias_mean": float(np.mean(bias)), "mae": float(np.mean(mae)),
            "rmse": float(np.sqrt(np.mean(rmse))),
            "sampled_mean": float(np.mean(cnt_mean)),
            "count_head_mean": float(np.mean(cnt_head_mean)),
            "note": "采样场景数量 vs 真实未来数（K=%d；部署口径含边界桶截断解码）" % args.K,
        },
        "cond_gap_real_vs_generated": {
            "n_slices_with_gap": int(len(gap_metrics)),
            "tv_gap_mean": (float(np.mean([g['tv_gap'] for g in gap_metrics]))
                            if gap_metrics else None),
            "tv_gap_ci": ([float(np.percentile([g['tv_gap'] for g in gap_metrics], 2.5)),
                           float(np.percentile([g['tv_gap'] for g in gap_metrics], 97.5))]
                          if gap_metrics else None),
            "note": "第二（重掩码）轮条件：部署保留=模型生成 token vs 训练保留=真实未来 token，"
                    "槽条件分布总变差（0=无 gap）",
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "deployment_sampling.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
