# -*- coding: utf-8 -*-
"""A3 正式步数选择（预声明规则，2026-10-01）：只用训练历史内部切分。

train=0..128、valid=128..160（固定切分/切片 seed）。对 train 上每 50 步的 checkpoint
计算 valid 上同口径计数 NLL，选**首个极小点**（窗口 ≥200 步防噪）作为 s*。
本脚本与结果只登记一次；s* 不随开发/在线结果回调。
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
from maskco_scenario import (MaskCOScenarioModel, load_model, MASK_TOKEN, PAD_TOKEN,
                             M_MAX, order_to_token, clock_bin_of)

TRAIN_SEED = 20260925
_NLL_FLOOR = 1e-12


def _softmax(x):
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


def _slices(history_days, cuts_per_day, rng):
    out = []
    for day_idx, day in enumerate(history_days):
        if not day:
            continue
        day_sorted = sorted(day, key=lambda o: o.reveal)
        rev_times = [min(float(o.reveal), 15.99) for o in day_sorted]
        n_cuts = min(cuts_per_day, len(rev_times))
        for cut in sorted(rng.choice(rev_times, size=n_cuts, replace=False)):
            vis = [order_to_token(o) for o in day_sorted if o.reveal <= cut + 1e-6]
            tgt = [order_to_token(o) for o in day_sorted if o.reveal > cut + 1e-6]
            if vis and tgt:
                out.append((int(day_idx), float(cut), vis, tgt, len(tgt),
                            clock_bin_of(float(cut))))
    return out


def _nll(model, slices, m_max):
    acc = []
    for _d, _c, vis, _t, n, cb in slices:
        vis_c = vis[:m_max]
        toks = np.full(2 * m_max, PAD_TOKEN, dtype=np.int32)
        toks[:len(vis_c)] = vis_c
        toks[len(vis_c):len(vis_c) + m_max] = MASK_TOKEN
        tl, cl = model(jnp.asarray(toks)[None],
                       clock_bin=jnp.asarray([cb], dtype=jnp.int32))
        p = _softmax(np.asarray(cl[0][:m_max + 1]))
        a = min(int(n), m_max)
        acc.append(-np.log(max(p[a], _NLL_FLOOR)))
    return float(np.mean(acc))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=None)
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--history-days", type=int, default=200)
    ap.add_argument("--valid-start", type=int, default=128)
    ap.add_argument("--valid-end", type=int, default=160)
    ap.add_argument("--cuts-per-day", type=int, default=4)
    ap.add_argument("--slice-seed", type=int, default=42)
    ap.add_argument("--min-window", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--allow-code-mismatch", action="store_true")
    ap.add_argument("--reuse-curve", default=None,
                    help="从既有 step_selection.json 复用曲线重算 s*（不重评估）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m_max = args.m_max
    if args.reuse_curve:
        old = json.load(open(args.reuse_curve, encoding="utf-8"))
        curve = []
        for c in old["curve"]:
            s = c["step"]
            curve.append((999999 if s == "final" else int(s),
                          float(c["valid_count_nll"])))
        curve.sort()
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.reuse_curve)),
                                "..", "config.json")
    else:
        cfg_path = os.path.join(args.ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        m_max = int(cfg["args"].get("m_max", M_MAX))
        args.dim = int(cfg["args"].get("dim", args.dim))
        args.arm = cfg["args"].get("arm", args.arm)
    if m_max is None:
        m_max = M_MAX

    if args.reuse_curve:
        bins = []
        valid_slices = []
    else:
        all_days = build_history(generate_dataset(args.history_days, args.n_orders,
                                                  TRAIN_SEED))
        valid_slices = _slices(all_days[args.valid_start:args.valid_end],
                               args.cuts_per_day, np.random.default_rng(args.slice_seed))
        print("valid slices:", len(valid_slices))

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
                    if 'ckpt_step_' in b else 999999)
            m = load_model(_fresh(), b)
            curve.append((step, _nll(m, valid_slices, m_max)))
        curve.sort()
    # 首个极小点（窗口 ≥ min_window 步防噪；2026-10-01 修正：前向窗口——
    # 首个 s 使 NLL(s) ≤ min NLL([s, s+min_window])，避免三样本局部极小误选）
    s_star, nll_star = None, None
    for i, (step, nll) in enumerate(curve):
        future = [v for s2, v in curve if step <= s2 <= step + args.min_window]
        if nll <= min(future) + 1e-9:
            s_star, nll_star = step, nll
            break
    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "curve": [{"step": s, "valid_count_nll": round(n, 4)} for s, n in curve],
        "s_star": s_star,
        "nll_at_s_star": nll_star,
        "rule": "首个极小点（min_window=%d），只用训练历史内部 valid 切分；登记后不回调"
                % args.min_window,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "step_selection.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
