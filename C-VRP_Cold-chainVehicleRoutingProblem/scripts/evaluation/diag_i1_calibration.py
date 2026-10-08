# -*- coding: utf-8 -*-
"""A-v1 步骤 3 · I1 联合校准诊断（纯离线，只用训练历史日 20260925）。

目标（路线图 I1「数量校准、时间/温区/空间/需求联合统计」）：
- 数量头校准：P(N_future | visible) 的 PIT/偏差/MAE/RMSE/秩相关；
- **多基线 NLL 增益（2026-10-01 用户复核修订）**：
  ①边际原始（训练日切片经验计数，未见计数用 1e-12 下限）；
  ②边际 α=1 平滑（后设平滑诊断，抵消未平滑下限对增益的夸大）；
  ③时钟桶条件（α=1，仅可见信息=切面时钟桶的非学习计数基线）；
- **按天聚类 bootstrap CI（2026-10-01 修订）**：切片来自 40 天，逐切片重采样低估
  不确定性——主 CI 改为整天聚类重采样（1000 次），逐切片 bootstrap 仅留作参考；
- 槽条件分布联合校准：(时间桶×温区×空间桶×需求桶) 及单字段边际 vs 经验频率；
- 真实未来 token 排序质量：top-1..5 覆盖 + 均值对数似然。

诚实口径：同一模型在「训练日切片」（拟合成验证，乐观）与「留出历史日切片」
（默认日 160..199，模型从未见过）分别报告；本脚本不触碰任何开发/论文测试种子。
结论措辞纪律：未平滑基线的正增益仅称「优于指定未平滑基线的离线预测线索」；
须按天聚类 CI 下界>0 且对平滑/时钟条件基线稳健，才可称稳健显著。
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
from scenario_saa import build_history
from maskco_scenario import (MaskCOScenarioModel, load_model, check_model_generation,
                             MASK_TOKEN, PAD_TOKEN, VOCAB_SIZE, M_MAX,
                             N_FIELDS, order_to_token, token_to_bucket,
                             clock_support_mask, clock_bin_of, _cond_probs)

TRAIN_SEED = 20260925
N_TIME_BINS, N_CLASSES, N_SPOTS, N_DEMAND_BINS = N_FIELDS
_NLL_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# 指标（纯 numpy，可单测）
# --------------------------------------------------------------------------- #
def _softmax(x):
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


def _nll_of(probs_dist, actuals):
    """probs_dist: (m_max+1,) 分布或 (S, m_max+1) 逐样本分布；actuals: (S,)。"""
    p = np.asarray(probs_dist, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :].repeat(len(actuals), axis=0)
    return float(-np.mean([np.log(max(p[i, a], _NLL_FLOOR))
                           for i, a in enumerate(actuals)]))


def count_metrics(actuals, cnt_logits):
    """数量头校准指标（不含基线；基线 NLL 由调用方给出）。"""
    probs = np.stack([_softmax(c) for c in cnt_logits])          # (S, m_max+1)
    k = np.arange(probs.shape[1], dtype=np.float64)
    mean_pred = (probs * k).sum(1)
    bias = float(np.mean(mean_pred - actuals))
    mae = float(np.mean(np.abs(mean_pred - actuals)))
    rmse = float(np.sqrt(np.mean((mean_pred - actuals) ** 2)))
    if len(actuals) > 1 and np.std(mean_pred) > 0 and np.std(actuals) > 0:
        rank = float(np.corrcoef(mean_pred, actuals)[0, 1])
    else:
        rank = float('nan')
    rng = np.random.default_rng(7)
    lows = np.array([probs[i, :a].sum() for i, a in enumerate(actuals)])
    highs = np.array([probs[i, :a + 1].sum() for i, a in enumerate(actuals)])
    u = rng.uniform(lows, highs)
    hist, _ = np.histogram(u, bins=10, range=(0.0, 1.0))
    return {"n_slices": int(len(actuals)), "bias_mean": bias, "mae": mae, "rmse": rmse,
            "rank_corr": rank, "pit_mean": float(u.mean()),
            "pit_histogram": hist.tolist(), "nll_model": _nll_of(probs, actuals)}


def _baselines_from_train(train_slices, m_max):
    """训练日切片 → 三个计数基线（只含训练日信息）：
    marg_raw（未平滑）、marg_sm1（α=1）、clock_cond_sm1（每时钟桶 α=1）。
    返回 dict name -> callable(cb_list, actuals) -> NLL。"""
    marg = np.zeros(m_max + 1)
    per_cb = {}
    for _d, _c, _v, _t, n, cb in train_slices:
        n = min(int(n), m_max)
        marg[n] += 1.0
        per_cb.setdefault(int(cb), np.zeros(m_max + 1))[n] += 1.0
    p_raw = marg / max(marg.sum(), 1e-12)
    p_sm1 = (marg + 1.0) / (marg + 1.0).sum()
    p_cb = {cb: (h + 1.0) / (h + 1.0).sum() for cb, h in per_cb.items()}
    fallback = p_sm1

    def _nll_marg_raw(cb_list, actuals):
        return _nll_of(p_raw, actuals)

    def _nll_marg_sm1(cb_list, actuals):
        return _nll_of(p_sm1, actuals)

    def _nll_clockcond(cb_list, actuals):
        out = []
        for a, cb in zip(actuals, cb_list):
            out.append(-np.log(max(p_cb.get(int(cb), fallback)[a], _NLL_FLOOR)))
        return float(np.mean(out))

    return {'marginal_raw': _nll_marg_raw, 'marginal_sm1': _nll_marg_sm1,
            'clock_cond_sm1': _nll_clockcond}


def bucket_calibration(slices, tok_logits, clocks):
    """槽条件分布 vs 经验频率（部署口径：时钟支持 + 非 NULL 条件）。
    slices: 6 元组 (day, clock, vis, tgt, n, cb)。"""
    agg = {"t": np.zeros(N_TIME_BINS), "c": np.zeros(N_CLASSES),
           "b": np.zeros(N_SPOTS), "d": np.zeros(N_DEMAND_BINS),
           "joint": np.zeros(N_FIELDS)}
    emp = {"t": np.zeros(N_TIME_BINS), "c": np.zeros(N_CLASSES),
           "b": np.zeros(N_SPOTS), "d": np.zeros(N_DEMAND_BINS),
           "joint": np.zeros(N_FIELDS)}
    n_emp_tot = 0
    for (_day, _clock, _vis, tgt, n_act, _cb), tl, clock in zip(slices, tok_logits, clocks):
        support = clock_support_mask(float(clock))
        p = _cond_probs(tl, support)                    # (m, VOCAB)
        p_slot = p.mean(0)                              # 每槽平均（槽对称）
        for tok in tgt:
            if not (0 < tok < VOCAB_SIZE):
                continue
            t, c, b, d = token_to_bucket(int(tok))
            emp["t"][t] += 1.0
            emp["c"][c] += 1.0
            emp["b"][b] += 1.0
            emp["d"][d] += 1.0
            emp["joint"][t, c, b, d] += 1.0
            n_emp_tot += 1
        w = float(n_act) * p_slot
        for tok in range(1, VOCAB_SIZE):
            t, c, b, d = token_to_bucket(tok)
            agg["t"][t] += w[tok]
            agg["c"][c] += w[tok]
            agg["b"][b] += w[tok]
            agg["d"][d] += w[tok]
            agg["joint"][t, c, b, d] += w[tok]
    out = {}
    for key in agg:
        pr = agg[key] / max(agg[key].sum(), 1e-12)
        er = emp[key] / max(emp[key].sum(), 1e-12)
        out[key] = {"pred_share": np.round(pr, 5).tolist(),
                    "emp_share": np.round(er, 5).tolist(),
                    "n_emp": emp[key].astype(int).tolist(),
                    "bias": float(np.sum(np.abs(pr - er)) / 2.0)}
    out["_n_emp_total"] = int(n_emp_tot)
    return out


def ranking_metrics(slices, tok_logits, clocks):
    """真实未来 token 排序质量：top-k 覆盖（任一槽命中即算）+ 均值对数概率。"""
    top_cov = {k: [] for k in (1, 2, 3, 5)}
    logps = []
    for (_day, _clock, _vis, tgt, _n_act, _cb), tl, clock in zip(slices, tok_logits, clocks):
        if not tgt:
            continue
        support = clock_support_mask(float(clock))
        p = _cond_probs(tl, support)                       # (m, VOCAB)
        topk_sets = [set(np.argsort(-p[:, 1:], axis=1)[:, :k2].ravel() + 1)
                     for k2 in (1, 2, 3, 5)]
        for tok in tgt:
            if not (0 < tok < VOCAB_SIZE):
                continue
            for k2 in (1, 2, 3, 5):
                cov = tok in topk_sets[(1, 2, 3, 5).index(k2)]
                top_cov[k2].append(1.0 if cov else 0.0)
            lp = np.log(np.maximum(p[:, tok].max(), 1e-12))
            logps.append(float(lp))
    return {"top_k_coverage": {k: (float(np.mean(v)) if v else None)
                               for k, v in top_cov.items()},
            "mean_logp_actual_token": float(np.mean(logps)) if logps else None,
            "n_actual_tokens": int(len(logps))}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _slice_with_clocks(history_days, cuts_per_day, rng):
    """镜像 make_training_slices，保留切面时钟与所属日（用于按天聚类 bootstrap）。
    返回 [(day_idx, clock, vis, tgt, n, cb)]，day_idx 为 history_days 内的相对日号。"""
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


def _summary_of(actuals, cnt_logits, cb_list, baselines_fn):
    m = count_metrics(actuals, cnt_logits)
    out = dict(m)
    for name, fn in baselines_fn.items():
        nll_b = fn(cb_list, actuals)
        out['nll_' + name] = nll_b
        out['gain_' + name] = nll_b - m['nll_model']
    return out


def _day_cluster_bootstrap(day_ids, actuals, cnt_logits, cb_list, baselines_fn,
                           rng, B=1000):
    """按天聚类 bootstrap（2026-10-01）：整天重采样（含该天全部切片）→ 各指标 CI。"""
    days = np.asarray(day_ids)
    uniq = np.unique(days)
    day_slices = {d: [i for i, x in enumerate(days) if x == d] for d in uniq}
    keys = ('bias_mean', 'nll_model', 'pit_mean',
            'gain_marginal_raw', 'gain_marginal_sm1', 'gain_clock_cond_sm1')
    acc = {k: [] for k in keys}
    for _ in range(B):
        sel = []
        for d in rng.choice(uniq, size=len(uniq), replace=True):
            sel.extend(day_slices[int(d)])
        if not sel:
            continue
        s = _summary_of([actuals[i] for i in sel], [cnt_logits[i] for i in sel],
                        [cb_list[i] for i in sel], baselines_fn)
        for k in keys:
            v = s.get(k)
            acc[k].append(v if np.isfinite(v) else None)
    out = {}
    for k in keys:
        vals = [v for v in acc[k] if v is not None]
        out[k] = ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                  if vals else None)
    out['_n_days_resampled'] = int(len(uniq))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", required=True)
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=None,
                    help="默认读 model.bin 同目录 config.json 的 m_max")
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--history-days", type=int, default=200,
                    help="训练历史总天数（seed 20260925 流）")
    ap.add_argument("--day-start", type=int, default=0)
    ap.add_argument("--day-end", type=int, default=None,
                    help="校准切片 [day_start, day_end)；默认全 200 天")
    ap.add_argument("--cuts-per-day", type=int, default=4)
    ap.add_argument("--slice-seed", type=int, default=42)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--allow-code-mismatch", action="store_true",
                    help="放行模型代码世代不匹配（显式标注，仅诊断用途）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    m_max = args.m_max
    gen = check_model_generation(args.model_bin,
                                 allow_mismatch=args.allow_code_mismatch)
    print("model generation check:", gen)
    if m_max is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.model_bin)),
                                "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            m_max = int(cfg["args"].get("m_max", M_MAX))
            args.dim = int(cfg["args"].get("dim", args.dim))
            args.arm = cfg["args"].get("arm", args.arm)
            print("config.json: m_max=%d dim=%d arm=%s" % (m_max, args.dim, args.arm))
        else:
            m_max = M_MAX
    day_end = args.day_end if args.day_end is not None else args.history_days
    assert 0 <= args.day_start < day_end <= args.history_days

    all_days = build_history(generate_dataset(args.history_days, args.n_orders,
                                              TRAIN_SEED))
    cal_days = all_days[args.day_start:day_end]
    rng_slices = np.random.default_rng(args.slice_seed)
    slices = _slice_with_clocks(cal_days, args.cuts_per_day, rng_slices)
    n_days = len({s[0] for s in slices})
    print("slices: %d over %d days (days %d..%d)"
          % (len(slices), n_days, args.day_start, day_end))

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

    tok_logits, cnt_logits, actuals, cb_list, day_ids = [], [], [], [], []
    for day_idx, clock, vis, tgt, n, cb in slices:
        vis_c = vis[:m_max]
        toks = np.full(2 * m_max, PAD_TOKEN, dtype=np.int32)   # 固定形状（编译一次）
        toks[:len(vis_c)] = vis_c
        toks[len(vis_c):len(vis_c) + m_max] = MASK_TOKEN
        cbv = jnp.asarray([cb], dtype=jnp.int32)
        tl, cl = model(jnp.asarray(toks)[None], clock_bin=cbv)
        tok_logits.append(np.asarray(tl[0][len(vis_c):len(vis_c) + m_max]))
        cnt_logits.append(np.asarray(cl[0][:m_max + 1]))
        actuals.append(min(int(n), m_max))
        cb_list.append(int(cb))
        day_ids.append(int(day_idx))

    clocks = [s[1] for s in slices]
    train_rng = np.random.default_rng(args.slice_seed + 1)
    train_slices = _slice_with_clocks(all_days[:args.day_start], args.cuts_per_day,
                                      train_rng)
    baselines_fn = _baselines_from_train(train_slices, m_max)
    print("baselines from %d train-day slices (days 0..%d)"
          % (len(train_slices), args.day_start))

    cnt = _summary_of(actuals, cnt_logits, cb_list, baselines_fn)
    buckets = bucket_calibration(slices, tok_logits, clocks)
    ranks = ranking_metrics(slices, tok_logits, clocks)

    # 主 CI = 按天聚类 bootstrap（2026-10-01 用户复核修订）
    day_rng = np.random.default_rng(args.slice_seed + 777)
    day_ci = _day_cluster_bootstrap(day_ids, actuals, cnt_logits, cb_list,
                                    baselines_fn, day_rng, B=1000)
    # 逐切片 bootstrap 仅参考（同一数据，非独立重复）
    boot_rng = np.random.default_rng(args.slice_seed + 999)
    S = len(slices)
    slice_acc = {}
    for _ in range(1000):
        idxs = boot_rng.integers(0, S, S)
        s = _summary_of([actuals[i] for i in idxs], [cnt_logits[i] for i in idxs],
                        [cb_list[i] for i in idxs], baselines_fn)
        for k, v in s.items():
            if isinstance(v, float) and np.isfinite(v):
                slice_acc.setdefault(k, []).append(v)
    slice_ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for k, v in slice_acc.items() if len(v)}

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "model_generation_check": gen,
        "train_seed": TRAIN_SEED,
        "history_days": {"total": args.history_days,
                         "calibration_range": [args.day_start, day_end],
                         "note": "训练历史日切片；[160,200) 为留出校准（模型未见）"},
        "n_eff": {"n_days": int(n_days),
                  "n_calibration_slices": int(S),
                  "n_marginal_train_slices": int(len(train_slices)),
                  "n_actual_tokens": int(ranks['n_actual_tokens'] or 0)},
        "n_slices": len(slices),
        "count": cnt,
        "count_day_cluster_ci": day_ci,
        "count_slice_bootstrap_ci_reference_only": slice_ci,
        "bucket_calibration": buckets,
        "ranking": ranks,
        "conclusion_discipline": ("未平滑基线正增益仅称「优于指定未平滑基线的离线预测线索」；"
                                  "须按天聚类 CI 下界>0 且对平滑/时钟条件基线稳健才可称稳健显著。"
                                  "预训练 vs 同架构随机为单种子消融，不推在线决策收益。"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(_nan2none(report), f, indent=2)
    print(json.dumps(_nan2none(report), indent=2))
    return report


def _nan2none(x):
    if isinstance(x, dict):
        return {k: _nan2none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan2none(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


if __name__ == "__main__":
    main()
