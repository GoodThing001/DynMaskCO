"""A 步骤 1：需求可预测性探针（无训练，纯 NumPy）。

回答：在带空间-时间结构 + 历史数据的合成需求过程下，今天已揭示的早期订单
能否预测未来订单（out-of-sample）？可预测 → A 的承重假设成立；否则改 DGP。

DGP（合成、结构化、现象真实——热点/时间峰/提前期都是真实生鲜配送现象）：
  - 连续潜在日因子 z~N(0,1) 决定：空间热点权重、峰值小时、总需求。
  - 3 个空间热点（高斯团），权重 softmax(log(base)+β·z)，β 使 z 高→热点0主导。
  - 时间速率单峰剖面，峰在 peak_hour(z)=clip(11+4z, 9, 19)。
  - 订单 = 非齐次泊松（thinning）；每单带 temp_class（与热点相关）与交付窗。

探针（k-NN，与 run_value_predictability 同级的简单探针，非神经网络）：
  - 特征：τ 前早期订单的每热点计数 + 每温区计数（6 维）。
  - 目标：未来（t>τ）每热点订单计数（3 维），总数=和。
  - 条件预测：k-NN 历史天；基线：边际均值。
  - 指标：未来总数 out-of-sample R²、每热点池化 R²、L1 误差下降、预测 spread。

判读：R²_total 显著 > 0 且 CI 下界 > 0 → 可预测（承重假设成立）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def generate_day(rng, z, T=16.0, tau=3.0):
    """生成一天订单，返回 (early_features 6-dim, future_spot_counts 3-dim, future_total)。"""
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
    beta = np.array([2.0, -0.5, -0.5])
    base = np.array([0.2, 0.4, 0.4])
    weights = _softmax(np.log(base) + beta * z)
    peak = float(np.clip(11.0 + 4.0 * z, 9.0, 19.0))
    total_per_h = 200.0 * np.exp(0.3 * z) / T

    lam_max = total_per_h * 2.2
    n = int(rng.poisson(lam_max * T))
    times = rng.uniform(0.0, T, n)
    prof = 1.0 + 1.2 * np.exp(-((times - peak) ** 2) / (2.0 * 1.5 ** 2))
    keep = rng.uniform(size=n) < (prof * total_per_h / lam_max)
    times = times[keep]
    spot = rng.choice(3, size=times.size, p=weights)
    class_probs = np.array([
        [0.1, 0.3, 0.6],
        [0.3, 0.5, 0.2],
        [0.6, 0.3, 0.1],
    ])
    temp_class = np.array([rng.choice(3, p=class_probs[s]) for s in spot])

    early = times <= tau
    feat = np.zeros(6)
    for s in range(3):
        feat[s] = np.sum(spot[early] == s)
        feat[3 + s] = np.sum(temp_class[early] == s)
    future_spot = np.array([np.sum(spot[~early] == s) for s in range(3)], dtype=np.float64)
    future_total = float((~early).sum())
    return feat, future_spot, future_total


def _knn_predict(Xq_norm, Xtr_norm, Ytr, k):
    out = np.zeros((Xq_norm.shape[0], Ytr.shape[1]))
    for i in range(Xq_norm.shape[0]):
        d = ((Xtr_norm - Xq_norm[i]) ** 2).sum(1)
        idx = np.argsort(d)[:k]
        out[i] = Ytr[idx].mean(0)
    return out


def _r2(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _bootstrap_r2(y, p, b, rng):
    vals = np.empty(b)
    n = len(y)
    for j in range(b):
        idx = rng.integers(0, n, size=n)
        vals[j] = _r2(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    train = [generate_day(rng, float(rng.normal()), tau=args.tau)
             for _ in range(args.train_days)]
    test = [generate_day(rng, float(rng.normal()), tau=args.tau)
            for _ in range(args.test_days)]

    Xtr = np.array([f for f, _, _ in train])
    Ytr = np.array([y for _, y, _ in train])
    Xte = np.array([f for f, _, _ in test])
    Yte = np.array([y for _, y, _ in test])
    Tte = np.array([t for _, _, t in test])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    P_spot = _knn_predict(Xte_n, Xtr_n, Ytr, args.k)
    P_total = P_spot.sum(1)

    r2_total = _r2(Tte, P_total)
    ci_lo, ci_hi = _bootstrap_r2(Tte, P_total, 1000, rng)
    r2_spot = _r2(Yte.ravel(), P_spot.ravel())

    l1_cond = float(np.abs(Yte - P_spot).sum(1).mean())
    l1_marg = float(np.abs(Yte - Ytr.mean(0)).sum(1).mean())
    l1_reduction = 1.0 - l1_cond / l1_marg if l1_marg > 0 else float("nan")

    spread = float(P_total.std() / max(P_total.mean(), 1e-9))

    if r2_total > 0.3 and ci_lo > 0.1:
        verdict = "PREDICTABLE"
    elif r2_total > 0.1 and ci_lo > 0.0:
        verdict = "WEAK"
    else:
        verdict = "NOT-PREDICTABLE"

    report = {
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "r2_total": r2_total,
        "r2_total_ci": [ci_lo, ci_hi],
        "r2_spot_pooled": r2_spot,
        "l1_reduction_vs_marginal": l1_reduction,
        "prediction_spread": spread,
        "verdict": verdict,
        "note": "synthetic DGP (hot-spots + time peak + continuous latent factor); "
                "scope = protocol validation, not real-data claim",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
