"""A 步骤 3b 地基（无训练）：绿色能耗预算下的承诺头腔门。

问题：在可预测需求 + 制冷能耗预算（绿色硬约束）下，条件前瞻（k-NN bootstrap）
相对 myopic 是否有材料因果头腔？头腔如何随预算松弛度 ρ 变化？

模型（文献标定，见 docs/当前规划/研究设计/地基约束规格_3b.md）：
- 订单 (time, class)。revenue = [10,20,30]；能耗 e(class) = [1,3,6]（∝ ΔT）。
- 资源：车队级每日制冷能耗预算 B = (1−ρ)·baseline。
- 决策：accept 消耗 e(class)；reject 丢 revenue。
- 同日品质衰减 ≤6% 略去（多日尺度另立）。

策略：
- myopic：来就收，装不下跳过（continue），直到预算耗尽。
- cond/uncond bootstrap：opportunity = 未来订单价值/能耗的分数边际 λ；accept iff rev/e > λ。
- oracle：精确 0/1 knapsack DP（真上界）。

判读：扫 ρ ∈ {0.10,0.20,0.30,0.40}；causal_cond 材料 >0 且 cond > uncond。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REV = {0: 10.0, 1: 20.0, 2: 30.0}
ENERGY = {0: 1.0, 1: 3.0, 2: 6.0}


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def generate_day(rng, z, T=16.0):
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
    cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
    idx = np.argsort(times)
    return times[idx], cls[idx]


def early_feature(times, cls, tau=3.0):
    feat = np.zeros(3)
    for t, c in zip(times[times <= tau], cls[times <= tau]):
        feat[int(c)] += 1
    return feat


def _knapsack_exact(vals, energys, R):
    """精确 0/1 knapsack DP（整数能耗权重），返回最大价值。"""
    W = int(round(R))
    if W <= 0:
        return 0.0
    dp = np.zeros(W + 1)
    for v, e in zip(vals, energys):
        ei = int(round(e))
        if ei > W:
            continue
        dp[ei:] = np.maximum(dp[ei:], dp[:-ei] + v)
    return float(dp[W])


def _marginal_lambda(vals, energys, R):
    """未来订单价值/能耗的分数边际 λ；预算不绑定（全装得下）时返回 0。"""
    if len(vals) == 0 or R <= 1e-12:
        return 0.0
    order = np.argsort(-(vals / np.maximum(energys, 1e-12)))
    rem = R
    for i in order:
        v, e = vals[i], energys[i]
        if e <= rem + 1e-12:
            rem -= e
        else:
            return float(v / e)  # 部分装下：分数边际
    return 0.0  # 全部装下：能耗不稀缺，接受一切正价值


def evaluate_day(times, cls, B, tau, hist, neighbors):
    morning = times <= tau
    afternoon = ~morning
    V_morning = float(np.sum([REV[int(c)] for c in cls[morning]]))
    E_morning = float(np.sum([ENERGY[int(c)] for c in cls[morning]]))
    R0 = B - E_morning

    all_v = np.array([REV[int(c)] for c in cls])
    all_e = np.array([ENERGY[int(c)] for c in cls])
    V_oracle = V_morning + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)

    # myopic：来就收，装不下跳过
    V_my = V_morning
    rem = R0
    for i in np.where(afternoon)[0]:
        e = ENERGY[int(cls[i])]
        if e <= rem + 1e-12:
            V_my += REV[int(cls[i])]
            rem -= e

    # bootstrap（分数边际 λ 规则）
    V_bs = V_morning
    rem = R0
    for i in np.where(afternoon)[0]:
        if rem <= 1e-12:
            break
        v_o = REV[int(cls[i])]
        e_o = ENERGY[int(cls[i])]
        t = times[i]
        lams = []
        for h in neighbors:
            t_arr = hist["tarr"][h]
            fut_mask = t_arr > t
            lams.append(_marginal_lambda(hist["varr"][h][fut_mask],
                                         hist["earr"][h][fut_mask], rem))
        lam = float(np.mean(lams))
        if e_o <= rem + 1e-12 and v_o > e_o * lam + 1e-9:
            V_bs += v_o
            rem -= e_o
    return V_my, V_bs, V_oracle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    hist = {
        "tarr": [d[0] for d in hist_days],
        "varr": [np.array([REV[int(c)] for c in d[1]]) for d in hist_days],
        "earr": [np.array([ENERGY[int(c)] for c in d[1]]) for d in hist_days],
    }
    Xtr = np.array([early_feature(d[0], d[1], args.tau) for d in hist_days])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd
    baseline = float(np.mean([hist["earr"][h].sum() for h in range(H)]))

    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]

    def stat(x):
        return {"mean": float(np.mean(x)),
                "ci_lo": float(np.percentile(x, 2.5)),
                "ci_hi": float(np.percentile(x, 97.5))}

    report = {"config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
              "baseline_energy_mean": baseline, "per_rho": {}}

    for rho in args.rhos:
        B = (1.0 - rho) * baseline
        V_my, V_cond, V_uncond, V_or = [], [], [], []
        for t_arr, c_arr in test_days:
            feat = early_feature(t_arr, c_arr, args.tau)
            x = (feat - mu) / sd
            d = ((Xtr_n - x) ** 2).sum(1)
            cond_neighbors = np.argsort(d)[:args.k]
            uncond_neighbors = rng.choice(H, size=args.k, replace=False)
            vm, vc, vo = evaluate_day(t_arr, c_arr, B, args.tau, hist, cond_neighbors)
            _, vu, _ = evaluate_day(t_arr, c_arr, B, args.tau, hist, uncond_neighbors)
            V_my.append(vm); V_cond.append(vc); V_uncond.append(vu); V_or.append(vo)

        V_my = np.array(V_my); V_cond = np.array(V_cond)
        V_uncond = np.array(V_uncond); V_or = np.array(V_or)
        fi = V_or - V_my; cc = V_cond - V_my; cu = V_uncond - V_my; rc = V_or - V_cond

        report["per_rho"][str(rho)] = {
            "B": B,
            "value": {"myopic": stat(V_my), "cond": stat(V_cond),
                      "uncond": stat(V_uncond), "oracle": stat(V_or)},
            "future_info": stat(fi),
            "causal_cond": stat(cc),
            "causal_uncond": stat(cu),
            "residual": stat(rc),
            "cond_share": float(np.mean(cc) / max(np.mean(fi), 1e-9)),
            "uncond_share": float(np.mean(cu) / max(np.mean(fi), 1e-9)),
        }
        print(f"\n=== rho={rho} (B={B:.1f} vs baseline {baseline:.1f}) ===")
        print(f"  myopic={np.mean(V_my):.0f}  cond={np.mean(V_cond):.0f}  "
              f"uncond={np.mean(V_uncond):.0f}  oracle={np.mean(V_or):.0f}")
        print(f"  future_info={np.mean(fi):.0f}  causal_cond={np.mean(cc):.0f} "
              f"({report['per_rho'][str(rho)]['cond_share']*100:.0f}%)  "
              f"causal_uncond={np.mean(cu):.0f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved:", os.path.join(args.out, "gate.json"))
    return report


if __name__ == "__main__":
    main()
