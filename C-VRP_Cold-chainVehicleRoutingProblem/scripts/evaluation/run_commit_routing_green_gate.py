"""A 步骤 3b 地基（无训练）：路由耦合能耗预算下的承诺头腔门。

与 run_commit_green_gate 的类级能耗版不同，本版能耗 = f(温区, 位置)：
  e_i = E_door/COP + UA·ΔT_class·(dist_i / speed) / COP
即"远单更耗能、冷冻单更耗能"——绿色预算通过位置+路由真正耦合进 accept/reject。

文献参数（地基约束规格_3b.md）：E_door=0.056 kWh/次、UA=0.03 kW/K、COP=1.5、
speed=30 km/h、1 距离单位=20 km。→ e_i ≈ 0.0373 + 0.0133·ΔT·dist。

DGP：z~N(0,1) 决定热点权重（→未来订单的位置与温区分布可预测）。
决策：accept 消耗 e(class,loc)；reject 丢 revenue；资源=每日能耗预算 B=(1−ρ)·baseline。
策略：myopic / cond(uncond) bootstrap（价值/能耗分数边际 λ）/ oracle（精确 knapsack DP）。
判读：扫 ρ，看 causal_cond 是否显著（CI 下界 > 0）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REV = {0: 10.0, 1: 20.0, 2: 30.0}
DT = {0: 7.0, 1: 21.0, 2: 43.0}
CENTERS = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
DEPOT = np.array([0.5, 0.5])
SCALE = 100.0  # 能耗 → 整数权重（knapsack DP）


def energy_of(cls, loc):
    dist = float(np.linalg.norm(loc - DEPOT))
    return 0.0373 + 0.0133 * DT[int(cls)] * dist


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def generate_day(rng, z, T=16.0):
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
    locs = CENTERS[spot] + rng.normal(0.0, 0.12, size=(times.size, 2))
    class_probs = np.array([
        [0.1, 0.3, 0.6],
        [0.3, 0.5, 0.2],
        [0.6, 0.3, 0.1],
    ])
    cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
    idx = np.argsort(times)
    return times[idx], cls[idx], locs[idx]


def early_feature(times, cls, locs, tau=3.0):
    feat = np.zeros(6)
    m = times <= tau
    for c in range(3):
        feat[3 + c] = float(np.sum(cls[m] == c))
    if m.sum() > 0:
        d = np.linalg.norm(locs[m][:, None, :] - CENTERS[None, :, :], axis=2)
        nearest = d.argmin(1)
        for s in range(3):
            feat[s] = float(np.sum(nearest == s))
    return feat


def _knapsack_exact(vals, energys, R):
    W = int(round(R * SCALE))
    if W <= 0:
        return 0.0
    dp = np.zeros(W + 1)
    for v, e in zip(vals, energys):
        ei = int(round(e * SCALE))
        if ei > W:
            continue
        dp[ei:] = np.maximum(dp[ei:], dp[:-ei] + v)
    return float(dp[W])


def _marginal_lambda(vals, energys, R):
    if len(vals) == 0 or R <= 1e-12:
        return 0.0
    order = np.argsort(-(vals / np.maximum(energys, 1e-12)))
    rem = R
    for i in order:
        v, e = vals[i], energys[i]
        if e <= rem + 1e-12:
            rem -= e
        else:
            return float(v / e)
    return 0.0


def evaluate_day(times, cls, locs, B, tau, hist, neighbors):
    morning = times <= tau
    afternoon = ~morning
    V_morning = float(np.sum([REV[int(c)] for c in cls[morning]]))
    E_morning = float(np.sum([energy_of(c, l) for c, l in zip(cls[morning], locs[morning])]))
    R0 = B - E_morning

    all_v = np.array([REV[int(c)] for c in cls])
    all_e = np.array([energy_of(c, l) for c, l in zip(cls, locs)])
    V_oracle = V_morning + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)

    V_my = V_morning
    rem = R0
    for i in np.where(afternoon)[0]:
        e = all_e[i]
        if e <= rem + 1e-12:
            V_my += all_v[i]
            rem -= e

    V_bs = V_morning
    rem = R0
    for i in np.where(afternoon)[0]:
        if rem <= 1e-12:
            break
        v_o = all_v[i]
        e_o = all_e[i]
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
        "earr": [np.array([energy_of(c, l) for c, l in zip(d[1], d[2])]) for d in hist_days],
    }
    Xtr = np.array([early_feature(d[0], d[1], d[2], args.tau) for d in hist_days])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd
    baseline = float(np.mean([hist["earr"][h].sum() for h in range(H)]))

    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]

    def stat(x):
        n = len(x)
        rng_b = np.random.default_rng(args.seed + 999)
        means = np.array([x[rng_b.integers(0, n, n)].mean() for _ in range(2000)])
        return {"mean": float(np.mean(x)),
                "ci_lo": float(np.percentile(means, 2.5)),
                "ci_hi": float(np.percentile(means, 97.5))}

    report = {"config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
              "baseline_energy_mean": baseline, "per_rho": {}}

    for rho in args.rhos:
        B = (1.0 - rho) * baseline
        V_my, V_cond, V_uncond, V_or = [], [], [], []
        for t_arr, c_arr, l_arr in test_days:
            feat = early_feature(t_arr, c_arr, l_arr, args.tau)
            x = (feat - mu) / sd
            d = ((Xtr_n - x) ** 2).sum(1)
            cond_neighbors = np.argsort(d)[:args.k]
            uncond_neighbors = rng.choice(H, size=args.k, replace=False)
            vm, vc, vo = evaluate_day(t_arr, c_arr, l_arr, B, args.tau, hist, cond_neighbors)
            _, vu, _ = evaluate_day(t_arr, c_arr, l_arr, B, args.tau, hist, uncond_neighbors)
            V_my.append(vm); V_cond.append(vc); V_uncond.append(vu); V_or.append(vo)

        V_my = np.array(V_my); V_cond = np.array(V_cond)
        V_uncond = np.array(V_uncond); V_or = np.array(V_or)
        fi = V_or - V_my; cc = V_cond - V_my; cu = V_uncond - V_my; rc = V_or - V_cond

        report["per_rho"][str(rho)] = {
            "B": B,
            "value": {"myopic": stat(V_my), "cond": stat(V_cond),
                      "uncond": stat(V_uncond), "oracle": stat(V_or)},
            "future_info": stat(fi), "causal_cond": stat(cc),
            "causal_uncond": stat(cu), "residual": stat(rc),
            "cond_share": float(np.mean(cc) / max(np.mean(fi), 1e-9)),
            "uncond_share": float(np.mean(cu) / max(np.mean(fi), 1e-9)),
        }
        print(f"\n=== rho={rho} (B={B:.1f} vs baseline {baseline:.1f}) ===")
        print(f"  myopic={np.mean(V_my):.0f}  cond={np.mean(V_cond):.0f}  "
              f"uncond={np.mean(V_uncond):.0f}  oracle={np.mean(V_or):.0f}")
        print(f"  future_info={np.mean(fi):.0f} CI[{np.percentile(fi,2.5):.0f},{np.percentile(fi,97.5):.0f}]  "
              f"causal_cond={np.mean(cc):.0f} CI[{np.percentile(cc,2.5):.0f},{np.percentile(cc,97.5):.0f}]")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved:", os.path.join(args.out, "gate.json"))
    return report


if __name__ == "__main__":
    main()
