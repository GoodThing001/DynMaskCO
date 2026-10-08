"""A 步骤 2：承诺头腔门（无训练，纯 NumPy）。

问题：在可预测需求下，lookahead（bootstrap SAA）相对 myopic（来就收）是否有
材料且因果的头腔？

决策模型（隔离 accept/reject 机制；路由抽象为容量 + 逐单价值）：
  - 每天：订单按到达顺序（时刻 + temp_class）。
  - 价值 revenue(class)：ambient=10 / chilled=20 / frozen=30（冷链：冷冻高值）。
  - 容量 C：每天最多服务 C 单。
  - 结构：morning[0,tau] 预下单已承诺；afternoon(tau,T] 逐单 accept/reject。

策略（同一 DGP、同一容量、同一价值、同一场景预算 K）：
  - Myopic：afternoon 来就收直到满。
  - Oracle：全知收 top-C（上界，隔离未来信息份额）。
  - Bootstrap-cond：每决策用 K 个 k-NN 相似历史天的未来订单估机会成本。
  - Bootstrap-uncond：每决策用 K 个随机历史天（无条件，对照；与 cond 同预算）。

指标（每天总价值 + 头腔分解，配对 bootstrap CI）：
  future_info = oracle - myopic；causal_cond = cond - myopic；
  causal_uncond = uncond - myopic；residual = oracle - cond。

判读：causal_cond 材料 > 0 且 cond > uncond → 有因果头腔且条件化有用 → 进入步骤 3。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REV = {0: 10.0, 1: 20.0, 2: 30.0}  # ambient / chilled / frozen


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
    return times[idx], spot[idx], cls[idx]


def early_feature(times, cls, tau=3.0):
    feat = np.zeros(6)
    for t, c in zip(times[times <= tau], cls[times <= tau]):
        feat[3 + c] += 1
    # spot index not retained in the gate features (class-only is enough to infer z
    # via the class mix); keep 3 class counts + total to match a 4-dim state.
    return feat


def _opp_cost(future_vals, remaining):
    if remaining <= 0 or future_vals.size < remaining:
        return 0.0
    return float(np.partition(future_vals, -remaining)[-remaining])


def evaluate_day(times, cls, C, tau, hist, neighbors):
    """返回 (V_myopic, V_bootstrap, V_oracle)。"""
    morning_mask = times <= tau
    V_morning = float(np.sum([REV[int(c)] for c in cls[morning_mask]]))
    R0 = C - int(morning_mask.sum())
    afternoon_idx = np.where(~morning_mask)[0]

    all_vals = np.sort(np.array([REV[int(c)] for c in cls]))[::-1]
    V_oracle = float(all_vals[:C].sum())

    V_my = V_morning
    rem = R0
    for i in afternoon_idx:
        if rem <= 0:
            break
        V_my += REV[int(cls[i])]
        rem -= 1

    V_bs = V_morning
    rem = R0
    for i in afternoon_idx:
        if rem <= 0:
            break
        v_o = REV[int(cls[i])]
        t = times[i]
        opps = []
        for h in neighbors:
            t_arr, v_arr = hist["tarr"][h], hist["varr"][h]
            fut = v_arr[t_arr > t]
            opps.append(_opp_cost(fut, rem))
        if v_o > float(np.mean(opps)) + 1e-9:
            V_bs += v_o
            rem -= 1
    return V_my, V_bs, V_oracle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--capacity", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    hist = {
        "tarr": [d[0] for d in hist_days],
        "varr": [np.array([REV[int(c)] for c in d[2]]) for d in hist_days],
    }
    Xtr = np.array([early_feature(d[0], d[2], args.tau) for d in hist_days])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd

    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]

    V_my, V_cond, V_uncond, V_or = [], [], [], []
    for t_arr, s_arr, c_arr in test_days:
        feat = early_feature(t_arr, c_arr, args.tau)
        x = (feat - mu) / sd
        d = ((Xtr_n - x) ** 2).sum(1)
        cond_neighbors = np.argsort(d)[:args.k]
        uncond_neighbors = rng.choice(H, size=args.k, replace=False)

        vm, vc, vo = evaluate_day(t_arr, c_arr, args.capacity, args.tau, hist, cond_neighbors)
        _, vu, _ = evaluate_day(t_arr, c_arr, args.capacity, args.tau, hist, uncond_neighbors)
        V_my.append(vm)
        V_cond.append(vc)
        V_uncond.append(vu)
        V_or.append(vo)

    V_my = np.array(V_my)
    V_cond = np.array(V_cond)
    V_uncond = np.array(V_uncond)
    V_or = np.array(V_or)

    def stat(x):
        return {"mean": float(np.mean(x)),
                "ci_lo": float(np.percentile(x, 2.5)),
                "ci_hi": float(np.percentile(x, 97.5))}

    future_info = V_or - V_my
    causal_cond = V_cond - V_my
    causal_uncond = V_uncond - V_my
    residual_cond = V_or - V_cond

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "total_value": {
            "myopic": stat(V_my), "cond": stat(V_cond),
            "uncond": stat(V_uncond), "oracle": stat(V_or),
        },
        "headroom": {
            "future_info_oracle_minus_myopic": stat(future_info),
            "causal_cond_minus_myopic": stat(causal_cond),
            "causal_uncond_minus_myopic": stat(causal_uncond),
            "residual_oracle_minus_cond": stat(residual_cond),
        },
        "cond_share_of_future_info": float(np.mean(causal_cond) / max(np.mean(future_info), 1e-9)),
        "uncond_share_of_future_info": float(np.mean(causal_uncond) / max(np.mean(future_info), 1e-9)),
    }

    m_cc = float(np.mean(causal_cond))
    cond_over_uncond = float(np.mean(causal_cond - causal_uncond))
    if m_cc <= 0:
        verdict = "NO-CAUSAL-HEADROOM"
    elif cond_over_uncond <= 0:
        verdict = "HEADROOM-BUT-CONDITIONING-USELESS"
    else:
        verdict = "CAUSAL-HEADROOM (proceed to step 3)"
    report["verdict"] = verdict

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
