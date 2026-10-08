"""A 步骤 3b：学习型条件场景生成 · 首个原型（条件计数回归）对比 k-NN。

先回答关键经验问题：参数化学习模型（条件计数回归）能否在 accept/reject 决策上
打赢 k-NN 条件 bootstrap？能 → 升级掩码生成（真方法）；不能 → k-NN 已近最优。

模型：MLP(上下文 10-dim → 未来 (class,spot) 计数 9-dim)。
上下文 = morning (class,spot) 计数(9) + 当前时刻 t(1)。
目标 = 未来（time>t）的 (class,spot) 计数(9)。MSE（学条件均值）。
策略：预测的未来计数 → (value,energy) 多重集 → 价值/能耗分数边际 λ → accept。

基线：myopic / 无条件 bootstrap / k-NN 条件 bootstrap / 学习型 / oracle（精确 knapsack）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

REV = {0: 10.0, 1: 20.0, 2: 30.0}
DT = {0: 7.0, 1: 21.0, 2: 43.0}
CENTERS = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
DEPOT = np.array([0.5, 0.5])
SCALE = 100.0

# 每 (class, spot) 的能耗（spot 中心距离近似）
E_CS = np.zeros((3, 3))
for _c in range(3):
    for _s in range(3):
        _d = np.linalg.norm(CENTERS[_s] - DEPOT)
        E_CS[_c, _s] = 0.0373 + 0.0133 * DT[_c] * _d


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
    class_probs = np.array([
        [0.1, 0.3, 0.6],
        [0.3, 0.5, 0.2],
        [0.6, 0.3, 0.1],
    ])
    cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
    idx = np.argsort(times)
    return times[idx], cls[idx], spot[idx]


def cs_counts(cls, spot, mask=None):
    out = np.zeros(9)
    if mask is None:
        mask = np.ones(len(cls), dtype=bool)
    for c in range(3):
        for s in range(3):
            out[c * 3 + s] = np.sum((cls == c) & (spot == s) & mask)
    return out


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
    order = np.argsort(-(np.asarray(vals) / np.maximum(np.asarray(energys), 1e-12)))
    rem = R
    for i in order:
        if energys[i] <= rem + 1e-12:
            rem -= energys[i]
        else:
            return float(vals[i] / energys[i])
    return 0.0


def lambda_from_counts(counts, R):
    vals, es = [], []
    for c in range(3):
        for s in range(3):
            n = int(round(counts[c * 3 + s]))
            vals.extend([REV[c]] * n)
            es.extend([E_CS[c, s]] * n)
    return _marginal_lambda(vals, es, R)


# ---------- 模型 ----------
def init_params(key, d_in=10, d_h=64, d_out=9):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (d_in, d_h)) * 0.2
    b1 = jnp.zeros(d_h)
    W2 = jax.random.normal(k2, (d_h, d_out)) * 0.2
    b2 = jnp.zeros(d_out)
    return (W1, b1, W2, b2)


def forward(params, x):
    W1, b1, W2, b2 = params
    h = jax.nn.relu(x @ W1 + b1)
    return h @ W2 + b2


@jax.jit
def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days

    # 历史天 + 训练切片
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    hist = {"tarr": [d[0] for d in hist_days],
            "varr": [np.array([REV[int(c)] for c in d[1]]) for d in hist_days],
            "earr": [np.array([E_CS[int(c), int(s)] for c, s in zip(d[1], d[2])]) for d in hist_days]}
    Xtr_morning = np.array([cs_counts(d[1], d[2], d[0] <= args.tau) for d in hist_days])

    X_train, Y_train = [], []
    slice_times = np.linspace(args.tau, 13.0, args.slices)
    for d in hist_days:
        t_arr, c_arr, s_arr = d
        for t in slice_times:
            ctx = np.concatenate([cs_counts(c_arr, s_arr, t_arr <= args.tau), [t / 16.0]])
            fut = cs_counts(c_arr, s_arr, t_arr > t)
            X_train.append(ctx)
            Y_train.append(fut)
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    X_train = (X_train - mu) / sd

    key = jax.random.PRNGKey(args.seed)
    params = init_params(key)
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    rngb = np.random.default_rng(1)
    for it in range(args.steps):
        idx = rngb.integers(0, len(X_train), 64)
        params, opt_state, _ = step(params, opt_state, X_train[idx], Y_train[idx])

    # 基线：k-NN 索引（按 morning 计数）
    Xtr_n = (Xtr_morning - Xtr_morning.mean(0)) / (Xtr_morning.std(0) + 1e-6)

    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]
    baseline = float(np.mean([hist["earr"][h].sum() for h in range(H)]))
    B = (1.0 - args.rho) * baseline

    def evaluate(day, neighbors):
        t_arr, c_arr, s_arr = day
        morning = t_arr <= args.tau
        afternoon = ~morning
        V_m = float(np.sum([REV[int(c)] for c in c_arr[morning]]))
        E_m = float(np.sum([E_CS[int(c), int(s)] for c, s in zip(c_arr[morning], s_arr[morning])]))
        R0 = B - E_m
        all_v = np.array([REV[int(c)] for c in c_arr])
        all_e = np.array([E_CS[int(c), int(s)] for c, s in zip(c_arr, s_arr)])
        V_or = V_m + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)

        V = V_m
        rem = R0
        for i in np.where(afternoon)[0]:
            if rem <= 1e-12:
                break
            v_o, e_o = all_v[i], all_e[i]
            t = t_arr[i]
            if neighbors is None:
                # 学习型：预测未来计数 → λ
                ctx = np.concatenate([cs_counts(c_arr, s_arr, morning), [t / 16.0]])
                ctx = (ctx - mu) / sd
                counts = np.asarray(forward(params, jnp.asarray(ctx, dtype=jnp.float32)))
                lam = lambda_from_counts(counts, rem)
            else:
                lams = []
                for h in neighbors:
                    ht = hist["tarr"][h]
                    f = ht > t
                    lams.append(_marginal_lambda(hist["varr"][h][f], hist["earr"][h][f], rem))
                lam = float(np.mean(lams))
            if e_o <= rem + 1e-12 and v_o > e_o * lam + 1e-9:
                V += v_o
                rem -= e_o
        return V, V_or

    def stat(x):
        n = len(x)
        r = np.random.default_rng(args.seed + 777)
        m = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
        return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(m, 2.5)),
                "ci_hi": float(np.percentile(m, 97.5))}

    V_my = np.zeros(args.test_days)
    V_kn = np.zeros(args.test_days)
    V_un = np.zeros(args.test_days)
    V_ln = np.zeros(args.test_days)
    V_or = np.zeros(args.test_days)
    for di, day in enumerate(test_days):
        t_arr, c_arr, s_arr = day
        # myopic / oracle
        morning = t_arr <= args.tau
        V_m = float(np.sum([REV[int(c)] for c in c_arr[morning]]))
        E_m = float(np.sum([E_CS[int(c), int(s)] for c, s in zip(c_arr[morning], s_arr[morning])]))
        R0 = B - E_m
        all_v = np.array([REV[int(c)] for c in c_arr])
        all_e = np.array([E_CS[int(c), int(s)] for c, s in zip(c_arr, s_arr)])
        afternoon = ~morning
        # oracle
        V_or[di] = V_m + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)
        # myopic
        rem = R0
        v = V_m
        for i in np.where(afternoon)[0]:
            if all_e[i] <= rem + 1e-12:
                v += all_v[i]; rem -= all_e[i]
        V_my[di] = v
        # learned / knn / uncond
        mc = cs_counts(c_arr, s_arr, morning)
        x = (mc - Xtr_morning.mean(0)) / (Xtr_morning.std(0) + 1e-6)
        d = ((Xtr_n - x) ** 2).sum(1)
        kn = np.argsort(d)[:args.k]
        un = rng.choice(H, size=args.k, replace=False)
        V_kn[di], _ = evaluate(day, kn)
        V_un[di], _ = evaluate(day, un)
        V_ln[di], _ = evaluate(day, None)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": B, "baseline": baseline},
        "value": {"myopic": stat(V_my), "knn": stat(V_kn), "uncond": stat(V_un),
                  "learned": stat(V_ln), "oracle": stat(V_or)},
        "learned_minus_myopic": stat(V_ln - V_my),
        "learned_minus_knn": stat(V_ln - V_kn),
        "knn_minus_myopic": stat(V_kn - V_my),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
