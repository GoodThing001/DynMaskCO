"""去风险测试：高维状态能否让学习模型反超 k-NN？

背景：低维（9 维 morning 计数）下 learned 打不过 k-NN。假说：状态高维后 k-NN
维度灾难、learned（参数化平滑）反超——这正是"冻结 encoder 表征充分性"的机制。

做法：morning 状态 = (class, spot, time-bucket) 直方图，维度 = 9 × n_tb。
扫 n_tb ∈ {1,4,8}，对每个：训练 learned（MLP 采样）与 k-NN（同特征）对比。
判读：若 n_tb 增大时 learned − kNN 由负转正 → 高维状态假设成立 → 上 GPU 建本体。
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
V_SLOTS = 200

E_CS = np.zeros((3, 3))
for _c in range(3):
    for _s in range(3):
        E_CS[_c, _s] = 0.0373 + 0.0133 * DT[_c] * np.linalg.norm(CENTERS[_s] - DEPOT)


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
    class_probs = np.array([[0.1, 0.3, 0.6], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
    cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
    idx = np.argsort(times)
    return times[idx], cls[idx], spot[idx]


def morning_feature(times, cls, spot, tau, n_tb):
    m = times <= tau
    tb = np.clip((times / tau * n_tb).astype(int), 0, n_tb - 1)
    feat = np.zeros(9 * n_tb)
    for c in range(3):
        for s in range(3):
            for b in range(n_tb):
                feat[(c * 3 + s) * n_tb + b] = np.sum((cls == c) & (spot == s) & (tb == b) & m)
    return feat


def future_counts(cls, spot, times, t):
    m = times > t
    out = np.zeros(9)
    for c in range(3):
        for s in range(3):
            out[c * 3 + s] = np.sum((cls == c) & (spot == s) & m)
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


def init_params(key, d_in, d_h=64):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (d_in, d_h)) * 0.2
    b1 = jnp.zeros(d_h)
    W2 = jax.random.normal(k2, (d_h, 10)) * 0.2
    b2 = jnp.zeros(10)
    return (W1, b1, W2, b2)


def forward(params, x):
    W1, b1, W2, b2 = params
    h = jax.nn.relu(x @ W1 + b1)
    return h @ W2 + b2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-sample", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--n-tb", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    hist = {"tarr": [d[0] for d in hist_days],
            "varr": [np.array([REV[int(c)] for c in d[1]]) for d in hist_days],
            "earr": [np.array([E_CS[int(c), int(s)] for c, s in zip(d[1], d[2])]) for d in hist_days]}
    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]
    baseline = float(np.mean([hist["earr"][h].sum() for h in range(H)]))
    B = (1.0 - args.rho) * baseline

    report = {"config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
              "baseline": baseline, "B": B, "per_ntb": {}}

    for n_tb in args.n_tb:
        feat_dim = 9 * n_tb
        Xtr_m = np.array([morning_feature(d[0], d[1], d[2], args.tau, n_tb) for d in hist_days])

        X_train, Y_train = [], []
        slice_times = np.linspace(args.tau, 13.0, args.slices)
        for d in hist_days:
            t_arr, c_arr, s_arr = d
            mf = morning_feature(t_arr, c_arr, s_arr, args.tau, n_tb)
            for t in slice_times:
                ctx = np.concatenate([mf, [t / 16.0]])
                fut = future_counts(c_arr, s_arr, t_arr, t)
                n_fut = int(np.sum(fut))
                target = np.zeros(10)
                target[:9] = fut
                target[9] = max(0, V_SLOTS - n_fut)
                target /= V_SLOTS
                X_train.append(ctx)
                Y_train.append(target)
        X_train = np.array(X_train, dtype=np.float32)
        Y_train = np.array(Y_train, dtype=np.float32)
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
        X_train = (X_train - mu) / sd

        key = jax.random.PRNGKey(args.seed + n_tb)
        params = init_params(key, feat_dim + 1)
        opt = optax.adam(1e-2)
        opt_state = opt.init(params)

        @jax.jit
        def loss_fn(p, x, y):
            logits = forward(p, x)
            return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=-1))

        @jax.jit
        def step(p, os_, x, y):
            l, g = jax.value_and_grad(loss_fn)(p, x, y)
            u, os_ = opt.update(g, os_)
            return optax.apply_updates(p, u), os_, l

        rngb = np.random.default_rng(1)
        for it in range(args.steps):
            idx = rngb.integers(0, len(X_train), 64)
            params, opt_state, _ = step(params, opt_state, X_train[idx], Y_train[idx])

        def sample_counts(ctx_np):
            logits = np.asarray(forward(params, jnp.asarray(ctx_np, dtype=jnp.float32)))
            p = np.exp(logits - logits.max()); p /= p.sum()
            toks = rng.choice(10, size=V_SLOTS, p=p)
            return np.bincount(toks, minlength=10)[:9]

        Xtr_n = (Xtr_m - Xtr_m.mean(0)) / (Xtr_m.std(0) + 1e-6)

        V_kn = np.zeros(args.test_days)
        V_ln = np.zeros(args.test_days)
        V_my = np.zeros(args.test_days)
        V_or = np.zeros(args.test_days)
        for di, day in enumerate(test_days):
            t_arr, c_arr, s_arr = day
            morning = t_arr <= args.tau
            V_m = float(np.sum([REV[int(c)] for c in c_arr[morning]]))
            E_m = float(np.sum([E_CS[int(c), int(s)] for c, s in zip(c_arr[morning], s_arr[morning])]))
            R0 = B - E_m
            all_v = np.array([REV[int(c)] for c in c_arr])
            all_e = np.array([E_CS[int(c), int(s)] for c, s in zip(c_arr, s_arr)])
            afternoon = ~morning
            V_or[di] = V_m + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)
            rem = R0; v = V_m
            for i in np.where(afternoon)[0]:
                if all_e[i] <= rem + 1e-12:
                    v += all_v[i]; rem -= all_e[i]
            V_my[di] = v
            mf = morning_feature(t_arr, c_arr, s_arr, args.tau, n_tb)
            x = (mf - Xtr_m.mean(0)) / (Xtr_m.std(0) + 1e-6)
            d = ((Xtr_n - x) ** 2).sum(1)
            kn = np.argsort(d)[:args.k]

            def run_policy(use_learned):
                V = V_m; rem = R0
                for i in np.where(afternoon)[0]:
                    if rem <= 1e-12:
                        break
                    v_o, e_o = all_v[i], all_e[i]
                    t = t_arr[i]
                    if use_learned:
                        ctx = np.concatenate([mf, [t / 16.0]])
                        ctx = (ctx - mu) / sd
                        lams = [lambda_from_counts(sample_counts(ctx), rem) for _ in range(args.k_sample)]
                        lam = float(np.mean(lams))
                    else:
                        lams = []
                        for h in kn:
                            ht = hist["tarr"][h]; f = ht > t
                            lams.append(_marginal_lambda(hist["varr"][h][f], hist["earr"][h][f], rem))
                        lam = float(np.mean(lams))
                    if e_o <= rem + 1e-12 and v_o > e_o * lam + 1e-9:
                        V += v_o; rem -= e_o
                return V

            V_kn[di] = run_policy(False)
            V_ln[di] = run_policy(True)

        def stat(x):
            n = len(x); r = np.random.default_rng(args.seed + 777)
            m = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
            return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(m, 2.5)),
                    "ci_hi": float(np.percentile(m, 97.5))}

        report["per_ntb"][str(n_tb)] = {
            "feat_dim": feat_dim,
            "knn_minus_myopic": stat(V_kn - V_my),
            "learned_minus_myopic": stat(V_ln - V_my),
            "learned_minus_knn": stat(V_ln - V_kn),
            "value": {"myopic": stat(V_my), "knn": stat(V_kn), "learned": stat(V_ln), "oracle": stat(V_or)},
        }
        print(f"\nn_tb={n_tb} (feat_dim={feat_dim}):")
        print(f"  knn-myopic={np.mean(V_kn-V_my):.0f}  learned-myopic={np.mean(V_ln-V_my):.0f}  "
              f"learned-knn={np.mean(V_ln-V_kn):.0f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved:", os.path.join(args.out, "result.json"))
    return report


if __name__ == "__main__":
    main()
