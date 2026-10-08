"""A 步骤 3b：学习型条件场景生成 · 生成式采样版（条件多项分布）对比 k-NN。

上一版（计数回归 MSE+round）是 predict-then-optimize 失败。本版改为：
- 模型输出未来 token（class×spot + null）的**分布**（softmax 10 类），CE 训练。
- 采样：从分布做 V 次多项采样 → 未来多重集（多样本 = 生成式场景）。
- 策略：K 个场景各自的 λ 取均值（无偏，方差低于 k-NN 的 10 近邻）。

基线：myopic / 无条件 bootstrap / k-NN 条件 bootstrap / 学习型 / oracle。
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
N_TOK = 9  # class×spot

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


# ---------- 模型：context -> 10 类 logits ----------
def init_params(key, d_in=10, d_h=64, d_out=10):
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
    # y: (batch, 10) 归一化目标分布（true_counts / V）
    logits = forward(params, x)
    log_p = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y * log_p, axis=-1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=100)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-sample", type=int, default=10, help="学习型采样场景数")
    ap.add_argument("--rho", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    H = args.train_days
    hist_days = [generate_day(rng, float(rng.normal())) for _ in range(H)]
    hist = {"tarr": [d[0] for d in hist_days],
            "varr": [np.array([REV[int(c)] for c in d[1]]) for d in hist_days],
            "earr": [np.array([E_CS[int(c), int(s)] for c, s in zip(d[1], d[2])]) for d in hist_days]}
    Xtr_morning = np.array([cs_counts(d[1], d[2], d[0] <= args.tau) for d in hist_days])

    # 训练数据：context(10) -> 未来 token 分布(10，含 null)
    X_train, Y_train = [], []
    slice_times = np.linspace(args.tau, 13.0, args.slices)
    for d in hist_days:
        t_arr, c_arr, s_arr = d
        for t in slice_times:
            ctx = np.concatenate([cs_counts(c_arr, s_arr, t_arr <= args.tau), [t / 16.0]])
            fut = cs_counts(c_arr, s_arr, t_arr > t)
            n_fut = int(np.sum(fut))
            target = np.zeros(10)
            target[:9] = fut
            target[9] = max(0, V_SLOTS - n_fut)  # null token
            target = target / V_SLOTS
            X_train.append(ctx)
            Y_train.append(target)
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

    def predict_dist(ctx_np):
        logits = np.asarray(forward(params, jnp.asarray(ctx_np, dtype=jnp.float32)))
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def sample_counts(ctx_np, rng_local):
        p = predict_dist(ctx_np)
        toks = rng_local.choice(10, size=V_SLOTS, p=p)
        return np.bincount(toks, minlength=10)[:9]

    Xtr_n = (Xtr_morning - Xtr_morning.mean(0)) / (Xtr_morning.std(0) + 1e-6)
    test_days = [generate_day(rng, float(rng.normal())) for _ in range(args.test_days)]
    baseline = float(np.mean([hist["earr"][h].sum() for h in range(H)]))
    B = (1.0 - args.rho) * baseline

    def evaluate(day, neighbors, use_learned):
        t_arr, c_arr, s_arr = day
        morning = t_arr <= args.tau
        afternoon = ~morning
        V_m = float(np.sum([REV[int(c)] for c in c_arr[morning]]))
        E_m = float(np.sum([E_CS[int(c), int(s)] for c, s in zip(c_arr[morning], s_arr[morning])]))
        R0 = B - E_m
        all_v = np.array([REV[int(c)] for c in c_arr])
        all_e = np.array([E_CS[int(c), int(s)] for c, s in zip(c_arr, s_arr)])
        V_or = V_m + _knapsack_exact(all_v[afternoon], all_e[afternoon], R0)
        mc = cs_counts(c_arr, s_arr, morning)

        V = V_m
        rem = R0
        for i in np.where(afternoon)[0]:
            if rem <= 1e-12:
                break
            v_o, e_o = all_v[i], all_e[i]
            t = t_arr[i]
            if use_learned:
                ctx = np.concatenate([mc, [t / 16.0]])
                ctx = (ctx - mu) / sd
                lams = [lambda_from_counts(sample_counts(ctx, rng), rem)
                        for _ in range(args.k_sample)]
                lam = float(np.mean(lams))
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

    V_my = np.zeros(args.test_days); V_kn = np.zeros(args.test_days)
    V_un = np.zeros(args.test_days); V_ln = np.zeros(args.test_days); V_or = np.zeros(args.test_days)
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
        mc = cs_counts(c_arr, s_arr, morning)
        x = (mc - Xtr_morning.mean(0)) / (Xtr_morning.std(0) + 1e-6)
        d = ((Xtr_n - x) ** 2).sum(1)
        kn = np.argsort(d)[:args.k]
        un = rng.choice(H, size=args.k, replace=False)
        V_kn[di], _ = evaluate(day, kn, False)
        V_un[di], _ = evaluate(day, un, False)
        V_ln[di], _ = evaluate(day, None, True)

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
