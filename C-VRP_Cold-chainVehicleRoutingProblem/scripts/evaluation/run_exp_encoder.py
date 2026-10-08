"""真 3b 实验 3：冻结 encoder 高维表征 vs 3维计数 k-NN（最小原型）。

问题：冻结 MaskCO encoder（DynamicColdChainModel，256 维，7D 输入）的高维可见状态，
能否让 k-NN / 学习头（MLP 预测未来边际 λ）反超 3 维 morning 温区计数 k-NN。

⚠️ 信息时点泄漏（2026-09-24 核查，尚未修复）：
  - 本文件 `EncoderKnnReplanner._lookahead_lambda` / `_count_feat` / `build_encoder_states`
    / `_instance_7d` 都用**固定 tau=3h** 的可见掩码 `reveal <= tau` 构造状态，却在
    clock < tau 的更早决策点使用——把「第 3 小时才可见」的订单信息泄漏到更早的决策。
  - `_train_mlp` 的标签是 `reveal > tau` 的未来边际 λ，却把该头用于全天所有决策（clock < tau
    时输入缺当前时钟，输出也不对）。这不是掩码生成失败的证据，是训练/部署时点错配。
  - 修复（第 4 步的「同日配对实验」前必须做）：把 H_state / count / MLP 输入都改成
    clock 依赖，可见掩码 = `reveal <= min(clock, tau)`，历史天用同一时间截面匹配。
    这需要在每个决策 clock 重编码（服务器 JAX），本文件尚未实现。

  **唯一已修复的路径**：本地三路（myopic / k-NN(3维计数) / clairvoyant）import 的
  `AcceptRejectReplanner._lookahead_lambda` 已在 run_causal_accept_gate.py 修正为
  `min(clock, 3h)` 时间截面；`clairvoyant_profit` 也已加 reveal_time。故本文件
  未加 `--use-encoder` 时的输出是因果正确的；加 `--use-encoder`/`--control-random`
  的输出仍含上述时点泄漏，不能作为论文结论。

本地（无 --use-encoder，纯 NumPy）：myopic / k-NN(3维计数) / clairvoyant 三路，
与 run_causal_accept_gate.py 完全一致（复用其模块级函数/常量）。

服务器（--use-encoder，需 JAX/Flax + step50000.ckpt）：
  - 对每个 (train/test) 日，构建 7D 特征 [x,y,dem/cap,tw_start/tw_max,tw_end/tw_max,
    temp_class/2, reveal/tw_max]，用 visible_mask（reveal<=tau 门控未来）编码得 H，
    对可见订单节点 mean-pool 得 256 维状态向量 H_state。
  - encoder_knn：k-NN 用 H_state 匹配历史天（替代 3 维计数），边际 λ 口径不变。
  - encoder_mlp：2 层 numpy MLP [H_state + rem/B] -> λ，训练标签 = 未来订单（reveal>tau）
    在给定剩余预算下的 knapsack 边际 λ。
  - random_proj_knn（--control-random）：把 3 维计数随机投影到 256 维再做 k-NN，
    区分「encoder 结构」vs「只是维度变高」。

能耗逐单 E_class 简化（物理路由能耗=下一阶段）；不声明成功，只报数字和 caveat。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SIM = os.path.join(_SCRIPTS, 'simulation')
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from strict_online_env import StrictOnlineEnv
from run_causal_accept_gate import (
    REV, E_CLASS, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,
    generate_dataset, instance_full_energy, _marginal_lambda,
    AcceptRejectReplanner, evaluate_trace, _stat, clairvoyant_profit,
)

DEFAULT_CKPT = os.path.join(_SCRIPTS, '..', 'ckpts', 'r1_5_baseline', 'typed_v1_edge',
                            'phase3c', 'seed42', 'step50000.ckpt')


# --------------------------------------------------------------------------- #
# 纯 NumPy 三路（复用上游 run_causal_accept_gate 的常量与求值）
# --------------------------------------------------------------------------- #
def run_replanner(test_ds, B, capacity, num_vehicles, replanner):
    """给定一个 replanner 实例，跑严格在线轨迹，返回 (profits, energies, fail_sets, n_served)。"""
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=None, booking_horizon=BOOKING_HORIZON)
    profits, energies, fail_sets, n_served = [], [], [], []
    for i in range(test_ds['coords'].shape[0]):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        profit, energy, fails = evaluate_trace(i, env, traces, test_ds, accepted_i, B)
        profits.append(profit)
        energies.append(energy)
        fail_sets.append(fails)
        n_served.append(len([s.node for t in traces for s in t.services]))
    return np.array(profits), np.array(energies), fail_sets, np.array(n_served)


def _count_feat(dataset, inst_idx, tau):
    """3 维 morning 温区计数（与 run_causal_accept_gate._lookahead_lambda 口径一致）。"""
    m = dataset['reveal_time'][inst_idx] <= tau
    feat = np.zeros(3)
    for c in range(3):
        feat[c] = float(np.sum(dataset['temp_class'][inst_idx][m] == c))
    return feat


class EncoderKnnReplanner(AcceptRejectReplanner):
    """用高维状态 H_state 做 k-NN 匹配；边际 λ 计算口径与 3 维计数 k-NN 完全一致。"""

    def __init__(self, budget, capacity, booking_horizon, hist, k, H_train, H_test):
        super().__init__(budget, capacity, booking_horizon, mode='knn', hist=hist, k=k)
        self.H_train = np.asarray(H_train, dtype=np.float64)
        self.H_test = np.asarray(H_test, dtype=np.float64)
        self.mu = self.H_train.mean(0)
        self.sd = self.H_train.std(0)
        self.Hn = (self.H_train - self.mu) / (self.sd + 1e-6)

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        # ⚠️ 时点泄漏（未修）：self.H_test[inst_idx] 是固定 tau 快照，clock<tau 时含未揭示订单信息。
        # 第 4 步需改为 clock 依赖 H_state：可见掩码 = reveal <= min(clock, tau)，历史天同截面。
        x = (self.H_test[inst_idx] - self.mu) / (self.sd + 1e-6)
        d = ((self.Hn - x) ** 2).sum(1)
        kn = np.argsort(d)[:self.k]
        rem = self.budget - float(np.sum(
            [E_CLASS[int(env.dataset['temp_class'][inst_idx, o2])] for o2 in self._accepted]))
        lams = []
        for h in kn:
            ht, hc = self.hist[h]
            f = ht > clock
            fv = np.array([REV[int(c)] for c in hc[f]])
            fe = np.array([E_CLASS[int(c)] for c in hc[f]])
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams))


class MlpLambdaReplanner(AcceptRejectReplanner):
    """用训练好的 MLP 头 [H_state + rem/B] -> λ 预测未来边际，替代 k-NN 查表。"""

    def __init__(self, budget, capacity, booking_horizon, hist, mlp):
        super().__init__(budget, capacity, booking_horizon, mode='knn', hist=hist, k=1)
        self.mlp = mlp

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        rem = self.budget - float(np.sum(
            [E_CLASS[int(env.dataset['temp_class'][inst_idx, o2])] for o2 in self._accepted]))
        return float(_mlp_predict(self.mlp, inst_idx, rem, self.budget))


# --------------------------------------------------------------------------- #
# 冻结 encoder 状态构造（服务器 JAX/Flax）
# --------------------------------------------------------------------------- #
def _setup_encoder_paths():
    from project_paths import EXTENSION_ROOT, MASKCO_ROOT
    _CVRPTW = str(EXTENSION_ROOT)
    _MASKCO = str(MASKCO_ROOT)
    for _p in (_MASKCO, _CVRPTW, os.path.join(_CVRPTW, 'scripts', 'models'),
               os.path.join(_CVRPTW, 'scripts', 'simulation')):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    return _CVRPTW


def load_encoder(ckpt_path):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModelConfig
    from cvrptw_utils import coord_normalize_visible

    print('loading encoder', ckpt_path, flush=True)
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError('empty params from checkpoint')
    model = nnx.merge(nnx.graphdef(model), params)
    model_in = int(model.init_proj.kernel.shape[0])
    D = int(model.init_proj.kernel.shape[1]) if hasattr(model.init_proj.kernel, 'shape') else 256

    @jax.jit
    def encode_batch(raw, vm):
        raw = raw.at[..., :2].set(coord_normalize_visible(raw[..., :2], vm))
        return model.encode(raw[..., :model_in], visible_mask=vm)

    print(f'encoder loaded: input_dim={model_in}', flush=True)
    return encode_batch, model_in


def _instance_7d(dataset, i, tau, cap):
    coords = dataset['coords'][i].astype(np.float32)
    demands = dataset['demands'][i].astype(np.float32)
    tw_start = dataset['tw_start'][i].astype(np.float32)
    tw_end = dataset['tw_end'][i].astype(np.float32)
    reveal = dataset['reveal_time'][i].astype(np.float32)
    temp_class = dataset['temp_class'][i].astype(np.float32)
    tw_max = float(tw_end.max())
    f_list = [
        coords,
        (demands / cap)[..., None],
        (tw_start / tw_max)[..., None],
        (tw_end / tw_max)[..., None],
        (temp_class / 2.0)[..., None],
        (reveal / tw_max)[..., None],
    ]
    raw = np.concatenate(f_list, axis=-1)  # (N+1, 7)
    visible = (reveal <= tau).astype(np.float32)
    visible[0] = 1.0
    return raw, visible


def build_encoder_states(dataset, indices, encode_fn, tau, cap=50.0, chunk=32):
    """对每个 instance 构建 7D 特征、编码、对可见订单节点 mean-pool，返回 (n, D)。"""
    import jax.numpy as jnp
    states = []
    for start in range(0, len(indices), chunk):
        idxs = indices[start:start + chunk]
        raws, vms = [], []
        for i in idxs:
            raw, vis = _instance_7d(dataset, i, tau, cap)
            raws.append(raw)
            vms.append(vis)
        raw_b = np.stack(raws, 0).astype(np.float32)
        vm_b = np.stack(vms, 0).astype(np.float32)
        H = np.asarray(encode_fn(jnp.asarray(raw_b), jnp.asarray(vm_b)))  # (B, N+1, D)
        for bi in range(len(idxs)):
            i = idxs[bi]
            dem = dataset['demands'][i]
            cust = np.zeros(H.shape[1], dtype=bool)
            cust[1:] = dem[1:] > 0
            mask = (vm_b[bi] > 0.5) & cust
            if mask.sum() > 0:
                states.append(H[bi][mask].mean(0))
            else:
                states.append(H[bi][0])
    return np.stack(states, 0).astype(np.float64)


# --------------------------------------------------------------------------- #
# 2 层 numpy MLP：[zscore(H_state) ; rem/B] -> λ（回归未来边际）
# --------------------------------------------------------------------------- #
def _train_mlp(H_train, hist, B, tau, hidden, steps, seed):
    rng = np.random.default_rng(seed)
    mu = H_train.mean(0)
    sd = H_train.std(0) + 1e-6
    X0 = (H_train - mu) / sd
    D = H_train.shape[1]

    Xs, ys = [], []
    rem_fracs = [0.25, 0.5, 0.75, 1.0]
    for i in range(len(hist)):
        ht, hc = hist[i]
        f = ht > tau
        fv = np.array([REV[int(c)] for c in hc[f]])
        fe = np.array([E_CLASS[int(c)] for c in hc[f]])
        for rf in rem_fracs:
            Xs.append(np.concatenate([X0[i], [rf]]))
            ys.append(_marginal_lambda(fv, fe, rf * B))
    X = np.stack(Xs, 0).astype(np.float64)
    y = np.asarray(ys, dtype=np.float64).reshape(-1, 1)
    y_mean = float(y.mean())
    y_std = float(y.std()) + 1e-6
    yn = (y - y_mean) / y_std

    W1 = rng.normal(0.0, 0.1, (D + 1, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, 0.1, (hidden, 1))
    b2 = np.zeros(1)
    # Adam state
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    def adam(p, g, m, v, t, lr=2e-3):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        return p - lr * mh / (np.sqrt(vh) + 1e-8), m, v

    m = X.shape[0]
    last_loss = None
    for t in range(1, steps + 1):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        yhat = a1 @ W2 + b2
        loss = float(((yhat - yn) ** 2).mean())
        last_loss = loss
        dz2 = 2.0 * (yhat - yn) / m
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1.0 - np.tanh(z1) ** 2)
        gW1 = X.T @ dz1
        gb1 = dz1.sum(0)
        W1, mW1, vW1 = adam(W1, gW1, mW1, vW1, t)
        b1, mb1, vb1 = adam(b1, gb1, mb1, vb1, t)
        W2, mW2, vW2 = adam(W2, gW2, mW2, vW2, t)
        b2, mb2, vb2 = adam(b2, gb2, mb2, vb2, t)

    return dict(mu=mu, sd=sd, y_mean=y_mean, y_std=y_std, W1=W1, b1=b1, W2=W2, b2=b2,
                hidden=hidden, D=D, last_loss=last_loss)


def _mlp_predict(mlp, inst_idx, rem, budget):
    H = mlp['H_test'][inst_idx]
    x = np.concatenate([(H - mlp['mu']) / (mlp['sd'] + 1e-6), [max(0.0, rem) / max(budget, 1e-12)]])
    a1 = np.tanh(x @ mlp['W1'] + mlp['b1'])
    yn = float((a1 @ mlp['W2'] + mlp['b2'])[0])
    return max(0.0, yn * mlp['y_std'] + mlp['y_mean'])


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--test-instances", type=int, default=60)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--mlp-steps", type=int, default=500)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--use-encoder", action="store_true")
    ap.add_argument("--control-random", action="store_true")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)
    train_full = np.array([instance_full_energy(train_ds, i) for i in range(args.train_instances)])
    B = (1.0 - args.rho) * float(train_full.mean())

    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    def fail_counts(f_sets):
        c = Counter()
        for f in f_sets:
            for r in f:
                c[r] += 1
        return dict(c)

    def feasible_rate(f_sets):
        return float(np.mean([len(f) == 0 for f in f_sets]))

    # 三路（本地，纯 NumPy）
    base = AcceptRejectReplanner(budget=B, capacity=args.capacity,
                                 booking_horizon=BOOKING_HORIZON, mode='myopic',
                                 hist=hist, k=args.k)
    p_my, e_my, f_my, ns_my = run_replanner(test_ds, B, args.capacity, args.num_vehicles, base)
    base = AcceptRejectReplanner(budget=B, capacity=args.capacity,
                                 booking_horizon=BOOKING_HORIZON, mode='knn',
                                 hist=hist, k=args.k)
    p_kn, e_kn, f_kn, ns_kn = run_replanner(test_ds, B, args.capacity, args.num_vehicles, base)

    p_cl, e_cl, f_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit(test_ds, i, B, args.capacity, args.num_vehicles)
        p_cl.append(prof if feas else float('nan'))
        e_cl.append(energy)
        f_cl.append(set() if feas else {'clairvoyant_infeasible'})

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "train_full_energy_mean": float(train_full.mean())},
        "myopic": {"feasible_rate": feasible_rate(f_my), "fail_counts": fail_counts(f_my),
                   "profit": _stat(p_my, args.seed), "n_served_mean": float(np.mean(ns_my))},
        "knn": {"feasible_rate": feasible_rate(f_kn), "fail_counts": fail_counts(f_kn),
                "profit": _stat(p_kn, args.seed), "n_served_mean": float(np.mean(ns_kn))},
        "clairvoyant": {"feasible_rate": feasible_rate(f_cl), "fail_counts": fail_counts(f_cl),
                        "profit": _stat(np.array(p_cl), args.seed)},
    }
    both_ok = np.array([len(f_my[i]) == 0 and len(f_kn[i]) == 0
                        for i in range(args.test_instances)])
    if both_ok.sum() > 0:
        report["knn_minus_myopic_paired"] = _stat(p_kn[both_ok] - p_my[both_ok], args.seed)
        report["paired_n"] = int(both_ok.sum())

    def add_arm(name, p, f, ns):
        report[name] = {"feasible_rate": feasible_rate(f), "fail_counts": fail_counts(f),
                        "profit": _stat(p, args.seed), "n_served_mean": float(np.mean(ns))}

    if args.use_encoder:
        _setup_encoder_paths()
        ckpt = args.encoder_ckpt or DEFAULT_CKPT
        encode_fn, model_in = load_encoder(ckpt)
        print('building encoder states (train/test)...', flush=True)
        H_train = build_encoder_states(train_ds, list(range(args.train_instances)),
                                       encode_fn, args.tau, cap=args.capacity)
        H_test = build_encoder_states(test_ds, list(range(args.test_instances)),
                                      encode_fn, args.tau, cap=args.capacity)
        print(f'H_state dim = {H_train.shape[1]}, train {H_train.shape} test {H_test.shape}',
              flush=True)

        # encoder k-NN
        ek = EncoderKnnReplanner(budget=B, capacity=args.capacity,
                                 booking_horizon=BOOKING_HORIZON, hist=hist, k=args.k,
                                 H_train=H_train, H_test=H_test)
        p_ek, e_ek, f_ek, ns_ek = run_replanner(test_ds, B, args.capacity, args.num_vehicles, ek)
        add_arm("encoder_knn", p_ek, f_ek, ns_ek)

        # encoder MLP 头
        mlp = _train_mlp(H_train, hist, B, args.tau, args.mlp_hidden, args.mlp_steps, args.seed)
        mlp['H_test'] = H_test
        report["mlp_meta"] = {"last_loss": mlp['last_loss'], "hidden": mlp['hidden'],
                              "y_mean": mlp['y_mean'], "y_std": mlp['y_std']}
        ml = MlpLambdaReplanner(budget=B, capacity=args.capacity,
                                booking_horizon=BOOKING_HORIZON, hist=hist, mlp=mlp)
        p_ml, e_ml, f_ml, ns_ml = run_replanner(test_ds, B, args.capacity, args.num_vehicles, ml)
        add_arm("encoder_mlp", p_ml, f_ml, ns_ml)

        if args.control_random:
            rng = np.random.default_rng(args.seed + 999)
            C_train = np.stack([_count_feat(train_ds, i, args.tau)
                                for i in range(args.train_instances)], 0).astype(np.float64)
            C_test = np.stack([_count_feat(test_ds, i, args.tau)
                               for i in range(args.test_instances)], 0).astype(np.float64)
            proj = rng.normal(0.0, 1.0, (3, H_train.shape[1]))
            rk = EncoderKnnReplanner(budget=B, capacity=args.capacity,
                                     booking_horizon=BOOKING_HORIZON, hist=hist, k=args.k,
                                     H_train=C_train @ proj, H_test=C_test @ proj)
            p_rk, e_rk, f_rk, ns_rk = run_replanner(test_ds, B, args.capacity, args.num_vehicles, rk)
            add_arm("random_proj_knn", p_rk, f_rk, ns_rk)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
