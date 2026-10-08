"""真 3b · 步骤4：逐时钟 H-state 的四臂同日配对实验（冻结 H / 随机同架构 H / 显式计数 / H-kNN）。

目的：判断「预训练 MaskCO 表征改善动态承诺的条件场景前瞻」是否有一个**稳定、可行的学习空间**
（冻结 H 必须稳定优于随机同架构 H 和 3 维显式计数，才值得投入真正的掩码生成器）。

相对 run_exp_encoder.py（旧版）的修正：
  1. **逐时钟 H-state**：不再用固定 tau=3 快照。对每个实例在网格 cut ∈ {0,0.5,...,tau} 上预计算
     H_state(cut) = mean_pool(encode(7D, visible = reveal <= cut))；决策时刻用
     cut = min(clock, tau) 向下取整到网格，历史天用**同一 cut** 匹配。消除旧版的时点泄漏。
  2. **随机对照 = 随机初始化同架构 encoder**（同 DynamicColdChainModel、同 7D 输入、256 维），
     而非「3 维计数随机投影」。区分「预训练表征」vs「只是高维/网络结构」。
  3. 同日配对：所有臂在同一批 test 实例上跑，按天配对差 + 可行率。

本地（无 --use-encoder）：myopic / count-knn（3 维显式计数）/ clairvoyant 三路（复用 run_exp_reserve 的
持久预留 + 因果口径）。服务器（--use-encoder）：frozen-H-knn / random-H-knn 两臂。

⚠️ 服务器臂需 JAX/Flax + ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt
（DynamicColdChainModel，256 维，7D 输入）。本脚本本地只跑三路；服务器命令见 main() docstring。
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

from run_exp_reserve import (REV, PVAL, K_EFF, E_CLASS, BOOKING_HORIZON, SPEED_KMH,
                             KM_PER_UNIT, FUEL_COST_PER_KM, generate_dataset,
                             instance_full_energy, _stat, _marginal_lambda,
                             ReserveReplanner, evaluate_trace, clairvoyant_profit)

DEFAULT_CKPT = os.path.join(_SCRIPTS, '..', 'ckpts', 'r1_5_baseline', 'typed_v1_edge',
                            'phase3c', 'seed42', 'step50000.ckpt')
GRID_STEP = 0.5


def run_replanner(test_ds, B, capacity, num_vehicles, replanner):
    from strict_online_env import StrictOnlineEnv
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


# --------------------------------------------------------------------------- #
# 逐时钟 H-state（服务器 JAX/Flax）
# --------------------------------------------------------------------------- #
def _grid_cuts(tau):
    return np.arange(0.0, tau + 1e-9, GRID_STEP)


def _cut_index(clock, tau, n_cuts):
    idx = int(min(float(clock), tau) / GRID_STEP)
    return min(idx, n_cuts - 1)


def _setup_encoder_paths():
    from project_paths import EXTENSION_ROOT, MASKCO_ROOT
    for _p in (str(MASKCO_ROOT), str(EXTENSION_ROOT),
               os.path.join(str(EXTENSION_ROOT), 'scripts', 'models'),
               os.path.join(str(EXTENSION_ROOT), 'scripts', 'simulation')):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    return str(EXTENSION_ROOT)


def _make_encode_fn(model):
    import jax
    import jax.numpy as jnp
    from cvrptw_utils import coord_normalize_visible
    model_in = int(model.init_proj.kernel.shape[0])

    @jax.jit
    def encode_batch(raw, vm):
        raw = raw.at[..., :2].set(coord_normalize_visible(raw[..., :2], vm))
        return model.encode(raw[..., :model_in], visible_mask=vm)
    return encode_batch, model_in


def load_frozen_encoder(ckpt_path):
    import jax
    from flax import nnx
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModelConfig

    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError('empty params from checkpoint')
    model = nnx.merge(nnx.graphdef(model), params)
    return _make_encode_fn(model)


def load_random_encoder():
    """随机初始化同架构 encoder（不加载任何预训练权重）。"""
    from DynamicColdChainModel import DynamicColdChainModelConfig
    model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
    model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()  # 随机初始化
    return _make_encode_fn(model)


def _instance_7d(dataset, i, cut, cap):
    coords = dataset['coords'][i].astype(np.float32)
    demands = dataset['demands'][i].astype(np.float32)
    tw_start = dataset['tw_start'][i].astype(np.float32)
    tw_end = dataset['tw_end'][i].astype(np.float32)
    reveal = dataset['reveal_time'][i].astype(np.float32)
    temp_class = dataset['temp_class'][i].astype(np.float32)
    tw_max = float(tw_end.max())
    f_list = [coords, (demands / cap)[..., None], (tw_start / tw_max)[..., None],
              (tw_end / tw_max)[..., None], (temp_class / 2.0)[..., None],
              (reveal / tw_max)[..., None]]
    raw = np.concatenate(f_list, axis=-1)
    visible = (reveal <= cut).astype(np.float32)
    visible[0] = 1.0
    return raw, visible


def build_clock_states(dataset, indices, encode_fn, tau, cap=50.0, chunk=32):
    """对每个 instance 在网格 cut 上预计算 H_state（(n_inst, n_cuts, D)）。"""
    import jax.numpy as jnp
    cuts = _grid_cuts(tau)
    n_cuts = len(cuts)
    states = []
    for start in range(0, len(indices), chunk):
        idxs = indices[start:start + chunk]
        per_inst = []
        for i in idxs:
            per_cut = []
            for cut in cuts:
                raw, vis = _instance_7d(dataset, i, cut, cap)
                per_cut.append((raw, vis))
            per_inst.append(per_cut)
        # 展平 (B*n_cuts) 编码
        raws, vms = [], []
        for per_inst_i in per_inst:
            for raw, vis in per_inst_i:
                raws.append(raw); vms.append(vis)
        raw_b = np.stack(raws, 0).astype(np.float32)
        vm_b = np.stack(vms, 0).astype(np.float32)
        H = np.asarray(encode_fn(jnp.asarray(raw_b), jnp.asarray(vm_b)))  # (B*n_cuts, N+1, D)
        H = H.reshape(len(idxs), n_cuts, H.shape[1], H.shape[2])
        for bi in range(len(idxs)):
            i = idxs[bi]
            dem = dataset['demands'][i]
            cust = np.zeros(H.shape[2], dtype=bool)
            cust[1:] = dem[1:] > 0
            per_cut_state = []
            for ci in range(n_cuts):
                vm = (vm_b[bi * n_cuts + ci] > 0.5) & cust
                if vm.sum() > 0:
                    per_cut_state.append(H[bi, ci][vm].mean(0))
                else:
                    per_cut_state.append(H[bi, ci][0])
            states.append(np.stack(per_cut_state, 0))  # (n_cuts, D)
    return np.stack(states, 0).astype(np.float64)  # (n_inst, n_cuts, D)


class EncoderKnnReplanner(ReserveReplanner):
    """用逐时钟 H_state 做 k-NN 匹配（历史天同 cut）；边际 λ 口径与 count-kNN 一致。"""

    def __init__(self, budget, capacity, booking_horizon, hist, k, H_train, H_test, tau):
        super().__init__(budget, capacity, booking_horizon, mode='knn', hist=hist, k=k)
        self.H_train = np.asarray(H_train, dtype=np.float64)  # (n_train, n_cuts, D)
        self.H_test = np.asarray(H_test, dtype=np.float64)    # (n_test, n_cuts, D)
        self.tau = tau
        self.n_cuts = self.H_train.shape[1]
        # 按 cut 归一化（每 cut 独立 zscore）
        self.mu = self.H_train.mean(axis=0, keepdims=True)  # (1, n_cuts, D)
        self.sd = self.H_train.std(axis=0, keepdims=True) + 1e-6
        self.Hn = (self.H_train - self.mu) / self.sd       # (n_train, n_cuts, D)

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        ci = _cut_index(clock, self.tau, self.n_cuts)
        x = (self.H_test[inst_idx, ci] - self.mu[0, ci]) / self.sd[0, ci]
        d = ((self.Hn[:, ci] - x) ** 2).sum(1)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--test-instances", type=int, default=40)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--use-encoder", action="store_true")
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

    def add_arm(report, name, p, f, ns):
        ok = np.array([len(x) == 0 for x in f])
        arm = {"feasible_rate": feasible_rate(f), "feasible_n": int(ok.sum()),
               "fail_counts": fail_counts(f), "profit": _stat(p, args.seed)}
        if ns is not None:
            arm["n_served_mean"] = float(np.mean(ns))
        report[name] = arm
        return ok

    report = {"config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
              "budget": {"B": float(B), "train_full_energy_mean": float(train_full.mean())}}

    # 本地三路（count-kNN 用 ReserveReplanner 的 3 维计数前瞻，因果口径）
    my = ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                          mode='myopic', hist=hist, k=args.k)
    p_my, e_my, f_my, ns_my = run_replanner(test_ds, B, args.capacity, args.num_vehicles, my)
    my_ok = add_arm(report, "myopic", p_my, f_my, ns_my)

    kn = ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                          mode='knn', hist=hist, k=args.k)
    p_kn, e_kn, f_kn, ns_kn = run_replanner(test_ds, B, args.capacity, args.num_vehicles, kn)
    kn_ok = add_arm(report, "count_knn", p_kn, f_kn, ns_kn)

    p_cl, e_cl, f_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit(test_ds, i, B, args.capacity, args.num_vehicles)
        p_cl.append(prof if feas else float('nan'))
        e_cl.append(energy)
        f_cl.append(set() if feas else {'clairvoyant_infeasible'})
    cl_ok = add_arm(report, "clairvoyant", np.array(p_cl), f_cl, None)

    # 服务器臂：frozen-H-knn / random-H-knn（逐时钟 H_state）
    if args.use_encoder:
        _setup_encoder_paths()
        ckpt = args.encoder_ckpt or DEFAULT_CKPT
        print('loading frozen encoder', ckpt, flush=True)
        frozen_fn, _ = load_frozen_encoder(ckpt)
        H_tr = build_clock_states(train_ds, list(range(args.train_instances)), frozen_fn, args.tau,
                                  cap=args.capacity)
        H_te = build_clock_states(test_ds, list(range(args.test_instances)), frozen_fn, args.tau,
                                  cap=args.capacity)
        print('frozen H_state dim', H_tr.shape, flush=True)
        ek = EncoderKnnReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                 hist=hist, k=args.k, H_train=H_tr, H_test=H_te, tau=args.tau)
        p_ek, e_ek, f_ek, ns_ek = run_replanner(test_ds, B, args.capacity, args.num_vehicles, ek)
        ek_ok = add_arm(report, "frozen_h_knn", p_ek, f_ek, ns_ek)

        print('building random-init same-arch encoder', flush=True)
        rand_fn, _ = load_random_encoder()
        R_tr = build_clock_states(train_ds, list(range(args.train_instances)), rand_fn, args.tau,
                                  cap=args.capacity)
        R_te = build_clock_states(test_ds, list(range(args.test_instances)), rand_fn, args.tau,
                                  cap=args.capacity)
        rk = EncoderKnnReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                 hist=hist, k=args.k, H_train=R_tr, H_test=R_te, tau=args.tau)
        p_rk, e_rk, f_rk, ns_rk = run_replanner(test_ds, B, args.capacity, args.num_vehicles, rk)
        rk_ok = add_arm(report, "random_h_knn", p_rk, f_rk, ns_rk)

    # 同日配对（count-kNN 为基准）
    both = kn_ok & my_ok
    if both.sum() > 0:
        report["count_knn_minus_myopic_paired"] = _stat(p_kn[both] - p_my[both], args.seed)
        report["paired_n"] = int(both.sum())

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
