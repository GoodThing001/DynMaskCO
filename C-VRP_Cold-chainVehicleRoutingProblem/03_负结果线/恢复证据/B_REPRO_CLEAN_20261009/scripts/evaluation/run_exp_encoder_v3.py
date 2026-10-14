"""真 3b · 步骤4 v3：所有臂共用 C0 合同/预算/完整认证 + 逐日配对输出。

相对 run_exp_encoder_v2.py 的修正：
  1. **接入 C0 绿色任务**（不再 coldchain_contract=None + 逐单 E_class）：所有臂
     （myopic / count-knn / frozen-H-knn / random-H-knn / clairvoyant）共用同一 C0 合同、
     预算 B=(1−ρ)×训练全接 C0 能耗、终局 c0_energy_from_traces 认证、接受预测 c0_marginal_energy。
  2. **逐日配对输出**：每臂逐日 profit + energy + hard_feasible + budget_violated，
     并给出 frozen−count / frozen−random / count−myopic 同日配对差（bootstrap CI）。
  3. 逐时钟 H-state（网格 cut 上 mean_pool，历史天同 cut 匹配）+ 随机初始化同架构 encoder 对照。

服务器臂（frozen-H-knn / random-H-knn）需 JAX/Flax + ckpts/.../step50000.ckpt（DynamicColdChainModel，
256 维，7D 输入）。本地只跑 myopic / count-knn / clairvoyant（C0，较慢）。

⚠️ 接受预测仍是「单温区×cooling_share」逐单代理，非整车全温区真实边际（见审计记录 est/true≈1.5）。
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
_COLD = os.path.join(_SCRIPTS, 'coldchain')
if _COLD not in sys.path:
    sys.path.insert(0, _COLD)

from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import (make_c0_contract, c0_energy_from_traces, c0_marginal_energy,
                               C0ReserveReplanner, evaluate_trace_c0, clairvoyant_profit_c0,
                               REV, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,
                               generate_dataset, _stat, _marginal_lambda)

DEFAULT_CKPT = os.path.join(_SCRIPTS, '..', 'ckpts', 'r1_5_baseline', 'typed_v1_edge',
                            'phase3c', 'seed42', 'step50000.ckpt')
GRID_STEP = 0.5


# --------------------------------------------------------------------------- #
# 逐时钟 H-state（服务器 JAX/Flax）
# --------------------------------------------------------------------------- #
def _grid_cuts(tau):
    return np.arange(0.0, tau + 1e-9, GRID_STEP)


def _cut_index(clock, tau, n_cuts):
    return min(int(min(float(clock), tau) / GRID_STEP), n_cuts - 1)


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
    from DynamicColdChainModel import DynamicColdChainModelConfig
    model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
    model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
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
    import jax.numpy as jnp
    cuts = _grid_cuts(tau)
    n_cuts = len(cuts)
    states = []
    for start in range(0, len(indices), chunk):
        idxs = indices[start:start + chunk]
        raws, vms = [], []
        for i in idxs:
            for cut in cuts:
                raw, vis = _instance_7d(dataset, i, cut, cap)
                raws.append(raw); vms.append(vis)
        raw_b = np.stack(raws, 0).astype(np.float32)
        vm_b = np.stack(vms, 0).astype(np.float32)
        H = np.asarray(encode_fn(jnp.asarray(raw_b), jnp.asarray(vm_b)))
        H = H.reshape(len(idxs), n_cuts, H.shape[1], H.shape[2])
        for bi in range(len(idxs)):
            i = idxs[bi]
            dem = dataset['demands'][i]
            cust = np.zeros(H.shape[2], dtype=bool)
            cust[1:] = dem[1:] > 0
            per_cut_state = []
            for ci in range(n_cuts):
                vm = (vm_b[bi * n_cuts + ci] > 0.5) & cust
                per_cut_state.append(H[bi, ci][vm].mean(0) if vm.sum() > 0 else H[bi, ci][0])
            states.append(np.stack(per_cut_state, 0))
    return np.stack(states, 0).astype(np.float64)


class EncoderKnnReplanner(C0ReserveReplanner):
    """逐时钟 H-state 做 k-NN 匹配（历史天同 cut）；边际 λ 能耗用 C0 c0_marginal_energy。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share, hist, k,
                 H_train, H_test, tau):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='knn', hist=hist, k=k)
        self.H_train = np.asarray(H_train, dtype=np.float64)
        self.H_test = np.asarray(H_test, dtype=np.float64)
        self.tau = tau
        self.n_cuts = self.H_train.shape[1]
        self.mu = self.H_train.mean(axis=0, keepdims=True)
        self.sd = self.H_train.std(axis=0, keepdims=True) + 1e-6
        self.Hn = (self.H_train - self.mu) / self.sd

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        ci = _cut_index(clock, self.tau, self.n_cuts)
        x = (self.H_test[inst_idx, ci] - self.mu[0, ci]) / self.sd[0, ci]
        d = ((self.Hn[:, ci] - x) ** 2).sum(1)
        kn = np.argsort(d)[:self.k]
        rem = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o2) for o2 in self._accepted]))
        lams = []
        for h in kn:
            ht, hc = self.hist[h]
            f = ht > clock
            fv = np.array([REV[int(c)] for c in hc[f]])
            fe = np.array([c0_marginal_energy(int(c), 2.0, self.cooling_share, self.contract)
                           for c in hc[f]])
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams))


# --------------------------------------------------------------------------- #
# 预算 + cooling_share（训练全接 C0 能耗）
# --------------------------------------------------------------------------- #
def compute_budget_and_dwell(train_ds, contract, num_vehicles, seed, rho, hist, k):
    """训练集「全接」C0 能耗 → 预算 B + 全温区制冷分摊 cooling_share（kWh/单）。"""
    cooling_share0 = 2.0  # 全接时 budget=inf，est 不用于决策，占位即可
    n_train = train_ds['coords'].shape[0]
    train_energy = np.zeros(n_train)
    total_orders = 0
    n_veh_used = 0
    for i in range(n_train):
        rp = C0ReserveReplanner(budget=float('inf'), capacity=50.0,
                                booking_horizon=BOOKING_HORIZON, contract=contract,
                                cooling_share=cooling_share0, mode='myopic', hist=hist, k=k)
        env = StrictOnlineEnv(train_ds, capacity=50.0, num_vehicles=num_vehicles,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        traces, _ = env.run(i)
        train_energy[i] = c0_energy_from_traces(traces)
        for t in traces:
            if t.final_coldchain_state is None:
                continue
            n_veh_used += 1
            total_orders += len(t.services)
    B = (1.0 - rho) * float(train_energy.mean())
    sum_cooling = float(sum(contract.thermal.cooling_power_kw))
    # 保守：每车运行时长上界 = booking_horizon（WAIT 到预约截止），制冷分摊 = 全温区功率 × 上界 / 每车单数
    mean_orders_per_vehicle = total_orders / max(n_veh_used, 1)
    cooling_share = sum_cooling * BOOKING_HORIZON / max(mean_orders_per_vehicle, 1)
    return B, cooling_share, float(train_energy.mean())


def run_replanner(test_ds, B, capacity, num_vehicles, contract, replanner):
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    n_test = test_ds['coords'].shape[0]
    profits, energies, fail_sets, budget_flags, n_served = [], [], [], [], []
    for i in range(n_test):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        profit, energy, fails, budget_violated = evaluate_trace_c0(
            i, env, traces, test_ds, accepted_i, B, contract)
        profits.append(profit)
        energies.append(energy)
        fail_sets.append(fails)
        budget_flags.append(budget_violated)
        n_served.append(len([s.node for t in traces for s in t.services]))
    return (np.array(profits), np.array(energies), fail_sets,
            np.array(budget_flags, dtype=bool), np.array(n_served))


def _stat_paired(x, seed):
    return _stat(x, seed)


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

    contract = make_c0_contract()
    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)

    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    B, cooling_share, train_energy_mean = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, args.seed, args.rho, hist, args.k)

    def fail_counts(f_sets):
        c = Counter()
        for f in f_sets:
            for r in f:
                c[r] += 1
        return dict(c)

    def hard_ok(f_sets):
        return np.array([len(f) == 0 for f in f_sets])

    arms = {}
    per_day = {}

    def record_arm(name, p, e, f, b):
        ok = hard_ok(f)
        arms[name] = {"feasible_rate": float(np.mean(ok & ~b)),
                      "hard_feasible_rate": float(np.mean(ok)),
                      "budget_violation_count": int(np.sum(b)),
                      "commitment_violation_count": int(fail_counts(f).get('commitment_violation', 0)),
                      "fail_counts": fail_counts(f),
                      "profit_hard_feasible": _stat(p[ok], args.seed),
                      "profit_budget_feasible": _stat(p[ok & ~b], args.seed)}
        per_day[name] = {"profit": p, "energy": e, "hard_feasible": ok,
                         "budget_violated": b}
        return ok

    # 本地 C0 三臂
    my = C0ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                            contract=contract, cooling_share=cooling_share, mode='myopic', hist=hist, k=args.k)
    p, e, f, b, ns = run_replanner(test_ds, B, args.capacity, args.num_vehicles, contract, my)
    my_ok = record_arm("myopic", p, e, f, b)

    ck = C0ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                            contract=contract, cooling_share=cooling_share, mode='knn', hist=hist, k=args.k)
    p, e, f, b, ns = run_replanner(test_ds, B, args.capacity, args.num_vehicles, contract, ck)
    ck_ok = record_arm("count_knn", p, e, f, b)

    p_cl, e_cl, b_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit_c0(test_ds, i, B, args.capacity,
                                                   args.num_vehicles, contract, cooling_share)
        p_cl.append(prof); e_cl.append(energy); b_cl.append(not feas)
    record_arm("clairvoyant", np.array(p_cl), np.array(e_cl), [set()] * args.test_instances,
               np.array(b_cl, dtype=bool))

    # 服务器臂
    if args.use_encoder:
        _setup_encoder_paths()
        ckpt = args.encoder_ckpt or DEFAULT_CKPT
        print('loading frozen encoder', ckpt, flush=True)
        frozen_fn, _ = load_frozen_encoder(ckpt)
        H_tr = build_clock_states(train_ds, list(range(args.train_instances)), frozen_fn,
                                  args.tau, cap=args.capacity)
        H_te = build_clock_states(test_ds, list(range(args.test_instances)), frozen_fn,
                                  args.tau, cap=args.capacity)
        print('frozen H_state', H_tr.shape, flush=True)
        ek = EncoderKnnReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                 contract=contract, cooling_share=cooling_share, hist=hist, k=args.k,
                                 H_train=H_tr, H_test=H_te, tau=args.tau)
        p, e, f, b, ns = run_replanner(test_ds, B, args.capacity, args.num_vehicles, contract, ek)
        ek_ok = record_arm("frozen_h_knn", p, e, f, b)

        print('building random-init same-arch encoder', flush=True)
        rand_fn, _ = load_random_encoder()
        R_tr = build_clock_states(train_ds, list(range(args.train_instances)), rand_fn,
                                  args.tau, cap=args.capacity)
        R_te = build_clock_states(test_ds, list(range(args.test_instances)), rand_fn,
                                  args.tau, cap=args.capacity)
        rk = EncoderKnnReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                 contract=contract, cooling_share=cooling_share, hist=hist, k=args.k,
                                 H_train=R_tr, H_test=R_te, tau=args.tau)
        p, e, f, b, ns = run_replanner(test_ds, B, args.capacity, args.num_vehicles, contract, rk)
        rk_ok = record_arm("random_h_knn", p, e, f, b)

    # 逐日配对
    def paired(a_name, b_name):
        if a_name not in per_day or b_name not in per_day:
            return None
        a, bb = per_day[a_name], per_day[b_name]
        ok = a["hard_feasible"] & bb["hard_feasible"]
        if ok.sum() == 0:
            return None
        return _stat(a["profit"][ok] - bb["profit"][ok], args.seed), int(ok.sum())

    paired_report = {}
    for a, b in [("count_knn", "myopic"), ("frozen_h_knn", "count_knn"),
                 ("frozen_h_knn", "random_h_knn"), ("frozen_h_knn", "myopic")]:
        r = paired(a, b)
        if r is not None:
            paired_report[f"{a}_minus_{b}"] = r[0]
            paired_report[f"{a}_minus_{b}_n"] = r[1]

    # 逐日数组落盘
    per_instance = []
    for i in range(args.test_instances):
        row = {"i": int(i)}
        for name in per_day:
            pd = per_day[name]
            row[name] = {
                "profit": (float(pd["profit"][i]) if np.isfinite(pd["profit"][i]) else None),
                "energy": float(pd["energy"][i]),
                "hard_feasible": bool(pd["hard_feasible"][i]),
                "budget_violated": bool(pd["budget_violated"][i]),
            }
        per_instance.append(row)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "train_accept_all_c0_energy_mean": train_energy_mean,
                   "cooling_share_kwh_per_order": float(cooling_share)},
        "arms": arms,
        "paired": paired_report,
        "per_instance": per_instance,
        "note": "步骤4 v3：所有臂共用 C0 合同/预算/完整认证；逐日配对输出；逐时钟 H-state + 随机同架构对照；"
                "接受预测仍是单温区×cooling_share 逐单代理（est/true≈1.5，见审计记录）。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
