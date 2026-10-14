"""真 3b · 强规则对照门：隔离 count-kNN 的 +349 中「筛选」vs「条件信息」的贡献。

问题：+349 = count-kNN − accept-if-feasible myopic。myopic 是无筛选弱基线，+349 可能主要来自
「筛选高 value/e 订单」（规则可捕获），而非「当天早期计数的条件预测」（学习可捕获）。

这道门加两个**因果、无条件（不看当天早期）、训练集校准**的强规则：
  1. fixed_threshold：固定 value/e 阈值 τ*，接受 REV[o]/e > τ*。
     τ* 从训练集离线 value/e 分布校准（累加 est_energy 到每天预算 B 的临界 value/e）。
  2. class_priority：按类别固定优先级（value/e 降序 = class 0 > 1 > 2）贪心接受。

主比较 = count_kNN − max(fixed_threshold, class_priority)（同日配对）。若不再显著为正，
则「条件信息」相对简单预留规则无独立价值，方向 A 的方法主张在该合成任务上收束。

所有臂共用 C0 合同/预算/完整认证；逐日输出利润、能耗、可行率、服务类别构成、路程、品质损失。
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
from run_exp_energy_c0 import (make_c0_contract, c0_marginal_energy, C0ReserveReplanner,
                               clairvoyant_profit_c0, c0_energy_from_traces, evaluate_trace_c0,
                               REV, PVAL, K_EFF, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,
                               FUEL_COST_PER_KM, generate_dataset, _stat, _marginal_lambda)
from run_exp_encoder_v3 import compute_budget_and_dwell


class FixedThresholdReplanner(C0ReserveReplanner):
    """固定 value/e 阈值 τ（无条件、训练集校准）：接受 REV[o]/e > τ。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share, tau, hist, k=10):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='myopic', hist=hist, k=k)
        self.tau = tau

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        tc = env.dataset['temp_class']
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        remaining = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o) for o in self._accepted]))
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(env, inst_idx, o)
            c = int(tc[inst_idx, o])
            if e > remaining + 1e-9:
                self._rejected.add(o); continue
            if REV[c] / max(e, 1e-12) <= self.tau:
                self._rejected.add(o); continue
            if not self._try_insert(env, inst_idx, o, vehicles, served_mask, clock):
                self._rejected.add(o); continue
            self._accepted.add(o)
            remaining -= e


class ClassPriorityReplanner(C0ReserveReplanner):
    """类别/价值-能耗优先级：接受 value/e > 1.0（价值覆盖能耗）的订单，同类按 reveal 顺序。

    τ=1.0 是先验固定阈值（不接受价值低于能耗的订单），对应 class 0/1（value/e≈2.7/1.25）
    优先于 class 2（value/e≈0.99）。区别于 fixed_threshold 的数据驱动 τ*。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share, hist, k=10):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='myopic', hist=hist, k=k)

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        tc = env.dataset['temp_class']
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        remaining = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o) for o in self._accepted]))
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(env, inst_idx, o)
            c = int(tc[inst_idx, o])
            if e > remaining + 1e-9:
                self._rejected.add(o); continue
            if REV[c] / max(e, 1e-12) <= 1.0:
                self._rejected.add(o); continue
            if not self._try_insert(env, inst_idx, o, vehicles, served_mask, clock):
                self._rejected.add(o); continue
            self._accepted.add(o)
            remaining -= e


class UncondDynamicReplanner(C0ReserveReplanner):
    """无条件动态前瞻：用全部训练日的未来订单算边际 value/e 门槛（不按当天早期计数选邻居）。

    接受 = 门槛（REV/e > lam）+ 真实 C0 certify_plan 硬认证（与 rolling 同口径）。
    """

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share, hist, k=10):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='knn', hist=hist, k=k)

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        return self._uncond_lookahead_lambda(env, inst_idx, clock)

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(env, inst_idx, o)
            lam = self._lookahead_lambda(env, inst_idx, clock, o)
            if REV[int(env.dataset['temp_class'][inst_idx, o])] <= e * lam + 1e-9:
                self._rejected.add(o)
                continue
            if not self._try_insert_certified(env, inst_idx, o, vehicles, served_mask, clock):
                self._rejected.add(o)
                continue
            self._accepted.add(o)


def calibrate_tau(train_ds, contract, cooling_share, B):
    """训练集离线 value/e 分布 → 固定阈值 τ*（累加 est_energy 到每天预算 B 的临界 value/e）。"""
    vals, energies = [], []
    n = train_ds['coords'].shape[0]
    for i in range(n):
        dem = train_ds['demands'][i]; tc = train_ds['temp_class'][i]
        for o in range(1, dem.shape[0]):
            if dem[o] > 0:
                c = int(tc[o]); q = float(dem[o])
                e = c0_marginal_energy(c, q, cooling_share, contract)
                vals.append(REV[c] / e); energies.append(e)
    vals = np.array(vals, float); energies = np.array(energies, float)
    order = np.argsort(-vals)
    cum = np.cumsum(energies[order])
    target = B * n  # 总预算
    idx = int(np.searchsorted(cum, target))
    idx = min(idx, len(vals) - 1)
    return float(vals[order][idx])


def evaluate_full(inst_idx, env, traces, dataset, accepted_set, B, contract):
    profit, energy, fails, budget_violated = evaluate_trace_c0(
        inst_idx, env, traces, dataset, accepted_set, B, contract)
    tc = dataset['temp_class'][inst_idx]
    served = [s.node for t in traces for s in t.services]
    n_served = len(served)
    class_counts = {c: int(sum(1 for o in served if int(tc[o]) == c)) for c in range(3)}
    total_distance_km = 0.0
    total_quality = 0.0
    for t in traces:
        if not t.services:
            continue
        d_units = sum(float(env.dist_mat[inst_idx, s.prev_node, s.node]) for s in t.services)
        d_units += float(env.dist_mat[inst_idx, t.services[-1].node, 0])
        total_distance_km += d_units * KM_PER_UNIT
        ret = float(t.return_arrival)
        for s in t.services:
            dwell = max(0.0, ret - float(s.service_finish))
            o = s.node
            total_quality += PVAL[int(tc[o])] * dataset['demands'][inst_idx, o] * (
                1.0 - np.exp(-K_EFF[int(tc[o])] * dwell))
    return dict(profit=profit, energy=energy, fails=fails, budget_violated=budget_violated,
                n_served=n_served, class_counts=class_counts,
                distance_km=total_distance_km, quality_loss=total_quality)


def run_full(test_ds, B, capacity, num_vehicles, contract, replanner):
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    n_test = test_ds['coords'].shape[0]
    rows = []
    for i in range(n_test):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        r = evaluate_full(i, env, traces, test_ds, accepted_i, B, contract)
        r['i'] = i
        rows.append(r)
    return rows


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
    tau_star = calibrate_tau(train_ds, contract, cooling_share, B)

    arms = {}
    rows_by_arm = {}

    def record(name, replanner):
        rows = run_full(test_ds, B, args.capacity, args.num_vehicles, contract, replanner)
        rows_by_arm[name] = rows
        p = np.array([r['profit'] for r in rows], float)
        e = np.array([r['energy'] for r in rows], float)
        ok = np.array([len(r['fails']) == 0 for r in rows])
        bv = np.array([r['budget_violated'] for r in rows])
        arms[name] = {
            "feasible_rate": float(np.mean(ok & ~bv)),
            "hard_feasible_rate": float(np.mean(ok)),
            "budget_violation_count": int(np.sum(bv)),
            "commitment_violation_count": int(sum(1 for r in rows if 'commitment_violation' in r['fails'])),
            "profit_hard_feasible": _stat(p[ok], args.seed),
            "mean_energy": float(e[ok].mean()) if ok.sum() else float('nan'),
            "mean_n_served": float(np.mean([r['n_served'] for r in rows])),
            "mean_class_counts": {str(c): float(np.mean([r['class_counts'][c] for r in rows])) for c in range(3)},
            "mean_distance_km": float(np.mean([r['distance_km'] for r in rows])),
            "mean_quality_loss": float(np.mean([r['quality_loss'] for r in rows])),
        }
        return ok

    record("myopic", C0ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                        contract=contract, cooling_share=cooling_share, mode='myopic', hist=hist, k=args.k))
    record("fixed_threshold", FixedThresholdReplanner(budget=B, capacity=args.capacity,
                                                      booking_horizon=BOOKING_HORIZON, contract=contract,
                                                      cooling_share=cooling_share, tau=tau_star, hist=hist, k=args.k))
    record("class_priority", ClassPriorityReplanner(budget=B, capacity=args.capacity,
                                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                                    cooling_share=cooling_share, hist=hist, k=args.k))
    record("count_knn", C0ReserveReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                           contract=contract, cooling_share=cooling_share, mode='knn', hist=hist, k=args.k))
    record("uncond_dynamic", UncondDynamicReplanner(budget=B, capacity=args.capacity,
                                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                                    cooling_share=cooling_share, hist=hist, k=args.k))

    # clairvoyant（离线贪心，逐日）
    cl_rows = []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit_c0(test_ds, i, B, args.capacity,
                                                   args.num_vehicles, contract, cooling_share)
        cl_rows.append(dict(profit=prof, energy=energy, budget_violated=not feas))
    arms["clairvoyant"] = {
        "feasible_rate": float(np.mean([not r['budget_violated'] for r in cl_rows])),
        "profit_hard_feasible": _stat(np.array([r['profit'] for r in cl_rows], float), args.seed),
        "mean_energy": float(np.mean([r['energy'] for r in cl_rows])),
    }
    rows_by_arm["clairvoyant"] = cl_rows

    # 同日配对（count_knn vs 各规则 / myopic）
    def paired(a_name, b_name):
        a = rows_by_arm[a_name]; bb = rows_by_arm[b_name]
        aok = np.array([len(r['fails']) == 0 for r in a])
        bok = np.array([len(r['fails']) == 0 for r in bb])
        ok = aok & bok
        if ok.sum() == 0:
            return None
        pa = np.array([r['profit'] for r in a], float)[ok]
        pb = np.array([r['profit'] for r in bb], float)[ok]
        return _stat(pa - pb, args.seed), int(ok.sum())

    paired_report = {}
    for a, b in [("count_knn", "myopic"), ("count_knn", "fixed_threshold"),
                 ("count_knn", "class_priority"), ("count_knn", "uncond_dynamic"),
                 ("uncond_dynamic", "fixed_threshold"), ("uncond_dynamic", "myopic"),
                 ("fixed_threshold", "myopic"), ("class_priority", "myopic")]:
        r = paired(a, b)
        if r is not None:
            paired_report[f"{a}_minus_{b}"] = r[0]
            paired_report[f"{a}_minus_{b}_n"] = r[1]

    per_instance = []
    for i in range(args.test_instances):
        row = {"i": int(i)}
        for name in rows_by_arm:
            r = rows_by_arm[name][i]
            row[name] = {
                "profit": (float(r['profit']) if np.isfinite(r['profit']) else None),
                "energy": float(r['energy']),
                "budget_violated": bool(r['budget_violated']),
            }
        per_instance.append(row)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "train_accept_all_c0_energy_mean": train_energy_mean,
                   "cooling_share_kwh_per_order": float(cooling_share),
                   "tau_star": tau_star},
        "arms": arms,
        "paired": paired_report,
        "per_instance": per_instance,
        "note": "强规则对照门：fixed_threshold(τ*=训练集离线value/e临界) + class_priority(类别0>1>2)；"
                "主比较 count_knn − 最强规则；隔离「筛选」vs「条件信息」价值。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
