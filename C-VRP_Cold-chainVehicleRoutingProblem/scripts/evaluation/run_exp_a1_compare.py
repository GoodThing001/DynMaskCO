"""A-v1 两臂对照：uncond_dynamic / rolling(滚动联合优化)，统一 v2 合同 + A-v1 评价 + 拒绝损失。

两臂统一接受硬认证（真实 C0 certify_plan）并共享无条件未来预算门槛 lam（训练集未来订单边际
value/e）；rolling 额外保留 relocate/swap 尾部局部搜索与 10s 时限。主表 p_c=(5,10,15)；
p_c=0 与 (10,20,30) 为敏感性（uncond_dynamic 决策与 p_c 无关，跑一次逐档重评效用）。
主比较 = MaskCO 事件掩码重构（待实现）vs 上述强对照。
A-v1 效用 = 服务收入 − 燃油 − 拒绝损失；硬约束 = 时间窗/容量/C0 能耗预算/承诺/鱼可售。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_contract import ColdChainContractV2
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1, REV
from run_exp_energy_c0 import (generate_dataset, _stat, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,
                               c0_energy_from_traces)
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_rule_control import UncondDynamicReplanner
from rolling_optimizer import RollingOptimizeReplanner
from strict_online_env import StrictOnlineEnv

REJECT_PENALTIES = {
    "p_c=0": {0: 0.0, 1: 0.0, 2: 0.0},
    "p_c=(5,10,15)": {0: 5.0, 1: 10.0, 2: 15.0},
    "p_c=(10,20,30)": {0: 10.0, 1: 20.0, 2: 30.0},
}
MAIN_PENALTY = "p_c=(5,10,15)"


def _single(ds, i):
    return {k: v[i] for k, v in ds.items()}


def run_raw(replanner, test_ds, contract, B, capacity, num_vehicles):
    """跑 env，逐实例返回 (traces, accepted, rejected) 原始结果 + 总耗时（不评效用）。"""
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    raw = []
    t0 = time.time()
    for i in range(test_ds['coords'].shape[0]):
        traces, _ = env.run(i)
        raw.append({'i': i, 'traces': traces,
                    'accepted': set(replanner._accepted),
                    'rejected': set(replanner._rejected)})
    elapsed = time.time() - t0
    return raw, elapsed


def eval_raw(raw, test_ds, contract, B, p_c):
    """用给定 p_c 评价原始轨迹，返回逐实例 A-v1 评价 dict。"""
    rows = []
    for r in raw:
        e = evaluate_trace_a1(r['traces'], _single(test_ds, r['i']), contract,
                              r['accepted'], r['rejected'], B, p_c)
        e['i'] = r['i']
        rows.append(e)
    return rows


def summarize(rows, B, seed, elapsed=None):
    p = np.array([r['utility'] for r in rows], float)
    ok = np.array([r['hard_feasible'] for r in rows])
    s = {
        "hard_feasible_rate": float(np.mean(ok)),
        "utility": _stat(p[ok], seed),
        "mean_n_served": float(np.mean([r['n_served'] for r in rows])),
        "mean_n_rejected": float(np.mean([r['n_rejected'] for r in rows])),
        "mean_fish_unsalable": float(np.mean([r['n_fish_unsalable'] for r in rows])),
        "service_rate": {c: float(np.mean([r['service_rate'][c] for r in rows]))
                         for c in ['0', '1', '2']},
        "reject_rate": {c: float(np.mean([r['reject_rate'][c] for r in rows]))
                        for c in ['0', '1', '2']},
        "mean_budget_usage": float(np.mean([r['energy_kwh'] / B for r in rows])),
    }
    if elapsed is not None:
        s["elapsed_s"] = float(elapsed)
    return s


def paired(a_rows, b_rows, seed):
    ok = np.array([r['hard_feasible'] for r in a_rows]) & np.array(
        [r['hard_feasible'] for r in b_rows])
    if ok.sum() == 0:
        return None
    pa = np.array([r['utility'] for r in a_rows], float)[ok]
    pb = np.array([r['utility'] for r in b_rows], float)[ok]
    return _stat(pa - pb, seed), int(ok.sum())


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
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    contract = ColdChainContractV2()
    contract.validate()
    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)
    test_ds = add_v2_initial_quality(test_ds, contract)

    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    B, cooling_share, train_energy_mean = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, args.seed, args.rho, hist, args.k)

    # uncond_dynamic：决策只依赖 lam 门槛 + certify，与 p_c 无关 → 跑一次，逐 p_c 只重评效用。
    ud = UncondDynamicReplanner(budget=B, capacity=args.capacity, booking_horizon=BOOKING_HORIZON,
                                contract=contract, cooling_share=cooling_share, hist=hist, k=args.k)
    ud_raw, ud_elapsed = run_raw(ud, test_ds, contract, B, args.capacity, args.num_vehicles)

    results = {}
    for pname, p_c in REJECT_PENALTIES.items():
        # rolling 决策依赖 p_c（reject_loss 进 _route_util）→ 逐 p_c 重跑
        ro = RollingOptimizeReplanner(budget=B, capacity=args.capacity,
                                      booking_horizon=BOOKING_HORIZON, contract=contract,
                                      cooling_share=cooling_share, reject_penalty=p_c,
                                      time_limit=args.time_limit, hist=hist, k=args.k)
        ro_raw, ro_elapsed = run_raw(ro, test_ds, contract, B, args.capacity, args.num_vehicles)

        ud_rows = eval_raw(ud_raw, test_ds, contract, B, p_c)
        ro_rows = eval_raw(ro_raw, test_ds, contract, B, p_c)

        block = {
            "reject_penalty": {str(k): float(v) for k, v in p_c.items()},
            "arms": {
                "uncond_dynamic": summarize(ud_rows, B, args.seed, ud_elapsed),
                "rolling": summarize(ro_rows, B, args.seed, ro_elapsed),
            },
        }
        r = paired(ro_rows, ud_rows, args.seed)
        if r is not None:
            block["paired"] = {"rolling_minus_uncond_dynamic": r[0],
                               "rolling_minus_uncond_dynamic_n": r[1]}
        results[pname] = block

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "cooling_share": float(cooling_share),
                   "train_accept_all_c0_energy_mean": float(train_energy_mean)},
        "main_penalty": MAIN_PENALTY,
        "results": results,
        "note": "A-v1 两臂对照（非学习）：uncond_dynamic / rolling(relocate+swap+certify)。"
                "两臂统一接受硬认证（certify_plan）并共享无条件未来预算门槛 lam；"
                "rolling 额外保留尾部局部搜索。效用=收入−燃油−拒绝损失；硬约束=TW/容量/C0能耗预算/承诺/鱼可售。"
                "主表 p_c=(5,10,15)；p_c=0 与 (10,20,30) 为敏感性。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
