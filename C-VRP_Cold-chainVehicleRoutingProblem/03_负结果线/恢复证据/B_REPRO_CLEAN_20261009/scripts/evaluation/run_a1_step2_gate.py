"""A-v1 步骤 2 无训练条件信息门驱动（三臂共用 SAA 决策闭环的按天配对比较）。

锁定（协议 §2.6）：
  - 臂 1 uncond_hist / 臂 2 cond_hist / 臂 3 explicit_feat，共用 SaaReplanner（同下游）；
  - train seed 20260925（200 天历史）、gate seed 20260926（40 天，全新未看）；
  - K=10 场景/决策、每决策 10s、超时统一降级（保计划+拒单）；
  - 主 p_c=(5,10,15)；p_c=0 与 (10,20,30) 为独立决策运行；
  - 不筛天：逐日保存全部结果与违规类型；存在硬违规 = 不宣称过门；
  - 主比较 = cond_hist − uncond_hist：均值 ≥ δ*=20 且 bootstrap 95% CI 下界 > 0。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, _stat, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history, build_history_times_classes,
                          make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

PENALTIES = {
    "p_c=0": {0: 0.0, 1: 0.0, 2: 0.0},
    "p_c=(5,10,15)": {0: 5.0, 1: 10.0, 2: 15.0},
    "p_c=(10,20,30)": {0: 10.0, 1: 20.0, 2: 30.0},
}
MAIN_PENALTY = "p_c=(5,10,15)"
DELTA_STAR = 20.0
TRAIN_SEED = 20260925
GATE_SEED = 20260926
FINAL_TEST_SEEDS = (20260927, 20260928)  # 只声明，不跑
ARM_SEEDS = {"uncond_hist": 7001, "cond_hist": 7002, "explicit_feat": 7003}


def _single(ds, i):
    return {k: v[i] for k, v in ds.items()}


def run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
            sampler, p_c, K, time_limit, arm_seed):
    rp = SaaReplanner(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=cooling_share, sampler=sampler,
                      reject_penalty=p_c, K=K, time_limit=time_limit, arm_seed=arm_seed)
    env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    rows = []
    for i in range(gate_ds['coords'].shape[0]):
        t0 = time.time()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract, acc, rej, B, p_c)
        rows.append(dict(
            i=int(i),
            utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
            hard_feasible=bool(ev['hard_feasible']),
            failures=sorted(set(ev['failures'])),
            n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
            energy_kwh=float(ev['energy_kwh']),
            budget_violated=bool(ev['budget_violated']),
            timeouts=int(rp.timeouts),
            elapsed_s=float(time.time() - t0)))
    return rows


def _worker(task):
    """(pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
        sampler, p_c, K, time_limit, arm_seed) -> (pname, name, rows)"""
    (pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
     sampler, p_c, K, time_limit, arm_seed) = task
    rows = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                   sampler, p_c, K, time_limit, arm_seed)
    return pname, name, rows


def summarize(rows, seed):
    ok = np.array([r['hard_feasible'] for r in rows])
    u = np.array([r['utility'] if r['utility'] is not None else float('nan')
                  for r in rows], float)
    fail_counts = Counter(f for r in rows for f in r['failures'])
    return {
        "n_days": int(len(rows)),
        "hard_feasible_rate": float(ok.mean()),
        "utility_all_days": _stat(u, seed),   # 不筛天
        "fail_counts": {k: int(v) for k, v in fail_counts.items()},
        "mean_timeouts": float(np.mean([r['timeouts'] for r in rows])),
        "timeout_days": int(sum(1 for r in rows if r['timeouts'] > 0)),
        "elapsed_total_s": float(np.sum([r['elapsed_s'] for r in rows])),
    }


def paired_all_days(a_rows, b_rows, seed):
    """不筛天：全部天进入配对（违规天的效用也如实计入；违规本身由 hard_feasible_rate 报告）。"""
    pa = np.array([r['utility'] if r['utility'] is not None else float('nan')
                   for r in a_rows], float)
    pb = np.array([r['utility'] if r['utility'] is not None else float('nan')
                   for r in b_rows], float)
    return _stat(pa - pb, seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--gate-instances", type=int, default=40)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--k-neighbors", type=int, default=10)
    ap.add_argument("--penalty", default=None,
                    help="只跑一档 p_c（烟测用）；默认全部三档")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 数（(p_c × 臂) 任务并行）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    contract = make_c0_contract_v2()
    train_ds = generate_dataset(args.train_instances, args.n_orders, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, train_energy_mean = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, TRAIN_SEED, args.rho, hist_tc,
        k=args.k_neighbors)
    gate_ds = add_v2_initial_quality(
        generate_dataset(args.gate_instances, args.n_orders, GATE_SEED), contract)

    samplers = {
        "uncond_hist": UncondHistoricalSampler(hist),
        "cond_hist": CondHistoricalSampler(hist, n_neighbors=args.k_neighbors),
        "explicit_feat": ExplicitFeatureSampler(hist),
    }

    penalty_items = ([(args.penalty, PENALTIES[args.penalty])]
                     if args.penalty else list(PENALTIES.items()))
    if args.penalty and args.penalty not in PENALTIES:
        raise SystemExit("unknown --penalty: " + args.penalty)

    tasks = []
    for pname, p_c in penalty_items:
        for name, sampler in samplers.items():
            tasks.append((pname, name, gate_ds, contract, B, args.capacity,
                          args.num_vehicles, cooling_share, sampler, p_c,
                          args.K, args.time_limit, ARM_SEEDS[name]))
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            task_results = pool.map(_worker, tasks)
    else:
        task_results = [_worker(t) for t in tasks]

    per_day = {}
    results = {}
    for pname, name, rows in task_results:
        results.setdefault(pname, {})["arms"] = results.setdefault(
            pname, {}).get("arms", {})
        results[pname]["arms"][name] = summarize(rows, GATE_SEED)
        per_day.setdefault(name, {})[pname] = rows

    for pname in results:
        cond_rows = per_day["cond_hist"][pname]
        uncond_rows = per_day["uncond_hist"][pname]
        st = paired_all_days(cond_rows, uncond_rows, GATE_SEED)
        arms = results[pname]["arms"]
        both_ok = (arms["cond_hist"]["hard_feasible_rate"] == 1.0
                   and arms["uncond_hist"]["hard_feasible_rate"] == 1.0)
        passed = bool(both_ok and np.isfinite(st["mean"])
                      and st["mean"] >= DELTA_STAR and st["ci_lo"] > 0)
        results[pname]["main_comparison"] = {
            "cond_minus_uncond": st,
            "delta_star": DELTA_STAR,
            "criterion": "mean>=delta_star AND ci_lo>0 AND both hard_feasible_rate==1.0",
            "passed": passed,
            "violations_present": not both_ok,
        }
        results[pname]["per_day"] = {name: per_day[name][pname] for name in samplers}

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "gate": GATE_SEED,
                  "final_test_reserved": list(FINAL_TEST_SEEDS)},
        "budget": {"B": float(B), "cooling_share": float(cooling_share),
                   "train_accept_all_c0_energy_mean": float(train_energy_mean)},
        "main_penalty": MAIN_PENALTY,
        "delta_star": DELTA_STAR,
        "gate": results,
        "note": "A-v1 步骤 2 无训练条件信息门：三臂（uncond_hist/cond_hist/explicit_feat）共用 SAA 决策闭环；"
                "主比较=cond_hist−uncond_hist；不筛天（逐日报告全部结果与违规类型）；"
                "p_c 三档独立决策运行；旧 40 天（seed 20260923/24）为开发证据，不参与本门。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
