"""A-v1 外部强求解器接单臂对比驱动（OR-Tools / PyVRP accept 决策）。

口径（与步骤 2/3 门完全一致）：
  - train seed 20260925（仅用于预算 B 的历史；B 从步骤 2 gate.json 直接读取，
    避免 jax 依赖）；开发集 = seed 20260926（40 天，已查看，标开发证据）；
    论文测试 = seed 20260927/20260928（默认不跑，--final-test 显式开启才跑）。
  - 每决策时限 = time_limit（超时降级 = 保旧计划 + 拒绝该单，计 timeouts）；
  - 不筛天；逐日保存全部结果与违规类型；
  - 三档 p_c 对同一轨迹独立评价（接单决策与 p_c 无关，只影响效用口径）；
  - 配对 = 按天配对 day-clustered bootstrap，与步骤 2 gate.json 的
    cond_hist / uncond_hist / explicit_feat 逐日行配对。

运行环境：OR-Tools 臂用 cc_ortools env（ortools 9.11.4210 + numpy），
PyVRP 臂用 cc_pyvrp env（pyvrp 0.14.0 + numpy）——同一轨迹口径、不同求解器。
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
from run_exp_reserve import (generate_dataset, _stat, BOOKING_HORIZON,
                             SPEED_KMH, KM_PER_UNIT)
from scenario_saa import make_c0_contract_v2
from solver_accept_replanner import SolverAcceptReplanner
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926
FINAL_TEST_SEEDS = (20260927, 20260928)

PENALTIES = {
    "p_c=0": {0: 0.0, 1: 0.0, 2: 0.0},
    "p_c=(5,10,15)": {0: 5.0, 1: 10.0, 2: 15.0},
    "p_c=(10,20,30)": {0: 10.0, 1: 20.0, 2: 30.0},
}
MAIN_PENALTY = "p_c=(5,10,15)"
BASELINE_ARMS = ("cond_hist", "uncond_hist", "explicit_feat")


def _single(ds, i):
    return {k: v[i] for k, v in ds.items()}


def run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
            solver, time_limit, solution_limit, pyvrp_max_iterations,
            days, penalty_names, day_off):
    rp = SolverAcceptReplanner(budget=B, capacity=capacity,
                               booking_horizon=BOOKING_HORIZON, contract=contract,
                               cooling_share=cooling_share, solver=solver,
                               time_limit=time_limit,
                               solution_limit=solution_limit,
                               pyvrp_max_iterations=pyvrp_max_iterations)
    env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    out = {pname: [] for pname in penalty_names}
    day_counters = []
    for i in range(days):
        t0 = time.time()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        elapsed = float(time.time() - t0)
        timeouts_day = int(rp.timeouts)
        # 计数器在 _reset_if_new 按天清零——每天结束即抓当天增量，再汇总
        day_counters.append(dict(
            n_solves=int(rp.n_solves), n_fail=int(rp.n_fail),
            solve_time_s=float(rp.solve_time_s),
            reject_reasons=dict(rp._reject_reasons)))
        for pname in penalty_names:
            ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract,
                                   acc, rej, B, PENALTIES[pname])
            out[pname].append(dict(
                i=int(day_off + i),
                utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
                hard_feasible=bool(ev['hard_feasible']),
                failures=sorted(set(ev['failures'])),
                n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
                energy_kwh=float(ev['energy_kwh']),
                budget_violated=bool(ev['budget_violated']),
                timeouts=timeouts_day,
                elapsed_s=elapsed))
    meta = dict(n_solves=int(sum(c['n_solves'] for c in day_counters)),
                n_fail=int(sum(c['n_fail'] for c in day_counters)),
                solve_time_s=float(sum(c['solve_time_s'] for c in day_counters)))
    rr = {}
    for c in day_counters:
        for k, v in c['reject_reasons'].items():
            rr[k] = rr.get(k, 0) + v
    meta['reject_reasons'] = {k: int(v) for k, v in rr.items()}
    meta['mean_solve_time_s'] = (meta['solve_time_s'] / meta['n_solves']
                                 if meta['n_solves'] else None)
    return out, meta


def summarize(rows, seed):
    ok = np.array([r['hard_feasible'] for r in rows])
    u = np.array([r['utility'] if r['utility'] is not None else float('nan')
                  for r in rows], float)
    fail_counts = Counter(f for r in rows for f in r['failures'])
    return {
        "n_days": int(len(rows)),
        "hard_feasible_rate": float(ok.mean()),
        "utility_all_days": _stat(u, seed),
        "fail_counts": {k: int(v) for k, v in fail_counts.items()},
        "mean_timeouts": float(np.mean([r['timeouts'] for r in rows])),
        "timeout_days": int(sum(1 for r in rows if r['timeouts'] > 0)),
        "elapsed_total_s": float(np.sum([r['elapsed_s'] for r in rows])),
    }


def paired_all_days(a_rows, b_rows, seed):
    """按天索引对齐配对（支持烟测子集：只配对两臂共有的天）。"""
    b_by_i = {r['i']: r['utility'] for r in b_rows}
    pa, pb = [], []
    for r in a_rows:
        if r['i'] not in b_by_i or r['utility'] is None or b_by_i[r['i']] is None:
            continue
        pa.append(r['utility'])
        pb.append(b_by_i[r['i']])
    if not pa:
        return {'mean': None, 'ci_lo': None, 'ci_hi': None, 'n': 0}
    return _stat(np.array(pa, float) - np.array(pb, float), seed)


def _worker(task):
    (gate_ds, contract, B, capacity, num_vehicles, cooling_share, solver,
     time_limit, solution_limit, pyvrp_max_iterations, days, day_off,
     penalty_names) = task
    sub = {k: v[day_off:day_off + days] for k, v in gate_ds.items()}
    out, meta = run_arm(sub, contract, B, capacity, num_vehicles, cooling_share,
                        solver, time_limit, solution_limit, pyvrp_max_iterations,
                        days, penalty_names, day_off)
    return day_off, out, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["ortools", "pyvrp"], required=True)
    ap.add_argument("--baseline-from", required=True,
                    help="步骤 2 gate.json（读取 B/cooling_share + 三基线臂逐日行）")
    ap.add_argument("--dev-instances", type=int, default=40)
    ap.add_argument("--final-test", action="store_true",
                    help="在论文测试日（20260927/28）上跑；默认开发集 20260926")
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--time-limit", type=float, default=10.0,
                    help="每决策时限（秒）")
    ap.add_argument("--solution-limit", type=int, default=30,
                    help="OR-Tools solution_limit（与冻结 OR7 选择一致）")
    ap.add_argument("--pyvrp-max-iterations", type=int, default=1000)
    ap.add_argument("--penalty", default=None, help="只评一档 p_c（烟测用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.penalty and args.penalty not in PENALTIES:
        raise SystemExit("unknown --penalty: " + args.penalty)
    penalty_names = [args.penalty] if args.penalty else list(PENALTIES)

    contract = make_c0_contract_v2()
    baseline = json.load(open(args.baseline_from, encoding="utf-8"))
    B = float(baseline["budget"]["B"])
    cooling_share = float(baseline["budget"]["cooling_share"])
    base_rows = {}
    for pname, blk in baseline["gate"].items():
        if "per_day" not in blk:
            continue
        for arm in BASELINE_ARMS:
            if arm in blk["per_day"]:
                base_rows[(pname, arm)] = blk["per_day"][arm]

    seeds = FINAL_TEST_SEEDS if args.final_test else (DEV_SEED,)
    results = {}
    for gate_seed in seeds:
        gate_ds = add_v2_initial_quality(
            generate_dataset(args.dev_instances, args.n_orders, gate_seed), contract)
        n_workers = max(1, args.workers)
        chunk = int(np.ceil(args.dev_instances / n_workers))
        tasks = []
        for off in range(0, args.dev_instances, chunk):
            tasks.append((gate_ds, contract, B, args.capacity, args.num_vehicles,
                          cooling_share, args.solver, args.time_limit,
                          args.solution_limit, args.pyvrp_max_iterations,
                          min(chunk, args.dev_instances - off), off, penalty_names))
        if n_workers > 1:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                task_results = pool.map(_worker, tasks)
        else:
            task_results = [_worker(t) for t in tasks]

        merged = {pname: [] for pname in penalty_names}
        meta_agg = dict(n_solves=0, n_fail=0, solve_time_s=0.0, reject_reasons={})
        for _off, out, meta in task_results:
            for pname, rows in out.items():
                merged[pname].extend(rows)
            for k in ("n_solves", "n_fail", "solve_time_s"):
                meta_agg[k] = meta_agg.get(k, 0) + meta.get(k, 0)
            for k, v in meta.get("reject_reasons", {}).items():
                meta_agg["reject_reasons"][k] = meta_agg["reject_reasons"].get(k, 0) + v
        meta_agg["mean_solve_time_s"] = (
            meta_agg["solve_time_s"] / meta_agg["n_solves"] if meta_agg["n_solves"] else None)

        for pname in penalty_names:
            rows = sorted(merged[pname], key=lambda r: r["i"])
            blk = results.setdefault(gate_seed, {}).setdefault(pname, {})
            blk.setdefault("arms", {})["%s_accept" % args.solver] = summarize(rows, gate_seed)
            blk["per_day"] = rows
            for arm in BASELINE_ARMS:
                b_rows = base_rows.get((pname, arm))
                if b_rows is None:
                    continue
                blk["arms"][arm] = summarize(b_rows, gate_seed)
                blk["%s_accept_minus_%s" % (args.solver, arm)] = paired_all_days(
                    rows, b_rows, gate_seed)
            blk["solver_meta"] = dict(meta_agg)
        results[gate_seed]["_solver"] = args.solver
        results[gate_seed]["_baseline_from"] = os.path.abspath(args.baseline_from)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED,
                  "final_test_reserved": list(FINAL_TEST_SEEDS),
                  "ran": [int(s) for s in seeds]},
        "budget": {"B": B, "cooling_share": cooling_share},
        "results": {str(s): results[s] for s in seeds},
        "note": ("A-v1 外部强求解器接单臂（myopic feasibility + 强路由）vs 步骤 2 三臂"
                 "（同信息边界/同 C0 认证/A-v1 效用口径、按天配对）。"
                 "主读点 = solver_accept − cond_hist 等配对差：路由强度对决策质量的贡献。"
                 "开发集（seed 20260926）结果 = 开发证据，非论文测试。"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
