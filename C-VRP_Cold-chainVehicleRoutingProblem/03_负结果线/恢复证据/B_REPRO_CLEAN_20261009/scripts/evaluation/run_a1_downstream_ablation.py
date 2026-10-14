"""A-v1 下游消融驱动：router（greedy/ortools/pyvrp）× vote（SAA 前瞻开/关）2×2 归因。

背景（2026-09-26 开发证据）：ortools_accept 显著胜 cond_hist（+247@p_c=0），
pyvrp_accept 显著负于 cond_hist（−141@p_c=0）——「myopic+强路由」不是良定义基线。
本消融分离两因子：
  - vote=off + router=greedy   （greedy_novote）：纯 myopic 贪心接单（无 SAA 投票）
  - vote=on  + router=greedy   （= 步骤2 cond_hist，已有）
  - vote=off + router=ortools  （= ortools_accept，已有）
  - vote=on  + router=ortools  （ortools_vote_cond，新增）
  - vote=on  + router=pyvrp    （pyvrp_vote_cond，新增）
归因：同 vote 下换 router → 路由强度贡献；同 router 下换 vote → SAA 前瞻贡献。

口径与步骤 2/3 完全一致：K=10、每决策 10s（超时=保计划+拒单）、真实 C0 certify、
A-v1 效用、不筛天、day-clustered bootstrap、三档 p_c 同轨迹独立评价。
vote 臂复用步骤 2 的 cond 采样器与 arm_seed=7002（同场景流，可逐位对照步骤2 cond_hist）。

运行环境：greedy_novote 只需 numpy；ortools/pyvrp 臂用 a1_ext（服务器）或
cc_ortools / cc_pyvrp（本地）。
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
from run_exp_energy_c0 import certify_plan
from run_exp_reserve import (generate_dataset, _stat, BOOKING_HORIZON,
                             SPEED_KMH, KM_PER_UNIT)
from scenario_saa import (SaaReplanner, CondHistoricalSampler, build_history,
                          make_c0_contract_v2)
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
BASELINE_ARMS = ("cond_hist", "uncond_hist", "explicit_feat")


class AblSaaReplanner(SaaReplanner):
    """SaaReplanner + router 选择 + vote 开关（只在接单侧换计划来源，场景评估保持贪心）。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 sampler, reject_penalty, K=10, time_limit=10.0, arm_seed=7001,
                 router='greedy', vote=True, solution_limit=30,
                 pyvrp_max_iterations=1000):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         sampler, reject_penalty, K=K, time_limit=time_limit,
                         arm_seed=arm_seed)
        if router not in ('greedy', 'ortools', 'pyvrp'):
            raise ValueError('router must be greedy|ortools|pyvrp')
        self.router = router
        self.vote = bool(vote)
        self.solution_limit = int(solution_limit)
        self.pyvrp_max_iterations = int(pyvrp_max_iterations)
        self._solver_rp = None
        if router != 'greedy':
            self._solver_rp = SolverAcceptReplanner(
                budget=budget, capacity=capacity, booking_horizon=booking_horizon,
                contract=contract, cooling_share=cooling_share, solver=router,
                time_limit=time_limit, solution_limit=solution_limit,
                pyvrp_max_iterations=pyvrp_max_iterations)

    def _try_insert_certified(self, env, inst_idx, o, vehicles, served_mask, clock):
        """接单侧计划来源：greedy 插入（父类）或强求解器完整覆盖 + 真实 C0 认证。"""
        if self.router == 'greedy':
            return super()._try_insert_certified(env, inst_idx, o, vehicles,
                                                 served_mask, clock)
        pool = sorted(set(int(x) for x in self._accepted
                          if not served_mask[x]) | {int(o)})
        saved = {k: list(v) for k, v in self._plan.items()}
        plan, reason = self._solver_rp._solve(env, inst_idx, clock, vehicles,
                                              served_mask, pool, time.time())
        if plan is None:
            self._plan = saved
            return False
        ok, _ = certify_plan(env, inst_idx, clock, vehicles, served_mask, plan,
                             self.contract, self.budget)
        if not ok:
            self._plan = saved
            return False
        self._plan = plan
        return True

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        if not self.vote:
            try:
                insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles,
                                                       served_mask, clock)
            except ValueError:
                insert_ok = False
            if insert_ok:
                self._accepted.add(o)
            else:
                self._rejected.add(o)
            return
        return super()._saa_decide(env, inst_idx, clock, vehicles, served_mask,
                                   o, scenarios, t0)

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = {int(i) for i in visible_ids if not served_mask[i]
                   and int(i) not in reserved and int(i) not in self._accepted
                   and int(i) not in self._rejected}
        if not new_set:
            return
        if not self.vote:
            for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
                self._saa_decide(env, inst_idx, clock, vehicles, served_mask, o,
                                 None, time.time())
            return
        super().on_reveal(env, inst_idx, clock, vehicles, served_mask, visible_ids)


def _single(ds, i):
    return {k: v[i] for k, v in ds.items()}


def run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
            arm_spec, p_c_names, day_off, days, hist, k_neighbors):
    sampler = None
    if arm_spec.get('sampler') == 'cond':
        sampler = CondHistoricalSampler(hist, n_neighbors=k_neighbors)
    rp = AblSaaReplanner(budget=B, capacity=capacity,
                         booking_horizon=BOOKING_HORIZON, contract=contract,
                         cooling_share=cooling_share, sampler=sampler,
                         reject_penalty=PENALTIES["p_c=(5,10,15)"],
                         K=10, time_limit=arm_spec.get('time_limit', 10.0),
                         arm_seed=arm_spec.get('arm_seed', 7002),
                         router=arm_spec['router'], vote=arm_spec['vote'],
                         solution_limit=arm_spec.get('solution_limit', 30),
                         pyvrp_max_iterations=arm_spec.get('pyvrp_max_iterations', 1000))
    env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    out = {pname: [] for pname in p_c_names}
    day_counters = []
    for j in range(days):
        i = day_off + j  # 全局实例索引：采样器 rng 依赖 inst_idx，必须与步骤2 一致
        t0 = time.time()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        elapsed = float(time.time() - t0)
        timeouts_day = int(rp.timeouts)
        day_counters.append(dict(
            n_solves=int(rp._solver_rp.n_solves) if rp._solver_rp is not None else 0,
            n_fail=int(rp._solver_rp.n_fail) if rp._solver_rp is not None else 0,
            solve_time_s=(float(rp._solver_rp.solve_time_s)
                          if rp._solver_rp is not None else 0.0),
            reject_reasons=(dict(rp._solver_rp._reject_reasons)
                            if rp._solver_rp is not None else {})))
        for pname in p_c_names:
            ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract,
                                   acc, rej, B, PENALTIES[pname])
            out[pname].append(dict(
                i=int(i),
                utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
                hard_feasible=bool(ev['hard_feasible']),
                failures=sorted(set(ev['failures'])),
                n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
                energy_kwh=float(ev['energy_kwh']),
                budget_violated=bool(ev['budget_violated']),
                timeouts=timeouts_day,
                elapsed_s=elapsed))
    meta = dict(router=arm_spec['router'], vote=bool(arm_spec['vote']),
                n_solves=int(sum(c['n_solves'] for c in day_counters)),
                n_fail=int(sum(c['n_fail'] for c in day_counters)),
                solve_time_s=float(sum(c['solve_time_s'] for c in day_counters)))
    rr = {}
    for c in day_counters:
        for k, v in c['reject_reasons'].items():
            rr[k] = rr.get(k, 0) + v
    meta['reject_reasons'] = {k: int(v) for k, v in rr.items()}
    if meta['n_solves']:
        meta['mean_solve_time_s'] = meta['solve_time_s'] / meta['n_solves']
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
    (gate_ds, contract, B, capacity, num_vehicles, cooling_share, arm_spec,
     p_c_names, days, day_off, hist, k_neighbors) = task
    # 不切片：env.run(全局索引)，保证采样器 rng 的 inst_idx 与步骤2 一致
    out, meta = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                        arm_spec, p_c_names, day_off, days, hist, k_neighbors)
    return day_off, out, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["greedy_novote", "greedy_vote_cond",
                             "ortools_vote_cond", "pyvrp_vote_cond"])
    ap.add_argument("--baseline-from", required=True,
                    help="步骤 2 gate.json（读取 B/cooling_share + 三基线臂逐日行）")
    ap.add_argument("--dev-instances", type=int, default=40)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--k-neighbors", type=int, default=10)
    ap.add_argument("--solution-limit", type=int, default=30)
    ap.add_argument("--pyvrp-max-iterations", type=int, default=1000)
    ap.add_argument("--penalty", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ARM_SPECS = {
        "greedy_novote": dict(router="greedy", vote=False, sampler=None),
        "greedy_vote_cond": dict(router="greedy", vote=True, sampler="cond"),
        "ortools_vote_cond": dict(router="ortools", vote=True, sampler="cond"),
        "pyvrp_vote_cond": dict(router="pyvrp", vote=True, sampler="cond"),
    }
    arm_spec = dict(ARM_SPECS[args.arm])
    arm_spec["time_limit"] = args.time_limit
    arm_spec["solution_limit"] = args.solution_limit
    arm_spec["pyvrp_max_iterations"] = args.pyvrp_max_iterations
    arm_spec["arm_seed"] = 7002  # 与步骤2 cond_hist 同场景流（可逐位对照）
    if args.penalty and args.penalty not in PENALTIES:
        raise SystemExit("unknown --penalty: " + args.penalty)
    p_c_names = [args.penalty] if args.penalty else list(PENALTIES)

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
    hist = build_history(generate_dataset(200, args.n_orders, TRAIN_SEED))

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
                          cooling_share, arm_spec, p_c_names,
                          min(chunk, args.dev_instances - off), off, hist,
                          args.k_neighbors))
        if n_workers > 1:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                task_results = pool.map(_worker, tasks)
        else:
            task_results = [_worker(t) for t in tasks]

        merged = {pname: [] for pname in p_c_names}
        meta_agg = {}
        for _off, out, meta in task_results:
            for pname, rows in out.items():
                merged[pname].extend(rows)
            for k, v in meta.items():
                if isinstance(v, dict):
                    meta_agg.setdefault(k, {}).update(
                        {kk: meta_agg.get(k, {}).get(kk, 0) + vv for kk, vv in v.items()})
                elif isinstance(v, bool) or not isinstance(v, (int, float)):
                    meta_agg[k] = v  # 标签（router/vote），各 worker 一致，直接覆盖
                else:
                    meta_agg[k] = meta_agg.get(k, 0) + v
        if meta_agg.get("n_solves"):
            meta_agg["mean_solve_time_s"] = meta_agg["solve_time_s"] / meta_agg["n_solves"]

        for pname in p_c_names:
            rows = sorted(merged[pname], key=lambda r: r["i"])
            blk = results.setdefault(gate_seed, {}).setdefault(pname, {})
            blk.setdefault("arms", {})[args.arm] = summarize(rows, gate_seed)
            blk["per_day"] = rows
            for arm in BASELINE_ARMS:
                b_rows = base_rows.get((pname, arm))
                if b_rows is None:
                    continue
                blk["arms"][arm] = summarize(b_rows, gate_seed)
                blk["%s_minus_%s" % (args.arm, arm)] = paired_all_days(
                    rows, b_rows, gate_seed)
            blk["arm_meta"] = dict(meta_agg)
        results[gate_seed]["_arm"] = args.arm
        results[gate_seed]["_baseline_from"] = os.path.abspath(args.baseline_from)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED,
                  "final_test_reserved": list(FINAL_TEST_SEEDS),
                  "ran": [int(s) for s in seeds]},
        "budget": {"B": B, "cooling_share": cooling_share},
        "results": {str(s): results[s] for s in seeds},
        "note": ("A-v1 下游消融（router × vote 2×2 归因）。vote 臂与步骤2 cond_hist 同场景流"
                 "（arm_seed=7002），可逐位对照。开发集结果 = 开发证据。"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
