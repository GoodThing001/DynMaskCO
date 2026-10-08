"""A-v1 SAA 下游归因/校准驱动：sampler × router × vote × sim_mode 全组合。

口径（与步骤 2/3 一致）：K=10、每决策 10s（超时=保计划+拒单）、真实 C0 certify、
A-v1 效用、不筛天、day-clustered bootstrap；vote 臂复用步骤 2 场景流（arm_seed=7002）。
每档 p_c 为独立决策轨迹（与步骤 2 的「三档 p_c 独立决策运行」一致）。

用法示例：
  # 步骤2 门口径（校准后）：cond vs uncond，主 p_c
  python run_a1_downstream_ablation.py --sampler cond  --sim-mode fixA --out .../cal_cond_fixA
  python run_a1_downstream_ablation.py --sampler uncond --sim-mode fixA --out .../cal_uncond_fixA
  # 2×2 投票行（校准后）
  python run_a1_downstream_ablation.py --sampler cond --router ortools --sim-mode fixC --out ...
  # myopic（无前瞻）
  python run_a1_downstream_ablation.py --vote off --router greedy --out ...
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

from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1, REV
from run_exp_energy_c0 import C0ReserveReplanner, certify_plan, c0_marginal_energy
from run_exp_reserve import (generate_dataset, _stat, BOOKING_HORIZON,
                             SPEED_KMH, KM_PER_UNIT)
from scenario_saa import (SaaReplanner, CondHistoricalSampler, UncondHistoricalSampler,
                          ExplicitFeatureSampler, build_history, make_c0_contract_v2)
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
    """SaaReplanner + router 选择 + vote 开关（只在接单侧换计划来源，场景评估=父类）。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 sampler, reject_penalty, K=10, time_limit=10.0, arm_seed=7001,
                 router='greedy', vote=True,
                 solution_limit=30, pyvrp_max_iterations=1000):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         sampler, reject_penalty, K=K, time_limit=time_limit,
                         arm_seed=arm_seed)
        if router not in ('greedy', 'ortools', 'pyvrp'):
            raise ValueError('router must be greedy|ortools|pyvrp')
        self.router = router
        self.vote = bool(vote)
        self._solver_rp = None
        if router != 'greedy':
            self._solver_rp = SolverAcceptReplanner(
                budget=budget, capacity=capacity, booking_horizon=booking_horizon,
                contract=contract, cooling_share=cooling_share, solver=router,
                time_limit=time_limit, solution_limit=solution_limit,
                pyvrp_max_iterations=pyvrp_max_iterations)

    def _try_insert_certified(self, env, inst_idx, o, vehicles, served_mask, clock):
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


class RuleAcceptReplanner(C0ReserveReplanner):
    """固定规则接单基线（2026-09-29，用户授权「对比方法扩充」）：
    myopic（lam=0：真实 C0 可行即收）或 value/e 固定门槛（lam>0：
    accept iff REV(o)/est(o) ≥ lam 且插入+认证可行）。无场景、无前瞻——纯规则对照。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 lam=0.0):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share)
        self.lam = float(lam)
        self.timeouts = 0
        self._reject_reasons = {}

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self.timeouts = 0
            self._reject_reasons = {}

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = {int(i) for i in visible_ids if not served_mask[i]
                   and int(i) not in reserved and int(i) not in self._accepted
                   and int(i) not in self._rejected}
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            tc = int(env.dataset['temp_class'][inst_idx, o])
            if self.lam > 0:
                e = c0_marginal_energy(tc, float(env.demands[inst_idx, o]),
                                       self.cooling_share, self.contract)
                if REV[tc] < self.lam * max(e, 1e-9):
                    self._rejected.add(o)
                    self._reject_reasons['below_lambda'] = \
                        self._reject_reasons.get('below_lambda', 0) + 1
                    continue
            try:
                insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles,
                                                       served_mask, clock)
            except ValueError:
                insert_ok = False
            if insert_ok:
                self._accepted.add(o)
            else:
                self._rejected.add(o)
                self._reject_reasons['infeasible_or_budget'] = \
                    self._reject_reasons.get('infeasible_or_budget', 0) + 1


def run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
            spec, pname, day_off, days, hist, k_neighbors):
    if spec.get('rule') in ('myopic', 'valuee'):
        rp = RuleAcceptReplanner(budget=B, capacity=capacity,
                                 booking_horizon=BOOKING_HORIZON, contract=contract,
                                 cooling_share=cooling_share,
                                 lam=spec.get('valuee_lambda', 0.0))
        env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract,
                              booking_horizon=BOOKING_HORIZON)
        rows = []
        day_rr = []
        for j in range(days):
            i = day_off + j
            t0 = time.time()
            traces, _ = env.run(i)
            acc, rej = set(rp._accepted), set(rp._rejected)
            elapsed = float(time.time() - t0)
            day_rr.append(dict(rp._reject_reasons))
            ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract,
                                   acc, rej, B, PENALTIES[pname])
            rows.append(dict(
                i=int(i),
                utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
                hard_feasible=bool(ev['hard_feasible']),
                failures=sorted(set(ev['failures'])),
                n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
                energy_kwh=float(ev['energy_kwh']),
                budget_violated=bool(ev['budget_violated']),
                timeouts=int(rp.timeouts),
                elapsed_s=elapsed))
        rr = {}
        for d in day_rr:
            for k, v in d.items():
                rr[k] = rr.get(k, 0) + v
        meta = dict(rule=spec['rule'], valuee_lambda=float(spec.get('valuee_lambda', 0.0)),
                    reject_reasons={k: int(v) for k, v in rr.items()})
        return rows, meta
    sampler = None
    if spec['sampler'] == 'cond':
        sampler = CondHistoricalSampler(hist, n_neighbors=k_neighbors)
    elif spec['sampler'] == 'uncond':
        sampler = UncondHistoricalSampler(hist)
    elif spec['sampler'] == 'explicit':
        sampler = ExplicitFeatureSampler(hist)
    elif spec['sampler'] != 'none':
        raise ValueError('unknown sampler ' + spec['sampler'])
    rp = AblSaaReplanner(budget=B, capacity=capacity,
                         booking_horizon=BOOKING_HORIZON, contract=contract,
                         cooling_share=cooling_share, sampler=sampler,
                         reject_penalty=PENALTIES[pname],
                         K=10, time_limit=spec.get('time_limit', 10.0),
                         arm_seed=7002,
                         router=spec['router'], vote=spec['vote'],
                         solution_limit=spec.get('solution_limit', 30),
                         pyvrp_max_iterations=spec.get('pyvrp_max_iterations', 1000))
    env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    rows = []
    day_counters = []
    prev = {'n_solves': 0, 'n_fail': 0, 'solve_time_s': 0.0}
    prev_rr = {}
    for j in range(days):
        i = day_off + j  # 全局实例索引：采样器 rng 依赖 inst_idx，必须与步骤2 一致
        t0 = time.time()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        elapsed = float(time.time() - t0)
        timeouts_day = int(rp.timeouts)
        cur = {'n_solves': (int(rp._solver_rp.n_solves) if rp._solver_rp is not None else 0),
               'n_fail': (int(rp._solver_rp.n_fail) if rp._solver_rp is not None else 0),
               'solve_time_s': (float(rp._solver_rp.solve_time_s)
                                if rp._solver_rp is not None else 0.0)}
        cur_rr = (dict(rp._solver_rp._reject_reasons)
                  if rp._solver_rp is not None else {})
        day_counters.append({k: cur[k] - prev[k] for k in prev})
        day_counters[-1]['reject_reasons'] = {
            k: cur_rr.get(k, 0) - prev_rr.get(k, 0) for k in set(cur_rr) | set(prev_rr)}
        prev, prev_rr = cur, cur_rr
        ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract,
                               acc, rej, B, PENALTIES[pname])
        rows.append(dict(
            i=int(i),
            utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
            hard_feasible=bool(ev['hard_feasible']),
            failures=sorted(set(ev['failures'])),
            n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
            energy_kwh=float(ev['energy_kwh']),
            budget_violated=bool(ev['budget_violated']),
            timeouts=timeouts_day,
            elapsed_s=elapsed))
    meta = dict(router=spec['router'], vote=bool(spec['vote']),
                sampler=spec['sampler'],
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
    return rows, meta


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
    (gate_ds, contract, B, capacity, num_vehicles, cooling_share, spec,
     pname, days, day_off, hist, k_neighbors) = task
    rows, meta = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                         spec, pname, day_off, days, hist, k_neighbors)
    return day_off, rows, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", required=True,
                    choices=["cond", "uncond", "explicit", "none"])
    ap.add_argument("--router", default="greedy", choices=["greedy", "ortools", "pyvrp"])
    ap.add_argument("--vote", default="on", choices=["on", "off"])
    ap.add_argument("--rule", default=None, choices=["myopic", "valuee"],
                    help="固定规则接单臂（忽略 sampler/vote；valuee 需 --valuee-lambda）")
    ap.add_argument("--valuee-lambda", type=float, default=0.0,
                    help="value/e 固定门槛 λ（rule=valuee 时生效）")
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
    ap.add_argument("--penalty", default="p_c=(5,10,15)",
                    help="单档 p_c（默认主档）；all = 三档独立轨迹")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.penalty != "all" and args.penalty not in PENALTIES:
        raise SystemExit("unknown --penalty: " + args.penalty)
    p_names = list(PENALTIES) if args.penalty == "all" else [args.penalty]
    spec = dict(sampler=args.sampler, router=args.router,
                vote=(args.vote == "on"), time_limit=args.time_limit,
                solution_limit=args.solution_limit,
                pyvrp_max_iterations=args.pyvrp_max_iterations,
                rule=args.rule, valuee_lambda=args.valuee_lambda)

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
        for pname in p_names:
            n_workers = max(1, args.workers)
            chunk = int(np.ceil(args.dev_instances / n_workers))
            tasks = []
            for off in range(0, args.dev_instances, chunk):
                tasks.append((gate_ds, contract, B, args.capacity, args.num_vehicles,
                              cooling_share, spec, pname,
                              min(chunk, args.dev_instances - off), off, hist,
                              args.k_neighbors))
            if n_workers > 1:
                from multiprocessing import Pool
                with Pool(n_workers) as pool:
                    task_results = pool.map(_worker, tasks)
            else:
                task_results = [_worker(t) for t in tasks]
            merged = []
            meta_agg = {}
            for _off, rows, meta in task_results:
                merged.extend(rows)
                for k, v in meta.items():
                    if isinstance(v, dict):
                        meta_agg.setdefault(k, {}).update(
                            {kk: meta_agg.get(k, {}).get(kk, 0) + vv
                             for kk, vv in v.items()})
                    elif k in ('rule', 'valuee_lambda', 'sampler', 'router', 'vote') \
                            or isinstance(v, bool) or not isinstance(v, (int, float)):
                        meta_agg[k] = v   # 标签/配置键：各 worker 一致，直接覆盖，不求和
                    else:
                        meta_agg[k] = meta_agg.get(k, 0) + v
            if meta_agg.get('n_solves'):
                meta_agg['mean_solve_time_s'] = meta_agg['solve_time_s'] / meta_agg['n_solves']
            rows = sorted(merged, key=lambda r: r['i'])
            blk = results.setdefault(gate_seed, {}).setdefault(pname, {})
            if spec.get('rule'):
                arm_name = 'rule_%s' % spec['rule']
                if spec['rule'] == 'valuee':
                    arm_name += '_lam%.2f' % spec['valuee_lambda']
            else:
                arm_name = '%s_%s_vote%s' % (spec['router'], spec['sampler'],
                                             'on' if spec['vote'] else 'off')
            blk.setdefault("arms", {})[arm_name] = summarize(rows, gate_seed)
            blk["per_day"] = rows
            for arm in BASELINE_ARMS:
                b_rows = base_rows.get((pname, arm))
                if b_rows is None:
                    continue
                blk["arms"][arm] = summarize(b_rows, gate_seed)
                blk["%s_minus_%s" % (arm_name, arm)] = paired_all_days(
                    rows, b_rows, gate_seed)
            blk["arm_meta"] = dict(meta_agg)
        results[gate_seed]["_spec"] = spec
        results[gate_seed]["_baseline_from"] = os.path.abspath(args.baseline_from)

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED,
                  "final_test_reserved": list(FINAL_TEST_SEEDS),
                  "ran": [int(s) for s in seeds]},
        "budget": {"B": B, "cooling_share": cooling_share},
        "results": {str(s): results[s] for s in seeds},
        "note": ("A-v1 SAA 下游归因（sampler×router×vote）。"
                 "vote 臂与步骤2 同场景流（arm_seed=7002）。"
                 "下游 = 修复后的 scenario_saa.py（T2-T5 + 影子车队修复，2026-09-26）。"
                 "开发集=开发证据。"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
