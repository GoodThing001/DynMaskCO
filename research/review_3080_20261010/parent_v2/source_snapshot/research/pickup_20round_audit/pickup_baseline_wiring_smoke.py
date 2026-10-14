"""Wiring smoke for the pickup main-gate baselines: can they be constructed and run at all?

Round 8 mapped the gate's missing arms to existing components:
  cond_hist       = SaaReplanner(sampler=CondHistoricalSampler(hist, n_neighbors))
  uncond_hist     = SaaReplanner(sampler=UncondHistoricalSampler(hist))
  strong_rolling  = RollingOptimizeReplanner(...)
  uncond_dynamic  = UncondDynamicReplanner(...)   (the rule-based arm already used by my pilots)

This checks they actually instantiate and complete a full A-v1 day under the strict-online env
before a panel is registered around them. A baseline that cannot run is the gap that wastes the
most effort, so it is verified cheaply and first.

Scope: wiring only. One or two instances, a short per-decision time limit, no comparison between
arms, no interval, and no claim of any kind. A arm reporting hard=False here is a wiring fault to
be fixed, not a result to report.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main(args):
    started = time.perf_counter()
    scripts = Path(args.scripts).resolve()
    for sub in ('evaluation', 'coldchain', 'simulation'):
        sys.path.insert(0, str(scripts / sub))
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from scenario_saa import (SaaReplanner, CondHistoricalSampler, UncondHistoricalSampler,
                              build_history)
    from rolling_optimizer import RollingOptimizeReplanner
    from strict_online_env import StrictOnlineEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    p_c = {0: 5.0, 1: 10.0, 2: 15.0}
    # history_days matters: generate_dataset's first argument is the INSTANCE COUNT, so the
    # original value of 1 gave the samplers a single history day, with which
    # CondHistoricalSampler has no conditional structure to select on and necessarily collapses
    # onto UncondHistoricalSampler.
    train_ds = generate_dataset(args.history_days, 200, 20260923)
    test_ds = add_v2_initial_quality(generate_dataset(args.instances, 200, 20260924), contract)
    hist_sampler = build_history(train_ds)
    hist_tc = []
    for i in range(train_ds['coords'].shape[0]):
        m = train_ds['demands'][i, 1:] > 0
        hist_tc.append((train_ds['reveal_time'][i, 1:][m],
                        train_ds['temp_class'][i, 1:][m].astype(int)))

    def build(name, seed):
        common = dict(budget=budget, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=args.cooling_share)
        if name == 'cond_hist':
            return SaaReplanner(**common, sampler=CondHistoricalSampler(hist_sampler,
                                                                       n_neighbors=args.n_neighbors),
                                reject_penalty=p_c, K=args.K, time_limit=args.time_limit,
                                arm_seed=seed)
        if name == 'uncond_hist':
            return SaaReplanner(**common, sampler=UncondHistoricalSampler(hist_sampler),
                                reject_penalty=p_c, K=args.K, time_limit=args.time_limit,
                                arm_seed=seed)
        if name == 'strong_rolling':
            return RollingOptimizeReplanner(**common, reject_penalty=p_c,
                                            time_limit=args.time_limit)
        if name == 'uncond_dynamic':
            return UncondDynamicReplanner(**common, hist=hist_tc, k=10)
        raise ValueError(name)

    arms = ['cond_hist', 'uncond_hist', 'strong_rolling', 'uncond_dynamic']
    rows = []
    for name in arms:
        for inst in range(args.instances):
            row = {'arm': name, 'instance': inst}
            try:
                rp = build(name, 7001 + inst)
                env = StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=15,
                                      tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                                      coldchain_contract=contract,
                                      booking_horizon=BOOKING_HORIZON)
                t0 = time.perf_counter()
                traces, _ = env.run(inst)
                row['wall_s'] = time.perf_counter() - t0
                ev = evaluate_trace_a1(traces, {k: v[inst] for k, v in test_ds.items()},
                                       contract, set(rp._accepted), set(rp._rejected), budget,
                                       p_c)
                row.update(ran=True, utility=float(ev['utility']), hard=bool(ev['hard_feasible']),
                           n_served=int(ev['n_served']), n_accepted=len(rp._accepted),
                           n_rejected=len(rp._rejected))
            except Exception as exc:                       # noqa: BLE001
                row.update(ran=False, failure='%s: %s' % (type(exc).__name__, exc))
            rows.append(row)
            print(json.dumps(row), flush=True)

    ran = [r for r in rows if r.get('ran')]
    result = {
        'kind': 'pickup_main_gate_baseline_wiring_smoke',
        'arms_requested': arms, 'instances': args.instances,
        'K': args.K, 'time_limit': args.time_limit,
        'n_rows': len(rows), 'n_ran': len(ran),
        'n_hard_feasible': sum(1 for r in ran if r['hard']),
        'all_arms_constructible': all(any(r['arm'] == a and r.get('ran') for r in rows)
                                      for a in arms),
        'rows': rows,
        'what_this_is': ['wiring verification that the main gate baselines instantiate and complete '
                         'a full A-v1 day under the strict-online environment'],
        'what_this_is_not': [
            'not a comparison: no arm is ranked, no interval computed, no claim made',
            'one or two instances with a short time limit, so no performance reading is possible',
            'nothing about stable gain, attribution, impossibility or equivalence'],
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_ran', 'n_hard_feasible', 'all_arms_constructible', 'wall_s')},
                     indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=1)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--time-limit', type=float, default=2.0)
    ap.add_argument('--history-days', type=int, default=1,
                    help='number of history days for the samplers; 1 makes the conditional arm '
                         'structurally vacuous')
    ap.add_argument('--n-neighbors', type=int, default=10,
                    help='CondHistoricalSampler neighbours; must be strictly smaller than '
                         '--history-days or the conditional sampler degenerates to uniform '
                         'sampling over the whole history and cannot differ from uncond_hist')
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
