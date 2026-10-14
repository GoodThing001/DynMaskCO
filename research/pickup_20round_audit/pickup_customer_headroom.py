"""Where does this operator actually have headroom: the position choice or the customer choice?

`pickup_operator_headroom.py` showed the POSITION choice has almost none: the immediate-delta argmin
and the completed-plan argmin coincide in about 90 percent of decisions, and where they differ the
plan-distance advantage is ~0.0036 units. So no term of the form `delta - w*gain` can win by
choosing positions better.

The operator makes a second decision that the same module never tested: WHICH customer to insert
next, chosen by the regret heuristic. This module measures headroom there with the same method:

  currency(customer)     = the regret value the rule actually uses,
  consequence(customer)  = total plan distance after inserting that customer at its best position
                           and greedily completing the rest.

If `headroom_rate_customer` is materially larger than `headroom_rate_position`, then the operator
change that could let a learned term win is to score CUSTOMERS, not positions - and that conclusion
is then based on a measurement rather than on a guess.

Measurement only: nothing is trained, no arm is compared, and no interval is a registered test.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def main(args):
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    scripts = Path(args.scripts).resolve()
    for p in (scripts / 'delivery', scripts / 'evaluation', scripts / 'coldchain',
              scripts / 'simulation'):
        sys.path.insert(0, str(p))
    from pickup_operator_headroom import (bootstrap, local_nodes, plan_distance,
                                          complete_greedily)
    bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from delivery_maskco import partial_adjacency
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = add_v2_initial_quality(generate_dataset(args.instances, 200, 20260924), contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        m = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][m],
                     train_ds['temp_class'][i][1:][m].astype(int)))

    def make_env():
        rp = UncondDynamicReplanner(budget=budget, capacity=50.0,
                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                    cooling_share=args.cooling_share, hist=hist, k=10)
        return StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=15,
                               tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                               coldchain_contract=contract,
                               booking_horizon=BOOKING_HORIZON), rp

    states = []
    for inst in range(args.instances):
        env, rp = make_env()
        got = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock) or len(got) >= args.states_per_instance:
                return
            for v in vehicles:
                tmpl = [int(x) for x in (v.mutable_suffix or [])]
                slots = [k for k, x in enumerate(tmpl) if x != 0 and not served_mask[x]
                         and float(env_.reveal_time[inst_idx, x]) <= clock + 1e-6]
                if args.min_orders <= len(slots) <= args.max_orders:
                    got.append({'inst': inst, 'clock': float(clock), 'orders': [tmpl[k] for k in slots],
                                'env': env_})
                    return

        env.snapshot_hook = hook
        env.run(inst)
        states.extend(got)

    rows = []
    for st in states:
        env, inst = st['env'], st['inst']
        orders = st['orders']
        coords, loc = local_nodes(env, inst, orders)
        inv = {v: k for k, v in loc.items()}
        nodes = len(orders) + 1
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        mask = set(map(int, np.random.default_rng(args.mask_seed).choice(
            np.arange(1, nodes), size=max(1, int(math.ceil(args.mask_fraction * len(orders)))),
            replace=False)))
        adj = partial_adjacency([[loc[o] for o in orders]], mask, nodes)
        route = [loc[o] for o in orders if loc[o] not in mask]
        remaining = sorted(mask)
        cands = []
        for i in remaining:
            opts = []
            for at in range(len(route) + 1):
                a = route[at - 1] if at else 0
                b = route[at] if at < len(route) else 0
                if adj[a, b] > 0:
                    continue
                delta = dist[a, i] + dist[i, b] - (0.0 if a == b else dist[a, b])
                opts.append((float(delta), at))
            if not opts:
                continue
            opts.sort()
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
            r2 = list(route)
            r2.insert(opts[0][1], i)
            rest = [x for x in remaining if x != i]
            done = complete_greedily(coords, loc, r2, rest, adj, dist)
            if done is None:
                continue
            cands.append({'customer': int(inv[i]), 'regret': float(regret),
                          'best_immediate_delta': float(opts[0][0]),
                          'completed_plan_distance': plan_distance(coords, [done])})
        if len(cands) < 2:
            continue
        by_regret = max(cands, key=lambda c: (c['regret'], -c['customer']))
        by_plan = min(cands, key=lambda c: (c['completed_plan_distance'], c['customer']))
        by_delta = min(cands, key=lambda c: (c['best_immediate_delta'], c['customer']))
        rows.append({
            'inst': inst, 'clock': st['clock'], 'orders': orders, 'n_candidates': len(cands),
            'argmax_by_regret_customer': by_regret['customer'],
            'argmin_by_plan_customer': by_plan['customer'],
            'argmin_by_delta_customer': by_delta['customer'],
            'regret_vs_plan_differ': bool(by_regret['customer'] != by_plan['customer']),
            'delta_vs_plan_differ': bool(by_delta['customer'] != by_plan['customer']),
            'plan_distance_gap_if_regret_used': float(by_regret['completed_plan_distance']
                                                      - by_plan['completed_plan_distance']),
            'plan_distance_gap_if_delta_used': float(by_delta['completed_plan_distance']
                                                     - by_plan['completed_plan_distance']),
            'candidates': cands})

    n = len(rows)
    rg = np.array([r['plan_distance_gap_if_regret_used'] for r in rows], dtype=np.float64)
    dg = np.array([r['plan_distance_gap_if_delta_used'] for r in rows], dtype=np.float64)
    result = {
        'kind': 'pickup_operator_customer_choice_headroom',
        'n_decisions': n,
        'regret_rule_headroom_rate': (sum(1 for r in rows if r['regret_vs_plan_differ']) / n)
                                     if n else None,
        'immediate_delta_rule_headroom_rate': (sum(1 for r in rows if r['delta_vs_plan_differ']) / n)
                                              if n else None,
        'mean_plan_gap_if_regret_used': float(rg.mean()) if n else None,
        'max_plan_gap_if_regret_used': float(rg.max()) if n else None,
        'mean_plan_gap_if_immediate_delta_used': float(dg.mean()) if n else None,
        'max_plan_gap_if_immediate_delta_used': float(dg.max()) if n else None,
        'per_decision': rows,
        'reading': ('Compare these two headroom rates with the position-choice headroom rate '
                    '(about 0.098 with gaps of ~0.0036). A larger rate or a larger gap here means '
                    'the CUSTOMER decision is where a learned term could plausibly win, and the '
                    'operator should be changed to score customers rather than positions.'),
        'what_this_is_not': [
            'not a method: nothing is trained and no arm is compared',
            'the plan-distance proxy is pure geometry, not A-v1 utility',
            'descriptive only; no interval is a registered test',
            'no impossibility, equivalence, attribution or stable-gain claim'],
        'mask_fraction': args.mask_fraction, 'mask_seed': args.mask_seed,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_decisions', 'regret_rule_headroom_rate',
                       'immediate_delta_rule_headroom_rate', 'mean_plan_gap_if_regret_used',
                       'max_plan_gap_if_regret_used', 'mean_plan_gap_if_immediate_delta_used',
                       'max_plan_gap_if_immediate_delta_used', 'wall_s')}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=14)
    ap.add_argument('--states-per-instance', type=int, default=3)
    ap.add_argument('--min-clock', type=float, default=2.0)
    ap.add_argument('--min-orders', type=int, default=4)
    ap.add_argument('--max-orders', type=int, default=6)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--mask-seed', type=int, default=2026120202)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
