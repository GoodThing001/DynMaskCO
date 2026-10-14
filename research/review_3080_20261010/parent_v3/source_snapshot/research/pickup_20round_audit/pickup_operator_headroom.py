"""Is there anything for a guide to learn in this operator? Local-win headroom measurement.

The mechanism diagnostic (`delivery_traintask_guide_diagnostic.py`) established, for delivery, that
the guide term in `score = delta/dscale - w*gain` can only push the choice AWAY from the
immediate-delta optimum, and that across 400 decisions neither the pretrained nor the trained guide
ever picked a strictly cheaper insertion at the same state. A learned term cannot beat the base rule
unless the option that is best by the base rule's own currency is NOT the option that is best once
you look past that currency.

This module measures exactly that, for pickup, before any further training is attempted:

  * for a real decision point, enumerate the legal options for the customer the operator would
    insert;
  * for each option, record (a) the IMMEDIATE insertion delta - the currency the base rule uses -
    and (b) the TOTAL plan distance after greedily completing the whole reconstruction from that
    option onward, which is the consequence the immediate delta cannot see;
  * report how often the immediate-delta argmin differs from the plan-distance argmin.

If those two argmins coincide on essentially every decision, no scoring function of the form
`delta - w*gain` can win locally and the operator itself must change - the guide has nothing to add.
If they differ often, the measured gap is the headroom a lookahead-aware learned term could target.

This is a measurement, not a method. It trains nothing, improves nothing, and no interval here is a
registered test. No impossibility, equivalence, attribution or stable-gain claim is made: a
negative result here speaks to THIS operator form on THESE states only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    here = Path(__file__).resolve().parent
    for p in (here, scripts / 'delivery', scripts / 'evaluation', scripts / 'coldchain',
              scripts / 'simulation'):
        sys.path.insert(0, str(p))
    return scripts


def plan_distance(coords, routes):
    """Total depot-to-depot travel distance of a set of routes (pure geometry, no physics)."""
    total = 0.0
    for r in routes:
        seq = [0] + list(r) + [0]
        for a, b in zip(seq, seq[1:]):
            total += float(np.linalg.norm(np.asarray(coords[a]) - np.asarray(coords[b])))
    return total


def local_nodes(env, inst, order_ids):
    coords = [tuple(float(x) for x in env.coords[inst, 0])]
    coords += [tuple(float(x) for x in env.coords[inst, o]) for o in order_ids]
    return (np.asarray(coords, np.float64),
            {int(o): i + 1 for i, o in enumerate(order_ids)})


def complete_greedily(coords, loc, route, remaining, adj, dist, retried=0):
    """Distance-only double-ended insertion to completion, mirroring the arm's own rule."""
    route = list(route)
    remaining = sorted(remaining)
    guard = 0
    while remaining:
        guard += 1
        if guard > 1000:
            return None
        best = None
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
                return None
            opts.sort()
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
            best = (regret, i, opts[0][1]) if best is None or \
                (regret, -i) > (best[0], -best[1]) else best
        if best is None:
            return None
        _, i, at = best
        route.insert(at, i)
        remaining.remove(i)
    return route


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
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
                    got.append({'inst': inst, 'clock': float(clock), 'slots': slots,
                                'orders': [tmpl[k] for k in slots],
                                'env': env_, 'served': np.array(served_mask, copy=True)})
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
        # decide which customer the operator would insert now: max regret, distance-only
        per_customer = []
        for i in remaining:
            opts = []
            for at in range(len(route) + 1):
                a = route[at - 1] if at else 0
                b = route[at] if at < len(route) else 0
                if adj[a, b] > 0:
                    continue
                delta = dist[a, i] + dist[i, b] - (0.0 if a == b else dist[a, b])
                opts.append((float(delta), at))
            if opts:
                opts.sort()
                regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
                per_customer.append((regret, i, opts))
        if not per_customer:
            continue
        _, i, opts = max(per_customer, key=lambda r: (r[0], -r[1]))
        options = []
        for delta, at in opts:
            r2 = list(route)
            r2.insert(at, i)
            rest = [x for x in remaining if x != i]
            done = complete_greedily(coords, loc, r2, rest, adj, dist)
            if done is None:
                continue
            options.append({'position': int(at), 'immediate_delta': float(delta),
                            'completed_plan_distance': plan_distance(coords, [done])})
        if len(options) < 2:
            continue
        by_delta = min(options, key=lambda o: (o['immediate_delta'], o['position']))
        by_plan = min(options, key=lambda o: (o['completed_plan_distance'], o['position']))
        rows.append({
            'inst': inst, 'clock': st['clock'], 'orders': orders,
            'customer': int(inv[i]), 'n_options': len(options),
            'argmin_by_delta_position': by_delta['position'],
            'argmin_by_plan_position': by_plan['position'],
            'argmins_differ': bool(by_delta['position'] != by_plan['position']),
            'plan_distance_gap': float(by_delta['completed_plan_distance']
                                       - by_plan['completed_plan_distance']),
            'delta_gap_of_plan_optimum': float(by_plan['immediate_delta']
                                              - by_delta['immediate_delta']),
            'options': options})

    n = len(rows)
    differ = sum(1 for r in rows if r['argmins_differ'])
    gaps = np.array([r['plan_distance_gap'] for r in rows], dtype=np.float64)
    dgaps = np.array([r['delta_gap_of_plan_optimum'] for r in rows], dtype=np.float64)
    result = {
        'kind': 'pickup_operator_local_win_headroom',
        'question': ('at a real decision, is the option that is best by the base rule\'s own '
                     'currency (immediate insertion delta) also the option that is best once the '
                     'whole reconstruction is completed?'),
        'n_decisions': n,
        'n_argmins_differ': int(differ),
        'headroom_rate': (differ / n) if n else None,
        'mean_plan_distance_gap_when_differing': (float(gaps[differ > 0].mean())
                                                  if differ else None),
        'max_plan_distance_gap': float(gaps.max()) if n else None,
        'mean_extra_immediate_delta_paid_for_the_plan_optimum': (float(dgaps.mean()) if n else None),
        'per_decision': rows,
        'reading': ('headroom_rate is the fraction of decisions where a lookahead-aware scoring '
                    'function COULD beat the immediate-delta rule. If it is near zero, no term of '
                    'the form delta - w*gain can win locally and the operator itself must change.'),
        'what_this_is_not': [
            'not a method and not a performance comparison: nothing is trained or improved here',
            'the plan-distance proxy is pure geometry; it is not the A-v1 utility and no claim is '
            'made that minimising it maximises utility',
            'descriptive only; no interval here is a registered test and no alpha applies',
            'no impossibility, equivalence, attribution or stable-gain claim: a low headroom rate '
            'speaks to THIS operator form on THESE states only'],
        'mask_fraction': args.mask_fraction, 'mask_seed': args.mask_seed,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_decisions', 'n_argmins_differ', 'headroom_rate',
                       'mean_plan_distance_gap_when_differing', 'max_plan_distance_gap',
                       'mean_extra_immediate_delta_paid_for_the_plan_optimum', 'wall_s')},
                     indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=6)
    ap.add_argument('--states-per-instance', type=int, default=2)
    ap.add_argument('--min-clock', type=float, default=2.0)
    ap.add_argument('--min-orders', type=int, default=4)
    ap.add_argument('--max-orders', type=int, default=5)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--mask-seed', type=int, default=2026120202)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
