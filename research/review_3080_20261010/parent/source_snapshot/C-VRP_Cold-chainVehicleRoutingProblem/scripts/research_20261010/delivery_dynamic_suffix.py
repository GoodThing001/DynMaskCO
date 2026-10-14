"""D-DYN-v1: does suffix reordering have room in the DYNAMIC online delivery setting?

Origin. Every delivery result so far - including the confirmed base ANY50 - lives in the
pre-dispatch snapshot setting, where the whole order set is known and one plan is
reconstructed. The registered dynamic engine (node n2, dynamic_delivery_engine.py) exists
and is engineering-qualified, but no policy has ever been run on it at research scale.
This panel answers the cheapest decisive question about the dynamic setting, following the
project's own screening rule: before porting the masked-reconstruction mechanism onto the
dynamic engine, count the action set that the mechanism would search.

Action set. The dynamic engine permits exactly one route-side intervention: after a
service finishes, a vehicle that still holds cargo may re-plan the ordering of its
remaining cargo, with the currently committed customer fixed and cross-vehicle handover
forbidden. So the question is whether reordering that suffix changes the terminal outcome
at all, given an identical initial plan.

Two policies, identical in every respect except the intervention:
  fixed    load, plan once by ascending order id, never re-plan
  reorder  same load and same first plan, then after every service completion re-plan the
           remaining cargo by nearest-neighbour from the current node

Both run on the same orders, the same coordinates, the same contract and the same dynamic
reveal/ready stream. Terminal outcome is compared on total travel distance, since both
policies serve exactly the same orders and differ only in ordering.

Verification. The engine carries its own independent event replay checker
(dynamic_delivery_engine.independent_replay), which rebuilds times, ownership, geometry,
temperature and energy from the event journal alone. Every run must pass it; the panel
aborts rather than reports if any run fails.

Scope. This tests the migration premise of the delivery mechanism, not the mechanism
itself: no mask and no anytime loop appear here, and both policies are non-learning. It
establishes whether the dynamic setting has route-side room, and nothing about MaskCO.

Usage:
    python delivery_dynamic_suffix.py --out NEWDIR [--instances N]
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('research_20261010', 'correction_engineering'):
    sys.path.insert(0, str(SCRIPTS / sub))

from dynamic_delivery_engine import (DynamicDeliveryContract,   # noqa: E402
                                     DynamicDeliveryEngine, independent_replay)

PROFILES = ('cherry_tomato_TA', 'chilled_mullet_overall_impression',
            'frozen_seabass_overall_acceptability_scenario')

PROTOCOL = {
    'version': 'D-DYN-v1',
    'kind': 'dynamic_action_set_screening_not_gain_claim',
    'parents': ['D-ANY50-CONFIRM-v1', 'D-GLS-v1'],
    'engine': 'dynamic_delivery_engine.DynamicDeliveryEngine (node n2, engineering-qualified)',
    'policies': ['fixed', 'reorder', 'reverse'],
    'intervention': ('after a service completion, re-plan the remaining cargo order by '
                     'nearest neighbour from the current node; committed customer fixed, '
                     'cross-vehicle handover forbidden by the engine'),
    'seed': 2026101221,
    'n_orders': 24,
    'n_vehicles': 3,
    'horizon_h': 8.0,
    'step_h': 0.005,
    'compared_quantity': 'total travel distance (both policies serve identical orders)',
    'independent_replay_required': True,
    'claims_online_gain': False,
    'stable_gain_established': False,
}


def make_orders(seed, n_orders):
    rng = np.random.default_rng(seed)
    coords = {0: (5.0, 5.0)}
    orders = {}
    for oid in range(1, n_orders + 1):
        coords[oid] = (float(rng.uniform(0, 10)), float(rng.uniform(0, 10)))
        zone = oid % 3
        release = 0.0 if oid % 4 else 0.20
        orders[oid] = {'quantity': 1.0, 'release': release, 'ready': release + 0.05,
                       'tw_start': 0.0, 'tw_end': 6.0, 'service': 0.05, 'zone': zone,
                       'profile_id': PROFILES[zone], 'quality_origin_h': 0.0,
                       'initial_quality': 1.0 if zone == 0 else 9.0,
                       'warehouse_product_temperature_c': (18.0, 4.0, -5.0)[zone]}
    return orders, coords


def choose_load(ready, obs, contract):
    """Largest prefix of the ready set that fits total and per-compartment capacity."""
    pick = []
    for oid in sorted(ready):
        trial = pick + [oid]
        if sum(obs['orders'][i]['quantity'] for i in trial) > contract.capacity + 1e-9:
            continue
        if any(sum(obs['orders'][i]['quantity'] for i in trial
                   if obs['orders'][i]['zone'] == z) > contract.compartment_capacity[z] + 1e-9
               for z in range(3)):
            continue
        pick = trial
    return pick


def build_suffix(cargo, obs, policy, node):
    ids = sorted(cargo)
    if policy == 'reverse':
        # Positive control: this arm MUST change the outcome. If it does not, the
        # re-planning intervention is not actually taking effect and the null result of
        # the other arms would be meaningless.
        return list(reversed(ids))
    if policy != 'reorder':
        return ids

    def point(i):
        return obs['depot'] if i == 0 else obs['orders'][i]['coords']

    def dist(a, b):
        pa, pb = point(a), point(b)
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    remaining, order, cur = set(ids), [], node
    while remaining:
        nxt = min(remaining, key=lambda i: dist(cur, i))
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return order


def run_policy(orders, coords, contract, policy, step_h):
    engine = DynamicDeliveryEngine(orders, coords, contract)
    t, guard = 0.0, 0
    while t < contract.horizon_h and guard < 100000:
        guard += 1
        engine.advance(min(t, contract.horizon_h))
        obs = engine.observation()
        for vid_key, fleet in obs['fleet'].items():
            vid = int(vid_key)
            if fleet.get('mode') != 'idle':
                continue
            node = fleet.get('node')
            cargo = fleet.get('cargo') or set()
            if node == 0 and not cargo:
                ready = [oid for oid, o in obs['orders'].items()
                         if o.get('ready') and o.get('owner') == 'warehouse']
                pick = choose_load(ready, obs, contract)
                if pick:
                    try:
                        engine.load(vid, pick)
                    except ValueError:
                        pass
            elif cargo:
                # Covers both a loaded vehicle at the depot (must depart) and a vehicle
                # holding cargo at a customer (may re-plan, then depart).
                #
                # Both policies use the SAME initial plan (nearest neighbour from the
                # depot at load time). They differ only in whether the remaining order may
                # be recomputed later: comparing against a deliberately bad initial order
                # would measure the baseline's weakness, not the room for re-planning.
                have_suffix = bool(fleet.get('suffix'))
                if not have_suffix:
                    # Every policy shares the identical initial plan, so any difference
                    # between arms can only come from re-planning after departure.
                    engine.plan(vid, build_suffix(cargo, obs, 'reorder', node) + [0])
                elif policy == 'reorder':
                    engine.plan(vid, build_suffix(cargo, obs, 'reorder', node) + [0])
                elif policy == 'reverse':
                    engine.plan(vid, build_suffix(cargo, obs, 'reverse', node) + [0])
                # 'fixed' deliberately does not re-plan.
                try:
                    engine.depart(vid)
                except ValueError:
                    pass
            else:
                # At a customer with nothing left to deliver: return to the depot.
                try:
                    engine.plan(vid, [0])
                    engine.depart(vid)
                except ValueError:
                    pass
        t += step_h
    result = engine.result()
    engine.close()
    return result


def summarise(result):
    op = result['operating']
    return {'delivered': len(op['delivered']),
            'distance_km': op.get('distance_km'),
            'energy_kwh': op.get('energy_kwh'),
            'quality_loss_value': op.get('quality_loss_value'),
            'trips': result.get('events') and sum(
                1 for e in result['events'] if e['kind'] == 'load_start')}


def main(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = replace(DynamicDeliveryContract(), vehicles=PROTOCOL['n_vehicles'],
                       horizon_h=PROTOCOL['horizon_h'])
    rows = []
    seeds = [PROTOCOL['seed'] + i for i in range(args.instances)]
    for seed in seeds:
        orders, coords = make_orders(seed, PROTOCOL['n_orders'])
        per = {}
        for policy in PROTOCOL['policies']:
            result = run_policy(orders, coords, contract, policy, PROTOCOL['step_h'])
            replay = independent_replay(orders, coords, contract, result)
            if not replay.get('independent_event_replay_pass'):
                raise RuntimeError('Independent event replay failed for policy %s seed %d; '
                                   'aborting' % (policy, seed))
            per[policy] = summarise(result)
            if seed == seeds[0]:
                with (out / ('result_seed%d_%s.json' % (seed, policy))).open(
                        'x', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, allow_nan=False)
        d_fixed = per['fixed']['distance_km']
        d_reorder = per['reorder']['distance_km']
        d_reverse = per['reverse']['distance_km']
        gain = (d_fixed - d_reorder) / d_fixed if d_fixed else 0.0
        rows.append({'seed': seed, 'served_identically':
                     per['fixed']['delivered'] == per['reorder']['delivered'],
                     'delivered': per['fixed']['delivered'],
                     'fixed_distance_km': d_fixed, 'reorder_distance_km': d_reorder,
                     'reverse_distance_km': d_reverse,
                     'relative_distance_gain': gain,
                     'positive_control_changed_outcome': abs(d_reverse - d_fixed) > 1e-9,
                     'independent_replay_pass': True})
        print(json.dumps({'seed': seed, 'fixed': round(d_fixed, 4),
                          'reorder': round(d_reorder, 4),
                          'reverse': round(d_reverse, 4),
                          'gain': round(gain, 6),
                          'control_changed': abs(d_reverse - d_fixed) > 1e-9}), flush=True)

    gains = [r['relative_distance_gain'] for r in rows]
    mean_gain = statistics.fmean(gains)
    n_with_gain = sum(1 for g in gains if g > 1e-9)
    result = {
        'protocol': PROTOCOL,
        'n_instances': len(rows),
        'rows': rows,
        'all_served_identically': all(r['served_identically'] for r in rows),
        'mean_relative_distance_gain': mean_gain,
        'max_relative_distance_gain': max(gains),
        'instances_with_any_gain': n_with_gain,
        'positive_control_changed_outcome_on_all_instances':
            all(r['positive_control_changed_outcome'] for r in rows),
        'verdict': ('dynamic_suffix_has_room' if mean_gain >= 0.01
                    else 'dynamic_suffix_has_no_material_room'),
        'interpretation': ('whether porting the masked-reconstruction mechanism onto the '
                           'dynamic engine is worth attempting, judged by whether the '
                           'route-side action set the mechanism would search can move the '
                           'outcome at all, against a control that uses the identical '
                           'initial plan'),
        'online_gain_established': False,
        'stable_gain_established': False,
        'maskco_attributable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in
                      ('n_instances', 'all_served_identically',
                       'mean_relative_distance_gain', 'max_relative_distance_gain',
                       'instances_with_any_gain', 'verdict', 'wall_s')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=8)
    main(ap.parse_args())
