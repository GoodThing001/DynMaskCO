"""P-CENSUS-v1: exact size and distance headroom of the same-vehicle tail action set.

Question. Before investing in a full encoder+decoder suffix reconstructor for pickup
(the registered node n3), measure whether the action set it would search has any room at
all. The action set is fixed by policy: accepted orders and per-vehicle ownership are
immutable, cross-vehicle reassignment is forbidden, and only the order of customers
already assigned to the same vehicle may change.

Method. Run the frozen JF1-H-F baseline once per instance, recover each vehicle's served
sequence from its trace, enumerate EVERY permutation of that sequence, keep the ones that
are time-window feasible under the same depot-departure clock the environment used, and
compare travel distance against the baseline order.

Scope, stated before the numbers. This panel reports STRUCTURE and distance headroom.
It is NOT a verdict on A-v1 utility headroom, because the composite cost is dominated by
refrigeration energy (measured on instance 0: distance_km 23.42 contributes 23.42 of a
coldchain_cost of 247.07, while the energy term contributes about 221), and a reordering
changes the duty time and therefore the energy. Deciding utility headroom needs a
time-line and cold-chain re-simulation, which is registered as the next step rather than
approximated here.

Self-validation. The distance recomputed from the baseline order MUST equal the frozen
evaluator's distance_km, and the baseline order MUST be time-window feasible under the
recomputed clock. If either fails the panel aborts rather than reporting.

Usage:
    python pickup_tail_census.py --out NEWDIR [--max-instances N]
"""
import argparse
from itertools import permutations
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import EXTENSION_ROOT  # noqa: E402

for sub in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, str(Path(EXTENSION_ROOT) / 'scripts' / sub))

from strict_online_env import StrictOnlineEnv            # noqa: E402
from jf1h_repair import make_continuation                # noqa: E402
from coldchain_contract import default_pilot_contract    # noqa: E402
from coldchain_evaluator import evaluate_coldchain_trace  # noqa: E402

DATA = Path(EXTENSION_ROOT) / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_dev_check_teacher.npz'
CAPACITY = 50.0
VEHICLES = 25
TW_SPEED = 1.0
MAX_ENUM = 7          # guard only; the observed maximum is 3

PROTOCOL = {
    'version': 'P-CENSUS-v1',
    'kind': 'action_set_structure_and_distance_headroom',
    'not_a_verdict_on': 'A-v1 utility headroom (energy and quality not re-simulated)',
    'action_set': 'permutations of the customers already served by the same vehicle',
    'frozen_within_action_set': ['accepted set', 'per-vehicle ownership',
                                 'cross-vehicle reassignment'],
    'data': 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz',
    'capacity': CAPACITY,
    'num_vehicles': VEHICLES,
    'tw_speed': TW_SPEED,
    'baseline': 'JF1-H-F (make_continuation)',
    'bootstrap_seed': 2026101191,
    'bootstrap_repetitions': 2000,
    'claims_online_gain': False,
    'regenerate_baseline_each_run': True,
}


def route_feasible(order, dispatch, dist_row, tw_start, tw_end, service_time, horizon):
    """Depot departure -> customers in `order` -> depot, with waiting at each customer."""
    t, prev = float(dispatch), 0
    for i in order:
        t = max(t + float(dist_row[prev, i]) / TW_SPEED, float(tw_start[i]))
        if t > float(tw_end[i]) + 1e-9:
            return False, None
        t += float(service_time[i])
        prev = i
    t += float(dist_row[prev, 0]) / TW_SPEED
    if t > horizon + 1e-9:
        return False, None
    return True, t


def route_distance(order, dist_row):
    prev, total = 0, 0.0
    for i in order:
        total += float(dist_row[prev, i])
        prev = i
    total += float(dist_row[prev, 0])
    return total


def main(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dataset = dict(np.load(DATA))
    contract = default_pilot_contract()
    env = StrictOnlineEnv(dataset, CAPACITY, TW_SPEED, VEHICLES,
                          replanner=make_continuation(), coldchain_contract=contract)
    n_inst = env.num_instances if args.max_instances is None else min(
        env.num_instances, args.max_instances)

    per_instance, vehicle_sizes, all_checks = [], [], []
    for idx in range(n_inst):
        traces, served = env.run(idx)
        inst = {'coords': env.coords[idx], 'tw_start': env.tw_start[idx],
                'tw_end': env.tw_end[idx], 'service_time': env.service_time[idx],
                'demands': env.demands[idx], 'dist_mat': env.dist_mat[idx],
                'temp_class': env.temp_class[idx],
                'initial_quality': env.initial_quality[idx], 'speed': env.tw_speed}
        report = evaluate_coldchain_trace(traces, inst, contract)
        horizon = float(np.max(env.tw_end[idx]))
        dist_row = env.dist_mat[idx]

        base_total, best_total, improved_vehicles, movable_vehicles = 0.0, 0.0, 0, 0
        sizes = {}
        for trace in traces:
            order = [int(s.node) for s in trace.services]
            if not order:
                continue
            sizes[len(order)] = sizes.get(len(order), 0) + 1
            vehicle_sizes.append(len(order))
            dispatch = float(trace.dispatch_time or 0.0)
            base_ok, _ = route_feasible(order, dispatch, dist_row, env.tw_start[idx],
                                        env.tw_end[idx], env.service_time[idx], horizon)
            base_d = route_distance(order, dist_row)
            base_total += base_d
            if not base_ok:
                all_checks.append({'instance': idx, 'vehicle': trace.vehicle_id,
                                   'baseline_feasible': False})
                raise RuntimeError('Baseline order is not feasible under the recomputed '
                                   'clock; aborting instead of reporting')
            if len(order) > MAX_ENUM:
                best_total += base_d
                continue
            best_d, best_order = base_d, order
            for perm in permutations(order):
                if perm == tuple(order):
                    continue
                ok, _ = route_feasible(perm, dispatch, dist_row, env.tw_start[idx],
                                       env.tw_end[idx], env.service_time[idx], horizon)
                if not ok:
                    continue
                d = route_distance(perm, dist_row)
                if d < best_d - 1e-12:
                    best_d, best_order = d, perm
            movable_vehicles += 1
            if best_d < base_d - 1e-12:
                improved_vehicles += 1
            best_total += best_d

        # Self-validation: recomputed baseline distance must equal the frozen evaluator.
        delta = abs(base_total - float(report['distance_km']))
        all_checks.append({'instance': idx, 'recomputed_baseline_km': base_total,
                           'evaluator_km': float(report['distance_km']),
                           'abs_delta': delta, 'baseline_feasible': True})
        if delta > 1e-6:
            raise RuntimeError('Recomputed baseline distance %.9f != evaluator %.9f on '
                               'instance %d; aborting' % (base_total,
                                                          float(report['distance_km']), idx))
        per_instance.append({
            'instance': idx, 'evaluator_distance_km': float(report['distance_km']),
            'evaluator_coldchain_cost': float(report['coldchain_cost']),
            'evaluator_energy_kwh': float(report['energy_kwh']),
            'evaluator_quality_loss': float(report['quality_loss']),
            'recomputed_baseline_km': base_total,
            'best_reordered_km': best_total,
            'gain_km': base_total - best_total,
            'relative_gain': (base_total - best_total) / base_total if base_total else 0.0,
            'vehicles_with_services': sum(sizes.values()),
            'vehicles_improved': improved_vehicles,
            'vehicle_size_histogram': sizes})
        print(json.dumps({'instance': idx, 'vehicles': sum(sizes.values()),
                          'base_km': round(base_total, 4), 'best_km': round(best_total, 4),
                          'improved_vehicles': improved_vehicles,
                          'sizes': sizes}), flush=True)

    gains = np.array([r['gain_km'] for r in per_instance], float)
    rel = np.array([r['relative_gain'] for r in per_instance], float)
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    idxs = rng.integers(0, len(rel), size=(PROTOCOL['bootstrap_repetitions'], len(rel)))
    means = rel[idxs].mean(axis=1)
    size_hist = {}
    for s in vehicle_sizes:
        size_hist[s] = size_hist.get(s, 0) + 1
    result = {
        'protocol': PROTOCOL,
        'n_instances': len(per_instance),
        'self_validation': {
            'recomputed_matches_evaluator_on_all_instances': True,
            'max_abs_delta_km': max(c['abs_delta'] for c in all_checks
                                    if 'abs_delta' in c),
            'baseline_feasible_on_all_vehicles': True},
        'vehicle_size_histogram': size_hist,
        'vehicles_with_single_customer': size_hist.get(1, 0),
        'vehicles_with_two_customers': size_hist.get(2, 0),
        'total_vehicles_with_services': len(vehicle_sizes),
        'mean_customers_per_vehicle': statistics.fmean(vehicle_sizes),
        'vehicles_with_distance_improvement': sum(r['vehicles_improved']
                                                  for r in per_instance),
        'distance_headroom': {
            'mean_gain_km': float(gains.mean()),
            'mean_relative_gain': float(rel.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_with_any_gain': int(sum(g > 1e-12 for g in gains)),
            'instances_with_no_gain': int(sum(g <= 1e-12 for g in gains))},
        'structural_note': (
            'A two-customer route has exactly two orderings, and under a symmetric '
            'distance matrix d(0,a)+d(a,b)+d(b,0) == d(0,b)+d(b,a)+d(a,0), so a '
            'two-customer vehicle has NO reordering headroom in distance at all. The '
            'entire same-vehicle tail action set is therefore bounded by the size '
            'histogram above.'),
        'utility_verdict_withheld_because': PROTOCOL['not_a_verdict_on'],
        'online_gain_established': False,
        'stable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'per_instance.json').open('x', encoding='utf-8') as f:
        json.dump(per_instance, f, indent=2, allow_nan=False)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in
                      ('vehicle_size_histogram', 'mean_customers_per_vehicle',
                       'vehicles_with_distance_improvement', 'distance_headroom',
                       'self_validation', 'wall_s')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-instances', type=int, default=None)
    main(ap.parse_args())
