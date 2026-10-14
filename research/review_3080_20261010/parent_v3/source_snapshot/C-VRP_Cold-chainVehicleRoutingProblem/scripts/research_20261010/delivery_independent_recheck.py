"""D-RECHECK-v1: independent recomputation of the quantities every delivery claim rests on.

Why this exists. Every delivery result in this program - the confirmed ANY50 base, its
win over destroy/regret-2 LNS, the GLS comparison, the destroy-operator screens - is
computed by one function, delivery_evaluator.evaluate, and produced by one reconstructor.
Nothing has ever recomputed any of those quantities with code that does not import the
evaluator. If evaluate had a systematic error, every conclusion would inherit it, and no
amount of internal consistency checking would notice.

What is genuinely independent here, and what is not, stated up front. This panel
recomputes, from the raw instance data alone:
  * TOTAL ROUTE DISTANCE - straight Euclidean sum over depot and customers, no project
    code involved, compared against the evaluator's distance_km
  * EXACT-ONCE SERVICE - every customer id appears exactly once across all routes
  * CAPACITY - per-vehicle demand sum against the contract capacity, from raw demands
  * COST DECOMPOSITION SELF-CONSISTENCY - the evaluator's own reported components must
    satisfy cost == w_distance*distance + w_energy*energy + w_quality*quality_loss with
    the contract's weights

It does NOT recompute refrigeration energy or cargo quality: those come from
transition_segment, the project's physical model, and reimplementing that model would
produce a second model rather than an independent check of the first. Time-window and
horizon checks are recomputed following the project's own documented semantics and are
therefore labelled as a transcription check, not an independent one.

Arms: the confirmed base ANY50 and the destroy/regret-2 LNS it beats, on the same
instances and the same budget, so the comparison itself is re-verified under independent
distance arithmetic rather than only under the evaluator's own number.

Usage:
    python delivery_independent_recheck.py --out NEWDIR [--instances N]
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_anytime import anytime_reconstruct
from delivery_baselines import improve_regret_lns
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import dump_new, generate

PROTOCOL = {
    'version': 'D-RECHECK-v1',
    'kind': 'independent_recomputation_of_shared_quantities',
    'parents': ['D-ANY50-CONFIRM-v1', 'D-DESTROY2-v1'],
    'dataset_seed': 2026101002,
    'instance_ids': list(range(32)),
    'mask_fraction': 0.5,
    'anytime_seed': 2026101131,
    'lns_seed': 41,
    'time_limit_s': 10.0,
    'independently_recomputed': ['total_route_distance', 'exact_once_service', 'capacity',
                                 'cost_decomposition_self_consistency'],
    'transcription_only': ['time_window', 'return_horizon'],
    'not_recomputed': ['refrigeration_energy', 'cargo_quality'],
    'not_recomputed_because': ('they come from transition_segment, the project physical '
                               'model; reimplementing it would create a second model, not '
                               'an independent check of the first'),
    'distance_tolerance_km': 1e-6,
    'cost_tolerance': 1e-6,
    'claims_online_gain': False,
}


def independent_distance(routes, coords):
    total = 0.0
    for route in routes:
        prev = 0
        for node in route:
            a, b = coords[prev], coords[node]
            total += math.hypot(a[0] - b[0], a[1] - b[1])
            prev = node
        a, b = coords[prev], coords[0]
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


def exact_once(routes, n_nodes):
    flat = [int(v) for route in routes for v in route]
    return sorted(flat) == list(range(1, n_nodes))


def capacity_ok(routes, demands, capacity):
    return all(sum(float(demands[i]) for i in route) <= capacity + 1e-9 for route in routes)


def transcription_time_windows(routes, data, contract):
    """Project semantics transcribed, not independently derived."""
    coords = np.asarray(data['coords'], float)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    horizon = float(np.max(data['tw_end']))
    for route in routes:
        t, prev = contract.loading_service_h, 0
        for i in route:
            t = max(t + float(dist[prev, i]) / contract.speed_kmph, float(data['tw_start'][i]))
            if t > float(data['tw_end'][i]) + 1e-9:
                return False, 'time_window'
            t += float(data['service_time'][i])
            prev = i
        t += float(dist[prev, 0]) / contract.speed_kmph
        if t > horizon + 1e-9:
            return False, 'return_horizon'
    return True, None


def main(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    dump_new(out / 'PROTOCOL.json', PROTOCOL)

    ids = PROTOCOL['instance_ids'][:args.instances]
    rows = []
    for instance_id in ids:
        data, initial = generate(PROTOCOL['dataset_seed'], instance_id)
        coords = data['coords']
        n_nodes = len(data['demands'])
        arms = {}
        res = anytime_reconstruct(data, initial, contract, PROTOCOL['time_limit_s'],
                                  PROTOCOL['mask_fraction'], PROTOCOL['anytime_seed'])
        arms['ANY50'] = res['routes']
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=PROTOCOL['time_limit_s'],
                                 seed=PROTOCOL['lns_seed'])
        arms['LNS'] = lns['routes']
        per_arm = {}
        for arm, routes in arms.items():
            checked = evaluate(data, routes, contract)
            ind_km = independent_distance(routes, coords)
            self_consistent = abs(
                float(checked['cost'])
                - (contract.distance_weight * float(checked['distance_km'])
                   + contract.energy_weight * float(checked['energy_kwh'])
                   + contract.quality_weight * float(checked['quality_loss_value']))) < 1e-6
            tw_ok, tw_reason = transcription_time_windows(routes, data, contract)
            per_arm[arm] = {
                'evaluator_cost': float(checked['cost']),
                'evaluator_distance_km': float(checked['distance_km']),
                'independent_distance_km': ind_km,
                'distance_agrees': abs(ind_km - float(checked['distance_km'])) < 1e-6,
                'exact_once': exact_once(routes, n_nodes),
                'capacity_ok': capacity_ok(routes, data['demands'], contract.capacity),
                'cost_decomposition_self_consistent': bool(self_consistent),
                'transcription_time_windows_ok': bool(tw_ok),
                'transcription_failure': tw_reason,
                'evaluator_hard': bool(checked['hard'])}
        rows.append({'instance_id': int(instance_id), 'arms': per_arm,
                     'independent_any50_cheaper': (per_arm['ANY50']['independent_distance_km']
                                                   < per_arm['LNS']['independent_distance_km']),
                     'evaluator_any50_cheaper': (per_arm['ANY50']['evaluator_cost']
                                                 < per_arm['LNS']['evaluator_cost'])})
        print(json.dumps({'instance_id': instance_id,
                          'ANY50_km': round(per_arm['ANY50']['independent_distance_km'], 4),
                          'LNS_km': round(per_arm['LNS']['independent_distance_km'], 4),
                          'dist_ok': all(per_arm[a]['distance_agrees'] for a in per_arm),
                          'self_consistent': all(
                              per_arm[a]['cost_decomposition_self_consistent'] for a in per_arm)}),
              flush=True)

    def all_arm(field):
        return all(r['arms'][a][field] for r in rows for a in r['arms'])

    n = len(rows)
    result = {
        'protocol': PROTOCOL, 'n_instances': n, 'rows': rows,
        'distance_recomputation_agrees_everywhere': all_arm('distance_agrees'),
        'max_distance_gap_km': max(abs(r['arms'][a]['independent_distance_km']
                                       - r['arms'][a]['evaluator_distance_km'])
                                   for r in rows for a in r['arms']),
        'exact_once_everywhere': all_arm('exact_once'),
        'capacity_ok_everywhere': all_arm('capacity_ok'),
        'cost_decomposition_consistent_everywhere': all_arm(
            'cost_decomposition_self_consistent'),
        'transcription_time_windows_ok_everywhere': all_arm('transcription_time_windows_ok'),
        'instances_where_independent_distance_ranks_ANY50_better':
            sum(1 for r in rows if r['independent_any50_cheaper']),
        'instances_where_evaluator_ranks_ANY50_better':
            sum(1 for r in rows if r['evaluator_any50_cheaper']),
        'ranking_agrees': all(r['independent_any50_cheaper'] == r['evaluator_any50_cheaper']
                              for r in rows),
        'independent_ANY50_mean_km': statistics.fmean(
            r['arms']['ANY50']['independent_distance_km'] for r in rows),
        'independent_LNS_mean_km': statistics.fmean(
            r['arms']['LNS']['independent_distance_km'] for r in rows),
        'relative_distance_advantage_ANY50_over_LNS': (
            (statistics.fmean(r['arms']['LNS']['independent_distance_km'] for r in rows)
             - statistics.fmean(r['arms']['ANY50']['independent_distance_km'] for r in rows))
            / statistics.fmean(r['arms']['LNS']['independent_distance_km'] for r in rows)),
        'verdict': 'shared_quantities_independently_confirmed',        'online_gain_established': False,
        'stable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    if not (result['distance_recomputation_agrees_everywhere']
            and result['cost_decomposition_consistent_everywhere']
            and result['exact_once_everywhere'] and result['capacity_ok_everywhere']):
        result['verdict'] = 'INDEPENDENT_RECOMPUTATION_DISAGREES'
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in
                      ('n_instances', 'distance_recomputation_agrees_everywhere',
                       'max_distance_gap_km', 'exact_once_everywhere',
                       'capacity_ok_everywhere',
                       'cost_decomposition_consistent_everywhere',
                       'transcription_time_windows_ok_everywhere',
                       'instances_where_independent_distance_ranks_ANY50_better',
                       'instances_where_evaluator_ranks_ANY50_better',
                       'ranking_agrees', 'independent_ANY50_mean_km',
                       'independent_LNS_mean_km', 'verdict',
                       'wall_s')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=12)
    main(ap.parse_args())
