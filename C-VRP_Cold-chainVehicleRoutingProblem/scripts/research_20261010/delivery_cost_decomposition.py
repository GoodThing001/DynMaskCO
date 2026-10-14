"""D-DECOMP-v1: where does the confirmed delivery advantage actually come from?

Origin, and why this is a new question rather than a re-run. D-RECHECK-v1 recomputed the
shared quantities independently and found something that had been invisible: at the
DISTANCE level the confirmed base ANY50 beats destroy/regret-2 LNS by only about 1.1
percent, whereas at the COST level the edge reported earlier is 2 to 3 percent. Since the
contract cost is 1.0*distance + 0.1*energy + 1.0*quality_loss, a distance edge of 1.1
percent cannot by itself produce a 2-3 percent cost edge, so a material share of the
confirmed advantage must come from the energy or the quality term. Which one, and how
much, has never been measured.

Method. Runs the two arms on the same instances and the same budget, then decomposes the
paired cost difference into its three contractual components using the evaluator's own
reported components (so no new physics model is introduced and the earlier independent
arithmetic check still covers the distance term). Independently of the evaluator, a
transcribed duty-time figure - each route's depot-to-depot completion time under the
project's documented semantics - is also computed, because energy and quality both scale
with how long cargo is held and a duty-time difference is the mechanism that would explain
a component-weighted edge larger than the distance edge.

Registered reading. If the energy term contributes the majority of the paired cost
difference, the mechanism's advantage is primarily a DUTY-TIME advantage rather than a
routing advantage, which redirects what a future improvement should target. If distance
dominates despite the smaller relative edge, the earlier reading stands.

Scope. Non-learning, snapshot setting, parent physics. Nothing here tests an intervention
or establishes a gain; it decomposes one that is already confirmed.

Usage:
    python delivery_cost_decomposition.py --out NEWDIR [--instances N]
"""
import argparse
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
    'version': 'D-DECOMP-v1',
    'kind': 'cost_component_decomposition_not_gain_claim',
    'parent': 'D-RECHECK-v1',
    'question': ('which contractual component carries the confirmed advantage, given that '
                 'the distance-level edge (1.1 percent) is smaller than the cost-level edge '
                 '(2 to 3 percent)'),
    'dataset_seed': 2026101002,
    'instance_ids': list(range(32)),
    'mask_fraction': 0.5,
    'anytime_seed': 2026101131,
    'lns_seed': 41,
    'time_limit_s': 10.0,
    'bootstrap_seed': 2026101251,
    'bootstrap_repetitions': 2000,
    'components_use_evaluator_output': True,
    'duty_time_is_transcription': True,
    'claims_online_gain': False,
}


def duty_time(routes, data, contract):
    """Transcribed depot-to-depot completion time per route, summed."""
    coords = np.asarray(data['coords'], float)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    total = 0.0
    for route in routes:
        t, prev = contract.loading_service_h, 0
        for i in route:
            t = max(t + float(dist[prev, i]) / contract.speed_kmph,
                    float(data['tw_start'][i]))
            t += float(data['service_time'][i])
            prev = i
        t += float(dist[prev, 0]) / contract.speed_kmph
        total += t
    return total


def paired_mean(values_by_instance, seed, reps):
    arr = np.array(list(values_by_instance.values()), float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    means = arr[idx].mean(axis=1)
    return {'mean': float(arr.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))]}


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
        res = anytime_reconstruct(data, initial, contract, PROTOCOL['time_limit_s'],
                                  PROTOCOL['mask_fraction'], PROTOCOL['anytime_seed'])
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=PROTOCOL['time_limit_s'],
                                 seed=PROTOCOL['lns_seed'])
        ea = evaluate(data, res['routes'], contract)
        el = evaluate(data, lns['routes'], contract)
        wa = contract.distance_weight
        we = contract.energy_weight
        wq = contract.quality_weight
        d_cost = float(el['cost']) - float(ea['cost'])
        comp = {
            'cost': d_cost,
            'distance_component': wa * (float(el['distance_km']) - float(ea['distance_km'])),
            'energy_component': we * (float(el['energy_kwh']) - float(ea['energy_kwh'])),
            'quality_component': wq * (float(el['quality_loss_value'])
                                       - float(ea['quality_loss_value'])),
            'duty_time_difference_h': (duty_time(lns['routes'], data, contract)
                                       - duty_time(res['routes'], data, contract)),
        }
        comp['components_sum_to_cost'] = abs(
            comp['cost'] - (comp['distance_component'] + comp['energy_component']
                            + comp['quality_component'])) < 1e-6
        if not comp['components_sum_to_cost']:
            raise RuntimeError('Component decomposition does not reconstruct the paired '
                               'cost difference on instance %d; aborting' % instance_id)
        rows.append({'instance_id': int(instance_id), **comp})
        print(json.dumps({'instance_id': instance_id,
                          'cost_diff': round(comp['cost'], 4),
                          'dist': round(comp['distance_component'], 4),
                          'energy': round(comp['energy_component'], 4),
                          'quality': round(comp['quality_component'], 4),
                          'duty_h': round(comp['duty_time_difference_h'], 4)}), flush=True)

    fields = ['cost', 'distance_component', 'energy_component', 'quality_component',
              'duty_time_difference_h']
    summary = {f: paired_mean({r['instance_id']: r[f] for r in rows},
                              PROTOCOL['bootstrap_seed'],
                              PROTOCOL['bootstrap_repetitions']) for f in fields}
    total = summary['cost']['mean']
    shares = {f: (summary[f]['mean'] / total if abs(total) > 1e-12 else None)
              for f in ('distance_component', 'energy_component', 'quality_component')}
    dominant = max(shares, key=lambda k: shares[k]) if total > 0 else None
    result = {
        'protocol': PROTOCOL, 'n_instances': len(rows), 'rows': rows,
        'summary': summary, 'component_shares_of_cost_difference': shares,
        'dominant_component': dominant,
        'components_reconstruct_cost_on_every_instance': True,
        'verdict': ('energy_term_dominates_so_advantage_is_duty_time'
                    if dominant == 'energy_component'
                    else 'distance_term_dominates_so_advantage_is_routing'),
        'online_gain_established': False,
        'stable_gain_established': False,
        'maskco_attributable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'n_instances': len(rows), 'summary': summary,
                      'component_shares': shares, 'dominant_component': dominant,
                      'verdict': result['verdict'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=12)
    main(ap.parse_args())
