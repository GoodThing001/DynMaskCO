"""D-ANY50-CONFIRM-v1: the held-out merge gate for the anytime masked reconstructor.

Why a third cohort. D-ANYTIME-v1 recorded ANY50 as the cheapest arm on both of its
cohorts and beat the strong non-learned LNS control, but that specific pairing was
computed after the run, and ANY50 was only a secondary mask-size probe. Arbor's merge
gate requires a candidate to improve on an evaluator the search never optimised
against, with the candidate fixed in advance. This panel therefore uses a THIRD
disjoint data seed and instance-id block and pre-registers ANY50 as the primary arm,
with ANY30 and LNS carried alongside as a consistency reference and the control.

Registered decision rule, fixed before data:
  * the gate PASSES only if ANY50 beats LNS with mean relative gain >= 1% and paired
    95% CI lower > 0 on this fresh cohort
  * ANY30 vs ANY50 is reported to check that the mask-granularity ordering reproduces
  * anything else, including a positive mean with a CI spanning zero, fails the gate
    and the candidate stays unmerged

Scope. Non-learning: no model is loaded and every arm has zero neural decodes. This
establishes that the masked-reconstruction action set with a 50% mask and a constant
wall budget is competitive with a strong same-contract control. It is NOT a MaskCO
attributable gain, because no learned component is involved, and it is a pre-dispatch
snapshot panel with simulated physics rather than an online dispatch result.

Usage:
    python delivery_any50_confirm.py --protocol P.json --out NEWDIR
"""
import argparse
import json
import os
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
    'version': 'D-ANY50-CONFIRM-v1',
    'kind': 'held_out_merge_gate_not_gain_claim',
    'parent_panels': ['D-ANYTIME-v1'],
    'primary_arm': 'ANY50',
    'control_arm': 'LNS',
    'reference_arm': 'ANY30',
    'dataset_seed': 2026101141,
    'instance_ids': list(range(2000, 2032)),
    'anytime_seed_primary': 2026101143,
    'anytime_seed_reference': 2026101144,
    'mask_fraction_primary': 0.5,
    'mask_fraction_reference': 0.3,
    'time_limit_s': 10.0,
    'lns_seed': 41,
    'bootstrap_seed': 2026101151,
    'bootstrap_repetitions': 2000,
    'gate_threshold': 0.01,
    'gate_rule': 'PASS only if ANY50 over LNS mean relative gain>=1% and paired 95% CI lower>0',
    'preregistered_primary': True,
    'cohort_disjoint_from': ['D-EQ8-v1 dev', 'D-ANYTIME-v1 dev', 'D-ANYTIME-v1 confirmation'],
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-ANY50-CONFIRM-v1', 'primary_arm': 'ANY50',
                'control_arm': 'LNS', 'dataset_seed': 2026101141,
                'anytime_seed_primary': 2026101143, 'anytime_seed_reference': 2026101144,
                'mask_fraction_primary': .5, 'mask_fraction_reference': .3,
                'time_limit_s': 10., 'lns_seed': 41, 'bootstrap_seed': 2026101151,
                'bootstrap_repetitions': 2000, 'gate_threshold': .01,
                'preregistered_primary': True}
    for k, v in expected.items():
        got = p.get(k)
        if isinstance(v, bool):
            # A registered boolean must be present and identical; a truthy stand-in
            # (1, "true") is not accepted in its place.
            if got is not v:
                raise ValueError('D-ANY50-CONFIRM-v1 registered field changed: ' + k)
        elif got != v or isinstance(got, bool):
            raise ValueError('D-ANY50-CONFIRM-v1 registered field changed; no mismatch bypass')
    if p.get('instance_ids') != list(range(2000, 2032)):
        raise ValueError('Held-out cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def paired(control, other, seed, reps, name_control, name_other):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'n_instances': len(shared), 'control': name_control, 'other': name_other,
            'mean_relative_gain': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs))}


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    dump_new(out / 'PROTOCOL.json', protocol)
    dump_new(out / 'identity_start.json', {
        'neural_decodes_in_any_arm': 0,
        'preregistered_primary_arm': protocol['primary_arm'],
        'cohort_disjoint_from': protocol['cohort_disjoint_from'],
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})

    rows = []
    for instance_id in protocol['instance_ids']:
        data, initial = generate(protocol['dataset_seed'], instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        arms = {}
        for name, frac, seed in (('ANY50', protocol['mask_fraction_primary'],
                                  protocol['anytime_seed_primary']),
                                 ('ANY30', protocol['mask_fraction_reference'],
                                  protocol['anytime_seed_reference'])):
            res = anytime_reconstruct(data, initial, contract, protocol['time_limit_s'],
                                      frac, seed)
            arms[name] = {'final_cost': res['final_endpoint_cost'],
                          'wall_s': res['wall_s'], 'cycles': res['cycles'],
                          'accepted': res['accepted'],
                          'complete_candidates': res['complete_candidates'],
                          'decode_calls': 0, 'routes': res['routes']}
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=protocol['time_limit_s'],
                                 seed=protocol['lns_seed'])
        arms['LNS'] = {'final_cost': lns['report']['cost'], 'wall_s': lns['wall_s'],
                       'timeout': lns['timeout'], 'tried': lns.get('tried', 0),
                       'decode_calls': 0, 'routes': lns['routes']}
        for arm in ('ANY50', 'ANY30', 'LNS'):
            replay = evaluate(data, arms[arm]['routes'], contract)
            if not replay['hard']:
                raise RuntimeError('Arm %s replay not hard feasible: %d' % (arm, instance_id))
            if abs(replay['cost'] - arms[arm]['final_cost']) > 1e-9:
                raise RuntimeError('Arm %s replay cost mismatch: %d' % (arm, instance_id))
        rows.append({'instance_id': int(instance_id), 'initial_cost': checked['cost'],
                     'arms': {k: {kk: vv for kk, vv in v.items() if kk != 'routes'}
                              for k, v in arms.items()}})
        print(json.dumps({'instance_id': instance_id, 'completed': len(rows),
                          'ANY50': round(arms['ANY50']['final_cost'], 4),
                          'ANY30': round(arms['ANY30']['final_cost'], 4),
                          'LNS': round(arms['LNS']['final_cost'], 4)}), flush=True)

    with (out / 'progress.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

    by = {arm: {r['instance_id']: r['arms'][arm]['final_cost'] for r in rows}
          for arm in ('ANY50', 'ANY30', 'LNS')}
    table = {}
    for arm in ('ANY50', 'ANY30', 'LNS'):
        table[arm] = {'n': len(by[arm]),
                      'mean_final_cost': statistics.fmean(by[arm].values()),
                      'mean_wall_s': statistics.fmean([r['arms'][arm]['wall_s'] for r in rows]),
                      'total_cycles': sum(r['arms'][arm].get('cycles',
                                                             r['arms'][arm].get('tried', 0))
                                          for r in rows),
                      'total_accepted': sum(r['arms'][arm].get('accepted', 0) for r in rows),
                      'n_timeout': sum(bool(r['arms'][arm].get('timeout')) for r in rows)}
    bs, reps = protocol['bootstrap_seed'], protocol['bootstrap_repetitions']
    gate = paired(by['LNS'], by['ANY50'], bs, reps, 'LNS', 'ANY50')
    reference = paired(by['ANY30'], by['ANY50'], bs, reps, 'ANY30', 'ANY50')
    passed = (gate['mean_relative_gain'] >= protocol['gate_threshold']
              and gate['descriptive_95_CI'][0] > 0)
    result = {'protocol': protocol, 'table': table,
              'primary_ANY50_over_LNS': gate,
              'reference_ANY50_over_ANY30': reference,
              'gate_passed': bool(passed),
              'gate_rule': protocol['gate_rule'],
              'online_gain_established': False,
              'maskco_attributable_gain_established': False,
              'stable_gain_established': bool(passed),
              'wall_s': time.perf_counter() - started}
    dump_new(out / 'result.json', result)
    print(json.dumps({'means': {k: round(v['mean_final_cost'], 4) for k, v in table.items()},
                      'primary_ANY50_over_LNS': gate, 'gate_passed': bool(passed),
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
