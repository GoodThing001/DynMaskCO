"""D-CTRL-v1: is the zero-guide masked reconstructor competitive with a strong
non-learned control at equal wall budget?

Why this exists. D-EQ8-v1 and D-DOSE-v1 established that no learned guide helps the
delivery reconstructor: the zero-guide arm reached 105.2283 on the dev cohort and
every nonzero guide weight was monotonically worse. The root delivery objective,
however, is a claim of cost reduction against "equal-schedule random/nonlearned and
same-contract strong rolling controls". A learned arm is not needed to ask the
prerequisite question: does the zero-guide masked reconstruction actually beat a
strong non-learned solver given the same initial solution and the same 10 seconds?

Arms (identical initial routes, identical contract, identical 10.0s wall budget):
  H     zero-guide masked reconstruction, eight fixed masks from the registered seed
  LNS   improve_regret_lns -- destroy/regret-2 repair LNS with simulated-annealing
        acceptance (scripts/delivery/delivery_baselines.py), the same-contract strong
        rolling control
  GLS   improve_ortools GLS from the same incumbent -- REQUIRES the ortools package,
        which is not installed on this machine, so this arm is registered but
        deferred to a server stage rather than silently dropped

Registered reading rule: H dominates the control if the paired mean relative gain
>= 1% with 95% CI lower > 0; the control dominates H if the paired mean <= -1% with
95% CI upper < 0; anything else is reported as indistinguishable. The primary cohort
is dev; the fresh-seed confirmation cohort is run so the control comparison is not
resting on burned instances alone.

This is a control calibration, not a gain claim: it cannot by itself establish any
MaskCO-attributable improvement, and it is a pre-dispatch snapshot panel with
simulated physics, not an online result.

Usage:
    python delivery_strong_control.py --protocol P.json --out NEWDIR
"""
import argparse
import hashlib
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

from delivery_baselines import improve_regret_lns
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import digest, dump_new, generate, make_schedule, run_arm

PROTOCOL = {
    'version': 'D-CTRL-v1',
    'kind': 'strong_control_calibration_not_gain_claim',
    'parent_panels': ['D-EQ8-v1', 'D-DOSE-v1'],
    'incumbent_to_beat': 'zero-guide masked reconstruction (H)',
    'dataset_seed_dev': 2026101002,
    'dataset_seed_confirmation': 2026101013,
    'dev_instance_ids': list(range(32)),
    'confirmation_instance_ids': list(range(1000, 1032)),
    'mask_seed_dev': 2026101041,
    'mask_seed_confirmation': 2026101052,
    'steps': 8,
    'mask_fraction': 0.3,
    'time_limit_s': 10.0,
    'lns_seed': 41,
    'bootstrap_seed': 2026101122,
    'bootstrap_repetitions': 2000,
    'dominance_threshold': 0.01,
    'deferred_arm': {'name': 'GLS', 'reason': 'ortools not installed on the local machine',
                     'status': 'registered_not_run'},
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-CTRL-v1', 'dataset_seed_dev': 2026101002,
                'dataset_seed_confirmation': 2026101013, 'mask_seed_dev': 2026101041,
                'mask_seed_confirmation': 2026101052, 'steps': 8, 'mask_fraction': .3,
                'time_limit_s': 10., 'lns_seed': 41, 'bootstrap_seed': 2026101122,
                'bootstrap_repetitions': 2000, 'dominance_threshold': .01}
    if any(p.get(k) != v or type(p.get(k)) is bool for k, v in expected.items()):
        raise ValueError('D-CTRL-v1 registered field changed; no mismatch bypass')
    if p.get('dev_instance_ids') != list(range(32)):
        raise ValueError('Dev cohort must be the thirty-two recorded instances')
    if p.get('confirmation_instance_ids') != list(range(1000, 1032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def run_cohort(cohort, instance_ids, dataset_seed, mask_seed, protocol, contract):
    rows = []
    for instance_id in instance_ids:
        data, initial = generate(dataset_seed, instance_id)
        schedule = make_schedule(51, 8, mask_seed, instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        arms = {}
        h = run_arm(data, initial, contract, schedule, None,
                    time_limit_s=protocol['time_limit_s'], guide_weight=0.)
        arms['H'] = {'final_cost': h['final_endpoint_cost'],
                     'first_cost': h['first_endpoint_cost'],
                     'timeout': h['timeout'],
                     'actual_attempts': h['actual_attempts'],
                     'complete_candidates': h['complete_candidates'],
                     'hard_candidates': h['hard_candidates'],
                     'decode_calls': h['decode_calls'],
                     'wall_s': h['wall_s']}
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=protocol['time_limit_s'],
                                 seed=protocol['lns_seed'])
        arms['LNS'] = {'final_cost': lns['report']['cost'], 'timeout': lns['timeout'],
                       'tried': lns.get('tried', 0), 'feasible': lns.get('feasible', 0),
                       'wall_s': lns['wall_s'],
                       'status': lns.get('status', 'candidate_certified')}
        for arm in ('H', 'LNS'):
            routes = h['routes'] if arm == 'H' else lns['routes']
            replay = evaluate(data, routes, contract)
            if not replay['hard']:
                raise RuntimeError('Arm %s replay not hard feasible: %d' % (arm, instance_id))
            if abs(replay['cost'] - arms[arm]['final_cost']) > 1e-9:
                raise RuntimeError('Arm %s replay cost mismatch: %d' % (arm, instance_id))
        arms['GLS'] = {'status': 'registered_not_run_ortools_unavailable', 'final_cost': None}
        row = {'cohort': cohort, 'instance_id': int(instance_id),
               'schedule_sha256': digest(schedule), 'initial_cost': checked['cost'],
               'arms': arms}
        rows.append(row)
        print(json.dumps({'cohort': cohort, 'instance_id': instance_id,
                          'completed': len(rows),
                          'H': round(arms['H']['final_cost'], 4),
                          'LNS': round(arms['LNS']['final_cost'], 4)}), flush=True)
    return rows


def paired(control, other, seed, reps, name_control, name_other):
    """Relative gain of `other` over `control`, positive means `other` is cheaper."""
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


def adjudicate(gain, threshold):
    lo, hi = gain['descriptive_95_CI']
    if gain['mean_relative_gain'] >= threshold and lo > 0:
        return 'other_dominates'
    if gain['mean_relative_gain'] <= -threshold and hi < 0:
        return 'control_dominates'
    return 'indistinguishable'


def summarise(rows, cohort, protocol):
    table = {}
    for arm in ('H', 'LNS', 'GLS'):
        costs = [r['arms'][arm]['final_cost'] for r in rows
                 if r['arms'][arm]['final_cost'] is not None]
        table[arm] = {
            'n': len(costs),
            'mean_final_cost': statistics.fmean(costs) if costs else None,
            'n_timeout': sum(bool(r['arms'][arm].get('timeout')) for r in rows),
            'total_decode_calls': sum(r['arms'][arm].get('decode_calls', 0) for r in rows),
            'total_lns_cycles': sum(r['arms'][arm].get('tried', 0) for r in rows),
            'status': ('run' if costs else
                       rows[0]['arms'][arm].get('status', 'not_run')),
        }
    h = {r['instance_id']: r['arms']['H']['final_cost'] for r in rows}
    lns = {r['instance_id']: r['arms']['LNS']['final_cost'] for r in rows}
    gain = paired(h, lns, protocol['bootstrap_seed'], protocol['bootstrap_repetitions'],
                  'H', 'LNS')
    return {'cohort': cohort, 'table': table,
            'lns_gain_over_H': gain,
            'verdict': adjudicate(gain, protocol['dominance_threshold'])}


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    dump_new(out / 'PROTOCOL.json', protocol)
    dump_new(out / 'identity_start.json', {
        'ortools_available': False,
        'deferred_arm': protocol['deferred_arm'],
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})
    result = {'protocol': protocol, 'deferred_arm': protocol['deferred_arm']}
    for cohort, ids, dseed, mseed, key in (
            ('dev', protocol['dev_instance_ids'], protocol['dataset_seed_dev'],
             protocol['mask_seed_dev'], 'dev_progress.jsonl'),
            ('confirmation', protocol['confirmation_instance_ids'],
             protocol['dataset_seed_confirmation'], protocol['mask_seed_confirmation'],
             'confirmation_progress.jsonl')):
        with (out / key).open('x', encoding='utf-8') as sink:
            rows = run_cohort(cohort, ids, dseed, mseed, protocol, contract)
            for row in rows:
                sink.write(json.dumps(row, allow_nan=False) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
        result[cohort] = summarise(rows, cohort, protocol)
    result['online_gain_established'] = False
    result['stable_gain_established'] = False
    result['wall_s'] = time.perf_counter() - started
    dump_new(out / 'result.json', result)
    print(json.dumps({'dev_verdict': result['dev']['verdict'],
                      'confirmation_verdict': result['confirmation']['verdict'],
                      'dev_mean_H': result['dev']['table']['H']['mean_final_cost'],
                      'dev_mean_LNS': result['dev']['table']['LNS']['mean_final_cost'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
