"""D-GLS-v1: fill the one missing strong control in the delivery line.

Origin. D-CTRL-v1 compared the zero-guide reconstructor against a non-learned
destroy/regret-2 LNS and registered an OR-Tools GLS arm that could not run because the
ortools package was absent on the local machine. D-ANYTIME-v1 then showed the same
reconstructor run on a deadline loop is indistinguishable from LNS at a 30 percent mask
and beats it at a 50 percent mask, and D-ANY50-CONFIRM-v1 confirmed that on a third
disjoint cohort. GLS therefore remains the only unmeasured strong control, and it is
strictly stronger than the LNS used so far, so it bounds how solid the confirmed base is.

This panel runs all three arms ON ONE MACHINE at one identical 10.0s wall budget and one
identical initial solution, so the comparison is not contaminated by machine speed:
  ANY50   the confirmed base - deadline-driven masked reconstruction, 50 percent mask,
          no neural decodes
  LNS     destroy/regret-2 repair with simulated-annealing acceptance
  GLS     OR-Tools guided local search seeded from the same incumbent, reported
          objective = travel distance, selection objective = the delivery contract cost

Registered reading: whichever arm is cheapest on the paired per-instance comparison
decides whether the confirmed base survives a stronger control. If GLS beats ANY50 with
mean relative gain >= 1 percent and paired 95 percent CI lower > 0, the base is dominated
and the mask/acceptance mechanism needs rethinking rather than tuning. If ANY50 is not
worse, the base holds against the strongest control available to this project.

Scope: non-learning throughout, pre-dispatch snapshot with simulated parent physics, no
online dispatch claim.

Usage:
    python delivery_gls_control.py --out NEWDIR
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_anytime import anytime_reconstruct
from delivery_baselines import improve_ortools, improve_regret_lns
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import dump_new, generate

PROTOCOL = {
    'version': 'D-GLS-v1',
    'kind': 'missing_strong_control_fill_not_gain_claim',
    'parents': ['D-CTRL-v1', 'D-ANYTIME-v1', 'D-ANY50-CONFIRM-v1'],
    'dataset_seed': 2026101002,
    'instance_ids': list(range(32)),
    'mask_fraction': 0.5,
    'anytime_seed': 2026101131,
    'lns_seed': 41,
    'time_limit_s': 10.0,
    'bootstrap_seed': 2026101211,
    'bootstrap_repetitions': 2000,
    'dominance_threshold': 0.01,
    'all_arms_same_machine_same_budget': True,
    'ortools_seed': 'library default (improve_ortools sets no random seed)',
    'claims_online_gain': False,
    'all_failures_retained': True,
}


def paired(control, other, seed, reps, name_control, name_other):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    mean = float(diffs.mean())
    verdict = ('other_dominates' if mean >= 0.01 and lo > 0 else
               'control_dominates' if mean <= -0.01 and hi < 0 else 'indistinguishable')
    return {'control': name_control, 'other': name_other, 'n_instances': len(shared),
            'mean_relative_gain': mean, 'descriptive_95_CI': [lo, hi],
            'instances_other_better': int(sum(d > 0 for d in diffs)),
            'instances_other_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs)), 'verdict': verdict}


def main(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    dump_new(out / 'PROTOCOL.json', PROTOCOL)

    try:
        import ortools
        ortools_version = getattr(ortools, '__version__', 'unknown')
    except ImportError:
        ortools_version = None
    if ortools_version is None:
        raise SystemExit('ortools is required for this panel; aborting instead of '
                         'silently dropping the GLS arm')
    dump_new(out / 'identity_start.json', {'ortools_version': ortools_version,
                                           'protocol': PROTOCOL})

    rows = []
    for instance_id in PROTOCOL['instance_ids']:
        data, initial = generate(PROTOCOL['dataset_seed'], instance_id)
        checked = evaluate(data, initial, contract)
        if not checked['hard']:
            raise RuntimeError('Initial data hard failure retained: %d' % instance_id)
        arms = {}
        res = anytime_reconstruct(data, initial, contract, PROTOCOL['time_limit_s'],
                                  PROTOCOL['mask_fraction'], PROTOCOL['anytime_seed'])
        arms['ANY50'] = {'final_cost': res['final_endpoint_cost'], 'wall_s': res['wall_s'],
                         'cycles': res['cycles'], 'decode_calls': 0, 'routes': res['routes']}
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=PROTOCOL['time_limit_s'],
                                 seed=PROTOCOL['lns_seed'])
        arms['LNS'] = {'final_cost': lns['report']['cost'], 'wall_s': lns['wall_s'],
                       'cycles': lns.get('tried', 0), 'decode_calls': 0,
                       'status': lns.get('status', 'candidate_certified'),
                       'routes': lns['routes']}
        gls = improve_ortools(data, initial, contract,
                              time_limit_s=PROTOCOL['time_limit_s'])
        arms['GLS'] = {'final_cost': gls['report']['cost'], 'wall_s': gls['wall_s'],
                       'cycles': 0, 'decode_calls': 0,
                       'status': gls.get('status', 'no_candidate'),
                       'timeout': gls.get('timeout', None), 'routes': gls['routes']}
        for arm in ('ANY50', 'LNS', 'GLS'):
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
                          'LNS': round(arms['LNS']['final_cost'], 4),
                          'GLS': round(arms['GLS']['final_cost'], 4)}), flush=True)

    with (out / 'progress.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
    by = {a: {r['instance_id']: r['arms'][a]['final_cost'] for r in rows}
          for a in ('ANY50', 'LNS', 'GLS')}
    table = {a: {'n': len(by[a]), 'mean_final_cost': statistics.fmean(by[a].values()),
                 'mean_wall_s': statistics.fmean(r['arms'][a]['wall_s'] for r in rows),
                 'total_cycles': sum(r['arms'][a]['cycles'] for r in rows)}
             for a in ('ANY50', 'LNS', 'GLS')}
    bs, reps = PROTOCOL['bootstrap_seed'], PROTOCOL['bootstrap_repetitions']
    gains = {'GLS_over_ANY50': paired(by['ANY50'], by['GLS'], bs, reps, 'ANY50', 'GLS'),
             'GLS_over_LNS': paired(by['LNS'], by['GLS'], bs, reps, 'LNS', 'GLS'),
             'ANY50_over_LNS': paired(by['LNS'], by['ANY50'], bs, reps, 'LNS', 'ANY50')}
    result = {'protocol': PROTOCOL, 'ortools_version': ortools_version,
              'table': table, 'gains': gains,
              'confirmed_base_dominated_by_GLS': gains['GLS_over_ANY50']['verdict'] == 'other_dominates',
              'online_gain_established': False,
              'stable_gain_established': False,
              'wall_s': time.perf_counter() - started}
    dump_new(out / 'result.json', result)
    print(json.dumps({'means': {a: round(t['mean_final_cost'], 4) for a, t in table.items()},
                      'gains': gains,
                      'confirmed_base_dominated_by_GLS': result['confirmed_base_dominated_by_GLS'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
