"""D-ANYTIME-v1: how much of the LNS dominance is iteration count rather than mechanism?

Origin. D-CTRL-v1 compared the zero-guide masked reconstructor (H) against a
non-learned destroy/regret-2 LNS at a nominally equal 10.0s wall budget and found
LNS cheaper by 31% on two cohorts. Inspecting the recorded wall times showed the
comparison was not effectively compute-equal: H is iteration-limited to a fixed
schedule of eight masks and self-terminated after a mean of 0.29s, so it never used
97% of its allowance, while LNS spent 9.99s and completed ~277 destroy-repair cycles.
The D-EQ8 draft had already warned that the panel "controls opportunities, not
consumed compute"; this panel separates the two.

Arms (same initial solution, same contract, same 10.0s wall limit):
  H8      the registered eight-fixed-mask reconstructor, unchanged (continuity
          check: its mean must reproduce 105.2283 on the dev cohort)
  ANY30   the same reconstructor run as an anytime loop -- fresh 30% masks until the
          deadline, greedy accept of certified improvements -- with no reversion,
          because stopping at the deadline is the intended behaviour rather than a
          failure
  ANY50   identical to ANY30 with a 50% mask fraction, as a pre-registered secondary
          probe of mask size
  LNS     the same-contract strong control, unchanged

Registered reading rule:
  * ANY30 vs H8 isolates the iteration effect. If ANY30 beats H8 with 95% CI lower > 0,
    the D-CTRL-v1 dominance is at least partly an artefact of the eight-mask schedule.
  * ANY30 vs LNS measures the residual mechanism effect after iterations are equalised.
  * If ANY30 still loses to LNS by >= 1% with CI upper < 0, the shortfall is
    attributable to the mask/acceptance mechanism rather than to iteration count.

Scope. Non-learning only (no model is loaded; every arm has zero neural decodes). This
is a pre-dispatch snapshot panel with simulated parent physics. It establishes nothing
about online dispatch, and because the anytime arms use deadline-termination rather
than fixed-schedule qualification, their results must not be merged into the
qualification state of D-EQ8-v1 or D-CTRL-v1.

Usage:
    python delivery_anytime.py --protocol P.json --out NEWDIR
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_baselines import improve_regret_lns, matrices
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import digest, dump_new, generate, make_schedule, run_arm, reconstruct
from delivery_maskco import partial_adjacency

PROTOCOL = {
    'version': 'D-ANYTIME-v1',
    'kind': 'iteration_effect_isolation_not_gain_claim',
    'parent_panels': ['D-EQ8-v1', 'D-CTRL-v1'],
    'dataset_seed_dev': 2026101002,
    'dataset_seed_confirmation': 2026101013,
    'dev_instance_ids': list(range(32)),
    'confirmation_instance_ids': list(range(1000, 1032)),
    'mask_seed_dev': 2026101041,
    'mask_seed_confirmation': 2026101052,
    'anytime_seed_dev': 2026101131,
    'anytime_seed_confirmation': 2026101132,
    'steps': 8,
    'mask_fraction': 0.3,
    'mask_fraction_secondary': 0.5,
    'time_limit_s': 10.0,
    'lns_seed': 41,
    'bootstrap_seed': 2026101133,
    'bootstrap_repetitions': 2000,
    'dominance_threshold': 0.01,
    'iteration_effect_rule': 'ANY30 beats H8 if mean relative gain>=1% and CI lower>0',
    'mechanism_residual_rule': 'ANY30 loses to LNS if mean relative gain<=-1% and CI upper<0',
    'deadline_termination_arms': ['ANY30', 'ANY50'],
    'qualification_merge_forbidden_with': ['D-EQ8-v1', 'D-CTRL-v1'],
    'claims_online_gain': False,
    'all_failures_retained': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-ANYTIME-v1', 'dataset_seed_dev': 2026101002,
                'dataset_seed_confirmation': 2026101013, 'mask_seed_dev': 2026101041,
                'mask_seed_confirmation': 2026101052, 'anytime_seed_dev': 2026101131,
                'anytime_seed_confirmation': 2026101132, 'steps': 8, 'mask_fraction': .3,
                'mask_fraction_secondary': .5, 'time_limit_s': 10., 'lns_seed': 41,
                'bootstrap_seed': 2026101133, 'bootstrap_repetitions': 2000,
                'dominance_threshold': .01}
    if any(p.get(k) != v or type(p.get(k)) is bool for k, v in expected.items()):
        raise ValueError('D-ANYTIME-v1 registered field changed; no mismatch bypass')
    if p.get('dev_instance_ids') != list(range(32)):
        raise ValueError('Dev cohort must be the thirty-two recorded instances')
    if p.get('confirmation_instance_ids') != list(range(1000, 1032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def anytime_reconstruct(instance, initial_routes, contract, time_limit_s, mask_fraction,
                        seed, clock=time.perf_counter):
    """Same reconstructor as the registered panel, driven by the deadline not by a list.

    No reversion on reaching the deadline: the best certified incumbent found so far
    is the arm's answer, because running until the deadline is the intended behaviour.
    """
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError('Invalid wall budget')
    if not 0 < mask_fraction <= 1:
        raise ValueError('Invalid mask fraction')
    started = clock()
    deadline = started + time_limit_s
    initial = evaluate(instance, initial_routes, contract)
    if not initial['hard']:
        raise ValueError('Unqualified common initial incumbent; do not screen/resample')
    dist = matrices(instance)
    nodes = len(dist)
    rng = np.random.default_rng(seed)
    size = max(1, int(math.ceil(mask_fraction * (nodes - 1))))
    best, report = deepcopy(initial_routes), initial
    cycles = complete = hard = accepted = 0
    while clock() < deadline:
        removed = set(map(int, rng.choice(np.arange(1, nodes), size=size, replace=False)))
        adjacency = partial_adjacency(best, removed, nodes)
        candidate, remaining, _insertions, status = reconstruct(
            instance, best, contract, removed, adjacency, None, 0., deadline, clock)
        cycles += 1
        if remaining:
            if status == 'deadline' or clock() >= deadline:
                break
            continue
        complete += 1
        if clock() >= deadline:
            break
        checked = evaluate(instance, candidate, contract, deadline=deadline, clock=clock)
        if checked['hard']:
            hard += 1
            if checked['cost'] < report['cost']:
                best, report = deepcopy([r for r in candidate if r]), checked
                accepted += 1
        if clock() >= deadline:
            break
    return {'routes': best, 'report': report, 'wall_s': clock() - started,
            'cycles': cycles, 'complete_candidates': complete, 'hard_candidates': hard,
            'accepted': accepted, 'first_endpoint_cost': initial['cost'],
            'final_endpoint_cost': report['cost'], 'mask_fraction': mask_fraction,
            'decode_calls': 0, 'termination': 'deadline',
            'qualification_semantics': 'deadline_terminated_not_fixed_schedule'}


def run_cohort(cohort, instance_ids, dataset_seed, mask_seed, anytime_seed, protocol, contract):
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
        arms['H8'] = {'final_cost': h['final_endpoint_cost'], 'wall_s': h['wall_s'],
                      'timeout': h['timeout'], 'actual_attempts': h['actual_attempts'],
                      'decode_calls': 0, 'routes': h['routes']}
        for name, frac, seed in (('ANY30', protocol['mask_fraction'], anytime_seed),
                                 ('ANY50', protocol['mask_fraction_secondary'], anytime_seed + 1)):
            out = anytime_reconstruct(data, initial, contract, protocol['time_limit_s'],
                                      frac, seed)
            arms[name] = {'final_cost': out['final_endpoint_cost'], 'wall_s': out['wall_s'],
                          'cycles': out['cycles'],
                          'complete_candidates': out['complete_candidates'],
                          'hard_candidates': out['hard_candidates'],
                          'accepted': out['accepted'], 'decode_calls': 0,
                          'status': 'deadline_complete', 'routes': out['routes']}
        lns = improve_regret_lns(data, initial, contract,
                                 time_limit_s=protocol['time_limit_s'],
                                 seed=protocol['lns_seed'])
        arms['LNS'] = {'final_cost': lns['report']['cost'], 'wall_s': lns['wall_s'],
                       'timeout': lns['timeout'], 'tried': lns.get('tried', 0),
                       'decode_calls': 0, 'routes': lns['routes']}
        for arm in ('H8', 'ANY30', 'ANY50', 'LNS'):
            replay = evaluate(data, arms[arm]['routes'], contract)
            if not replay['hard']:
                raise RuntimeError('Arm %s replay not hard feasible: %d' % (arm, instance_id))
            if abs(replay['cost'] - arms[arm]['final_cost']) > 1e-9:
                raise RuntimeError('Arm %s replay cost mismatch: %d' % (arm, instance_id))
        rows.append({'cohort': cohort, 'instance_id': int(instance_id),
                     'schedule_sha256': digest(schedule),
                     'initial_cost': checked['cost'],
                     'arms': {k: {kk: vv for kk, vv in v.items() if kk != 'routes'}
                              for k, v in arms.items()}})
        print(json.dumps({'cohort': cohort, 'instance_id': instance_id,
                          'completed': len(rows),
                          'H8': round(arms['H8']['final_cost'], 4),
                          'ANY30': round(arms['ANY30']['final_cost'], 4),
                          'ANY50': round(arms['ANY50']['final_cost'], 4),
                          'LNS': round(arms['LNS']['final_cost'], 4)}), flush=True)
    return rows


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


def summarise(rows, cohort, protocol):
    table = {}
    for arm in ('H8', 'ANY30', 'ANY50', 'LNS'):
        costs = [r['arms'][arm]['final_cost'] for r in rows]
        table[arm] = {
            'n': len(costs), 'mean_final_cost': statistics.fmean(costs),
            'mean_wall_s': statistics.fmean([r['arms'][arm]['wall_s'] for r in rows]),
            'n_timeout': sum(bool(r['arms'][arm].get('timeout')) for r in rows),
            'total_cycles': sum(r['arms'][arm].get('cycles', r['arms'][arm].get('tried', 0))
                                for r in rows),
            'total_accepted': sum(r['arms'][arm].get('accepted', 0) for r in rows),
            'total_decode_calls': sum(r['arms'][arm].get('decode_calls', 0) for r in rows)}
    by = {arm: {r['instance_id']: r['arms'][arm]['final_cost'] for r in rows}
          for arm in ('H8', 'ANY30', 'ANY50', 'LNS')}
    bs = protocol['bootstrap_seed']
    reps = protocol['bootstrap_repetitions']
    gains = {
        'ANY30_over_H8': paired(by['H8'], by['ANY30'], bs, reps, 'H8', 'ANY30'),
        'ANY50_over_H8': paired(by['H8'], by['ANY50'], bs, reps, 'H8', 'ANY50'),
        'LNS_over_ANY30': paired(by['ANY30'], by['LNS'], bs, reps, 'ANY30', 'LNS'),
        'LNS_over_H8': paired(by['H8'], by['LNS'], bs, reps, 'H8', 'LNS')}
    thr = protocol['dominance_threshold']
    def verdict(g):
        lo, hi = g['descriptive_95_CI']
        if g['mean_relative_gain'] >= thr and lo > 0:
            return 'other_dominates'
        if g['mean_relative_gain'] <= -thr and hi < 0:
            return 'control_dominates'
        return 'indistinguishable'
    return {'cohort': cohort, 'table': table, 'gains': gains,
            'verdicts': {k: verdict(v) for k, v in gains.items()}}


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
        'deadline_termination_arms': protocol['deadline_termination_arms'],
        'qualification_merge_forbidden_with': protocol['qualification_merge_forbidden_with'],
        'smoke_test_truncated': args.max_instances is not None,
        'smoke_test_max_instances': args.max_instances,
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})
    result = {'protocol': protocol,
              'smoke_test_truncated': args.max_instances is not None}
    for cohort, ids, dseed, mseed, aseed, key in (
            ('dev', protocol['dev_instance_ids'], protocol['dataset_seed_dev'],
             protocol['mask_seed_dev'], protocol['anytime_seed_dev'], 'dev_progress.jsonl'),
            ('confirmation', protocol['confirmation_instance_ids'],
             protocol['dataset_seed_confirmation'], protocol['mask_seed_confirmation'],
             protocol['anytime_seed_confirmation'], 'confirmation_progress.jsonl')):
        if args.max_instances is not None:
            ids = ids[:args.max_instances]
        rows = run_cohort(cohort, ids, dseed, mseed, aseed, protocol, contract)
        with (out / key).open('x', encoding='utf-8') as sink:
            for row in rows:
                sink.write(json.dumps(row, allow_nan=False) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
        result[cohort] = summarise(rows, cohort, protocol)
    result['online_gain_established'] = False
    result['stable_gain_established'] = False
    result['wall_s'] = time.perf_counter() - started
    dump_new(out / 'result.json', result)
    print(json.dumps({'dev_means': {a: t['mean_final_cost'] for a, t in result['dev']['table'].items()},
                      'confirmation_means': {a: t['mean_final_cost']
                                             for a, t in result['confirmation']['table'].items()},
                      'dev_verdicts': result['dev']['verdicts'],
                      'confirmation_verdicts': result['confirmation']['verdicts'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-instances', type=int, default=None)
    main(parser.parse_args())
