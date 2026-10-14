"""D-DOSE-v1: dose-response of the guide weight in the delivery reconstructor.

Parent evidence. D-EQ8-v1 (n=32, equal eight-attempt schedule) found the pretrained
CVRP guide WORSE than a random same-architecture guide and both worse than no guide
at all (114.699 / 111.656 / 105.228). D-GUIDE-v1 then showed, on the recorded
decision states, that the guide is harmful in both of its insertion roles and that
the position role dominates (+2.18 per decision versus +0.31 for the regret role).

Hypothesis under test. The guide term's *magnitude*, not its direction, is what
makes it harmful: at a guide weight scaled below 1.0 the pretrained guide will beat
the zero-guide reconstructor on terminal contract cost under an identical schedule.

Design. One factor (guide weight) at a registered grid, on two disjoint cohorts:
  dev          -- the already-burned D-EQ8-v1 instance population, used to SELECT
  confirmation -- a fresh data seed and fresh instance ids, run ONLY for the
                  selected weight, and used to ADMIT or reject
Selection on dev, admission on confirmation. The zero-guide arm is present in both
cohorts and is the control, so no external baseline is needed. Because the guide
term is multiplied by the weight, the weight-0 arm must be exactly the no-guide
reconstructor; that identity is asserted rather than assumed.

Registered decision rule, fixed before any data:
  * select gw* = argmax mean paired relative improvement over gw=0 on dev among arms
    with mean improvement >= 1% and paired 95% CI lower > 0; tie-break smaller weight
  * if no arm qualifies, close the hypothesis negative and DO NOT run confirmation
  * confirm only if the confirmation cohort shows mean improvement >= 1% and a paired
    95% CI lower > 0; the weight-1.0 arm is carried into confirmation as a
    registered sanity arm expected to stay negative
Nothing is reselected after seeing results: masks, instances, weights and seeds are
all protocol constants.

Scope limits. This is a pre-dispatch route-snapshot panel with simulated physics
from the parent protocol; it is not an online dynamic result, and simulated energy
and literature quality kinetics are not a real-vehicle or food-safety claim.

Usage:
    python delivery_guide_dose.py --stage qa    --checkpoint CKPT --upstream ROOT --out NEWDIR
    python delivery_guide_dose.py --stage panel --checkpoint CKPT --upstream ROOT \
        --out NEWDIR --qa-artifact QA/result.json
"""
import argparse
from copy import deepcopy
import hashlib
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

from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import (digest, dump_new, generate, make_schedule,
                                     parameter_digest, run_arm)
from delivery_maskco import MaskCORouteProvider

PROTOCOL = {
    'version': 'D-DOSE-v1',
    'parent_panel': 'D-EQ8-v1',
    'parent_diagnosis': 'D-GUIDE-v1',
    'factor': 'guide_weight',
    'dataset_seed_dev': 2026101002,
    'dataset_seed_confirmation': 2026101013,
    'dev_instance_ids': list(range(32)),
    'confirmation_instance_ids': list(range(1000, 1032)),
    'mask_seed_dev': 2026101041,
    'mask_seed_confirmation': 2026101052,
    'arm_order_seed_dev': 2026101099,
    'arm_order_seed_confirmation': 2026101063,
    'guide_weight_grid_P': [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
    'guide_weight_grid_R': [0.0, 0.1, 0.25, 1.0],
    'steps': 8,
    'mask_fraction': 0.3,
    'time_limit_s': 10.0,
    'random_model_seed': 42,
    'bootstrap_seed': 2026101111,
    'bootstrap_repetitions': 2000,
    'exploratory_mean_relative_threshold': 0.01,
    'selection_rule': ('argmax mean paired relative improvement over guide_weight 0 on dev '
                       'among arms with mean>=0.01 and paired 95% CI lower>0; tie-break smaller '
                       'weight; if none qualify close negative and skip confirmation'),
    'confirmation_gate': 'mean paired relative improvement>=0.01 and paired 95% CI lower>0',
    'claims_online_gain': False,
    'all_failures_retained': True,
    'original_panels_immutable': True,
    'physics_note': 'simulated parent-physics snapshot; not physical calibration',
}


def verify_protocol(p):
    expected = {'version': 'D-DOSE-v1', 'dataset_seed_dev': 2026101002,
                'dataset_seed_confirmation': 2026101013, 'mask_seed_dev': 2026101041,
                'mask_seed_confirmation': 2026101052, 'arm_order_seed_dev': 2026101099,
                'arm_order_seed_confirmation': 2026101063, 'steps': 8,
                'mask_fraction': .3, 'time_limit_s': 10., 'random_model_seed': 42,
                'bootstrap_seed': 2026101111, 'bootstrap_repetitions': 2000,
                'guide_weight_grid_P': [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
                'guide_weight_grid_R': [0.0, 0.1, 0.25, 1.0],
                'exploratory_mean_relative_threshold': .01}
    if any(p.get(k) != v or type(p.get(k)) is bool for k, v in expected.items()):
        raise ValueError('D-DOSE-v1 registered field changed; no mismatch bypass')
    if p.get('dev_instance_ids') != list(range(32)):
        raise ValueError('Dev cohort must be the thirty-two recorded instances')
    if p.get('confirmation_instance_ids') != list(range(1000, 1032)):
        raise ValueError('Confirmation cohort ids are registered and fixed')
    if set(p['guide_weight_grid_P']) & set(p['guide_weight_grid_R']) != {0.0, 0.1, 0.25, 1.0}:
        raise ValueError('Weight-0 control must be present in both grids')
    if p.get('claims_online_gain') is not False:
        raise ValueError('Scope cannot become an online gain claim')
    return expected


def build_providers(checkpoint, upstream, recorded_identity):
    providers = {'P': MaskCORouteProvider(str(checkpoint), upstream),
                 'R': MaskCORouteProvider(str(checkpoint), upstream,
                                          random_seed=PROTOCOL['random_model_seed'])}
    local = {arm: parameter_digest(providers[arm]) for arm in ('P', 'R')}
    for arm in ('P', 'R'):
        if local[arm] != recorded_identity['model_parameters'][arm]:
            raise ValueError('Local model parameters differ from the recorded identity: ' + arm)
        providers[arm].prewarm(51, DeliveryContract().capacity)
    if local['P'] == local['R']:
        raise ValueError('Pretrained and random parameters must differ')
    return providers, local


def synthetic_input():
    nodes = 51
    coords = np.stack([np.arange(nodes) / 10., np.zeros(nodes)], axis=1)
    data = {'coords': coords.tolist(), 'demands': [0] + [1] * 50,
            'temp_class': [0] + [i % 3 for i in range(50)],
            'reveal_time': [0.] * nodes, 'ready_time': [0.] * nodes,
            'tw_start': [0.] * nodes, 'tw_end': [8.] * nodes,
            'service_time': [0.] + [.03] * 50}
    initial = [list(range(1 + 10 * k, 11 + 10 * k)) for k in range(5)]
    return data, initial, make_schedule(nodes, 8, 9001, 0)


def dose_arms():
    arms = [('P', gw) for gw in PROTOCOL['guide_weight_grid_P']]
    arms += [('R', gw) for gw in PROTOCOL['guide_weight_grid_R'] if gw != 0.0]
    return arms


def arm_key(provider_arm, gw):
    return f'{provider_arm}@gw{gw:g}'


def run_instance(cohort, instance_id, dataset_seed, mask_seed, arm_order_seed,
                 contract, providers, arms, time_limit_s, identity):
    data, initial = generate(dataset_seed, instance_id)
    schedule = make_schedule(51, 8, mask_seed, instance_id)
    checked = evaluate(data, initial, contract)
    if not checked['hard']:
        raise RuntimeError('Initial data hard failure retained; no replacement: ' + str(instance_id))
    order = list(np.random.default_rng(np.random.SeedSequence(
        [arm_order_seed, instance_id])).permutation([arm_key(a, g) for a, g in arms]))
    results = {}
    for key in order:
        provider_arm, gw = next((a, g) for a, g in arms if arm_key(a, g) == key)
        provider = providers[provider_arm]
        out = run_arm(data, initial, contract, schedule, provider,
                      time_limit_s=time_limit_s, guide_weight=gw)
        replay = evaluate(data, out['routes'], contract)
        if not replay['hard'] or abs(replay['cost'] - out['report']['cost']) > 1e-9:
            raise RuntimeError('Submitted route replay mismatch: ' + key)
        results[key] = {'provider': provider_arm, 'guide_weight': gw,
                        'final_endpoint_cost': out['final_endpoint_cost'],
                        'first_endpoint_cost': out['first_endpoint_cost'],
                        'timeout': out['timeout'],
                        'fixed_schedule_qualified': out['fixed_schedule_qualified'],
                        'actual_attempts': out['actual_attempts'],
                        'complete_candidates': out['complete_candidates'],
                        'hard_candidates': out['hard_candidates'],
                        'decode_calls': out['decode_calls']}
    zero_keys = [k for k, v in results.items() if v['guide_weight'] == 0.0]
    costs = {results[k]['final_endpoint_cost'] for k in zero_keys}
    return {'cohort': cohort, 'instance_id': int(instance_id),
            'schedule_sha256': digest(schedule), 'arm_order': order,
            'zero_weight_arms_agree': len(costs) == 1, 'arms': results}


def paired_bootstrap(control, other, seed, reps):
    shared = sorted(set(control) & set(other))
    diffs = np.array([(control[i] - other[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'n_instances': len(shared), 'mean_relative_improvement': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_improved': int(sum(d > 0 for d in diffs)),
            'instances_worse': int(sum(d < 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs))}


def analyse(rows, arms):
    """Per-arm cost table plus paired comparison against the weight-0 control."""
    control = {}
    for row in rows:
        for key, value in row['arms'].items():
            if value['guide_weight'] == 0.0:
                control[row['instance_id']] = value['final_endpoint_cost']
                break
    table = {}
    for arm, gw in arms:
        key = arm_key(arm, gw)
        per_instance = {row['instance_id']: row['arms'][key]['final_endpoint_cost']
                        for row in rows if key in row['arms']}
        entry = {'provider': arm, 'guide_weight': gw, 'n': len(per_instance),
                 'mean_final_endpoint_cost': statistics.fmean(per_instance.values()),
                 'n_timeout': sum(row['arms'][key]['timeout'] for row in rows if key in row['arms']),
                 'n_unqualified': sum(not row['arms'][key]['fixed_schedule_qualified']
                                      for row in rows if key in row['arms']),
                 'total_decode_calls': sum(row['arms'][key]['decode_calls']
                                           for row in rows if key in row['arms'])}
        if gw == 0.0:
            entry['comparison_vs_zero_weight'] = None
        else:
            entry['comparison_vs_zero_weight'] = paired_bootstrap(
                control, per_instance, PROTOCOL['bootstrap_seed'],
                PROTOCOL['bootstrap_repetitions'])
        table[key] = entry
    return table


def main(args):
    started = time.perf_counter()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    checkpoint = Path(args.checkpoint)
    recorded = json.loads(Path(args.recorded_identity).read_text(encoding='utf-8'))
    weight_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if weight_sha != recorded['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match the recorded parent identity')
    providers, local_params = build_providers(checkpoint, args.upstream, recorded)

    import jax
    import flax
    identity = {'checkpoint_sha256': weight_sha,
                'local_model_parameters': local_params,
                'recorded_model_parameters': recorded['model_parameters'],
                'local_jax_version': jax.__version__,
                'local_flax_version': flax.__version__,
                'local_numpy_version': np.__version__,
                'server_jax_version': '0.6.2',
                'threads': {k: os.environ.get(k) for k in
                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    dump_new(out / 'PROTOCOL.json', protocol)
    dump_new(out / 'identity_start.json', identity)

    if args.stage == 'qa':
        arms = dose_arms()
        data, initial, schedule = synthetic_input()
        qa = {'stage': 'synthetic_engineering_qa', 'arms': {}}
        for arm, gw in arms:
            result = run_arm(data, initial, contract, schedule, providers[arm],
                             time_limit_s=protocol['time_limit_s'], guide_weight=gw)
            qa['arms'][arm_key(arm, gw)] = {
                'fixed_schedule_qualified': result['fixed_schedule_qualified'],
                'timeout': result['timeout'],
                'actual_attempts': result['actual_attempts'],
                'complete_candidates': result['complete_candidates'],
                'hard_candidates': result['hard_candidates'],
                'decode_calls': result['decode_calls']}
        qa['all_arms_qualified'] = all(v['fixed_schedule_qualified'] and not v['timeout'] and
                                       v['actual_attempts'] == 8 and v['complete_candidates'] == 8
                                       and v['hard_candidates'] == 8 for v in qa['arms'].values())
        qa['identity'] = {'checkpoint_sha256': weight_sha, 'model_parameters': local_params}
        dump_new(out / 'result.json', qa)
        print(json.dumps({'qa_pass': qa['all_arms_qualified'], 'arms': len(qa['arms']),
                          'wall_s': round(time.perf_counter() - started, 1)}, indent=2))
        if not qa['all_arms_qualified']:
            raise SystemExit('Synthetic QA failed; do not run the panel')
        return

    if args.stage != 'panel':
        raise ValueError('Unknown stage')
    if not args.qa_artifact:
        raise ValueError('Actual successful synthetic QA artifact required')
    qa_raw = Path(args.qa_artifact).read_bytes()
    qa = json.loads(qa_raw)
    if not qa.get('all_arms_qualified') or qa.get('stage') != 'synthetic_engineering_qa':
        raise ValueError('Synthetic QA did not qualify the dose arms')
    if qa.get('identity', {}).get('model_parameters') != local_params:
        raise ValueError('Synthetic QA identity differs from this run')
    identity['synthetic_qa_artifact_sha256'] = hashlib.sha256(qa_raw).hexdigest()

    arms = dose_arms()
    dev_arms = list(arms)
    dev_rows = []
    with (out / 'dev_progress.jsonl').open('x', encoding='utf-8') as sink:
        for instance_id in protocol['dev_instance_ids']:
            row = run_instance('dev', instance_id, protocol['dataset_seed_dev'],
                               protocol['mask_seed_dev'], protocol['arm_order_seed_dev'],
                               contract, providers, dev_arms, protocol['time_limit_s'], identity)
            sink.write(json.dumps(row, allow_nan=False) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
            dev_rows.append(row)
            print(json.dumps({'stage': 'dev', 'instance_id': instance_id,
                              'completed': len(dev_rows)}), flush=True)
    dev_table = analyse(dev_rows, dev_arms)

    threshold = protocol['exploratory_mean_relative_threshold']
    qualified = []
    for key, entry in dev_table.items():
        cmp_ = entry.get('comparison_vs_zero_weight')
        if cmp_ and (cmp_['mean_relative_improvement'] >= threshold
                     and cmp_['descriptive_95_CI'][0] > 0):
            qualified.append((cmp_['mean_relative_improvement'], -entry['guide_weight'], key, entry))
    qualified.sort(reverse=True)
    selected = None
    if qualified:
        selected = {'arm_key': qualified[0][2],
                    'provider': qualified[0][3]['provider'],
                    'guide_weight': qualified[0][3]['guide_weight'],
                    'dev_mean_relative_improvement': qualified[0][0],
                    'dev_ci': qualified[0][3]['comparison_vs_zero_weight']['descriptive_95_CI']}

    result = {'identity': identity, 'protocol': protocol,
              'dev_table': dev_table, 'dev_selected': selected,
              'dev_zero_weight_arms_agree_on_all_instances':
                  all(row['zero_weight_arms_agree'] for row in dev_rows),
              'dev_all_arms_qualified': all(v['n_unqualified'] == 0 for v in dev_table.values()),
              'confirmation_table': None, 'confirmation_gate': None}

    if selected is None:
        result['outcome'] = 'development_stage_found_no_arm_meeting_the_registered_selectivity_rule'
        result['confirmation_run'] = False
        result['online_gain_established'] = False
        result['stable_gain_established'] = False
    else:
        confirm_weights = sorted({0.0, selected['guide_weight'], 1.0})
        confirm_arms = [(selected['provider'], gw) for gw in confirm_weights]
        confirm_rows = []
        with (out / 'confirmation_progress.jsonl').open('x', encoding='utf-8') as sink:
            for instance_id in protocol['confirmation_instance_ids']:
                row = run_instance('confirmation', instance_id,
                                   protocol['dataset_seed_confirmation'],
                                   protocol['mask_seed_confirmation'],
                                   protocol['arm_order_seed_confirmation'],
                                   contract, providers, confirm_arms,
                                   protocol['time_limit_s'], identity)
                sink.write(json.dumps(row, allow_nan=False) + '\n')
                sink.flush()
                os.fsync(sink.fileno())
                confirm_rows.append(row)
                print(json.dumps({'stage': 'confirmation', 'instance_id': instance_id,
                                  'completed': len(confirm_rows)}), flush=True)
        confirm_table = analyse(confirm_rows, confirm_arms)
        result['confirmation_table'] = confirm_table
        result['confirmation_run'] = True
        key = arm_key(selected['provider'], selected['guide_weight'])
        gate = confirm_table[key]['comparison_vs_zero_weight']
        passed = (gate['mean_relative_improvement'] >= threshold
                  and gate['descriptive_95_CI'][0] > 0
                  and all(v['n_unqualified'] == 0 for v in confirm_table.values()))
        result['confirmation_gate'] = {'arm_key': key, 'passed': bool(passed), **gate}
        result['outcome'] = ('confirmation_passed' if passed
                             else 'confirmation_failed_no_admissible_gain')
        result['online_gain_established'] = False
        result['stable_gain_established'] = bool(passed)
    dump_new(out / 'identity_end.json', identity)
    dump_new(out / 'result.json', result)
    print(json.dumps({'outcome': result['outcome'], 'dev_selected': selected,
                      'wall_s': round(time.perf_counter() - started, 1)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('qa', 'panel'))
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--recorded-identity', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--qa-artifact')
    main(parser.parse_args())
