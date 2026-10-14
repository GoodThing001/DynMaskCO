"""Re-evaluate archived terminal traces and describe all teacher states.

Read-only supplemental audit, never a new training/model-selection gate.
Does not rerun an online policy or select days by their reward/sign.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_capture_positions(collection, day, actual_count=None):
    """Independently derive the registered quartiles from inert counts."""
    if 'capture_ordinals_by_day' not in collection:
        positions = collection['capture_ordinals']
    else:
        design = collection['capture_design']
        if (design['rule'], design['numerators'], design['denominator']) != (
                'ceil(q*N_day)-1', [1,3,5,7], 8):
            raise ValueError('Unregistered teacher sampling design')
        count = design['actual_orders_by_day'][str(day)]
        if type(count) is not int or count < 4:
            raise ValueError('Insufficient actual customers for four positions')
        positions = [(n * count + 7) // 8 - 1 for n in (1,3,5,7)]
        if collection['capture_ordinals_by_day'][str(day)] != positions:
            raise ValueError('Mapped teacher sampling differs from registered quartiles')
        if actual_count is not None and actual_count != count:
            raise ValueError('Regenerated customer count differs from mapped sampling')
    if (len(set(positions)) != len(positions) or positions != sorted(positions)
            or actual_count is not None and any(i >= actual_count for i in positions)):
        raise ValueError('Invalid or nonexistent teacher sampling position')
    return positions


def restore_traces(payload, *, VehicleTrace, ServiceRecord, restore_state, restore_delivery):
    traces = []
    for trace in payload:
        values = deepcopy(trace)
        services = []
        for item in values['services']:
            service = dict(item)
            for key in ('coldchain_before', 'coldchain_after'):
                service[key] = restore_state(service[key])
            services.append(ServiceRecord(**service))
        values['services'] = services
        values['final_coldchain_state'] = restore_state(values['final_coldchain_state'])
        values['depot_unload_records'] = [restore_delivery(item) for item in values['depot_unload_records']]
        traces.append(VehicleTrace(**values))
    return traces


def check_entry(entry, row, dataset, contract, collection, backend):
    day, ordinal = entry['day_id'], entry['ordinal']
    if (row['day_id'], row['ordinal']) != (day, ordinal) or row['same_action_parity'] is not True:
        raise ValueError('Wrong report state or missing original-action parity')
    record, branches = entry['record'], entry['branches']
    if any(order.reveal > record.visible.clock + 1e-6 for order in record.visible.orders):
        raise ValueError('Hidden future in actor-visible condition')
    if record.visible.clock != record.snapshot['clock']:
        raise ValueError('Generator and physical clocks differ')
    if not record.snapshot['visible_mask'][record.candidate]:
        raise ValueError('Candidate was not revealed')
    preferences = backend['teacher_preference'](*branches, minimum_gap=collection['minimum_gap'])
    if entry['preference'] != preferences or any(row[key] != value for key, value in preferences.items()):
        raise ValueError('Report/archive preference mismatch')
    summaries = []
    for accept, branch in zip((True, False), branches):
        if not branch['action_feasible']:
            if branch.get('utility') is not None or branch.get('hard_feasible') is not None or 'traces' in branch:
                raise ValueError('Infeasible action was given a fabricated terminal trajectory')
            summaries.append({'forced_accept': accept, 'action_feasible': False})
            continue
        traces = restore_traces(branch['traces'], **{key: backend[key] for key in
            ('VehicleTrace', 'ServiceRecord', 'restore_state', 'restore_delivery')})
        recomputed = backend['evaluate'](traces, {key: values[day] for key, values in dataset.items()},
            contract, set(branch['accepted']), set(branch['rejected']), collection['budget'], {0: 5, 1: 10, 2: 15})
        for metric in ('utility', 'revenue', 'fuel_cost', 'reject_loss', 'distance_km', 'energy_kwh'):
            if (not np.isfinite(recomputed[metric]) or not np.isfinite(branch[metric]) or
                    not np.isclose(recomputed[metric], branch[metric], rtol=0, atol=1e-8)):
                raise ValueError('Archived terminal metric mismatch: ' + metric)
        if any(recomputed[key] != branch[key] for key in ('hard_feasible', 'failures', 'n_served', 'n_rejected')):
            raise ValueError('Archived terminal feasibility or order counts mismatch')
        if branch['timeouts'] != 0:
            raise ValueError('Infinite-time teacher unexpectedly contains a timeout')
        target, opposite = ('accepted', 'rejected') if accept else ('rejected', 'accepted')
        if record.candidate not in branch[target] or record.candidate in branch[opposite]:
            raise ValueError('Forced branch chose the wrong action')
        summaries.append({'forced_accept': accept, 'action_feasible': True,
            **{key: recomputed[key] for key in ('utility', 'revenue', 'fuel_cost', 'reject_loss',
                'distance_km', 'energy_kwh', 'hard_feasible', 'failures', 'n_served', 'n_rejected')}})
    margin = row['hist_resampled_margin']
    if margin is not None and not np.isfinite(margin):
        raise ValueError('Nonfinite conditional-history diagnostic margin')
    valid = preferences['teacher_valid']
    regret = (abs(preferences['teacher_delta']) if (margin > 0) != (preferences['teacher_delta'] > 0) else 0.) \
        if valid and margin is not None else None
    if regret != row['hist_teacher_regret']:
        raise ValueError('Conditional-history regret differs from raw sign rule')
    delta = preferences['teacher_delta']
    decomposition = None
    if valid:
        decomposition = {key: summaries[0][key] - summaries[1][key]
                         for key in ('revenue', 'fuel_cost', 'reject_loss')}
        if not np.isclose(delta, decomposition['revenue'] - decomposition['fuel_cost'] -
                          decomposition['reject_loss'], rtol=0, atol=1e-8):
            raise ValueError('Teacher marginal utility fails revenue/fuel/rejection accounting')
    return {'day_id': day, 'ordinal': ordinal, 'clock': record.visible.clock, 'candidate': record.candidate,
        'visible_order_count': len(record.visible.orders), **preferences,
        'hist_resampled_margin': margin, 'hist_teacher_regret': regret,
        'hist_direction_wrong': bool(regret > 0) if regret is not None else None,
        'branch_summaries': summaries, 'delta_decomposition': decomposition,
        'terminal_trace_recomputed': True}


def summarize(rows, expected_keys):
    keys = [(row['day_id'], row['ordinal']) for row in rows]
    if not keys or len(set(keys)) != len(keys) or set(keys) != {tuple(key) for key in expected_keys}:
        raise ValueError('Teacher audit cannot trim/duplicate/miss states')
    valid = [row for row in rows if row['teacher_valid']]
    eligible = [row for row in rows if row['decision_eligible']]
    evaluable = [row for row in valid if row['hist_teacher_regret'] is not None]
    delta = np.array([row['teacher_delta'] for row in valid], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite valid teacher delta')
    notes = []
    if not eligible:
        notes.append('No actor-eligible labels; registered training remains fixed, but decision learning has no active signal')
    if not any(row['teacher_delta'] < 0 for row in valid) or not any(row['teacher_delta'] > 0 for row in valid):
        notes.append('Teacher directions are one-sided; this alone does not prove a discriminative learning benefit')
    if evaluable and all(row['hist_teacher_regret'] == 0 for row in evaluable):
        notes.append('Conditional history agrees with every evaluable teacher state; no excess teacher opportunity demonstrated')
    by_day = {str(day): {key: sum(bool(row[key]) for row in rows if row['day_id'] == day)
        for key in ('teacher_valid', 'decision_eligible')} for day in sorted({row['day_id'] for row in rows})}
    return {'states': len(rows), 'days': len(by_day), 'valid_teachers': len(valid),
        'invalid_teachers': len(rows) - len(valid), 'actor_eligible_states': len(eligible),
        'valid_small_or_tie_states': len(valid) - len(eligible),
        'signs': {'positive': sum(row['teacher_delta'] > 0 for row in valid),
                 'negative': sum(row['teacher_delta'] < 0 for row in valid),
                 'tie': sum(row['teacher_delta'] == 0 for row in valid)},
        'valid_delta_quantiles': dict(zip(('min', 'q25', 'median', 'q75', 'max'),
            map(float, np.quantile(delta, [0., .25, .5, .75, 1.])))) if len(delta) else None,
        'conditional_hist_evaluable_states': len(evaluable),
        'conditional_hist_wrong_directions': sum(row['hist_direction_wrong'] for row in evaluable),
        'conditional_hist_regret_sum': float(sum(row['hist_teacher_regret'] for row in evaluable)),
        'by_day': by_day, 'notes': notes,
        'scope': 'All registered states; valid-label statistics are explicitly conditional, no filtering of training states',
        'online_gain_established': False, 'model_selection_changed': False}


def attach_backend(ext):
    for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(Path(ext) / 'scripts' / folder))
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from i3_counterfactual_teacher import teacher_preference
    from recourse_snapshot import _restore_coldchain_state, _restore_delivery_record
    from strict_online_env import VehicleTrace, ServiceRecord
    return {'evaluate': evaluate_trace_a1, 'teacher_preference': teacher_preference,
        'restore_state': _restore_coldchain_state, 'restore_delivery': _restore_delivery_record,
        'VehicleTrace': VehicleTrace, 'ServiceRecord': ServiceRecord}


def verify_shard(main, ext, day):
    cfg_path = Path(main) / 'teacher_collection_config.json'
    config = read(cfg_path)
    folder = Path(main) / ('teacher_day_%03d' % day)
    report_path = folder / 'report.json'
    report_digest = sha(report_path)
    report = read(report_path)
    protocol = read(Path(main) / 'protocol.json')
    if (config['collection'] != protocol['teacher_collection'] or
            report['qualification'] != 'formal_development' or report['config_sha256'] != sha(cfg_path) or
            not report['source_stable'] or report['source_start'] != report['source_end'] or
            report['source_start'] != config['source_sha256'] or report['collection'] != config['collection']):
        raise ValueError('Teacher shard differs from immutable formal configuration')
    for rel, want in report['source_start'].items():
        if sha(Path(ext) / rel) != want:
            raise ValueError('Frozen source mismatch: ' + rel)
    expected = [(day, ordinal) for ordinal in expected_capture_positions(config['collection'], day)]
    if sorted(tuple(key) for key in report['expected_keys']) != expected:
        raise ValueError('Shard capture keys differ from registration')
    backend = attach_backend(ext)
    from i3_teacher_archive import load_archive
    from scenario_saa import make_c0_contract_v2
    from run_exp_reserve import generate_dataset
    from run_identity import dataset_identity, dataset_meta_sha256
    from coldchain_evaluator_a1 import add_v2_initial_quality
    restored = load_archive(folder / 'private.json', archive_sha256=report['archive_sha256'],
        expected_context=report['context'], expected_keys=expected)
    contract = make_c0_contract_v2()
    if contract.contract_hash != report['context']['contract_hash']:
        raise ValueError('Frozen contract differs from teacher archive')
    history = generate_dataset(config['collection']['history_days'], config['collection']['orders'],
                               config['collection']['train_seed'])
    canonical_data = add_v2_initial_quality(history, contract)
    expected_capture_positions(config['collection'], day,
        int(np.count_nonzero(canonical_data['demands'][day, 1:] > 0)))
    context = report['context']
    if (context['collection_config'] != config['collection'] or
            dataset_identity(history) != context['history_dataset_sha256'] or
            dataset_meta_sha256(history) != context['history_dataset_meta_sha256'] or
            dataset_identity(canonical_data) != context['dataset_sha256'] or
            dataset_meta_sha256(canonical_data) != context['dataset_meta_sha256']):
        raise ValueError('Teacher dataset differs from independently regenerated training history')
    indexed = {(row['day_id'], row['ordinal']): row for row in report['records']}
    if len(indexed) != len(report['records']) or set(indexed) != set(expected):
        raise ValueError('Shard public rows are missing or duplicated')
    rows = [check_entry(entry, indexed[(entry['day_id'], entry['ordinal'])], restored['dataset'],
                       contract, config['collection'], backend) for entry in restored['entries']]
    if sha(report_path) != report_digest or sha(folder / 'private.json') != report['archive_sha256']:
        raise ValueError('Teacher artifact changed during verification')
    return {'report_sha256': sha(report_path), 'archive_sha256': report['archive_sha256'],
            'summary': summarize(rows, expected), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--frozen-ext', required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify_shard(args.main, args.frozen_ext, args.day)
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('TEACHER_TERMINAL_AUDIT_PASS day=%s states=%s' % (args.day, result['summary']['states']))
