"""Sealed, per-day I3 teacher collection; retains every registered state.

Formal collection activates only after a qualified A3 failure and a matching
externally sealed protocol. Shards use global dataset day IDs, allowing merge
without changing instance/RNG identities. No development/final-test inputs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(spec):
    if sha(spec['path']) != spec['digest']:
        raise ValueError('External registration/activation identity mismatch')
    return json.loads(Path(spec['path']).read_text(encoding='utf-8'))


def validate_config(config, day):
    cc = config['collection']
    if (config['schema'] != 'i3_teacher_collection_v1' or
            config['qualification'] not in ('engineering', 'formal_development') or
            cc['train_seed'] != 20260925 or cc['continuation_time_limit'] != 'infinity' or
            (cc['future_policy'], cc['energy_pricing'], cc['shadow_mode']) !=
            ('density', 'marginal', 'greedy')):
        raise ValueError('Unsupported collection seed/qualification/policy')
    days, ordinals = cc['days'], cc['capture_ordinals']
    if (not days or not ordinals or day not in days or
            len(set(days)) != len(days) or len(set(ordinals)) != len(ordinals) or
            any(type(d) is not int or not 0 <= d < cc['history_days'] for d in days) or
            any(type(i) is not int or not 0 <= i < cc['orders'] for i in ordinals) or
            ordinals != sorted(ordinals) or cc['K'] <= 0 or cc['minimum_gap'] != 5):
        raise ValueError('Explicit valid global days/all state ordinals required')
    if not config['source_sha256']:
        raise ValueError('Missing source closure')
    excluded = cc['pool_excluded_days']
    if (len(set(excluded)) != len(excluded) or
            any(type(d) is not int or not 0 <= d < cc['history_days'] for d in excluded)):
        raise ValueError('Invalid teacher continuation pool exclusion')
    if config['qualification'] == 'formal_development':
        if cc['not_formal_training_config'] is not False or (cc['history_days'], cc['orders']) != (200, 200):
            raise ValueError('Formal teacher collection requires the registered 200-day problem')
        activation = sealed(config['activation_a3'])
        if (activation.get('formal_adjudicable') is not True or activation.get('partial') or
                activation.get('n_batches') != 9):
            raise ValueError('Formal I3 requires a complete qualified A3 answer')
        for seed in (20260926, 20260930, 20261001):
            tier = activation['summary_by_arm_seed']['B'][str(seed)]['main_tier']
            if tier['formal_adjudication'] is not True or tier['numeric_passed'] is not False:
                raise ValueError('I3 failure branch requires B main gates to fail formally')
        protocol = sealed(config['protocol'])
        if protocol.get('predeclared') is not True or protocol.get('teacher_collection') != cc:
            raise ValueError('Teacher configuration differs from registered protocol')
    elif cc['not_formal_training_config'] is not True:
        raise ValueError('Engineering collection must remain explicitly engineering')


def collect(config_path, config_sha256, day, out_path):
    if sha(config_path) != config_sha256:
        raise ValueError('Collection configuration identity mismatch')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(config, day)
    cc = config['collection']
    required = {'scripts/training/collect_i3_teachers.py',
                'scripts/training/i3_counterfactual_teacher.py',
                'scripts/training/i3_teacher_archive.py', 'scripts/simulation/recourse_snapshot.py',
                'scripts/evaluation/run_exp_encoder_v3.py', 'scripts/evaluation/run_identity.py'}
    for rel, want in config['source_sha256'].items():
        path = EXT / rel
        if not path.resolve().is_relative_to(EXT.resolve()) or sha(path) != want:
            raise ValueError('Collection source mismatch: ' + rel)
    for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(ROOT / folder))
    import numpy as np
    from i3_counterfactual_teacher import (CaptureTeacherReplanner, ReplaySafeSaaReplanner,
                                           replay_action, scenario_margin, teacher_preference)
    from i3_teacher_archive import write_archive, load_archive
    from scenario_saa import (build_history, build_history_times_classes,
                              CondHistoricalSampler, make_c0_contract_v2)
    from run_identity import dataset_identity, dataset_meta_sha256, SHARED_SOURCE_FILES
    from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from strict_online_env import StrictOnlineEnv
    if not (required | set(SHARED_SOURCE_FILES)) <= set(config['source_sha256']):
        raise ValueError('Incomplete teacher source closure')
    if not all(np.isfinite(cc[k]) and cc[k] > 0 for k in ('budget', 'cooling_share', 'capacity')):
        raise ValueError('Invalid physical collection configuration')
    history_data = generate_dataset(cc['history_days'], cc['orders'], cc['train_seed'])
    contract = make_c0_contract_v2()
    history = build_history(history_data)
    tc = build_history_times_classes(history_data)
    budget, cooling, _ = compute_budget_and_dwell(history_data, contract, cc['vehicles'],
        cc['train_seed'], cc['rho'], tc, k=cc['k_neighbors'])
    np.testing.assert_allclose([budget, cooling], [cc['budget'], cc['cooling_share']], rtol=0, atol=1e-8)
    data = add_v2_initial_quality(history_data, contract)
    source_start = dict(config['source_sha256'])
    data_sha, data_meta = dataset_identity(data), dataset_meta_sha256(data)
    # Dataset retains its original global IDs, including days not captured.
    pool = [h for i, h in enumerate(history) if i not in set(cc['pool_excluded_days']) | {day}]
    if not pool:
        raise ValueError('Teacher continuation pool is empty')

    def factory(capture=False):
        cls = CaptureTeacherReplanner if capture else ReplaySafeSaaReplanner
        opts = {'capture_ordinals': cc['capture_ordinals']} if capture else {}
        return cls(budget=cc['budget'], capacity=cc['capacity'], booking_horizon=BOOKING_HORIZON,
            contract=contract, cooling_share=cc['cooling_share'],
            sampler=CondHistoricalSampler(pool, n_neighbors=cc['k_neighbors']),
            reject_penalty={0: 5, 1: 10, 2: 15}, K=cc['K'], time_limit=float('inf'), arm_seed=7001,
            shadow_mode='greedy', energy_pricing='marginal', future_policy='density', **opts)

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=False)
    out.joinpath('submitted_config.json').write_bytes(Path(config_path).read_bytes())
    start = time.perf_counter()
    policy, records = factory(capture=True), []
    env = StrictOnlineEnv(data, capacity=cc['capacity'], tw_speed=SPEED_KMH / KM_PER_UNIT,
        num_vehicles=cc['vehicles'], replanner=policy,
        coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.snapshot_hook = policy.capture_hook(records, train_seed=cc['train_seed'],
        dataset_sha256=data_sha, dataset_meta_sha256=data_meta)
    traces, _ = env.run(day)
    original = evaluate_trace_a1(traces, {k: v[day] for k, v in data.items()}, contract,
        policy._accepted, policy._rejected, policy.budget, policy.reject_penalty)
    if len(records) != len(cc['capture_ordinals']) or not original['hard_feasible']:
        raise RuntimeError('Incomplete captures or invalid original trajectory')
    entries, rows = [], []
    with out.joinpath('progress.jsonl').open('x', encoding='utf-8') as stream:
        for ordinal, record in zip(cc['capture_ordinals'], records):
            branches = [replay_action(record, data, factory, contract, BOOKING_HORIZON,
                                      SPEED_KMH / KM_PER_UNIT, accept) for accept in (True, False)]
            preferred = teacher_preference(*branches, minimum_gap=cc['minimum_gap'])
            chosen = branches[0 if record.candidate in policy._accepted else 1]
            parity = (chosen['action_feasible'] and chosen['accepted'] == sorted(policy._accepted)
                and chosen['rejected'] == sorted(policy._rejected)
                and all(abs(chosen[key] - original[key]) < 1e-8 for key in
                        ('utility', 'energy_kwh', 'distance_km', 'revenue', 'fuel_cost', 'reject_loss')))
            if not parity:
                raise RuntimeError('Original action trajectory replay mismatch')
            scenes = factory().sampler.sample(record.visible,
                np.random.default_rng([9031, day, ordinal]), cc['K'])
            diagnostic = scenario_margin(record, data, factory, contract, BOOKING_HORIZON,
                                         SPEED_KMH / KM_PER_UNIT, scenes)
            margin = diagnostic['margin']
            regret = None
            if preferred['teacher_valid'] and margin is not None:
                regret = abs(preferred['teacher_delta']) if (margin > 0) != (preferred['teacher_delta'] > 0) else 0.
            row = {'day_id': day, 'ordinal': ordinal, 'same_action_parity': True,
                   'hist_resampled_margin': margin, 'hist_teacher_regret': regret, **preferred}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            saved = [{**branch, **({'traces': [asdict(t) for t in branch['traces']]}
                        if 'traces' in branch else {})} for branch in branches]
            entries.append({'day_id': day, 'ordinal': ordinal, 'record': record,
                            'branches': saved, 'preference': preferred})
    context = {'train_seed': cc['train_seed'], 'dataset_sha256': data_sha,
        'dataset_meta_sha256': data_meta, 'contract_hash': contract.contract_hash,
        'source_sha256': {rel.removeprefix('scripts/'): h for rel, h in source_start.items()},
        'generator_config': {'history_days': cc['history_days'], 'orders': cc['orders'],
                            'seed': cc['train_seed'], 'probe_days': cc['history_days'], 'initial_quality': 'v2'},
        'collection_config': cc, 'history_dataset_sha256': dataset_identity(history_data),
        'history_dataset_meta_sha256': dataset_meta_sha256(history_data)}
    keys = [(day, i) for i in cc['capture_ordinals']]
    digest = write_archive(out / 'private.json', context=context, dataset=data,
                           entries=entries, expected_keys=keys)
    load_archive(out / 'private.json', archive_sha256=digest, expected_context=context, expected_keys=keys)
    source_end = {rel: sha(EXT / rel) for rel in source_start}
    if source_start != source_end or sha(config_path) != config_sha256:
        raise RuntimeError('Teacher source changed during collection')
    report = {'qualification': config['qualification'], 'config_sha256': config_sha256,
        'collection': cc, 'context': context, 'expected_keys': keys, 'records': rows,
        'archive_sha256': digest, 'source_stable': True, 'source_start': source_start,
        'source_end': source_end, 'elapsed_s': time.perf_counter() - start,
        'formal_training': False, 'online_gain_established': False}
    out.joinpath('report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('COLLECTION_COMPLETE day=%s states=%s qualification=%s' % (day, len(keys), config['qualification']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--day-id', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    collect(args.config, args.config_sha256, args.day_id, args.out)
