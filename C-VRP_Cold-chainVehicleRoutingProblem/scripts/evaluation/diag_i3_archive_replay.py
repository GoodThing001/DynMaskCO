"""Independent replay of a complete PRIVATE engineering teacher archive.

Only supports the explicit probe configurations; not a formal collection or
training runner. Verifies caller report checksum/context and current source,
rebuilds the sealed historical pool, then reruns BOTH branches for every state.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_teacher_archive import load_archive
from i3_counterfactual_teacher import TRAIN_SEED, ReplaySafeSaaReplanner, replay_action
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import make_c0_contract_v2, build_history, CondHistoricalSampler
from run_identity import dataset_identity, dataset_meta_sha256
from recourse_snapshot import snapshot_state_hash


def verify(report_path, archive_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    config = report['config']
    if (config['train_seed'] != TRAIN_SEED or not config['not_formal_training_config'] or
            config['continuation_time_limit'] != 'infinity' or
            config['version'] not in ('engineering_small_train_probe_v1',
                                     'engineering_200_order_train_probe_v1')):
        raise ValueError('Unsupported or non-training engineering archive')
    for rel, want in report['source_sha256'].items():
        path = ROOT / rel
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Source path escapes extension scripts')
        if hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise ValueError('Replay source mismatch: ' + rel)
    context = {'train_seed': TRAIN_SEED, 'dataset_sha256': report['dataset_sha256'],
        'dataset_meta_sha256': report['dataset_meta_sha256'],
        'contract_hash': report['contract_hash'], 'source_sha256': report['source_sha256'],
        'generator_config': {'history_days': config['history_days'], 'orders': config['orders'],
                             'seed': TRAIN_SEED, 'probe_days': config['probe_days'],
                             'initial_quality': 'v2'},
        'collection_config': config,
        'history_dataset_sha256': report['history_dataset_sha256'],
        'history_dataset_meta_sha256': report['history_dataset_meta_sha256']}
    keys = [(d, i) for d in range(config['probe_days']) for i in config['capture_ordinals']]
    restored = load_archive(archive_path, archive_sha256=report['private_archive']['sha256'],
                            expected_context=context, expected_keys=keys)
    history_data = generate_dataset(config['history_days'], config['orders'], TRAIN_SEED)
    if (dataset_identity(history_data) != context['history_dataset_sha256'] or
            dataset_meta_sha256(history_data) != context['history_dataset_meta_sha256']):
        raise ValueError('Regenerated teacher history identity mismatch')
    history, contract = build_history(history_data), make_c0_contract_v2()
    if contract.contract_hash != context['contract_hash']:
        raise ValueError('Replay contract mismatch')
    count = 0
    for entry in restored['entries']:
        day = entry['day_id']
        pool = [h for i, h in enumerate(history) if i != day]

        def factory():
            return ReplaySafeSaaReplanner(budget=config['budget'], capacity=50,
                booking_horizon=BOOKING_HORIZON, contract=contract,
                cooling_share=config.get('cooling_share', 1),
                sampler=CondHistoricalSampler(pool), reject_penalty={0: 5, 1: 10, 2: 15},
                K=config['K'], time_limit=float('inf'), arm_seed=7001,
                shadow_mode='greedy', energy_pricing='marginal', future_policy='density')

        for accept, saved in zip((True, False), entry['branches']):
            branch = replay_action(entry['record'], restored['dataset'], factory, contract,
                                   BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, accept)
            if branch['action_feasible'] != saved['action_feasible']:
                raise ValueError('Branch action feasibility changed')
            if branch['action_feasible']:
                for metric in ('utility', 'energy_kwh', 'distance_km', 'revenue',
                               'fuel_cost', 'reject_loss'):
                    np.testing.assert_allclose(branch[metric], saved[metric], rtol=0, atol=1e-8)
                if (branch['hard_feasible'] != saved['hard_feasible'] or
                        branch['accepted'] != saved['accepted'] or
                        branch['rejected'] != saved['rejected'] or
                        branch['timeouts'] != saved['timeouts']):
                    raise ValueError('Replayed terminal branch outcomes changed')
                traces = [asdict(trace) for trace in branch['traces']]
                if snapshot_state_hash({'traces': traces}) != snapshot_state_hash({'traces': saved['traces']}):
                    raise ValueError('Full archived branch trace changed')
            count += 1
    return {'engineering_archive_replay_pass': True, 'records': len(keys),
            'independently_replayed_branches': count,
            'dataset_sha256': context['dataset_sha256'],
            'archive_sha256': report['private_archive']['sha256'],
            'source_sha256': report['source_sha256'],
            'formal_training': False, 'online_gain_established': False,
            'scope': 'All predeclared engineering states, both full infinite-budget branches'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify(args.report, args.archive)
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('ALL PASS: independent complete private archive replay, states=%s branches=%s' %
          (result['records'], result['independently_replayed_branches']))
