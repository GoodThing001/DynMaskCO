"""Real generator collector/merge path and qualified-activation guards."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent
for folder in ('training', 'evaluation', 'simulation', 'coldchain', 'tests'):
    sys.path.insert(0, str(ROOT / folder))
from collect_i3_teachers import collect, validate_config, sha
from merge_i3_teachers import merge
from run_identity import SHARED_SOURCE_FILES
from run_exp_reserve import generate_dataset
from scenario_saa import build_history_times_classes, make_c0_contract_v2
from run_exp_encoder_v3 import compute_budget_and_dwell
from test_i3_counterfactual_teacher import check_raises


def configuration():
    ds = generate_dataset(4, 8, 20260925)
    b, c, _ = compute_budget_and_dwell(ds, make_c0_contract_v2(), 3, 20260925, .6,
                                       build_history_times_classes(ds), k=2)
    files = set(SHARED_SOURCE_FILES) | {'scripts/training/collect_i3_teachers.py',
        'scripts/training/i3_counterfactual_teacher.py', 'scripts/training/i3_teacher_archive.py',
        'scripts/simulation/recourse_snapshot.py', 'scripts/evaluation/run_exp_encoder_v3.py',
        'scripts/evaluation/run_identity.py'}
    cc = {'version': 'engineering_collector_path_v1', 'train_seed': 20260925,
        'history_days': 4, 'orders': 8, 'days': [0, 2], 'capture_ordinals': [0, 4],
        'budget': float(b), 'cooling_share': float(c), 'vehicles': 3, 'capacity': 50.,
        'rho': .6, 'k_neighbors': 2, 'K': 1, 'minimum_gap': 5,
        'continuation_time_limit': 'infinity', 'future_policy': 'density',
        'energy_pricing': 'marginal', 'shadow_mode': 'greedy',
        'pool_excluded_days': [], 'not_formal_training_config': True}
    return {'schema': 'i3_teacher_collection_v1', 'qualification': 'engineering',
            'collection': cc, 'source_sha256': {r: sha(EXT / r) for r in files}}


def test(out_root=None):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(out_root) if out_root else Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        cfg = configuration()
        path = root / 'collection_config.json'
        path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        shards = []
        for day in cfg['collection']['days']:
            folder = root / ('day_%s' % day)
            report = collect(path, sha(path), day, folder)
            assert report['source_stable'] and len(report['records']) == 2
            assert all(r['same_action_parity'] for r in report['records'])
            shards.append({'report': str(folder / 'report.json'),
                'archive': str(folder / 'private.json'), 'report_sha256': sha(folder / 'report.json')})
        merged = merge(shards, root / 'merged')
        assert merged['merged_states'] == 4 and merged['merged_days'] == 2
        assert merged['expected_keys'] == [(0, 0), (0, 4), (2, 0), (2, 4)]
        check_raises(lambda: merge(shards[:1], root / 'missing'))
        check_raises(lambda: merge(shards + shards[:1], root / 'duplicate'))
        bad = deepcopy(cfg)
        bad['qualification'] = 'formal_development'
        check_raises(lambda: validate_config(bad, 0))
        bad = deepcopy(cfg)
        bad['collection']['train_seed'] = 20260927
        check_raises(lambda: validate_config(bad, 0))
        check_raises(lambda: validate_config(cfg, 1))
    print('ALL PASS: real generator global-day teacher shards and complete merge; original branches replay, formal activation and no missing/duplicate/test-day promotion')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--persistent-out')
    test(ap.parse_args().persistent_out)
