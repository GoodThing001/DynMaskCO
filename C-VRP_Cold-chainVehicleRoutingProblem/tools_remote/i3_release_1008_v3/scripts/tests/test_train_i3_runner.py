"""Real small archive CLI-path smoke plus fail-closed formal/identity guards.

Test settings/random warmstart are engineering only, not formal I3 choices.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent
for folder in ('tests', 'training', 'models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from train_i3_scenario import run, validate_config
from i3_training_engine import TrainingSettings
from maskco_scenario import MaskCOScenarioModel, save_model
from test_i3_counterfactual_teacher import check_raises


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(report_path, archive_path, warmstart):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    cc = report['config']
    ctx = {'train_seed': 20260925, 'dataset_sha256': report['dataset_sha256'],
        'dataset_meta_sha256': report['dataset_meta_sha256'],
        'contract_hash': report['contract_hash'], 'source_sha256': report['source_sha256'],
        'generator_config': {'history_days': cc['history_days'], 'orders': cc['orders'],
                             'seed': 20260925, 'probe_days': cc['probe_days'], 'initial_quality': 'v2'},
        'collection_config': cc, 'history_dataset_sha256': report['history_dataset_sha256'],
        'history_dataset_meta_sha256': report['history_dataset_meta_sha256']}
    sources = ('train_i3_scenario.py', 'i3_training_engine.py', 'i3_optimizer.py',
               'i3_decision_objective.py', 'i3_actor_draws.py', 'i3_pool_lineage.py')
    hashes = {'scripts/training/' + name: digest(ROOT / 'training' / name) for name in sources}
    hashes['scripts/models/maskco_scenario.py'] = digest(ROOT / 'models/maskco_scenario.py')
    return {'schema': 'i3_fixed_training_config_v1', 'qualification': 'engineering',
        'train_seed': 20260925, 'model_selection': 'fixed_last_step',
        'train_days': [0], 'holdout_days': [1, 2], 'split_scope': 'teacher_objective_only',
        'train_keys': [[0, ordinal] for ordinal in cc['capture_ordinals']],
        'teacher': {'path': str(Path(archive_path).resolve()),
            'sha256': report['private_archive']['sha256'], 'context': ctx,
            'expected_keys': [[day, ordinal] for day in range(cc['probe_days'])
                              for ordinal in cc['capture_ordinals']]},
        'runtime': {'device': 'cpu', 'one_cpu_affinity': False},
        'model': {'arm': 'random', 'dim': 16, 'initialization_seed': 343,
            'warmstart_path': str(Path(warmstart).resolve()), 'warmstart_sha256': digest(warmstart),
            'warmstart_training_days': []},
        'settings': {'steps': 2, 'K': 2, 'independent_sets': 2, 'learning_rate': 1e-5,
            'weight_decay': 0., 'max_grad_norm': 1., 'reconstruction_weight': .1,
            'tau': 20., 'weight_cap': 100., 'baseline_decay': .9,
            'schedule_seed': 23, 'actor_seed': 507, 'label_seed': 71,
            'objective': 'decision_reconstruction'}, 'source_sha256': hashes}


def test_runner(report_path, archive_path, persistent_out=None):
    with tempfile.TemporaryDirectory() as temp:
        model = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(343))
        model.head_count.kernel.value = jnp.zeros_like(model.head_count.kernel.value)
        model.head_count.bias.value = jnp.full_like(model.head_count.bias.value, -10.).at[2].set(10.)
        warm = Path(temp) / 'warm.bin'
        save_model(model, warm)
        cfg = configuration(report_path, archive_path, warm)
        cfg_path = Path(temp) / 'config.json'
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        sha = digest(cfg_path)
        out = Path(persistent_out) if persistent_out else Path(temp) / 'trained'
        result = run(cfg_path, sha, out)
        assert result['qualification'] == 'engineering' and result['steps_completed'] == 2
        assert result['checkpoint_reload_verified'] and result['source_stable']
        assert result['initial_trainable_hash'] != result['final_trainable_hash']
        assert not result['online_gain_established']
        assert not set(r['day_id'] for r in result['logs']) & set(cfg['holdout_days'])
        assert result['history_pool_lineage']['excluded_days'] == cfg['holdout_days']
        check_raises(lambda: run(cfg_path, '0' * 64, Path(temp) / 'wrong_sha'))
        bad = deepcopy(cfg)
        bad['qualification'] = 'formal_development'
        check_raises(lambda: validate_config(bad))
        bad = deepcopy(cfg)
        bad['train_keys'].pop()
        check_raises(lambda: validate_config(bad))
        bad = deepcopy(cfg)
        bad['holdout_days'] = [0]
        check_raises(lambda: validate_config(bad))
        bad = deepcopy(cfg)
        bad['split_scope'] = 'whole_day_holdout'
        bad['model']['warmstart_training_days'] = [1]
        check_raises(lambda: validate_config(bad))
        invalid = deepcopy(cfg['settings'])
        invalid['tau'] = 0
        check_raises(lambda: TrainingSettings(**invalid).validate())
    print('ALL PASS: sealed-config/archive real-generator training runner, fixed model selection, reload/source proof, holdout state exclusion, formal-engineering barrier, omitted states/split/config guards')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--persistent-out')
    args = parser.parse_args()
    test_runner(args.report, args.archive, args.persistent_out)
