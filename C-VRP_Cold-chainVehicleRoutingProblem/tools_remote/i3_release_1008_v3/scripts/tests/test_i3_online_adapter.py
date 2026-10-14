"""Real trained engineering models through four-arm adapter; no paper claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent
for folder in ('tests', 'training', 'evaluation', 'models'):
    sys.path.insert(0, str(ROOT / folder))
from test_train_i3_runner import configuration
from test_i3_counterfactual_teacher import check_raises
from train_i3_scenario import run, sha
from maskco_scenario import MaskCOScenarioModel, save_model
from run_i3_online_compare import main, I3OnlineAdapter


def test(archive_root, output=None):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(output) if output else Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        warm_dir = root / 'warm'
        warm_dir.mkdir()
        model = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(343))
        model.head_count.kernel.value = jnp.zeros_like(model.head_count.kernel.value)
        model.head_count.bias.value = jnp.full_like(model.head_count.bias.value, -10.).at[2].set(10.)
        warm = warm_dir / 'model.bin'
        save_model(model, warm)
        warm_dir.joinpath('config.json').write_text(json.dumps({'qualification': 'engineering',
            'training_complete': True, 'train_seed': 20260925,
            'args': {'arm': 'random', 'dim': 16, 'm_max': 200, 'train_instances': 12},
            'model_source_sha256': {'scripts/models/maskco_scenario.py': sha(ROOT / 'models/maskco_scenario.py')}}),
            encoding='utf-8')
        archive = Path(archive_root)
        cfg = configuration(archive / 'report.json', archive / 'private.json', warm)
        cfg['actor_pool_excluded_days'] = []  # Teacher-objective-only split, full B decoding pool.
        models, reports = {}, {}
        for name, objective in (('maskco', 'decision_reconstruction'),
                                ('reconstruction_control', 'reconstruction_control')):
            copy = deepcopy(cfg)
            copy['settings']['objective'] = objective
            path = root / (name + '_train_config.json')
            path.write_text(json.dumps(copy, indent=2), encoding='utf-8')
            folder = root / name
            reports[name] = run(path, sha(path), folder)
            models[name] = {'path': str((folder / 'model.bin').resolve()),
                'sha256': sha(folder / 'model.bin'), 'card': str((folder / 'config.json').resolve()),
                'card_sha256': sha(folder / 'config.json')}
        assert reports['maskco']['history_pool_lineage']['excluded_days'] == []
        models['warmstart'] = {'path': str(warm.resolve()), 'sha256': sha(warm),
            'card': str((warm_dir / 'config.json').resolve()), 'card_sha256': sha(warm_dir / 'config.json')}
        config = {'schema': 'i3_online_comparison_v1', 'qualification': 'engineering',
            'models': models, 'history_pool_lineage': reports['maskco']['history_pool_lineage'],
            'online_parameters': {'arm': 'random', 'dim': 16, 'm_max': 200, 'train_instances': 12,
                'dev_instances': 1, 'n_orders': 20, 'num_vehicles': 3, 'K': 2,
                'time_limit': 10., 'future_policy': 'density', 'energy_pricing': 'marginal',
                'main_only': True, 'workers': 2}}
        path = root / 'online_config.json'
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        argv = ['--online-config', str(path), '--online-config-sha256', sha(path),
            '--model-bin', models['maskco']['path'], '--arm', 'random', '--dim', '16',
            '--train-instances', '12', '--dev-instances', '1', '--n-orders', '20',
            '--num-vehicles', '3', '--K', '2', '--future-policy', 'density',
            '--energy-pricing', 'marginal', '--main-only', '--workers', '2', '--out', str(root / 'online')]
        report = main(argv)
        tier = report['results']['20260926']['p_c=(5,10,15)']
        assert set(tier['arms']) == {'maskco', 'cond_hist', 'warmstart', 'reconstruction_control'}
        assert all(tier['arms'][name]['n_days'] == 1 for name in tier['arms'])
        assert tier['main_comparison']['formal_adjudication'] is False
        assert not report['model_adapter']['qualification'] == 'formal_development'
        assert 'maskco_minus_reconstruction_control' in tier and 'maskco_minus_warmstart' in tier
        check_raises(lambda: I3OnlineAdapter(path, '0' * 64))
        bad = deepcopy(config)
        bad['models']['warmstart']['sha256'] = '0' * 64
        bad_path = root / 'bad_online.json'
        bad_path.write_text(json.dumps(bad), encoding='utf-8')
        check_raises(lambda: main(['--online-config', str(bad_path), '--online-config-sha256', sha(bad_path)] + argv[4:]))
    print('ALL PASS: real I3/reconstruction-control/warmstart/cond online adapter, full pool deployment lineage, engineering nonformal barrier and weight/config identity guards')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-root', required=True)
    ap.add_argument('--persistent-out')
    args = ap.parse_args()
    test(args.archive_root, args.persistent_out)
