"""Sealed-config I3 training runner; engineering runs never become formal data.

Config and archive must carry external SHA identities. Fixed last-step model
selection only. Formal mode additionally requires qualified A3 failure, a
registered matching protocol and a formal 200-order teacher collection.
GPU/CPU environment is selected before importing JAX or model libraries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sealed(path, digest):
    if sha(path) != digest:
        raise ValueError('External file identity mismatch: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_config(config):
    if (config['schema'] != 'i3_fixed_training_config_v1' or
            config['qualification'] not in ('engineering', 'formal_development') or
            config['train_seed'] != 20260925 or
            config['model_selection'] != 'fixed_last_step'):
        raise ValueError('Unsupported training qualification/seed/selection')
    train, holdout = config['train_days'], config['holdout_days']
    if (not train or len(set(train)) != len(train) or len(set(holdout)) != len(holdout) or
            set(train) & set(holdout) or any(type(d) is not int or d < 0 for d in train + holdout)):
        raise ValueError('Training/holdout days must be explicit, unique and disjoint')
    keys = [tuple(k) for k in config['train_keys']]
    expected = [tuple(k) for k in config['teacher']['expected_keys'] if k[0] in train]
    if len(set(keys)) != len(keys) or sorted(keys) != sorted(expected):
        raise ValueError('Use all predeclared training states; no eligibility/sign trimming')
    ctx = config['teacher']['context']
    if ctx['train_seed'] != config['train_seed']:
        raise ValueError('Teacher training seed mismatch')
    captured_days = {k[0] for k in config['teacher']['expected_keys']}
    if not set(train + holdout) <= captured_days:
        raise ValueError('Declared split days are absent from the teacher archive')
    if config['split_scope'] not in ('teacher_objective_only', 'whole_day_holdout'):
        raise ValueError('Declare what the internal holdout excludes')
    if config['split_scope'] == 'whole_day_holdout':
        if set(holdout) & set(config['model']['warmstart_training_days']):
            raise ValueError('Whole-day holdout was used by warmstart')
        if not set(holdout) <= set(ctx['collection_config'].get('pool_excluded_days', [])):
            raise ValueError('Teacher continuation pool includes claimed whole-day holdout')
    if config['qualification'] == 'formal_development':
        if 'actor_pool_excluded_days' not in config:
            raise ValueError('Formal training must explicitly register its decoding pool')
        cc = ctx['collection_config']
        if (cc.get('not_formal_training_config') is not False or
                cc['orders'] != 200 or cc['history_days'] != 200 or not holdout):
            raise ValueError('Engineering/undersized teacher archives cannot be promoted')
        activation = read_sealed(**config['activation_a3'])
        if (activation.get('formal_adjudicable') is not True or activation.get('partial') or
                activation.get('n_batches') != 9):
            raise ValueError('Formal I3 requires a complete qualified original A3 answer')
        b = activation['summary_by_arm_seed']['B']
        if any(not b[str(s)]['main_tier']['formal_adjudication'] or
               b[str(s)]['main_tier']['numeric_passed'] for s in (20260926, 20260930, 20261001)):
            raise ValueError('Registered I3 failure branch requires all B main gates to fail')
        protocol = read_sealed(**config['protocol'])
        expected_settings = dict(protocol['settings'])
        if config['settings']['objective'] == 'reconstruction_control':
            if 'reconstruction_control' not in protocol.get('controls', []):
                raise ValueError('Reconstruction control was not registered')
            expected_settings['objective'] = 'reconstruction_control'
        if (protocol.get('predeclared') is not True or
                protocol.get('main_comparison') != 'I3_minus_cond_hist' or
                protocol.get('delta_3') != 20 or protocol.get('online_days') != 40 or
                expected_settings != config['settings'] or
                protocol.get('teacher_collection') != cc or
                protocol.get('actor_pool_excluded_days') != config['actor_pool_excluded_days'] or
                protocol.get('warmstart_sha256') != config['model']['warmstart_sha256']):
            raise ValueError('Formal configuration differs from registered I3 protocol')


def _run(config_path, config_sha256, out_path):
    config = read_sealed(config_path, config_sha256)
    validate_config(config)
    runtime = config['runtime']
    if runtime['device'] == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif runtime['device'] == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(runtime['gpu_id'])
    else:
        raise ValueError('Choose CPU or GPU before importing JAX')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if runtime.get('one_cpu_affinity'):
        if hasattr(os, 'sched_getaffinity'):
            allowed = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {min(allowed)})
        os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for folder in ('', 'models', 'training', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(ROOT / folder))
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from i3_teacher_archive import load_archive
    from i3_training_engine import TrainingSettings, train_fixed_schedule, reconstruction_example
    from i3_optimizer import parameter_hash
    from i3_pool_lineage import pool_lineage
    from i3_counterfactual_teacher import ReplaySafeSaaReplanner
    from maskco_scenario import MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools, save_model, load_model
    from scenario_saa import build_history, CondHistoricalSampler, make_c0_contract_v2
    from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
    from run_identity import dataset_identity, dataset_meta_sha256

    settings = TrainingSettings(**config['settings'])
    settings.validate()
    for rel, digest in config['source_sha256'].items():
        path = EXT / rel
        if not path.resolve().is_relative_to(EXT.resolve()) or sha(path) != digest:
            raise ValueError('Training source identity mismatch: ' + rel)
    required = {'scripts/training/train_i3_scenario.py', 'scripts/training/i3_training_engine.py',
                'scripts/training/i3_optimizer.py', 'scripts/training/i3_decision_objective.py',
                'scripts/training/i3_actor_draws.py', 'scripts/models/maskco_scenario.py'}
    required.add('scripts/training/i3_pool_lineage.py')
    if not required <= set(config['source_sha256']):
        raise ValueError('Training/source closure is incomplete')
    teacher = config['teacher']
    ctx = teacher['context']
    for rel, digest in ctx['source_sha256'].items():
        path = ROOT / rel
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha(path) != digest:
            raise ValueError('Teacher/SAA generation source mismatch: ' + rel)
    archive = load_archive(teacher['path'], archive_sha256=teacher['sha256'],
        expected_context=ctx, expected_keys=teacher['expected_keys'])
    cc = ctx['collection_config']
    history_data = generate_dataset(cc['history_days'], cc['orders'], config['train_seed'])
    if (dataset_identity(history_data) != ctx['history_dataset_sha256'] or
            dataset_meta_sha256(history_data) != ctx['history_dataset_meta_sha256']):
        raise ValueError('Historical sampling pool identity mismatch')
    history = build_history(history_data)
    contract = make_c0_contract_v2()
    if contract.contract_hash != ctx['contract_hash']:
        raise ValueError('Teacher/training contract mismatch')
    spec = config['model']
    if sha(spec['warmstart_path']) != spec['warmstart_sha256']:
        raise ValueError('Warmstart checkpoint identity mismatch')
    if config['qualification'] == 'formal_development':
        warm_card = read_sealed(spec['warmstart_card'], spec['warmstart_card_sha256'])
        warm_args = warm_card['args']
        if (warm_card['train_seed'] != config['train_seed'] or
                warm_args['arm'] != spec['arm'] or warm_args['dim'] != spec['dim'] or
                warm_args['train_instances'] != cc['history_days'] or warm_args['n_orders'] != cc['orders']):
            raise ValueError('Warmstart lineage differs from registered architecture/data')
        from maskco_scenario import check_model_generation
        if check_model_generation(spec['warmstart_path']) != 'ok':
            raise ValueError('Unknown warmstart source generation')
    cvrp = None
    if spec['arm'] in ('pretrained', 'random_cvrp'):
        if sha(spec['cvrp_path']) != spec['cvrp_sha256']:
            raise ValueError('CVRP checkpoint identity mismatch')
        from mpre import load_cvrp_model
        cvrp, cvrp_config, _ = load_cvrp_model(spec['cvrp_path'])
        if spec['arm'] == 'random_cvrp':
            from dataclasses import replace
            cvrp = replace(cvrp_config, rngs=spec['initialization_seed']).construct_model()
    if spec['arm'] not in ('pretrained', 'random_cvrp', 'random'):
        raise ValueError('Unsupported architecture')
    model = MaskCOScenarioModel(dim=spec['dim'], arm=spec['arm'], cvrp_model=cvrp,
                                rngs=nnx.Rngs(spec['initialization_seed']))
    load_model(model, spec['warmstart_path'])
    # Legacy engineering configs excluded holdout by default. New formal
    # teacher-objective-only splits may retain all historical information;
    # whole-day holdout must exclude those days everywhere, including warmstart.
    excluded = config.get('actor_pool_excluded_days', config['holdout_days'])
    if config['split_scope'] == 'whole_day_holdout' and not set(config['holdout_days']) <= set(excluded):
        raise ValueError('Whole-day holdout appears in actor decoding pool')
    decoding_pool = pool_lineage(history_data, train_seed=config['train_seed'], excluded_days=excluded)
    pools = HistoryPools([history[i] for i in decoding_pool['included_days']])
    sampler = MaskCOScenarioSampler(model, pools)
    keys = [tuple(k) for k in config['train_keys']]
    entries_by_key = {(e['day_id'], e['ordinal']): e for e in archive['entries']}
    entries = [entries_by_key[k] for k in keys]
    private_history = {i: day for i, day in enumerate(build_history(archive['dataset']))}

    def factory_for_day(day):
        excluded = set(cc.get('pool_excluded_days', [])) | {day}
        pool = [h for i, h in enumerate(history) if i not in excluded]
        return lambda: ReplaySafeSaaReplanner(budget=cc['budget'], capacity=50,
            booking_horizon=BOOKING_HORIZON, contract=contract,
            cooling_share=cc.get('cooling_share', 1), sampler=CondHistoricalSampler(pool),
            reject_penalty={0: 5, 1: 10, 2: 15}, K=cc['K'], time_limit=float('inf'),
            arm_seed=7001, shadow_mode='greedy', energy_pricing='marginal', future_policy='density')

    source_start = {p.relative_to(EXT).as_posix(): sha(p) for p in ROOT.rglob('*.py')}
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'submitted_config.json').write_bytes(Path(config_path).read_bytes())
    (out / 'input_warmstart.bin').write_bytes(Path(spec['warmstart_path']).read_bytes())
    status_path = out / 'status.json'
    status_path.write_text(json.dumps({'stage': 'running', 'qualification': config['qualification'],
                                      'training_complete': False}), encoding='utf-8')
    provisional = {'qualification': config['qualification'], 'training_complete': False,
        'args': {'arm': spec['arm'], 'dim': spec['dim'], 'm_max': 200,
                 'train_instances': cc['history_days']}, 'train_seed': config['train_seed'],
        'model_source_sha256': config['source_sha256']}
    (out / 'config.json').write_text(json.dumps(provisional, indent=2), encoding='utf-8')
    with (out / 'progress.jsonl').open('x', encoding='utf-8') as progress:
        def on_step(row):
            progress.write(json.dumps(row, allow_nan=False) + '\n')
            progress.flush()
        result = train_fixed_schedule(sampler, entries, archive['dataset'], private_history,
            factory_for_day, contract, BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, settings,
            minimum_teacher_gap=cc['minimum_gap'], on_step=on_step)
    model_path = out / 'model.bin'
    save_model(model, model_path)
    restored = MaskCOScenarioModel(dim=spec['dim'], arm=spec['arm'], cvrp_model=cvrp,
                                   rngs=nnx.Rngs(0))
    load_model(restored, model_path)
    if (parameter_hash(restored, frozen=False) != result['final_trainable_hash'] or
            parameter_hash(restored, frozen=True) != result['frozen_hash'] or
            result['initial_trainable_hash'] == result['final_trainable_hash']):
        raise RuntimeError('Updated model checkpoint reload failed')
    recon = reconstruction_example(sampler, entries[0]['record'].visible,
        private_history[entries[0]['day_id']], np.random.default_rng(settings.label_seed))
    inputs = jnp.asarray(recon['recon_x']), jnp.asarray(recon['recon_cb'])
    for a, b in zip(model(inputs[0], clock_bin=inputs[1]), restored(inputs[0], clock_bin=inputs[1])):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    source_end = {p.relative_to(EXT).as_posix(): sha(p) for p in ROOT.rglob('*.py')}
    if source_start != source_end:
        raise RuntimeError('Training source changed during run')
    result.update(config_sha256=config_sha256, qualification=config['qualification'],
        source_start=source_start, source_end=source_end, source_stable=True,
        model_bin_sha256=sha(model_path), checkpoint_reload_verified=True,
        teacher_archive_sha256=teacher['sha256'], split_scope=config['split_scope'],
        history_pool_lineage=decoding_pool,
        holdout_days=config['holdout_days'], model_selection='fixed_last_step')
    (out / 'train_report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    model_sources = {rel: source_start[rel] for rel in required}
    (out / 'config.json').write_text(json.dumps({'qualification': config['qualification'],
        'training_complete': True, 'args': {'arm': spec['arm'], 'dim': spec['dim'],
        'm_max': 200, 'train_instances': cc['history_days']}, 'train_seed': config['train_seed'],
        'model_source_sha256': model_sources, 'i3_training': result,
        'history_pool_lineage': decoding_pool,
        'warmstart_sha256': spec['warmstart_sha256']}, indent=2), encoding='utf-8')
    status_path.write_text(json.dumps({'stage': 'finished', 'qualification': config['qualification'],
        'training_complete': True, 'source_stable': True,
        'checkpoint_reload_verified': True}), encoding='utf-8')
    print('TRAINING_COMPLETED qualification=%s steps=%s reload_verified=1 online_gain=0' %
          (config['qualification'], result['steps_completed']))
    return result


def run(config_path, config_sha256, out_path):
    out = Path(out_path)
    if out.exists():
        raise FileExistsError('Keep earlier attempts; use a new output directory')
    try:
        return _run(config_path, config_sha256, out_path)
    except Exception as exc:
        if out.exists():
            (out / 'status.json').write_text(json.dumps({'stage': 'failed',
                'training_complete': False, 'error_type': type(exc).__name__,
                'error': str(exc)}), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.config, args.config_sha256, args.out)
