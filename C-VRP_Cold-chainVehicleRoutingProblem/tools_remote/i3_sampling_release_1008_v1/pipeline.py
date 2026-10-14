"""Finite registered experiment: QA + A3 activation -> teachers -> two fixed models -> dev gate.

Never runs reserved final seeds, alters parameters, retries failed numerical
experiments, writes working source, or uses a GPU. Every child has a timeout.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from i3_collection_cpu_leases_1008 import collect_shards


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return sha(path)


def run(out, project, ext):
    out, project, ext = map(Path, (out, project, ext))
    python = sys.executable
    protocol_path = out / 'protocol.json'
    reg, protocol_sha = read(protocol_path), sha(protocol_path)
    source_manifest_sha = sha(out / 'SOURCE_MANIFEST.sha256')
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 48:
        raise RuntimeError('Registered CPU allocations unavailable; do not silently change scheduling')
    scripts = ext / 'scripts'
    for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(scripts / folder))
    from run_identity import SHARED_SOURCE_FILES, dataset_identity, dataset_meta_sha256, source_identity
    from run_exp_reserve import generate_dataset
    from scenario_saa import make_c0_contract_v2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from i3_pool_lineage import pool_lineage
    from diag_i3_online_adjudicate import adjudicate

    def state(stage, **fields):
        out.joinpath('pipeline_state.json').write_text(json.dumps({'stage': stage,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), **fields}, indent=2), encoding='utf-8')
        print(stage, flush=True)

    def seal():
        result = subprocess.run(['sha256sum', '-c', str(out / 'SOURCE_MANIFEST.sha256')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode or sha(protocol_path) != protocol_sha or sha(out / 'SOURCE_MANIFEST.sha256') != source_manifest_sha:
            raise RuntimeError('Frozen source or protocol changed')

    def child(script, argv, folder, timeout, cpus=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        command = [python, str(scripts / script)] + list(map(str, argv))
        if cpus:
            command = ['taskset', '-c', ','.join(map(str, cpus))] + command
        with folder.joinpath('run.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, cwd=ext)
        folder.joinpath('run.exit').write_text('RUN_EXIT=%s\n' % result.returncode, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Child failed; preserve log and stop: ' + str(folder))
        seal()

    state('waiting_release_qa')
    deadline = time.monotonic() + 7200
    while not out.joinpath('qa.exit').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Release QA did not complete within its fixed deadline')
        time.sleep(30)
    exits = out.joinpath('qa.exit').read_text().splitlines()
    if not {'RUN_EXIT=0', 'SEAL_EXIT=0'} <= set(exits) or not read(out / 'QA_PASS.json')['engineering_release_pass']:
        raise RuntimeError('Release QA failed; formal experiment not activated')
    seal()
    state('waiting_qualified_a3_failure')
    activation = project / 'results/a3_repair_chain_1007_2250/A3_SAME_SOURCE_ADJUDICATION.json'
    deadline = time.monotonic() + 86400
    while not activation.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('A3 activation missing after 24h; no formal collection')
        time.sleep(30)
    answer = read(activation)
    if (not answer['formal_adjudicable'] or answer['partial'] or answer['n_batches'] != 9 or
            any(not answer['summary_by_arm_seed']['B'][str(seed)]['main_tier']['formal_adjudication'] or
                answer['summary_by_arm_seed']['B'][str(seed)]['main_tier']['numeric_passed']
                for seed in reg['development_seeds'])):
        state('activation_not_met', explanation='Do not trigger I3 from unqualified/positive original A3')
        return
    saved_activation = out / 'A3_ACTIVATION.json'
    write(saved_activation, answer)
    files = set(SHARED_SOURCE_FILES) | {'scripts/training/collect_i3_teachers.py',
        'scripts/training/i3_counterfactual_teacher.py', 'scripts/training/i3_teacher_archive.py',
        'scripts/training/i3_capture_schedule.py',
        'scripts/simulation/recourse_snapshot.py', 'scripts/evaluation/run_exp_encoder_v3.py',
        'scripts/evaluation/run_identity.py'}
    cfg = {'schema': 'i3_teacher_collection_v1', 'qualification': 'formal_development',
        'collection': reg['teacher_collection'], 'source_sha256': {r: sha(ext / r) for r in files},
        'activation_a3': {'path': str(saved_activation), 'digest': sha(saved_activation)},
        'protocol': {'path': str(protocol_path), 'digest': protocol_sha}}
    cfg_path = out / 'teacher_collection_config.json'
    cfg_sha = write(cfg_path, cfg)
    state('collecting_registered_teachers', days=len(cfg['collection']['days']), workers=8)

    def collect(day, cpu):
        target = out / ('teacher_day_%03d' % day)
        log = out / ('teacher_job_%03d' % day)
        child('training/collect_i3_teachers.py', ['--config', cfg_path, '--config-sha256', cfg_sha,
            '--day-id', day, '--out', target], log, reg['resources']['teacher_day_timeout_s'],
            cpus=[cpu])
        return {'report': str(target / 'report.json'), 'archive': str(target / 'private.json'),
                'report_sha256': sha(target / 'report.json')}

    def collection_failure(day, exc):
        write(out / 'collection_failure.json', {'day_id': day, 'error_type': type(exc).__name__,
            'error': str(exc), 'scientific_parameters_changed': False, 'online_gain_established': False})
    shards = collect_shards(cfg['collection']['days'], allowed[8:16], collect, collection_failure)
    merge_cfg = out / 'merge_input.json'
    merge_sha = write(merge_cfg, {'shards': shards})
    child('training/merge_i3_teachers.py', ['--input-config', merge_cfg, '--input-sha256', merge_sha,
        '--out', out / 'teacher_merged'], out / 'merge_job', 900)
    merged = read(out / 'teacher_merged/report.json')
    # Read-only independent audit is a qualification barrier, not a label or
    # reward-selection gate. Every registered state remains in the experiment.
    audit_out = project / 'results/i3_teacher_data_audit_1008_v3'
    audit_path = audit_out / 'TEACHER_DATA_AUDIT.json'
    state('waiting_independent_teacher_audit', states=128)
    audit_deadline = time.monotonic() + 1800
    while not audit_path.exists():
        if (audit_out / 'failure.json').exists():
            raise RuntimeError('Independent teacher qualification audit failed: ' + str(read(audit_out / 'failure.json')))
        if time.monotonic() > audit_deadline:
            raise RuntimeError('Independent teacher audit unavailable after fixed 30min qualification wait')
        time.sleep(10)
    audit = read(audit_path)
    expected_inputs = {(row['report_sha256'], sha(row['archive'])) for row in shards}
    observed_inputs = {(row['report_sha256'], row['archive_sha256']) for row in audit['inputs']}
    if (audit['main_output'] != str(out) or audit['protocol_sha256'] != protocol_sha or
            audit['audit_source_sha256'] != sha(out / 'teacher_data_audit.py') or
            not audit['terminal_trace_reevaluation_complete'] or
            (audit['summary']['states'], audit['summary']['days']) != (128, 32) or
            audit['split']['training']['states'] != 96 or
            audit['split']['teacher_objective_holdout']['states'] != 32 or
            len(audit['inputs']) != 32 or expected_inputs != observed_inputs or
            sorted((row['day_id'], row['ordinal']) for row in audit['rows']) !=
            sorted(tuple(key) for key in merged['expected_keys'])):
        raise RuntimeError('Independent teacher audit does not match every sealed registered shard')
    seal()
    write(out / 'TEACHER_AUDIT_ACCEPTED.json', {'audit_path': str(audit_path),
        'audit_sha256': sha(audit_path), 'all_registered_states': 128, 'qualification_verified': True})
    settings = reg['settings']
    required = ['train_i3_scenario.py', 'i3_training_engine.py', 'i3_optimizer.py',
                'i3_decision_objective.py', 'i3_actor_draws.py', 'i3_pool_lineage.py']
    training_sources = {'scripts/training/' + f: sha(scripts / 'training' / f) for f in required}
    training_sources['scripts/models/maskco_scenario.py'] = sha(scripts / 'models/maskco_scenario.py')
    model = {'arm': 'pretrained', 'dim': 128, 'initialization_seed': 42,
        'warmstart_path': str(out / 'warmstart/model.bin'), 'warmstart_sha256': reg['warmstart_sha256'],
        'warmstart_card': str(out / 'warmstart/config.json'), 'warmstart_card_sha256': sha(out / 'warmstart/config.json'),
        'warmstart_training_days': list(range(200)),
        'cvrp_path': str(project.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'),
        'cvrp_sha256': reg['cvrp_sha256']}
    base_cfg = {'schema': 'i3_fixed_training_config_v1', 'qualification': 'formal_development',
        'train_seed': 20260925, 'model_selection': 'fixed_last_step',
        'train_days': reg['train_days'], 'holdout_days': reg['holdout_days'],
        'split_scope': reg['split_scope'], 'actor_pool_excluded_days': reg['actor_pool_excluded_days'],
        'train_keys': [key for key in merged['expected_keys'] if key[0] in reg['train_days']],
        'teacher': {'path': str(out / 'teacher_merged/private.json'), 'sha256': merged['archive_sha256'],
                    'context': merged['context'], 'expected_keys': merged['expected_keys']},
        'runtime': {'device': 'cpu', 'one_cpu_affinity': False}, 'model': model,
        'settings': settings, 'source_sha256': training_sources,
        'activation_a3': {'path': str(saved_activation), 'digest': sha(saved_activation)},
        'protocol': {'path': str(protocol_path), 'digest': protocol_sha}}
    state('training_fixed_models', steps=settings['steps'], selection='fixed_last_step')

    def train(name):
        cfg = deepcopy(base_cfg)
        if name == 'reconstruction_control':
            cfg['settings']['objective'] = 'reconstruction_control'
        path = out / (name + '_training_config.json')
        digest = write(path, cfg)
        child('training/train_i3_scenario.py', ['--config', path, '--config-sha256', digest,
            '--out', out / name], out / (name + '_training_job'), reg['resources']['training_timeout_s'],
            allowed[16:20] if name == 'maskco' else allowed[20:24])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(train, ['maskco', 'reconstruction_control']))
    history_data = generate_dataset(200, 200, 20260925)
    decoding = pool_lineage(history_data, train_seed=20260925, excluded_days=[])
    contract = make_c0_contract_v2()
    models = {}
    for name in ('maskco', 'reconstruction_control', 'warmstart'):
        folder = out / name
        models[name] = {'path': str(folder / 'model.bin'), 'sha256': sha(folder / 'model.bin'),
            'card': str(folder / 'config.json'), 'card_sha256': sha(folder / 'config.json')}

    def online(seed, offset):
        data = add_v2_initial_quality(generate_dataset(40, 200, seed), contract)
        expected = {'shared_source_sha256': source_identity(SHARED_SOURCE_FILES),
            'contract_sha256': contract.contract_hash, 'data_sha256': dataset_identity(data),
            'data_meta_sha256': dataset_meta_sha256(data), 'budget_B': reg['teacher_collection']['budget'],
            'cooling_share': reg['teacher_collection']['cooling_share'],
            'seeds': {'train': 20260925, 'gate': seed, 'final_test_reserved': [20260927, 20260928]}}
        cfg = {'schema': 'i3_online_comparison_v1', 'qualification': 'formal_development',
            'models': models, 'history_pool_lineage': decoding, 'cvrp_sha256': reg['cvrp_sha256'],
            'online_parameters': reg['online_parameters'], 'arm_order_seed': 101 + seed,
            'worker_cpu_sets': [allowed[offset + 2*i:offset + 2*i + 2] for i in range(4)],
            'protocol': {'path': str(protocol_path), 'sha256': protocol_sha},
            'frozen_extension': str(ext), 'expected_identity': expected}
        path = out / ('online_config_%s.json' % seed)
        digest = write(path, cfg)
        args = ['--online-config', path, '--online-config-sha256', digest,
                '--model-bin', models['maskco']['path'], '--cvrp-ckpt', model['cvrp_path'],
                '--gate-seed', seed, '--out', out / ('online_%s' % seed)]
        for key, value in reg['online_parameters'].items():
            flag = '--' + key.replace('_', '-')
            if value is True:
                args.append(flag)
            elif value is not False:
                args.extend([flag, value])
        target = out / ('online_%s' % seed)
        child('evaluation/run_i3_online_compare.py', args, target, reg['resources']['online_timeout_s'])
        answer = adjudicate(target / 'gate.json', path, digest, protocol_path, protocol_sha)
        write(target / 'INDEPENDENT_ADJUDICATION.json', answer)
        return answer

    state('online_first_development_seed', seed=20260926)
    first = online(20260926, 24)
    if not first['by_seed']['20260926']['formal_passed']:
        state('registered_primary_not_passed', first=first, next='New mechanism protocol; do not tune this model from dev')
        return
    state('online_robustness', seeds=[20260930, 20261001])
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(lambda pair: online(*pair), [(20260930, 32), (20261001, 40)]))
    all_answers = [first] + answers
    robust = all(v['formal_passed'] for answer in all_answers for v in answer['by_seed'].values())
    write(out / 'I3_DEVELOPMENT_SUMMARY.json', {'robust_primary_passed': robust,
        'answers': all_answers, 'final_test_run': False, 'iteration_gain_established': False,
        'strong_dynamic_comparison_complete': False})
    state('registered_development_complete', robust_primary_passed=robust,
          next='Same-weight iteration and effective dynamic controls; final seeds remain reserved')


if __name__ == '__main__':
    try:
        run(*sys.argv[1:4])
    except Exception as exc:
        target = Path(sys.argv[1]) / 'pipeline_failure.json'
        target.write_text(json.dumps({'error_type': type(exc).__name__, 'error': str(exc),
            'parameters_changed': False, 'final_test_run': False}, indent=2), encoding='utf-8')
        raise
