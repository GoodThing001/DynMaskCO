"""Same-protocol online continuation after a pre-data import failure.

Reuses the completed fixed checkpoints, never recollects/retrains or selects
weights. Reads origin artifacts; all new outputs use a separate frozen run.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

PROTOCOL_SHA = '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, v):
    with Path(p).open('x', encoding='utf-8') as stream:
        json.dump(v, stream, indent=2, allow_nan=False)
    return sha(p)


def run(out, project, origin):
    out, project, origin = map(Path, (out, project, origin))
    ext = out / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
    scripts = ext / 'scripts'
    for folder in ('training', 'evaluation', 'models', 'coldchain', 'simulation'):
        sys.path.insert(0, str(scripts / folder))
    from diag_i3_online_adjudicate import adjudicate
    from run_identity import SHARED_SOURCE_FILES, source_identity, dataset_identity, dataset_meta_sha256
    from run_exp_reserve import generate_dataset
    from scenario_saa import make_c0_contract_v2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    import numpy as np

    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('PYTHONPATH', None)
    protocol = out / 'protocol.json'
    assert sha(protocol) == sha(origin / 'protocol.json') == PROTOCOL_SHA
    reg = read(protocol)
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 50:
        raise RuntimeError('Registered CPU allocation unavailable')
    manifest_sha = sha(out / 'SOURCE_MANIFEST.sha256')

    def state(stage, **extra):
        (out / 'pipeline_state.json').write_text(json.dumps({'stage': stage,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), **extra}, indent=2), encoding='utf-8')
        print(stage, flush=True)

    def seal():
        result = subprocess.run(['sha256sum', '-c', str(out / 'SOURCE_MANIFEST.sha256')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode or sha(protocol) != PROTOCOL_SHA or sha(out / 'SOURCE_MANIFEST.sha256') != manifest_sha:
            raise RuntimeError('Recovery source/protocol seal failed')
        result = subprocess.run(['sha256sum', '-c', str(origin / 'SOURCE_MANIFEST.sha256')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Original training source seal changed')

    state('independent_fixed_checkpoint_audit')
    seal()
    teacher_barrier = read(origin / 'TEACHER_AUDIT_ACCEPTED.json')
    if (not teacher_barrier['qualification_verified'] or teacher_barrier['all_registered_states'] != 128
            or sha(teacher_barrier['audit_path']) != teacher_barrier['audit_sha256']):
        raise RuntimeError('No qualified 128-state teacher barrier')
    template = read(origin / 'online_config_20260926.json')
    reports = {arm: read(origin / arm / 'train_report.json')
               for arm in ('maskco', 'reconstruction_control')}
    progress = {}
    for arm, report in reports.items():
        path = origin / arm
        rows = [json.loads(line) for line in (path / 'progress.jsonl').read_text().splitlines()]
        cfg = read(origin / (arm + '_training_config.json'))
        expected_settings = deepcopy(reg['settings'])
        if arm == 'reconstruction_control':
            expected_settings['objective'] = 'reconstruction_control'
        if (cfg['settings'] != expected_settings or report['settings'] != expected_settings
                or report['train_keys'] != cfg['train_keys']
                or len(cfg['train_keys']) != 96
                or sha(cfg['teacher']['path']) != cfg['teacher']['sha256']
                or cfg['train_days'] != reg['train_days'] or cfg['holdout_days'] != reg['holdout_days']):
            raise RuntimeError('Training registration or teacher provenance mismatch')
        if (report['steps_completed'] != 192 or [r['step'] for r in rows] != list(range(1, 193))
                or report['logs'] != rows or report['model_selection'] != 'fixed_last_step'
                or not report['checkpoint_reload_verified'] or not report['source_stable']
                or report['source_start'] != report['source_end']
                or report['teacher_archive_sha256'] != cfg['teacher']['sha256']
                or report['config_sha256'] != sha(origin / (arm + '_training_config.json'))
                or sha(path / 'model.bin') != report['model_bin_sha256']
                or sha(path / 'input_warmstart.bin') != reg['warmstart_sha256']):
            raise RuntimeError('Unqualified fixed training: ' + arm)
        for rel, digest in report['source_start'].items():
            if sha(origin / 'src/C-VRP_Cold-chainVehicleRoutingProblem' / rel) != digest:
                raise RuntimeError('Training provenance differs from actual original source')
        rng = np.random.default_rng(cfg['settings']['schedule_seed'])
        expected = [tuple(cfg['train_keys'][i]) for _ in range(2)
                    for i in rng.permutation(len(cfg['train_keys']))]
        observed = [(row['day_id'], row['ordinal']) for row in rows]
        if observed != expected or len(set(observed)) != 96:
            raise RuntimeError('Fixed 96-state/two-pass schedule mismatch')
        if (rows[0]['trainable_before'] != report['initial_trainable_hash']
                or rows[-1]['trainable_after'] != report['final_trainable_hash']
                or any(a['trainable_after'] != b['trainable_before'] for a, b in zip(rows, rows[1:]))):
            raise RuntimeError('Parameter-update continuity mismatch')
        progress[arm] = rows
    if (reports['maskco']['initial_trainable_hash'] != reports['reconstruction_control']['initial_trainable_hash']
            or [(r['day_id'], r['ordinal']) for r in progress['maskco']] !=
               [(r['day_id'], r['ordinal']) for r in progress['reconstruction_control']]):
        raise RuntimeError('Training arms do not match')

    # Independently load actual checkpoints into fresh model templates and
    # recompute both parameter groups, rather than trusting report flags.
    os.sched_setaffinity(0, set(allowed[48:50]))
    from flax import nnx
    import jax.numpy as jnp
    from maskco_scenario import MaskCOScenarioModel, load_model, check_model_generation
    from i3_optimizer import parameter_hash
    from mpre import load_cvrp_model
    cvrp_path = read(origin / 'maskco_training_config.json')['model']['cvrp_path']
    assert sha(cvrp_path) == reg['cvrp_sha256']
    cvrp, _, _ = load_cvrp_model(cvrp_path)
    loaded = {}
    outputs = {}
    x = jnp.zeros((1, 200), dtype=jnp.int32)
    cb = jnp.zeros((1,), dtype=jnp.int32)
    for arm in ('warmstart', 'maskco', 'reconstruction_control'):
        p = origin / arm
        if check_model_generation(p / 'model.bin') != 'ok':
            raise RuntimeError('Model generation mismatch')
        m = MaskCOScenarioModel(dim=128, arm='pretrained', cvrp_model=cvrp, rngs=nnx.Rngs(0))
        load_model(m, p / 'model.bin')
        loaded[arm] = {'trainable_hash': parameter_hash(m, frozen=False),
            'frozen_hash': parameter_hash(m, frozen=True), 'model_sha256': sha(p / 'model.bin')}
        outputs[arm] = [np.asarray(v) for v in m(x, clock_bin=cb)]
        if not all(np.isfinite(v).all() for v in outputs[arm]):
            raise RuntimeError('Nonfinite fixed synthetic inference')
        if arm != 'warmstart':
            if (loaded[arm]['trainable_hash'] != reports[arm]['final_trainable_hash']
                    or loaded[arm]['frozen_hash'] != reports[arm]['frozen_hash']):
                raise RuntimeError('Independent runtime reload hash mismatch')
        del m
    for arm in ('maskco', 'reconstruction_control'):
        if (loaded[arm]['trainable_hash'] == loaded['warmstart']['trainable_hash']
                or loaded[arm]['frozen_hash'] != loaded['warmstart']['frozen_hash']
                or reports[arm]['initial_trainable_hash'] != loaded['warmstart']['trainable_hash']):
            raise RuntimeError('Frozen branch or actual parameter update failed')
    save(out / 'INDEPENDENT_FIXED_MODEL_ACCEPTANCE.json', {'qualified': True,
        'steps_per_arm': 192, 'same_fixed_schedule': True, 'same_initialization': True,
        'actual_saved_models_independently_reloaded': True, 'actual_cvrp_frozen': True,
        'models': loaded, 'teacher_origin': str(origin), 'protocol_sha256': PROTOCOL_SHA,
        'online_gain_established': False})
    del cvrp, outputs

    # True isolated subprocess regression reproduces the original entry failure.
    # No inherited sys.path and no online utility is observed in this QA.
    state('isolated_online_entry_qa')
    driver = scripts / 'evaluation/run_i3_online_compare.py'
    old_driver = origin / 'src/C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_i3_online_compare.py'
    def import_check(path):
        snippet = "import runpy;runpy.run_path(%r,run_name='isolated_import_qa')" % str(path)
        result = subprocess.run([sys.executable, '-c', snippet], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, cwd=ext)
        return result.returncode, result.stdout.decode('utf-8', 'replace')
    old_rc, old_log = import_check(old_driver)
    new_rc, new_log = import_check(driver)
    if old_rc == 0 or "No module named 'coldchain_contract'" not in old_log or new_rc != 0:
        raise RuntimeError('Isolated-entry regression does not prove the engineering fix: ' + new_log)
    save(out / 'ENTRY_IMPORT_REGRESSION.json', {'old_returncode': old_rc, 'new_returncode': new_rc,
        'old_log': old_log, 'new_log': new_log, 'engineering_only': True})
    # No dataset sampling or policy action in this acceptance check.
    import run_i3_online_compare as entry
    from types import SimpleNamespace
    old_cfg_path = origin / 'online_config_20260926.json'
    adapter = entry.I3OnlineAdapter(old_cfg_path, sha(old_cfg_path))
    args = SimpleNamespace(**template['online_parameters'], only=None, baseline_from=None,
        allow_code_mismatch=False, final_test=False, gate_seed=20260926,
        model_bin=template['models']['maskco']['path'], cvrp_ckpt=cvrp_path)
    adapter.qualify(args)
    os.sched_setaffinity(0, set(allowed))
    save(out / 'QA_PASS.json', {'entry_import_and_actual_model_qualification': True,
        'scientific_parameters_changed': False, 'model_weights_changed': False,
        'online_utility_observed': False})
    seal()

    def online(seed, offset):
        seal()
        cfg = deepcopy(template)
        cfg['frozen_extension'] = str(ext)
        cfg['protocol'] = {'path': str(protocol), 'sha256': PROTOCOL_SHA}
        if seed != 20260926:
            contract = make_c0_contract_v2()
            data = add_v2_initial_quality(generate_dataset(40, 200, seed), contract)
            cfg['expected_identity'] = {**template['expected_identity'],
                'shared_source_sha256': source_identity(SHARED_SOURCE_FILES),
                'data_sha256': dataset_identity(data), 'data_meta_sha256': dataset_meta_sha256(data),
                'seeds': {'train': 20260925, 'gate': seed, 'final_test_reserved': [20260927, 20260928]}}
            cfg['arm_order_seed'] = 101 + seed
            cfg['worker_cpu_sets'] = [allowed[offset+2*i:offset+2*i+2] for i in range(4)]
        cfg_path = out / ('online_config_%s.json' % seed)
        digest = save(cfg_path, cfg)
        target = out / ('online_%s' % seed)
        target.mkdir()
        argv = [sys.executable, str(driver), '--online-config', str(cfg_path),
            '--online-config-sha256', digest, '--model-bin', cfg['models']['maskco']['path'],
            '--cvrp-ckpt', cvrp_path, '--gate-seed', str(seed), '--out', str(target)]
        for key, value in reg['online_parameters'].items():
            flag = '--' + key.replace('_', '-')
            if value is True: argv.append(flag)
            elif value is not False: argv.extend([flag, str(value)])
        with (target / 'run.log').open('x') as log:
            result = subprocess.run(argv, env=env, cwd=ext, stdout=log, stderr=subprocess.STDOUT,
                timeout=reg['resources']['online_timeout_s'])
        (target / 'run.exit').write_text('RUN_EXIT=%s\n' % result.returncode)
        if result.returncode:
            raise RuntimeError('Online child failed; preserve evidence: ' + str(target))
        seal()
        answer = adjudicate(target / 'gate.json', cfg_path, digest, protocol, PROTOCOL_SHA)
        save(target / 'INDEPENDENT_ADJUDICATION.json', answer)
        return answer

    state('online_first_development_seed', seed=20260926)
    first = online(20260926, 24)
    if not first['by_seed']['20260926']['formal_passed']:
        state('registered_primary_not_passed', first=first, no_numerical_retry=True)
        return
    state('online_robustness', seeds=[20260930, 20261001])
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(lambda pair: online(*pair), [(20260930, 32), (20261001, 40)]))
    all_answers = [first] + answers
    robust = all(v['formal_passed'] for answer in all_answers for v in answer['by_seed'].values())
    save(out / 'I3_DEVELOPMENT_SUMMARY.json', {'robust_primary_passed': robust,
        'answers': all_answers, 'final_test_run': False, 'iteration_gain_established': False,
        'strong_dynamic_comparison_complete': False})
    state('registered_development_complete', robust_primary_passed=robust)


if __name__ == '__main__':
    try:
        run(*sys.argv[1:4])
    except Exception as exc:
        save(Path(sys.argv[1]) / 'pipeline_failure.json', {'error_type': type(exc).__name__,
            'error': str(exc), 'scientific_parameters_changed': False, 'final_test_run': False})
        raise
