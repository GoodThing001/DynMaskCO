"""Finite supplemental diagnosis after immutable fixed-step training completes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return sha(path)


def run(out, project):
    out, project = Path(out), Path(project)
    main = project / 'results/i3_registered_1008_v5'
    ext = main / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
    addendum = read(out / 'addendum.json')
    if sha(main / 'protocol.json') != addendum['main_protocol_sha256']:
        raise ValueError('Supplement does not refer to the registered main experiment')
    code, config_identity = sha(out / 'diagnostic.py'), sha(out / 'addendum.json')
    cpu_set = sorted(os.sched_getaffinity(0))[48:52]
    if len(cpu_set) != addendum['logical_cpu_count']:
        raise ValueError('Separate diagnostic CPUs unavailable')
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}

    def state(stage, **fields):
        out.joinpath('state.json').write_text(json.dumps({'stage': stage,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), **fields}, indent=2), encoding='utf-8')
        print(stage, flush=True)

    def sealed():
        if sha(out / 'diagnostic.py') != code or sha(out / 'addendum.json') != config_identity:
            raise ValueError('Supplementary code/addendum changed')
        result = subprocess.run(['sha256sum', '-c', str(main / 'SOURCE_MANIFEST.sha256')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError('Main frozen experiment changed; do not diagnose mixed sources')

    def evaluate(config, config_name, target, timeout):
        path = out / config_name
        digest = write(path, config)
        with out.joinpath(config_name + '.log').open('x') as log:
            result = subprocess.run([sys.executable, str(out / 'diagnostic.py'), '--config', str(path),
                '--config-sha256', digest, '--out', str(target)], env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        out.joinpath(config_name + '.exit').write_text('RUN_EXIT=%s\n' % result.returncode)
        if result.returncode:
            raise RuntimeError('Diagnostic failed; preserve attempt: ' + str(path))
        sealed()
        return read(target / 'report.json')

    def binding(folder):
        return {'path': str(folder / 'model.bin'), 'sha256': sha(folder / 'model.bin'),
                'card': str(folder / 'config.json'), 'card_sha256': sha(folder / 'config.json')}

    sealed()
    state('waiting_release_qa')
    qa_deadline = time.monotonic() + 7200
    while not (main / 'qa.exit').exists():
        if time.monotonic() > qa_deadline:
            raise RuntimeError('Recovery QA unavailable after fixed 2h wait')
        time.sleep(30)
    if not {'RUN_EXIT=0', 'SEAL_EXIT=0'} <= set((main / 'qa.exit').read_text().splitlines()):
        raise RuntimeError('Recovery QA failed; no teacher risk evaluation')
    state('engineering_real_teacher_smoke')
    qa_cfg = main / 'engineering_actual_teacher_config.json'
    # Predeclared engineering check only: same three existing real teacher states.
    smoke = {'qualification': 'engineering', 'training_config': {'path': str(qa_cfg), 'sha256': sha(qa_cfg)},
        'frozen_extension': str(ext), 'evaluation_keys': [[0,24], [0,99], [0,174]],
        'models': {'maskco': binding(main / 'engineering_actual_teacher_training'),
                   'warmstart': binding(main / 'warmstart')},
        'cpu_set': cpu_set, 'independent_sets': 2, 'K': 2, 'sampling_seed': 907,
        'bootstrap_samples': 2000, 'bootstrap_seed': 1777}
    report = evaluate(smoke, 'engineering_config.json', out / 'engineering', 1800)
    if report['states'] != 3 or report['days'] != 1 or report['evaluated_states'] != 2:
        raise ValueError('Engineering diagnostic dropped real states or evaluated invalid teacher')
    write(out / 'ENGINEERING_PASS.json', {'real_teacher_diagnostic_pass': True,
        'states': 3, 'evaluated_states': 2, 'engineering_only': True, 'online_gain_established': False})
    state('waiting_fixed_training')
    deadline = time.monotonic() + addendum['dependency_wait_timeout_s']
    while True:
        if (main / 'pipeline_failure.json').exists() or (main / 'collection_failure.json').exists():
            state('main_experiment_failed', explanation=read(main / ('pipeline_failure.json' if (main / 'pipeline_failure.json').exists() else 'collection_failure.json')))
            return
        if (main / 'pipeline_state.json').exists() and read(main / 'pipeline_state.json')['stage'] == 'activation_not_met':
            state('main_failure_branch_not_activated')
            return
        try:
            ready = all(read(main / name / 'status.json')['stage'] == 'finished' and
                        read(main / name / 'config.json')['training_complete'] is True
                        for name in ('maskco', 'reconstruction_control'))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            ready = False
        if ready:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Fixed training unavailable within registered diagnostic wait')
        time.sleep(30)
    sealed()
    training_cfg = main / 'maskco_training_config.json'
    cfg = {'qualification': 'teacher_objective_diagnostic',
        'training_config': {'path': str(training_cfg), 'sha256': sha(training_cfg)},
        'frozen_extension': str(ext),
        'evaluation_keys': [[day, ordinal] for day in addendum['evaluation_days'] for ordinal in addendum['ordinals_by_day'][str(day)]],
        'models': {name: binding(main / name) for name in ('maskco', 'warmstart', 'reconstruction_control')},
        'cpu_set': cpu_set, **{key: addendum[key] for key in
            ('independent_sets', 'K', 'sampling_seed', 'bootstrap_samples', 'bootstrap_seed')}}
    state('evaluating_teacher_objective_holdout', states=32, days=8)
    evaluate(cfg, 'holdout_config.json', out / 'holdout', addendum['evaluation_timeout_s'])
    state('completed', checkpoint_selection_changed=False, online_gain_established=False)


if __name__ == '__main__':
    try:
        run(*sys.argv[1:3])
    except Exception as exc:
        Path(sys.argv[1], 'failure.json').write_text(json.dumps({'error_type': type(exc).__name__,
            'error': str(exc), 'main_experiment_changed': False, 'online_gain_established': False}, indent=2))
        raise
