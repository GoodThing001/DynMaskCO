"""Observe the unchanged frozen day-12 baseline; no teacher/model promotion.

Uses the exact registered collection entry point and intercepts only its first
terminal evaluation to preserve the missing fault evidence. Stops before any
counterfactual replay or training. Frozen source and protocol remain untouched.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(out):
    root = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
    main = root / 'results/i3_registered_1008_v4'
    out = Path(out)
    assert out.is_dir() and out.resolve().parent == root / 'results'
    frozen = main / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
    config_path = main / 'teacher_collection_config.json'
    protocol_sha = '345604e3c3ed0ff2c40e049ded3c4a5a6784fcde1f464fe221dbc98850d8f5a5'
    assert sha(main / 'protocol.json') == protocol_sha
    assert json.loads((main / 'collection_failure.json').read_text())['day_id'] == 12
    config_sha = sha(config_path)
    for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(frozen / 'scripts' / folder))
    import collect_i3_teachers
    import coldchain_evaluator_a1
    from i3_counterfactual_teacher import CaptureTeacherReplanner
    from i3_teacher_archive import _pack, _canonical
    from strict_online_env import StrictOnlineEnv
    from run_identity import dataset_identity, dataset_meta_sha256

    observed = {}
    original_run = StrictOnlineEnv.run
    original_hook = CaptureTeacherReplanner.capture_hook
    original_eval = coldchain_evaluator_a1.evaluate_trace_a1

    class BaseObserved(Exception):
        pass

    def observe_run(env, *args, **kwargs):
        observed['env'] = env
        return original_run(env, *args, **kwargs)

    def observe_hook(policy, records, **kwargs):
        observed['policy'], observed['records'] = policy, records
        return original_hook(policy, records, **kwargs)

    def observe_evaluation(traces, instance, contract, accepted, rejected, budget, penalty):
        result = original_eval(traces, instance, contract, accepted, rejected, budget, penalty)
        env, policy, records = observed['env'], observed['policy'], observed['records']
        payload = _pack({'records': records, 'dataset': env.dataset,
                         'traces': [asdict(t) for t in traces],
                         'accepted': accepted, 'rejected': rejected,
                         'planner_state': policy.export_state()})
        raw = _canonical(payload)
        path = out / 'base_private_typed.json'
        with path.open('xb') as stream:
            stream.write(raw)
        config = json.loads(config_path.read_text())
        answer = {'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'purpose': 'read-only collection failure diagnosis', 'day_id': 12,
                  'qualification': 'diagnostic_only_not_formal_teacher',
                  'protocol_sha256': protocol_sha, 'collection_config_sha256': config_sha,
                  'dataset_sha256': dataset_identity(env.dataset),
                  'dataset_meta_sha256': dataset_meta_sha256(env.dataset),
                  'captured_count': len(records),
                  'expected_count': len(config['collection']['capture_ordinals']),
                  'decisions_seen': policy._capture_ordinal,
                  'captured_candidates': [r.candidate for r in records],
                  'original_evaluation': result,
                  'private_payload_sha256': hashlib.sha256(raw).hexdigest(),
                  'scientific_parameters_changed': False,
                  'frozen_source_modified': False, 'formal_states_promoted': False,
                  'online_gain_established': False}
        write(out / 'BASE_DIAGNOSTIC.json', answer)
        print(json.dumps(answer, indent=2), flush=True)
        raise BaseObserved()

    StrictOnlineEnv.run = observe_run
    CaptureTeacherReplanner.capture_hook = observe_hook
    coldchain_evaluator_a1.evaluate_trace_a1 = observe_evaluation
    write(out / 'state.json', {'stage': 'running_unchanged_day12_base',
                              'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                              'config_sha256': config_sha, 'protocol_sha256': protocol_sha})
    try:
        collect_i3_teachers.collect(config_path, config_sha, 12, out / 'observed_collection')
        raise RuntimeError('The observational stop was not reached')
    except BaseObserved:
        seal = subprocess.run(['sha256sum', '-c', str(main / 'SOURCE_MANIFEST.sha256')],
                              capture_output=True, text=True)
        assert seal.returncode == 0 and sha(config_path) == config_sha
        assert sha(main / 'protocol.json') == protocol_sha
        write(out / 'COMPLETED.json', {'stage': 'completed',
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seal_exit': seal.returncode,
            'source_items_ok': sum(s.endswith(': OK') for s in seal.stdout.splitlines()),
            'diagnostic_sha256': sha(out / 'BASE_DIAGNOSTIC.json'),
            'formal_training_started': False, 'online_gain_established': False})


if __name__ == '__main__':
    try:
        run(sys.argv[1])
    except Exception as exc:
        write(Path(sys.argv[1]) / 'failure.json', {'error_type': type(exc).__name__,
            'error': str(exc), 'formal_training_started': False})
        raise
