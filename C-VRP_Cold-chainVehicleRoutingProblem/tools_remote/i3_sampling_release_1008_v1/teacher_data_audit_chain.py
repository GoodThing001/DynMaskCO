"""Finite, read-only terminal audit of each registered teacher shard as it finishes."""
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


def run(out, root):
    out, root = Path(out), Path(root)
    main = root / 'results/i3_registered_1008_v5'
    ext = main / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
    protocol_sha = '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c'
    if sha(main / 'protocol.json') != protocol_sha:
        raise ValueError('Teacher audit refers to a different protocol')
    reg = read(main / 'protocol.json')
    code_digest = sha(out / 'audit.py')
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'JAX_PLATFORMS': 'cpu'}
    cpu = sorted(os.sched_getaffinity(0))[52]
    expected = [(day, ordinal) for day in reg['teacher_collection']['days']
                for ordinal in reg['teacher_collection']['capture_ordinals_by_day'][str(day)]]

    def state(stage, **fields):
        out.joinpath('state.json').write_text(json.dumps({'stage': stage,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), **fields}, indent=2), encoding='utf-8')

    def seal():
        if sha(out / 'audit.py') != code_digest or sha(main / 'protocol.json') != protocol_sha:
            raise ValueError('Audit or main protocol changed')
        result = subprocess.run(['sha256sum', '-c', str(main / 'SOURCE_MANIFEST.sha256')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError('Main frozen source seal failed')

    seal()
    deadline = time.monotonic() + 108000  # Finite 30h, no numerical retries.
    reports, seen = [], set()
    state('waiting_teacher_shards', audited_days=0)
    while len(seen) < len(reg['teacher_collection']['days']):
        if (main / 'pipeline_failure.json').exists() or (main / 'collection_failure.json').exists():
            state('main_experiment_failed', failure=read(main / ('pipeline_failure.json' if (main / 'pipeline_failure.json').exists() else 'collection_failure.json')))
            return
        for day in reg['teacher_collection']['days']:
            if day in seen or not (main / ('teacher_day_%03d/report.json' % day)).exists():
                continue
            seal()
            target = out / ('day_%03d.json' % day)
            with out.joinpath('day_%03d.log' % day).open('x') as log:
                result = subprocess.run(['taskset', '-c', str(cpu), sys.executable, str(out / 'audit.py'),
                    '--main', str(main), '--frozen-ext', str(ext), '--day', str(day), '--out', str(target)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            out.joinpath('day_%03d.exit' % day).write_text('RUN_EXIT=%s\n' % result.returncode)
            if result.returncode:
                raise RuntimeError('Teacher trace audit failed: day=%s' % day)
            reports.append(read(target))
            seen.add(day)
            state('auditing_teacher_shards', audited_days=len(seen), audited_states=sum(len(r['rows']) for r in reports))
        if time.monotonic() > deadline:
            raise RuntimeError('Registered teacher shards unavailable after fixed 30h wait')
        if len(seen) < len(reg['teacher_collection']['days']):
            time.sleep(30)
    seal()
    # The audit's pure aggregation code; no working-directory training imports.
    import importlib.util
    spec = importlib.util.spec_from_file_location('independent_teacher_audit', out / 'audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = sorted([row for report in reports for row in report['rows']], key=lambda r: (r['day_id'], r['ordinal']))
    summary = module.summarize(rows, expected)
    split = {name: module.summarize([r for r in rows if r['day_id'] in days],
                [(day, ordinal) for day, ordinal in expected if day in days])
             for name, days in [('training', reg['train_days']), ('teacher_objective_holdout', reg['holdout_days'])]}
    with out.joinpath('TEACHER_DATA_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump({'main_output': str(main), 'summary': summary, 'split': split, 'rows': rows,
            'inputs': [{'report_sha256': r['report_sha256'], 'archive_sha256': r['archive_sha256']} for r in reports],
            'audit_source_sha256': code_digest, 'protocol_sha256': protocol_sha,
            'terminal_trace_reevaluation_complete': True,
            'complete_policy_rerun_performed': False, 'parameters_changed': False,
            'training_gate_changed': False, 'online_gain_established': False}, stream, indent=2, allow_nan=False)
    state('completed', audited_days=32, audited_states=128, online_gain_established=False)


if __name__ == '__main__':
    try:
        run(*sys.argv[1:3])
    except Exception as exc:
        Path(sys.argv[1], 'failure.json').write_text(json.dumps({'error_type': type(exc).__name__,
            'error': str(exc), 'main_experiment_changed': False}, indent=2), encoding='utf-8')
        raise
