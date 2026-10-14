"""Read-only server evidence for the A3 -> registered I3 transition."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

chain = ROOT / 'results/a3_repair_chain_1007_2250'
answer_path = chain / 'A3_SAME_SOURCE_ADJUDICATION.json'
answer = read(answer_path)
assert answer['formal_adjudicable'] and not answer['partial'] and answer['n_batches'] == 9
files, weights = {}, {}
for name, batch in answer['per_batch'].items():
    gate = ROOT / 'results' / name / 'gate.json'
    files[gate.relative_to(ROOT).as_posix()] = sha(gate)
    files[gate.with_name('run.exit').relative_to(ROOT).as_posix()] = sha(gate.with_name('run.exit'))
    doc = read(gate)
    ident = doc['results'][batch['seed']]['_identity']
    weight = Path(ident['model_bin'])
    digest = sha(weight)
    assert digest == ident['model_bin_sha256']
    weights[str(weight)] = digest
    baseline = ROOT / batch['baseline_gate']
    files[baseline.relative_to(ROOT).as_posix()] = sha(baseline)
    files[baseline.with_name('run.exit').relative_to(ROOT).as_posix()] = sha(baseline.with_name('run.exit'))
files[answer_path.relative_to(ROOT).as_posix()] = sha(answer_path)

main = ROOT / 'results/i3_registered_1008_v3'
proto, cfg = read(main / 'protocol.json'), read(main / 'teacher_collection_config.json')
assert cfg['collection'] == proto['teacher_collection']
# The supervisor seals its own serialization; verify content against the original,
# and verify the saved-copy bytes against the collection binding separately.
assert read(main / 'A3_ACTIVATION.json') == answer
assert sha(main / 'A3_ACTIVATION.json') == cfg['activation_a3']['digest']
assert sha(main / 'protocol.json') == cfg['protocol']['digest']
ext = main / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
assert all(sha(ext / rel) == digest for rel, digest in cfg['source_sha256'].items())
seal = subprocess.run(['sha256sum', '-c', str(main / 'SOURCE_MANIFEST.sha256')],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
assert seal.returncode == 0
jobs, old_jobs = [], []
for path in Path('/proc').iterdir():
    if not path.name.isdecimal():
        continue
    try:
        command = (path / 'cmdline').read_bytes().decode().split('\0')
        if any(str(ROOT / 'results/i3_registered_1008_v2') in arg for arg in command):
            old_jobs.append({'pid': int(path.name), 'command': command})
        if str(ext / 'scripts/training/collect_i3_teachers.py') in command:
            pid = int(path.name)
            assert command[command.index('--config') + 1] == str(main / 'teacher_collection_config.json')
            assert command[command.index('--config-sha256') + 1] == sha(main / 'teacher_collection_config.json')
            jobs.append({'pid': pid, 'day_id': int(command[command.index('--day-id') + 1]),
                         'cpu_affinity': sorted(os.sched_getaffinity(pid)),
                         'frozen_collector_path': str(ext / 'scripts/training/collect_i3_teachers.py')})
    except (ProcessLookupError, FileNotFoundError):
        continue
assert not old_jobs
assert len(jobs) <= 8 and len({row['day_id'] for row in jobs}) == len(jobs)
assert all(len(row['cpu_affinity']) == 1 for row in jobs)
result = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'a3_artifact_hashes': files,
          'a3_actual_weight_hashes': weights, 'a3_qualification_complete': True,
          'i3_protocol_sha256': sha(main / 'protocol.json'),
          'i3_activation_sha256': sha(main / 'A3_ACTIVATION.json'),
          'i3_activation_content_equals_original': True,
          'i3_collection_config_sha256': sha(main / 'teacher_collection_config.json'),
          'i3_collection_equals_registered': True, 'i3_source_seal_ok': True,
          'i3_seal_files_checked': len(seal.stdout.splitlines()),
          'i3_live_collectors': jobs, 'old_v2_live_processes': old_jobs,
          'completed_teacher_days': [int(p.parent.name.split('_')[-1]) for p in main.glob('teacher_day_*/report.json')],
          'i3_pipeline_state': read(main / 'pipeline_state.json'),
          'supplement_state': read(ROOT / 'results/i3_teacher_risk_1008_v1/state.json'),
          'supplement_engineering_pass': read(ROOT / 'results/i3_teacher_risk_1008_v1/ENGINEERING_PASS.json'),
          'maskco_online_gain_established': False, 'reserved_final_tests_run': False}
out = chain / ('CORE_GAIN_TRANSITION_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
with out.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({'out': out.relative_to(ROOT).as_posix(), 'all_checks_passed': True,
                  'collectors': len(jobs), 'completed_teacher_days': result['completed_teacher_days']}))
