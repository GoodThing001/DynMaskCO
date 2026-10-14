"""Preserve and stop only the exact I3 v3 chain and its two read-only observers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
MAIN = ROOT / 'results/i3_registered_1008_v3'


def processes():
    rows = []
    for q in Path('/proc').iterdir():
        if not q.name.isdecimal():
            continue
        try:
            stat = (q / 'stat').read_text().rsplit(')', 1)[1].split()
            rows.append({'pid': int(q.name), 'pgid': os.getpgid(int(q.name)),
                         'start_ticks': stat[19], 'zombie': stat[0] == 'Z',
                         'argv': (q / 'cmdline').read_bytes().decode().strip('\0').split('\0'),
                         'cpu': sorted(os.sched_getaffinity(int(q.name)))})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def stop(name, script_name):
    out = ROOT / 'results' / name
    script = str(out / script_name)
    supervisors = [p for p in processes() if p['argv'][1:2] == [script] and not p['zombie']]
    if len(supervisors) != 1:
        raise RuntimeError('Expected exactly one owned supervisor: ' + name)
    owner = supervisors[0]
    group = [p for p in processes() if p['pgid'] == owner['pgid']]
    if owner['pgid'] == os.getpgrp() or any(not p['zombie'] and not any(str(out) in a for a in p['argv']) for p in group):
        raise RuntimeError('Unexpected process in exact owned group; no signals sent')
    record = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'supervisor': owner,
              'before': group, 'reason': 'day 3 archive write/reload identity failure; preserve engineering-invalid attempt',
              'scientific_parameters_changed': False, 'online_results_observed': False,
              'new_attempt': 'results/i3_registered_1008_v4', 'team_processes_touched': False}
    save(out / 'engineering_stop_started.json', record)
    # Recheck identity immediately before signaling, guarding PID reuse.
    now = next((p for p in processes() if p['pid'] == owner['pid']), None)
    if now is None or (now['argv'], now['start_ticks'], now['pgid']) != (owner['argv'], owner['start_ticks'], owner['pgid']):
        raise RuntimeError('Owned supervisor identity changed; no signals sent')
    os.killpg(owner['pgid'], signal.SIGTERM)
    time.sleep(3)
    remaining = [p for p in processes() if p['pgid'] == owner['pgid'] and not p['zombie']]
    if remaining:
        if any(not any(str(out) in a for a in p['argv']) for p in remaining):
            raise RuntimeError('Unexpected surviving process; do not escalate')
        os.killpg(owner['pgid'], signal.SIGKILL)
        time.sleep(1)
    record['after'] = [p for p in processes() if p['pgid'] == owner['pgid']]
    record['live_remaining'] = sum(not p['zombie'] for p in record['after'])
    save(out / 'engineering_stop_completed.json', record)
    if record['live_remaining']:
        raise RuntimeError('Owned process survived')
    print(json.dumps({'chain': name, 'live_remaining': 0, 'old_outputs_preserved': True}))


if __name__ == '__main__':
    failed = MAIN / 'teacher_job_003/run.log'
    if 'Teacher record/context identity mismatch' not in failed.read_text():
        raise RuntimeError('Expected exact engineering failure is absent')
    save(MAIN / 'ENGINEERING_INVALIDATION.json', {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'day': 3,
        'failure_log_sha256': hashlib.sha256(failed.read_bytes()).hexdigest(),
        'qualification': 'engineering_invalid_no_formal_training_or_online',
        'reports_preserved': sorted(p.parent.name for p in MAIN.glob('teacher_day_*/report.json')),
        'reason': 'archive validation passed before write but record identity changed after typed roundtrip',
        'frozen_source_modified': False, 'scientific_parameters_changed': False})
    stop('i3_registered_1008_v3', 'pipeline.py')
    stop('i3_teacher_data_audit_1008_v1', 'chain.py')
    stop('i3_teacher_risk_1008_v1', 'chain.py')
