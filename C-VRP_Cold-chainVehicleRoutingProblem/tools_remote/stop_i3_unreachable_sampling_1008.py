"""Stop only the exact owned v4 chain/observers after structural qualification failure."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
MAIN = ROOT / 'results/i3_registered_1008_v4'
DIAG = ROOT / 'results/i3_day12_collection_failure_diag_1008_v1'


def processes():
    rows = []
    for q in Path('/proc').iterdir():
        if not q.name.isdecimal():
            continue
        try:
            pid = int(q.name)
            stat = (q / 'stat').read_text().rsplit(')', 1)[1].split()
            rows.append({'pid': pid, 'pgid': os.getpgid(pid),
                'start_ticks': stat[19], 'zombie': stat[0] == 'Z',
                'argv': (q / 'cmdline').read_bytes().decode().strip('\0').split('\0')})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def stop(name, script_name):
    out = ROOT / 'results' / name
    script = str(out / script_name)
    live = processes()
    owners = [p for p in live if p['argv'][1:2] == [script] and not p['zombie']]
    if not owners:
        if any(not p['zombie'] and any(str(out) in a for a in p['argv']) for p in live):
            raise RuntimeError('Owned supervisor absent but possible children remain; no signal sent')
        save(out / 'sampling_stop_completed.json', {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
             'already_completed': True, 'live_remaining': 0, 'team_processes_touched': False})
        return
    if len(owners) != 1:
        raise RuntimeError('Expected one exact owned supervisor; no signal sent')
    owner = owners[0]
    group = [p for p in live if p['pgid'] == owner['pgid']]
    if owner['pgid'] == os.getpgrp() or any(not p['zombie'] and
            not any(str(out) in a for a in p['argv']) for p in group):
        raise RuntimeError('Unexpected group member; no signal sent')
    record = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'supervisor': owner,
        'before': group, 'reason': 'registered 128-state sampling is structurally impossible: only 125 ordinals exist',
        'sampling_rule_changed': False, 'old_outputs_preserved': True,
        'formal_training_started': False, 'team_processes_touched': False}
    save(out / 'sampling_stop_started.json', record)
    now = next((p for p in processes() if p['pid'] == owner['pid']), None)
    if now is None or (now['argv'], now['start_ticks'], now['pgid']) != (
            owner['argv'], owner['start_ticks'], owner['pgid']):
        raise RuntimeError('Supervisor identity changed; no signal sent')
    os.killpg(owner['pgid'], signal.SIGTERM)
    time.sleep(3)
    remaining = [p for p in processes() if p['pgid'] == owner['pgid'] and not p['zombie']]
    if remaining:
        if any(not any(str(out) in a for a in p['argv']) for p in remaining):
            raise RuntimeError('Unexpected surviving process; no further signals')
        os.killpg(owner['pgid'], signal.SIGKILL)
        time.sleep(1)
    record['live_remaining'] = sum(p['pgid'] == owner['pgid'] and not p['zombie']
                                   for p in processes())
    save(out / 'sampling_stop_completed.json', record)
    assert record['live_remaining'] == 0
    print(json.dumps({'chain': name, 'live_remaining': 0, 'old_outputs_preserved': True}))


if __name__ == '__main__':
    audit_path = DIAG / 'INDEPENDENT_CARDINALITY_AUDIT.json'
    audit = json.loads(audit_path.read_text())
    assert audit['old_existing_states'] == 125 and audit['old_expected_states'] == 128
    assert audit['old_missing_days'] == [12, 25, 29]
    assert audit['proposed_rule_approved'] is False
    base = json.loads((DIAG / 'BASE_DIAGNOSTIC.json').read_text())
    assert base['captured_count'] == 3 and base['decisions_seen'] == 150
    assert base['original_evaluation']['hard_feasible'] is True
    assert base['original_evaluation']['failures'] == []
    assert json.loads((MAIN / 'collection_failure.json').read_text())['day_id'] == 12
    save(MAIN / 'COLLECTION_QUALIFICATION_FAILURE.json', {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'classification': 'registered_sampling_design_infeasible',
        'census_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'expected_states': 128, 'existing_ordinals': 125,
        'missing_days': [12, 25, 29], 'original_day12_hard_feasible': True,
        'formal_training_started': False, 'online_started': False,
        'frozen_source_modified': False, 'sampling_amendment_approved': False})
    stop('i3_registered_1008_v4', 'pipeline.py')
    stop('i3_teacher_data_audit_1008_v2', 'chain.py')
    stop('i3_teacher_risk_1008_v2', 'chain.py')
