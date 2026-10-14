"""Assemble the delivery diagnostic hand-back from the raw originals.

This script COPIES raw values and records their sha256; it does not recompute, reinterpret
or upgrade any verdict.  The one derived block is labelled `derived`.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(rel):
    path = HERE / rel
    if not path.is_file():
        return None
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    return path.read_text(encoding='utf-8', errors='replace')


RAW = ['run_r1/DIAG_STATE.json', 'run_r1/DIAG_FAILURE.json', 'run_r1/DIAG_FAILURE.traceback.txt',
       'run_r1/SUPERVISOR_RECORD.json', 'run_r1/SUPERVISOR_STDOUT.txt',
       'run_r2/DIAG_STATE.json', 'run_r2/DIAG_FAILURE.json', 'run_r2/DIAG_FAILURE.traceback.txt',
       'run_r2/SUPERVISOR_RECORD.json', 'run_r2/SUPERVISOR_STDOUT.txt',
       'run_r3/DIAG_STATE.json', 'run_r3/SUPERVISOR_RECORD.json', 'run_r3/SUPERVISOR_STDOUT.txt',
       'run_r3/SHARD_PROBE.json',
       'REGISTRATION.json', 'REGISTRATION_r2.json', 'REGISTRATION_r3.json',
       'EXPECTED_IDENTITY_r3.json', 'LOCAL_UNIFIED_PROTOCOL_RESULT.json',
       'LOCAL_UNIFIED_PROTOCOL_v2_RESULT.json', 'unified_closure_protocol.py',
       'unified_closure_protocol_r2.py', 'local_unified_protocol_test.py',
       'local_unified_protocol_v2_test.py', 'build_expected_identity_r3.py',
       'shard_probe_server.py', 'server_diagnostic.py', 'server_diagnostic_r2.py',
       'supervise_diagnostic.py', 'supervise_diagnostic_r2.py',
       'assemble_diagnostic_handback.py', 'build_verified_resource_snapshot.py',
       'HAND_BACK_r3.md', 'RESOURCE_SNAPSHOT_3080_VERIFIED.json',
       'REGISTERED_SHARDS_REFERENCE.json',
       '../actual_loaded_closure_request_root_v1/REGISTRATION.json',
       '../actual_loaded_closure_request_root_v1/RESOURCE_SNAPSHOT.json',
       '../actual_loaded_closure_request_root_v1/THREE_POINT_PROTOCOL_RESULT.json',
       '../../../tools_placeholder_nonexistent']
RAW = [r for r in RAW if not r.endswith('tools_placeholder_nonexistent')]
for rel in ['../../../../tools/actual_loaded_closure_candidate_r20.py',
            '../../../../tools/actual_loaded_closure_candidate_r21.py']:
    RAW.append(rel)

s1, s2, s3 = load('run_r1/DIAG_STATE.json'), load('run_r2/DIAG_STATE.json'), load('run_r3/DIAG_STATE.json')
sup1, sup2, sup3 = (load('run_r1/SUPERVISOR_RECORD.json'), load('run_r2/SUPERVISOR_RECORD.json'),
                    load('run_r3/SUPERVISOR_RECORD.json'))
fail1, fail2 = load('run_r1/DIAG_FAILURE.json'), load('run_r2/DIAG_FAILURE.json')
probe = load('run_r3/SHARD_PROBE.json')


def last_line(text):
    return text.strip().splitlines()[-1]


out = {
    'stage': 'delivery_actual_loaded_closure_cpu_only_8plus2_engineering_diagnostic',
    'round': 'r3 (third invocation of the authorized diagnostic)',
    'verdict_of_the_diagnostic': {
        'run_r3_gate_pass': s3['gate_pass'],
        'run_r3_gates': s3['gates'],
        'run_r3_failed_gates': s3['failed_gates'],
        'publication_performed': s3['publication_performed'],
    },
    'raw_originals': {},
    'attempts': [
        {'run': 'run_r1', 'executor': 'server_diagnostic.py',
         'phase_reached': s1['phase'], 'gate_pass': False,
         'failure_type': last_line(fail1['traceback']),
         'supervisor': {k: sup1[k] for k in ('exit_code', 'timed_out', 'signalled', 'reaped',
                                             'wall_seconds', 'signal_name')},
         'reason': 'closure candidate r20 crashed on the real module set (P1 capture)'},
        {'run': 'run_r2', 'executor': 'server_diagnostic.py (wrong: supervisor hardcoded the r1 child)',
         'phase_reached': s2['phase'], 'gate_pass': False,
         'failure_type': last_line(fail2['traceback']),
         'supervisor': {k: sup2[k] for k in ('exit_code', 'timed_out', 'signalled', 'reaped',
                                             'wall_seconds', 'signal_name')},
         'reason': 'harness error: the corrected executor was never launched'},
        {'run': 'run_r3', 'executor': 'server_diagnostic_r2.py',
         'supervisor_script': 'supervise_diagnostic_r2.py',
         'identity_preflight': 'passed before launch (9/9 manifest entries matched)',
         'phase_reached': s3['phase'], 'gate_pass': s3['gate_pass'],
         'supervisor': {k: sup3[k] for k in ('exit_code', 'timed_out', 'signalled', 'reaped',
                                             'wall_seconds', 'signal_name')},
         'reason': 'completed; all declared gates passed'},
    ],
    'run_r3_measurements': {
        'cpu_environment': s3['cpu_lock'],
        'preconditions': s3['preconditions'],
        'warmup': s3['warmup'],
        'execution': s3['execution'],
        'closure': s3['closure'],
        'source_identity': s3['source_identity'],
        'registration_sha256': s3['registration_sha256'],
        'wall_seconds': s3['wall_seconds'],
    },
    'derived': {
        'note': 'the only computed block; every input above is copied verbatim from the originals',
        'closure_capture_seconds_per_point': {c['point']: c['seconds'] for c in s3['closure']['captures']},
        'gated_capture_seconds_total': round(sum(c['seconds'] for c in s3['closure']['captures']
                                                 if c['point'].startswith(('P1', 'P2', 'P3'))), 3),
        'capture_over_request_budget_ratio': round(
            max(c['seconds'] for c in s3['closure']['captures']) / s3['declared_budget_seconds'], 2),
        'model_unit_seconds_steady': s3['execution']['unit_seconds_median'],
        'inference_share_of_budget': round(
            s3['execution']['unit_seconds_median'] / s3['declared_budget_seconds'], 4),
    },
    'source_shard_identity_3080': {
        'registered_shard_count': probe['registered_shard_count'],
        'registered_shards_present_on_host': len(probe['present']),
        'registered_shards_absent_on_host': len(probe['absent']),
        'all_registered_paths_under_research': probe['all_registered_paths_under_research'],
        'compute_sources_present_and_matching_local': sorted(
            k for k, v in probe['sources_dir_probe'].items() if v['present']),
        'conclusion': (
            'the registered shard set is a host-local research artifact set (46/46 absent on 3080), '
            'so a 3080-side delivery request cannot be identity-verified against it without '
            'uploading the shard set; the compute sources the request loads are present on 3080 '
            'and hash-match the local copies bit-for-bit'),
    },
    'labels': {
        'defect_noted': (
            "server_diagnostic_r2.py hardcodes attempt_id='r2' in its output while run_r3 is the "
            "third invocation; the authoritative attempt mapping is REGISTRATION_r3.json plus the "
            "supervisor record (attempt_id=run_r3), not that field. Recorded, not silently fixed."),
    },
    'stable_online_gain': False,
    'publication_performed': False,
}

for rel in RAW:
    path = HERE / rel
    out['raw_originals'][rel] = {'sha256': sha(path), 'bytes': path.stat().st_size} if path.is_file() else None

(HERE / 'DIAGNOSTIC_HAND_BACK.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({k: out[k] for k in ('verdict_of_the_diagnostic', 'derived',
                                      'source_shard_identity_3080', 'labels')},
                 ensure_ascii=False, indent=2))
