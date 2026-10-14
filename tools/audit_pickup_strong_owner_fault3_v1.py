"""Independent byte and authority audit of the one new fault-3 request.

No admission, scoring, C0, worker, or old normal-request replay.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROUND = BASE / 'research/review_3080_20261010/goal_round_20261013_r11_fault3_head_arithmetic'
OUT = BASE / 'research/review_3080_20261010/pickup/strong_independent_owner_fault3_root_v1'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg = json.loads((ROUND / 'PICKUP_FAULT3_REGISTRATION.json').read_text(encoding='utf-8'))
    result = read('RESULT.json')
    row = result['records'][0]
    producer_reg = read('REGISTRATION.json')
    start = read('SOURCE_START_0.json')
    compute_end = read('SOURCE_COMPUTE_END_0.json')
    response = dict(read('RESPONSE_ATTEMPT_0.json')['__mapping__'])
    checks = {}
    checks['new_program_registered'] = sha(BASE / 'tools/qualify_pickup_strong_owner_fault3_v1.py') == reg['program_sha256']
    checks['prior_owner_registered'] = sha(BASE / 'research/review_3080_20261010/pickup/strong_independent_owner_root_v3/OWNER_RESULT_0.json') == reg['prior_owner_sha256']
    checks['single_registered_fault'] = len(result['records']) == 1 and row['fault'] == reg['fixed_fault'] == 3
    checks['stage_reached'] = row['stage_reached'] == 3 and row['promotion'] is not None
    checks['prepared_before_fault'] = row['prepared'] == response['compiled_state_sha256'] and len(row['prepared']) == 64
    checks['no_physical_authority'] = row['primary_advance'] is None and row['actual_process_owner_selected'] is False
    checks['old_owner_exact'] = row['baseline'] == row['old_owner_after'] == row['ready']['baseline']
    checks['cancel_and_reap'] = row['error'] == 'advance_cancelled' and row['reaped'] and 0 < row['wall_s'] < row['cap_s']
    checks['source_compute_closed'] = start['files'] == compute_end['files'] and start['module_files'] == compute_end['module_files']
    checks['old_worker_natural_end_not_fabricated'] = not (OUT / 'OWNER_RESULT_0.json').exists() and not (OUT / 'SOURCE_END_0.json').exists()
    checks['original_result_preserved'] = result['checks_count'] == 5 and result['failed'] == 0 and read('RUN_EXIT.json')['exit_code'] == 0
    checks['registered_scope'] = producer_reg['fault_cap_s'] == 10 and producer_reg['fault_reserve_s'] == 2 and not producer_reg['new_scientific_data']
    report = {'scope': 'single actual fault3 post-selection pre-physical authority, saved evidence only',
              'checks': checks, 'failed': [k for k,v in checks.items() if not v],
              'component_fault_recovery_qualified': all(checks.values()),
              'full_live_request_qualified': False, 'stable_online_gain': False}
    target = OUT / 'INDEPENDENT_FAULT3_AUTHORITY.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'checks': len(checks), 'failed': report['failed']}))
    return int(bool(report['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
