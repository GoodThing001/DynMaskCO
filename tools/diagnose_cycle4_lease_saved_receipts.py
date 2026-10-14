"""Narrow saved-only diagnosis; never rerun locks or rewrite original FAILs."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research/review_3080_20261010/delivery/local_source_lease_boundary_root_v1'
NAMES = ['RESULT.json', 'INDEPENDENT_AUDIT.json', 'leased_RESULT.json', 'owner_held_RESULT.json',
         'released_RESULT.json', 'owner_exited_RESULT.json', 'OWNER_EXIT.json', 'SOURCE_START.json',
         'SOURCE_END.json', 'RUN_EXIT.json', 'AUDIT_RUN_EXIT.json', 'probe.py', 'owned/unleased_extra.py']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(name):
    return json.loads((STAGE / name).read_text(encoding='utf-8'))

def create(name, value):
    with (STAGE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    if sys.argv[1] == 'register':
        create('SAVED_FAILURE_DIAGNOSIS_REGISTRATION_R2.json', {
            'inputs': {n: sha(STAGE / n) for n in NAMES}, 'program_sha256': sha(Path(__file__)),
            'scope': 'saved bytes, original failed predicate names, error receipts and actual source-set difference only',
            'no_lease_probes_or_original_audit_rerun': True,
            'does_not_change_original_registered_error_code_or_source_gates': True})
        return
    assert sys.argv[1] == 'diagnose'
    reg = read('SAVED_FAILURE_DIAGNOSIS_REGISTRATION_R2.json')
    checks = []
    def ck(name, value):
        checks.append({'name': name, 'pass': bool(value)})
    for n, expected in reg['inputs'].items():
        ck('input:' + n, sha(STAGE / n) == expected)
    ck('program', sha(Path(__file__)) == reg['program_sha256'])
    production = read('RESULT.json'); audit = read('INDEPENDENT_AUDIT.json')
    leased = read('leased_RESULT.json'); held = read('owner_held_RESULT.json')
    six = leased['rows'][:6]
    ck('six_actual_denials', len(six) == 6 and all(r['success'] is False for r in six))
    ck('original_production_fail_names', [r['name'] for r in production['checks'] if not r['pass']] ==
       ['six_mutations_blocked_by_sharing', 'separate_owner_blocks_write', 'parent_source_closed'])
    ck('original_audit_fail_names', [r['name'] for r in audit['checks'] if not r['pass']] ==
       ['sharing_rejection:write', 'sharing_rejection:truncate', 'sharing_rejection:replace',
        'sharing_rejection:pyc_write', 'new_file_real', 'parent_closed'])
    ck('original_fail_counts_kept', production['failed'] == 3 and audit['failed'] == 6)
    ck('held_write_denied', held['rows'][0]['success'] is False)
    ck('released_write_allowed', read('released_RESULT.json')['rows'][0]['success'] is True)
    ck('owner_exited_write_allowed', read('owner_exited_RESULT.json')['rows'][0]['success'] is True)
    ck('before_after_bytes_and_ID_exact', production['before'] == production['after'])
    raw_extra = (STAGE / 'owned/unleased_extra.py').read_bytes()
    ck('actual_extra_CRLF', raw_extra == b'UNREGISTERED=1\r\n')
    tree = ast.parse((STAGE / 'probe.py').read_text(encoding='utf-8'))
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and n.func.attr == 'open' and
             any(isinstance(a, ast.Constant) and a.value == 'x' for a in n.args)]
    ck('original_text_writer_default_newline', len(opens) == 1 and
       not any(k.arg == 'newline' for k in opens[0].keywords))
    begin = read('SOURCE_START.json'); end = read('SOURCE_END.json')
    added = sorted(set(end['files']) - set(begin['files']))
    changed = sorted(p for p, h in begin['files'].items() if end['files'].get(p) != h)
    ck('three_missing_start_files', len(added) == 3 and
       {Path(p).name for p in added} == {'kernel.appcore.dll', 'gbk.py', 'gbk.cpython-312.pyc'})
    ck('no_existing_file_changed', not changed)
    ck('original_natural_failure_and_reap', all(read(n)['exit_code'] == 1 and read(n)['reaped']
       for n in ('RUN_EXIT.json', 'AUDIT_RUN_EXIT.json')))
    create('SAVED_FAILURE_DIAGNOSIS_R2.json', {
        'checks_count': len(checks), 'failed': sum(not r['pass'] for r in checks), 'checks': checks,
        'original_production': {'checks': 10, 'failures': 3, 'exit_code': 1},
        'original_independent': {'checks': 951, 'failures': 6, 'exit_code': 1},
        'actual_error_receipts': six, 'original_extra_file_hex': raw_extra.hex(),
        'missing_start_files': added, 'existing_byte_changes': changed,
        'interpretation': 'All six attempts were denied and IDs/bytes unchanged. Original exact winerror32 assumptions and LF expectation do not match saved receipts. Sharing-specific cause for replace winerror5 not uniquely isolated; no released replace counterfactual exists. Missing-start files remain real qualification failure, and first-load trigger is not uniquely localized.',
        'full_runtime_guard_qualified': False, 'source_qualified': False,
        'full_request_budget_qualified': False, 'stable_online_gain': False,
        'old_results_rewritten': False, 'old_probes_or_audit_executions': 0})
    print(json.dumps({'checks_count': len(checks), 'failed': sum(not r['pass'] for r in checks),
                      'original_FAIL_preserved': True, 'source_qualified': False}))
    assert all(r['pass'] for r in checks)

if __name__ == '__main__':
    main()
