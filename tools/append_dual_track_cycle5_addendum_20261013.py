"""Append-only provenance and source-identity addendum for cycle 5."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
CYCLE = TREE / 'dual_track_optimization_20261013_root_v1/cycle5_20261013_root_v1'
PK = TREE / 'pickup/lazy_base_distance_reuse_root_v1'
D1 = TREE / 'delivery/action_matching_head_update_root_v1'
D2 = TREE / 'delivery/action_matching_head_update_root_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


pickup_run = json.loads((PK / 'RUN_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))
pickup_audit = json.loads((PK / 'AUDIT_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))
d1_result = json.loads((D1 / 'RESULT.json').read_text(encoding='utf-8'))
d1_exit = json.loads((D1 / 'RUN_EXIT.json').read_text(encoding='utf-8'))
d2_result = json.loads((D2 / 'RESULT.json').read_text(encoding='utf-8'))
d2_audit = json.loads((D2 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))

for arm, arm_dir in (('pickup', PK), ('delivery_v2', D2)):
    for name in ('RESULT.json', 'RUN_EXIT.json'):
        pass
assert (PK / 'RESULT.json').exists() and (D2 / 'RESULT.json').exists()

addendum = {
    'scope': 'append-only addendum; the already written ITERATION_RECEIPT.json and evidence hashes are unchanged',
    'execution_provenance': {
        'pickup_n11': {
            'executed_by': pickup_run['identity']['owner'],
            'run_pid': pickup_run['identity']['pid'],
            'run_started_perf_counter': pickup_run['started'],
            'audit_pid': pickup_audit['identity']['pid'],
            'note': ('The pickup n11 component was executed and independently audited earlier inside cycle 5 by the '
                     'preceding cycle execution (artifacts timestamped 2026-10-13 before this continuation). This '
                     'continuation re-read and reconciled its recorded evidence (producer/audit exits, check counts, '
                     'terminal hexes, registration and evidence hashes) and deliberately did not re-run it, per the '
                     'no-repeat rule. It is not re-executed or upgraded here.'),
        },
        'delivery_n12': {
            'v1_owner': d1_result['resource'].get('cpu_affinity') and 'recorded in RESULT.json',
            'v1_attempt': rel(D1),
            'v1_failed_checks': d1_result['failed'],
            'v1_exit_code': d1_exit['exit_code'],
            'v1_note': ('v1 was executed inside cycle 5 by the preceding cycle execution and is preserved unchanged; '
                        'its two FAILs and exit 1 are not rewritten.'),
            'v2_executed_by': json.loads((D2 / 'RUN_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))['identity']['owner'],
            'v2_pid': json.loads((D2 / 'RUN_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))['identity']['pid'],
            'v2_audit_pid': json.loads((D2 / 'AUDIT_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))['identity']['pid'],
            'v2_note': ('v2 was registered before execution by this continuation, then run, then independently audited in '
                        'a separate process; its two audit FAILs are preserved and were not converted to PASS.'),
        },
    },
    'preserved_originals_unchanged_now': {
        rel(D1 / 'RESULT.json'): sha(D1 / 'RESULT.json'),
        rel(D1 / 'RUN_EXIT.json'): sha(D1 / 'RUN_EXIT.json'),
        'v1_source_end_absent': not (D1 / 'SOURCE_END.json').exists(),
        'v1_failed_checks_now': d1_result['failed'],
        'v1_exit_code_now': d1_exit['exit_code'],
        'v2_failed_checks_now': d2_result['failed'],
        'v2_audit_failed_now': d2_audit['failed'],
        'v2_audit_failure_names_now': d2_audit['failure_names'],
    },
    'source_identity_not_in_the_main_evidence_block': {
        rel(D2 / 'optimizer_reproducibility_diagnosis.py'): sha(D2 / 'optimizer_reproducibility_diagnosis.py'),
        rel(D1 / 'probe_inputs.py'): sha(D1 / 'probe_inputs.py'),
        rel(D2 / 'prepare.py'): sha(D2 / 'prepare.py'),
        rel(D2 / 'supervise.py'): sha(D2 / 'supervise.py'),
    },
    'read_only_preparation': [
        rel(D1 / 'probe_inputs.py') + ' (read-only probe of the sealed head inputs; produced no stage artifact)',
    ],
    'shared_resource': dict(cpu_affinity=[7], serial=True, pickup_first=True,
                            gpu_used=False, server_used=False, shared_MD_uploaded=False),
    'stable_online_gain': False,
    'scientific_family3_started': False,
}
out = CYCLE / 'EVIDENCE_ADDENDUM.json'
assert not out.exists(), out
out.write_text(json.dumps(addendum, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'written': rel(out), 'v1_failed': d1_result['failed'], 'v2_audit_failed': d2_audit['failed']}))
