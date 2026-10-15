"""Seal the already completed V1.9 pass; never starts or retries a worker."""
import hashlib
import json
import platform
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E = ROOT / 'v19_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, content):
    with open(p, 'x', encoding='utf-8') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def main():
    audit = json.loads((E / 'FORMAL_64_INDEPENDENT_AUDIT.json').read_text())
    result = json.loads((E / 'formal_collection/SUPERVISOR_RESULT.json').read_text())
    ident = json.loads((E / 'RUNTIME_IDENTITY_V19.json').read_text())
    ledger = [json.loads(s) for s in (E / 'formal_collection/STATE_LEDGER.jsonl').read_text().splitlines()]
    acceptance = json.loads((E / 'engineering_release/ACCEPTANCE.json').read_text())
    kill = json.loads((E / 'kill_release/KILL_E2E.json').read_text())
    auth = json.loads((E / 'authorization_release/AUTHORIZATION_GATE.json').read_text())
    regression = json.loads((E / 'REGRESSION.json').read_text())
    source_checks = {n: sha(EXT / rec['path']) == rec['sha256'] for n, rec in ident['modules'].items()}
    checks = {
        'engineering_44': acceptance['n_fail'] == 0 and acceptance['n_pass'] == 44,
        'kill_eight_real_trees': kill['n_fail'] == 0 and len(kill['rows']) == 8,
        'authorization_20': auth['n_fail'] == 0 and auth['n_pass'] == 20,
        'shared_regression_38': regression['n_fail'] == 0 and regression['n_pass'] == 38,
        'formal_64_published': result['verdict'] == 'BATCH_COMPLETE' and result['slots_published'] == 64,
        'independent_1498': audit['n_fail'] == 0 and audit['n_pass'] == 1498 and audit['qualified_slots'] == 64,
        'source_stable_and_current': result['source_stable'] and all(source_checks.values()),
        'no_failed_or_unreached_slot': not any(r['event'] in ('FAILED', 'NOT_ATTEMPTED_DUE_TO_TERMINATION') for r in ledger),
    }
    report = {
        'classification': 'completed_one_pass_F_free_state_collection_not_gain_evidence',
        'checks': checks, 'all_collection_engineering_checks_pass': all(checks.values()),
        'machine': platform.platform(), 'python': sys.version, 'run_id': result['run_id'],
        'prelaunch_commit': '806efd559eea66571dee0ad4894c751a5d46175a',
        'formal_slots_collected': 64, 'denominator': 64,
        'authorization': 'HUMAN_CONDITIONAL_AUTHORIZATION_RECORDED_AND_CONSUMED_ONCE',
        'authorization_sha256': sha(E / 'USER_CONDITIONAL_AUTHORIZATION.json'),
        'runtime_identity_sha256': sha(E / 'RUNTIME_IDENTITY_V19.json'),
        'runtime_source_archive_sha256': sha(E / 'RUNTIME_SOURCE_V19.zip'),
        'independent_audit_sha256': sha(E / 'FORMAL_64_INDEPENDENT_AUDIT.json'),
        'source_stability_and_current': source_checks, 'runtime_identity': ident,
        'timing': {'batch_wall_s': result['batch_wall_s'],
                   'max_day_rollout_wall_s': max(r['day_rollout_wall_s'] for r in ledger if r['event'] == 'DAY_ROLLOUT_FINISHED'),
                   'max_slot_span_wall_s': max(r['slot_span_wall_s'] for r in ledger if r['event'] == 'SLOT_FINISHED'),
                   'frozen_caps_s': result['caps']},
        'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'F_D_Geo_comparisons': 0, 'training_runs': 0, 'L2_L3_batches': 0,
        'final_test_seeds_touched': False, 'additional_formal_collection_authorization': 'NOT_AUTHORIZED',
        'remaining': ['certified common post-baseline incumbent and actual F input qualification',
                      'E2 and D8 full state/contract-bound installation evidence',
                      'separately frozen and authorized L1 attribution, L2 quality and L3 paired-day utility gates',
                      'cross-condition stability; retain I3 negative result and historical backbone uncertainty'],
        'linux_5090_3080_execution_qualification': 'NOT_ESTABLISHED_NOT_EXECUTED',
    }
    write_new(E / 'COLLECTION_ADJUDICATION_V19.json', report)
    gates = ROOT / 'l1_gates'
    copies = {
        'A1_D3_L1_COLLECTION_ADJUDICATION_V19.json': E / 'COLLECTION_ADJUDICATION_V19.json',
        'A1_D3_L1_FORMAL_64_INDEPENDENT_AUDIT_V19.json': E / 'FORMAL_64_INDEPENDENT_AUDIT.json',
        'A1_D3_L1_P0C_RUNTIME_IDENTITY_V19.json': E / 'RUNTIME_IDENTITY_V19.json',
        'A1_D3_L1_STATE_MANIFEST_64_V19.json': E / 'STATE_MANIFEST_64_V19.json',
    }
    for name, original in copies.items():
        with open(gates / name, 'xb') as fh:
            fh.write(original.read_bytes())
    files = list(E.rglob('*')) + list(gates.glob('*V19.json'))
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(files) if p.is_file()}
    write_new(E / 'ARTIFACT_SHA256_V19.json', manifest)
    print(json.dumps({'checks': checks, 'timing': report['timing'], 'raw_files_sealed': len(manifest)}))
    return int(not all(checks.values()))


if __name__ == '__main__':
    sys.exit(main())
