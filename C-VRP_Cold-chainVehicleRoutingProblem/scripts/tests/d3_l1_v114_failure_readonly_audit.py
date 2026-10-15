"""Read-only adjudication of a failed F pass and synthetic persistence repair evidence."""
import hashlib
import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/tests', 'scripts/simulation', 'scripts/coldchain'):
    sys.path.insert(0, str(EXT / part))
from d3_l1_v18_readonly_audit import typed_bytes
import numpy as np

ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out=None):
    e = ROOT / 'v111_evidence'
    report = json.loads((e / 'REAL_F_FORWARD_QUALIFICATION_V111.json').read_text())
    source_start = json.loads((e / 'IDENTITY_START.json').read_text())
    input_report = json.loads((ROOT / 'v110_evidence/INPUT_QUALIFICATION_V110.json').read_text())
    checks = []
    def ck(name, ok):
        checks.append({'name': name, 'ok': bool(ok)})
    ck('whole_pass_is_blocked', report['verdict'] == 'REAL_INPUT_F_INTERFACE_BLOCKED' and report['science_ready'] is False)
    ck('all_64_in_original_order', [(r['day_id'], r['position']) for r in report['rows']]
       == [(d, p) for d in range(16) for p in range(1, 5)])
    ck('22_complete_42_uncompleted_no_retry', report['pass'] == 22 and report['failed_or_missing'] == 42
       and report['recorded_F_forward_calls'] == 45)
    ck('real_process_exited_failure_without_descendants', report['returncode'] == 1 and report['tree_active_after'] == 0)
    failure = json.loads((e / 'WORKER_FAILURE.json').read_text())
    ck('failure_not_parity_but_windows_progress_write', failure['type'] == 'PermissionError'
       and 'STAGE.json.tmp' in failure['error'] and 'os.replace' in failure['traceback'])
    ck('missing_worker_origin_end_is_not_claimed_closed', not (e / 'RUNTIME_ORIGINS.json').exists())
    with zipfile.ZipFile(e / 'PREFAILURE_SOURCE_BYTES_V111.zip') as z:
        for name, expected in source_start['source_files'].items():
            ck('frozen_original_source:' + name, hashlib.sha256(z.read(name)).hexdigest() == expected)
    for i, row in enumerate(report['rows']):
        if row['status'] != 'FORWARD_CHECKS_PASS':
            ck('uncompleted_%d_no_score_evidence' % i, 'raw_path' not in row)
            continue
        prefix = 'completed_%d:' % i
        raw = e / row['raw_path']
        ck(prefix + 'raw_sha256', sha(raw) == row['raw_sha256'])
        data = pickle.loads(raw.read_bytes())
        a, b = data['scores_first'], data['scores_repeat']
        source = input_report['rows'][i]
        ck(prefix + 'exact_repeated_scores', typed_bytes(a) == typed_bytes(b))
        ck(prefix + 'finite_and_shape', np.isfinite(a).all() and a.shape == tuple(row['scores_shape'])
           == (source['n_legal'],))
        ck(prefix + 'exact_float_payloads', a.dtype.str == row['scores_dtype']
           and a.tobytes().hex() == row['scores_first_hex'] == row['scores_repeat_hex'])
        ck(prefix + 'common_input_and_ordered_legal_bound', row['input_sha256'] == data['input_sha256'] == source['input_sha256']
           and row['legal_sha256'] == data['legal_sha256'] == source['legal_sha256'])
    phase = json.loads((ROOT / 'v114_evidence/PHASE_JOURNAL_ACCEPTANCE_V114.json').read_text())
    ck('phase_primitive_11_pass_but_no_F_reexecution', phase['n_fail'] == 0 and phase['n_pass'] == 11
       and phase['F_forward_calls'] == 0 and phase['science_ready'] is False)
    for v in (112, 113):
        old = ROOT / ('v%d_evidence' % v)
        records = list(old.glob('PHASE_JOURNAL_ACCEPTANCE*.json'))
        ck('old_failed_candidate_%d_preserved' % v, len(records) == 1 and json.loads(records[0].read_text())['n_fail'] == 1)
    records = [json.loads(p.read_text()) for p in sorted((ROOT / 'v114_evidence/phases').glob('phase_*.json'))
               if 'invalid' not in p.name]
    ck('actual_phase_records_immutable_sequence', len(records) == 1002
       and [r['sequence'] for r in records] == list(range(1002)))
    adjudication = {'checks': checks, 'n_pass': sum(r['ok'] for r in checks),
        'n_fail': sum(not r['ok'] for r in checks), 'classification': 'independent_audit_confirms_failed_F_pass',
        'F_qualification': 'BLOCKED_22_OF_64_COMPLETE', 'repair_primitive': '11_PASS_0_FAIL_ONLY',
        'repaired_F_harness_executed': False, 'new_F_forward_calls': 0, 'stable_online_gain': False,
        'source_sha256': sha(__file__), 'failed_original_report_sha256': sha(e / 'REAL_F_FORWARD_QUALIFICATION_V111.json')}
    target = Path(out) if out else ROOT / 'v114_evidence/INDEPENDENT_FAILURE_AUDIT_V114.json'
    with open(target, 'x', encoding='utf-8') as fh:
        json.dump(adjudication, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in adjudication.items() if k != 'checks'}))
    return int(adjudication['n_fail'] != 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='New independent output path; never overwrite archived results')
    sys.exit(main(parser.parse_args().out))
