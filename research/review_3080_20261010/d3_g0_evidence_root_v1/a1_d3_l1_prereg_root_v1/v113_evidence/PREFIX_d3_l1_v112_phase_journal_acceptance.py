"""Real subprocess persistence stress and a deterministic Windows open-reader counterexample."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import d3_l1_v112_phase_journal as journal_module
from d3_l1_v112_phase_journal import PhaseJournal, latest

EXT = Path(__file__).resolve().parents[2]
ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
OUT = ROOT / 'v113_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def producer():
    journal = PhaseJournal(OUT / 'phases')
    for i in range(1000):
        journal.publish('slot', day_id=i // 4, position=i % 4 + 1, index=i)


def main():
    OUT.mkdir(exist_ok=False)
    checks, evidence = {}, {}
    dest, temp = OUT / 'legacy.json', OUT / 'legacy.json.tmp'
    dest.write_text('{"old":true}');temp.write_text('{"new":true}')
    with open(dest, 'rb') as reader:
        try:
            os.replace(temp, dest)
            evidence['legacy_replace_result'] = 'SUCCEEDED_ON_THIS_PLATFORM'
        except PermissionError as exc:
            evidence['legacy_replace_result'] = {'type': type(exc).__name__, 'winerror': exc.winerror, 'errno': exc.errno}
    checks['original_open_reader_failure_reproduced_on_windows'] = (
        isinstance(evidence['legacy_replace_result'], dict) if os.name == 'nt' else True)
    start = time.monotonic()
    with open(OUT / 'producer.stdout', 'wb') as stdout, open(OUT / 'producer.stderr', 'wb') as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--producer'], stdout=stdout, stderr=stderr)
        held = [];observed = [];errors = []
        try:
            while process.poll() is None:
                if time.monotonic() - start > 30.:
                    process.kill();process.wait(timeout=5);errors.append('stress_deadline');break
                try:
                    row = latest(OUT / 'phases')
                    if row:
                        observed.append(row['sequence'])
                        files = sorted((OUT / 'phases').glob('phase_*.json'))
                        if files and len(held) < 64:
                            held.append(open(files[-1], 'rb'))  # deliberately retain open readers
                except Exception as exc:
                    errors.append('%s: %s' % (type(exc).__name__, exc))
                time.sleep(.001)
            process.wait(timeout=5)
        finally:
            for fh in held:
                fh.close()
            if process.poll() is None:
                process.kill();process.wait(timeout=5)
    files = sorted((OUT / 'phases').glob('phase_*.json'))
    records = [json.loads(p.read_text()) for p in files]
    checks['real_producer_process_exited_zero'] = process.returncode == 0
    checks['one_thousand_durable_unmodified_records'] = len(records) == 1000 and [r['index'] for r in records] == list(range(1000))
    checks['reader_never_saw_partial_or_invalid_record'] = not errors
    checks['reader_observed_monotonic_latest'] = len(observed) > 1 and observed == sorted(observed)
    checks['held_readers_did_not_block_new_publications'] = len(held) > 0 and process.returncode == 0
    checks['no_stray_temp_files'] = not list((OUT / 'phases').glob('*.tmp'))
    immutable = {p.name: sha(p) for p in files}
    restarted = PhaseJournal(OUT / 'phases');restarted.publish('completed', index=1000)
    checks['producer_restart_does_not_overwrite'] = latest(OUT / 'phases')['sequence'] == 1000 and all(
        sha(OUT / 'phases' / name) == value for name, value in immutable.items())
    checks['reader_ignores_unpublished_partial_temp'] = True
    (OUT / 'phases/incomplete.tmp').write_bytes(b'{')
    checks['reader_ignores_unpublished_partial_temp'] = latest(OUT / 'phases')['sequence'] == 1000
    from unittest.mock import patch
    t_denied = time.monotonic()
    denied_raised = False
    with patch.object(journal_module, '_read_json', side_effect=PermissionError('injected persistent denial')):
        try:
            latest(OUT / 'phases', max_wait_s=.02)
        except PermissionError:
            denied_raised = True
    checks['persistent_read_denial_is_bounded_hard_error'] = denied_raised and time.monotonic() - t_denied < .2
    (OUT / 'phases/phase_999999999_invalid.json').write_bytes(b'{')
    invalid_raised = False
    try:
        latest(OUT / 'phases')
    except json.JSONDecodeError:
        invalid_raised = True
    checks['corrupt_final_record_is_hard_error'] = invalid_raised
    report = {'checks': checks, 'n_pass': sum(checks.values()), 'n_fail': sum(not v for v in checks.values()),
        'evidence': evidence, 'producer_pid': process.pid, 'producer_returncode': process.returncode,
        'reader_errors': errors, 'n_observations': len(observed), 'held_read_handles': len(held),
        'wall_s': time.monotonic() - start, 'journal_source_sha256': sha(Path(__file__).with_name('d3_l1_v112_phase_journal.py')),
        'acceptance_source_sha256': sha(__file__), 'F_forward_calls': 0, 'rollouts': 0,
        'formal_states_consumed': 0, 'science_ready': False,
        'read_diagnostics': journal_module.READ_DIAGNOSTICS,
        'scope': 'persistence primitive only; qualification is conditional on n_fail=0; repaired F harness NOT re-executed'}
    with open(OUT / 'PHASE_JOURNAL_ACCEPTANCE_V113.json', 'x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report))
    return int(report['n_fail'] != 0)


if __name__ == '__main__':
    if '--producer' in sys.argv:
        producer()
    else:
        sys.exit(main())
