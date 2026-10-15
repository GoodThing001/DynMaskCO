"""Real subprocess/tree termination through the SAME production supervisor loop."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'scripts/simulation'))
import d3_l1_collection_runner as R
import d3_l1_p0c_v18 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args0 = ap.parse_args()
    root = Path(args0.outdir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    # Frozen test plan: one run each, no result-dependent case selection or retries.
    stages = ('before_capture', 'capture', 'parity', 'publish', 'worker_error', 'batch',
              'delayed_capture', 'publish_pending_fsync')
    for stage in stages:
        out = root / stage
        out.mkdir()
        args = SimpleNamespace(outdir=str(out), dataset=R.DEFAULT_DATASET,
            identity=R.DEFAULT_IDENTITY, runtime_identity=str(out / 'RUNTIME_IDENTITY_V18.json'),
            purpose='synthetic', run_id='v18-kill-' + stage, days=2, day_id=0,
            slot_cap_s=.40, day_cap_s=120., day_rollout_cap_s=.60,
            batch_cap_s=3.0 if stage == 'batch' else 15., fault='none', inject_block_s=0.,
            expect_dataset_sha256=R.DEV_CHECK_SHA)
        # Explicit prior registration. Production never generates this automatically.
        identity = V.execution_identity(R, args)
        V.write_json(R, args.runtime_identity, identity)

        def command(day, go):
            return [sys.executable, str(EXT / 'scripts/tests/d3_l1_v18_synthetic_worker.py'),
                    '--outdir', str(out), '--go', str(go), '--day', str(day),
                    '--stage', stage if day == 0 else 'success', '--runtime-identity',
                    args.runtime_identity, '--noise']

        started = time.monotonic()
        rc = V.supervise(R, args, synthetic_command=command)
        elapsed = time.monotonic() - started
        result = json.loads((out / 'SUPERVISOR_RESULT.json').read_text())
        ledger = R.read_ledger(out / 'STATE_LEDGER.jsonl')
        first = result['days'][0]
        expected = ('rollout_deadline_killed' if stage == 'before_capture' else
                    'batch_deadline_killed' if stage == 'batch' else
                    'worker_error' if stage == 'worker_error' else 'slot_deadline_killed')
        cleanup = first.get('cleanup') or {}
        day0 = [r for r in ledger if r.get('day_id') == 0]
        final = [R.slot_state(day0, 0, pos)[0] for pos in range(1, 5)]
        wanted = (['NOT_ATTEMPTED_DUE_TO_TERMINATION'] * 4 if stage in ('before_capture', 'batch')
                  else ['FAILED'] + ['NOT_ATTEMPTED_DUE_TO_TERMINATION'] * 3)
        # Restart with a different run_id cannot capture either terminated day again.
        before = sum(r.get('event') == 'FIXTURE_STARTED' for r in ledger)
        retry = copy.copy(args)
        retry.run_id += '-new-run-id'
        retry_rc = V.supervise(R, retry, synthetic_command=command)
        after = sum(r.get('event') == 'FIXTURE_STARTED' for r in R.read_ledger(out / 'STATE_LEDGER.jsonl'))
        checks = {'real_kill_and_reap': cleanup.get('cleanup_ok') is True
                  and cleanup.get('tree_active_before', 0) >= 1
                  and cleanup.get('tree_active_after') == 0,
                  'exit_nonzero': first['returncode'] != 0,
                  'expected_deadline_window': first['status'] == expected,
                  'terminal_accounting': final == wanted,
                  'fixed_denominator': result['slots_in_denominator'] == 64,
                  'no_false_batch_complete': result['verdict'] != 'BATCH_COMPLETE' and rc != 0,
                  'new_run_cannot_recapture': before == after,
                  'next_day_continues_except_batch':
                      (len(result['days']) == 1 if stage == 'batch' else
                       len(result['days']) == 2 and result['days'][1]['status'] == 'completed'),
                  'bounded_test': elapsed < 20.}
        rows.append({'stage': stage, 'checks': checks, 'ok': all(checks.values()),
                     'elapsed': elapsed, 'first_run_result': result,
                     'new_run_id_exit': retry_rc,
                     'terminal_states': final})
        print(stage, checks, flush=True)
    result = {'purpose': 'synthetic_real_process_tree_acceptance', 'formal_slots_collected': 0,
              'production_defaults': V.FROZEN_PARAMETERS,
              'rows': rows, 'n_fail': sum(not row['ok'] for row in rows),
              'argv': sys.argv, 'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    V.write_json(R, root / 'KILL_E2E.json', result)
    return int(result['n_fail'] != 0)


if __name__ == '__main__':
    sys.exit(main())
