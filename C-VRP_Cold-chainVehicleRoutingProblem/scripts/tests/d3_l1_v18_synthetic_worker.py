"""Isolated watchdog fixture. No dataset load, simulation, solver or model calls."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'scripts/simulation'))
import d3_l1_collection_runner as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--go', required=True)
    ap.add_argument('--day', type=int, required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--runtime-identity', required=True)
    ap.add_argument('--noise', action='store_true')
    args = ap.parse_args()
    deadline = time.monotonic() + 10
    while not Path(args.go).exists():
        if time.monotonic() > deadline:
            return 3
        time.sleep(.01)
    out = Path(args.outdir)
    ledger = out / 'STATE_LEDGER.jsonl'

    def emit(event, **kw):
        R.append_jsonl(ledger, dict(event=event, day_id=args.day, run_id='synthetic',
                                   synthetic_only=True, **kw))
    emit('FIXTURE_STARTED', pid=os.getpid(), stage=args.stage)
    if args.stage == 'delayed_capture':
        time.sleep(.7)  # longer than slot cap, before capture and before rollout
    if args.stage in ('success', 'long_validation', 'identity_drift'):
        emit('DAY_ROLLOUT_STARTED')
        emit('DAY_ROLLOUT_FINISHED')
        for pos in range(1, 5):
            emit('ATTEMPT_STARTED', position=pos)
            if args.stage == 'long_validation' and pos == 1:
                time.sleep(.8)  # > rollout cap, after rollout has FINISHED
            emit('PUBLISHED', position=pos,
                 slot_span_wall_s=.8 if args.stage == 'long_validation' and pos == 1 else .001)
            emit('SLOT_FINISHED', position=pos,
                 slot_span_wall_s=.8 if args.stage == 'long_validation' and pos == 1 else .001)
        identity = json.loads(Path(args.runtime_identity).read_text())
        from d3_l1_p0c_v18 import write_json
        if args.stage == 'identity_drift':
            identity['runner_sha256'] = '0' * 64
            write_json(R, Path(args.runtime_identity), identity)
            emit('FIXTURE_RUNTIME_IDENTITY_CHANGED')
        write_json(R, out / ('WORKER_IDENTITY_END_DAY%02d.json' % args.day),
                   {'ok': True, 'identity': identity, 'synthetic_completion_marker': True})
        return 0
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    emit('FIXTURE_CHILD_SPAWNED', child_pid=child.pid, pid=os.getpid())
    if args.noise:
        # A pipe-based supervisor would hang once these buffers fill.
        sys.stdout.write('o' * 100000)
        sys.stderr.write('e' * 100000)
        sys.stdout.flush()
        sys.stderr.flush()
    if args.stage != 'batch':
        emit('DAY_ROLLOUT_STARTED')
    if args.stage not in ('before_capture', 'batch'):
        emit('ATTEMPT_STARTED', position=1)
    if args.stage in ('parity', 'publish', 'worker_error'):
        emit('DAY_ROLLOUT_FINISHED')
    if args.stage == 'publish':
        R.atomic_write_bytes(out / 'staged/half.pkl', b'unpublished fixture bytes')
        emit('FIXTURE_WRITE_ENTERED', position=1)
    if args.stage == 'publish_pending_fsync':
        emit('DAY_ROLLOUT_FINISHED')
        emit('PUBLISHED', position=1, slot_span_wall_s=.001)
    if args.stage == 'worker_error':
        return 17  # deliberately leaves a live child: supervisor must still clean it
    time.sleep(30)
    return 99


if __name__ == '__main__':
    sys.exit(main())
