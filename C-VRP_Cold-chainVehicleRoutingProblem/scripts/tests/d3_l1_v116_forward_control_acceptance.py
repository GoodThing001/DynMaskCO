#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.16 acceptance for the revised forward supervisor (no F, no real data, no state collection).

Closes the half-finished item: after the per-slot PASS rows are durably written, a failing TRAILING
IDENTITY record must make the PARENT revoke the whole round's qualification.

Scenarios, one attempt per directory (fixed in advance, never repeated on failure):
  S1 64 durable rows + valid tail + control ok            -> ROUND_PASS (real contained subprocess)
  S2 64 durable rows, tail DELETED                        -> ROUND_REVOKED, all 64 downgraded
  S3 64 durable rows, tail CORRUPT                        -> ROUND_REVOKED
  S4 64 durable rows, end identity MISMATCH               -> control HARD_STOP_IDENTITY_END + REVOKED
  S5 63 durable rows + valid tail                         -> ROUND_REVOKED (INCOMPLETE_ROWS)
  S6 blocked slot -> real deadline kill                   -> terminated/not-attempted statuses + REVOKED
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/coldchain'):
    sys.path.insert(0, str(EXT / p))
sys.path.insert(0, str(HERE))

import d3_l1_v116_forward_control as C                                    # noqa: E402

WORKER = r'''
import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[1]); mode = sys.argv[2]
(out / 'phases').mkdir(parents=True, exist_ok=True)
(out / 'scores').mkdir(parents=True, exist_ok=True)
seq = 0
def phase(stage, **kw):
    global seq
    rec = {'sequence': seq, 'mono': time.monotonic(), 'stage': stage, **kw}
    (out / 'phases' / ('phase_%06d.json' % seq)).write_text(json.dumps(rec))
    seq += 1
(out / 'READY.json').write_text('{"ready": true}')
if mode == 'block_slot':
    phase('slot', day_id=0, position=1)
    time.sleep(30)
    sys.exit(0)
n_rows = 63 if mode == 'rows_63' else 64
for day in range(16):
    for pos in range(1, 5):
        if len(list((out / 'scores').glob('*.json'))) >= n_rows:
            break
        phase('slot', day_id=day, position=pos)
        payload = {'day_id': day, 'position': pos, 'value': day * 4 + pos}
        rec = dict(payload, payload_sha=__import__('hashlib').sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest())
        (out / 'scores' / ('day%02d_pos%d.json' % (day, pos))).write_text(
            json.dumps(rec, sort_keys=True))
        phase('slot_finished', day_id=day, position=pos)
phase('completed')
if mode == 'mutate_watched':
    wp = os.environ.get('V116_WATCHED')
    if wp:
        Path(wp).write_text('mutated-by-worker\n')   # 64 rows already durable; END identity must fail
'''


def verify_record(record, out):
    payload = {k: v for k, v in record.items() if k != 'payload_sha'}
    want = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    ok = record.get('payload_sha') == want and isinstance(record.get('day_id'), int) \
        and isinstance(record.get('position'), int)
    return ok, {'payload_sha_ok': record.get('payload_sha') == want}


def scenario(root, name, mode, *, destroy_tail=None, watched=None, slot_cap=10.0):
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    worker = root / 'synthetic_worker.py'
    if not worker.is_file():
        worker.write_text(WORKER, encoding='utf-8')
    watched_file = root / 'WATCHED_IDENTITY.txt'
    if not watched_file.is_file():
        watched_file.write_text('identity-v116\n', encoding='utf-8')
    identity = {str(watched_file): C.sha(watched_file)}
    env = dict(os.environ)
    if watched == 'mutate':
        # start identity must PASS: the worker mutates the watched file while it runs, so only the
        # END identity fails (this is the HARD_STOP_IDENTITY_END path, not a start-identity stop)
        env['V116_WATCHED'] = str(watched_file)
    control = C.run_contained([sys.executable, str(worker), str(out), mode], out, identity,
                              init_cap=30., slot_cap=slot_cap, batch_cap=120., env=env)
    rows = C.reconcile_rows(out, control, verify_record=verify_record)
    if destroy_tail == 'delete':
        (out / 'CONTROL_IDENTITY_END.json').unlink(missing_ok=True)
    elif destroy_tail == 'corrupt':
        (out / 'CONTROL_IDENTITY_END.json').write_text('{not json', encoding='utf-8')
    report, rows = C.round_verdict(out, control, rows, expected_rows=64)
    return {'name': name, 'control_ok': control.get('ok'), 'control_reason': control.get('reason'),
            'rows_provisional': sum(1 for r in rows if r.get('qualified')),
            'round': report, 'statuses': sorted({r.get('status') for r in rows}),
            'n_qualified': sum(1 for r in rows if r.get('qualified')),
            'round_files': sorted(p.name for p in out.glob('ROUND_*.json'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='v116_'))
    acc = []
    try:
        s1 = scenario(root, 'S1_pass', 'rows_64')
        acc.append(('S1 full 64 rows + valid tail -> ROUND_PASS',
                    s1['round']['status'] == 'ROUND_PASS' and s1['n_qualified'] == 64
                    and s1['control_ok'] and 'ROUND_PASS.json' in s1['round_files'], s1))
        s2 = scenario(root, 'S2_tail_deleted', 'rows_64', destroy_tail='delete')
        acc.append(('S2 64 durable rows but tail DELETED -> ROUND_REVOKED and all rows downgraded',
                    s2['round']['status'] == 'ROUND_REVOKED' and s2['n_qualified'] == 0
                    and s2['round']['rows_qualified_provisionally'] == 64
                    and s2['round']['reason'] == 'TAIL_IDENTITY_MISSING_OR_INVALID', s2))
        s3 = scenario(root, 'S3_tail_corrupt', 'rows_64', destroy_tail='corrupt')
        acc.append(('S3 64 durable rows but tail CORRUPT -> ROUND_REVOKED',
                    s3['round']['status'] == 'ROUND_REVOKED' and s3['n_qualified'] == 0, s3))
        s4 = scenario(root, 'S4_end_identity', 'mutate_watched', watched='mutate')
        acc.append(('S4 end identity mismatch -> control HARD_STOP_IDENTITY_END and round REVOKED',
                    s4['control_reason'] == 'HARD_STOP_IDENTITY_END' and s4['control_ok'] is False
                    and s4['round']['status'] == 'ROUND_REVOKED', s4))
        s5 = scenario(root, 'S5_rows_63', 'rows_63')
        acc.append(('S5 only 63 durable rows -> ROUND_REVOKED (INCOMPLETE_ROWS)',
                    s5['round']['status'] == 'ROUND_REVOKED'
                    and s5['round']['reason'] == 'INCOMPLETE_ROWS', s5))
        s6 = scenario(root, 'S6_blocked_slot', 'block_slot', slot_cap=2.0)
        acc.append(('S6 a blocked slot is killed by the real supervisor deadline and the round is '
                    'REVOKED with terminated/not-attempted statuses',
                    s6['control_reason'] == 'slot_span_deadline' and s6['round']['status']
                    == 'ROUND_REVOKED'
                    and any('TERMINATED' in (x or '') for x in s6['statuses']), s6))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for _, ok, _ in acc if not ok)
    payload = {'rows': [{'name': n, 'ok': bool(ok), 'detail': d} for n, ok, d in acc],
               'n_rows': len(acc), 'n_pass': len(acc) - n_fail, 'n_fail': n_fail,
               'tool_sha256': C.sha(__file__), 'control_sha256':
                   C.sha(EXT / 'scripts/tests/d3_l1_v116_forward_control.py')}
    (out / 'A1_D3_L1_V116_FORWARD_CONTROL_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    for n, ok, d in acc:
        print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', n,
                                 json.dumps({'status': d['round']['status'],
                                             'n_qualified': d['n_qualified'],
                                             'reason': d['round']['reason'],
                                             'control_reason': d['control_reason']},
                                            default=repr, sort_keys=True)[:220]))
    print('\n=== V1.16 acceptance: %d rows, %d PASS, %d FAIL ===' % (len(acc),
                                                                    len(acc) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
