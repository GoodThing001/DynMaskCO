#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.22 REAL-F FORWARD ADJUDICATION: the formal verdict on the single authorised batch.

Reads the executed run directory and produces the adjudication, recomputing every claim from the
artifacts instead of trusting the supervisor's summary:

  A  the round verdict and the per-row qualification
  B  per-row score contract: length equals the ordered legal length, all finite, the two forwards are
     IDENTICAL in dtype/shape/bytes, and the raw artifact reproduces the recorded hex strings
  C  input identity: every row's input digest equals the pre-qualified V1.10 digest and every
     ordered-legal digest matches
  D  source and asset identity: the 34 frozen watch paths still hash to the frozen digests, and the
     run's start/end identity records agree
  E  execution cost against the FROZEN budgets: per-slot spans, initialization, batch wall clock and
     the recorded forward-call total
  F  consumption: exactly one one-shot receipt exists for this run, and the credential file still
     hashes to the digest that was consumed (no post-hoc edit)
  G  what this does NOT establish: no divergence, no utility, no significance, no MaskCO claim

The verdict is ENGINEERING_PASS only if A-F all hold.  G is recorded unconditionally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as AUTH                                       # noqa: E402
import d3_l1_v118_persistence as persistence                                 # noqa: E402

ROWS = []


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--consumption-dir', required=True)
    ap.add_argument('--credential', required=True)
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cons_dir = Path(args.consumption_dir).expanduser().resolve()
    reader = persistence.VerifiedReader(run_dir)

    row_records = []
    for p in sorted((run_dir / 'scores').glob('SLOT_day*_pos*.json')):
        row_records.append(json.loads(p.read_text(encoding='utf-8')))
    fwd = reader.latest('FORWARD_TARGETS') or {}
    targets = {(int(r['day_id']), int(r['position'])): r for r in fwd.get('rows', [])}
    supervision = reader.latest('SUPERVISION_RESULT') or {}

    # ---- A round verdict ---------------------------------------------------------------
    a_ok = (len(row_records) == 64
            and supervision.get('verdict') == 'REAL_F_FORWARD_ENGINEERING_PASS'
            and (supervision.get('round') or {}).get('status') == 'ROUND_PASS'
            and supervision.get('n_rows_qualified') == 64
            and all(r.get('status') == 'FORWARD_CHECKS_PASS' for r in row_records))
    ck('A the round PASSED: 64 row records, 64 qualified, ROUND_PASS, verdict '
       'REAL_F_FORWARD_ENGINEERING_PASS', a_ok,
       {'n_rows': len(row_records), 'verdict': supervision.get('verdict'),
        'round': (supervision.get('round') or {}).get('status'),
        'n_rows_qualified': supervision.get('n_rows_qualified')})

    # ---- B per-row score contract ------------------------------------------------------
    b_fail = []
    for r in row_records:
        t = targets.get((r['day_id'], r['position'])) or {}
        first, repeat = r['scores_first_hex'], r['scores_repeat_hex']
        raw_path = run_dir / r['raw_relpath']
        try:
            blob = pickle.loads(raw_path.read_bytes())
            replay = (bytes(blob['scores_first']).hex() == first
                      and bytes(blob['scores_repeat']).hex() == repeat
                      and blob.get('input_sha256') == r['input_sha256'])
        except Exception:                                                  # noqa: BLE001
            replay = False
        problems = []
        if not (r['scores_shape'] and len(r['scores_shape']) == 1
                and r['scores_shape'][0] == t.get('n_legal_expected')):
            problems.append('score length != ordered legal length')
        if first != repeat:
            problems.append('forwards not byte-identical')
        if r['scores_dtype'] != '<f4':
            problems.append('dtype %s' % r['scores_dtype'])
        if not replay:
            problems.append('raw artifact does not reproduce the record')
        if r.get('F_calls') != 2:
            problems.append('F_calls %r' % r.get('F_calls'))
        if not all(r.get('checks', {}).values()):
            problems.append('a worker check failed')
        if problems:
            b_fail.append({'row': [r['day_id'], r['position']], 'problems': problems})
    ck('B every row satisfies the score contract: length == ordered legal length, all finite, the two '
       'forwards byte-identical, and the raw artifact reproduces the record', not b_fail,
       {'n_checked': len(row_records), 'n_failed': len(b_fail), 'failures': b_fail[:5],
        'note': 'finiteness is asserted by the worker and re-checked by the main verifier; here the '
                'length/dtype/repeat/replay contract is recomputed from the raw artifacts'})

    # ---- C input identity ----------------------------------------------------------------
    c_fail = []
    for r in row_records:
        t = targets.get((r['day_id'], r['position'])) or {}
        if r['input_sha256'] != t.get('input_sha256'):
            c_fail.append({'row': [r['day_id'], r['position']], 'field': 'input_sha256'})
        if r['legal_sha256'] != t.get('legal_sha256'):
            c_fail.append({'row': [r['day_id'], r['position']], 'field': 'legal_sha256'})
        if r.get('state_hash') != t.get('state_hash'):
            c_fail.append({'row': [r['day_id'], r['position']], 'field': 'state_hash'})
    ck('C every row was scored on the PRE-QUALIFIED input: input digest, ordered-legal digest and '
       'state hash all equal the frozen target', not c_fail,
       {'n_checked': len(row_records), 'n_failed': len(c_fail), 'failures': c_fail[:5]})

    # ---- D source and asset identity ------------------------------------------------------
    idt = reader.latest('IDENTITY_TARGETS') or {}
    drift = [p for p, want in (idt.get('paths') or {}).items() if sha(p) != want]
    # The worker publishes its start/end identity records into the PHASE JOURNAL directory, so a
    # reader bound to the run root finds none of them and would silently compare empty dicts.
    phase_reader = persistence.VerifiedReader(run_dir / 'phases')
    start_rec = phase_reader.latest('identity_start') or {}
    end_rec = phase_reader.latest('identity_end') or {}
    d_ok = (not drift and len(idt.get('paths') or {}) == 34
            and start_rec.get('paths') == end_rec.get('paths')
            and end_rec.get('paths') == {p: sha(p) for p in (idt.get('paths') or {})})
    ck('D all 34 frozen watch paths still hash to their frozen digests, and the worker\'s start/end '
       'identity records agree with each other and with the freeze', d_ok,
       {'n_paths': len(idt.get('paths') or {}), 'n_drifted': len(drift), 'drifted': drift[:5]})

    # ---- E execution cost against the FROZEN budgets --------------------------------------
    ident = reader.latest('RUN_IDENTITY') or {}
    caps = (ident.get('budgets') or {}).get('caps') or {}
    spans = [r['span_wall_s'] for r in row_records]
    warm = reader.latest('WARMUP') or {}
    work = reader.latest('WORKER_RESULT') or {}
    ctrl = supervision.get('control') or {}
    e_detail = {'caps': caps, 'max_span_s': max(spans), 'median_span_s': sorted(spans)[len(spans) // 2],
                'worker_wall_s': work.get('worker_wall_s'),
                'init_and_compile_s': warm.get('wall_s'),
                'supervisor_wall_s': ctrl.get('worker_control_wall_s'),
                'F_calls_recorded': sum(r.get('F_calls', 0) for r in row_records) + (1 if warm else 0)}
    e_ok = (caps == {'initialize_compile_s': 60.0, 'slot_span_s': 10.0, 'batch_s': 180.0}
            and max(spans) <= caps['slot_span_s']
            and (warm.get('wall_s') or 0) <= caps['initialize_compile_s']
            and (work.get('worker_wall_s') or 0) <= caps['batch_s']
            and e_detail['F_calls_recorded'] <= 129)
    ck('E the batch stayed inside the FROZEN budgets and used at most 129 forwards', e_ok, e_detail)

    # ---- F consumption ---------------------------------------------------------------------
    receipts = []
    for p in sorted(cons_dir.glob('authorization_consumption_*.json')):
        rec = json.loads(p.read_text(encoding='utf-8'))
        rec['path'] = str(p)          # the record does not carry its own path; the caller does
        receipts.append(rec)
    mine = [r for r in receipts
            if r.get('run_id') == (ident.get('run_id'))]
    credential = AUTH.load_credential(Path(args.credential).expanduser().resolve())
    f_ok = (len(mine) == 1 and mine[0].get('one_shot') is True
            and AUTH.content_digest(credential) == mine[0].get('credential_content_sha256')
            and AUTH.content_digest(credential) == mine[0].get('credential_content_sha256'))
    ck('F exactly ONE one-shot consumption receipt exists for this run, and the credential file '
       'still hashes to the digest that was consumed', f_ok,
       {'n_receipts_total': len(receipts), 'n_for_this_run': len(mine),
        'receipt': {k: mine[0].get(k) for k in ('consumed_utc', 'consumer_pid', 'approved_by',
                                                'credential_content_sha256')} if mine else None,
        'credential_content_sha256_now': AUTH.content_digest(credential)})

    g = {'what_this_does_NOT_establish': [
        'any F-vs-D-Geo action divergence (L1 is a separate, independent authorisation)',
        'any decision-quality or online-utility gain',
        'any significance, learning contribution or MaskCO claim',
        'that the historical training backbone is the current file (label unchanged)'],
        'labels': ident.get('labels'),
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none'}

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_v122_real_f_forward_adjudication',
        'run_id': ident.get('run_id'),
        'verdict': ('REAL_F_FORWARD_ENGINEERING_PASS_ADJUDICATED' if not fails
                    else 'REAL_F_FORWARD_ADJUDICATION_REJECTED'),
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'execution_summary': e_detail,
        'consumption': {'n_receipts_for_run': len(mine),
                        'receipt_path': mine[0]['path'] if mine else None},
        'claim_limits': g,
        'opportunity_consumed': True,
        'no_rerun_policy': 'PASS/FAIL/timeout/kill consume the single opportunity; no re-run, no '
                           'filling missing states, never splicing the 42 missing V1.11 records',
        'tool_sha256': sha(Path(__file__)),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== V1.22 real-F forward adjudication: %d rows, %d PASS, %d FAIL | %s ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails), payload['verdict']))
    print(json.dumps(e_detail, indent=1, default=repr))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
