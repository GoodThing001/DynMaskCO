#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent read-only audit: the V1.19 run has performed ZERO real F work.

The claim "no real F forward has been executed" must be checkable by someone who does not trust the
runner's own report.  This tool reads the run directory's published artifacts only -- it imports no
runner, no supervisor and no scorer -- and counts the things that only a real forward could leave
behind:

  * slot row records (`SLOT_day*_pos*_*.json`)                      -> must be 0
  * score raw artifacts (`SCORES_*.json`)                           -> must be 0
  * phase-journal records (any `phase_*.json` / `slot` event)       -> must be 0
  * MODEL_LOAD records (a real run loads the checkpoint first)      -> must be 0
  * WARMUP records (a real run compiles/forwards once first)        -> must be 0
  * identity_start / identity_end tail records                      -> must be 0
  * consumption receipts for this run's credential                  -> must be 0
  * WORKER_RESULT / WORKER_FAILURE records                          -> must be 0

and that the artifacts which MUST exist do exist and verify:

  * RUN_IDENTITY (parseable, correct run id)
  * FORWARD_TARGETS with 64 rows bound to the pre-qualified digests
  * IDENTITY_TARGETS whose every path still hashes to its frozen digest (runtime watch closure)
  * a READINESS record whose verdict and `real_F_forward_executed` are read back and re-adjudicated
  * the authorization template, present, with the human approval fields EMPTY and refused by the
    gate (a refusal artifact must be on disk as evidence of the attempt)

Verdict `ZERO_REAL_F_WORK_CONFIRMED` requires ALL of the above.  Anything else is reported with the
specific counter-evidence, because "no evidence of work" and "evidence of no work" are different
claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / _p))

CASES = []
REC = {}

MUST_BE_ABSENT = {
    'slot_row_records': 'SLOT_day*_pos*_*.json',
    'raw_score_artifacts': 'SCORES_*.json',
    'phase_journal_records': 'phase_*.json',
    'model_load_records': 'MODEL_LOAD_*.json',
    'warmup_records': 'WARMUP_*.json',
    'identity_start_records': 'identity_start_*.json',
    'identity_end_records': 'identity_end_*.json',
    'worker_results': 'WORKER_RESULT_*.json',
    'worker_failures': 'WORKER_FAILURE_*.json',
    'first_failure_records': 'FIRST_FAILURE_*.json',
    'supervision_results': 'SUPERVISION_RESULT_*.json',
    'round_records': 'ROUND_*.json',
}


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
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:240]))


def read_records(directory, prefix):
    """Records sorted by the publisher's own monotonic clock, exactly like the production reader.

    Sorting by file name would pick an arbitrary record: the run directory legitimately holds SEVERAL
    immutable READINESS records (one per verification run, because the runner never overwrites a
    record), so "latest" must mean latest by publication time, not by name.
    """
    out = []
    if not Path(directory).is_dir():
        return out
    for p in sorted(Path(directory).glob('%s*.json' % prefix)):
        try:
            record = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:                                          # noqa: BLE001
            record = {'_unreadable': repr(exc)}
        out.append((p.name, record))
    out.sort(key=lambda t: (float(t[1].get('published_mono') or 0.0), t[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--consumption-dir')
    ap.add_argument('--expect-run-id', required=True)
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    REC['run_dir'] = str(run_dir)
    REC['expect_run_id'] = args.expect_run_id

    # ---- Z1 the run directory exists and holds the freeze artifacts -------------------------
    ident = read_records(run_dir, 'RUN_IDENTITY')
    fwd = read_records(run_dir, 'FORWARD_TARGETS')
    idt = read_records(run_dir, 'IDENTITY_TARGETS')
    ck('Z1 the run directory carries exactly one run identity, one forward-target list and one '
       'identity-target list',
       len(ident) == 1 and len(fwd) == 1 and len(idt) == 1 and not ident[0][1].get('_unreadable'),
       {'n_identity': len(ident), 'n_targets': len(fwd), 'n_identity_targets': len(idt)})
    if not (ident and fwd and idt):
        print('FATAL: the run directory is not frozen')
        return 1
    REC['run_id'] = ident[0][1].get('run_id')
    REC['n_target_rows'] = fwd[0][1].get('n_rows')
    ck('Z2 the run identity is the expected run and the target list has 64 pre-qualified rows',
       REC['run_id'] == args.expect_run_id and REC['n_target_rows'] == 64,
       {'run_id': REC['run_id'], 'n_target_rows': REC['n_target_rows'],
        'targets_sha256': fwd[0][1].get('targets_sha256')})

    # ---- Z3 runtime watch closure, recomputed here -----------------------------------------
    watch = {p: {'frozen': want, 'current': sha(p), 'ok': sha(p) == want}
             for p, want in (idt[0][1].get('paths') or {}).items()}
    drifted = sorted(p for p, r in watch.items() if not r['ok'])
    REC['runtime_watch'] = {'n_paths': len(watch), 'n_drifted': len(drifted), 'drifted': drifted[:8]}
    ck('Z3 every path in the frozen runtime watch set still hashes to its frozen digest',
       watch and not drifted, REC['runtime_watch'])

    # ---- Z4 NOTHING a real forward would leave behind --------------------------------------
    counts, present = {}, {}
    for label, pattern in MUST_BE_ABSENT.items():
        prefix = pattern.split('*')[0]
        directory = run_dir / 'scores' if label in ('slot_row_records', 'raw_score_artifacts') \
            else run_dir / 'phases' if label in ('phase_journal_records', 'identity_start_records',
                                                 'identity_end_records') else run_dir
        found = sorted(p.name for p in directory.glob(pattern)) if directory.is_dir() else []
        counts[label] = len(found)
        present[label] = found[:5]
    REC['absent_counts'] = counts
    REC['absent_samples'] = present
    ck('Z4 the run directory contains NO artifact that only a real forward could produce '
       '(slot rows, raw scores, phase records, model load, warm-up, identity tail, supervision)',
       all(v == 0 for v in counts.values()), counts)

    # ---- Z5 no credential was ever consumed for this run -----------------------------------
    cons_dir = Path(args.consumption_dir) if args.consumption_dir else None
    consumed = []
    if cons_dir and cons_dir.is_dir():
        for p in sorted(cons_dir.glob('authorization_consumption_*.json')):
            try:
                rec = json.loads(p.read_text(encoding='utf-8'))
            except Exception:                                             # noqa: BLE001
                rec = {}
            if rec.get('run_id') == args.expect_run_id:
                consumed.append({'path': str(p), 'run_id': rec.get('run_id'),
                                 'consumed_utc': rec.get('consumed_utc')})
    REC['consumption_for_this_run'] = consumed
    ck('Z5 no authorization credential for THIS run id has ever been consumed',
       not consumed, {'consumption_dir': str(cons_dir) if cons_dir else None,
                      'n_for_this_run': len(consumed)})

    # ---- Z6 the authorization template is present and is NOT an approval -------------------
    templates = sorted(run_dir.parent.glob('AUTHORIZATION_TEMPLATE_*.json'))
    tmpl = json.loads(templates[-1].read_text(encoding='utf-8')) if templates else {}
    humans_empty = [f for f in ('approved_by', 'approved_utc', 'approval_statement')
                    if not str(tmpl.get(f) or '').strip()]
    refusals = read_records(run_dir, 'AUTHORIZATION_REFUSED')
    REC['template'] = {'path': str(templates[-1]) if templates else None,
                       'human_fields_empty': humans_empty,
                       'credential_sha256': tmpl.get('credential_sha256')}
    REC['refusal_records'] = [n for n, _ in refusals]
    ck('Z6 the authorization template exists, its human approval fields are EMPTY, and the gate '
       'left a refusal record on disk',
       bool(templates) and len(humans_empty) == 3 and bool(refusals)
       and 'no_human_approval' in json.dumps([r for _n, r in refusals]),
       {'n_refusal_records': len(refusals), 'human_fields_empty': humans_empty})

    # ---- Z7 the readiness verdict, re-adjudicated ------------------------------------------
    readiness = read_records(run_dir, 'READINESS')
    last = readiness[-1][1] if readiness else {}
    checks = last.get('readiness', {}).get('checks', {})
    REC['readiness'] = {'n_records': len(readiness), 'verdict': last.get('verdict'),
                        'real_F_forward_executed': checks.get('real_F_forward_executed'),
                        'missing_before_signature':
                            last.get('readiness', {}).get('missing_before_signature')}
    ck('Z7 the readiness record reports READY_FOR_USER_SIGNATURE while recording that the real F '
       'forward has NOT been executed, and both are consistent with Z4/Z5',
       last.get('verdict') == 'READY_FOR_USER_SIGNATURE'
       and checks.get('real_F_forward_executed') is False
       and not last.get('readiness', {}).get('missing_before_signature')
       and all(v == 0 for v in counts.values()) and not consumed,
       REC['readiness'])

    fails = [c for c in CASES if not c[1]]
    verdict = 'ZERO_REAL_F_WORK_CONFIRMED' if not fails else 'ZERO_F_CLAIM_REJECTED'
    payload = {'schema': 'd3_l1_v119_zero_f_audit',
               'run_id': args.expect_run_id, 'run_dir': str(run_dir),
               'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
               'records': REC, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
               'n_fail': len(fails), 'verdict': verdict,
               'independence': 'this audit imports no runner, no supervisor and no scorer; it '
                               'counts artifacts on disk and re-hashes the frozen watch set',
               'claim_limits': [
                   'this proves no real F work happened in THIS run directory; it does not prove '
                   'anything about any other directory',
                   'a zero-work state is not a pass: it is the state a signature would authorise '
                   'leaving',
                   'the readiness verdict is read back and re-adjudicated, never taken on trust'],
               'tool_sha256': sha(Path(__file__))}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== V1.19 zero-real-F audit: %d rows, %d PASS, %d FAIL | %s ==='
          % (len(CASES), len(CASES) - len(fails), len(fails), verdict))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
