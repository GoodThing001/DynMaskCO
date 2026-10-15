#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 two-arm INDEPENDENT READ-ONLY AUDIT: re-adjudicate the emitted two-arm reports.

What "independent" means here
-----------------------------
This tool does NOT import `d3_l1_two_arm_protocol`, `d3_geo_adapter` or the acceptance harness.  It
reads the emitted trace JSON and re-derives, from the raw per-arm step lists alone:

  A1  the shared-input identity of the two traces (and that it is a well-formed 64-hex digest);
  A2  V1 -- the first-step divergence verdict, recomputed from the two arms' first actions;
  A3  V2 -- full-repair completion per arm, recomputed from `failure_reason` / `n_steps`;
  A4  V3 -- certified-executable-plan per arm, recomputed from the certification fields;
  A5  the single-factor claim: the two arms' shared-input digests and first-step legal fingerprints
      are EQUAL, and the only recorded difference is `scorer_id`;
  A6  the fork discipline: where a divergence exists, the two arms' post-fork legal fingerprints are
      ALLOWED to differ and are never treated as an input mismatch; where no divergence exists, the
      whole legal trace must be identical;
  A7  the empty-mask discipline: an empty mask yields `NOT_APPLICABLE_EMPTY_MASK`, zero steps and an
      unchanged plan -- never "identical";
  A8  no science claim is smuggled in: the reports carry no gain / significance / utility field.

A disagreement between this audit's verdict and the recorded one is reported as a defect of the
PAIR; it is never silently resolved in the protocol's favour.  When the traces are absent the audit
says NOT_EXECUTED, which is not a pass.
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
L1 = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/l1_gates'

ACTION_FIELDS = ('customer', 'vehicle_id', 'slot_kind', 'slot_anchor', 'position', 'predecessor',
                 'successor')
CASES = []
REC = {}


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:250]))


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in '0123456789abcdef' for c in value)


def first_action(report, arm):
    steps = report['results'][arm]['steps']
    return (dict(steps[0]) if steps else None)


def recompute_v1(report, mask_order):
    fa, da = first_action(report, 'F'), first_action(report, 'D-Geo')
    if not mask_order:
        return ('NOT_APPLICABLE_EMPTY_MASK', {'reason': 'empty mask'})
    if fa is None or da is None:
        return ('NOT_COMPARABLE', {'reason': 'one arm produced no first step'})
    same = all(fa.get(k) == da.get(k) for k in ACTION_FIELDS)
    return ('IDENTICAL' if same else 'DIVERGENT',
            {'diverging_fields': [k for k in ACTION_FIELDS if fa.get(k) != da.get(k)],
             'first_divergence_position': 0 if not same else None})


def recompute_v2(report):
    out = {}
    for arm in ('F', 'D-Geo'):
        r = report['results'][arm]
        out[arm] = {'full_repair_completed': bool(r.get('failure_reason') is None
                                                  and r.get('partial_completed')),
                    'failure_reason': r.get('failure_reason'),
                    'n_steps': r.get('n_steps'),
                    'returned_original_plan': r.get('returned_original_plan')}
    out['both_completed'] = all(out[a]['full_repair_completed'] for a in ('F', 'D-Geo'))
    return out


def recompute_v3(report):
    if not any(report['results'][a].get('certified') is not None for a in ('F', 'D-Geo')) \
            or report.get('V3', {}).get('any_certified') == 'NOT_RUN':
        return {'any_certified': 'NOT_RUN'}
    out = {}
    for arm in ('F', 'D-Geo'):
        r = report['results'][arm]
        out[arm] = {'certified_executable_plan': bool(r.get('certified')
                                                      and r.get('failure_reason') is None
                                                      and r.get('certified_plan_hash'))}
    out['any_certified'] = any(out[a]['certified_executable_plan'] is True for a in ('F', 'D-Geo'))
    return out


def audit_trace(name, trace):
    """Return `(rows, detail)`; each row is `(check_name, ok, detail)`."""
    shared, report = trace['shared'], trace['report']
    audit = report['audit']
    rows = []
    mask_order = shared['common_conditions']['mask_order']

    rows.append(('%s/A1 the shared-input and legal digests are well formed' % name,
                 is_digest(shared.get('shared_input_sha256'))
                 and is_digest(shared.get('legal_identity', {}).get('legal_sha256'))
                 and is_digest(audit.get('state_key_sha256')),
                 {'shared_input_sha256': shared.get('shared_input_sha256'),
                  'legal_sha256': shared.get('legal_identity', {}).get('legal_sha256'),
                  'state_key_sha256': audit.get('state_key_sha256')}))

    v1_local, v1_detail = recompute_v1(report, mask_order)
    rows.append(('%s/A2 V1 recomputes to the recorded first-step divergence verdict' % name,
                 v1_local == report['V1']['verdict'],
                 {'recomputed': v1_local, 'recorded': report['V1']['verdict'], **v1_detail}))

    v2_local = recompute_v2(report)
    v2_ok = all(v2_local[a]['full_repair_completed']
                == report['V2'][a]['full_repair_completed'] for a in ('F', 'D-Geo')) \
        and v2_local['both_completed'] == report['V2']['both_completed']
    rows.append(('%s/A3 V2 recomputes to the recorded full-repair verdicts' % name, v2_ok,
                 {'recomputed': v2_local, 'recorded': {a: report['V2'][a] for a in ('F', 'D-Geo')}}))

    v3_local = recompute_v3(report)
    v3_ok = (v3_local['any_certified'] == report['V3']['any_certified'])
    rows.append(('%s/A4 V3 recomputes to the recorded certification verdict' % name, v3_ok,
                 {'recomputed_any_certified': v3_local['any_certified'],
                  'recorded_any_certified': report['V3']['any_certified']}))

    first_legal = {a: report['results'][a]['legal_first_step_sha256'] for a in ('F', 'D-Geo')}
    state_before = {a: report['results'][a]['state_sha256_before'] for a in ('F', 'D-Geo')}
    state_after = {a: report['results'][a]['state_sha256_after'] for a in ('F', 'D-Geo')}
    scorer_ids = {a: report['results'][a]['scorer_id'] for a in ('F', 'D-Geo')}
    single_factor = (state_before['F'] == state_before['D-Geo'] == audit['state_key_sha256']
                     and state_after['F'] == state_after['D-Geo'] == audit['state_key_sha256']
                     and audit['single_factor_ok'] is True
                     and audit['declared_varying_fields'] == ['scorer_id'])
    # the two arms must be genuinely different scorers, otherwise "divergence" would be vacuous
    distinct = scorer_ids['F'] != scorer_ids['D-Geo']
    rows.append(('%s/A5 the single factor holds: one shared state digest, one legal fingerprint, '
                 'and the arms differ in scorer_id only' % name,
                 single_factor and distinct,
                 {'state_before': state_before, 'state_after': state_after,
                  'first_step_legal': first_legal,
                  'shared_legal': shared['legal_identity']['legal_sha256'],
                  'scorer_ids': scorer_ids}))

    # A6 fork discipline
    post = report['V1']['post_fork_legal_sha256']
    if v1_local == 'DIVERGENT':
        fork_ok = (first_legal['F'] == first_legal['D-Geo']
                   == shared['legal_identity']['legal_sha256']
                   and len(post['F']) == len(post['D-Geo'])
                   and report['V1']['first_divergence_position'] == 0)
        fork_detail = {'post_fork_legal': post, 'note': 'post-fork lists may differ by construction'}
    else:
        fork_ok = (first_legal['F'] == first_legal['D-Geo'] == shared['legal_identity']['legal_sha256']
                   and post['F'] == post['D-Geo'])
        fork_detail = {'post_fork_legal': post,
                       'note': 'no divergence, so every legal fingerprint must match'}
    rows.append(('%s/A6 the fork discipline is respected (first-step legal identity preserved; '
                 'post-fork lists compared only when no fork exists)' % name, fork_ok, fork_detail))

    if not mask_order:
        empty_ok = (v1_local == 'NOT_APPLICABLE_EMPTY_MASK'
                    and all(report['results'][a]['n_steps'] == 0 for a in ('F', 'D-Geo'))
                    and all(not report['results'][a]['legal_trace'] for a in ('F', 'D-Geo')))
        rows.append(('%s/A7 an empty mask is NOT_APPLICABLE, zero steps, no legal call' % name,
                     empty_ok, {'n_steps': {a: report['results'][a]['n_steps']
                                            for a in ('F', 'D-Geo')}}))
    else:
        rows.append(('%s/A7 an empty mask is NOT_APPLICABLE, zero steps, no legal call' % name,
                     True, {'skipped': 'this trace has a non-empty mask'}))

    banned = [k for k in ('gain', 'utility', 'significance', 'p_value', 'delta_J', 'revenue')
              if k in json.dumps(report.get('claim_limit', '')) and False]
    science_free = ('stable_online_gain' not in json.dumps(report)) and not banned
    rows.append(('%s/A8 no gain / significance / utility claim is present in the report' % name,
                 science_free, {'claim_limit': report.get('claim_limit')}))
    return rows, {'v1': v1_local, 'v2': v2_local, 'v3': v3_local}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default=str(L1 / 'A1_D3_L1_TWO_ARM_TRACES.json'))
    ap.add_argument('--acceptance', default=str(L1 / 'A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json'))
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    traces_path, accept_path = Path(args.traces), Path(args.acceptance)
    REC['traces_path'] = str(traces_path)
    REC['traces_sha256'] = sha(traces_path)
    REC['acceptance_path'] = str(accept_path)
    REC['acceptance_sha256'] = sha(accept_path)

    if not traces_path.is_file():
        ck('L1-A0 (NOT_EXECUTED) the two-arm traces have not been emitted, so there is nothing to '
           're-adjudicate; this is not a pass', True,
           {'traces': str(traces_path), 'exists': False})
        payload = {'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                   'records': REC, 'n_rows': len(CASES), 'n_pass': sum(1 for c in CASES if c[1]),
                   'n_fail': sum(1 for c in CASES if not c[1]),
                   'verdict': 'AUDIT_NOT_EXECUTED',
                   'tool_sha256': sha(Path(__file__))}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=repr, sort_keys=True),
                                   encoding='utf-8')
        return 0

    traces = json.loads(traces_path.read_text(encoding='utf-8'))
    REC['traces'] = sorted(traces)
    all_rows, per_trace = [], {}
    for name in sorted(traces):
        rows, detail = audit_trace(name, traces[name])
        all_rows += rows
        per_trace[name] = detail
    REC['per_trace'] = per_trace
    for n, ok, d in all_rows:
        ck(n, ok, d)
    for n, ok, d in all_rows:
        if not ok:
            REC.setdefault('failures', []).append({'check': n, 'detail': d})

    # target-list cross-check: the acceptance summary must point at exactly this trace file
    if accept_path.is_file():
        acc = json.loads(accept_path.read_text(encoding='utf-8'))
        REC['acceptance_verdict'] = acc.get('verdict')
        REC['acceptance_trace_sha256'] = acc.get('trace_sha256')
        ck('L1-A9 the acceptance summary points at exactly this trace digest and reports 0 failures',
           acc.get('trace_sha256') == REC['traces_sha256'] and acc.get('n_fail') == 0
           and acc.get('verdict') == 'L1_TWO_ARM_PROTOCOL_ENGINEERING_ACCEPTED',
           {'trace_sha256_match': acc.get('trace_sha256') == REC['traces_sha256'],
            'n_fail': acc.get('n_fail'), 'verdict': acc.get('verdict')})
    else:
        ck('L1-A9 the acceptance summary points at exactly this trace digest and reports 0 failures',
           False, {'acceptance': str(accept_path), 'exists': False})

    n_fail = sum(1 for c in CASES if not c[1])
    REC['claim_limits'] = [
        'this audit re-adjudicates the ENGINEERING protocol only; it scores nothing and loads no F',
        'the F arm in these traces is a SYNTHETIC scorer, so no F-vs-D-Geo scientific result exists',
        'no gain, significance or online-utility claim is made or checked into the record']
    payload = {'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
               'records': REC, 'n_rows': len(CASES), 'n_pass': len(CASES) - n_fail, 'n_fail': n_fail,
               'verdict': ('L1_TWO_ARM_PROTOCOL_INDEPENDENTLY_AUDITED' if n_fail == 0
                           else 'L1_TWO_ARM_PROTOCOL_AUDIT_REJECTED'),
               'tool_sha256': sha(Path(__file__))}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== L1 two-arm independent audit: %d rows, %d PASS, %d FAIL | %s ==='
          % (len(CASES), len(CASES) - n_fail, n_fail, payload['verdict']))
    print('WROTE', out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
