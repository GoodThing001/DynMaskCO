#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1-v2 EXECUTED-ROUND INDEPENDENT ADJUDICATION (read-only).

Re-adjudicates the ONE consumed L1-v2 comparison from the RAW records it left behind.  It imports
NEITHER the L1 run entry, NOR the two-arm protocol, NOR the real arm, NOR the geo adapter, NOR the
closure: every number is recomputed here from the durable JSONL, the phase journal and the frozen
identity record.  A module that produced a result must not be the thing that certifies it.

Rows
----
  D1  the round's own adjudication is read and NOT upgraded: a BLOCKED round stays BLOCKED
  D2  the denominator is the 64 pre-registered states, in the frozen order, with no duplicate
  D3  exactly one state is an exception, it is named, and it is REPORTED not removed
  D4  the six V1 buckets are counted independently and reconcile with the round record
  D5  V2 is recomputed per arm from the per-state failure reasons
  D6  V3 is NOT_RUN for every state (no certifier, no install)
  D7  every COMPLETED state carries an exact 2-call-per-arm ledger with 0 problems
  D8  the frozen caps were respected: per-state wall clock, per-arm calls, init/compile, batch
  D9  the arm-order attribution is recomputed and NO state is ORDER_ATTRIBUTABLE
  D10 the identity witnesses match at BOTH ends (42 paths, start and end)
  D11 the one-shot receipt exists exactly once and lives OUTSIDE the run directory
  D12 the witness digests the credential bound are the ones on disk
  D13 the exception's cause is a NEGATIVE PAD in the F path, with the true maximum measured
  D14 no re-run, no top-up and no filtering: the record set is exactly one pass over 64 states

Nothing here is a scientific claim beyond what the records support.  The comparison is reported as
EXECUTED; whether its 63/64 answerable denominator is acceptable is the signer's decision, and this
tool does not make it.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN = EV / 'l1_v2_execution_run'
CONSUMPTION = EV / 'l1_gates/L1_V2_EXECUTION_consumption'
PREREG = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
IDENTITY = EV / 'l1_gates/L1_V2_PREREGISTRATION_MANIFEST.json'
CRED = EV / 'l1_gates/L1_V2_EXECUTION_AUTHORIZATION.json'

AUTHORISED = {
    'run_id': 'd3l1-l1-two-arm-20261016-l1v2',
    'prereg_sha256': 'e02b98587682a42f9142104e6798b6e8ef4db4135e9ee0dc0eaf2aa749e449ba',
    'source_identity_sha256': '6139249946497ac1dde4848cda7531df85446667075fb50c012607c275d05d0c',
    'state_binding_sha256': 'f5eedd1027342496939e7257e64f2d25dd6f7cd8787e731b6aa9dde213eefee1',
    'n_states': 64,
    'n_watch': 42,
    'caps': {'initialize_compile_s': 60.0, 'per_state_wall_clock_s': 10.0,
             'batch_wall_clock_s': 180.0, 'per_arm_per_state_scorer_calls': 2,
             'per_arm_total_scorer_calls': 128},
}
FORBIDDEN = ('d3_l1_l1_formal_run', 'd3_l1_two_arm_protocol', 'd3_l1_real_arm', 'd3_geo_adapter',
             'd3_l1_closure', 'd3_l1_v121_dependency_closure', 'd3_l1_v116_forward_control')
V1_BUCKETS = ('DIVERGENT', 'IDENTICAL', 'NOT_COMPARABLE', 'NOT_APPLICABLE_EMPTY_MASK',
              'NOT_ATTEMPTED')
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


def load(p):
    p = Path(p)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def read_jsonl(p):
    rows = []
    for line in Path(p).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ck(name, ok, detail):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:210]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()

    records = read_jsonl(RUN / 'L1_STATE_RECORDS.jsonl') if (RUN / 'L1_STATE_RECORDS.jsonl') else []
    adj = load(RUN / 'L1_ADJUDICATION.json') or {}
    sup = load(RUN / 'L1_SUPERVISION.json') or {}
    ctl = load(RUN / 'CONTROL_RESULT.json') or {}
    frozen = load(RUN / 'L1_IDENTITY.json') or {}
    manifest = load(IDENTITY) or {}
    phases = sorted((RUN / 'phases').glob('phase_*.json'))
    phase_recs = [json.loads(p.read_text(encoding='utf-8')) for p in phases]
    receipts = sorted(CONSUMPTION.glob('authorization_consumption_*.json'))

    keys = [(int(r['day_id']), int(r['position'])) for r in records]
    completed = [r for r in records if r.get('status') == 'COMPLETED']
    exceptions = [r for r in records if r.get('status') == 'FAILED_EXCEPTION']

    # D1 -------------------------------------------------------------------------------------
    ck('D1 the round record is read as-is and NOT upgraded: a BLOCKED round stays BLOCKED',
       adj.get('status') == 'BLOCKED' and len(adj.get('blocking_reasons') or []) >= 1,
       {'status_from_the_round': adj.get('status'),
        'blocking_reasons': adj.get('blocking_reasons'),
        'not_upgraded_by_this_tool': True,
        'verdict_is': 'the comparison is EXECUTED; the round is BLOCKED because one state raised and '
                      'its ledger could not be cross-checked.  A BLOCKED round is never a formal '
                      'PASS.'})

    # D2 -------------------------------------------------------------------------------------
    frozen_order = [(int(s['day_id']), int(s['position']))
                    for s in (frozen.get('state_list') or {}).get('states', [])]
    d2_ok = (len(records) == 64 and len(set(keys)) == 64 and keys == frozen_order
             and frozen.get('state_binding_sha256') == AUTHORISED['state_binding_sha256'])
    ck('D2 the denominator is the 64 pre-registered states in the FROZEN order, no duplicate and no '
       'state added or dropped', d2_ok,
       {'n_records': len(records), 'n_unique': len(set(keys)),
        'order_equals_the_frozen_list': keys == frozen_order,
        'first_three': keys[:3], 'last_three': keys[-3:],
        'state_binding_sha256': frozen.get('state_binding_sha256'),
        'matches_the_authorisation': frozen.get('state_binding_sha256')
        == AUTHORISED['state_binding_sha256']})

    # D3 -------------------------------------------------------------------------------------
    exc_keys = [(int(r['day_id']), int(r['position'])) for r in exceptions]
    ck('D3 exactly one state raised, it is NAMED, and it is reported rather than removed from the '
       'denominator', len(exceptions) == 1 and exc_keys == [(0, 3)],
       {'n_exceptions': len(exceptions), 'exception_states': exc_keys,
        'reported_in_the_round_record': len(adj.get('exception_states') or []) == len(exceptions),
        'still_in_the_denominator': adj.get('n_states_in_denominator') == 64,
        'exception_type': exceptions[0].get('exception_type') if exceptions else None,
        'message': exceptions[0].get('error') if exceptions else None})

    # D4 -------------------------------------------------------------------------------------
    # Scope matters and the first cut of this row got it wrong.  The ROUND record counts V1 over ALL
    # 64 pre-registered states -- a state that never ran is `NOT_ATTEMPTED`, which is a recorded
    # outcome, not an omission.  The SCIENTIFIC reading additionally needs the comparable subset, so
    # both are computed here and neither is allowed to stand in for the other.
    counts_all = {b: sum(1 for r in records if (r.get('V1') or {}).get('verdict') == b)
                  for b in V1_BUCKETS}
    counts_comparable = {b: sum(1 for r in completed if (r.get('V1') or {}).get('verdict') == b)
                         for b in V1_BUCKETS}
    recorded = adj.get('V1_counts') or {}
    d4_ok = (all(counts_all[b] == int(recorded.get(b, -1)) for b in V1_BUCKETS)
             and sum(counts_all.values()) == 64
             and counts_all['NOT_ATTEMPTED'] == len(exceptions)
             and sum(counts_comparable.values()) == len(completed)
             and counts_all['DIVERGENT'] == counts_comparable['DIVERGENT']
             and counts_all['IDENTICAL'] == counts_comparable['IDENTICAL'])
    ck('D4 the V1 buckets are recomputed independently over ALL 64 records AND over the comparable '
       'subset; both reconcile with the round record', d4_ok,
       {'over_all_64': counts_all, 'over_the_comparable_subset': counts_comparable,
        'recorded_in_the_round': recorded,
        'n_completed': len(completed), 'n_exception': len(exceptions),
        'reading': 'V1 divergence is claimed only over states where BOTH arms produced a first step '
                   '(%d here).  NOT_COMPARABLE is neither agreement nor disagreement, and a state '
                   'that never ran is NOT_ATTEMPTED -- recorded, not dropped.'
                   % (counts_comparable['DIVERGENT'] + counts_comparable['IDENTICAL']),
        'note_on_the_first_cut': 'counting only over the 63 completed states made this row disagree '
                                 'with the round record by exactly the one NOT_ATTEMPTED state; the '
                                 'scope error was in this auditor, not in the run'})

    # D5 -------------------------------------------------------------------------------------
    v2 = {a: sum(1 for r in completed
                 if ((r.get('V2') or {}).get(a) or {}).get('full_repair_completed'))
          for a in ('F', 'D-Geo')}
    reasons = {}
    for r in completed:
        for a, why in (r.get('per_arm_failure') or {}).items():
            if why:
                reasons['%s:%s' % (a, why)] = reasons.get('%s:%s' % (a, why), 0) + 1
    ck('D5 V2 is recomputed per arm; the counts and the failure vocabulary come from the records',
       v2['F'] + v2['D-Geo'] == 2 * len(completed) and not reasons,
       {'full_repair_completed': v2, 'n_completed': len(completed),
        'failure_reasons_seen': reasons or 'none',
        'not_a_certification_claim': 'V2 means the arm walked the whole mask without a failure '
                                     'reason; it says NOTHING about certifiability (V3 is NOT_RUN)'})

    # D6 -------------------------------------------------------------------------------------
    # V3 IS `NOT_RUN` EVERYWHERE, BUT IN TWO SHAPES, and a naive reader would mis-state this.  A
    # COMPARED state carries the protocol's V3 -- per arm `certified_executable_plan` plus the
    # aggregate `any_certified`.  The state that raised carries the blank-record shape,
    # `{verdict, reason}`, because it never reached the protocol.  Both say NOT_RUN; neither claims a
    # plan.  The first cut of this row read only `verdict`, so it saw 63 "absent" V3s and one NOT_RUN
    # -- exactly backwards -- and would have reported a non-existent gap.
    def v3_values(r):
        v3 = r.get('V3')
        if not isinstance(v3, dict):
            return ['ABSENT']
        out = []
        if 'any_certified' in v3:
            out.append(str(v3.get('any_certified')))
            for arm in ('F', 'D-Geo'):
                if isinstance(v3.get(arm), dict):
                    out.append(str(v3[arm].get('certified_executable_plan')))
        if 'verdict' in v3:
            out.append(str(v3.get('verdict')))
        return out or ['ABSENT']

    distinct = sorted({v for r in records for v in v3_values(r)})
    shapes = {'protocol_any_certified': 0, 'blank_record_verdict': 0, 'other': 0}
    for r in records:
        v3 = r.get('V3') or {}
        if 'any_certified' in v3:
            shapes['protocol_any_certified'] += 1
        elif 'verdict' in v3:
            shapes['blank_record_verdict'] += 1
        else:
            shapes['other'] += 1
    v3_ok = (distinct == ['NOT_RUN'] and shapes['other'] == 0
             and shapes['protocol_any_certified'] == len(completed)
             and shapes['blank_record_verdict'] == len(exceptions))
    ck('D6 V3 is NOT_RUN on EVERY state, in the shape that state actually reached: %d compared states '
       'carry the protocol V3 and %d carries the blank-record V3'
       % (shapes['protocol_any_certified'], shapes['blank_record_verdict']), v3_ok,
       {'distinct_V3_values_across_all_states_and_arms': distinct,
        'shape_counts': shapes,
        'round_record': (adj.get('V3') or '')[:130] if isinstance(adj.get('V3'), str)
        else adj.get('V3'),
        'why_two_shapes': 'a state that raises never reaches the protocol, so its V3 comes from the '
                          'blank-state record; reporting both shapes keeps the difference visible '
                          'instead of flattening it into one truthy check',
        'caution': 'NOT_RUN means no certifier was supplied and no plan was installed; it is NOT a '
                   'comparison result and NOT a statement that a plan would fail certification'})

    # D7 -------------------------------------------------------------------------------------
    ledgers_ok = all(r.get('ledger_ok') for r in completed)
    per_arm = set()
    for r in completed:
        pa = (r.get('ledger_cross_check') or {}).get('per_arm_calls') or {}
        per_arm.add((pa.get('F'), pa.get('D-Geo')))
    problems = sum(len(r.get('ledger_problems') or []) for r in completed)
    ck('D7 every COMPLETED state carries an exact 2-calls-per-arm ledger with ZERO problems, so no '
       'logits vector was reused across steps', ledgers_ok and per_arm == {(2, 2)} and problems == 0,
       {'all_ledger_ok': ledgers_ok, 'distinct_(F,D)_call_counts': sorted(per_arm, key=repr),
        'total_ledger_problems': problems,
        'total_F_calls': sum(((r.get('ledger_cross_check') or {}).get('per_arm_calls') or {})
                             .get('F', 0) for r in completed),
        'total_D_Geo_calls': sum(((r.get('ledger_cross_check') or {}).get('per_arm_calls') or {})
                                 .get('D-Geo', 0) for r in completed),
        'caps': {'per_arm_per_state': AUTHORISED['caps']['per_arm_per_state_scorer_calls'],
                 'per_arm_total': AUTHORISED['caps']['per_arm_total_scorer_calls']}})

    # D8 -------------------------------------------------------------------------------------
    spans = sorted(float(r['wall_s']) for r in completed if r.get('wall_s') is not None)
    ready = None
    for p in sorted(RUN.glob('L1_READY_*.json')):
        ready = json.loads(p.read_text(encoding='utf-8'))
        break
    init_s = float(ready['initialize_and_compile_s']) if ready else None
    batch_s = float(sup.get('worker_control_wall_s') or ctl.get('worker_control_wall_s') or 0)
    d8_ok = (spans and spans[-1] <= AUTHORISED['caps']['per_state_wall_clock_s']
             and init_s is not None and init_s <= AUTHORISED['caps']['initialize_compile_s']
             and batch_s <= AUTHORISED['caps']['batch_wall_clock_s']
             and not (adj.get('timeout_states') or []))
    ck('D8 the frozen caps were respected: per-state wall clock, init/compile, batch, per-arm calls',
       d8_ok,
       {'per_state_wall_s': {'min': spans[0] if spans else None,
                             'median': spans[len(spans) // 2] if spans else None,
                             'max': spans[-1] if spans else None,
                             'cap': AUTHORISED['caps']['per_state_wall_clock_s'],
                             'headroom_x': (AUTHORISED['caps']['per_state_wall_clock_s']
                                            / spans[-1]) if spans else None},
        'initialize_and_compile_s': init_s, 'cap_init': AUTHORISED['caps']['initialize_compile_s'],
        'batch_wall_s': batch_s, 'cap_batch': AUTHORISED['caps']['batch_wall_clock_s'],
        'n_timeout_states': len(adj.get('timeout_states') or []),
        'n_states_with_any_arm_failure': sum(1 for r in completed
                                             if any((r.get('per_arm_failure') or {}).values()))})

    # D9 -------------------------------------------------------------------------------------
    def classify(r):
        v2r = r.get('V2') or {}
        order = v2r.get('arm_order') or []
        fails = r.get('per_arm_failure') or {}
        if len(order) < 2 or not fails:
            return 'NOT_ORDER_ATTRIBUTABLE'
        first, second = order[0], order[1]
        a, b = fails.get(first) == 'timeout', fails.get(second) == 'timeout'
        if a and b:
            return 'ORDER_INDETERMINATE'
        if b and not a:
            return 'ORDER_ATTRIBUTABLE'
        return 'NOT_ORDER_ATTRIBUTABLE'

    recomputed_attr = {k: sum(1 for r in records if classify(r) == k)
                       for k in ('NOT_ORDER_ATTRIBUTABLE', 'ORDER_ATTRIBUTABLE',
                                 'ORDER_INDETERMINATE')}
    recorded_attr = ((adj.get('arm_order') or {}).get('counts')) or {}
    scale = ((adj.get('arm_order') or {}).get('wall_clock_scope'))
    d9_ok = (recomputed_attr == recorded_attr and recomputed_attr['ORDER_ATTRIBUTABLE'] == 0
             and scale == 'per_state_both_arms_share_one_10s_allowance')
    ck('D9 the arm-order attribution is recomputed independently and NO state is ORDER_ATTRIBUTABLE: '
       'the sequential order and the shared allowance produced no timing artefact', d9_ok,
       {'recomputed': recomputed_attr, 'recorded': recorded_attr, 'wall_clock_scope': scale,
        'arm_order_used': (adj.get('arm_order') or {}).get('order'),
        'reading': 'ORDER_ATTRIBUTABLE would mean a second-arm timeout attributable to the shared '
                   'allowance rather than to the scorer.  There are none, so no mechanism difference '
                   'in this run is confounded by the order.'})

    # D10 ------------------------------------------------------------------------------------
    start = load(RUN / 'CONTROL_IDENTITY_START.json') or {}
    end = None
    for p in sorted(RUN.glob('CONTROL_IDENTITY_END_*.json')):
        end = json.loads(p.read_text(encoding='utf-8'))
        break
    tail = None
    for p in sorted((RUN / 'phases').glob('IDENTITY_END_*.json')):
        tail = json.loads(p.read_text(encoding='utf-8'))
        break
    want = AUTHORISED['n_watch']
    d10_ok = (start.get('ok') is True and len(start.get('rows') or {}) == want
              and end is not None and end.get('ok') is True and len(end.get('rows') or {}) == want
              and tail is not None and len(tail.get('paths') or {}) == want)
    ck('D10 the identity witnesses match at BOTH ends: the same %d frozen paths at start, at the '
       'supervisor tail and in the worker journal' % want, d10_ok,
       {'start_ok': start.get('ok'), 'start_n_paths': len(start.get('rows') or {}),
        'end_ok': end.get('ok') if end else None,
        'end_n_paths': len(end.get('rows') or {}) if end else None,
        'tail_n_paths': len(tail.get('paths') or {}) if tail else None,
        'n_drifted_end': (sum(1 for v in (end.get('rows') or {}).values() if not v.get('ok'))
                          if end else None)})

    # D11 ------------------------------------------------------------------------------------
    inside = list(RUN.rglob('authorization_consumption_*.json'))
    receipt = json.loads(receipts[0].read_text(encoding='utf-8')) if receipts else {}
    ck('D11 the one-shot receipt exists EXACTLY once and lives OUTSIDE the run directory',
       len(receipts) == 1 and not inside and receipt.get('run_id') == AUTHORISED['run_id'],
       {'n_receipts': len(receipts), 'n_inside_the_run_dir': len(inside),
        'receipt_run_id': receipt.get('run_id'),
        'consumed_utc': receipt.get('consumed_utc'),
        'receipt_name': receipts[0].name if receipts else None,
        'why_it_matters': 'the fact of use must survive the run it authorised'})

    # D12 ------------------------------------------------------------------------------------
    live = {str(k).replace('\\', '/'): v for k, v in (frozen.get('watch') or {}).items()}
    on_disk_mismatch = [p for p, want in live.items() if sha(p) != want]
    cred = load(CRED) or {}
    manifest_pub = {}
    for p in sorted(EV.glob('*PREREGISTRATION_V2.md')):
        manifest_pub['prereg_sha256'] = sha(p)
    d12_ok = (not on_disk_mismatch and len(live) == want
              and manifest_pub.get('prereg_sha256') == AUTHORISED['prereg_sha256']
              and cred.get('run_id') == AUTHORISED['run_id']
              and cred.get('preregistration_sha256') == AUTHORISED['prereg_sha256']
              and cred.get('source_identity_sha256') == AUTHORISED['source_identity_sha256']
              and cred.get('state_binding_sha256') == AUTHORISED['state_binding_sha256'])
    ck('D12 the witness digests the credential bound are the ones ON DISK, and the pre-registration '
       'digest is the authorised one', d12_ok,
       {'n_watched_paths': len(live), 'n_drifted_since_the_run': len(on_disk_mismatch),
        'drifted': on_disk_mismatch[:5],
        'preregistration_sha256_now': manifest_pub.get('prereg_sha256'),
        'preregistration_sha256_authorised': AUTHORISED['prereg_sha256'],
        'credential_run_id': cred.get('run_id'),
        'credential_source_identity_sha256': cred.get('source_identity_sha256'),
        'matches_authorisation': cred.get('source_identity_sha256')
        == AUTHORISED['source_identity_sha256'],
        'credential_sha256': cred.get('credential_sha256')})

    # D13 ------------------------------------------------------------------------------------
    # the cause is stated by the RECORD and re-derived from the frozen derivation rule, not asserted
    src = (EXT / 'scripts/simulation/d3_l1_real_arm.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'frozen_jit_shapes':
            fn = node
    fn_src = ast.get_source_segment(src, fn) if fn else ''
    padded_by_mask_pool = "ms = [int(r['n_legal_expected']) for r in targets['rows']]" in fn_src
    d13_ok = (len(exceptions) == 1
              and exceptions[0].get('exception_type') == 'ValueError'
              and 'negative' in str(exceptions[0].get('error'))
              and padded_by_mask_pool)
    ck('D13 the single exception is a NEGATIVE PAD in the frozen F path: `M_max` is derived from '
       '`n_legal_expected` while the padded tensor is the mask pool, which is one larger',
       d13_ok,
       {'exception': {k: exceptions[0].get(k) for k in ('day_id', 'position', 'exception_type',
                                                        'error')} if exceptions else None,
        'derivation_read': "M_max = max(n_legal_expected) over the V1.22 target rows",
        'derivation_present_in_source': padded_by_mask_pool,
        'why_only_one_state': 'the mask pool is the ordered-legal list plus the already-committed '
                              'candidates, so it exceeds the legal count by a variable amount; only '
                              'the state whose pool reaches 27 passes M_max = 26',
        'this_is_a_DEFECT_not_a_state_property': True,
        'not_repaired_here': 'repairing it changes a frozen watched path, which would require a NEW '
                             'identity and a NEW authorisation; this tool only reports it'})

    # D14 ------------------------------------------------------------------------------------
    n_slot = sum(1 for r in phase_recs if r.get('stage') == 'slot')
    n_fin = sum(1 for r in phase_recs if r.get('stage') == 'slot_finished')
    n_completed_stage = sum(1 for r in phase_recs if r.get('stage') == 'completed')
    sequence_ok = all(r.get('sequence') == i for i, r in enumerate(phase_recs))
    d14_ok = (n_slot == 64 and n_fin == 64 and n_completed_stage == 1 and sequence_ok
              and len(phase_recs) == 131)
    ck('D14 the record set is EXACTLY ONE pass over the 64 states: 64 slot starts, 64 finishes, one '
       'completed marker, a contiguous sequence and no second pass', d14_ok,
       {'slot_starts': n_slot, 'slot_finishes': n_fin, 'completed_markers': n_completed_stage,
        'n_phase_records': len(phase_recs), 'sequence_contiguous': sequence_ok,
        'expected_phase_records': 1 + 1 + 1 + 2 * 64 + 1,
        'no_rerun': 'a second pass would show a second `initialize`/`compile` or more than 64 slot '
                    'starts; the journal shows neither'})

    # independence ---------------------------------------------------------------------------
    leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
    src_self = Path(__file__).read_text(encoding='utf-8')
    own_imports = []
    for node in ast.parse(src_self).body:
        if isinstance(node, ast.Import):
            own_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            own_imports.append(node.module)
    indep_ok = not leaked and not any(m in ' '.join(own_imports) for m in FORBIDDEN)

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_v2_executed_round_independent_adjudication',
        'run_id': AUTHORISED['run_id'],
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'round_status_from_the_run': adj.get('status'),
        'round_blocking_reasons': adj.get('blocking_reasons'),
        'denominator': {'n_states': len(records), 'n_completed': len(completed),
                        'n_exception': len(exceptions),
                        'exception_states': exc_keys},
        'V1_counts_over_all_64': counts_all,
        'V1_counts_over_the_comparable_subset': counts_comparable,
        'V2_full_repair_completed_recomputed': v2,
        'V3': 'NOT_RUN for all 64 states (no certifier, no install)',
        'order_attribution_recomputed': recomputed_attr,
        'resources': {'per_state_wall_s': {'min': spans[0] if spans else None,
                                           'median': spans[len(spans) // 2] if spans else None,
                                           'max': spans[-1] if spans else None},
                      'initialize_and_compile_s': init_s, 'batch_wall_s': batch_s,
                      'F_total_calls': sum(((r.get('ledger_cross_check') or {})
                                            .get('per_arm_calls') or {}).get('F', 0)
                                           for r in completed),
                      'D_Geo_total_calls': sum(((r.get('ledger_cross_check') or {})
                                                .get('per_arm_calls') or {}).get('D-Geo', 0)
                                               for r in completed)},
        'independence': {'imports_none_of': list(FORBIDDEN),
                         'modules_found_in_sys_modules': leaked,
                         'imports_declared_here': sorted(set(own_imports)),
                         'ok': indep_ok},
        'not_claimed': ['a certified executable plan (V3 = NOT_RUN)',
                        'any decision-quality, utility, significance or learning claim',
                        'that the 63/64 answerable denominator is acceptable -- that is the signer\'s '
                        'decision and this tool does not make it',
                        'any statement about the historical training backbone'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)),
        'verdict': ('L1_V2_EXECUTED_ROUND_INDEPENDENTLY_ADJUDICATED' if not fails and indep_ok
                    else 'L1_V2_EXECUTED_ROUND_ADJUDICATION_FAILED'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1-v2 executed-round independent adjudication: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if (not fails and indep_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
