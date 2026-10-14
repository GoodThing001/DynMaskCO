#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 ACTION-CONSISTENCY AUDIT (user-ordered P0 follow-up, 2026-10-14).

Purpose: close the gap the user found in the previous round's evidence.  Two defects were
confirmed in that evidence and are fixed HERE, in a NEW audit tool -- the frozen tree
`freeze_tree_5273739d/`, its raw results and every historical test are left untouched:

  1. `apply_calls` reference pollution: the previous suite stored the module-global list itself
     into the archived record, so a later `clear()` rewrote already-archived records.  Here every
     scenario owns a Recorder whose snapshot is a deep copy taken at capture time.
  2. The previous "choice vs applied action" probe did not complete a clean run: its record
     carries `failure_reason='score_len'` with `n_steps=1`.  Here each scenario reports its own
     run-level `failure_reason`, and the scoring is length-matched BY CONSTRUCTION
     (`G.geo_score`), because `stepwise_sample_d3` is MULTI-STEP: a hand-fed constant score
     vector is only valid for the steps whose legal count equals its length.

Checks, each judged separately:
  A. single-step legal control (same customer, two legal actions, EXACT tie) -- field-by-field
     comparison of every applied `FleetAction` against the corresponding step record, plus a
     plan-level comparison (the plan actually returned vs the plan rebuilt from the step action).
  B. full transaction flow (`run_repair_transaction` + the TEST-ONLY certifier): the applied
     action must match the assembly's recorded choice for the identical fixture, and the
     certified/installed plan identity must match the plan rebuilt from that action.
  C. reproduction of the `score_len` mechanism (constant vector vs length-matched scorer).
  D. isolation proof: scenario A's archived snapshot is byte-identical before and after B runs.

A field mismatch is an IMPLEMENTATION DEFECT and is reported as a FAIL, never explained away.

The D3 modules are imported from the FROZEN tree whose path is passed in (`--d3-sim`), so the
audit runs against the registered identity, not the worktree copy.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

CASES = []
RECORDS = {}
ACTION_FIELDS = ('customer', 'slot_kind', 'slot_anchor', 'position', 'predecessor', 'successor')


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def rec(key, value):
    RECORDS[key] = value


class Recorder:
    """Per-scenario recorder.  Snapshots are deep copies, so no two scenarios share a list."""

    def __init__(self, name, mutate=False, G=None, max_calls=12):
        self.name = name
        self.mutate = mutate
        self.G = G
        self.max_calls = max_calls
        self.calls = []

    # -- capture -------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError('AUDIT_RECORDER_GUARD: more than %d apply calls in %s'
                               % (self.max_calls, self.name))
        plans = args[0] if args else kwargs.get('plans')
        action = None
        for a in list(args[1:]) + list(kwargs.values()):
            if _looks_like_action(a):
                action = a
                break
        applied = self._apply(plans, action) if self.mutate else plans
        self.calls.append({
            'step_index': len(self.calls),
            'action_repr': repr(action),
            'action_fields': _action_fields(action),
            'plan_in_identity': self._ident(plans),
            'plan_out_identity': self._ident(applied),
            'kwargs_keys': sorted(kwargs),
        })
        return applied

    def _apply(self, plans, action):
        if action is None or plans is None:
            return plans
        new = copy.deepcopy(plans)
        kind, anchor = action.slot.kind, action.slot.anchor
        target = None
        for vid, plan in new.items():
            if kind == 'new_route':
                if getattr(plan, 'anchor_node', None) == 0 and not getattr(plan, 'suffix', None):
                    target = vid
                    break
            elif getattr(plan, 'anchor_node', None) == anchor:
                target = vid
                break
        if target is None:
            return new
        plan = new[target]
        suffix = list(getattr(plan, 'suffix', []))
        pos = max(0, min(int(action.position), len(suffix)))
        suffix.insert(pos, int(action.customer))
        try:
            new[target] = dataclasses.replace(plan, suffix=suffix)
        except Exception:
            plan2 = copy.copy(plan)
            plan2.suffix = suffix
            new[target] = plan2
        return new

    def _ident(self, plans):
        if self.G is not None and hasattr(self.G, '_plan_identity'):
            try:
                return self.G._plan_identity(plans)
            except Exception as exc:
                return '<identity-error: %r>' % (exc,)
        return repr(plans)

    def snapshot(self):
        """Deep copy: the archived record can never be mutated by a later scenario."""
        return copy.deepcopy({'name': self.name, 'n_calls': len(self.calls), 'calls': self.calls})

    @property
    def n_calls(self):
        return len(self.calls)


def _looks_like_action(obj):
    return (hasattr(obj, 'customer') and hasattr(obj, 'slot') and hasattr(obj, 'position')
            and hasattr(obj, 'predecessor') and hasattr(obj, 'successor'))


def _action_fields(action):
    if action is None:
        return None
    return {'customer': int(action.customer), 'slot_kind': action.slot.kind,
            'slot_anchor': int(action.slot.anchor), 'position': int(action.position),
            'predecessor': int(action.predecessor), 'successor': int(action.successor),
            'incumbent': bool(getattr(action, 'incumbent', False))}


def _subset(fields):
    return {k: fields[k] for k in ACTION_FIELDS} if fields else None


def _step_fields(step):
    if step is None:
        return None
    return {'customer': step.get('customer'), 'slot_kind': step.get('slot_kind'),
            'slot_anchor': step.get('slot_anchor'), 'position': step.get('position'),
            'predecessor': step.get('predecessor'), 'successor': step.get('successor')}


def _field_compare(step, action_fields):
    want, got = _step_fields(step), action_fields
    if want is None or got is None:
        return None, None, None, False
    per = {k: (want[k] == got[k]) for k in ACTION_FIELDS}
    return want, {k: got[k] for k in ACTION_FIELDS}, per, all(per.values())


def _public(obj):
    out = {}
    for name in dir(obj):
        if name.startswith('_'):
            continue
        try:
            out[name] = repr(getattr(obj, name))
        except Exception as exc:                                   # pragma: no cover
            out[name] = '<unreadable: %r>' % (exc,)
    return out


def _load(tree_sim, tree_tests, fallback):
    for p in reversed([str(f) for f in fallback]):
        sys.path.insert(0, p)
    sys.path.insert(0, str(tree_tests))
    sys.path.insert(0, str(tree_sim))
    G = importlib.import_module('d3_geo_adapter')
    MP = importlib.import_module('mpre_policy')
    AC = importlib.import_module('action_contract')
    H = importlib.import_module('test_d3_geo_adapter')
    return G, MP, AC, H


def identity_of(module):
    path = Path(getattr(module, '__file__', '<none>'))
    return {'file': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d3-sim', required=True, help='frozen tree scripts/simulation')
    ap.add_argument('--d3-tests', required=True, help='frozen tree scripts/tests')
    ap.add_argument('--fallback', nargs='*', default=[],
                    help='non-frozen helper dirs (coldchain_state, ...)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    G, MP, AC, H = _load(args.d3_sim, args.d3_tests, args.fallback)
    _cand, _plans, _identity_apply = H._cand, H._plans, H._identity_apply

    rec('imported_modules', {'d3_geo_adapter': identity_of(G), 'mpre_policy': identity_of(MP),
                             'action_contract': identity_of(AC),
                             'test_d3_geo_adapter': identity_of(H)})
    rec('primary_constants', {
        'EPS_D_PRIMARY_REL': getattr(G, 'EPS_D_PRIMARY_REL', '<missing>'),
        'EPS_D_PRIMARY_ABS': getattr(G, 'EPS_D_PRIMARY_ABS', '<missing>'),
        'TIE_BREAK_LEGAL_ORDER': getattr(G, 'TIE_BREAK_LEGAL_ORDER', '<missing>'),
        'INSTALL_REQUIRES_CERTIFICATION': getattr(G, 'INSTALL_REQUIRES_CERTIFICATION',
                                                  '<missing>')})

    P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
    MUT = {1, 2, 3}
    EXTRACT = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

    # EXACT tie: identical incremental_distance, so the tie-break (legal order) decides; and the
    # scorer is the adapter's own, so every step's score vector is length-matched by construction.
    A_POS1 = _cand(5, 'anchored', 3, 1, 5.0)
    A_POS0 = _cand(5, 'anchored', 3, 0, 5.0)
    ORDER = [A_POS1, A_POS0]          # legal head = larger canonical key
    LEGAL = lambda P, rem: [t for t in ORDER if t[0] in rem]           # noqa: E731
    # two DIFFERENT customers (canonical order reversed) -- the second control
    TIE2 = [_cand(5, 'anchored', 3, 0, 5.0), _cand(4, 'anchored', 3, 1, 5.0)]
    LEGAL2 = lambda P, rem: [t for t in TIE2 if t[0] in rem]           # noqa: E731

    print('--- A: single-step legal control, per-field action comparison ---')
    recA = Recorder('A_same_customer_exact_tie', mutate=True, G=G)
    exc_a = None
    try:
        plan_a, steps_a, _r_a, fr_a, _d_a = G.stepwise_sample_d3(
            score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=LEGAL, apply_fn=recA, plans=P0,
            mask_set=[5], mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True)
    except BaseException as exc:
        plan_a, steps_a, fr_a = None, [], None
        exc_a = '%s: %s\n%s' % (type(exc).__name__, exc, traceback.format_exc())
    snap_a_before = recA.snapshot()
    rec('A_run', {'failure_reason': fr_a, 'n_steps': len(steps_a), 'n_applied': recA.n_calls,
                  'legal_count': len(LEGAL(None, {5})), 'scorer': 'G.geo_score',
                  'exception': exc_a,
                  'legal_order_positions': [t[1].position for t in ORDER],
                  'plan_identity_returned': recA._ident(plan_a),
                  'snapshot': snap_a_before})
    ck('A0 the assembly terminates without an exception or the recorder guard',
       exc_a is None and recA.n_calls < recA.max_calls,
       {'exception': exc_a, 'n_applied': recA.n_calls, 'max_calls': recA.max_calls})
    ck('A1 the legal fixture runs with no CONTRACT-VIOLATION reason (score_len/score_shape/'
       'no_certifier) and every step has a recorded applied action (len(steps) == len(applied) '
       '>= 1); a terminal no_legal is recorded verbatim, not hidden',
       fr_a not in ('score_len', 'score_shape', 'no_certifier')
       and len(steps_a) == recA.n_calls and recA.n_calls >= 1,
       {'failure_reason': fr_a, 'n_steps': len(steps_a), 'n_applied': recA.n_calls})

    pairs = []
    all_ok = len(steps_a) == recA.n_calls and recA.n_calls > 0
    for i, step in enumerate(steps_a):
        applied = recA.calls[i] if i < recA.n_calls else None
        want, got, per, ok = _field_compare(step, applied['action_fields'] if applied else None)
        pairs.append({'step_index': i, 'step_fields': want, 'applied_fields': got,
                      'per_field_equal': per, 'all_equal': bool(ok),
                      'step_incremental_distance': step.get('incremental_distance'),
                      'applied_plan_in': applied['plan_in_identity'] if applied else None,
                      'applied_plan_out': applied['plan_out_identity'] if applied else None})
        all_ok = all_ok and bool(ok)
    rec('A_field_comparison', pairs)
    ck('A2 FIELD-BY-FIELD: every action actually passed to apply_fn equals the corresponding '
       'step record in all six action fields (customer/slot_kind/slot_anchor/position/'
       'predecessor/successor)',
       all_ok, {'pairs': pairs})

    plan_from_step = None
    if steps_a:
        step0 = steps_a[0]
        action_from_step = next((c for c in ORDER if _subset(_action_fields(c[1]))
                                 == _step_fields(step0) and step0.get('customer') == c[0]), None)
        if action_from_step is not None:
            builder = Recorder('plan_builder', mutate=True, G=G)
            plan_from_step = builder(P0, action_from_step[1])
    rec('A_plan_level', {'plan_identity_returned_by_assembly': recA._ident(plan_a),
                         'plan_identity_rebuilt_from_step0': recA._ident(plan_from_step)
                         if plan_from_step is not None else None,
                         'step0': repr(steps_a[0]) if steps_a else None})
    ck('A3 PLAN LEVEL: the plan the assembly returns equals the plan rebuilt from the recorded '
       'step action (identity comparison through the adapter _plan_identity)',
       plan_from_step is not None
       and recA._ident(plan_a) == recA._ident(plan_from_step),
       RECORDS['A_plan_level'])

    print('--- B: full transaction flow (TEST-ONLY certifier) ---')
    CERT_CALLS = []

    def _test_only_certifier(plan):
        CERT_CALLS.append(G._plan_identity(plan) if hasattr(G, '_plan_identity') else repr(plan))
        return True, 'TEST_ONLY_DETERMINISTIC_FIXTURE_CERTIFIER (not a C0 certification)'

    recB = Recorder('B_transaction_flow', mutate=True, G=G)
    installed = []
    ledger = G.InstallLedger(install_fn=lambda p: installed.append(p))
    raised = None
    try:
        ok_b = G.run_repair_transaction(
            ledger=ledger, score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=LEGAL,
            apply_fn=recB, plans=P0, mask_set=[5], mutable_ids=MUT,
            selector=G.SELECTOR_GEO, deterministic=True, certify_fn=_test_only_certifier)
    except BaseException as exc:
        ok_b, raised = None, '%s: %s\n%s' % (type(exc).__name__, exc, traceback.format_exc())
    dump_b = _public(ledger)
    snap_b = recB.snapshot()
    rec('B_run', {'returned': repr(ok_b), 'raised': raised, 'n_applied': recB.n_calls,
                  'certifier_calls': len(CERT_CALLS), 'certified_identities': CERT_CALLS,
                  'install_fn_calls': len(installed), 'snapshot': snap_b,
                  'ledger': dump_b, 'reason': dump_b.get('failure_reason')})
    ck('B1 the full transaction flow completes with the TEST-ONLY certifier: 1 install, '
       'certifier called once, no exception',
       ok_b is True and dump_b.get('n_installs') == '1' and len(installed) == 1
       and len(CERT_CALLS) == 1 and raised is None,
       {'returned': repr(ok_b), 'n_installs': dump_b.get('n_installs'),
        'install_fn_calls': len(installed), 'certifier_calls': len(CERT_CALLS),
        'raised': raised})

    # cross-mode: the action applied inside the transaction must equal the assembly's recorded
    # first choice for the identical fixture
    a_first = recA.calls[0]['action_fields'] if recA.n_calls else None
    b_first = recB.calls[0]['action_fields'] if recB.n_calls else None
    rec('B_cross_mode', {'assembly_first_applied': a_first, 'transaction_first_applied': b_first,
                         'equal': a_first == b_first})
    ck('B2 CROSS-MODE: the action applied inside the transaction equals the action the assembly '
       'records/applies for the identical fixture (all six fields)',
       a_first is not None and b_first is not None
       and all(a_first[k] == b_first[k] for k in ACTION_FIELDS),
       RECORDS['B_cross_mode'])

    cert_ids = CERT_CALLS
    installed_ids = [G._plan_identity(p) for p in installed] if hasattr(G, '_plan_identity') \
        else [repr(p) for p in installed]
    plan_from_b = None
    b_sub = {k: b_first[k] for k in ACTION_FIELDS} if b_first else None
    if b_sub is not None:
        action_b = next((c for c in ORDER if _subset(_action_fields(c[1])) == b_sub), None)
        if action_b is not None:
            builder = Recorder('plan_builder_b', mutate=True, G=G)
            plan_from_b = builder(P0, action_b[1])
    rebuilt = recB._ident(plan_from_b) if plan_from_b is not None else None
    b3_ok = (plan_from_b is not None and len(cert_ids) == 1 and len(installed_ids) == 1
             and cert_ids[0] == rebuilt and installed_ids[0] == cert_ids[0])
    rec('B_plan_level', {'certified_identities': cert_ids,
                         'actually_installed_plan_identities': installed_ids,
                         'plan_identity_rebuilt_from_applied_action': rebuilt,
                         'ledger_installs_raw': dump_b.get('installs'),
                         'three_way_binding': b3_ok})
    ck('B3 PLAN/IDENTITY LEVEL: three-way binding -- the identity of the plan rebuilt from the '
       'applied action == the certified identity == the identity of the plan actually handed to '
       'install_fn (and the ledger records the same certified hash)',
       b3_ok, RECORDS['B_plan_level'])

    print('--- C: score_len mechanism reproduction ---')
    recC0 = Recorder('C_constant_vector', mutate=False, G=G)
    plan_c0, steps_c0, _r, fr_c0, _d = G.stepwise_sample_d3(
        score_fn=lambda st: [1.0, 1.0], extract_fn=EXTRACT, legal_fn=LEGAL2, apply_fn=recC0,
        plans=P0, mask_set=[4, 5], mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True)
    recC1 = Recorder('C_length_matched', mutate=False, G=G)
    plan_c1, steps_c1, _r2, fr_c1, _d2 = G.stepwise_sample_d3(
        score_fn=G.geo_score, extract_fn=EXTRACT, legal_fn=LEGAL2, apply_fn=recC1, plans=P0,
        mask_set=[4, 5], mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True)
    rec('C_score_len_mechanism', {
        'constant_2_vector': {'failure_reason': fr_c0, 'n_steps': len(steps_c0),
                              'n_applied': recC0.n_calls},
        'length_matched_geo_score': {'failure_reason': fr_c1, 'n_steps': len(steps_c1),
                                     'n_applied': recC1.n_calls},
        'two_distinct_customers_fixture': [{'customer': t[0], 'position': t[1].position}
                                           for t in TIE2]})
    ck('C1 the previous probe\'s `failure_reason` is REPRODUCED as a multi-step artefact: a '
       'hand-fed constant 2-vector dies with score_len while steps from earlier iterations '
       'remain',
       fr_c0 == 'score_len' and len(steps_c0) >= 1,
       RECORDS['C_score_len_mechanism']['constant_2_vector'])
    ck('C2 with the length-matched adapter scorer the SAME two-customer fixture no longer dies '
       'with score_len (it reaches its natural terminal state) -- so the earlier probe was not a '
       'clean control, but the fixture itself is legal',
       fr_c1 != 'score_len' and len(steps_c1) >= 1,
       RECORDS['C_score_len_mechanism']['length_matched_geo_score'])

    print('--- D: isolation proof (no shared mutable list) ---')
    snap_a_after_b = recA.snapshot()
    rec('D_isolation', {'A_snapshot_before_B': snap_a_before, 'A_snapshot_after_B': snap_a_after_b,
                        'identical': snap_a_before == snap_a_after_b,
                        'A_list_object_id': id(recA.calls), 'B_list_object_id': id(recB.calls),
                        'distinct_lists': id(recA.calls) != id(recB.calls)})
    ck('D1 ISOLATION: scenario A\'s archived snapshot is byte-identical before and after B ran, '
       'and A/B never share a list object (the previous round\'s pollution cannot recur)',
       snap_a_before == snap_a_after_b and id(recA.calls) != id(recB.calls)
       and len(snap_a_before['calls']) == snap_a_before['n_calls'],
       {'identical_before_after': snap_a_before == snap_a_after_b,
        'distinct_lists': id(recA.calls) != id(recB.calls),
        'A_n_calls_archived': snap_a_before['n_calls'],
        'A_n_calls_live': recA.n_calls})

    rec('previous_round_defects_confirmed', {
        'reference_pollution': ('the previous suite stored the module-global APPLY_CALLS list '
                                'itself, so a later clear() rewrote archived records; confirmed '
                                'by the archived C1 record showing chosen_step.position=1 with an '
                                'apply_calls entry whose action is the LATER probe\'s action'),
        'non_clean_control': ('the archived C_coherence_probe record has failure_reason="score_len" '
                              'with n_steps=1 (multi-step artefact), reproduced in C1/C2 above')})

    fails = [c for c in CASES if not c[1]]
    print('\n=== %d rows, %d PASS, %d FAIL ===' % (len(CASES), len(CASES) - len(fails),
                                                   len(fails)))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
             'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
             'n_fail': len(fails)}, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
