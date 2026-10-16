#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 SIGNING PACKAGE: the seven-element pre-registration, the frozen identity and the budget
derivation, produced by MECHANICAL acceptance rather than by prose.

It performs the pre-signature checks the user asked for and refuses to emit a signable package while
any of them fails:

  M1  V1.22 shows 64/64 PASS and exactly one one-shot consumption receipt
  M2  the REAL F logits correspond to the first step's ORDERED legal list (length and order), i.e.
      the arm's logits length equals that step's legal length and the scorer is the V1.22-accepted one
  M3  F and D-Geo share a BYTE-IDENTICAL initial input (the protocol's single-factor machinery)
  M4  after a fork the two arms' legal lists may differ, and BOTH are recorded (never compared)
  M5  no later repair step reuses logits computed for the first step's legal list (the per-step
      ledger's legal fingerprints and logits lengths are checked step by step)
  M6  failures, timeouts, rollbacks and an empty pool all stay in the denominator -- and the L1
      adjudication cannot lose one, because it takes an `expected_states` bound
  M7  L1's OWN sources, the D-Geo adapter, the F scorer, the state materialisation and every actual
      dependency are frozen by SHA, derived from the L1 ENTRY (`d3_l1_closure`) rather than from the
      F-forward closure
  M8  an independent read-only audit can re-adjudicate the three verdicts and the whole identity
      WITHOUT importing the algorithms under test
  M9  the C0 budget stays an explicitly UNRESOLVED, non-blocking item
  M10 the three reviewed sign-blockers (freeze coverage, READY + per-state supervision, the 64-state
      denominator) are closed by executable counterexamples

The budget is DERIVED, not invented: the F cost measured in V1.22 is used, with the step accounting
spelled out, and any quantity that cannot be derived from the available evidence is recorded as a
GAP instead of being filled with a guessed multiplier.

The package is emitted ONLY when every gate passes.  A package that cannot be signed is not written at
all, so a stale file can never be mistaken for a fresh authorisation.
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
for _p in ('scripts', 'scripts/simulation', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_closure as l1_closure                                        # noqa: E402
import d3_l1_v118_persistence as persistence                              # noqa: E402

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V122 = EV / 'v122_evidence'
L1 = EV / 'l1_gates'
RUN_V122 = V122 / 'run_v122'

# The L1 identity is materialised under its own name so a v1 package is never overwritten in place.
RUN_ID = 'd3l1-l1-two-arm-20261015-l1v1'
L1_IDENTITY_FILE = L1 / 'L1_SIGNING_IDENTITY_V1.json'
L1_BLOCKER_ACCEPTANCE_FILE = L1 / 'L1_SIGN_BLOCKER_ACCEPTANCE.json'
L1_AUDIT_FILE = L1 / 'L1_SIGN_BLOCKER_INDEPENDENT_AUDIT.json'
L1_SYNTHETIC_FILE = L1 / 'L1_SUPERVISED_SYNTHETIC_RUN.json'
OLD_PACKAGE_FILE = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE.json'


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
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def derive_budget(adj):
    """Derive the L1 caps from the MEASURED V1.22 F cost, with the step accounting made explicit.

    Measured facts (V1.22 adjudication):
      * 0.422 s  = the span of ONE STATE covering TWO scorer calls, i.e. ~0.211 s per F call at the
                   first step (the largest legal list).  It is NOT the cost of a multi-step repair.
      * 33.0 s   = the worker wall clock for 64 states x 2 calls + init/compile 6.688 s
                   => (33.0 - 6.688) / 128 = 0.2057 s per F call
      * 129      = the recorded forward total

    What is NOT derivable from existing evidence: the per-step cost of steps 2..k, because V1.22
    scored each state exactly ONCE at the first step and never ran a multi-step repair.  A repair of
    m mask steps costs m F calls per arm per state, and the later steps have SMALLER legal lists, so
    the first-step cost is an UPPER bound rather than a typical value -- but the total call count is
    m, which no timing sample can pin down.  The cap is therefore expressed as a CALL ceiling (which
    is exact) plus a wall-clock ceiling derived from the worst-case per-call cost times that call
    ceiling, and the residual uncertainty is recorded as a GAP rather than smoothed over.
    """
    ex = (adj or {}).get('execution_summary') or {}
    per_state_span = ex.get('max_span_s')
    worker_wall = ex.get('worker_wall_s')
    init_compile = ex.get('init_and_compile_s')
    calls = ex.get('F_calls_recorded')
    per_call_measured = None
    if worker_wall is not None and init_compile is not None and calls:
        per_call_measured = (float(worker_wall) - float(init_compile)) / float(calls - 1)
    worst_per_call = max([v for v in (per_state_span / 2.0 if per_state_span else None,
                                      per_call_measured) if v])
    return {
        'source': 'V1.22 real-F measurements (v122_evidence/REAL_F_FORWARD_ADJUDICATION_V122.json)',
        'measured': {'per_state_span_two_calls_s': per_state_span,
                     'worker_wall_s': worker_wall, 'init_and_compile_s': init_compile,
                     'recorded_forward_calls': calls,
                     'per_call_from_wall_s': per_call_measured,
                     'worst_per_call_s': worst_per_call},
        'interpretation': {
            '0.422_s_is_not_a_repair_cost': 'that span covers ONE state and TWO first-step calls; a '
                                            'multi-step MaskCO repair costs one call per step',
            'later_steps_are_cheaper': 'the legal list shrinks as the mask is consumed, so the '
                                       'first-step cost is an UPPER bound per call',
            'call_count_is_exact_but_unknown_until_m_is_fixed': 'the repair length m (mask size) '
                                                                'decides the call total',
        },
        'gaps': [
            'the per-step cost of steps 2..m was never measured: V1.22 scored each state once at the '
            'first step, so a multi-step repair has NO direct timing sample',
            'no L1 pilot may be run to fill this gap without a separate authorisation; no multiplier '
            'is invented here',
        ],
        'not_relaxed': {'d3_single_decision_10s': 'the existing D3 10 s per-decision ceiling is '
                                                  'carried over unchanged and is NOT widened',
                        'offline_vs_online': 'an offline mechanism diagnostic may NOT claim to have '
                                             'passed the L3 online publication gate'},
    }


def l1_identity():
    """The L1 run identity as `build_l1_identity` derives it -- the object that is actually frozen.

    Reading the identity from the ENTRY rather than from the V1.22 F-forward record is the whole
    point of blocker A: the F record's watch set covers the F path, and gating the signature on it
    let M7 read PASS while `l1_sources_in_closure` was empty.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'l1_entry_for_signing', str(EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry_for_signing'] = module
    spec.loader.exec_module(module)
    ident = module.build_l1_identity(RUN_ID, created_utc='2026-10-15T00:00:00Z')
    # a stable, on-disk copy of exactly the object the freeze uses, so the signer can read it back
    L1_IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    L1_IDENTITY_FILE.write_text(json.dumps(ident, indent=2, sort_keys=True, default=repr),
                                encoding='utf-8')
    return ident, module


def blocker_acceptance():
    """The three-blocker acceptance produced by `d3_l1_l1_blocker_acceptance.py`."""
    return load(L1_BLOCKER_ACCEPTANCE_FILE) or {}


def mechanical_acceptance(check_only=False):
    adj = load(V122 / 'REAL_F_FORWARD_ADJUDICATION_V122.json') or {}
    verifier = load(V122 / 'INDEPENDENT_READONLY_VERIFIER_EXECUTED.json') or {}
    supervision = persistence.VerifiedReader(RUN_V122).latest('SUPERVISION_RESULT') or {}
    fwd = persistence.VerifiedReader(RUN_V122).latest('FORWARD_TARGETS') or {}
    l1_acc = load(L1 / 'A1_D3_L1_TWO_ARM_PROTOCOL_ACCEPTANCE.json') or {}
    l1_audit = load(L1 / 'A1_D3_L1_TWO_ARM_INDEPENDENT_AUDIT.json') or {}
    next_step = load(V122 / 'L1_NEXT_STEP_PACKAGE.json') or {}
    receipts = sorted((V122 / 'consumption').glob('authorization_consumption_*.json'))

    # THE L1 IDENTITY AND ITS DERIVED CLOSURE -- not the F-forward one
    ident, entry = l1_identity()
    watch = ident['watch']
    closure_info = l1_closure.build_watch_set()
    cc_ok, cc = l1_closure.cross_check_watch_set(list(watch))
    critical = list(l1_closure.critical_l1_sources())
    l1_own = [c for c in critical if c in set(closure_info['watch'])]
    blockers = blocker_acceptance()
    audit = load(L1_AUDIT_FILE) or {}
    synthetic = load(L1_SYNTHETIC_FILE) or {}
    states = ident['state_list']
    drift = [p for p, want in watch.items() if sha(p) != want]
    # the L1 budget is DERIVED from the measured V1.22 F cost -- `derive_budget` is what carries the
    # step accounting and the GAP list, so it must actually be called
    budget = derive_budget(adj)

    checks = {}
    checks['M1_v122_pass_and_single_receipt'] = {
        'ok': (adj.get('verdict') == 'REAL_F_FORWARD_ENGINEERING_PASS_ADJUDICATED'
               and adj.get('n_fail') == 0
               and verifier.get('verdict') == 'VERIFIED'
               and len(receipts) == 1),
        'detail': {'adjudication': adj.get('verdict'), 'n_pass': adj.get('n_pass'),
                   'n_fail': adj.get('n_fail'), 'verifier': verifier.get('verdict'),
                   'n_receipts': len(receipts),
                   'rows_independently_ok': ((verifier.get('records') or {}).get('row_state') or {})
                   .get('n_independently_ok'),
                   'supervision_verdict': supervision.get('verdict')}}
    checks['M2_real_F_logits_match_first_step_legal'] = {
        'ok': bool(l1_acc.get('verdict') == 'L1_TWO_ARM_PROTOCOL_ENGINEERING_ACCEPTED')
        and bool((adj.get('execution_summary') or {}).get('F_calls_recorded')),
        'detail': {'protocol_acceptance': l1_acc.get('verdict'),
                   'protocol_n_pass': l1_acc.get('n_pass'),
                   'v122_F_calls': (adj.get('execution_summary') or {}).get('F_calls_recorded'),
                   'real_arm_module': 'scripts/simulation/d3_l1_real_arm.py',
                   'real_arm_sha256': sha(EXT / 'scripts/simulation/d3_l1_real_arm.py'),
                   'why_this_is_the_right_check': 'the L1 real arm wraps the real scorer so that a '
                                                  'logits vector whose length differs from THIS '
                                                  "step's ordered legal list raises instead of "
                                                  'being reused'}}
    checks['M3_shared_initial_input_byte_identical'] = {
        'ok': bool(l1_acc.get('n_fail') == 0),
        'detail': {'protocol_acceptance': l1_acc.get('verdict'),
                   'single_factor_rule': 'state_sha256 before == after == shared for BOTH arms; '
                                         'first-step legal fingerprint equal on both arms; only '
                                         'scorer_id declared as varying',
                   'enforced_by': 'd3_l1_two_arm_protocol.run_two_arms / assert_single_factor'}}
    checks['M4_post_fork_legal_recorded_not_compared'] = {
        'ok': bool(l1_acc.get('n_fail') == 0 and l1_audit.get('n_fail') == 0),
        'detail': {'audit_verdict': l1_audit.get('verdict'),
                   'rule': 'the first-step legal fingerprint must match on both arms; later '
                           'fingerprints are recorded per arm and never used as an input mismatch'}}
    checks['M5_no_cross_step_logits_reuse'] = {
        'ok': bool(l1_acc.get('n_fail') == 0),
        'detail': {'mechanism': "d3_l1_real_arm.Ledger records, per scorer call, the step's legal "
                                'fingerprint, the logits length and the logits digest; the L1 run '
                                'report carries the ledger and the audit re-reads it',
                   'hard_failure': 'make_score_fn raises when logits length != that step legal length'}}
    # M6 now has a MECHANICAL basis, not a prose one: the adjudication takes an `expected_states`
    # bound and reports BLOCKED when the denominator is not the pre-registered list.
    checks['M6_failures_stay_in_denominator'] = {
        'ok': bool(blockers.get('results', {}).get('C', {}).get('C1', {}).get('ok')
                   and blockers.get('results', {}).get('C', {}).get('C4', {}).get('ok')
                   and blockers.get('results', {}).get('C', {}).get('C6', {}).get('ok')),
        'detail': {'rule': 'the state list is frozen BEFORE the run; every pre-registered state gets '
                           'its own record, and exception / timeout / empty-pool / not-comparable / '
                           'not-attempted states are listed SEPARATELY rather than removed',
                   'pre_fix_defect': 'adjudicate([r for r in records if r.get("status") != '
                                     '"FAILED_EXCEPTION")]) shrank the denominator to the survivors',
                   'mechanical_basis': 'd3_l1_l1_blocker_acceptance rows C1 (inject one fault, '
                                       'denominator stays 64), C4 (a missing state -> BLOCKED), '
                                       'C6 (no filtered call site remains)',
                   'expected_states_bound': int(entry.EXPECTED_STATES),
                   'c1': blockers.get('results', {}).get('C', {}).get('C1'),
                   'c4': blockers.get('results', {}).get('C', {}).get('C4')}}
    # M7 -- BLOCKER A.  The gate now reads the L1 closure and requires the L1 modules BY NAME.
    checks['M7_closure_frozen_by_sha'] = {
        'ok': bool(cc_ok and watch
                   and len(watch) == len(closure_info['watch']) + len(l1_closure.NON_SOURCE_ASSETS)
                   and all(c in closure_info['watch'] for c in critical) and not drift),
        'detail': {'n_watch_paths': len(watch),
                   'n_derived_source': closure_info['n_watch'],
                   'n_non_source_assets': len(l1_closure.NON_SOURCE_ASSETS),
                   'covers_closure': bool(cc_ok),
                   'closure_diff': cc,
                   'closure_module': 'scripts/data/d3_l1_closure.py',
                   'critical_l1_sources': critical,
                   'l1_sources_in_closure': sorted(
                       p for p in closure_info['watch']
                       if p.endswith(('d3_l1_l1_formal_run.py', 'd3_l1_real_arm.py',
                                      'd3_l1_two_arm_protocol.py', 'd3_geo_adapter.py',
                                      'd3_l1_v110_input_qualification.py', 'd3_l1_closure.py'))),
                   'n_drifted_at_gate_time': len(drift),
                   'pre_fix_defect': 'the watch set was the F-FORWARD closure: this same gate read '
                                     'PASS while l1_sources_in_closure == [] and n_watch_paths == 34, '
                                     'i.e. it froze code that decides the F forward and none of the '
                                     'code that decides the L1 comparison',
                   'derived_from': 'the L1 formal entry, via the transitive local-import graph'}}
    checks['M8_independent_audit_without_importing_the_algorithm'] = {
        'ok': bool(l1_audit.get('verdict') == 'L1_TWO_ARM_PROTOCOL_INDEPENDENTLY_AUDITED'
                   and l1_audit.get('n_fail') == 0
                   and audit.get('verdict') == 'L1_SIGN_BLOCKERS_INDEPENDENTLY_AUDITED'
                   and audit.get('n_fail') == 0),
        'detail': {'protocol_audit': l1_audit.get('verdict'),
                   'protocol_audit_n_pass': l1_audit.get('n_pass'),
                   'blocker_audit': audit.get('verdict'),
                   'blocker_audit_n_pass': audit.get('n_pass'),
                   'blocker_audit_n_fail': audit.get('n_fail'),
                   'independence': 'the audits read the emitted trace / package JSON and import '
                                   'neither the protocol, nor the geo adapter, nor the acceptance '
                                   'harness'}}
    c0 = ((next_step.get('2_l1_preconditions') or {}).get('checks') or {}).get(
        'c0_budget_for_certification') or {}
    checks['M9_c0_classification'] = {
        'ok': True,
        'classification': 'NON_BLOCKING_UNRESOLVED',
        'detail': {'original_record': c0,
                   'ruling': 'the C0 formal budget stays an independent open item; L1 reports V3 = '
                             'NOT_RUN, so no formal budget is required to answer the L1 mechanism '
                             'question, and no plan is installed',
                   'not_a_pass': 'this is recorded as UNRESOLVED, never as a PASS'}}
    # M10 -- BLOCKERS B and C, plus the executable counterexamples for A.
    b_rows = blockers.get('results', {}).get('B', {})
    a_rows = blockers.get('results', {}).get('A', {})
    checks['M10_sign_blockers_closed_by_counterexample'] = {
        'ok': bool(blockers.get('verdict') == 'L1_SIGN_BLOCKERS_CLOSED'
                   and blockers.get('n_fail') == 0
                   and all(a_rows.get(k, {}).get('ok') for k in ('A1', 'A3', 'A4', 'A5', 'A6',
                                                                 'A7', 'A8'))
                   and all(b_rows.get(k, {}).get('ok') for k in ('B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                                                                 'B7'))
                   and all(blockers.get('results', {}).get('C', {}).get(k, {}).get('ok')
                           for k in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'))
                   and synthetic.get('verdict') == 'L1_SUPERVISED_SYNTHETIC_RUN_ACCEPTED'),
        'detail': {'blocker_acceptance_verdict': blockers.get('verdict'),
                   'blocker_acceptance_n_pass': blockers.get('n_pass'),
                   'blocker_acceptance_n_fail': blockers.get('n_fail'),
                   'A_freeze_covers_l1': {k: a_rows.get(k, {}).get('ok')
                                          for k in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8')},
                   'B_ready_and_per_state': {k: b_rows.get(k, {}).get('ok')
                                             for k in ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')},
                   'C_denominator': {k: blockers.get('results', {}).get('C', {}).get(k, {}).get('ok')
                                     for k in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6')},
                   'supervised_synthetic_run': synthetic.get('verdict'),
                   'wall_clock_scope': getattr(entry, 'WALL_CLOCK_SCOPE', None),
                   'frozen_caps_unchanged': ident['budget'],
                   'budget_derivation_source': budget['source'],
                   'budget_derivation_gaps': budget['gaps']}}
    return checks, ident, entry, closure_info, cc, fwd, blockers, audit, synthetic, budget, adj, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='emit the package even when a gate fails, writing it to <json>.blocked '
                         'instead: a blocked package is evidence about WHAT is missing, and it is '
                         'never written at the signable path')
    args = ap.parse_args()
    (checks, ident, entry, closure_info, cc, fwd, blockers, audit, synthetic, budget,
     adj, states) = mechanical_acceptance()
    gates = {k: v['ok'] for k, v in checks.items()}
    ok = all(gates.values())

    package = {
        'schema': 'd3_l1_l1_formal_execution_preregistration_and_signing_package',
        'package_version': 'L1_V2',
        'run_id': RUN_ID,
        '1_state_source': {
            'reuse': 'the 64 frozen V1.9 dev_check states, LEDGER PUBLICATION ORDER, unchanged',
            'ledger_sha256': sha(EV / 'v19_evidence/formal_collection/STATE_LEDGER.jsonl'),
            'state_manifest_sha256': sha(EV / 'v19_evidence/STATE_MANIFEST_64_V19.json'),
            'n_states': ident['state_list']['n_states'],
            'states_frozen_before_the_run': True,
            'empty_pool_and_failures': 'kept in the denominator as-is',
            'denominator_rule': 'every pre-registered state keeps its own record; exception / '
                                'timeout / empty-pool / not-comparable / not-attempted states are '
                                'listed separately and none is removed or replaced',
        },
        '2_arms': {
            'F': {'scorer': 'the REAL MaskCO scorer accepted by V1.22',
                  'run_id': ident['reuses']['run_id'],
                  'f_ckpt_sha256': ident['identities']['f_ckpt']['sha256'],
                  'backbone_sha256': ident['identities']['backbone']['sha256'],
                  'label': ident['arms']['F']['label'],
                  'synthetic_fallback': 'NONE: load_real_f_scorer raises on a digest mismatch'},
            'D_Geo': {'scorer': 'registered geo_score = -incremental_distance',
                      'selector': 'geo_unified_minimum', 'eps_rel': 0.0, 'eps_abs': 0.0,
                      'tie_break': 'legal_order_first'},
        },
        '3_shared_inputs': {
            'p0': 'the same P0 from the accepted V1.10 materialisation path',
            'strict_online_visibility': 'identical for both arms',
            'b_pool': 'build_pool_B, built BEFORE sampling',
            'ordered_legal': "the environment's enumerator order, verbatim",
            'mask': 'identical', 'contract': 'identical (effective contract hash frozen)',
            'tie_break': 'legal_order_first for both', 'resources': 'identical BudgetGate ceilings',
            'declared_varying_fields': ['scorer_id'],
        },
        '4_three_verdicts': {
            'V1_first_step_divergence': 'DIVERGENT / IDENTICAL / NOT_COMPARABLE / '
                                        'NOT_APPLICABLE_EMPTY_MASK / NOT_ATTEMPTED',
            'V2_full_repair_completed': 'per arm, excluding certification',
            'V3_certified_executable_plan': 'NOT_RUN for this execution, with the certifier identity '
                                            'recorded as absent; no plan is installed',
        },
        '5_budget': dict(budget, **{
            'wall_clock_scope': entry.WALL_CLOCK_SCOPE,            'wall_clock_scope_statement': (
                'the frozen per_state_wall_clock_s = 10.0 is the allowance for ONE STATE '
                'COMPARISON -- both arms TOGETHER.  Both arms\' gates are charged from ONE monotonic '
                'origin, so the second arm pays for the time the first one spent.  The alternative '
                'reading (10 s for F plus 10 s for D-Geo) would let one state consume 20 s while '
                'every durable record still declared "10 s", and is refused.  The per-arm CALL bound '
                'is per arm because the pre-registration names it per arm.'),
            'step_accounting': {
                'm_mask_steps': 'min(2, |B pool|) -- the same mask rule the accepted V1.10 input '
                                'materialisation used',
                'F_calls_per_state': 'm (one per step)',
                'D_geo_calls_per_state': 'm (one per step)',
                'F_calls_total_max': 128, 'D_geo_calls_total_max': 128,
                'both_arms_identical_ceiling': True,
                'no_bonus_for_being_slower': 'the CLOCK ceiling is shared and the CALL ceilings are '
                                             'identical, so F receives no extra time or calls for '
                                             'being slower',
            },
            'frozen_caps': dict(ident['budget'], unchanged_by_the_blocker_fixes=True,
                                provenance='the pre-registration fixed 60 / 10 / 180 s and the '
                                           '128 / 2 / 2 call counts; the blocker fixes did NOT '
                                           'change any of them (asserted by blocker row B7 and by '
                                           'the independent audit row R12)'),
            'how_the_caps_relate_to_the_measured_cost': {
                'measured': budget['measured'],
                'gaps': budget['gaps'],
                'not_relaxed': budget['not_relaxed'],
                'rule': 'the caps are the FROZEN engineering ceilings and are DERIVED here only in '
                        'the sense that the measured V1.22 F cost is shown to fit inside them; they '
                        'are not widened to accommodate a measurement, and the unmeasured part of a '
                        'multi-step repair is recorded as a GAP rather than filled in',
            },
        }),
        '6_identity_freeze': {
            'derived_from': 'the L1 formal entry (scripts/data/d3_l1_closure.py)',
            'watch_paths': len(ident['watch']),
            'watch_paths_with_sha256': ident['watch'],
            'watch_derivation': ident['watch_derivation'],
            'source_closure': closure_info['watch'],
            'closure_module': 'scripts/data/d3_l1_closure.py',
            'non_source_assets': closure_info['non_source_assets'],
            'critical_l1_sources': list(l1_closure.critical_l1_sources()),
            'state_binding_sha256': ident['state_binding_sha256'],
            'forward_targets_sha256': ident['forward_targets_sha256'],
            'drift_at_package_time': [p for p, want in ident['watch'].items() if sha(p) != want],
            'live_recheck': ('the supervisor authorises over a LIVE re-derivation of every watched '
                             'path, so drift detected at consumption time refuses the credential '
                             'BEFORE the one-shot opportunity is spent'),
        },
        '7_mechanical_acceptance': checks,
        'signature_scope': {
            'authorises': 'ONE L1 F-vs-D-Geo mechanism comparison over the 64 frozen states, '
                          'offline, no install, V3 = NOT_RUN',
            'does_NOT_authorise': ['E2 training', 'L2', 'L3', 'the 40-day batch', 'the final test',
                                   'installing any plan', 'the C0 budget ruling',
                                   'any utility or significance claim'],
            'failure_consumes_the_opportunity': True,
            'no_rerun': 'PASS/FAIL/timeout/kill all consume the single opportunity; no state may be '
                        'added, replaced or re-run',
        },
        'run_entry': {
            'module': 'scripts/simulation/d3_l1_l1_formal_run.py',
            'sha256': sha(EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'),
            'modes': ['--freeze-identity', '--supervise', '--worker', '--synthetic-run',
                      '--guard-probe'],
            'ready_semantics': ('the supervisor decides "initialization is over" from the worker\'s '
                                'immutable, uniquely named L1_READY record, verified through the '
                                'V1.18 persistence reader -- the SAME convention the V1.22 F path '
                                'proved, never the legacy fixed-name `READY.json` default that a '
                                'V1.18+ worker can never publish'),
            'per_state_supervision': ('the worker publishes one `slot` / `slot_finished` phase pair '
                                      'per pre-registered state, so the supervisor can tell "a state '
                                      'is running" from "the worker is stuck in initialization" and '
                                      'the V1.16 timeline audit adjudicates the journal unchanged'),
            'guard': 'shared with the V1.22 F path (d3_l1_v120_execution_context): a real L1 run '
                     'cannot start without a consumed, identity-bound credential',
        },
        'evidence': {
            'l1_identity': {'file': str(L1_IDENTITY_FILE.relative_to(REPO)),
                            'sha256': sha(L1_IDENTITY_FILE)},
            'blocker_acceptance': {'file': str(L1_BLOCKER_ACCEPTANCE_FILE.relative_to(REPO)),
                                   'sha256': sha(L1_BLOCKER_ACCEPTANCE_FILE),
                                   'verdict': blockers.get('verdict'),
                                   'n_pass': blockers.get('n_pass'), 'n_fail': blockers.get('n_fail'),
                                   'tool_sha256': blockers.get('tool_sha256')},
            'independent_audit': {'file': str(L1_AUDIT_FILE.relative_to(REPO)),
                                  'sha256': sha(L1_AUDIT_FILE),
                                  'verdict': audit.get('verdict'), 'n_pass': audit.get('n_pass'),
                                  'n_fail': audit.get('n_fail')},
            'run_entry_acceptance': {
                'file': 'l1_gates/L1_RUN_ENTRY_ACCEPTANCE.json',
                'sha256': sha(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json'),
                'verdict': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('verdict'),
                'n_pass': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('n_pass'),
                'n_fail': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('n_fail'),
                'scorer': 'SYNTHETIC: the acceptance says nothing about MaskCO or D-Geo'},
            'supervised_synthetic_run': {'file': str(L1_SYNTHETIC_FILE.relative_to(REPO)),
                                         'sha256': sha(L1_SYNTHETIC_FILE),
                                         'verdict': synthetic.get('verdict')},
            'real_arm_acceptance': {
                'file': 'l1_gates/L1_REAL_ARM_ACCEPTANCE.json',
                'sha256': sha(L1 / 'L1_REAL_ARM_ACCEPTANCE.json'),
                'verdict': (load(L1 / 'L1_REAL_ARM_ACCEPTANCE.json') or {}).get('verdict'),
                'what_it_proves': 'the REAL MaskCO scorer drives the two-arm protocol on ONE state; '
                                  'it makes no L1 divergence claim'},
        },
        'superseded': {
            'invalidated_package': str(OLD_PACKAGE_FILE.relative_to(REPO)),
            'invalidated_sha256': sha(OLD_PACKAGE_FILE),
            'reason': 'three sign-blocking defects were found in independent source review and are '
                      'fixed in this package: (A) M7 gated on the F-FORWARD closure, so it read PASS '
                      'while l1_sources_in_closure was empty and none of the L1 decision modules was '
                      'frozen; (B) run_contained was called with no ready_probe and the worker '
                      'published no READY and no per-state phase records, so initialization was '
                      'measured as the whole batch; (C) FAILED_EXCEPTION states were filtered out '
                      'before adjudication, silently shrinking the 64-state denominator',
            'not_deleted': 'the invalidated package is left BYTE-INTACT; this package supersedes it '
                           'explicitly instead of overwriting it',
        },
        'execution_command': (
            'python scripts/simulation/d3_l1_l1_formal_run.py --supervise '
            '--out-dir <frozen L1 run directory> '
            '--authorization <signed credential> --preregistration '
            '<A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md>'),
        'conclusions_this_can_support': [
            'whether F and D-Geo produce different FIRST-STEP actions on the same ordered legal list',
            'per arm, whether a complete repair was produced (uncertified)',
            'the failure/empty-pool/timeout distribution, with every state kept in the denominator'],
        'conclusions_this_cannot_support': [
            'that either repair was CERTIFIABLE (V3 is NOT_RUN)',
            'any decision-quality or online-utility difference',
            'any significance, learning contribution or MaskCO claim',
            'any statement about the historical training backbone'],
        'all_gates_pass': ok,
        'blocking_gates': [k for k, v in gates.items() if not v],
        'not_run': ['the L1 comparison itself', 'training', 'L2', 'L3', 'the 40-day batch',
                    'the final test'],
        'human_approval': 'NOT FILLED: no human approval field is populated by this tool, and this '
                          'package is NOT an authorisation',
        'stable_online_gain': False, 'science_ready': False,
        'tool_sha256': sha(Path(__file__)),
    }
    out = Path(args.json)
    if not ok and not args.force:
        print(json.dumps({'gates': gates, 'all_gates_pass': False,
                          'blocking_gates': package['blocking_gates'],
                          'wrote_nothing': True,
                          'reason': 'a package with a failing gate is never written at the signable '
                                    'path; pass --force to write it to <json>.blocked instead'},
                         indent=1, default=repr))
        return 1
    if not ok:
        out = out.with_name(out.name + '.blocked')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print(json.dumps({'gates': gates, 'all_gates_pass': ok,
                      'blocking_gates': package['blocking_gates'],
                      'budget_worst_per_call_s': budget['measured']['worst_per_call_s'],
                      'budget_gaps': len(budget['gaps']),
                      'm7_l1_sources_in_closure': len(
                          checks['M7_closure_frozen_by_sha']['detail']['l1_sources_in_closure']),
                      'm7_n_watch_paths': checks['M7_closure_frozen_by_sha']['detail']
                      ['n_watch_paths']}, indent=1, default=repr))
    print('WROTE', out)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
