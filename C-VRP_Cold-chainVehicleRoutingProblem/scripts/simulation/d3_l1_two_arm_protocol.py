#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 two-arm protocol: SHARED input, two scoring arms, THREE SEPARATE verdicts.

The L1 question is narrow: *given the exact same legal action set, do F (restored MaskCO logits)
and D-Geo (``-incremental_distance``) choose different EXECUTABLE repairs?*  This module builds the
common input and runs the two arms; it never compares F with D-Geo scientifically, because the real
F forward over the 64 states is the single item awaiting the user's signature.

Single-factor discipline (the whole point)
------------------------------------------
Two different scoring functions give "different actions" for many uninteresting reasons.  So the
input is built ONCE and handed to both arms as the SAME OBJECT, and every input identity is
hashed before and after each arm:

  * `shared_input_sha256`   -- hash of the immutable, audit-visible part of the input record;
  * `state_key_sha256`      -- hash of the scorer state CONTAINING its numpy arrays.  If any arm
                               mutated the state in place, this changes and the run is REFUSED.
  * per-arm `state_sha256`  -- must equal the shared identity on BOTH arms.

`assert_single_factor()` then verifies that the two arms differ in exactly one declared field:
`scorer_id`.  Everything else -- initial snapshot, P0, mask, ordered legal, mutable vehicle set,
capacity / tw_max / timestep convention, contract and information boundary -- must be equal, and
the equality is mechanical, not asserted in prose.

Three verdicts, never merged (a first-step difference with a later failure is NOT "F produced a
different executable plan")
------------------------------------------------------------------------------------------------
  V1 `first_step_divergence`      -- the two arms' first chosen actions differ
  V2 `full_repair_completed`      -- per arm: did it consume the whole mask without failure
  V3 `certified_executable_plan`  -- per arm: is there a plan that PASSED certification

`certify_fn` is injectable and its identity is recorded.  When no certifier is supplied, V3 is
reported as `NOT_RUN` rather than True or False: an absent check is not a passed check.

Fail-closed vocabulary is inherited from `d3_geo_adapter` (`no_legal`, `nan`, `timeout`,
`cert_fail`, `score_shape`, `score_len`, `geo_len`, `no_certifier`); a failure returns the ORIGINAL
plan and keeps the partial plan only as evidence.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402

SCHEMA = 'd3_l1_two_arm_v1'

ARM_F = 'F'
ARM_D_GEO = 'D-Geo'
ARMS = (ARM_F, ARM_D_GEO)

# The ONLY field the two arms may differ in.  Adding to this tuple is a protocol change and must be
# recorded in the pre-registration, because it widens the single factor.
DECLARED_VARYING_FIELDS = ('scorer_id',)

SHARED_INPUT_SECTIONS = ('state_identity', 'common_conditions', 'legal_identity', 'tensor_identity',
                         'budget', 'contract', 'certifier')


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def json_digest(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=repr).encode('utf-8'))


def array_digest(array):
    """Type + dtype + shape + raw bytes.  Never a tolerance comparison."""
    a = np.asarray(array)
    raw = a.tobytes()
    return {'dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': sha256_bytes(raw),
            'nbytes': len(raw)}


def legal_row(customer, action, candidate):
    """The canonical per-candidate identity of the ORDERED legal list.

    One function, used both for the shared input identity and for each arm's per-step trace, so the
    two can be compared byte-for-byte.  A separate ad-hoc encoder on either side would make the
    single-factor check compare strings rather than the same object.
    """
    return [int(customer), str(action.slot.kind), int(action.slot.anchor), int(action.position),
            int(action.predecessor), int(action.successor), float(candidate.incremental_distance)]


def legal_fingerprint(legal):
    """Ordered fingerprint of a `(customer, action, candidate)` list, order included."""
    return json_digest([legal_row(c, a, cand) for c, a, cand in legal])


# ---------------------------------------------------------------------------------------------
# the shared input
# ---------------------------------------------------------------------------------------------
def build_shared_input(*, env, inst_idx, day, clock, vehicles, served_mask, visible_ids, plans,
                       pool_B, pool_diag, mask, partial, legal, mutable_ids, protected_ids,
                       inputs, contract, capacity=50.0, tw_max=24.0, snapshot=None,
                       state_hash=None, reveal_time=None, visible_fingerprint=None,
                       budget=None, certifier_identity=None, legal_fn=None):
    """Build the ONE object both arms receive.

    `legal` is the reference ordered candidate list, produced by the environment's own enumerator
    (`mpre_policy.enumerate_legal_actions`) on the SAME `plans` and mask the arms will start from.
    Its ORDER is part of the input identity, because both arms use `first argmax` semantics and a
    re-ordered list would silently change the tie outcome.  When `legal_fn` is given, `legal` is
    recomputed from it here so the recorded identity cannot describe one list while the arms are
    handed another.
    """
    if legal_fn is not None:
        legal = legal_fn(plans, [int(c) for c in mask])
    legal_order = []
    for c, a, cand in legal:
        legal_order.append({'customer': int(c), 'slot_kind': str(a.slot.kind),
                            'slot_anchor': int(a.slot.anchor), 'position': int(a.position),
                            'predecessor': int(a.predecessor), 'successor': int(a.successor),
                            'incremental_distance': float(cand.incremental_distance)})
    tensor_hashes = {k: array_digest(v) for k, v in sorted(inputs.items()) if hasattr(v, 'dtype')}
    record = {
        'schema': SCHEMA,
        'state_identity': {'inst_idx': int(inst_idx), 'day': int(day), 'clock': float(clock),
                           'state_hash': state_hash, 'reveal_time': reveal_time,
                           'visible_fingerprint': visible_fingerprint,
                           'n_visible': len(visible_ids),
                           'snapshot_present': snapshot is not None},
        'common_conditions': {'mask_order': [int(c) for c in mask],
                              'mask_sha256': json_digest([int(c) for c in mask]),
                              'mutable_ids': sorted(int(v) for v in mutable_ids),
                              'protected_ids': sorted(int(v) for v in protected_ids),
                              'pool_B': [int(c) for c in pool_B],
                              'pool_B_rule': (pool_diag or {}).get('rule'),
                              'capacity': float(capacity), 'tw_max': float(tw_max),
                              'num_vehicles': len(vehicles),
                              'n_legal': len(legal_order),
                              'tie_break': GEO.TIE_BREAK_LEGAL_ORDER,
                              'eps_rel': GEO.EPS_D_PRIMARY_REL, 'eps_abs': GEO.EPS_D_PRIMARY_ABS,
                              'selector_d_geo': GEO.SELECTOR_GEO,
                              'selector_f': GEO.SELECTOR_HISTORICAL},
        'legal_identity': {'legal_order': legal_order,
                           'legal_sha256': legal_fingerprint(legal),
                           'order_rule': 'the environment enumerator order, preserved verbatim; '
                                         'both arms use first-maximum argmax on THIS order'},
        'tensor_identity': tensor_hashes,
        'contract': {'effective_contract_hash': getattr(contract, 'contract_hash', None)},
        'budget': budget or {'wall_clock_limit_s': None, 'n_steps_cap': None,
                             'note': 'no budget supplied by the caller'},
        'certifier': {'identity': certifier_identity,
                      'run_certifier_identity': None},
        'transport': {'env_class': type(env).__name__, 'inst_idx': int(inst_idx), 'day': int(day),
                      'clock': float(clock), 'vehicles': vehicles, 'served_mask': served_mask,
                      'visible_ids': [int(c) for c in visible_ids], 'plans': plans,
                      'partial': partial, 'legal': legal, 'mutable_ids': mutable_ids,
                      'inputs': inputs, 'contract': contract, 'legal_fn': legal_fn},
    }
    if legal_fn is not None:
        record['legal_identity']['legal_source'] = ('recomputed by the caller-supplied legal_fn on '
                                                    'the same plans + mask the arms start from')
    else:
        record['legal_identity']['legal_source'] = 'supplied by the caller (V1.10 materialisation)'
    record['shared_input_sha256'] = json_digest(
        {k: record[k] for k in SHARED_INPUT_SECTIONS})
    record['state'] = make_scorer_state(record)
    record['state_key_before_sha256'] = json_digest(record['state'])
    return record


def make_scorer_state(record):
    """The dict handed to `extract_fn` / `score_fn`.

    It carries the SAME tensor objects (by reference, never copies) plus the transport views a
    scorer needs: the current plans, the remaining mask, and the ordered legal list.  Both arms
    receive this one dict, so a scorer that mutated it in place would be caught by the state digest
    check in `run_two_arms`.
    """
    transport = record['transport']
    return dict(transport['inputs'], plans=transport['plans'],
                mask=record['common_conditions']['mask_order'],
                mask_set=record['common_conditions']['mask_order'],
                legal=transport['legal'], partial=transport['partial'],
                mutable_ids=transport['mutable_ids'])


# ---------------------------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------------------------
def arm_scorers(f_score_fn=None, *, geo_extract_fn=None, base_extract_fn=None,
                allowed_vehicle_ids=None, f_label=None, d_label=None):
    """Return `(scorers, identities)`.

    `scorers['F'](state)`   -> the restored MaskCO logits for the ordered legal list
    `scorers['D-Geo'](state)` -> `-incremental_distance(a)`

    Both receive the SAME state object.  D-Geo's distance vector is supplied by the SHARED extractor
    (`make_geo_extract_fn`) and never recomputed, so a distance bug cannot be mistaken for a policy
    difference.  When `f_score_fn` is None the F arm REFUSES to run (a placeholder must never be
    scored as if it were F).
    """
    if geo_extract_fn is None:
        geo_extract_fn = GEO.make_geo_extract_fn(base_extract_fn,
                                                 allowed_vehicle_ids=allowed_vehicle_ids)

    def f_scorer(state):
        if f_score_fn is None:
            raise RuntimeError('the F arm has no restored scorer; refusing to substitute a '
                               'placeholder for real MaskCO logits')
        scores = np.asarray(f_score_fn(state))
        return scores

    scorers = {ARM_F: f_scorer, ARM_D_GEO: lambda state: np.asarray(GEO.geo_score(state))}
    identities = {
        ARM_F: {'scorer_id': 'restored_maskco_logits', 'label': f_label,
                'callable': GEO._callable_identity(f_score_fn), 'available': f_score_fn is not None},
        ARM_D_GEO: {'scorer_id': GEO.SELECTOR_GEO, 'label': d_label or 'geo_score = -delta_d',
                    'callable': GEO._callable_identity(GEO.geo_score), 'available': True},
    }
    return {'scorers': scorers, 'extract_fn': geo_extract_fn, 'identities': identities}


class BudgetGate(object):
    """Per-arm resource ceiling with FAIL-CLOSED semantics.

    Every unit is counted separately because the units are not interchangeable: a scorer CALL is not
    a wall-clock second, and neither is a plan certification.  When a ceiling is crossed the arm
    fails with `failure_reason='timeout'` (wall clock) or `'budget_calls'` (call count) and the
    ORIGINAL plan is returned -- a run that overspends never yields a candidate.

    The gate exists because the two arms have very different cost: D-Geo is a vector read while F is
    a neural forward.  Without an explicit shared ceiling, "F got more time because it is slower"
    would be an unregistered advantage.  The ceiling is therefore set per arm BEFORE the comparison
    and is identical for both.
    """

    def __init__(self, *, max_scorer_calls=None, max_wall_seconds=None, started_mono=None):
        """`started_mono` lets SEVERAL gates share ONE wall-clock origin.

        L1 pre-registers `per_state_wall_clock_s = 10.0` for a STATE, not for an arm.  Giving each
        arm its own origin would hand the state two separate ten-second allowances while the record
        still claimed the state was bounded by ten seconds.  The caller therefore passes one origin
        to both arms' gates, and the second arm is charged the time the first one spent: the sum of
        the two arms can never exceed the single pre-registered per-state allowance.  The default
        (``None``) keeps the original per-gate semantics for every existing caller.
        """
        self.max_scorer_calls = max_scorer_calls
        self.max_wall_seconds = max_wall_seconds
        self.scorer_calls = 0
        self.started = time.monotonic() if started_mono is None else float(started_mono)
        self.shared_origin = started_mono is not None
        self.exceeded = None

    def charge_call(self):
        self.scorer_calls += 1
        if self.max_scorer_calls is not None and self.scorer_calls > self.max_scorer_calls:
            self.exceeded = 'budget_calls'
        return self.exceeded

    def charge_elapsed(self):
        elapsed = time.monotonic() - self.started
        if self.max_wall_seconds is not None and elapsed > self.max_wall_seconds:
            self.exceeded = 'timeout'
        return self.exceeded

    def remaining_wall(self):
        if self.max_wall_seconds is None:
            return None
        return max(0.0, self.max_wall_seconds - (time.monotonic() - self.started))

    def snapshot(self):
        return {'max_scorer_calls': self.max_scorer_calls,
                'max_wall_seconds': self.max_wall_seconds,
                'scorer_calls': self.scorer_calls,
                'wall_seconds': round(time.monotonic() - self.started, 6),
                'wall_origin_shared_across_arms': self.shared_origin,
                'exceeded': self.exceeded}


def _wrap_scorer_with_gate(arm, scorer, gate, on_step=None):
    """Wrap a scorer so every call is charged and a crossed ceiling becomes a failure, not a crash."""
    def gated(state):
        if gate is None:
            return scorer(state)
        if gate.charge_call():
            raise BudgetExceeded(arm, gate.exceeded)
        if on_step is not None:
            on_step(arm, state, gate)
        if gate.charge_elapsed():
            raise BudgetExceeded(arm, gate.exceeded)
        return scorer(state)
    return gated


class BudgetExceeded(RuntimeError):
    """Raised inside a gated scorer; `stepwise_sample_d3` converts it into a fail-closed rollback."""

    def __init__(self, arm, reason):
        super(BudgetExceeded, self).__init__('%s exceeded its %s ceiling' % (arm, reason))
        self.arm = arm
        self.reason = reason


def run_arm(arm, shared, *, scorer, extract_fn, legal_fn, apply_fn, certify_fn=None,
            wall_clock_limit_s=None, tie_break=GEO.TIE_BREAK_LEGAL_ORDER, gate=None,
            on_score_step=None):
    """Run ONE arm to completion (or failure) on the shared input.  Never mutates `shared`.

    The arm wraps `legal_fn` so that the ORDERED legal list is fingerprinted at EVERY step.  The
    first-step fingerprint is a shared-input identity (both arms must agree); after a fork the two
    arms legitimately enumerate different lists, so later fingerprints are recorded evidence only
    and are never used to declare an input mismatch.

    `gate` (a `BudgetGate`) charges every scorer call and the elapsed wall clock, and `on_score_step`
    observes `(arm, state, gate)` on each call -- the hook the L1 real-F arm uses to verify that the
    logits it receives correspond to THIS step's ordered legal list rather than to a cached one.
    """
    transport = shared['transport']
    plans = transport['plans']
    budget = GEO.Budget()
    legal_trace = []

    def traced_legal_fn(P, remaining):
        legal = legal_fn(P, remaining)
        legal_trace.append({'remaining': [int(c) for c in remaining],
                            'n_legal': len(legal),
                            'legal_sha256': legal_fingerprint(legal)})
        return legal

    gated_scorer = _wrap_scorer_with_gate(arm, scorer, gate, on_score_step)
    try:
        plan, steps, records, failure_reason, diag = GEO.stepwise_sample_d3(
            score_fn=gated_scorer, extract_fn=extract_fn, legal_fn=traced_legal_fn,
            apply_fn=apply_fn, plans=plans, mask_set=shared['common_conditions']['mask_order'],
            mutable_ids=transport['mutable_ids'], selector=GEO.SELECTOR_HISTORICAL,
            deterministic=True, certify_fn=certify_fn, budget=budget,
            wall_clock_limit_s=wall_clock_limit_s, tie_break=tie_break)
    except BudgetExceeded as exc:
        # The gate crossed a ceiling: fail CLOSED with the original plan, exactly like a deadline.
        plan, steps, records, failure_reason, diag = (
            dict(plans), [], [], exc.reason,
            {'budget_exceeded': exc.reason, 'n_steps': 0, 'certified': False, 'certifier': None,
             'certified_plan_hash': None, 'deadline_ok': False,
             'rollback': True, 'rollback_to': 'the original plans passed in by the caller'})
    first = records[0] if records else None
    out = {
        'arm': arm,
        'first_step': None if first is None else {
            'chosen_index_in_legal': first['chosen'],
            'customer': first['customer'], 'vehicle_id': first['vehicle_id'],
            'incremental_distance': first['incremental_distance'],
            'selection_rule': first['selection']['rule'],
            'score_exact_ties': first['score_exact_ties'],
            'canonical_key_of_selected': first['selection'].get('canonical_key_of_selected'),
        },
        'first_step_action': None if not steps else dict(steps[0]),
        'n_steps': len(steps),
        'steps': steps,
        'records': records,
        'legal_trace': legal_trace,
        'legal_first_step_sha256': (legal_trace[0]['legal_sha256'] if legal_trace else None),
        'legal_after_fork_sha256': [t['legal_sha256'] for t in legal_trace[1:]],
        'failure_reason': failure_reason,
        'partial_completed': failure_reason is None,
        'plan_after': {int(v): list(getattr(p, 'suffix', [])) for v, p in plan.items()},
        'plan_hash_after': GEO._plan_identity(plan),
        'returned_original_plan': bool(failure_reason is not None),
        'budget_gate': (gate.snapshot() if gate is not None else None),
        'certified': diag.get('certified'),
        'certified_plan_hash': diag.get('certified_plan_hash'),
        'certifier': diag.get('certifier'),
        'budget': budget.snapshot(),
        'diag': {k: v for k, v in diag.items() if k != 'partial_plan_at_failure'},
        'partial_plan_at_failure': diag.get('partial_plan_at_failure'),
    }
    return out


def run_two_arms(shared, arms, *, legal_fn, apply_fn, certify_fn=None, wall_clock_limit_s=None,
                 certify_not_run_reason=None, gates=None, on_score_step=None):
    """Run both arms and produce the THREE SEPARATE verdicts plus the mechanical audit.

    `gates` is `{arm: BudgetGate}`.  The gates are supplied BY THE CALLER, one per arm, with the
    SAME ceilings for both: the protocol never grants an arm extra time because it is slower.
    """
    import d3_geo_adapter as _geo
    if certify_fn is not None:
        shared['certifier']['run_certifier_identity'] = _geo._callable_identity(certify_fn)
    scorers = arms['scorers']
    extract_fn = arms['extract_fn']
    gates = gates or {}
    results = {}
    for arm in ARMS:
        before = json_digest(shared['state'])
        results[arm] = run_arm(arm, shared, scorer=scorers[arm], extract_fn=extract_fn,
                               legal_fn=legal_fn, apply_fn=apply_fn, certify_fn=certify_fn,
                               wall_clock_limit_s=wall_clock_limit_s, gate=gates.get(arm),
                               on_score_step=on_score_step)
        # Refuse a run in which an arm mutated the shared state: that would break the single factor.
        after = json_digest(shared['state'])
        results[arm]['state_key_unchanged'] = bool(before == after)
        results[arm]['state_sha256_before'] = before
        results[arm]['state_sha256_after'] = after
        results[arm]['scorer_id'] = arms['identities'][arm]['scorer_id']

    shared_state_sha = shared['state_key_before_sha256']
    audit = {
        'schema': SCHEMA,
        'shared_input_sha256': shared['shared_input_sha256'],
        'state_key_sha256': shared_state_sha,
        'legal_sha256': shared['legal_identity']['legal_sha256'],
        'mask_sha256': shared['common_conditions']['mask_sha256'],
        'contract_hash': shared['contract']['effective_contract_hash'],
        'declared_varying_fields': list(DECLARED_VARYING_FIELDS),
        'arm_identities': arms['identities'],
        'certifier_identity': shared['certifier']['identity'],
        'state_unchanged_both_arms': all(results[a]['state_key_unchanged'] for a in ARMS),
        'state_identity_agrees_both_arms': all(
            results[a]['state_sha256_before'] == shared_state_sha
            and results[a]['state_sha256_after'] == shared_state_sha for a in ARMS),
    }
    audit['single_factor_ok'] = bool(audit['state_unchanged_both_arms']
                                     and audit['state_identity_agrees_both_arms'])
    audit['single_factor_note'] = (
        'both arms were given the SAME state object; the only declared difference is scorer_id')

    # ---- V1 first-step divergence ---------------------------------------------------------
    fa, da = results[ARM_F]['first_step_action'], results[ARM_D_GEO]['first_step_action']
    v1 = {'verdict': None, 'reason': None}
    if not shared['common_conditions']['mask_order']:
        # An empty B pool is a recorded denominator entry, not a comparison outcome: there is no
        # first step to compare, so V1 is NOT APPLICABLE.  It must never be reported as "identical".
        v1.update(verdict='NOT_APPLICABLE_EMPTY_MASK',
                  reason='the B mask pool is empty, so no first step exists to compare; this state '
                         'stays in the denominator as an empty-pool entry')
    elif fa is None or da is None:
        v1.update(verdict='NOT_COMPARABLE',
                  reason='one arm produced no first step (failure: %s / %s)'
                         % (results[ARM_F]['failure_reason'], results[ARM_D_GEO]['failure_reason']))
    else:
        same = all(fa[k] == da[k] for k in ('customer', 'vehicle_id', 'slot_kind', 'slot_anchor',
                                            'position', 'predecessor', 'successor'))
        diverging_fields = [k for k in ('customer', 'vehicle_id', 'slot_kind', 'slot_anchor',
                                        'position', 'predecessor', 'successor')
                            if fa[k] != da[k]]
        v1.update(verdict='DIVERGENT' if not same else 'IDENTICAL',
                  f_action=fa, d_geo_action=da,
                  # the ONLY position at which both arms provably received the same legal list is
                  # the first step, so a divergence can only ever be located there
                  first_divergence_position=0 if not same else None,
                  diverging_fields=diverging_fields,
                  explanation='the first step of the mask sequence is the only position where the '
                              'two arms provably received the same legal list; after a fork the '
                              'partial plans differ and the legal lists may legitimately differ')
    v1['mask_order'] = shared['common_conditions']['mask_order']
    # Fork discipline: the FIRST-step legal fingerprint is a shared-input identity and must be equal
    # on both arms.  Later fingerprints may differ BY CONSTRUCTION and must never be read as an
    # input mismatch -- they are the policy's own consequence.
    v1['first_step_legal_sha256'] = {a: results[a]['legal_first_step_sha256'] for a in ARMS}
    v1['first_step_legal_identical'] = (results[ARM_F]['legal_first_step_sha256']
                                        == results[ARM_D_GEO]['legal_first_step_sha256']
                                        == shared['legal_identity']['legal_sha256'])
    v1['post_fork_legal_sha256'] = {a: results[a]['legal_after_fork_sha256'] for a in ARMS}
    v1['post_fork_legal_not_compared'] = (
        'after the first divergence the arms run different partial plans, so their subsequent '
        'legal lists are allowed to differ; this is a consequence of the policies, not an input '
        'identity mismatch')

    # ---- V2 full repair completion --------------------------------------------------------
    v2 = {a: {'full_repair_completed': bool(results[a]['partial_completed']
                                            and results[a]['failure_reason'] is None),
              'failure_reason': results[a]['failure_reason'],
              'n_steps': results[a]['n_steps'],
              'returned_original_plan': results[a]['returned_original_plan']} for a in ARMS}
    v2['both_completed'] = all(v2[a]['full_repair_completed'] for a in ARMS)
    # ---------------------------------------------------------------------------------------
    # ARM ORDER AND THE SHARED WALL-CLOCK ALLOWANCE (fairness instrumentation)
    #
    # The arms run SEQUENTIALLY and, for L1, are charged from ONE monotonic origin, because the
    # pre-registration fixes `per_state_wall_clock_s = 10.0` as the allowance for the STATE -- both
    # arms together -- rather than for each arm.  A direct consequence is that the SECOND arm's
    # remaining time is `10.0 s - (what the first arm spent)`, so the second arm's time budget
    # depends on the first arm's COST.
    #
    # That is an instrumentation property, not a property of either scorer, and it must never be
    # readable as a mechanism difference.  These fields make it explicit and mechanically separable
    # WITHOUT relaxing any budget and WITHOUT changing either arm:
    #   `wall_budget_source`   whether the clock origin was shared (one 10 s for the state) or
    #                          private (one 10 s per arm -- the default gate semantics);
    #   `order_sensitive`      TRUE when the gate's `timeout` is attributable to the SHARED origin,
    #                          i.e. when the arm that timed out was charged for time it did not
    #                          spend -- which can only be the second arm;
    #   `arm_order_note`       the reason in one sentence, for the reader and for the adjudication.
    # A first-arm timeout is NOT order-sensitive: nothing had run before it.
    # ---------------------------------------------------------------------------------------
    wall_source = ('shared_single_allowance'
                   if any((results[a].get('budget_gate') or {}).get('wall_origin_shared_across_arms')
                          for a in ARMS) else 'per_arm_private_allowance')
    order_sensitive = [a for a in ARMS
                       if v2[a]['failure_reason'] == 'timeout'
                       and a != ARMS[0] and wall_source == 'shared_single_allowance']
    v2['arm_order'] = list(ARMS)
    v2['wall_budget_source'] = wall_source
    v2['order_sensitive'] = bool(order_sensitive)
    v2['order_sensitive_arms'] = list(order_sensitive)
    v2['second_arm'] = ARMS[1]
    if wall_source == 'shared_single_allowance':
        v2['arm_order_note'] = (
            'the arms ran in the frozen order %s and were charged from ONE monotonic origin, so the '
            'second arm (%s) received "10.0 s minus what %s spent".  A `timeout` on %s is therefore '
            'ATTRIBUTABLE TO THE SHARED ALLOWANCE AND THE ORDER, not to its scorer being slower or '
            'worse, and may never be reported as a mechanism advantage for %s.  A `timeout` on the '
            'FIRST arm (%s) is not order-sensitive: nothing had run before it.'
            % (list(ARMS), ARMS[1], ARMS[0], ARMS[1] or 'the second arm', ARMS[0], ARMS[0]))
    else:
        v2['arm_order_note'] = (
            'each arm held its OWN wall-clock allowance (the default per-gate origin), so no arm can '
            'be charged for time another arm spent and the order carries no timing confound')
    v2['v1_order_sensitivity'] = (
        'V1 (the first-step comparison) is ORDER-INVARIANT BY CONSTRUCTION: no scorer call happens '
        'before the first step, so at that point the second arm has spent nothing and no deadline can '
        'have fired.  Only V2 can be order-sensitive, and only through the shared wall clock.')

    # ---- V3 certified executable plan ------------------------------------------------------
    if certify_fn is None:
        v3 = {a: {'certified_executable_plan': 'NOT_RUN',
                  'reason': certify_not_run_reason or 'no certifier was supplied'} for a in ARMS}
        v3['any_certified'] = 'NOT_RUN'
    else:
        v3 = {a: {'certified_executable_plan': bool(results[a]['certified']
                                                    and results[a]['failure_reason'] is None
                                                    and results[a]['certified_plan_hash']),
                  'certified_plan_hash': results[a]['certified_plan_hash'],
                  'certifier': results[a]['certifier'],
                  'failure_reason': results[a]['failure_reason']} for a in ARMS}
        v3['any_certified'] = any(v3[a]['certified_executable_plan'] is True for a in ARMS)

    return {'results': results, 'audit': audit, 'V1': v1, 'V2': v2, 'V3': v3,
            'claim_limit': 'L1 only reports whether attributable ACTION DIVERGENCE exists. It is '
                           'not a gain, a significance test, or an online-utility claim, and a '
                           'first-step divergence is NOT a different executable plan.'}


# ---------------------------------------------------------------------------------------------
# replayable acceptance (deterministic: same inputs -> same verdicts)
# ---------------------------------------------------------------------------------------------
def replay_digest(report):
    """A digest of everything a replay must reproduce.  Timings are excluded: they are not results."""
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in sorted(obj.items())
                    if k not in ('budget', 'wall_seconds', 'span_wall_s', 'published_mono')
                    and not k.endswith('_wall_s')}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj
    return json_digest({'audit': strip(report['audit']), 'V1': strip(report['V1']),
                        'V2': strip(report['V2']), 'V3': strip(report['V3']),
                        'plans': {a: report['results'][a]['plan_after'] for a in ARMS},
                        'hashes': {a: report['results'][a]['plan_hash_after'] for a in ARMS}})
