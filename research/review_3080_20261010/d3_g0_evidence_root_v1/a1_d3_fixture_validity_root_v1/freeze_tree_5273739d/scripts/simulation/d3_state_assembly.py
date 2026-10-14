#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B1: READ-ONLY state-level assembly.

Authorised scope (user ruling 2026-10-14): a LIMITED B1 engineering implementation.
APPROVED here: construct C1-C5 state-level counterexamples on the real `StrictOnlineEnv` +
`VehicleState`; provide a READ-ONLY interface for post-prepare state capture and per-vehicle
mask-eligibility checking; verify committed / served / protected / reveal-invalidation; build a
ZERO-write-back no-op assembly path and check legal-input equivalence with the baseline; record
real state, candidate pools, legal actions and every failure reason.
NOT APPROVED and NOT implemented here: real environment install write-back, modifying `_commit`,
a composed certifier taking over production decisions, a cross-process state-hash protocol,
persistent `P_after_install`, training, or any effect experiment.

STOP CONDITION (user ruling): if the real environment state cannot be captured WITHOUT changing
production behaviour, stop and report the gap instead of editing the core env commit logic.
This module therefore adds NO behaviour: it only reads, and every read goes through existing,
already-sanctioned entry points:
  * `recourse_snapshot.capture_recourse_snapshot` / `restore_recourse_snapshot`
  * `strict_online_env.StrictOnlineEnv.prepare_decision_point` (on a CLONE, never on live state)
  * `dynmaskco_cc_context.mask_candidate_pool` / `decision_pool_from_vehicles`
  * `mpre_policy.enumerate_legal_actions`
  * `action_contract.plan_hash` / `build_vehicle_plans`

Why a clone is the enabler: `restore_recourse_snapshot(snapshot)` takes NO env argument and
returns `(vehicles, traces, served_mask)` (recourse_snapshot.py:214-315), so the assembly can
replay `prepare_decision_point` + an incumbent planner + legal enumeration on a detached copy
while the live run keeps running untouched. Using the live `vehicles` for those steps would
mutate the run, which is exactly what must not happen.

Two identities, deliberately separate (user ruling R1):
  * `pre_prepare_context`  -- what a decision-time model may legitimately see (strict-online
                              causality audit). The env's `snapshot_hook` fires BEFORE
                              `prepare_decision_point`, so this is the hook-time state.
  * `post_prepare_state`   -- the state certification and installation must agree on.
`strict_online_env.prepare_decision_point` (line 608) REWRITES `ready_time` (idle -> clock,
ready -> max(ready, clock)), so the two are different objects and must never be conflated.
Feeding full post-prepare state into a scorer would open a new information channel; that is
why the credential carries it but the visible context does not.

Diagnostics that must NOT be turned into failure conditions (user ruling R2): `get_reserved_customers`
includes EVERY vehicle's mutable suffix, while B-mask eligibility allows a replannable vehicle's
suffix to be re-planned. So `pool_B & reserved != {}` is NOT automatically a violation. The
intersection is REGISTERED as a diagnostic and each member is explained by vehicle replannability.
The real failure conditions are: committed, served, protected, or an immutable prefix.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

B1_SCOPE = 'read_only_state_level'
B1_WRITES_PRODUCTION_STATE = False
PLAN_DIGEST_SCHEMA = 'plan_digest_anchor_v1'

# Eligibility exclusion reasons (each is a REAL failure condition, per user ruling R2/R6-adjacent)
EXCLUSION_REASONS = ('committed', 'served', 'protected', 'not_b_in_eligible',
                     'not_visible', 'not_in_decision_pool')


class WritesForbiddenError(RuntimeError):
    """Raised when a read-only assembly step appears to have changed live state."""


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------
def plan_digest_anchor_v1(plans):
    """Audit digest that COVERS `anchor_time` / `anchor_load`, unlike `action_contract.plan_hash`.

    `plan_hash` hashes only (dict key, anchor_node, repr(tuple(suffix))) and truncates to 16 hex
    (action_contract.py:301-309), so two plans differing only in `anchor_time` hash identically.
    This digest is a NEW, independent audit identity (user ruling R4 pattern: add a versioned
    identity rather than changing the meaning of an existing one). It does NOT replace
    `plan_hash` anywhere; B1 only records both.
    """
    h = hashlib.sha256()
    h.update(PLAN_DIGEST_SCHEMA.encode())
    for vid in sorted(plans):
        p = plans[vid]
        h.update(record_bytes([int(vid), int(getattr(p, 'vehicle_id', vid)),
                               int(p.anchor_node), repr(float(p.anchor_time)),
                               repr(float(p.anchor_load)), [int(x) for x in p.suffix]]))
    return h.hexdigest()


def record_bytes(obj):
    """Deterministic, type-tagged encoding (no PYTHONHASHSEED dependence, no numpy ambiguity)."""
    h = hashlib.sha256()
    _update(h, obj)
    return h.digest()


def _update(h, v):
    if isinstance(v, np.ndarray):
        h.update(b'arr' + v.dtype.str.encode() + v.tobytes())
    elif isinstance(v, (bool, np.bool_)):
        h.update(b'bool1' if v else b'bool0')
    elif isinstance(v, (int, np.integer)):
        h.update(b'int' + str(int(v)).encode())
    elif isinstance(v, (float, np.floating)):
        h.update(b'flt' + repr(float(v)).encode())
    elif isinstance(v, str):
        h.update(b'str' + v.encode())
    elif isinstance(v, (list, tuple)):
        h.update(b'lst' + str(len(v)).encode())
        for x in v:
            _update(h, x)
    elif isinstance(v, dict):
        h.update(b'dct' + str(len(v)).encode())
        for k in sorted(v):
            _update(h, k)
            _update(h, v[k])
    elif v is None:
        h.update(b'none')
    else:
        h.update(b'oth' + repr(v).encode())


def contract_identity(env):
    """Contract identity is reachable from the ENV, never from the state (survey finding Q2).

    Returns None-valued fields (not a raised error) when the env has no cold-chain contract, so
    a caller can record "no contract" explicitly instead of guessing.
    """
    c = getattr(env, 'coldchain_contract', None)
    if c is None:
        return {'contract_hash': None, 'schema_version': None, 'shared_capacity': None}
    return {'contract_hash': c.contract_hash,
            'schema_version': getattr(c, 'schema_version', None),
            'shared_capacity': float(c.operational.shared_vehicle_capacity)}


def scenario_identity(env):
    """Dataset/scenario identity reachable from the env (arrays, not files)."""
    return {
        'num_instances': int(env.num_instances),
        'num_nodes': int(env.num_nodes),
        'capacity': float(env.capacity),
        'num_vehicles': int(env.num_vehicles),
        'booking_horizon': (None if env.booking_horizon is None
                            else float(env.booking_horizon)),
        'dist_mat_shape': list(np.asarray(env.dist_mat).shape),
        'dist_mat_bytes_sha': hashlib.sha256(
            np.ascontiguousarray(np.asarray(env.dist_mat)).tobytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# state capture / clone (read-only)
# ---------------------------------------------------------------------------
def capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                           served_mask, all_customers=None):
    """In-process state capture. Returns `(snapshot, state_hash)`.

    Never relies on `env.snapshot_hook` being registered: the assembly calls the existing
    capture function directly, because the hook defaults to None (strict_online_env.py:179) and
    the live decision path has no state identity of its own (survey finding Q1).
    """
    from recourse_snapshot import capture_recourse_snapshot
    snap = capture_recourse_snapshot(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                    traces, served_mask, all_customers=all_customers)
    return snap, snap['state_hash']


def clone_state(snapshot):
    """Detached `(vehicles, traces, served_mask)` copy of a captured state.

    `restore_recourse_snapshot` takes no `env`, so it cannot mutate the live environment. It
    also refuses snapshots whose `state_hash` is missing or inconsistent (recourse_snapshot.py:219-225),
    so a corrupt capture cannot silently become the basis of a claim.
    """
    from recourse_snapshot import restore_recourse_snapshot
    return restore_recourse_snapshot(snapshot)


def live_state_fingerprint(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                           served_mask, all_customers=None):
    """Cheap fingerprint of the LIVE decision state, used to prove the assembly wrote nothing.

    Computed through the same sanctioned capture path, so it covers `env.replan_count`,
    `env.force_suffix` and `env.replanner.export_state()` as well as every vehicle field: if a
    read-only step accidentally touched any of them, the fingerprint would move.
    """
    return capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                                 served_mask, all_customers=all_customers)[1]


# ---------------------------------------------------------------------------
# post-prepare preparation on a CLONE
# ---------------------------------------------------------------------------
def prepare_clone(env, inst_idx, clock, clone_vehicles, clone_served_mask, visible_ids,
                  continuation=None):
    """Run `prepare_decision_point` + an incumbent planner on the CLONE only.

    `continuation` must be a FRESH planner instance (e.g. `jf1h_repair.make_continuation()`),
    never `env.replanner`: a planner carries its own `deferred_customers` / `status_inst`, and
    driving the live one from a probe would mutate the run.
    Returns `(post_prepare_vehicles, plans, replan_ids)`.
    """
    from action_contract import build_vehicle_plans
    env.prepare_decision_point(clock, clone_vehicles)          # mutates the CLONE list only
    if continuation is None:
        from jf1h_repair import make_continuation
        continuation = make_continuation()
    replan_ids = {v.vehicle_id for v in clone_vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, clone_vehicles, clone_served_mask,
                      [int(c) for c in visible_ids], replan_ids=replan_ids)
    return clone_vehicles, build_vehicle_plans(env, inst_idx, clone_vehicles), replan_ids


# ---------------------------------------------------------------------------
# mask eligibility: per-vehicle resolution, one state version (user clarification, 2026-10-14)
# ---------------------------------------------------------------------------
def mask_eligibility(env, inst_idx, clock, vehicles, served_mask, visible_ids, plans,
                     mutable_ids, protected=frozenset()):
    """Resolve mask eligibility PER VEHICLE at ONE state version.

    The user's clarification: a customer may have been in a replannable vehicle's suffix but have
    been committed by another vehicle before certification, so a union-of-suffixes membership
    test can wrongly accept it. Eligibility is therefore resolved by vehicle ownership:
        B-mask eligible  = the customer appears in the CURRENT suffix of some vehicle in
                           `mutable_ids` (recorded with the owning vehicle ids), AND
        not committed    = it is not any vehicle's live `committed_next`, AND
        not served       = `served_mask` is False and it is in no `served_route`, AND
        not protected    = it is in no protected vehicle's suffix.
    All four are computed from the SAME `vehicles`/`plans`/`served_mask` objects.

    Returns a report dict; `eligible` is the sorted customer list that satisfies all four.
    """
    mutable_ids = set(int(v) for v in (mutable_ids or ()))
    protected = set(int(v) for v in (protected or ()))
    visible = set(int(c) for c in visible_ids)

    b_by_customer = {}
    for v in vehicles:
        if int(v.vehicle_id) not in mutable_ids:
            continue
        plan = plans.get(int(v.vehicle_id))
        suffix = list(plan.suffix) if plan is not None else list(v.mutable_suffix)
        for x in suffix:
            x = int(x)
            if x > 0:
                b_by_customer.setdefault(x, []).append(int(v.vehicle_id))

    committed_by_customer = {}
    for v in vehicles:
        cn = v.committed_next
        if cn not in (None, 0):
            committed_by_customer[int(cn)] = int(v.vehicle_id)
    committed_status = {int(v.vehicle_id) for v in vehicles if v.status == 'committed'}

    served_from_mask = {int(c) for c in visible if bool(served_mask[int(c)])}
    served_from_route = set()
    for v in vehicles:
        for x in v.served_route:
            if int(x) > 0:
                served_from_route.add(int(x))
    served = served_from_mask | served_from_route

    protected_tail = {}
    for vid in protected:
        plan = plans.get(vid)
        suffix = list(plan.suffix) if plan is not None else next(
            (list(v.mutable_suffix) for v in vehicles if int(v.vehicle_id) == vid), [])
        for x in suffix:
            x = int(x)
            if x > 0:
                protected_tail.setdefault(x, []).append(vid)

    reasons = {}

    def _note(c, why):
        reasons.setdefault(int(c), [])
        if why not in reasons[int(c)]:
            reasons[int(c)].append(why)

    universe = sorted(visible)
    for c in universe:
        if c not in b_by_customer:
            _note(c, 'not_b_in_eligible')
        if c in committed_by_customer:
            _note(c, 'committed')
        if c in served:
            _note(c, 'served')
        if c in protected_tail:
            _note(c, 'protected')
    eligible = [c for c in universe if not reasons.get(c)]

    return {
        'state_version': {'inst_idx': int(inst_idx), 'clock': float(clock)},
        'mutable_ids': sorted(mutable_ids),
        'protected_ids': sorted(protected),
        'b_eligible_by_customer_vehicle': {int(k): sorted(v) for k, v in b_by_customer.items()},
        'committed_next_by_customer_vehicle': {int(k): int(v) for k, v in
                                               committed_by_customer.items()},
        'committed_vehicle_ids': sorted(committed_status),
        'served_from_mask_count': len(served_from_mask),
        'served_from_route_count': len(served_from_route),
        'served_nodes': sorted(served),
        'served_mask_vs_served_route_discrepancy': sorted(served_from_mask ^ served_from_route),
        'protected_tail_by_customer_vehicle': {int(k): sorted(v) for k, v in
                                                protected_tail.items()},
        'exclusion_reasons': {int(k): sorted(v) for k, v in reasons.items()},
        'eligible': eligible,
        'n_eligible': len(eligible),
    }


# ---------------------------------------------------------------------------
# pools: authoritative + diagnostics (never shrink the main pool)
# ---------------------------------------------------------------------------
def pool_report(env, inst_idx, clock, vehicles, served_mask, visible_ids, plans, mutable_ids,
                protected=frozenset(), eligibility=None):
    """Authoritative pool + diagnostics, with the empty pool PRESERVED.

    Authoritative = `mask_candidate_pool` (the registered B-mask domain). `get_reserved_customers`
    is reported as a DIAGNOSTIC ONLY and is never intersected into the main pool (user ruling R2):
    it locks every vehicle's mutable suffix, whereas B semantics permits re-planning a
    replannable vehicle's suffix, so a non-empty intersection is not a violation.
    """
    from dynmaskco_cc_context import (mask_candidate_pool as _mcp,
                                      decision_pool_from_vehicles as _dp)
    pool_auth = [int(c) for c in _mcp(vehicles, served_mask, visible_ids,
                                     protected=protected, plans=plans)]
    pool_dec = [int(c) for c in _dp(vehicles, served_mask, visible_ids)]
    reserved = sorted(int(c) for c in env.get_reserved_customers(vehicles))
    elig = sorted(int(c) for c in (eligibility or {}).get('eligible', []))
    return {
        'authoritative_rule': 'mask_candidate_pool',
        'pool_B': pool_auth,                 # returned AS-IS, even when empty
        'empty': len(pool_auth) == 0,
        'n_pool_B': len(pool_auth),
        'decision_pool': pool_dec,
        'n_decision_pool': len(pool_dec),
        'reserved_diagnostic_only': reserved,
        'diag_pool_B_inter_reserved': sorted(set(pool_auth) & set(reserved)),
        'diag_decision_pool_minus_pool_B': sorted(set(pool_dec) - set(pool_auth)),
        'diag_eligible_minus_pool_B': sorted(set(elig) - set(pool_auth)),
        'diag_pool_B_minus_eligible': sorted(set(pool_auth) - set(elig)),
        'note': ('intersection with reserved is a DIAGNOSTIC: a replannable vehicle\'s suffix is '
                 'legitimately re-plannable, so it is not by itself a violation'),
    }


def collect_legal_order(env, inst_idx, plans, remaining, mutable_ids):
    """Legal actions from the SHARED enumerator, recorded in order with resolved vehicle ids."""
    from mpre_policy import enumerate_legal_actions
    legal = enumerate_legal_actions(env, inst_idx, plans, list(remaining), set(mutable_ids))
    return [{'customer': int(c), 'slot_kind': a.slot.kind, 'slot_anchor': int(a.slot.anchor),
             'position': int(a.position), 'incremental_distance': float(cand.incremental_distance)}
            for (c, a, cand) in legal]


# ---------------------------------------------------------------------------
# B2-shaped credential (B1 only COMPUTES and COMPARES it; nothing is gated on it)
# ---------------------------------------------------------------------------
def decision_credential(env, inst_idx, clock, event_id, state_hash, plans,
                        pre_prepare_hash=None):
    """The triple-identity record B2 will later gate on. B1 never gates anything with it."""
    from action_contract import plan_hash
    return {
        'credential_schema': 'b1_decision_credential_v1',
        'state_hash': state_hash,
        'state_hash_scope': 'in_process_same_run_only',   # user ruling R4: NOT cross-process
        'state_capture_anchor': 'post_prepare',
        'pre_prepare_context_hash': pre_prepare_hash,
        'contract': contract_identity(env),
        'plan_hash': plan_hash(plans),
        'plan_hash_anchor_schema': PLAN_DIGEST_SCHEMA,
        'plan_hash_anchor': plan_digest_anchor_v1(plans),
        'anchor_fields': {int(vid): [int(p.anchor_node), repr(float(p.anchor_time)),
                                     repr(float(p.anchor_load))]
                          for vid, p in sorted(plans.items())},
        'inst_idx': int(inst_idx), 'clock': float(clock), 'event_id': int(event_id),
    }


def credential_diff(a, b):
    """Field-by-field difference between two credentials, for state-drift detection."""
    keys = ('state_hash', 'pre_prepare_context_hash', 'plan_hash', 'plan_hash_anchor',
            'plan_hash_anchor_schema')
    out = {k: (a.get(k) != b.get(k)) for k in keys}
    out['contract_changed'] = a.get('contract') != b.get('contract')
    out['clock'] = [a.get('clock'), b.get('clock')]
    out['event_id'] = [a.get('event_id'), b.get('event_id')]
    out['any_relevant_change'] = any(v for k, v in out.items() if isinstance(v, bool))
    return out
