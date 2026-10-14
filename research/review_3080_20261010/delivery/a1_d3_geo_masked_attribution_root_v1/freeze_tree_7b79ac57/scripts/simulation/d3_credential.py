#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B2-1: `CertificationCredentialV1` -- the triple-identity credential.

Authorised scope (user ruling 2026-10-14): build a credential that is COMPUTABLE, REFUSABLE and
INDEPENDENTLY TESTABLE, without taking over the real commit loop.  This module does not install
anything, does not touch `_commit` / `_plan_and_commit`, and makes no cross-process claim.

Three identities, and why all three are needed (measured in B1):
  * `state_hash`               -- the REAL post-prepare state.  The live decision path has no
                                  state identity of its own (B1 survey), so the assembly captures
                                  it in-process; `state_hash_scope` says so explicitly.
  * `contract_hash` /
    `effective_contract_hash` /
    `profile_hash`             -- measured to DIFFER for this project's frozen profile
                                  (raw 32124bfd... vs effective edc1b2ea...), and the env only
                                  carries the effective one, so the raw contract must be injected
                                  by the caller and its absence recorded, never faked.
  * `plan_hash_legacy` +
    `plan_digest_anchor_v1`    -- `plan_hash` hashes only (key, anchor_node, repr(suffix)) and
                                  truncates to 16 hex, so it is BLIND to `anchor_time` /
                                  `anchor_load` changes (B1 D4).  The legacy hash is kept for
                                  historical comparison and is never replaced or redefined.

`state_identity_information_scope` (user requirement): the state identity must depend only on
decision-time state, the already-revealed set and the contract identity.  Reading future order
attributes merely to compute the hash would poison the strict-online audit, so the scope is
DECLARED and AUDITED here rather than assumed.
"""
from __future__ import annotations

import hashlib

import numpy as np

from d3_state_assembly import (PLAN_DIGEST_SCHEMA, contract_identity, plan_digest_anchor_v1,
                              scenario_identity)

CREDENTIAL_SCHEMA = 'certification_credential_v1'
STATE_HASH_SCOPE = 'in_process_same_run_only'
REQUIRED_STATE_CAPTURE_ANCHOR = 'post_prepare'
VALID_ANCHORS = ('pre_prepare', 'post_prepare')

# Three-state quality outcome: never a bare True/False, because "not checkable online" is not
# "satisfied" (user ruling: 不能默认合格).
QUALITY_OK = 'CHECKED_SATISFIED'
QUALITY_FAILED = 'CHECKED_FAILED'
QUALITY_CAPABILITY_GAP = 'CAPABILITY_GAP_BLOCKING'

REQUIRED_FIELDS = ('credential_schema', 'state_hash', 'state_hash_scope', 'state_capture_anchor',
                   'contract_hash', 'raw_contract_reachable', 'effective_contract_hash',
                   'profile_hash', 'plan_hash_legacy', 'plan_digest_anchor_v1',
                   'plan_digest_anchor_schema', 'anchor_fields', 'inst_idx', 'event_id', 'clock',
                   'source_identity', 'certification_results', 'state_identity_information_scope')


def _sha256_of_array(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def state_identity_information_scope(env, inst_idx, clock, snapshot, plans=None):
    """DECLARE and AUDIT what the state identity depends on.

    Returns a report with:
      * `classified_keys`  -- every snapshot key bucketed as decision_time_state /
                              revealed_so_far / contract_identity / coordinate / replanner_state;
      * `visible_mask_is_causal` -- `visible_mask == {c : reveal_time[c] <= clock}`;
      * `unrevealed_customers_contributing` -- customers placed in a plan suffix while NOT yet
                              revealed at `clock` (must be empty);
      * `future_only_keys_present` -- snapshot keys that would carry future order attributes
                              (`reveal_time` / `demands` / `tw_start` / `tw_end` /
                              `initial_quality` / `temp_class`); must be empty, because the
                              capture does not copy per-customer static data at all;
      * a sensitivity proof that `visible_mask` IS an input to the hash.
    """
    reveal = np.asarray(env.reveal_time[inst_idx])
    vis_expected = np.array([bool(reveal[c] <= clock + 1e-6) for c in range(env.num_nodes)])
    vis_expected[0] = True
    vis_actual = np.asarray(snapshot['visible_mask'], dtype=bool).copy()
    if vis_actual.shape[0] == vis_expected.shape[0]:
        vis_actual[0] = True
    visible_causal = bool(np.array_equal(vis_actual, vis_expected))

    future_only = ('reveal_time', 'demands', 'tw_start', 'tw_end', 'initial_quality',
                   'temp_class', 'coords', 'dist_mat', 'service_time', 'routes', 'opt_costs')
    future_only_present = sorted(k for k in snapshot if k in future_only)

    revealed = {c for c in range(1, env.num_nodes) if bool(reveal[c] <= clock + 1e-6)}
    unrevealed_in_plans = []
    if plans:
        for vid, p in plans.items():
            for x in p.suffix:
                x = int(x)
                if x > 0 and x not in revealed:
                    unrevealed_in_plans.append({'vehicle_id': int(vid), 'customer': x})
    unrevealed_in_plans.sort(key=lambda r: (r['vehicle_id'], r['customer']))

    classified = {}
    for k in sorted(snapshot):
        if k in ('schema_version', 'instance_id', 'event_id', 'replan_count', 'clock', 'reveal_idx',
                 'num_vehicles', 'num_nodes', 'customer_universe'):
            bucket = 'coordinate'
        elif k == 'coldchain_contract_hash':
            bucket = 'contract_identity'
        elif k == 'replanner_state':
            bucket = 'replanner_state'
        elif k in ('visible_mask', 'served_mask'):
            bucket = 'revealed_so_far'
        else:
            bucket = 'decision_time_state'
        classified[k] = bucket

    # sensitivity proof: the visible set IS bound into the identity
    from recourse_snapshot import snapshot_state_hash
    base = snapshot_state_hash(snapshot)
    probe = dict(snapshot)
    probe.pop('state_hash', None)
    vm = np.asarray(snapshot['visible_mask'], dtype=bool).copy()
    flip = next((c for c in range(1, env.num_nodes) if bool(reveal[c] > clock + 1e-6)), None)
    probe['visible_mask'] = vm
    probe_hash_before = snapshot_state_hash(probe)
    if flip is not None:
        vm2 = vm.copy()
        vm2[flip] = True
        probe['visible_mask'] = vm2
    probe_hash_after = snapshot_state_hash(probe)

    return {
        'declared_scope': 'revealed_and_committable_state_only',
        'classified_keys': classified,
        'n_keys': len(classified),
        'visible_mask_is_causal': visible_causal,
        'future_only_keys_present_in_snapshot': future_only_present,
        'unrevealed_customers_contributing': unrevealed_in_plans,
        'unrevealed_customers_contributing_is_empty': not unrevealed_in_plans,
        'n_revealed_at_clock': len(revealed),
        'visible_mask_is_bound_into_hash': probe_hash_before != probe_hash_after,
        'probe_customer_flipped': flip,
        'base_state_hash_matches_snapshot': (base == snapshot.get('state_hash')),
        'note': ('the capture copies served_mask / visible_mask / vehicle state / traces / the '
                 'contract hash -- not the per-customer static tables -- so future order '
                 'attributes cannot enter the identity; this audit proves the visible set is '
                 'bound while no unrevealed customer contributes'),
    }


def make_credential(env, inst_idx, clock, event_id, snapshot, state_hash, plans,
                    certification_results, raw_contract=None, profile=None,
                    source_identity=None, state_capture_anchor=REQUIRED_STATE_CAPTURE_ANCHOR):
    """Build `CertificationCredentialV1`.

    `raw_contract` is INJECTED because it is not reachable from the env (the env carries the
    effective contract only).  When it is absent, `raw_contract_reachable` is False and
    `contract_hash` is None -- the effective hash is never passed off as the raw one.
    """
    from action_contract import plan_hash
    if state_capture_anchor not in VALID_ANCHORS:
        raise ValueError('unknown state_capture_anchor %r' % state_capture_anchor)
    eff = contract_identity(env)
    return {
        'credential_schema': CREDENTIAL_SCHEMA,
        'state_hash': state_hash,
        'state_hash_scope': STATE_HASH_SCOPE,
        'state_capture_anchor': state_capture_anchor,
        'contract_hash': (raw_contract.contract_hash if raw_contract is not None else None),
        'raw_contract_reachable': raw_contract is not None,
        'effective_contract_hash': eff['contract_hash'],
        'profile_hash': (profile.profile_hash if profile is not None else None),
        'plan_hash_legacy': plan_hash(plans) if plans else None,
        'plan_digest_anchor_schema': PLAN_DIGEST_SCHEMA,
        'plan_digest_anchor_v1': plan_digest_anchor_v1(plans) if plans else None,
        'anchor_fields': ({int(vid): [int(p.anchor_node), repr(float(p.anchor_time)),
                                      repr(float(p.anchor_load))]
                           for vid, p in sorted(plans.items())} if plans else {}),
        'inst_idx': int(inst_idx), 'event_id': int(event_id), 'clock': float(clock),
        'source_identity': source_identity or {},
        'certification_results': certification_results,
        'state_identity_information_scope': state_identity_information_scope(
            env, inst_idx, clock, snapshot, plans),
        'scenario_identity': scenario_identity(env),
    }


def validate_credential(cred, require_anchor=REQUIRED_STATE_CAPTURE_ANCHOR):
    """Structural completeness. Returns `(ok, reasons)`.

    Missing identity is a REFUSAL (`*_missing`), never a default pass.
    """
    reasons = []
    for f in REQUIRED_FIELDS:
        if f not in cred:
            reasons.append('%s_missing' % f)
    if reasons:
        return False, reasons
    if cred['credential_schema'] != CREDENTIAL_SCHEMA:
        reasons.append('credential_schema_unknown')
    for f in ('state_hash', 'effective_contract_hash'):
        v = cred.get(f)
        if not isinstance(v, str) or len(v) != 64:
            reasons.append('%s_missing' % f)
    if not isinstance(cred.get('plan_digest_anchor_v1'), str):
        reasons.append('plan_digest_anchor_v1_missing')
    if cred.get('state_hash_scope') != STATE_HASH_SCOPE:
        reasons.append('state_hash_scope_unexpected')
    if require_anchor is not None and cred.get('state_capture_anchor') != require_anchor:
        reasons.append('anchor_mismatch')
    if cred.get('raw_contract_reachable') is False and cred.get('contract_hash') is not None:
        reasons.append('raw_contract_inconsistent')
    return (not reasons), reasons


def credential_identity_check(cert, install):
    """The three equalities required before installation.

    Returns `(ok, reasons, detail)`.  Identity matching is NECESSARY BUT NOT SUFFICIENT: the
    caller must still check the deadline, the completeness of certification and (in B3) the
    actual install result.  This function deliberately cannot express those checks.
    """
    reasons, detail = [], {}
    for field, name in (('state_hash', 'state_hash_mismatch'),
                        ('effective_contract_hash', 'effective_contract_hash_mismatch'),
                        ('contract_hash', 'contract_hash_mismatch'),
                        ('profile_hash', 'profile_hash_mismatch'),
                        ('plan_digest_anchor_v1', 'plan_digest_anchor_mismatch'),
                        ('plan_hash_legacy', 'plan_hash_legacy_mismatch')):
        same = cert.get(field) == install.get(field)
        detail[field] = {'cert': cert.get(field), 'install': install.get(field), 'equal': same}
        if not same:
            # a None-vs-None "match" on the raw contract is acceptable; a None that appears on
            # one side only is a mismatch, and is caught here because equality fails.
            reasons.append(name)
    detail['anchor'] = {'cert': cert.get('state_capture_anchor'),
                        'install': install.get('state_capture_anchor')}
    if cert.get('state_capture_anchor') != install.get('state_capture_anchor'):
        reasons.append('anchor_mismatch')
    if cert.get('inst_idx') != install.get('inst_idx'):
        reasons.append('instance_mismatch')
    return (not reasons), reasons, detail
