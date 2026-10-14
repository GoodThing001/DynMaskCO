#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT1: the AUTHORISED COMMIT TRANSACTION (isolated implementation).

Authorised scope (user ruling 2026-10-14): implement the transaction **in isolation** -- new module,
one-time credential registry, failure-state receipt, isolated-copy tests.  It is explicitly NOT
registered as `StrictOnlineEnv`'s real commit entry, and `B3-AT2` (production wiring / real
write-back) is NOT authorised.

What this module is responsible for (the four design requirements):

  R1  SINGLE AUTHORISED WRITE INTERFACE.  The gate DECIDES, the transaction WRITES, and the two are
      separate objects: `commit()` takes a `gate_decision_fn` and refuses to publish unless that
      decision allows.  A caller that passes the gate and then calls `_commit` on its own is *not*
      stopped by this module -- preventing that belongs to B3-AT2, which owns the write capability.
  R2  ONE-TIME CREDENTIAL.  Every authorisation is a ticket bound to (decision point, state version,
      plan identity, certifier).  Consumption lives in `CommitTicketRegistry`, so a NEW transaction
      object reusing the same ticket is refused -- the replay protection is not per-object.
  R3  RE-VERIFY AND WRITE ARE INSEPARABLE.  Everything (re-verify, ticket consumption, candidate
      construction, publish) happens inside one `_Boundary`.  Inside it the state version is
      re-checked immediately before the publish, so a state change that lands between the
      re-verification and the write is caught.  Adding one more hash comparison outside a boundary
      does not achieve this.
  R4  FAILURE CONSISTENCY.  The candidate state is fully constructed and validated on a detached
      staging structure; the default publish is then a pure attribute assignment that cannot fail
      part-way.  A deliberately NON-atomic publisher that raises is reported as `PARTIAL_COMMIT`
      with the real per-vehicle fields -- never as a rollback, because there is no durable
      transaction mechanism and no state-level restore is proven.

Honest limits, stated here rather than discovered later:
  * `CommitTicketRegistry` is IN-PROCESS ONLY (`scope='in_process_module_singleton'`).  No
    cross-process or durable claim is made.
  * the boundary detects a state change; it cannot exclude a writer that mutates *during* the
    publish itself.  True exclusivity needs AT2's ownership of the write capability.
  * this module writes only to DETACHED copies of a real `StrictOnlineEnv` state.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace

COMMIT_SCHEMA = 'authorized_commit_transaction_v1'
TICKET_SCHEMA = 'commit_ticket_v1'
BOUNDARY_SCHEMA = 'commit_boundary_v1'

OUTCOME_COMMITTED = 'COMMITTED'
OUTCOME_REFUSED = 'REFUSED_NO_MUTATION'
OUTCOME_WAIT = 'WAIT_BY_ENV_PROTOCOL'
OUTCOME_BLOCKED_D8 = 'BLOCKED_BY_D8'
OUTCOME_PARTIAL = 'PARTIAL_COMMIT'
OUTCOME_UNKNOWN = 'OUTCOME_UNKNOWN'
OUTCOMES = (OUTCOME_COMMITTED, OUTCOME_REFUSED, OUTCOME_WAIT, OUTCOME_BLOCKED_D8,
            OUTCOME_PARTIAL, OUTCOME_UNKNOWN)

# Publish-state vocabulary (user ruling 2026-10-14, B3-AT2A section 2).  These are QUALIFICATIONS,
# not decoration: `PUBLISHING` means real state modification has begun and overall consistency is
# NOT yet confirmed, and `COMMITTED_VERIFIED` is reserved for a publish whose post-state was
# independently re-read.  This module records the trace, and it stops at `COMMITTED`: it has no
# independent post-commit re-read, so it must never claim `COMMITTED_VERIFIED`.
PUBLISH_STAGED_VALIDATED = 'STAGED_VALIDATED'
PUBLISH_PUBLISHING = 'PUBLISHING'
PUBLISH_COMMITTED = 'COMMITTED'
PUBLISH_COMMITTED_VERIFIED = 'COMMITTED_VERIFIED'
PUBLISH_PARTIAL = 'PARTIAL_COMMIT'
PUBLISH_UNKNOWN = 'OUTCOME_UNKNOWN'
PUBLISH_STATES = (PUBLISH_STAGED_VALIDATED, PUBLISH_PUBLISHING, PUBLISH_COMMITTED,
                  PUBLISH_COMMITTED_VERIFIED, PUBLISH_PARTIAL, PUBLISH_UNKNOWN)

# IMPLEMENTATION MODE, NOT A SYSTEM PROPERTY.  The user ruling: "在拥有充分的发布线性化证据前，
# `atomic` 只能作为实现模式名，不能作为已建立的系统性质."  The mode is therefore named after what
# the code does (stage everything, validate, then one pass) and the old string is accepted only as a
# deprecated alias so no caller silently loses the corrected wording.
PUBLISH_MODE = 'staged_then_single_pass'
PUBLISH_MODE_LEGACY_ALIAS = 'atomic'
PUBLISH_MODE_NOTE = ('implementation mode name only: staging failures publish nothing and the '
                     'tested failure injections left no partial publish, but this module does NOT '
                     'claim that a multi-vehicle publish is unobservable in intermediate states, '
                     'nor that an arbitrary mid-publish exception leaves no partial state')

WRITE_ATTRS = ('mutable_suffix', 'committed_next')


# ---------------------------------------------------------------------------------------------
# plan identity
# ---------------------------------------------------------------------------------------------
def commit_identity(plans):
    """Identity of a plan set that INCLUDES `anchor_time` and `anchor_load`.

    `action_contract.plan_hash` hashes only (vid, anchor_node, suffix), so it is blind to the anchor
    time and load -- which is exactly how the B3-AC `AC-R7` exclusivity gap arises.  The transaction
    therefore uses this identity and never `plan_hash` alone for authorisation.
    """
    h = hashlib.sha256()
    for vid in sorted(plans):
        p = plans[vid]
        h.update(('%s|%s|%.12g|%.12g|%s;' % (int(vid), int(p.anchor_node), float(p.anchor_time),
                                            float(p.anchor_load), tuple(int(x) for x in p.suffix))
                 ).encode())
    return h.hexdigest()[:32]


def legacy_plan_hash(plans):
    from action_contract import plan_hash
    return plan_hash(plans)


def state_version(env, inst, clock, event, reveal_idx, cv, ct, csm, all_customers):
    """State version used for BOTH the credential anchor and the in-boundary re-check.

    It must be the same function as the credential's state hash, otherwise the transaction compares
    two different digests.  Measured defect of the first version: the transaction used
    `live_state_fingerprint` (which returns the snapshot's own `state_hash`) while the credential was
    built from `snapshot_state_hash(snapshot)`; the two are not interchangeable, so every row
    reported a spurious `state_unchanged=False` (and the fixture assertion
    `reprepared_state_matches_the_fixture_post_hash` was false for a state that had not moved).
    """
    from d3_state_assembly import capture_decision_state
    from recourse_snapshot import snapshot_state_hash
    snap, _h = capture_decision_state(env, inst, clock, event, reveal_idx, cv, ct, csm,
                                     all_customers)
    return snapshot_state_hash(snap)


# ---------------------------------------------------------------------------------------------
# R2: one-time authorisation registry
# ---------------------------------------------------------------------------------------------
class CommitTicketRegistry(object):
    """IN-PROCESS registry of issued/consumed commit authorisations.

    Consumption is keyed on the ticket id, so reuse is refused even from a brand-new transaction
    object.  No persistence, no cross-process claim.
    """

    scope = 'in_process_module_singleton'

    def __init__(self):
        self._issued = {}
        self._consumed = {}

    def issue(self, *, inst, event, clock, state_hash, commit_id, certifier, plan_hash=None):
        raw = '%s|%s|%.9f|%s|%s|%s' % (inst, event, float(clock), state_hash, commit_id, certifier)
        ticket_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        rec = {'ticket_schema': TICKET_SCHEMA, 'ticket_id': ticket_id,
               'inst': int(inst), 'event': int(event), 'clock': float(clock),
               'state_hash': state_hash, 'commit_id': commit_id, 'certifier': certifier,
               'plan_hash_legacy': plan_hash}
        self._issued[ticket_id] = rec
        return dict(rec)

    def consume(self, ticket_id, *, consumer):
        if ticket_id in self._consumed:
            return False, {'reason': 'ticket_already_consumed',
                           'first_consumer': self._consumed[ticket_id]['consumer'],
                           'attempted_consumer': consumer,
                           'consumed_at': self._consumed[ticket_id]['sequence']}
        self._consumed[ticket_id] = {'consumer': consumer, 'sequence': len(self._consumed) + 1}
        return True, {'reason': None, 'sequence': self._consumed[ticket_id]['sequence']}

    def state_of(self, ticket_id):
        if ticket_id in self._consumed:
            return 'consumed'
        if ticket_id in self._issued:
            return 'issued'
        return 'unknown'

    def snapshot(self):
        return {'scope': self.scope, 'n_issued': len(self._issued), 'n_consumed': len(self._consumed),
                'issued': sorted(self._issued), 'consumed': sorted(self._consumed),
                'note': ('in-process only: no nonce survives the process, and no cross-process '
                         'claim is made')}


DEFAULT_REGISTRY = CommitTicketRegistry()


# ---------------------------------------------------------------------------------------------
# runtime write monitor
# ---------------------------------------------------------------------------------------------
class WriteMonitor(object):
    """In-process instrument around the live write primitives.

    AST analysis can only see the calls and inheritance it models; dynamic attribute access and
    external injection need a runtime record.  This monitor wraps `_commit` / `_plan_and_commit` on
    ONE env instance, records every invocation with its caller and whether it happened inside the
    transaction boundary, and restores the originals on exit.

    DETECTION IS NOT PREVENTION: a recorded out-of-transaction write means the probe observed a
    bypass, not that the transaction blocked one.
    """

    def __init__(self, env):
        self.env = env
        self.inside_transaction = False
        self.records = []
        self._orig = {}

    def _caller(self):
        for fr in inspect.stack()[2:]:
            mod = fr.frame.f_globals.get('__name__', '')
            if mod != __name__:
                return '%s:%d' % (fr.filename.replace('\\', '/'), fr.lineno)
        return 'unknown'

    def __enter__(self):
        self._orig['_commit'] = self.env._commit
        self._orig['_plan_and_commit'] = self.env._plan_and_commit
        env = self.env
        mon = self

        def _commit(inst_idx, clock, v, traces, node):
            mon.records.append({'primitive': '_commit', 'caller': mon._caller(),
                                'inside_transaction': bool(mon.inside_transaction),
                                'vehicle_id': int(v.vehicle_id), 'node': (None if node is None
                                                                          else int(node)),
                                'clock': float(clock)})
            return mon._orig['_commit'](inst_idx, clock, v, traces, node)

        def _plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers):
            mon.records.append({'primitive': '_plan_and_commit', 'caller': mon._caller(),
                                'inside_transaction': bool(mon.inside_transaction),
                                'vehicle_id': None, 'node': None, 'clock': float(clock),
                                'n_vehicles': len(vehicles)})
            return mon._orig['_plan_and_commit'](inst_idx, clock, vehicles, traces, served_mask,
                                                 all_customers)

        env._commit = _commit
        env._plan_and_commit = _plan_and_commit
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, fn in self._orig.items():
            setattr(self.env, name, fn)
        return False

    def out_of_transaction(self):
        return [r for r in self.records if not r['inside_transaction']]

    def snapshot(self):
        return {'n_records': len(self.records), 'records': self.records,
                'n_out_of_transaction': len(self.out_of_transaction()),
                'detection_is_not_prevention': True}


# ---------------------------------------------------------------------------------------------
# R1/R3/R4: the transaction
# ---------------------------------------------------------------------------------------------
class AuthorizedCommitTransaction(object):
    """The single authorised write interface for this scope.

    `registered_on_env = False` is a class attribute, and the AT1 tests assert mechanically that no
    module outside `scripts/tests` imports this module and that no env in the batch has it
    registered -- i.e. it does not take over the production path.
    """

    registered_on_env = False
    module_manages_writes = True

    def __init__(self, *, registry=None, monitor=None, name='authorized-commit-tx'):
        self.registry = registry if registry is not None else DEFAULT_REGISTRY
        self.monitor = monitor
        self.name = name
        self.commits = 0
        self.partial_commits = 0
        self.refusals = 0
        self.wait_seen = 0

    # ---- R3: the boundary -------------------------------------------------------------------
    class _Boundary(object):
        SCHEMA = BOUNDARY_SCHEMA

        def __init__(self, monitor):
            self.monitor = monitor
            self.opened = False
            self.closed = False

        def __enter__(self):
            self.opened = True
            if self.monitor is not None:
                self.monitor.inside_transaction = True
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.monitor is not None:
                self.monitor.inside_transaction = False
            self.closed = True
            return False

    def _boundary(self):
        return self._Boundary(self.monitor)

    def _finalize(self, base, env, inst, clock, event, reveal_idx, cv, ct, csm, all_customers,
                  **extra):
        """Fill the after-state evidence on EVERY return path.

        Measured defect of the first version: the refusal paths returned before computing
        `state_hash_after`, so `state_unchanged` evaluated to `None == hash` -> False and the matrix
        reported a state change for refusals that had written nothing.  A refusal must be able to
        prove "nothing moved", so the after-state is computed here for every outcome.
        """
        base.update(extra)
        try:
            base['state_hash_after'] = state_version(env, inst, clock, event, reveal_idx, cv, ct,
                                                     csm, all_customers)
        except BaseException as exc:                                      # noqa: BLE001
            base['state_hash_after'] = None
            base['state_hash_after_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:80])
        if base.get('state_hash_before') is not None:
            base['state_unchanged'] = (base['state_hash_after'] == base['state_hash_before'])
        base['evidence_after'] = self._field_evidence(cv)
        if self.monitor is not None:
            base['monitor'] = self.monitor.snapshot()
        return base

    # ---- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _pending(plans):
        return {int(vid): [int(x) for x in p.suffix if int(x) > 0]
                for vid, p in plans.items() if len(p.suffix) > 0}

    @staticmethod
    def _field_evidence(vehicles):
        out = []
        for v in sorted(vehicles, key=lambda x: int(x.vehicle_id)):
            out.append({'vehicle_id': int(v.vehicle_id), 'status': v.status,
                        'current_node': int(v.current_node),
                        'ready_time': float(v.ready_time),
                        'committed_next': (None if v.committed_next is None
                                           else int(v.committed_next)),
                        'mutable_suffix': [int(x) for x in v.mutable_suffix]})
        return out

    @staticmethod
    def _validate_candidate(staged, served_mask, n_nodes):
        """Candidate validation BEFORE any write: nothing served, no duplicates, in range."""
        problems = []
        seen = {}
        for vid, suffix in sorted(staged.items()):
            for c in suffix:
                if c <= 0 or c >= n_nodes:
                    problems.append({'why': 'out_of_range', 'vehicle_id': vid, 'customer': c})
                if bool(served_mask[c]):
                    problems.append({'why': 'already_served', 'vehicle_id': vid, 'customer': c})
                seen[c] = seen.get(c, 0) + 1
        for c, k in sorted(seen.items()):
            if k > 1:
                problems.append({'why': 'duplicate_in_candidate', 'customer': c, 'count': k})
        return problems

    # ---- the commit -------------------------------------------------------------------------
    def commit(self, *, env, inst, clock, event, cv, ct, csm, plans, all_customers,
               cert_result, cred_cert, cred_install, gate_decision_fn, deadline_ok=True,
               publish=PUBLISH_MODE, publish_fn=None, pre_publish_hook=None,
               expected_commit_id=None, budget_role='FORMAL', expected_ticket=None,
               state_hash_before=None, reveal_idx=0):
        """Decide outside, write inside one boundary.  Never touches the live environment."""
        certifier = 'compose_certification+pre_install_reverify'
        staged_all = self._pending(plans)
        base = {
            'commit_schema': COMMIT_SCHEMA,
            'not_a_production_path': True,
            'registered_on_env': bool(self.registered_on_env),
            'budget_role': budget_role,
            'decision_point': {'inst': int(inst), 'event': int(event), 'clock': float(clock)},
            'gate': None, 'refusals': [], 'ticket': None, 'ticket_state': None,
            'commit_id': commit_identity(plans),
            'plan_hash_legacy': legacy_plan_hash(plans),
            'state_hash_before': state_hash_before,
            'state_hash_after': None, 'state_unchanged': None,
            'evidence_before': self._field_evidence(cv), 'evidence_after': None,
            'evidence_source': 'live_vehicle_fields',
            'boundary': {'schema': BOUNDARY_SCHEMA, 'opened': False, 'closed': False,
                         'state_recheck_inside_boundary': False},
            'candidate_validation': None,
            'monitor': None,
            'warnings': [],
        }

        # ---- WAIT first (H3): a WAIT is not a refused write ---------------------------------
        if not staged_all:
            self.wait_seen += 1
            base.update({'outcome': OUTCOME_WAIT, 'counts_as_refusal_capability': False,
                         'why': ('no pending action exists, so there is no refused WRITE to '
                                 'attribute; this must never be counted as refusal capability')})
            return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                   all_customers)

        # ---- R1: the gate decides -----------------------------------------------------------
        decision = gate_decision_fn(cert_result=cert_result, cred_cert=cred_cert,
                                   cred_install=cred_install, deadline_ok=deadline_ok)
        base['gate'] = decision
        if not decision.get('allow'):
            gaps = list(cert_result.get('blocking_capability_gaps') or ())
            only_quality = (bool(gaps) and not list(cert_result.get('reasons') or ())
                            and all(g == 'quality_salability' for g in gaps))
            self.refusals += 1
            base.update({'outcome': OUTCOME_BLOCKED_D8 if only_quality else OUTCOME_REFUSED,
                         'refusals': list(decision.get('refusals') or []),
                         'counts_as_refusal_capability': ('quality_dimension_only' if only_quality
                                                          else True),
                         'why': ('the gate refused outside the boundary, so nothing was written; '
                                 'the denial is attributed to the gate reasons above')})
            return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                   all_customers)

        # ---- R3: everything below happens inside ONE boundary -------------------------------
        with self._boundary() as b:
            base['boundary']['opened'] = True
            h_at_reverify = None
            if state_hash_before is not None:
                h_at_reverify = state_version(env, inst, clock, event, reveal_idx, cv, ct, csm,
                                             all_customers)
            # TEST SEAM: runs INSIDE the boundary, between the re-verification and the write.
            if pre_publish_hook is not None:
                pre_publish_hook(cv, csm)
            h_before_publish = None
            if state_hash_before is not None:
                h_before_publish = state_version(env, inst, clock, event, reveal_idx, cv, ct,
                                                 csm, all_customers)
                base['boundary']['state_recheck_inside_boundary'] = True
                base['boundary']['state_hash_at_reverify'] = h_at_reverify
                base['boundary']['state_hash_before_publish'] = h_before_publish
                if h_before_publish != state_hash_before:
                    self.refusals += 1
                    base.update({
                        'outcome': OUTCOME_REFUSED,
                        'refusals': ['state_changed_inside_boundary'],
                        'counts_as_refusal_capability': True,
                        'ticket_state': 'unconsumed_after_refusal',
                        'why': ('the state moved between the re-verification and the write, and the '
                                'boundary caught it BEFORE publishing; nothing was written')})
                    base['boundary']['closed'] = b.closed
                    return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                          all_customers)

            # identity: the transaction must not rely on the anchor-blind legacy hash
            if expected_commit_id is not None and expected_commit_id != base['commit_id']:
                self.refusals += 1
                base.update({'outcome': OUTCOME_REFUSED,
                             'refusals': ['commit_identity_mismatch'],
                             'counts_as_refusal_capability': True,
                             'ticket_state': 'unconsumed_after_refusal',
                             'why': ('`commit_identity` (which includes anchor_time / anchor_load) '
                                     'differs from the authorised identity; `plan_hash_legacy` '
                                     'alone would NOT have detected this')})
                base['boundary']['closed'] = b.closed
                return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                       all_customers)

            # ticket: issued, then consumed INSIDE the boundary, in the shared registry
            ticket = expected_ticket or self.registry.issue(
                inst=inst, event=event, clock=clock, state_hash=state_hash_before,
                commit_id=base['commit_id'], certifier=certifier,
                plan_hash=base['plan_hash_legacy'])
            base['ticket'] = dict(ticket)
            ok, info = self.registry.consume(ticket['ticket_id'], consumer=self.name)
            if not ok:
                self.refusals += 1
                base.update({'outcome': OUTCOME_REFUSED,
                             'refusals': ['ticket_already_consumed'],
                             'counts_as_refusal_capability': True,
                             'ticket_state': 'consumed',
                             'consume_info': info,
                             'why': ('one authorisation is consumed once; a NEW transaction object '
                                     'reusing the same ticket is refused because consumption lives '
                                     'in the shared registry, not in the transaction object')})
                base['boundary']['closed'] = b.closed
                return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                       all_customers)
            base['ticket_state'] = 'consumed_in_boundary'
            base['consume_info'] = info

            # candidate built and VALIDATED before any write
            problems = self._validate_candidate(staged_all, csm, env.num_nodes)
            base['candidate_validation'] = {'problems': problems, 'staged': staged_all}
            if problems:
                self.refusals += 1
                base.update({'outcome': OUTCOME_REFUSED,
                             'refusals': ['candidate_invalid'],
                             'counts_as_refusal_capability': True,
                             'ticket_state': 'consumed_then_refused_candidate_invalid',
                             'why': ('the candidate failed validation BEFORE publishing; the '
                                     'authorisation is recorded as consumed on a refusal path so '
                                     'it cannot be retried silently')})
                base['boundary']['closed'] = b.closed
                return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                       all_customers)

            by_id = {int(v.vehicle_id): v for v in cv}
            base['publish_mode'] = PUBLISH_MODE
            base['publish_mode_note'] = PUBLISH_MODE_NOTE
            if publish in (PUBLISH_MODE, PUBLISH_MODE_LEGACY_ALIAS):
                # R4: build the complete new state first, then a single pass that performs no
                # per-vehicle failure-prone work.  NOTE the vocabulary: this reaches `COMMITTED`,
                # NOT `COMMITTED_VERIFIED` -- there is no independent post-commit re-read here.
                base['publish_state_trace'] = [PUBLISH_STAGED_VALIDATED, PUBLISH_PUBLISHING]
                new_suffix = {vid: list(sfx) for vid, sfx in staged_all.items()}
                for vid, v in by_id.items():
                    v.mutable_suffix = list(new_suffix.get(vid, []))
                    v.needs_replan = False
                for v in cv:
                    if v.status in ('idle', 'ready') and v.mutable_suffix:
                        nxt = int(v.mutable_suffix[0])
                        v.mutable_suffix = v.mutable_suffix[1:]
                        env._commit(inst, clock, v, ct, nxt)
                    v.needs_replan = False
                self.commits += 1
                base['publish_state_trace'].append(PUBLISH_COMMITTED)
                base.update({'outcome': OUTCOME_COMMITTED, 'publish': PUBLISH_MODE,
                             'committed_verified': False,
                             'why_no_verified_state': ('`COMMITTED_VERIFIED` is reserved for a publish '
                                                       'whose post-state was independently re-read; '
                                                       'that is the B3-AT2A coordinator\'s job'),
                             'counts_as_refusal_capability': False})
            else:
                if publish_fn is None:
                    raise ValueError("publish mode %r requires publish_fn" % (publish,))
                base['publish_state_trace'] = [PUBLISH_STAGED_VALIDATED, PUBLISH_PUBLISHING]
                published = []
                try:
                    for vid in sorted(staged_all):
                        publish_fn(by_id[vid], staged_all[vid])
                        published.append({'vehicle_id': int(vid),
                                          'after': self._field_evidence([by_id[vid]])[0]})
                    self.commits += 1
                    base['publish_state_trace'].append(PUBLISH_COMMITTED)
                    base.update({'outcome': OUTCOME_COMMITTED, 'publish': 'per_vehicle_injected',
                                 'committed_verified': False,
                                 'published_vehicles': published,
                                 'counts_as_refusal_capability': False})
                except BaseException as exc:                                   # noqa: BLE001
                    self.partial_commits += 1
                    base['publish_state_trace'].append(PUBLISH_PARTIAL)
                    base.update({
                        'outcome': OUTCOME_PARTIAL, 'publish': 'per_vehicle_injected',
                        'published_vehicles': published,
                        'n_published_before_failure': len(published),
                        'exception': '%s: %s' % (type(exc).__name__, str(exc)[:120]),
                        'ticket_state': 'consumed_on_partial_commit',
                        'counts_as_refusal_capability': False,
                        'partial_commit_report': True,
                        'rollback_claimed': False,
                        'why': ('a partial publish is REPORTED as PARTIAL_COMMIT with the real '
                                'per-vehicle fields; it is NOT called a rollback, because no '
                                'durable transaction mechanism exists and no state-level restore '
                                'is proven (H5)')})

            base['boundary']['closed'] = b.closed
            return self._finalize(base, env, inst, clock, event, reveal_idx, cv, ct, csm,
                                   all_customers)


def default_gate_decision_fn(*, cert_result, cred_cert, cred_install, deadline_ok):
    """The REAL gate: `installable` is the only thing that can authorise the write.

    While the `quality_salability` capability gap stands, `installable` is False for every plan, so
    this function refuses everything -- which is the honest state of the system, and is why the AT1
    positive control has to be labelled as a mechanisms-only control.
    """
    from d3_install_gate import pre_install_reverify
    complete = bool(cert_result.get('installable'))
    decision = pre_install_reverify(cred_cert, cred_install, deadline_ok=deadline_ok,
                                    certification_complete=complete)
    decision['certification_complete_used'] = complete
    return decision


def isolation_report(repo_scripts):
    """Mechanical isolation statement for this batch.

    Returns the reachable-call graph (which non-test modules import this module), the D3-scope
    finding that no `d3_*` module calls a live write primitive, and the explicit statement that the
    seven historical bypass sites are untouched.  All derived with `ast`, no source modified.
    """
    import ast
    from pathlib import Path
    scripts = Path(repo_scripts)
    me = 'd3_commit_transaction'
    importers, d3_write_primitive_calls, d3_live_state_assigns = [], [], []
    for p in sorted(scripts.rglob('*.py')):
        rel = p.relative_to(scripts.parent).as_posix()
        if '__pycache__' in rel:
            continue
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        except SyntaxError:
            continue
        is_test = '/tests/' in rel
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name == me for a in node.names):
                if not is_test:
                    importers.append(rel)
            if isinstance(node, ast.ImportFrom) and (node.module or '') == me and not is_test:
                importers.append(rel)
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else None)
                if name in ('_commit', '_plan_and_commit') and not is_test:
                    d3_write_primitive_calls.append({'file': rel, 'line': node.lineno,
                                                     'call': name})
            if isinstance(node, ast.Assign) and not is_test:
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr in WRITE_ATTRS:
                        d3_live_state_assigns.append({'file': rel, 'line': t.lineno,
                                                      'attr': t.attr})
    d3_files = [p for p in sorted(scripts.rglob('d3_*.py'))
                if '__pycache__' not in p.as_posix()]
    d3_rel = {p.relative_to(scripts.parent).as_posix() for p in d3_files}
    return {
        'module': me,
        'non_test_modules_importing_the_transaction': sorted(set(importers)),
        'registered_on_env_anywhere': False,
        'd3_module_files': sorted(d3_rel),
        'd3_write_primitive_calls': [c for c in d3_write_primitive_calls if c['file'] in d3_rel],
        'd3_live_state_assigns': [c for c in d3_live_state_assigns if c['file'] in d3_rel],
        'historical_bypass_sites_modified_this_batch': [],
        'claim_allowed': ('within the D3 module set APART FROM THIS TRANSACTION MODULE ITSELF, no '
                          'live write primitive is referenced and no live state attribute is '
                          'assigned, so an ungated write is UNREACHABLE from those modules; this '
                          'transaction module is the one authorised writer and is the only entry '
                          'listed above.  This is NOT proposition P2 ("every real commit goes '
                          'through the gate") and does not claim the transaction took over the '
                          'production path'),
    }
