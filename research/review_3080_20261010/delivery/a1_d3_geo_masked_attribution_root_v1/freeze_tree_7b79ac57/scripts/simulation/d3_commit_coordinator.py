#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2A: the D3 COMMIT COORDINATOR -- unique writer, mutual exclusion, post-commit
read-only verification.  Isolated: it is NOT registered as any production run's commit entry.

User ruling (2026-10-14, B3-AT2A):

    "一个 D3 专用提交协调器持有真实车辆状态的写入权限，其他模块只能提交候选计划与授权凭据."
    "如果这三种写入方式仍然可以同时修改同一 D3 状态，便无法宣称真正的唯一写入者."
    and: 锁只有在所有相关写入者都遵守时才有效, so the boundary must impose a write CONSTRAINT on
    the D3 execution instance rather than only locking inside the new module.

What this module builds:

  * `WriteCapability` -- the token that carries the right to write.  Held by the coordinator.
  * `D3WriteBoundary` -- installed on ONE D3 execution instance.  It wraps that instance's
    `_commit` / `_plan_and_commit` and REFUSES any invocation that does not present the live
    capability from inside the coordinator's critical section.  This is forcible prevention at the
    D3-execution-instance level (stronger than B3-AT1's detection), and it touches no source file:
    the originals are restored by `restore()`.
  * `D3CommitCoordinator` -- the only writer.  Other modules may only call `submit(...)` with a
    candidate plan set and an authorisation credential.
  * mutual exclusion + epoch control: a re-entrancy-safe lock plus a state `epoch` captured at
    authorisation time and re-checked inside the critical section, so a publish that starts after
    another writer moved the state is refused, and two coordinators publishing the same ticket are
    serialised (one consumes, the other is refused).
  * `COMMITTED_VERIFIED` only after an INDEPENDENT post-commit re-read that is compared against the
    authorised commit result; otherwise the publish state stops at `COMMITTED`.
  * publish-state vocabulary: STAGED_VALIDATED / PUBLISHING / COMMITTED / COMMITTED_VERIFIED /
    PARTIAL_COMMIT / OUTCOME_UNKNOWN.  `atomic` is an implementation-mode name only.

Scope limits stated up front:
  * the guard covers the two commit primitives.  A direct assignment to `VehicleState.mutable_suffix`
    by a rogue in-process writer is NOT blocked by the guard -- it is DETECTED by the epoch re-check
    (and the AT2A rows record that distinction explicitly).
  * in-process only: no durable transaction, no cross-process exclusion.
"""
from __future__ import annotations

import hashlib
import threading

from d3_commit_transaction import (PUBLISH_COMMITTED, PUBLISH_COMMITTED_VERIFIED, PUBLISH_PARTIAL,
                                   PUBLISH_PUBLISHING, PUBLISH_STAGED_VALIDATED, PUBLISH_UNKNOWN,
                                   PUBLISH_MODE, PUBLISH_MODE_NOTE, PUBLISH_STATES,
                                   commit_identity, legacy_plan_hash, state_version,
                                   OUTCOME_BLOCKED_D8, OUTCOME_PARTIAL, OUTCOME_REFUSED,
                                   OUTCOME_UNKNOWN, OUTCOME_WAIT)

COORDINATOR_SCHEMA = 'd3_commit_coordinator_v1'
GUARD_SCHEMA = 'd3_write_boundary_v1'

CAUSE_ENERGY = 'energy_budget'
CAUSE_QUALITY = 'quality_salability'
CAUSE_AUTH = 'transaction_authorization'
CAUSES = (CAUSE_ENERGY, CAUSE_QUALITY, CAUSE_AUTH)


class UnauthorizedWrite(RuntimeError):
    """Raised by the instance guard when a write is attempted without the coordinator's capability."""


# Thread-local critical-section marker.  Measured defect of the first version: the guard asked ONE
# coordinator object whether the current thread was inside its critical section, so a second
# coordinator sharing the same guarded instance had every write refused (the AT2-ONCE row came back
# as an exception instead of a clean ticket refusal).  The marker is now process-level and carries the
# capability that opened the section, so several coordinators can legitimately share one guarded
# instance while every write stays tied to the capability of the coordinator that is publishing.
_CRITICAL = threading.local()
_BOUNDARIES = {}


def _cs_depth():
    return getattr(_CRITICAL, 'depth', 0)


def _cs_capability():
    return getattr(_CRITICAL, 'capability', None)


def boundary_for(env):
    """One guard per environment instance, shared by every coordinator working on that instance."""
    key = id(env)
    b = _BOUNDARIES.get(key)
    if b is None:
        b = D3WriteBoundary(env)
        _BOUNDARIES[key] = b
    return b


def drop_boundary(env):
    b = _BOUNDARIES.pop(id(env), None)
    if b is not None:
        b.restore()
    return b


class WriteCapability(object):
    """The right to write.  Identity-compared, never serialised, never handed to another module."""

    def __init__(self, owner):
        self.owner = owner
        self._nonce = hashlib.sha256(('%s|%d' % (owner, id(self))).encode()).hexdigest()[:12]

    def __repr__(self):                                                    # pragma: no cover
        return '<WriteCapability owner=%s nonce=%s>' % (self.owner, self._nonce)


class D3WriteBoundary(object):
    """Instance-level write constraint for ONE D3 execution instance.

    `install()` wraps `_commit` / `_plan_and_commit`.  A call is allowed only when the current thread
    is inside a coordinator critical section AND the capability that opened that section is still
    live on this boundary.  Everything else raises `UnauthorizedWrite` and performs NO write.
    """

    def __init__(self, env):
        self.env = env
        self.installed = False
        self.live_capabilities = {}
        self.refusals = []
        self.allowed = 0
        self._orig = {}

    def grant(self, capability):
        self.live_capabilities[id(capability)] = capability

    def revoke(self, capability):
        self.live_capabilities.pop(id(capability), None)

    def is_live(self, capability):
        return capability is not None and id(capability) in self.live_capabilities

    def install(self):
        if self.installed:
            return self
        self._orig['_commit'] = self.env._commit
        self._orig['_plan_and_commit'] = self.env._plan_and_commit
        bound = self

        def _commit(inst_idx, clock, v, traces, node):
            bound._authorise('_commit', v.vehicle_id, node)
            return bound._orig['_commit'](inst_idx, clock, v, traces, node)

        def _plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers):
            bound._authorise('_plan_and_commit', None, None)
            return bound._orig['_plan_and_commit'](inst_idx, clock, vehicles, traces, served_mask,
                                                   all_customers)

        self.env._commit = _commit
        self.env._plan_and_commit = _plan_and_commit
        self.installed = True
        return self

    def _authorise(self, primitive, vehicle_id, node):
        cap = _cs_capability()
        holder = threading.get_ident() if _cs_depth() > 0 else None
        if not self.is_live(cap) or holder != threading.get_ident():
            self.refusals.append({'primitive': primitive, 'vehicle_id': (None if vehicle_id is None
                                                                        else int(vehicle_id)),
                                  'node': (None if node is None else int(node)),
                                  'capability_present': cap is not None,
                                  'capability_live': self.is_live(cap),
                                  'critical_section_owner': holder,
                                  'caller_thread': threading.get_ident()})
            raise UnauthorizedWrite(
                'write via %s refused: the D3 write boundary requires the coordinator capability '
                'inside its critical section' % primitive)
        self.allowed += 1

    def restore(self):
        for name, fn in self._orig.items():
            setattr(self.env, name, fn)
        self.installed = False

    def snapshot(self):
        return {'guard_schema': GUARD_SCHEMA, 'installed': self.installed,
                'n_allowed_writes': self.allowed, 'n_refused_writes': len(self.refusals),
                'refused_writes': self.refusals,
                'installed_on_a_production_instance': False,
                'coverage_note': ('the guard covers `_commit` / `_plan_and_commit`; a direct '
                                  '`mutable_suffix` assignment by a rogue in-process writer is not '
                                  'blocked here and is caught by the epoch re-check instead')}


class D3CommitCoordinator(object):
    """The ONLY writer for a D3 execution instance.  Others submit candidates + credentials."""

    registered_on_env = False
    holds_write_capability = True

    def __init__(self, env, *, registry, name='d3-coordinator'):
        self.env = env
        self.registry = registry
        self.name = name
        self._capability = None
        self._lock = threading.RLock()
        self._owner = None                 # thread id currently inside the critical section
        self.boundary = boundary_for(env)
        self.n_submitted = 0
        self.n_committed = 0
        self.n_refused = 0
        self.n_partial = 0
        self.n_wait = 0

    # ---- capability / critical section ------------------------------------------------------
    def open(self):
        self._capability = WriteCapability(self.name)
        self.boundary.grant(self._capability)
        return self.boundary.install()

    def close(self):
        if self._capability is not None:
            self.boundary.revoke(self._capability)
        self.boundary.restore()
        drop_boundary(self.env)
        self._capability = None

    def current_capability(self):
        return self._capability

    def critical_section_owner(self):
        return threading.get_ident() if _cs_depth() > 0 else None

    class _Critical(object):
        def __init__(self, coord):
            self.coord = coord
            self._prev_cap = None

        def __enter__(self):
            self.coord._lock.acquire()
            self._prev_cap = _cs_capability()
            _CRITICAL.capability = self.coord.current_capability()
            _CRITICAL.depth = _cs_depth() + 1
            self.coord._owner = threading.get_ident()
            return self

        def __exit__(self, exc_type, exc, tb):
            _CRITICAL.depth = max(0, _cs_depth() - 1)
            _CRITICAL.capability = self._prev_cap
            if _cs_depth() == 0:
                self.coord._owner = None
            self.coord._lock.release()
            return False

    def critical_section(self):
        return self._Critical(self)

    # ---- cause classification (three separate causes, never merged) -------------------------
    @staticmethod
    def classify_causes(cert_result, auth_ok):
        """Report `energy_budget` / `quality_salability` / `transaction_authorization` separately."""
        reasons = list((cert_result or {}).get('reasons') or ())
        gaps = list((cert_result or {}).get('blocking_capability_gaps') or ())
        return {
            CAUSE_ENERGY: {'failed': any('energy' in r for r in reasons),
                           'evidence': [r for r in reasons if 'energy' in r]},
            CAUSE_QUALITY: {'failed': bool(gaps) or any('quality' in r for r in reasons),
                            'evidence': gaps + [r for r in reasons if 'quality' in r]},
            CAUSE_AUTH: {'failed': not auth_ok, 'evidence': [] if auth_ok else ['not_authorized']},
            'note': ('the three causes are reported separately; a formal-budget failure must NOT be '
                     'attributed to quality, and the formal budget is never relaxed to make a '
                     'transaction test succeed'),
        }

    # ---- submissions ------------------------------------------------------------------------
    def submit(self, *, inst, clock, event, reveal_idx, cv, ct, csm, plans, all_customers,
               cert_result, cred_cert, cred_install, gate_decision_fn, deadline_ok=True,
               expected_commit_id=None, expected_ticket=None, before_publish_start_hook=None,
               registry=None,
               publish_hook=None, after_publish_hook=None, publish_fn=None, budget_role='FORMAL',
               active_publishes=None):
        """Submit a candidate + credential.  Only the coordinator writes, and only inside the
        critical section, and only with the capability it holds."""
        self.n_submitted += 1
        staged = {int(vid): [int(x) for x in p.suffix if int(x) > 0]
                  for vid, p in plans.items() if len(p.suffix) > 0}
        out = {'coordinator_schema': COORDINATOR_SCHEMA, 'name': self.name,
               'not_a_production_path': True, 'registered_on_env': False,
               'budget_role': budget_role, 'publish_mode': PUBLISH_MODE,
               'publish_mode_note': PUBLISH_MODE_NOTE,
               'decision_point': {'inst': int(inst), 'event': int(event), 'clock': float(clock)},
               'commit_id': commit_identity(plans), 'plan_hash_legacy': legacy_plan_hash(plans),
               'publish_state': None, 'publish_state_trace': [],
               'refusals': [], 'causes': None, 'ticket': None, 'ticket_state': None,
               'post_read': None, 'committed_verified': False,
               'field_evidence_before': self._fields(cv),
               'field_evidence_after': None}

        # WAIT first: no pending action is not a refused write
        if not staged:
            self.n_wait += 1
            out.update({'outcome': OUTCOME_WAIT, 'counts_as_refusal_capability': False,
                        'why': 'no pending action, so there is no refused write to attribute'})
            return out

        # the gate decides, OUTSIDE the critical section
        decision = gate_decision_fn(cert_result=cert_result, cred_cert=cred_cert,
                                   cred_install=cred_install, deadline_ok=deadline_ok)
        auth_ok = bool(decision.get('allow'))
        out['causes'] = self.classify_causes(cert_result, auth_ok)
        out['gate'] = decision
        if not auth_ok:
            self.n_refused += 1
            only_quality = (out['causes'][CAUSE_QUALITY]['failed']
                            and not out['causes'][CAUSE_ENERGY]['failed'])
            out.update({'outcome': OUTCOME_BLOCKED_D8 if only_quality else OUTCOME_REFUSED,
                        'refusals': list(decision.get('refusals') or []),
                        'counts_as_refusal_capability': ('quality_dimension_only' if only_quality
                                                         else True),
                        'why': 'the gate refused before any write; causes are reported separately'})
            return out

        epoch_at_authorisation = state_version(self.env, inst, clock, event, reveal_idx, cv, ct,
                                               csm, all_customers)
        out['epoch_at_authorisation'] = epoch_at_authorisation

        with self.critical_section():
            # (a0) TEST SEAM for AT2-F8: models an out-of-band writer (a direct `mutable_suffix`
            # assignment, which the instance guard deliberately does NOT cover) landing between the
            # authorisation and the start of the publish.  It must be caught by the epoch re-check.
            if before_publish_start_hook is not None:
                before_publish_start_hook(cv, csm, out)
            # (a) epoch re-check: publish must not start on a moved state
            epoch_now = state_version(self.env, inst, clock, event, reveal_idx, cv, ct, csm,
                                      all_customers)
            out['epoch_at_publish_start'] = epoch_now
            if epoch_now != epoch_at_authorisation:
                self.n_refused += 1
                out.update({'outcome': OUTCOME_REFUSED, 'publish_state': None,
                            'refusals': ['state_changed_before_publish'],
                            'counts_as_refusal_capability': True,
                            'why': 'the state moved between authorisation and publish start'})
                return self._finish(out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers)

            # (b) identity: the transaction identity, not the anchor-blind legacy hash
            if expected_commit_id is not None and expected_commit_id != out['commit_id']:
                self.n_refused += 1
                out.update({'outcome': OUTCOME_REFUSED,
                            'refusals': ['commit_identity_mismatch'],
                            'counts_as_refusal_capability': True})
                return self._finish(out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers)

            # (c) one-time ticket, consumed inside the critical section
            reg = registry if registry is not None else self.registry
            ticket = expected_ticket or reg.issue(
                inst=inst, event=event, clock=clock, state_hash=epoch_at_authorisation,
                commit_id=out['commit_id'], certifier='coordinator',
                plan_hash=out['plan_hash_legacy'])
            out['ticket'] = dict(ticket)
            ok, info = reg.consume(ticket['ticket_id'], consumer=self.name)
            out['consume_info'] = info
            if not ok:
                self.n_refused += 1
                out.update({'outcome': OUTCOME_REFUSED, 'refusals': ['ticket_already_consumed'],
                            'ticket_state': 'consumed',
                            'counts_as_refusal_capability': True})
                return self._finish(out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers)
            out['ticket_state'] = 'consumed_in_critical_section'

            # (d) staging: build and validate before any write
            problems = self._validate(staged, csm, self.env.num_nodes)
            out['candidate_validation'] = {'problems': problems, 'staged': staged}
            if problems:
                self.n_refused += 1
                out.update({'outcome': OUTCOME_REFUSED, 'refusals': ['candidate_invalid'],
                            'ticket_state': 'consumed_then_candidate_invalid',
                            'counts_as_refusal_capability': True})
                return self._finish(out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers)
            out['publish_state'] = PUBLISH_STAGED_VALIDATED
            out['publish_state_trace'] = [PUBLISH_STAGED_VALIDATED]

            # (e) the publish itself, with the write capability held and the critical section owned
            out['publish_state_trace'].append(PUBLISH_PUBLISHING)
            out['publish_state'] = PUBLISH_PUBLISHING
            if active_publishes is not None:
                active_publishes['active'] = active_publishes.get('active', 0) + 1
                active_publishes['max_concurrent'] = max(active_publishes.get('max_concurrent', 0),
                                                         active_publishes['active'])
                active_publishes.setdefault('windows', []).append(
                    {'thread': threading.get_ident(), 'state': 'enter'})
            try:
                if publish_hook is not None:
                    publish_hook(cv, csm, out)
                by_id = {int(v.vehicle_id): v for v in cv}
                if publish_fn is None:
                    for vid, v in by_id.items():
                        v.mutable_suffix = list(staged.get(vid, []))
                        v.needs_replan = False
                    for v in cv:
                        if v.status in ('idle', 'ready') and v.mutable_suffix:
                            nxt = int(v.mutable_suffix[0])
                            v.mutable_suffix = v.mutable_suffix[1:]
                            self.env._commit(inst, clock, v, ct, nxt)
                        v.needs_replan = False
                    out['publish_state'] = PUBLISH_COMMITTED
                    out['publish_state_trace'].append(PUBLISH_COMMITTED)
                    out['outcome'] = PUBLISH_COMMITTED
                    self.n_committed += 1
                else:
                    published = []
                    try:
                        for vid in sorted(staged):
                            publish_fn(by_id[vid], staged[vid])
                            published.append({'vehicle_id': int(vid),
                                              'after': self._fields([by_id[vid]])[0]})
                        out['publish_state'] = PUBLISH_COMMITTED
                        out['publish_state_trace'].append(PUBLISH_COMMITTED)
                        out['outcome'] = PUBLISH_COMMITTED
                        out['published_vehicles'] = published
                        self.n_committed += 1
                    except BaseException as exc:                            # noqa: BLE001
                        self.n_partial += 1
                        out['publish_state'] = PUBLISH_PARTIAL
                        out['publish_state_trace'].append(PUBLISH_PARTIAL)
                        out['outcome'] = OUTCOME_PARTIAL
                        out['published_vehicles'] = published
                        out['n_published_before_failure'] = len(published)
                        out['exception'] = '%s: %s' % (type(exc).__name__, str(exc)[:120])
                        out['rollback_claimed'] = False
                        out['why'] = ('a partial publish is reported as PARTIAL_COMMIT with the '
                                      'true per-vehicle fields; it is NOT called a rollback')
                if after_publish_hook is not None:
                    after_publish_hook(cv, csm, out)
            finally:
                if active_publishes is not None:
                    active_publishes['active'] -= 1
                    active_publishes.setdefault('windows', []).append(
                        {'thread': threading.get_ident(), 'state': 'exit'})

        out = self._finish(out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers)

        # (f) INDEPENDENT post-commit read-only verification -> COMMITTED_VERIFIED
        if out.get('outcome') == PUBLISH_COMMITTED:
            post = self._post_read(cv, staged)
            out['post_read'] = post
            if post['matches_the_authorised_result']:
                out['committed_verified'] = True
                out['publish_state'] = PUBLISH_COMMITTED_VERIFIED
                out['publish_state_trace'].append(PUBLISH_COMMITTED_VERIFIED)
            else:
                out['publish_state'] = PUBLISH_UNKNOWN
                out['publish_state_trace'].append(PUBLISH_UNKNOWN)
                out['outcome'] = OUTCOME_UNKNOWN
                out['why'] = ('the post-commit re-read does not match the authorised result, so the '
                              'publish state is OUTCOME_UNKNOWN rather than verified')
        out['counts_as_refusal_capability'] = False
        return out

    # ---- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _fields(vehicles):
        return [{'vehicle_id': int(v.vehicle_id), 'status': v.status,
                 'current_node': int(v.current_node),
                 'committed_next': (None if v.committed_next is None else int(v.committed_next)),
                 'mutable_suffix': [int(x) for x in v.mutable_suffix]}
                for v in sorted(vehicles, key=lambda x: int(x.vehicle_id))]

    @staticmethod
    def _validate(staged, served_mask, n_nodes):
        problems, seen = [], {}
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

    def _post_read(self, cv, staged):
        """Independent re-read of the actual vehicle fields and comparison with the authorisation."""
        rows, mismatches = [], []
        by_id = {int(v.vehicle_id): v for v in cv}
        for vid in sorted(staged):
            v = by_id.get(vid)
            observed_head = (None if v.committed_next is None else int(v.committed_next))
            expected_head = int(staged[vid][0]) if staged[vid] else None
            rows.append({'vehicle_id': int(vid), 'expected_committed_head': expected_head,
                         'observed_committed_head': observed_head,
                         'observed_status': v.status,
                         'observed_suffix_after_pop': [int(x) for x in v.mutable_suffix]})
            if expected_head != observed_head:
                mismatches.append({'vehicle_id': int(vid), 'expected': expected_head,
                                   'observed': observed_head})
        return {'method': 'read-only re-read of live vehicle fields after the publish',
                'rows': rows, 'mismatches': mismatches,
                'matches_the_authorised_result': not mismatches,
                'note': ('only a publish whose post-state is independently re-read may be labelled '
                         'COMMITTED_VERIFIED')}

    def _finish(self, out, inst, clock, event, reveal_idx, cv, ct, csm, all_customers):
        out['field_evidence_after'] = self._fields(cv)
        out['state_hash_after'] = state_version(self.env, inst, clock, event, reveal_idx, cv, ct,
                                               csm, all_customers)
        before = out.get('epoch_at_authorisation')
        if before is not None:
            out['state_unchanged'] = (out['state_hash_after'] == before)
        out['boundary'] = self.boundary.snapshot()
        return out
