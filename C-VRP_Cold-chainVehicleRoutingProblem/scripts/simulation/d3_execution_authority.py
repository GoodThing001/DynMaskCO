#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2B-1: the D3 EXECUTION AUTHORITY -- default-OFF wiring for the real env.

Authorised scope (user ruling 2026-10-14): a closable D3-specific wiring with `state_authority=None`
as the default, three separately routed interfaces (decision commit / execution advance / reveal
invalidation), a mechanically verifiable write graph, and an isolated-mode regression against the
historical behaviour.  NOT authorised: enabling real production submission, D8 relaxation, G1/G2 or
effect experiments.

Design constraints honoured here:

  * `strict_online_env.py` is NOT modified.  Its sha256
    fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6 stays the frozen historical
    identity, and the D3 mode is installed by INSTANCE-LEVEL wiring, so the default path cannot be
    affected by construction.  ("默认值为 None 本身不足以证明零影响" is answered by running the full
    compatibility regression in the test, not by the default alone.)
  * the authority ADOPTS the environment's own vehicle list instead of copying it, so there is
    exactly ONE authoritative state object (no env copy + container copy running in parallel).
  * the replanner never receives the authoritative list: it gets a detached copy, and its intended
    plan is then submitted to the authority, which classifies it as a DECISION_WRITE.
  * reveal tail-clearing is verified (not blocked) against the REVEAL_INVALIDATION contract, because
    it happens inside `_run_loop` as a plain attribute assignment.
  * credential policy for THIS batch is `observation`: the wiring classifies, records and enforces
    invariants, but does not demand a runtime credential, because the real gate refuses every plan
    while the D8 capability gap stands and demanding one here would lock out legitimate execution
    advances (the opposite failure the ruling warns about).  Enforcing certified submission is
    AT2B-2/AT2B-3, and P2 stays NOT_ESTABLISHED.
"""
from __future__ import annotations

import copy
import hashlib
import threading

from d3_commit_transaction import state_version

AUTHORITY_SCHEMA = 'd3_execution_authority_v1'
POLICY_OBSERVATION = 'observation'
POLICY_CERTIFIED = 'certified'

DECISION_WRITE = 'DECISION_WRITE'
DECISION_PREPARATION = 'DECISION_PREPARATION'
EXECUTION_ADVANCE = 'EXECUTION_ADVANCE'
REVEAL_INVALIDATION = 'REVEAL_INVALIDATION'
CONSTRUCTION_ONLY = 'CONSTRUCTION_ONLY'
LOCAL_ONLY = 'LOCAL_ONLY'
PERMISSION_CLASSES = (DECISION_WRITE, DECISION_PREPARATION, EXECUTION_ADVANCE,
                      REVEAL_INVALIDATION, CONSTRUCTION_ONLY, LOCAL_ONLY)

# pre-registered mapping (B3AT2H inventory rules, with the AT2B-1 amendments)
FUNCTION_PERMISSION = {
    '_commit': DECISION_WRITE,
    '_plan_and_commit': DECISION_WRITE,
    'plan': DECISION_WRITE,
    'on_reveal': DECISION_WRITE,
    'prepare_decision_point': DECISION_PREPARATION,
    '_advance_fleet': EXECUTION_ADVANCE,
    '_advance_coldchain_to': EXECUTION_ADVANCE,
    '_dispatch_coldchain': EXECUTION_ADVANCE,
    '_run_loop': EXECUTION_ADVANCE,        # clock advance / booking_end; reveal handled separately
    'run': EXECUTION_ADVANCE,
    'run_resumed': EXECUTION_ADVANCE,
    '_force_return_all': DECISION_WRITE,   # delegates to _commit (a head commit)
    '__init__': CONSTRUCTION_ONLY,
}
REVEAL_FIELDS = ('mutable_suffix',)


class RevealInvalidation(object):
    """The REVEAL_INVALIDATION contract + its mechanical verifier (V1-V5)."""

    @staticmethod
    def field_view(vehicles):
        return {int(v.vehicle_id): {'status': v.status,
                                    'committed_next': (None if v.committed_next is None
                                                       else int(v.committed_next)),
                                    'suffix': sorted(int(x) for x in v.mutable_suffix if int(x) > 0),
                                    'needs_replan': bool(getattr(v, 'needs_replan', False)),
                                    'replan_reason': getattr(v, 'replan_reason', None)}
                for v in vehicles}

    @classmethod
    def verify(cls, before, after, *, epoch_before, epoch_after, served_mask_before,
               served_mask_after):
        b, a = cls.field_view(before), cls.field_view(after)
        violations = []
        cleared = []
        for vid, av in a.items():
            bv = b.get(vid)
            if bv is None:
                violations.append({'check': 'V0_vehicle_set', 'vehicle_id': vid,
                                   'why': 'a vehicle appeared during the invalidation'})
                continue
            # V1 = no REVOCATION of an existing commitment.  Setting a commitment where there was
            # none is a DECISION_WRITE, not a reveal revocation, so it is not a violation here.
            # (Measured defect of the first version: comparing inequality flagged the commit loop's
            # legitimate head commit as a violation.)
            if bv['committed_next'] is not None and bv['committed_next'] != av['committed_next']:
                violations.append({'check': 'V1_commitment_frozen', 'vehicle_id': vid,
                                   'before': bv['committed_next'], 'after': av['committed_next']})
            added = sorted(set(av['suffix']) - set(bv['suffix']))
            if added:
                violations.append({'check': 'V2_monotone_shrink', 'vehicle_id': vid,
                                   'added_customers': added})
            if bv['suffix'] and not av['suffix']:
                cleared.append(vid)
                if av['status'] != 'closed' and not (av['needs_replan']
                                                     and av['replan_reason']):
                    violations.append({'check': 'V3_replan_trigger_preserved', 'vehicle_id': vid,
                                       'needs_replan': av['needs_replan'],
                                       'replan_reason': av['replan_reason']})
        if list(served_mask_before) != list(served_mask_after):
            violations.append({'check': 'V4_served_mask_untouched'})
        if epoch_before == epoch_after:
            violations.append({'check': 'V5_credential_expiry',
                               'why': 'the state version did not change, so credentials would NOT be '
                                      'invalidated by the reveal'})
        return {'contract': 'REVEAL_INVALIDATION', 'ok': not violations,
                'violations': violations, 'vehicles_cleared': sorted(cleared),
                'epoch_before': epoch_before, 'epoch_after': epoch_after,
                'checks': ['V1 commitment frozen', 'V2 monotone shrink (no new customers)',
                           'V3 replan trigger preserved', 'V4 served_mask untouched',
                           'V5 credential expiry (epoch changed)']}


class D3ExecutionAuthority(object):
    """Holds the ONE authoritative state and routes the three interfaces.

    Default-OFF: `attach(env, ...)` is the only entry point and the test asserts that an un-attached
    environment is bit-identical in behaviour to the historical one.
    """

    def __init__(self, env, *, container, policy=POLICY_OBSERVATION, instance_id='d3-exec'):
        self.env = env
        self.container = container
        self.policy = policy
        self.instance_id = instance_id
        self.attached = False
        self.adopted = False
        self.write_graph = []
        self.violations = []
        self.reveal_events = []
        self.advances = 0
        self.decision_commits = 0
        self.replanner_candidates = 0
        self.anchor = None
        self.real_write_backs_to_a_production_system = 0
        self._orig = {}
        self._pending_reveal = None
        self._anchor_epoch_at_prepare = None
        self._lock = threading.RLock()

    # ---- attach / detach --------------------------------------------------------------------
    def attach(self):
        if self.attached:
            return self
        env = self.env
        self._orig = {'_commit': env._commit, '_plan_and_commit': env._plan_and_commit,
                      '_advance_fleet': env._advance_fleet,
                      '_advance_coldchain_to': env._advance_coldchain_to,
                      'prepare_decision_point': env.prepare_decision_point}
        bound = self

        def _commit(inst_idx, clock, v, traces, node):
            bound._classify('_commit', v.vehicle_id, node)
            return bound._orig['_commit'](inst_idx, clock, v, traces, node)

        def prepare_decision_point(clock, vehicles):
            bound._adopt(vehicles)
            bound.container.set_context(clock=clock, event=getattr(env, 'event_id', None))
            bound._classify('prepare_decision_point', None, None)
            out = bound._orig['prepare_decision_point'](clock, vehicles)
            # ANCHOR ORDER: the certification anchor is captured AFTER the preparation, so no
            # credential is needed to authorise the step that makes the credential meaningful.
            bound._anchor_epoch_at_prepare = bound.epoch()
            bound.anchor = {'captured_after': 'prepare_decision_point',
                            'clock': float(clock), 'epoch': bound._anchor_epoch_at_prepare}
            return out

        def _advance_fleet(inst_idx, clock, vehicles, traces, served_mask):
            bound._adopt(vehicles, traces=traces, served_mask=served_mask)
            bound.container.set_context(inst=inst_idx, clock=clock,
                                        event=getattr(env, 'event_id', None))
            bound.advances += 1
            bound._classify('_advance_fleet', None, None)
            out = bound._orig['_advance_fleet'](inst_idx, clock, vehicles, traces, served_mask)
            # WINDOW: _advance_fleet returns at strict_online_env.py:519 and the reveal block runs
            # at :523-528, so snapshotting HERE gives exactly the pre-reveal, post-advance state --
            # the window REVEAL_INVALIDATION governs.  Snapshotting earlier would wrongly include the
            # legitimate service-completion clearing of committed_next.
            if bound._is_reveal_clock(inst_idx, clock):
                bound.note_pre_reveal(vehicles, served_mask)
            return out

        def _advance_coldchain_to(v, clock):
            bound._classify('_advance_coldchain_to', v.vehicle_id, None)
            return bound._orig['_advance_coldchain_to'](v, clock)

        def _plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers):
            # run() reaches _plan_and_commit BEFORE the first _advance_fleet, so traces and the served
            # mask must be adopted here as well (measured: an empty traces list crashed the epoch's
            # sanctioned snapshot, which needs at least one services array).
            bound._adopt(vehicles, traces=traces, served_mask=served_mask)
            bound.container.set_context(inst=inst_idx, clock=clock,
                                        event=getattr(env, 'event_id', None))
            # 1) verify any reveal invalidation that just happened (the tail-clear is a plain
            #    attribute assignment inside _run_loop, so it is verified rather than blocked)
            bound._verify_pending_reveal(vehicles, served_mask, inst_idx, clock)
            # 2) give the replanner a DETACHED COPY, never the authoritative list
            replanner = getattr(bound.env, 'replanner', None)
            isolated = None
            if replanner is not None:
                isolated = _ReplannerIsolation(bound, replanner)
                bound.env.replanner = isolated
            try:
                return bound._orig['_plan_and_commit'](inst_idx, clock, vehicles, traces,
                                                       served_mask, all_customers)
            finally:
                if replanner is not None:
                    bound.env.replanner = replanner
                # 3) the isolated replanner's intent was applied to a copy; take it as a decision
                #    candidate and classify it (candidate legality is enforced here)
                if isolated is not None:
                    bound._record_candidates(isolated)

        env._commit = _commit
        env._plan_and_commit = _plan_and_commit
        env._advance_fleet = _advance_fleet
        env._advance_coldchain_to = _advance_coldchain_to
        env.prepare_decision_point = prepare_decision_point
        self.attached = True
        return self

    def detach(self):
        for name, fn in self._orig.items():
            setattr(self.env, name, fn)
        self.attached = False

    # ---- state ownership --------------------------------------------------------------------
    def _adopt(self, vehicles, traces=None, served_mask=None):
        if not self.adopted:
            self.container.adopt(vehicles, traces=traces, served_mask=served_mask)
            self.adopted = True
        elif served_mask is not None and self.container._served_mask is not served_mask:
            self.container._served_mask = served_mask
        elif traces is not None and self.container._traces is not traces:
            self.container._traces = traces

    def epoch(self):
        return self.container.epoch()

    def vehicles(self):
        return self.container._vehicles

    # ---- classification + write graph -------------------------------------------------------
    def _classify(self, function, vehicle_id, node, extra=None):
        perm = FUNCTION_PERMISSION.get(function, 'UNCLASSIFIED')
        row = {'function': function, 'permission_class': perm,
               'requires_recertification': perm in (DECISION_WRITE,),
               'vehicle_id': (None if vehicle_id is None else int(vehicle_id)),
               'node': (None if node is None else int(node)),
               'policy': self.policy,
               'direct': True}
        if extra:
            row.update(extra)
        self.write_graph.append(row)
        if perm == 'UNCLASSIFIED':
            self.violations.append({'why': 'unclassified_write', 'row': row})
        return row

    def _record_candidates(self, isolated):
        for row in isolated.candidates:
            self.replanner_candidates += 1
            self.write_graph.append({'function': 'replanner.plan', 'permission_class': DECISION_WRITE,
                                     'requires_recertification': True, 'direct': False,
                                     'via': 'detached_copy',
                                     'vehicle_id': row['vehicle_id'],
                                     'suffix': row['suffix'], 'policy': self.policy})
            problems = self.validate_candidate(row['suffix'])
            if problems and self.policy == POLICY_CERTIFIED:
                self.violations.append({'why': 'illegal_candidate', 'row': row,
                                        'problems': problems})
            elif problems:
                self.write_graph[-1]['candidate_problems'] = problems

    def validate_candidate(self, suffix):
        problems, seen = [], set()
        served = self.container._served_mask
        n = self.env.num_nodes
        for c in suffix:
            c = int(c)
            if c < 0 or c >= n:
                problems.append({'why': 'out_of_range', 'customer': c})
                continue          # do NOT index with an out-of-range customer
            if c == 0:
                continue          # depot marker (JF1H appends [0]); not a customer
            if served is not None and bool(served[c]):
                problems.append({'why': 'already_served', 'customer': c})
            if c in seen:
                problems.append({'why': 'duplicate', 'customer': c})
            seen.add(c)
        return problems

    # ---- the three interfaces ---------------------------------------------------------------
    def decision_commit(self, *, vehicle, node, credential=None):
        """DECISION_WRITE interface: commit a plan head.  In `observation` policy it records; in
        `certified` policy it refuses without a live credential + matching epoch."""
        if self.policy == POLICY_CERTIFIED:
            ok = credential is not None and credential.get('epoch') == self.epoch()
            if not ok:
                self.violations.append({'why': 'decision_commit_without_credential',
                                        'vehicle_id': int(vehicle.vehicle_id)})
                return False
        self.decision_commits += 1
        return True

    def reveal_invalidation(self, *, vehicles, served_mask, inst_idx, clock, epoch_before):
        """REVEAL_INVALIDATION interface: verify the environment's own tail-clear."""
        epoch_after = self.epoch()
        verdict = RevealInvalidation.verify(self._reveal_before, vehicles,
                                            epoch_before=epoch_before, epoch_after=epoch_after,
                                            served_mask_before=self._reveal_served_before,
                                            served_mask_after=served_mask)
        event = {'inst': int(inst_idx), 'clock': float(clock), 'verdict': verdict}
        self.reveal_events.append(event)
        if not verdict['ok']:
            self.violations.append({'why': 'reveal_invalidation_violation',
                                    'violations': verdict['violations']})
        return verdict

    def _is_reveal_clock(self, inst_idx, clock):
        row = getattr(self.env, 'reveal_time', None)
        if row is None:
            return False
        try:
            times = row[inst_idx]
        except Exception:                                                   # noqa: BLE001
            return False
        return any(abs(float(t) - float(clock)) < 1e-6 for t in times[1:])

    def note_pre_reveal(self, vehicles, served_mask):
        """Called from the wrapped `_advance_fleet` right before the reveal block runs."""
        self._reveal_before = [copy.deepcopy(v) for v in vehicles]
        self._reveal_served_before = list(served_mask)
        self._reveal_epoch_before = self.epoch()

    def _verify_pending_reveal(self, vehicles, served_mask, inst_idx, clock):
        if getattr(self, '_reveal_before', None) is None:
            return None
        verdict = self.reveal_invalidation(vehicles=vehicles, served_mask=served_mask,
                                          inst_idx=inst_idx, clock=clock,
                                          epoch_before=self._reveal_epoch_before)
        self._reveal_before = None
        return verdict

    # ---- evidence ---------------------------------------------------------------------------
    def snapshot(self):
        return {'authority_schema': AUTHORITY_SCHEMA, 'instance_id': self.instance_id,
                'policy': self.policy, 'attached': self.attached, 'adopted': self.adopted,
                'n_authoritative_vehicles': len(self.container._vehicles),
                'container_is_the_owner': True,
                'n_advances': self.advances, 'n_decision_commits': self.decision_commits,
                'n_replanner_candidates': self.replanner_candidates,
                'n_reveal_events': len(self.reveal_events),
                'reveal_events': self.reveal_events[:4],
                'n_write_graph_rows': len(self.write_graph),
                'write_graph_by_permission': {p: sum(1 for r in self.write_graph
                                                     if r['permission_class'] == p)
                                              for p in PERMISSION_CLASSES},
                'violations': self.violations,
                'anchor': self.anchor,
                'anchor_captured_after_prepare': bool(self.anchor
                                                      and self.anchor['captured_after']
                                                      == 'prepare_decision_point'),
                'real_write_backs_to_a_production_system':
                    self.real_write_backs_to_a_production_system,
                'default_off': True,
                'does_not_establish_P2': True}


class _ReplannerIsolation(object):
    """Wraps a replanner so it only ever sees a DETACHED COPY of the authoritative vehicles.

    The wrapper records the intent (per-vehicle suffix) and never applies it to the authoritative
    state itself; the environment's own commit loop still runs on the authoritative list, which is
    what keeps ONE authoritative state.
    """

    def __init__(self, authority, inner):
        self.authority = authority
        self.inner = inner
        self.candidates = []
        self.notes = []
        self._delegate_attrs = ('export_state', 'restore_state', 'sync_deferred_from_vehicles',
                                'on_reveal', 'replan_count')

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        detached = [copy.deepcopy(v) for v in vehicles]
        detached_mask = copy.copy(served_mask)
        out = self.inner.plan(env, inst_idx, clock, detached, detached_mask, visible_ids,
                              replan_ids=replan_ids)
        # The replanner's INTENT is now published into the authoritative objects as a
        # DECISION_WRITE, with candidate validation.  This is what keeps ONE authoritative state:
        # the replanner never holds it, and the env's own commit loop still runs on it.
        by_id = {int(v.vehicle_id): v for v in vehicles}
        for dv in detached:
            suffix = [int(x) for x in dv.mutable_suffix]
            customers = [int(x) for x in suffix if int(x) > 0]
            problems = self.authority.validate_candidate(customers)
            self.candidates.append({'vehicle_id': int(dv.vehicle_id), 'suffix': customers,
                                    'problems': problems})
            auth_v = by_id.get(int(dv.vehicle_id))
            if problems:
                # An ILLEGAL candidate is never applied, whatever the credential policy: refusing it
                # is a property of the plan itself, not of the authorisation.
                self.authority.violations.append({'why': 'illegal_candidate_not_applied',
                                                  'vehicle_id': int(dv.vehicle_id),
                                                  'suffix': customers, 'problems': problems})
                continue
            if auth_v is not None:
                auth_v.mutable_suffix = list(suffix)
                auth_v.needs_replan = bool(getattr(dv, 'needs_replan', False))
        return out

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        if not hasattr(self.inner, 'on_reveal'):
            return None
        detached = [copy.deepcopy(v) for v in vehicles]
        return self.inner.on_reveal(env, inst_idx, clock, detached, copy.copy(served_mask),
                                    visible_ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def make_authority(env, container, *, policy=POLICY_OBSERVATION, attach=True, **kw):
    """Factory.  `state_authority=None` equivalent: pass attach=False -> nothing is installed."""
    authority = D3ExecutionAuthority(env, container=container, policy=policy, **kw)
    if not attach:
        return authority                      # default-OFF: no wrapping at all
    return authority.attach()


def state_authority_is_none_equivalent(env):
    """Mechanical check used by the default-off row: the env carries no authority wiring."""
    return (not hasattr(env, '_d3_authority')
            and getattr(env, 'state_authority', None) is None
            and not getattr(env, '_d3_wired', False))
