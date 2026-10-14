#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2G: the D3 STATE CONTAINER -- state ownership, so the attribute-write bypass
is closed.  Isolated: not wired to any production run.

Why this module exists (user ruling 2026-10-14, after B3-AT2A):

    "AT2-F10 = REFUSED_BUT_ONE_WRITE_PATH_ESCAPED ... 实例守卫覆盖 _commit 和 _plan_and_commit，但
     没有覆盖 vehicle.mutable_suffix = ...  这说明守卫保护的是部分函数入口，而不是状态对象本身。"
    "仅把更多方法包裹起来也不充分，因为旧代码仍可能持有 VehicleState 的可变引用。"
    "由 D3 专用状态容器独占真实 VehicleState，外部模块只接收只读快照或脱离副本。"

The enforcement is therefore structural rather than a longer list of wrapped methods:

  * `D3StateContainer` INGESTS the real state by deep copy.  The authoritative `VehicleState` objects
    are private to the container and are never returned to any caller.
  * `read_only_view()` returns a frozen projection (immutable sequence of frozen vehicle views whose
    `mutable_suffix` is a tuple): assignment, in-place list mutation and element replacement all
    raise, so there is no attribute to assign to.
  * `detached_copy()` returns a deep copy for candidate work; mutating it cannot touch the container.
  * `publish(...)` is the ONLY mutation path: it requires the coordinator's write capability and a
    matching state epoch, applies the certified candidate to the container's OWN objects, and then
    performs an independent post-read of those same objects.

Ownership is NOT atomicity, and this module does not claim otherwise: the vocabulary stays
STAGED_VALIDATED / PUBLISHING / COMMITTED / COMMITTED_VERIFIED / PARTIAL_COMMIT / OUTCOME_UNKNOWN, and
`max_concurrent_publish_windows == 1` only says the controlled publish windows did not overlap -- it
does not say an external observer can never see an intermediate state.
"""
from __future__ import annotations

import copy
import hashlib
import threading

from d3_commit_transaction import (PUBLISH_COMMITTED, PUBLISH_COMMITTED_VERIFIED, PUBLISH_PARTIAL,
                                   PUBLISH_PUBLISHING, PUBLISH_STAGED_VALIDATED, PUBLISH_UNKNOWN,
                                   PUBLISH_STATES, commit_identity, state_version)

CONTAINER_SCHEMA = 'd3_state_container_v1'
VIEW_SCHEMA = 'd3_read_only_state_view_v1'


class FrozenStateError(RuntimeError):
    """Raised when a caller tries to mutate a read-only projection of the authoritative state."""


class FrozenVehicle(object):
    """An immutable projection of one authoritative vehicle.  Every mutation route raises."""

    __slots__ = ('_f',)

    def __init__(self, vehicle):
        object.__setattr__(self, '_f', {
            'vehicle_id': int(vehicle.vehicle_id),
            'status': str(vehicle.status),
            'current_node': int(vehicle.current_node),
            'current_load': float(vehicle.current_load),
            'ready_time': float(vehicle.ready_time),
            'committed_next': (None if vehicle.committed_next is None
                               else int(vehicle.committed_next)),
            'committed_arrive': getattr(vehicle, 'committed_arrive', None),
            'needs_replan': bool(getattr(vehicle, 'needs_replan', False)),
            # tuple, not list: `view.mutable_suffix.append(...)` cannot even be attempted
            'mutable_suffix': tuple(int(x) for x in vehicle.mutable_suffix),
            'coldchain_frozen': True,
        })

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, '_f')[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        raise FrozenStateError('read-only projection: cannot set %r on FrozenVehicle' % name)

    def __delattr__(self, name):
        raise FrozenStateError('read-only projection: cannot delete %r on FrozenVehicle' % name)

    def __repr__(self):                                                    # pragma: no cover
        return '<FrozenVehicle %d %s>' % (self._f['vehicle_id'], self._f['status'])


class ReadOnlyStateView(object):
    """Immutable sequence of `FrozenVehicle` -- the only thing external modules may receive."""

    SCHEMA = VIEW_SCHEMA
    frozen = True

    def __init__(self, vehicles, *, epoch, container_name):
        self._items = tuple(FrozenVehicle(v) for v in vehicles)
        self._epoch = epoch
        self._container_name = container_name

    # sequence protocol (read-only)
    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._items[i]
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    # every mutation route raises
    def __setitem__(self, i, value):
        raise FrozenStateError('read-only view: cannot replace element %r' % (i,))

    def __delitem__(self, i):
        raise FrozenStateError('read-only view: cannot delete element %r' % (i,))

    def append(self, value):
        raise FrozenStateError('read-only view: cannot append')

    def extend(self, values):
        raise FrozenStateError('read-only view: cannot extend')

    def insert(self, i, value):
        raise FrozenStateError('read-only view: cannot insert')

    def pop(self, *a):
        raise FrozenStateError('read-only view: cannot pop')

    def sort(self, *a, **kw):
        raise FrozenStateError('read-only view: cannot sort')

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        raise FrozenStateError('read-only view: cannot set %r' % name)

    @property
    def epoch(self):
        return self._epoch

    def snapshot(self):
        return {'view_schema': VIEW_SCHEMA, 'container': self._container_name,
                'epoch': self._epoch,
                'vehicles': [{'vehicle_id': v.vehicle_id, 'status': v.status,
                              'current_node': v.current_node,
                              'committed_next': v.committed_next,
                              'mutable_suffix': list(v.mutable_suffix)}
                             for v in self._items]}


class D3StateContainer(object):
    """The single owner of the authoritative D3 vehicle state.

    `hand_out_*` never returns an authoritative object; `publish` is the only mutator and it demands
    the coordinator's capability plus a matching epoch.
    """

    registered_on_env = False

    def __init__(self, vehicles, traces, served_mask, *, name='d3-state', env=None, inst=None,
                 clock=None, event=None, reveal_idx=0, all_customers=None):
        # INGEST BY DEEP COPY: the caller keeps its own objects, and later mutation of them cannot
        # reach the authoritative state (this is what makes stale references ineffective).
        self._vehicles = [copy.deepcopy(v) for v in vehicles]
        self._traces = copy.deepcopy(traces)
        self._served_mask = copy.deepcopy(served_mask)
        self.name = name
        self.env = env
        self.inst = inst
        self.clock = clock
        self.event = event
        self.reveal_idx = reveal_idx
        self.all_customers = list(all_customers or [])
        self.adopted = False
        self.publishes = 0
        self.partial_publishes = 0
        self.refused = 0
        self.events = []

    # ---- owned state, never exposed -----------------------------------------------------------
    def _authoritative(self):
        return self._vehicles

    # ---- D3 mode: ADOPT the environment's own list (no copy) ---------------------------------
    def adopt(self, vehicles, traces=None, served_mask=None):
        """Take ownership of the environment's own list WITHOUT copying.

        Used by the AT2B-1 D3 mode: the environment keeps a borrowed reference, the container owns
        the objects and is the only mutator, so there is exactly ONE authoritative state (the user
        ruling forbids an env copy and a container copy running in parallel).  The ingest-by-deepcopy
        path above remains the behaviour for external callers.
        """
        self._vehicles = vehicles
        if traces is not None:
            self._traces = traces
        if served_mask is not None:
            self._served_mask = served_mask
        self.adopted = True
        return self

    def read_only_view(self):
        return ReadOnlyStateView(self._vehicles, epoch=self.epoch(), container_name=self.name)

    def detached_copy(self):
        return ([copy.deepcopy(v) for v in self._vehicles], copy.deepcopy(self._traces),
                copy.deepcopy(self._served_mask))

    def set_context(self, inst=None, clock=None, event=None, reveal_idx=None):
        """Keep the sanctioned (inst, clock, event, reveal_idx) context current during a run.

        Measured defect of the first version: the container was built for a single decision point
        (inst/clock set) and then reused for a live run where they were None, so `epoch()` crashed
        inside `capture_recourse_snapshot`.  The epoch now falls back to a field hash when the
        context is incomplete, and the authority keeps the context up to date.
        """
        if inst is not None:
            self.inst = inst
        if clock is not None:
            self.clock = clock
        if event is not None:
            self.event = event
        if reveal_idx is not None:
            self.reveal_idx = reveal_idx
        return self

    def epoch(self):
        """Version of the authoritative state, computed from the container's OWN objects."""
        if (self.env is not None and self.inst is not None and self.clock is not None
                and self._traces):
            return state_version(self.env, self.inst, self.clock, self.event, self.reveal_idx,
                                 self._vehicles, self._traces, self._served_mask, self.all_customers)
        h = hashlib.sha256()
        for v in sorted(self._vehicles, key=lambda x: int(x.vehicle_id)):
            h.update(('%d|%s|%d|%s|%s;' % (int(v.vehicle_id), v.status, int(v.current_node),
                                           None if v.committed_next is None else int(v.committed_next),
                                           tuple(int(x) for x in v.mutable_suffix))).encode())
        return h.hexdigest()

    def field_evidence(self):
        return [{'vehicle_id': int(v.vehicle_id), 'status': v.status,
                 'current_node': int(v.current_node),
                 'committed_next': (None if v.committed_next is None else int(v.committed_next)),
                 'mutable_suffix': [int(x) for x in v.mutable_suffix]}
                for v in sorted(self._vehicles, key=lambda x: int(x.vehicle_id))]

    # ---- the ONLY mutation path ---------------------------------------------------------------
    def publish(self, staged, *, capability, expected_epoch, is_capability_live,
                in_critical_section=None, publish_fn=None, container_hook=None):
        """Publish the certified candidate into the container's OWN objects.

        Requirements: the capability must be live on the coordinator boundary, the publisher must be
        inside the coordinator's critical section, and the epoch must still match.  A failure
        mid-publish is reported as PARTIAL_COMMIT with the true fields, and the post-read is what
        decides COMMITTED vs COMMITTED_VERIFIED.

        MEASURED DEFECT of the first version: the authorisation check happened only implicitly, when
        the guarded `_commit` raised *after* the container had already staged `mutable_suffix` on its
        own objects.  That left an unauthorised mutation behind, which then tripped the epoch check of
        the legitimate publish.  The authorisation pre-flight is therefore the FIRST thing that
        happens, before any mutation.
        """
        out = {'container_schema': CONTAINER_SCHEMA, 'container': self.name,
               'publish_state': None, 'publish_state_trace': [],
               'epoch_at_publish_start': self.epoch(), 'capability_presented': capability is not None,
               'capability_live': bool(is_capability_live(capability)),
               'in_critical_section': (None if in_critical_section is None
                                       else bool(in_critical_section()))}
        if not out['capability_live'] or (in_critical_section is not None
                                         and not out['in_critical_section']):
            self.refused += 1
            out.update({'publish_state': None, 'outcome': 'REFUSED_NO_MUTATION',
                        'refusals': ['not_authorised_before_any_mutation'],
                        'nothing_written': True,
                        'why': ('the container is the only mutator and it demands the live capability '
                                'from inside the coordinator critical section; the refusal happens '
                                'BEFORE any mutation, so nothing is left behind')})
            return out
        if expected_epoch is not None and out['epoch_at_publish_start'] != expected_epoch:
            self.refused += 1
            out.update({'publish_state': None, 'outcome': 'REFUSED_NO_MUTATION',
                        'refusals': ['state_changed_before_publish'],
                        'nothing_written': True,
                        'why': 'the authoritative epoch moved since the authorisation'})
            return out

        by_id = {int(v.vehicle_id): v for v in self._vehicles}
        out['publish_state'] = PUBLISH_STAGED_VALIDATED
        out['publish_state_trace'] = [PUBLISH_STAGED_VALIDATED]
        out['publish_state'] = PUBLISH_PUBLISHING
        out['publish_state_trace'].append(PUBLISH_PUBLISHING)
        pre = self.field_evidence()
        if container_hook is not None:
            container_hook(self, out)
        published = []
        try:
            if publish_fn is None:
                for vid, v in by_id.items():
                    v.mutable_suffix = list(staged.get(vid, []))
                    v.needs_replan = False
                for v in self._vehicles:
                    if v.status in ('idle', 'ready') and v.mutable_suffix:
                        nxt = int(v.mutable_suffix[0])
                        v.mutable_suffix = v.mutable_suffix[1:]
                        if self.env is not None:
                            self.env._commit(self.inst, self.clock, v, self._traces, nxt)
                        else:
                            v.committed_next = nxt
                            v.status = 'committed'
                out['publish_state'] = PUBLISH_COMMITTED
                out['publish_state_trace'].append(PUBLISH_COMMITTED)
                out['outcome'] = PUBLISH_COMMITTED
                self.publishes += 1
                self.events.append({'action': 'publish', 'n_vehicles': len(staged)})
            else:
                for vid in sorted(staged):
                    publish_fn(by_id[vid], staged[vid])
                    published.append(int(vid))
                out['publish_state'] = PUBLISH_COMMITTED
                out['publish_state_trace'].append(PUBLISH_COMMITTED)
                out['outcome'] = PUBLISH_COMMITTED
                out['published_vehicles'] = published
                self.publishes += 1
        except BaseException as exc:                                            # noqa: BLE001
            name = type(exc).__name__
            out['exception'] = '%s: %s' % (name, str(exc)[:120])
            post = self.field_evidence()
            if name in ('UnauthorizedWrite', 'FrozenStateError'):
                # An authorisation / projection refusal is NOT a partial publish: nothing was
                # written.  Reporting it as PARTIAL_COMMIT would break the vocabulary
                # ("PARTIAL_COMMIT = 已确认部分状态改变") and would hide an authorisation failure
                # inside a state-change report.  Measured defect of the first version.
                self.refused += 1
                out['publish_state'] = None
                out['publish_state_trace'].append('REFUSED_NO_MUTATION')
                out['outcome'] = 'REFUSED_NO_MUTATION'
                out['refusals'] = ['write_refused_by_the_instance_guard']
                out['nothing_written'] = (post == pre)
                out['why'] = ('refused by the instance guard: the container may only be mutated from '
                              'inside the coordinator critical section (or by a live capability); no '
                              'state change occurred')
            elif post != pre:
                self.partial_publishes += 1
                out['publish_state'] = PUBLISH_PARTIAL
                out['publish_state_trace'].append(PUBLISH_PARTIAL)
                out['outcome'] = PUBLISH_PARTIAL
                out['rollback_claimed'] = False
                out['n_published_before_failure'] = len(published)
                out['partial_change_confirmed'] = True
                out['why'] = ('a partial publish is reported as PARTIAL_COMMIT with the true fields '
                              'of the container; it is NOT called a rollback')
            else:
                out['publish_state'] = PUBLISH_UNKNOWN
                out['publish_state_trace'].append(PUBLISH_UNKNOWN)
                out['outcome'] = PUBLISH_UNKNOWN
                out['why'] = ('an exception was raised and the container fields did NOT change, so '
                              'this is neither a commit nor a partial commit: OUTCOME_UNKNOWN')
        out['field_evidence_after'] = self.field_evidence()
        return out

    def post_read(self, staged):
        """Independent re-read OF THE CONTAINER'S OWN OBJECTS, compared with the authorisation."""
        rows, mismatches = [], []
        by_id = {int(v.vehicle_id): v for v in self._vehicles}
        for vid in sorted(staged):
            v = by_id.get(vid)
            observed = None if v is None or v.committed_next is None else int(v.committed_next)
            expected = int(staged[vid][0]) if staged[vid] else None
            rows.append({'vehicle_id': int(vid), 'expected_committed_head': expected,
                         'observed_committed_head': observed,
                         'source': 'container._vehicles (the authoritative objects)'})
            if expected != observed:
                mismatches.append({'vehicle_id': int(vid), 'expected': expected,
                                   'observed': observed})
        return {'method': 'read-only re-read of the container OWN objects',
                'rows': rows, 'mismatches': mismatches,
                'from_the_authoritative_container': True,
                'matches_the_authorised_result': not mismatches}

    def snapshot(self):
        return {'container_schema': CONTAINER_SCHEMA, 'name': self.name,
                'registered_on_env': False, 'n_authoritative_vehicles': len(self._vehicles),
                'n_publishes': self.publishes, 'n_partial_publishes': self.partial_publishes,
                'n_refused': self.refused, 'epoch': self.epoch(), 'events': self.events,
                'owns_the_real_vehicles': True,
                'external_modules_receive': ['read_only_view (frozen)', 'detached_copy (deep copy)']}
