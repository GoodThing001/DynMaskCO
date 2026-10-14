#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2B-2a: OBJECT-LEVEL write exclusion for the D3 authoritative state.

The user ruling after AT2B-1:

    "容器采用 adopt() 接管 env 原有车辆对象，而 D3ExecutionAuthority.vehicles() 仍返回权威可变对象。
     因此存在一个明确的逻辑缺口：状态只有一份，并不意味着只有一个合法写入者。...
     事后 epoch 检测不能代替事前拒绝。"

The gap is closed STRUCTURALLY, not by more bookkeeping.  The container's adopted vehicles are
converted, in place, into `AuthoritativeVehicleState` instances whose `__setattr__` consults a
thread-local write scope, and whose list-valued fields are `GuardedList`s whose mutating methods
consult the same scope.  A write is therefore refused BEFORE it happens unless the calling thread is
inside an authority-held scope:

    scope = None        -> every guarded write raises UnauthorizedWrite
    scope = 'execution' -> the environment's own routed primitives may write (the engine is the writer)
    scope = 'reveal'    -> ONLY `mutable_suffix` may be written, because the external reveal event
                           invalidates the revocable tail; commitment fields stay refused, so the
                           object-level guard itself enforces "a reveal may not revoke a commitment"

Honest scope limits, stated up front:
  * `served_mask` is a numpy array; in-place element writes on it are NOT intercepted at the object
    level (only the reveal verifier's V4 check covers it).  Recorded as a residual.
  * this is a single-process, cooperative-object guard: it does not defend against arbitrary
    reflection (`object.__setattr__`), ctypes, or malicious memory modification.
  * `detach()` restores the plain environment; exclusion is a property of the attached D3 mode.
"""
from __future__ import annotations

import copy
import threading

from strict_online_env import VehicleState

SCOPE_EXECUTION = 'execution'
SCOPE_REVEAL = 'reveal'
SCOPE_DECISION_CANDIDATE = 'decision_candidate'      # detached candidates ONLY (no authoritative write)
SCOPE_COMPAT_INSTALL = 'compat_candidate_install'    # LABELLED isolation compatibility mode
SCOPE_DECISION_COMMIT = 'decision_commit'            # authoritative write WITH certification
WRITE_SCOPES = (None, SCOPE_EXECUTION, SCOPE_REVEAL, SCOPE_DECISION_CANDIDATE,
                SCOPE_COMPAT_INSTALL, SCOPE_DECISION_COMMIT)

# fields a reveal invalidation may write: the revocable tail AND the replan trigger.  The user
# ruling is explicit that the reveal must not silently lose the replanning opportunity
# ("保留或正确设置重规划需求"), and the AT2B-1 measurement showed the reveal block sets
# needs_replan/replan_reason in the SAME block that clears the tail.  Commitment fields stay
# refused, so the object-level guard itself enforces "a reveal may not revoke a commitment".
# First-execution defect (preserved in raw/B3AT2B2A_run1_*): this tuple held only 'mutable_suffix',
# so the legitimate reveal write at strict_online_env.py:526 raised UnauthorizedWrite.
REVEAL_WRITABLE_FIELDS = ('mutable_suffix', 'needs_replan', 'replan_reason')
# every mutable field that the object guard protects
GUARDED_FIELDS = ('status', 'current_node', 'current_load', 'ready_time', 'committed_next',
                  'committed_arrive', 'committed_finish', 'committed_coldchain_before',
                  'mutable_suffix', 'served_route', 'dispatch_time', 'return_finish',
                  'needs_replan', 'replan_reason', 'coldchain_state', 'coldchain_updated_time',
                  'coldchain_accounting_anchor')

_SCOPE = threading.local()
_ATTEMPTS = []
_ATTEMPTS_LOCK = threading.Lock()


class UnauthorizedWrite(RuntimeError):
    """Raised BEFORE the write happens when the thread holds no authority write scope."""


def current_scope():
    return getattr(_SCOPE, 'mode', None)


def scope_owner():
    return getattr(_SCOPE, 'owner', None)


class write_scope(object):
    """Context manager that grants write permission to the current thread for one authority."""

    def __init__(self, owner, mode):
        if mode not in WRITE_SCOPES or mode is None:
            raise ValueError('write_scope needs %r or %r' % (SCOPE_EXECUTION, SCOPE_REVEAL))
        self.owner, self.mode = owner, mode
        self._prev = None

    def __enter__(self):
        self._prev = (current_scope(), scope_owner())
        _SCOPE.mode, _SCOPE.owner = self.mode, self.owner
        return self

    def __exit__(self, exc_type, exc, tb):
        _SCOPE.mode, _SCOPE.owner = self._prev
        return False


def record_attempt(kind, detail):
    with _ATTEMPTS_LOCK:
        _ATTEMPTS.append({'kind': kind, 'detail': detail, 'scope': current_scope(),
                          'thread': threading.get_ident()})


def attempts():
    return list(_ATTEMPTS)


def reset_attempts():
    with _ATTEMPTS_LOCK:
        _ATTEMPTS[:] = []


def _is_monotone_shrink(old, new):
    """True when `new` is `old` with its head consumed (or old is empty and new is anything? no)."""
    o = [int(x) for x in (old or ())]
    n = [int(x) for x in (new or ())]
    if len(n) > len(o):
        return False
    return o[len(o) - len(n):] == n if n else True


def _authorised(owner, field, old=None, new=None, policy=None):
    mode = current_scope()
    if mode == SCOPE_COMPAT_INSTALL and policy == 'certified':
        return False, 'compat_install_not_allowed_under_certified_policy'
    if mode is None or scope_owner() != owner:
        return False, 'no_write_scope'
    if mode == SCOPE_DECISION_CANDIDATE:
        # A detached-candidate scope must NEVER write the authoritative state (user ruling section 2).
        if field in GUARDED_FIELDS:
            return False, 'decision_candidate_must_not_write_authoritative_state'
        return True, 'decision_candidate_scope'
    if mode in (SCOPE_COMPAT_INSTALL, SCOPE_DECISION_COMMIT):
        # The ONLY scopes that may install a NEW suffix into the authoritative state.
        # SCOPE_COMPAT_INSTALL is the explicitly labelled isolation compatibility mode: every write
        # under it is recorded with certification_complete=false and is NOT a certified install.
        return True, mode + '_scope'
    if mode == SCOPE_EXECUTION:
        # RISK A (user ruling): a scope must not authorise arbitrary new suffixes by virtue of the
        # function that opened it.  In the execution scope `mutable_suffix` may only SHRINK (head
        # consumption); writing a new/unrelated suffix requires the dedicated decision-candidate
        # scope, which the authority opens only for a VALIDATED candidate.
        if field == 'mutable_suffix' and mode == SCOPE_EXECUTION:
            o = [int(x) for x in (old or ())]
            n = [int(x) for x in (new or ())]
            if n == o:
                return True, 'execution_scope_no_op'
            if len(n) == len(o) - 1 and o[1:] == n:
                # head-drop by exactly one == the commit loop's pop.  DEFERRED REFINEMENT (recorded):
                # tying this to "the head was actually committed" needs the vehicle context; without
                # it a bare head-drop is still allowed, which is disclosed as a residual.
                return True, 'execution_scope_head_consumed'
            return False, 'execution_scope_may_only_consume_the_head_or_refuse'
        return True, mode + '_scope'
    if mode == SCOPE_REVEAL:
        if field in REVEAL_WRITABLE_FIELDS:
            if field == 'mutable_suffix' and not _is_monotone_shrink(old, new):
                return False, 'reveal_scope_only_shrinks_the_tail'
            return True, 'reveal_scope_revocable_tail'
        return False, 'reveal_scope_may_not_write_%s' % field
    return False, 'unknown_scope'


class GuardedList(list):
    """A list whose every mutating method requires the owner's write scope.

    `__deepcopy__` deliberately returns a PLAIN list: the user ruling for run7 is that a detached
    copy must lose the authoritative identity across the WHOLE object graph, not only on the object
    that happened to raise first (`strict_online_env.py:368` refused `served_route.append`).
    """

    def __deepcopy__(self, memo):
        return [copy.deepcopy(x, memo) for x in self]

    def __copy__(self):
        return list(self)

    def __init__(self, values=(), owner=None, field=None):
        list.__init__(self, values)
        self._owner, self._field = owner, field

    def _check(self, op):
        ok, why = _authorised(self._owner, self._field)
        if not ok:
            record_attempt('in_place_list_%s' % op, {'field': self._field, 'why': why})
            raise UnauthorizedWrite('in-place %s on %r refused: %s' % (op, self._field, why))
        return True

    def append(self, x):
        self._check('append'); return list.append(self, x)

    def extend(self, xs):
        self._check('extend'); return list.extend(self, xs)

    def insert(self, i, x):
        self._check('insert'); return list.insert(self, i, x)

    def pop(self, *a):
        self._check('pop'); return list.pop(self, *a)

    def remove(self, x):
        self._check('remove'); return list.remove(self, x)

    def clear(self):
        self._check('clear'); return list.clear(self)

    def sort(self, *a, **kw):
        self._check('sort'); return list.sort(self, *a, **kw)

    def reverse(self):
        self._check('reverse'); return list.reverse(self)

    def __setitem__(self, i, v):
        self._check('setitem'); return list.__setitem__(self, i, v)

    def __delitem__(self, i):
        self._check('delitem'); return list.__delitem__(self, i)

    def __iadd__(self, xs):
        self._check('iadd'); return list.__iadd__(self, xs)


class AuthoritativeVehicleState(VehicleState):
    """`VehicleState` with an object-level write guard.

    Constructed from a plain instance (or from scratch) and then locked; from that moment every
    guarded attribute write goes through `_authorised`, so an unauthorised writer is refused BEFORE
    the state changes rather than detected afterwards.
    """

    def __init__(self, *args, owner=None, **kwargs):
        object.__setattr__(self, '_d3_owner', owner)
        object.__setattr__(self, '_d3_locked', False)
        VehicleState.__init__(self, *args, **kwargs)

    @classmethod
    def from_plain(cls, plain, owner):
        obj = cls(vehicle_id=int(plain.vehicle_id), owner=owner)
        for f in GUARDED_FIELDS:
            object.__setattr__(obj, f, copy.deepcopy(getattr(plain, f)))
        obj._guard_lists()
        object.__setattr__(obj, '_d3_locked', True)
        return obj

    def __deepcopy__(self, memo):
        """Every detached copy handed out is a PLAIN, freely mutable VehicleState.

        Measured defect of the second execution (preserved in raw/B3AT2B2A_run2_*): the default
        deepcopy copied _d3_locked=True and the owner, so the replanner's isolated copy was itself
        guarded and jf1h_repair.py:207 raised suffix_growth_requires_decision_scope.  Detached copies
        exist so external modules can work freely on state that cannot reach the authoritative
        objects, so they must NOT carry the guard.
        """
        plain = VehicleState(vehicle_id=int(self.vehicle_id))
        for f in GUARDED_FIELDS:
            object.__setattr__(plain, f, copy.deepcopy(getattr(self, f), memo))
        return plain

    def __copy__(self):
        return self.__deepcopy__({})

    def _guard_lists(self):
        for f in ('mutable_suffix', 'served_route'):
            values = list(getattr(self, f) or ())
            object.__setattr__(self, f, GuardedList(values, owner=self._d3_owner, field=f))

    def __setattr__(self, name, value):
        if name.startswith('_d3_') or not object.__getattribute__(self, '_d3_locked'):
            object.__setattr__(self, name, value)
            return
        if name in GUARDED_FIELDS:
            old_val = object.__getattribute__(self, name) if name in ('mutable_suffix',
                                                                        'served_route') else None
            ok, why = _authorised(self._d3_owner, name, old_val, value)
            if not ok:
                record_attempt('attribute_write', {'field': name, 'why': why,
                                                   'vehicle_id': int(self.vehicle_id)})
                raise UnauthorizedWrite('write to %r refused: %s' % (name, why))
            if name in ('mutable_suffix', 'served_route') and not isinstance(value, GuardedList):
                value = GuardedList(list(value or ()), owner=self._d3_owner, field=name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in GUARDED_FIELDS and object.__getattribute__(self, '_d3_locked'):
            old_val = object.__getattribute__(self, name) if name in ('mutable_suffix',
                                                                        'served_route') else None
            ok, why = _authorised(self._d3_owner, name, old_val, value)
            if not ok:
                record_attempt('attribute_delete', {'field': name, 'why': why})
                raise UnauthorizedWrite('delete of %r refused: %s' % (name, why))
        object.__delattr__(self, name)


def guard_authoritative_state(container, owner):
    """Convert the container's adopted vehicles in place into guarded authoritative instances.

    The environment keeps using the SAME list object (it borrowed it), so replacing the elements is
    what reorganises the env's state references without touching any historical code.  The replaced
    plain objects are returned so the test can prove that a reference held from before the handover
    can no longer move the authoritative state.
    """
    plain = list(container._vehicles)
    container._vehicles[:] = [AuthoritativeVehicleState.from_plain(v, owner) for v in plain]
    for tr in container._traces:
        for f in ('services', 'depot_unload_records'):
            values = list(getattr(tr, f) or ())
            setattr(tr, f, GuardedList(values, owner=owner, field='traces.%s' % f))
    return plain


def read_only_view(vehicles):
    from d3_state_container import ReadOnlyStateView
    return ReadOnlyStateView(vehicles, epoch=None, container_name='authoritative')
