#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2B-2a: OBJECT-LEVEL write exclusion installed ON TOP of the AT2B-1 authority.

User ruling (2026-10-14):

    "容器采用 adopt() 接管 env 原有车辆对象，而 D3ExecutionAuthority.vehicles() 仍返回权威可变对象。
     因此存在一个明确的逻辑缺口：状态只有一份，并不意味着只有一个合法写入者。...
     事后 epoch 检测不能代替事前拒绝。"

Layering (nothing in the AT2B-1 module is modified, and nothing in `strict_online_env.py`):

    strict_online_env  ->  D3ExecutionAuthority routers (AT2B-1)  ->  WriteExclusion (this module)

`WriteExclusion.attach()` wraps the six primitives the environment uses and enforces:

  * TOP-LEVEL entries (`_plan_and_commit`, `_advance_fleet`, `_force_return_all`) OPEN an EXECUTION
    write scope for the duration of the environment's own call, so the engine remains the writer;
  * NESTED primitives (`_commit`, `_advance_coldchain_to`, `prepare_decision_point`) REQUIRE an open
    scope -- so a direct unauthorised `env._commit(...)` is refused BEFORE any write;
  * the REVEAL window (between the end of `_advance_fleet` and the start of `_plan_and_commit`, where
    `_run_loop` runs the reveal/booking branches) holds a REVEAL scope that permits ONLY the revocable
    tail and the replan trigger -- commitment fields stay refused by the object guard itself;
  * the authoritative objects are converted, in place, into `AuthoritativeVehicleState`, so a write
    attempted from outside every scope (another thread, a hook callback, a stale reference, or code
    after the run) raises `UnauthorizedWrite` and the state does not move;
  * `vehicles()` returns a READ-ONLY view -- the mutable authoritative list is never handed out.

Residuals, recorded honestly: `served_mask` is a numpy array and in-place writes on it are not
intercepted (the reveal verifier's V4 covers it); arbitrary reflection (`object.__setattr__`), ctypes
and malicious memory modification are outside the threat model.
"""
from __future__ import annotations

import copy

import d3_authoritative_state as AS
import d3_state_container as SC

TOP_LEVEL = ('_plan_and_commit', '_advance_fleet', '_force_return_all')
NESTED = ('_commit', '_advance_coldchain_to', 'prepare_decision_point')


class WriteExclusion(object):
    """Installs object-level write exclusion and read-only accessors for one authority."""

    MODES = ('strict', 'compat_isolation')

    def __init__(self, authority, *, name=None, mode='strict'):
        if mode not in self.MODES:
            raise ValueError('unknown exclusion mode %r' % (mode,))
        self.mode = mode
        self.candidate_records = []
        self.authority = authority
        self.env = authority.env
        self.owner = name or (authority.instance_id + ':excl')
        self.attached = False
        self.guarded = False
        self.pre_handover_objects = []
        self.scope_openings = []
        self.refusals = []
        self._orig = {}
        self._reveal_scope = None

    # ---- attach -----------------------------------------------------------------------------
    def attach(self):
        if self.attached:
            return self
        env = self.env
        bound = self
        self._orig = {n: getattr(env, n) for n in TOP_LEVEL + NESTED}

        def _exec_scope():
            # In `compat_isolation` the labelled compatibility scope is ALSO held, so the AT2B-1
            # candidate publication can still reach the authoritative state; every such write is
            # recorded as an uncertified compatibility install and is never counted as certified.
            return AS.write_scope(self.owner,
                                  AS.SCOPE_COMPAT_INSTALL if self.mode == 'compat_isolation'
                                  else AS.SCOPE_EXECUTION)

        def _require(name):
            if AS.current_scope() is None or AS.scope_owner() != self.owner:
                bound.refusals.append({'primitive': name, 'why': 'no_write_scope_for_this_owner',
                                       'scope': AS.current_scope(),
                                       'scope_owner': AS.scope_owner()})
                AS.record_attempt('unauthorised_primitive_call', {'primitive': name})
                raise AS.UnauthorizedWrite(
                    '%s refused: no authority write scope is held by this thread' % name)

        def _commit(inst_idx, clock, v, traces, node):
            _require('_commit')
            return bound._orig['_commit'](inst_idx, clock, v, traces, node)

        def prepare_decision_point(clock, vehicles):
            _require('prepare_decision_point')
            return bound._orig['prepare_decision_point'](clock, vehicles)

        def _advance_coldchain_to(v, clock):
            _require('_advance_coldchain_to')
            return bound._orig['_advance_coldchain_to'](v, clock)

        def _advance_fleet(inst_idx, clock, vehicles, traces, served_mask):
            bound._guard_after_adoption()
            bound.scope_openings.append({'primitive': '_advance_fleet', 'mode': 'execution',
                                         'clock': float(clock)})
            with _exec_scope():
                out = bound._orig['_advance_fleet'](inst_idx, clock, vehicles, traces, served_mask)
            if (bound._is_clock(inst_idx, clock) or bound._is_booking_clock(clock)):
                if bound._reveal_scope is None:
                    bound._reveal_scope = AS.write_scope(bound.owner, AS.SCOPE_REVEAL)
                    bound._reveal_scope.__enter__()
                    bound.scope_openings.append({'primitive': 'reveal_window', 'mode': 'reveal',
                                                 'clock': float(clock)})
            return out

        def _force_return_all(inst_idx, clock, vehicles, traces, served_mask):
            bound._guard_after_adoption()
            bound.scope_openings.append({'primitive': '_force_return_all', 'mode': 'execution',
                                         'clock': float(clock)})
            with _exec_scope():
                return bound._orig['_force_return_all'](inst_idx, clock, vehicles, traces,
                                                        served_mask)

        def _plan_and_commit(inst_idx, clock, vehicles, traces, served_mask, all_customers):
            bound._guard_after_adoption()
            if bound._reveal_scope is not None:
                bound._reveal_scope.__exit__(None, None, None)
                bound._reveal_scope = None
            bound.scope_openings.append({'primitive': '_plan_and_commit', 'mode': 'execution',
                                         'clock': float(clock)})
            with _exec_scope():
                return bound._orig['_plan_and_commit'](inst_idx, clock, vehicles, traces,
                                                       served_mask, all_customers)

        for name, fn in (('_commit', _commit), ('prepare_decision_point', prepare_decision_point),
                         ('_advance_coldchain_to', _advance_coldchain_to),
                         ('_advance_fleet', _advance_fleet),
                         ('_force_return_all', _force_return_all),
                         ('_plan_and_commit', _plan_and_commit)):
            setattr(env, name, fn)
        self.attached = True
        return self

    def detach(self):
        for name, fn in self._orig.items():
            setattr(self.env, name, fn)
        self.attached = False

    # ---- enforcement helpers ----------------------------------------------------------------
    def _guard_after_adoption(self):
        """Convert the container's adopted vehicles into guarded instances, in place (once)."""
        container = self.authority.container
        if self.guarded or not container.adopted:
            return
        self.pre_handover_objects = list(container._vehicles)          # the PLAIN objects
        AS.guard_authoritative_state(container, self.owner)
        self.guarded = True

    def _is_clock(self, inst_idx, clock):
        return self.authority._is_reveal_clock(inst_idx, clock)

    def _is_booking_clock(self, clock):
        bh = getattr(self.env, 'booking_horizon', None)
        return bh is not None and abs(float(bh) - float(clock)) < 1e-6

    # ---- accessors: the mutable list is never handed out ------------------------------------
    def vehicles(self):
        return AS.read_only_view(self.authority.container._vehicles)

    def authoritative(self):
        """The mutable list, for the authority's own use inside a scope.  Not for callers."""
        return self.authority.container._vehicles

    def snapshot(self):
        return {'mode': self.mode, 'exclusion_attached': self.attached, 'objects_guarded': self.guarded,
                'n_candidate_records': len(self.candidate_records),
                'candidate_records': self.candidate_records[:4],
                'compat_isolation_writes_are_uncertified': self.mode == 'compat_isolation',
                'owner': self.owner,
                'n_pre_handover_plain_objects': len(self.pre_handover_objects),
                'n_scope_openings': len(self.scope_openings),
                'scope_openings_head': self.scope_openings[:6],
                'n_refusals': len(self.refusals), 'refusals': self.refusals[:6],
                'write_attempts_observed': len(AS.attempts()),
                'residual_not_intercepted': ['served_mask (numpy array) in-place element writes'],
                'threat_model_excludes': ['arbitrary reflection (object.__setattr__)', 'ctypes',
                                          'malicious memory modification'],
                'vehicles_accessor_returns_read_only_view': True}


def make_authority_with_exclusion(env, container, *, instance_id='d3-exec-2a',
                                  mode='strict'):
    """Convenience: the AT2B-1 authority plus object-level write exclusion.

    `mode='strict'` is the default and refuses every uncertified authoritative write;
    `mode='compat_isolation'` is the LABELLED isolation compatibility mode that lets the AT2B-1
    candidate publication reach the authoritative state and records those writes as uncertified
    (never as a `certified` install).
    """
    import d3_execution_authority as EA
    authority = EA.D3ExecutionAuthority(env, container=container, instance_id=instance_id)
    authority.attach()
    exclusion = WriteExclusion(authority, name=instance_id + ':excl', mode=mode)
    exclusion.attach()
    return authority, exclusion
