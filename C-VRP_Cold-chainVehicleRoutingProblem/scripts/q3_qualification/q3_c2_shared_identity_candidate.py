#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 MINIMAL ISOLATED CANDIDATE  --  `C2_SHARED_IDENTITY_V1`.

Builds ONE trusted strict-online frozen state, a projection that can be RECOMPUTED from it, and a
guard that makes it impossible for a certifier to read an unrevealed order's attributes through the
raw instance arrays.

WHY THE GUARD EXISTS
====================
The previous round measured that M1-a and C0 consume FULL instance arrays (`env.demands`,
`env.dist_mat`, `env.tw_class`, ...) indexed by whatever order ids the caller supplies.  A frozen
projection that merely *hides* unrevealed orders while the arrays still contain them is not a
strict-online guarantee: the certifier can still index the future.  So this candidate hands the real
certifiers a GUARDED view whose node indices outside the visible set raise
`HIDDEN_FUTURE_ATTRIBUTE_REQUESTED` and are logged.

THE VISIBLE SET IS NOT DECIDED HERE
===================================
`visible_ids_authoritative()` reproduces the EVENT ENGINE's own comparison
(`reveal_time <= clock + eps_event`, `strict_online_env.py:490/:514/:538`) with the engine's own
tolerance.  No private tolerance is invented -- the previous round used 1e-9 without justification and
that is registered as AUD-2.  A caller-supplied visible set is accepted only if it EQUALS the
authoritative one; a superset is refused.

WHAT THIS IS NOT
================
Not a production contract, not a certification, not install atomicity, and not a sandbox: the guard is
a Python-level check at indexing depth 1 (nested array views are not re-guarded), which is recorded in
`DOMAIN_LIMITS`.  Identity agreement never confers physical qualification.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'C2_SHARED_IDENTITY_V1'
SCHEMA = 'q3_c2_shared_identity'

#: the EVENT ENGINE's own reveal comparison, with the engine's own tolerance (do not invent one)
EPS_EVENT = 1e-6
VISIBLE_RULE_SOURCE = ('strict_online_env.py:490/:514/:538 -- '
                       '`reveal_time[inst, i] <= clock + 1e-6`')

DOMAIN_LIMITS = (
    'the guard checks indexing at depth 1: a nested numpy view is not re-guarded',
    'in-process Python checks are not a security sandbox against hostile code',
    'no coverage of threads, external processes or escaped references',
    'no statement about install atomicity (B-9b is out of G-3 and out of this candidate)',
    'identity agreement is never a physical qualification',
)


class Refusal(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


# =============================================================================================
# encoding (reuses the audited conventions: delimited fields, bit-exact floats, full digest)
# =============================================================================================
def enc_bits(x):
    return struct.pack('>d', float(x)).hex()


def _f(name, payload):
    payload = str(payload)
    return '%d:%s%d:%s' % (len(name), name, len(payload), payload)


def _time(x, field):
    v = float(x)
    if not math.isfinite(v):
        raise Refusal('NON_FINITE_TIME', {'field': field, 'got': repr(x)})
    if v == 0.0 and math.copysign(1.0, v) < 0:
        raise Refusal('NEGATIVE_ZERO_TIME', {'field': field})
    return enc_bits(v)


# =============================================================================================
# the authoritative visible set (the ENGINE's rule, not ours)
# =============================================================================================
def visible_ids_authoritative(reveal_time_row, clock, eps_event=EPS_EVENT):
    """Reproduce the event engine's own visibility comparison exactly."""
    out = []
    for i in range(1, len(reveal_time_row)):
        rt = float(reveal_time_row[i])
        if rt <= float(clock) + float(eps_event):
            out.append(int(i))
    return sorted(out)


# =============================================================================================
# the frozen visible state -- unrevealed attributes are NEVER read
# =============================================================================================
@dataclasses.dataclass(frozen=True)
class FrozenOrder(object):
    order_id: int
    demand: float
    tw_start: float
    tw_end: float
    temp_class: int
    dtypes: tuple


@dataclasses.dataclass(frozen=True)
class FrozenVehicle(object):
    vehicle_id: int
    anchor_node: int
    anchor_time: float
    anchor_load: float
    status: str


@dataclasses.dataclass(frozen=True)
class FrozenVisibleState(object):
    clock: float
    instance_id: int
    event_id: int
    contract_hash: str
    orders: tuple                 # ONLY the visible ones, by construction
    vehicles: tuple
    executed_prefix: tuple
    committed_order_ids: tuple
    source_identity: tuple
    visible_ids: tuple
    eps_event: float

    def order_ids(self):
        return tuple(o.order_id for o in self.orders)


def freeze_visible_state(*, clock, instance_id, event_id, contract_hash, reveal_time_row,
                         demands_row, tw_start_row, tw_end_row, temp_class_row,
                         vehicles, executed_prefix, committed_order_ids, source_identity,
                         claimed_visible_ids=None, eps_event=EPS_EVENT):
    """Freeze the AUTHORITATIVE visible part of the state.

    Attributes of unrevealed orders are never touched: the loop runs over the authoritative visible
    set only, so a future order's demand / time window / temperature class cannot enter the identity.
    """
    authoritative = visible_ids_authoritative(reveal_time_row, clock, eps_event)
    if claimed_visible_ids is not None:
        claimed = sorted(int(i) for i in claimed_visible_ids)
        if claimed != authoritative:
            extra = sorted(set(claimed) - set(authoritative))
            raise Refusal('VISIBLE_SET_MISMATCH', {
                'claimed': claimed, 'authoritative': authoritative, 'not_visible': extra,
                'why': 'a caller may not invent or extend the visible set; the engine decides it'})
    orders = []
    for i in authoritative:
        rt = float(reveal_time_row[i])
        if rt > float(clock) + float(eps_event):
            raise Refusal('HIDDEN_FUTURE_ATTRIBUTE_REQUESTED', {'order': i})
        orders.append(FrozenOrder(order_id=int(i),
                                  demand=float(demands_row[i]),
                                  tw_start=float(tw_start_row[i]),
                                  tw_end=float(tw_end_row[i]),
                                  temp_class=int(temp_class_row[i]),
                                  dtypes=('float', 'float', 'float', 'int')))
    frozen = FrozenVisibleState(
        clock=float(clock), instance_id=int(instance_id), event_id=int(event_id),
        contract_hash=str(contract_hash), orders=tuple(orders),
        vehicles=tuple(FrozenVehicle(int(v['vehicle_id']), int(v['anchor_node']),
                                     float(v['anchor_time']), float(v['anchor_load']),
                                     str(v['status'])) for v in vehicles),
        executed_prefix=tuple(int(c) for c in executed_prefix),
        committed_order_ids=tuple(int(c) for c in committed_order_ids),
        source_identity=tuple(sorted((str(k), str(v)) for k, v in (source_identity or {}).items())),
        visible_ids=tuple(authoritative), eps_event=float(eps_event))
    assert_visible_only(frozen)
    return frozen


def assert_visible_only(frozen):
    """No order in the frozen state may be outside its own visible set."""
    vis = set(frozen.visible_ids)
    illegal = [o.order_id for o in frozen.orders if o.order_id not in vis]
    if illegal:
        raise Refusal('FROZEN_STATE_CONTAINS_UNREVEALED', {'orders': illegal})
    return True


# =============================================================================================
# the identity: visible-only, delimited, bit-exact
# =============================================================================================
def visible_identity(frozen):
    assert_visible_only(frozen)
    parts = [_f('schema', SCHEMA), _f('version', IMPLEMENTATION_VERSION),
             _f('clock', _time(frozen.clock, 'clock')),
             _f('instance', str(frozen.instance_id)), _f('event', str(frozen.event_id)),
             _f('contract', frozen.contract_hash),
             _f('eps_event', repr(frozen.eps_event)),
             _f('n_visible', str(len(frozen.visible_ids)))]
    for c in frozen.visible_ids:
        parts.append(_f('visible', str(int(c))))
    for o in frozen.orders:
        parts += [_f('order', str(o.order_id)), _f('demand', _time(o.demand, 'demand')),
                  _f('tw_start', _time(o.tw_start, 'tw_start')),
                  _f('tw_end', _time(o.tw_end, 'tw_end')),
                  _f('temp_class', str(o.temp_class))]
    for v in frozen.vehicles:
        parts += [_f('vehicle', str(v.vehicle_id)), _f('anchor_node', str(v.anchor_node)),
                  _f('anchor_time', _time(v.anchor_time, 'anchor_time')),
                  _f('anchor_load', _time(v.anchor_load, 'anchor_load')),
                  _f('status', v.status)]
    for c in frozen.executed_prefix:
        parts.append(_f('executed', str(int(c))))
    for c in frozen.committed_order_ids:
        parts.append(_f('committed', str(int(c))))
    for k, v in frozen.source_identity:
        parts.append(_f('source', '%s=%s' % (k, v)))
    return hashlib.sha256(''.join(parts).encode('utf-8')).hexdigest()


# =============================================================================================
# the projection and its verification (P-a..P-d, with RECOMPUTATION)
# =============================================================================================
def content_digest(orders):
    return hashlib.sha256(''.join(_f('order', str(int(o))) for o in orders).encode('utf-8')).hexdigest()


def project_visible(frozen):
    orders = list(frozen.order_ids())
    return {'derived_from': visible_identity(frozen), 'content': orders,
            'content_digest': content_digest(orders),
            'visible_ids': list(frozen.visible_ids), 'eps_event': frozen.eps_event,
            'rule_source': VISIBLE_RULE_SOURCE}


def verify_projection(proj, frozen):
    """P-a label, P-b self-consistency, P-c RECOMPUTE, P-d strict-online visibility."""
    problems = []
    if proj.get('derived_from') != visible_identity(frozen):
        problems.append('P-a source label mismatch')
    if proj.get('content_digest') != content_digest(proj.get('content') or []):
        problems.append('P-b content/digest mismatch')
    recomputed = list(frozen.order_ids())
    if [int(x) for x in (proj.get('content') or [])] != recomputed:
        problems.append('P-c content != recomputed projection')
    if [int(x) for x in (proj.get('visible_ids') or [])] != list(frozen.visible_ids):
        problems.append('P-d visible set mismatch')
    if float(proj.get('eps_event', -1)) != float(frozen.eps_event):
        problems.append('P-d tolerance mismatch (a projection may not carry its own tolerance)')
    bad = [int(o) for o in (proj.get('content') or []) if int(o) not in set(frozen.visible_ids)]
    if bad:
        problems.append('P-d unrevealed order present: %s' % bad)
    return {'verified': not problems, 'problems': problems}


# =============================================================================================
# the guard: make the raw instance arrays unreachable for unrevealed orders
# =============================================================================================
class GuardedArray(object):
    """An instance-array view that REFUSES node indices outside the visible set.

    Access at depth 1 is guarded and logged; a returned numpy slice is not re-guarded, which is
    recorded in DOMAIN_LIMITS rather than glossed over.
    """

    def __init__(self, array, name, visible, log, instance_axis=True):
        self._a = array
        self._name = name
        self._vis = set(int(x) for x in visible) | {0}
        self._log = log
        self._inst = instance_axis

    def __getitem__(self, key):
        key_t = key if isinstance(key, tuple) else (key,)
        idxs = key_t[1:] if self._inst else key_t
        for k in idxs:
            if isinstance(k, (int,)) or type(k).__name__.startswith('int'):
                if int(k) not in self._vis:
                    self._log.append({'field': self._name, 'index': int(k), 'violation': True})
                    raise Refusal('HIDDEN_FUTURE_ATTRIBUTE_REQUESTED',
                                  {'field': self._name, 'index': int(k),
                                   'visible': sorted(self._vis)})
        self._log.append({'field': self._name, 'key': str(key), 'violation': False})
        return self._a[key]

    def __float__(self):
        return float(self._a)


def guard_env(env, visible, log):
    """Wrap the node-indexed arrays of a real env so unrevealed indices raise."""
    import numpy as np
    for name in ('demands', 'tw_start', 'tw_end', 'service_time', 'reveal_time', 'temp_class',
                 'dist_mat'):
        arr = getattr(env, name, None)
        if arr is not None:
            object.__setattr__(env, name, GuardedArray(arr, name, visible, log))
    ds = getattr(env, 'dataset', None)
    if ds is not None and 'temp_class' in ds:
        ds = dict(ds)
        ds['temp_class'] = GuardedArray(ds['temp_class'], 'dataset.temp_class', visible, log)
        object.__setattr__(env, 'dataset', ds)
    return env


# =============================================================================================
# consumption records (what each party actually used)
# =============================================================================================
def projection_digest(frozen):
    """Digest of the WHOLE projection, so two different states cannot share it."""
    proj = {'derived_from': visible_identity(frozen),
            'content_digest': content_digest(list(frozen.order_ids()))}
    return hashlib.sha256(json.dumps(proj, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def consumption_record(frozen, party, verdict, extra=None, reported_identity=None):
    """What a party actually consumed, plus what it CLAIMS it consumed.

    A party can report one digest while computing over another state, so the record carries BOTH and
    flags the disagreement instead of assuming it away.
    """
    actual = visible_identity(frozen)
    rep = actual if reported_identity is None else str(reported_identity)
    rec = {'party': party, 'verdict': verdict,
           'frozen_identity': actual,
           'reported_identity': rep,
           'reported_identity_bound': (rep == actual),
           'consumed_projection_digest': projection_digest(frozen),
           'visible_ids': list(frozen.visible_ids),
           'instance_id': frozen.instance_id, 'event_id': frozen.event_id,
           'contract_hash': frozen.contract_hash,
           'source_identity': [list(kv) for kv in frozen.source_identity],
           'physical_qualification': False,
           'domain_limits': list(DOMAIN_LIMITS)}
    if extra:
        rec.update(extra)
    return rec


def joint_verdict(records, g2_budget_qualified):
    """Identity agreement never becomes installability."""
    identities = {r['frozen_identity'] for r in records}
    projections = {r['consumed_projection_digest'] for r in records}
    reasons = []
    if len(identities) != 1:
        reasons.append('FROZEN_IDENTITY_DISAGREEMENT')
    if len(projections) != 1:
        reasons.append('CONSUMED_PROJECTION_DISAGREEMENT')
    unbound = [r['party'] for r in records if r.get('reported_identity_bound') is False]
    if unbound:
        reasons.append('REPORTED_IDENTITY_NOT_BOUND')
    if not g2_budget_qualified:
        reasons.append('G2_BUDGET_NOT_QUALIFIED')
    return {'verdict': 'JOINT_NOT_CERTIFIABLE' if reasons else 'JOINT_IDENTITY_AGREEMENT_ONLY',
            'identity_agreement': len(identities) == 1,
            'projection_agreement': len(projections) == 1,
            'reported_identity_unbound_parties': unbound,
            'installable': False if reasons else False,   # never installable in this candidate
            'reasons': reasons,
            'note': 'installability is False in every branch: this candidate produces no physical '
                    'qualification and no install certificate'}
