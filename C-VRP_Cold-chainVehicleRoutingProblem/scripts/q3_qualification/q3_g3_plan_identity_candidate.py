#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-3 ISOLATED CANDIDATE  --  `G3_PLAN_IDENTITY_V1`: a deep-frozen plan identity with a guarded
certify -> install boundary.

WHAT THIS EXISTS FOR
====================
The current boundary in `d3_geo_adapter.py:666-696` computes the identity FIRST, then hands the
ORIGINAL mutable plan to `install_fn` (:689), and only copies it afterwards (:691).  An independent
review showed that a callback can therefore change `anchor_time` / `anchor_load` during install while
the ledger still records the pre-callback hash, and that no hash strength fixes that: the problem is
that the object CHECKED and the object USED are not required to be the same object.

This module is the isolated candidate that closes it, in the shape the review demanded:

    L-1  deep-freeze the plan (recursive: frozen dataclasses, tuples, no dict/list/set anywhere)
         and re-hash THAT object
    L-2  hand the callback a read-only view of the frozen plan, and re-hash before AND after
    L-3  archive the identity of the pre-install frozen plan, never a copy of whatever the callback
         still holds
    L-4  the certifier returns the full identity plus per-field evidence for the object it audited
    L-5  cross-check the anchors against the frozen execution state

And -- from amendment 03 -- it must NOT pretend that a post-hoc hash is install atomicity.  Outcomes
are tiered and each tier claims exactly what it can prove:

    REFUSED_BEFORE_INSTALL     callback never called, n_installs = 0
    INSTALL_COMPLETED          callback returned, frozen identity unchanged before/after
    INSTALL_ROLLED_BACK        callback ran, post-check failed, rollback executed AND verified
    INSTALL_OUTCOME_UNVERIFIED callback ran, no verified rollback -- "no install" may NOT be claimed

WHAT THIS IS NOT
================
Not a security sandbox.  A read-only Python wrapper does not stop hostile in-process code (reflection,
ctypes, gc tricks), and nothing here covers threads, external processes, escaped references, or real
executor side effects.  Those are recorded as applicability-domain limits, in the receipt and in the
report, not hidden in a comment.

Not a production change: no helper is imported for mutation, none is replaced, and
`production_consumable` is False.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from types import MappingProxyType

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G3_PLAN_IDENTITY_V1'
SCHEMA = 'q3_g3_plan_identity'

#: outcome tiers -- the ONLY four values `install_with_frozen_identity` may return
REFUSED_BEFORE_INSTALL = 'REFUSED_BEFORE_INSTALL'
INSTALL_COMPLETED = 'INSTALL_COMPLETED'
INSTALL_ROLLED_BACK = 'INSTALL_ROLLED_BACK'
INSTALL_OUTCOME_UNVERIFIED = 'INSTALL_OUTCOME_UNVERIFIED'

NULL = 'null'

#: what the candidate explicitly does NOT cover (carried in the receipt, quoted in the report)
DOMAIN_LIMITS = (
    'in-process read-only wrappers are NOT a security sandbox: hostile code can bypass them via '
    'reflection, ctypes or gc tricks',
    'no coverage of threads or concurrent mutation',
    'no coverage of external processes or out-of-process mutation',
    'a reference captured BEFORE the freeze still points at the caller object; mutating it does not '
    'change the frozen plan, but the candidate does not claim the caller object is protected',
    'real executor side effects (disk, orders, state machines) are not observed; only a verified '
    'rollback callback can narrow that gap',
)


class Refusal(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                      # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


# =============================================================================================
# L-1: the encoders (lossless, and DELIMITED -- the old hash concatenated with no separator)
# =============================================================================================
def enc_bits(x):
    """Bit-exact canonical form of a double; total on -0.0, nan, inf."""
    return struct.pack('>d', float(x)).hex()


def enc_int(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal('NON_NUMERIC_INT', {'got': repr(n)})
    return str(int(n))


def _enc_field(name, payload):
    """Length-prefixed field: the encoding cannot be ambiguous across field boundaries."""
    if not isinstance(payload, str):
        payload = str(payload)
    return '%d:%s%d:%s' % (len(name), name, len(payload), payload)


def enc_time(x, field):
    """A time must be finite, >= 0 and not negative zero (same rule as the G-1 identity)."""
    if x is None:
        raise Refusal('MISSING_TIME', {'field': field})
    if isinstance(x, bool):
        raise Refusal('NON_NUMERIC_TIME', {'field': field, 'got': repr(x)})
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise Refusal('NON_NUMERIC_TIME', {'field': field, 'got': repr(x)[:60]})
    if not math.isfinite(v):
        raise Refusal('NON_FINITE_TIME', {'field': field, 'got': repr(v)})
    if v == 0.0 and math.copysign(1.0, v) < 0:
        raise Refusal('NEGATIVE_ZERO_TIME', {'field': field})
    if v < 0.0:
        raise Refusal('NEGATIVE_TIME', {'field': field, 'got': v})
    return enc_bits(v)


def source_dtype(x):
    dt = getattr(x, 'dtype', None)
    return str(dt) if dt is not None else type(x).__name__


# =============================================================================================
# L-1: deep freeze
# =============================================================================================
@dataclasses.dataclass(frozen=True)
class FrozenVehiclePlan(object):
    """A vehicle plan that cannot be mutated through normal attribute assignment."""

    vehicle_id: int
    anchor_node: int
    anchor_time: float
    anchor_load: float
    suffix: tuple
    dtypes: tuple


@dataclasses.dataclass(frozen=True)
class FrozenFleet(object):
    """The frozen fleet: an ordered tuple of frozen plans plus a read-only mapping view."""

    plans: tuple

    def items(self):
        for p in self.plans:
            yield p.vehicle_id, p

    def by_id(self):
        return {p.vehicle_id: p for p in self.plans}


def deep_freeze_plan(plan):
    """Freeze a plan mapping RECURSIVELY, copying every value so no mutable reference is shared."""
    if not isinstance(plan, Mapping):
        raise Refusal('PLAN_NOT_A_MAPPING', {'type': type(plan).__name__})
    frozen = []
    seen = set()
    for vid in sorted(int(v) for v in plan):
        if vid in seen:
            raise Refusal('DUPLICATE_VEHICLE_KEY', {'vehicle_id': vid})
        seen.add(vid)
        p = None
        for k, v in plan.items():
            if int(k) == vid:
                p = v
                break
        if p is None:                                                      # pragma: no cover
            raise Refusal('PLAN_LOOKUP_FAILED', {'vehicle_id': vid})
        suffix = tuple(int(x) for x in (getattr(p, 'suffix', ()) or ()))
        if any(int(x) == 0 for x in suffix):
            raise Refusal('SUFFIX_CONTRACT_VIOLATION', {
                'vehicle_id': vid, 'suffix': list(suffix),
                'why': 'a production suffix never contains 0 (build_vehicle_plans filters them), so a '
                       '0 anywhere means the input is not production-shaped'})
        frozen.append(FrozenVehiclePlan(
            vehicle_id=vid,
            anchor_node=int(getattr(p, 'anchor_node', 0)),
            anchor_time=float(getattr(p, 'anchor_time', 0.0)),
            anchor_load=float(getattr(p, 'anchor_load', 0.0)),
            suffix=suffix,
            dtypes=(
                ('anchor_node', source_dtype(getattr(p, 'anchor_node', 0))),
                ('anchor_time', source_dtype(getattr(p, 'anchor_time', 0.0))),
                ('anchor_load', source_dtype(getattr(p, 'anchor_load', 0.0))),
                ('suffix', source_dtype(getattr(p, 'suffix', ()))),
            ),
        ))
    fleet = FrozenFleet(plans=tuple(frozen))
    assert_deep_frozen(fleet)
    return fleet


def assert_deep_frozen(obj, _path='', _seen=None):
    """Recursive proof that nothing mutable survived the freeze.

    A `MappingProxyType` over a dict of live `VehiclePlan`s is NOT frozen: the inner objects are
    still assignable.  This walk exists so that such a structure is REFUSED instead of trusted.
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:                                                       # pragma: no cover
        return True
    problems = []

    def walk(node, path):
        if isinstance(node, (list, set, bytearray)):
            problems.append({'path': path, 'type': type(node).__name__,
                             'why': 'mutable container'})
            return
        if isinstance(node, dict):
            problems.append({'path': path, 'type': 'dict', 'why': 'mutable mapping'})
            for k, v in node.items():
                walk(v, '%s[%r]' % (path, k))
            return
        if isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, '%s[%r]' % (path, k))
            return
        if isinstance(node, tuple):
            for i, v in enumerate(node):
                walk(v, '%s[%d]' % (path, i))
            return
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            params = getattr(type(node), '__dataclass_params__', None)
            if params is None or not params.frozen:
                problems.append({'path': path, 'type': type(node).__name__,
                                 'why': 'dataclass is not frozen'})
                return
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name), '%s.%s' % (path, f.name))
            return
        if isinstance(node, (int, float, str, bool, type(None))):
            return
        problems.append({'path': path, 'type': type(node).__name__, 'why': 'unclassified object'})

    walk(obj, _path or '<root>')
    if problems:
        raise Refusal('FREEZE_NOT_DEEP', {'problems': problems[:8],
                                          'why': 'the freeze must be recursive; a shallow read-only '
                                                 'wrapper leaves the inner plans mutable'})
    return True


def read_only_view(frozen):
    """The mapping handed to the install callback: no setter, and the values are frozen plans."""
    assert_deep_frozen(frozen)
    return MappingProxyType({p.vehicle_id: p for p in frozen.plans})


# =============================================================================================
# L-4: the identity, with per-field evidence
# =============================================================================================
def encode_plan(frozen):
    """Canonical, DELIMITED encoding of the whole fleet.  Returns bytes."""
    assert_deep_frozen(frozen)
    parts = [_enc_field('schema', SCHEMA), _enc_field('version', IMPLEMENTATION_VERSION),
             _enc_field('n_vehicles', enc_int(len(frozen.plans)))]
    for p in frozen.plans:
        parts.append(_enc_field('vehicle_id', enc_int(p.vehicle_id)))
        parts.append(_enc_field('anchor_node', enc_int(p.anchor_node)))
        parts.append(_enc_field('anchor_time', enc_time(p.anchor_time, 'anchor_time')))
        parts.append(_enc_field('anchor_load', enc_time(p.anchor_load, 'anchor_load')))
        parts.append(_enc_field('anchor_time_dtype', p.dtypes[1][1]))
        parts.append(_enc_field('anchor_load_dtype', p.dtypes[2][1]))
        parts.append(_enc_field('n_suffix', enc_int(len(p.suffix))))
        for c in p.suffix:
            parts.append(_enc_field('suffix_item', enc_int(c)))
    return ''.join(parts).encode('utf-8')


def plan_identity(frozen):
    """Full 64-hex identity.  A truncated prefix is never the identity."""
    return hashlib.sha256(encode_plan(frozen)).hexdigest()


def per_field_evidence(frozen):
    """Every bound field as encoded bits, so a verifier can compare field-by-field (L-4)."""
    assert_deep_frozen(frozen)
    return {
        'schema': SCHEMA,
        'vehicles': [
            {'vehicle_id': enc_int(p.vehicle_id),
             'anchor_node': enc_int(p.anchor_node),
             'anchor_time': enc_time(p.anchor_time, 'anchor_time'),
             'anchor_load': enc_time(p.anchor_load, 'anchor_load'),
             'anchor_time_dtype': p.dtypes[1][1],
             'anchor_load_dtype': p.dtypes[2][1],
             'suffix': [enc_int(c) for c in p.suffix]}
            for p in frozen.plans],
        'n_vehicles': enc_int(len(frozen.plans)),
    }


def certify_frozen_plan(frozen):
    """The certifier's output: identity of the object it ACTUALLY audited, plus field evidence."""
    assert_deep_frozen(frozen)
    ident = plan_identity(frozen)
    return {'schema': SCHEMA + '_certificate', 'identity': ident, 'length_hex': len(ident),
            'per_field': per_field_evidence(frozen),
            'audited_object_class': type(frozen).__name__,
            'deep_freeze_verified': True,
            'not_for_production': True, 'production_consumable': False}


# =============================================================================================
# L-5: anchor cross-check against the frozen execution state
# =============================================================================================
def anchor_cross_check(frozen, frozen_state):
    """The anchors must be DERIVABLE from the frozen execution state, not from caller belief."""
    assert_deep_frozen(frozen)
    if not frozen_state:
        raise Refusal('FROZEN_STATE_MISSING', {'why': 'L-5 requires the frozen execution state'})
    problems = []
    for p in frozen.plans:
        st = frozen_state.get(p.vehicle_id)
        if st is None:
            problems.append({'vehicle_id': p.vehicle_id, 'why': 'vehicle absent from frozen state'})
            continue
        node = int(st.get('current_node', -1))
        committed = st.get('committed_next')
        if committed not in (None, 0):
            node = int(committed)
            expect_time = st.get('committed_finish')
            expect_load = st.get('load')  # + demand applied by the caller; not re-derived here
        else:
            expect_time = st.get('ready_time')
            expect_load = st.get('load')
        if p.anchor_node != node:
            problems.append({'vehicle_id': p.vehicle_id, 'field': 'anchor_node',
                             'plan': p.anchor_node, 'frozen_state': node})
        if expect_time is not None and abs(p.anchor_time - float(expect_time)) > 1e-12:
            problems.append({'vehicle_id': p.vehicle_id, 'field': 'anchor_time',
                             'plan': p.anchor_time, 'frozen_state': float(expect_time)})
        if expect_load is not None:
            plan_load = float(p.anchor_load)
            state_load = float(expect_load)
            if abs(plan_load - state_load) > 1e-9:
                problems.append({'vehicle_id': p.vehicle_id, 'field': 'anchor_load',
                                 'plan': plan_load, 'frozen_state': state_load})
    if problems:
        raise Refusal('ANCHOR_NOT_FROM_FROZEN_STATE', {'problems': problems[:8]})
    return True


# =============================================================================================
# the guarded install boundary
# =============================================================================================
def install_with_frozen_identity(plan, *, certificate, install_fn, rollback_fn=None,
                                 frozen_state=None, require_frozen_state=False):
    """Install exactly the object that was certified, and claim only what can be proven.

    Order (deliberately the reverse of d3_geo_adapter.py:666-696, which copies AFTER the callback):
        freeze -> hash -> compare with the certificate -> cross-check anchors
        -> hand a read-only view to the callback -> re-hash the FROZEN object -> archive that
    """
    receipt = {
        'schema': SCHEMA + '_install', 'EXPERIMENTAL_NOT_FOR_PRODUCTION': True,
        'production_consumable': False, 'implementation_version': IMPLEMENTATION_VERSION,
        'callback_called': False, 'n_installs': 0, 'pre_identity': None, 'post_identity': None,
        'certified_identity': (certificate or {}).get('identity'),
        'refusals': [], 'rollback_attempted': False, 'rollback_verified': False,
        'side_effects_possible': False, 'archived_plan': None, 'archived_identity': None,
        'domain_limits': list(DOMAIN_LIMITS),
    }

    # ---- 1. freeze and hash the object that WILL be used (L-1) -----------------------------
    try:
        frozen = deep_freeze_plan(plan)
        pre = plan_identity(frozen)
        pre_fields = per_field_evidence(frozen)
    except Refusal as exc:
        receipt['refusals'].append(exc.as_dict())
        receipt['outcome'] = REFUSED_BEFORE_INSTALL
        receipt['proves'] = 'the install callback was never called and nothing was installed'
        return receipt
    receipt['pre_identity'] = pre

    # ---- 2. the certificate must describe THIS object (L-4 comparison) ----------------------
    if not certificate or not certificate.get('identity'):
        receipt['refusals'].append({'code': 'CERTIFIED_IDENTITY_MISSING'})
    elif certificate['identity'] != pre:
        receipt['refusals'].append({'code': 'CERTIFIED_INSTALLED_MISMATCH',
                                    'detail': {'certified': certificate['identity'],
                                               'frozen': pre}})
    elif certificate.get('per_field') != pre_fields:
        receipt['refusals'].append({'code': 'CERTIFIED_FIELD_EVIDENCE_MISMATCH'})

    # ---- 3. anchors must come from the frozen state (L-5) ----------------------------------
    if require_frozen_state or frozen_state is not None:
        try:
            anchor_cross_check(frozen, frozen_state)
        except Refusal as exc:
            receipt['refusals'].append(exc.as_dict())

    if receipt['refusals']:
        receipt['outcome'] = REFUSED_BEFORE_INSTALL
        receipt['proves'] = ('the install callback was never called (callback_called=False, '
                             'n_installs=0) and nothing was installed')
        return receipt

    # ---- 4. run the callback on a read-only view of the FROZEN object (L-2) ----------------
    view = read_only_view(frozen)
    error = None
    try:
        install_fn(view)
        receipt['callback_called'] = True
    except BaseException as exc:                                           # noqa: BLE001
        receipt['callback_called'] = True
        error = '%s: %s' % (type(exc).__name__, str(exc)[:120])
    receipt['side_effects_possible'] = receipt['callback_called']

    # ---- 5. re-hash the FROZEN object: the only thing the callback could have touched -----
    try:
        post = plan_identity(frozen)
    except Refusal as exc:
        post = None
        receipt['refusals'].append(exc.as_dict())
    receipt['post_identity'] = post
    # A re-hash that could not complete is treated as "not identical": if the frozen object can no
    # longer be encoded (e.g. the callback wrote nan into it), that is a mutation, not a pass.
    mutated = (post is None or post != pre)

    # ---- 6. outcome tier ----------------------------------------------------------------
    if not mutated and error is None:
        receipt['outcome'] = INSTALL_COMPLETED
        receipt['n_installs'] = 1
        # L-3: archive the FROZEN plan and the PRE identity, never a fresh copy of anything else
        receipt['archived_plan'] = frozen
        receipt['archived_identity'] = pre
        receipt['proves'] = ('the frozen plan was identical before and after the callback, and the '
                             'archive is bound to that identity')
        return receipt

    if mutated:
        receipt['refusals'].append({'code': 'PLAN_MUTATED_DURING_INSTALL',
                                    'detail': {'pre': pre, 'post': post}})
    if error is not None:
        receipt['refusals'].append({'code': 'INSTALL_CALLBACK_RAISED', 'detail': {'error': error}})

    # a rollback is only "verified" when it ran and reported success
    if rollback_fn is not None:
        receipt['rollback_attempted'] = True
        try:
            receipt['rollback_verified'] = bool(rollback_fn(frozen))
        except BaseException as exc:                                       # noqa: BLE001
            receipt['rollback_verified'] = False
            receipt['refusals'].append({'code': 'ROLLBACK_RAISED',
                                        'detail': {'error': '%s' % type(exc).__name__}})

    if receipt['rollback_verified']:
        receipt['outcome'] = INSTALL_ROLLED_BACK
        receipt['n_installs'] = 0
        receipt['proves'] = ('the callback ran and the rollback ran and reported success; the state '
                             'AFTER the rollback is not independently verified by this module')
    else:
        receipt['outcome'] = INSTALL_OUTCOME_UNVERIFIED
        receipt['n_installs'] = 0
        receipt['proves'] = ('only that the callback was called; it does NOT prove that nothing was '
                             'installed, and it does NOT prove a rollback happened')
    return receipt
