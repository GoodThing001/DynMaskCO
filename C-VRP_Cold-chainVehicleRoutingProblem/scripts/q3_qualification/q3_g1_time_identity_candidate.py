#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-1 ISOLATED CANDIDATE  --  `G1_TIME_IDENTITY_V1`: a lossless, deterministic time identity.

WHAT THIS REPLACES
==================
`m1b_immutable_snapshot.snapshot_state_identity` canonicalises four fields with `round(x, 9)`.  That
is a FILTER, not a canonical form: two distinct doubles around 24.0 (1e-12 apart) hash to the same
string, and the collapse window is about **281474 ulp** (measured, probe P-02).  The identity is
therefore blind to exactly the drift it exists to detect.

This module is an ISOLATED candidate.  It does not modify the production module, it is not wired into
the certifier or the install entry, and `production_consumable` is False.  Replacing the production
function needs its own pre-registration and its own authorisation.

ENCODING RULES (the whole point)
================================
  R-1  every time / load is encoded as `struct.pack('>d', x).hex()` -- big-endian IEEE-754 bits.
       `float.hex()` is NOT used for the digest (it would be a second source of truth).
  R-2  a time must be finite, >= 0 and not negative zero.  Violations are REFUSED, never encoded
       and explained later.
  R-3  an absent value (`committed_finish is None`) is the sentinel `'null'`, which differs from the
       bit pattern of 0.0.
  R-4  no `round`, no decimal formatting, no `Decimal` quantisation as a canonical form; the module
       scans its own source and refuses if it finds one (`LOSSY_CANONICALISER`).
  R-5  integers are decimal strings (they never pass through a float); booleans are 'true'/'false'.
  R-6  the SOURCE DTYPE is recorded next to every value, so a silent float32 -> float64 promotion is
       visible instead of being absorbed.

WHAT IT REFUSES TO DO
====================
It does not endorse a G-2 bound.  If the G-2 time budget is not qualified (unknown distance model, no
clock bound, non-unit scale, no monitor receipt), the identity can still be *consistent* -- and it
will say so -- but `certifiable` stays False.  Identity consistency is never used as a substitute for
budget qualification.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
from pathlib import Path

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G1_TIME_IDENTITY_V1'
SCHEMA = 'q3_g1_time_identity'

NULL = 'null'
LOSSY_TOKENS = ('round(', 'format(', '%.9')

#: declared classification of every verdict this module can emit
GRADING = {
    'PROVED_CONDITIONAL': 'proved from the source + a witness that is itself conditional',
    'DIAGNOSTIC': 'observed on synthetic data only',
    'UNKNOWN': 'not established',
}


class Refusal(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}

    def as_dict(self):
        return {'code': self.code, 'detail': self.detail}


# =============================================================================================
# R-1 / R-3 / R-5 / R-6: the encoders
# =============================================================================================
def enc_bits(x):
    """Bit-exact canonical form of a double.  Total: also encodes -0.0, nan and inf."""
    return struct.pack('>d', float(x)).hex()


def enc_int(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise Refusal('NON_NUMERIC_INT', {'got': repr(n)})
    return str(int(n))


def enc_bool(b):
    return 'true' if bool(b) else 'false'


def source_dtype(x):
    """The dtype the value arrived with, so an upcast cannot pass unnoticed (R-6)."""
    dt = getattr(x, 'dtype', None)
    if dt is not None:
        return str(dt)
    return type(x).__name__


def _to_float(x, field):
    if isinstance(x, bool):
        raise Refusal('NON_NUMERIC_TIME', {'field': field, 'got': repr(x)})
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise Refusal('NON_NUMERIC_TIME', {'field': field, 'got': repr(x)[:60]})
    if not math.isfinite(v):
        raise Refusal('NON_FINITE_TIME', {'field': field, 'got': repr(v)})
    if v == 0.0 and math.copysign(1.0, v) < 0:
        raise Refusal('NEGATIVE_ZERO_TIME', {'field': field,
                                             'why': '-0.0 == 0.0 numerically but has a different bit '
                                                    'pattern; an ambiguous timestamp is refused'})
    if v < 0.0:
        raise Refusal('NEGATIVE_TIME', {'field': field, 'got': v})
    return v


def enc_time(x, field, allow_null=False):
    """R-2 + R-3 + R-6: validate, then encode bits plus dtype."""
    if x is None:
        if allow_null:
            return {'value': NULL, 'dtype': 'absent'}
        raise Refusal('MISSING_TIME', {'field': field})
    v = _to_float(x, field)
    return {'value': enc_bits(v), 'dtype': source_dtype(x)}


def require_unit(hours_per_time_unit):
    """The unit scale must be a real positive number; only exactly 1.0 is certifiable."""
    if isinstance(hours_per_time_unit, bool) or hours_per_time_unit is None:
        raise Refusal('ILLEGAL_UNIT', {'hours_per_time_unit': repr(hours_per_time_unit)})
    try:
        hpu = float(hours_per_time_unit)
    except (TypeError, ValueError):
        raise Refusal('ILLEGAL_UNIT', {'hours_per_time_unit': repr(hours_per_time_unit)[:60]})
    if not math.isfinite(hpu) or hpu <= 0:
        raise Refusal('ILLEGAL_UNIT', {'hours_per_time_unit': hpu})
    return {'hours_per_time_unit': hpu, 'unit_certifiable': (hpu == 1.0),
            'note': ('a non-unit scale adds one rounding per advance that the L1 slack budget does '
                     'not cover, so the identity can be consistent but not certifiable')}


def canon(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha256_json(payload):
    return hashlib.sha256(canon(payload).encode('utf-8')).hexdigest()


# =============================================================================================
# R-4: the self-check
# =============================================================================================
def lossy_canonicaliser_scan(source_path=None):
    """Refuse to be a lossy canonicaliser: scan this module's own AST for `round` / formatting."""
    path = Path(source_path) if source_path else Path(__file__).resolve()
    tree = ast.parse(path.read_text(encoding='utf-8'))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == 'round':
                hits.append({'line': node.lineno, 'what': 'round(...)'})
            if isinstance(f, ast.Attribute) and f.attr in ('format', 'quantize'):
                hits.append({'line': node.lineno, 'what': '.' + f.attr + '(...)'})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) \
                and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) \
                and ('.9' in node.left.value or '.%' in node.left.value):
            hits.append({'line': node.lineno, 'what': 'percent-format with a precision'})
    if hits:
        raise Refusal('LOSSY_CANONICALISER', {'hits': hits,
                                              'why': 'R-4 forbids a lossy canonical form anywhere in '
                                                     'this module'})
    return {'clean': True, 'scanned': path.name, 'forbidden_tokens': list(LOSSY_TOKENS)}


# =============================================================================================
# the identity object
# =============================================================================================
def build_identity(*, clock, anchor, vehicles=None, contract_hash, hours_per_time_unit,
                   distance_km_per_unit=1.0, instance_id, event_id, served_prefix=(),
                   committed_next=(), T_clock_sup, n_max_per_vehicle,
                   travel_domain, source=None, census_digest=None, domain=None,
                   g2_qualification=None, monitor_receipt=None, declared_resume=False,
                   segment_durations=()):
    """Build the identity.  Every check is explicit; nothing is silently absorbed.

    `identity_consistent` says the internal bindings are complete and validated.
    `certifiable` additionally requires the G-2 budget to be QUALIFIED, the unit to be 1.0 and the
    monitor receipt to be clean -- because a consistent identity over an unqualified bound would
    launder the bound.
    """
    conditional, refusals = [], []

    if not isinstance(contract_hash, str) or not contract_hash:
        raise Refusal('CONTRACT_IDENTITY_UNBOUND', {'contract_hash': repr(contract_hash)[:60]})
    unit = require_unit(hours_per_time_unit)
    if not unit['unit_certifiable']:
        conditional.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                            'detail': {'hours_per_time_unit': unit['hours_per_time_unit']}})
    if not math.isfinite(float(distance_km_per_unit)) or float(distance_km_per_unit) <= 0:
        raise Refusal('ILLEGAL_UNIT', {'distance_km_per_unit': repr(distance_km_per_unit)})

    if T_clock_sup is None or n_max_per_vehicle is None:
        raise Refusal('TRUSTED_TIME_BOUND_MISSING', {
            'T_clock_sup': repr(T_clock_sup), 'n_max_per_vehicle': repr(n_max_per_vehicle),
            'why': 'without a clock bound and an advance bound the identity has nothing to bind; it '
                   'must not fall back to the horizon or to max_iter'})
    tsup = _to_float(T_clock_sup, 'T_clock_sup')
    if tsup <= 0:
        raise Refusal('TRUSTED_TIME_BOUND_MISSING', {'T_clock_sup': tsup, 'why': 'non-positive'})
    if isinstance(n_max_per_vehicle, bool) or not isinstance(n_max_per_vehicle, int) \
            or n_max_per_vehicle <= 0:
        raise Refusal('TRUSTED_TIME_BOUND_MISSING',
                      {'n_max_per_vehicle': repr(n_max_per_vehicle)})

    if not isinstance(travel_domain, dict) or not travel_domain.get('models'):
        raise Refusal('DISTANCE_DOMAIN_MISSING', {
            'travel_domain': repr(travel_domain)[:120],
            'why': 'the travel bound only exists inside a declared distance model; sqrt(2) alone is '
                   'not a repo-wide upper bound (G-2 errata D-2/D-3)'})
    speed = travel_domain.get('tw_speed')
    if speed is None or isinstance(speed, bool):
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': repr(speed)})
    try:
        sp = float(speed)
    except (TypeError, ValueError):
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': repr(speed)[:60]})
    if not math.isfinite(sp) or sp <= 0:
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': sp})

    if declared_resume:
        refusals.append({'code': 'SNAPSHOT_RESTORE_INSIDE_INTERVAL',
                         'detail': {'why': 'a restore path can inject the cold-chain anchor from '
                                           'outside the event loop'}})
    if refusals:
        raise Refusal(refusals[0]['code'], refusals[0]['detail'])

    # ---- the bits block -------------------------------------------------------------------
    bits = {
        'clock': enc_time(clock, 'clock'),
        'anchor': enc_time(anchor, 'anchor', allow_null=True),
        'segment_durations': [enc_time(d, 'segment_duration') for d in segment_durations],
        'vehicles': [],
    }
    for v in (vehicles or ()):
        bits['vehicles'].append({
            'vehicle_id': enc_int(int(v['vehicle_id'])),
            'ready_time': enc_time(v.get('ready_time'), 'ready_time'),
            'current_load': enc_time(v.get('current_load'), 'current_load', allow_null=True),
            'committed_finish': enc_time(v.get('committed_finish'), 'committed_finish',
                                         allow_null=True),
            'return_finish': enc_time(v.get('return_finish'), 'return_finish', allow_null=True),
            'status': str(v.get('status')),
            'current_node': enc_int(int(v.get('current_node', 0))),
        })

    # ---- the G-2 budget binding ------------------------------------------------------------
    g2 = dict(g2_qualification or {})
    g2_qualified = bool(g2.get('budget_qualified')) and str(g2.get('verdict')) in (
        'PROVED_CONDITIONAL', 'PROVED')
    if not g2_qualified:
        conditional.append({'code': 'G2_BUDGET_NOT_QUALIFIED',
                            'detail': {'g2_verdict': g2.get('verdict'),
                                       'why': 'identity consistency must never be used to endorse a '
                                              'time bound that G-2 did not qualify'}})
    monitor_ok = bool((monitor_receipt or {}).get('certifiable'))
    if not monitor_ok:
        conditional.append({'code': 'CALLBACK_ADVANCES_UNWITNESSED',
                            'detail': {'why': 'no clean monitor receipt covers the OPEN callbacks'}})

    payload = {
        'schema': SCHEMA, 'version': IMPLEMENTATION_VERSION,
        'not_for_production': True, 'production_consumable': PRODUCTION_CONSUMABLE,
        'bits': bits,
        'execution': {
            'instance_id': enc_int(int(instance_id)),
            'event_id': enc_int(int(event_id)),
            'served_prefix': [enc_int(int(c)) for c in served_prefix],
            'committed_next': [enc_int(int(c)) for c in committed_next],
        },
        'binding': {
            'contract_hash': contract_hash,
            'hours_per_time_unit': enc_bits(unit['hours_per_time_unit']),
            'distance_km_per_unit': enc_bits(float(distance_km_per_unit)),
            'T_clock_sup': enc_bits(tsup),
            'n_max_per_vehicle': enc_int(int(n_max_per_vehicle)),
            'travel_domain': {'models': list(travel_domain.get('models') or ()),
                              'multiplier_ceiling': enc_bits(
                                  float(travel_domain.get('multiplier_ceiling', 1.0))),
                              'tw_speed': enc_bits(sp)},
            'source': dict(source or {}),
            'census_digest': census_digest,
        },
        'domain': dict(domain or {}),
        'g2_qualification': g2,
        'monitor': {'receipt_digest': (monitor_receipt or {}).get('receipt_digest'),
                    'certifiable': monitor_ok},
    }
    identity_consistent = True
    certifiable = (identity_consistent and unit['unit_certifiable'] and g2_qualified and monitor_ok)
    payload['identity_consistent'] = identity_consistent
    payload['certifiable'] = certifiable
    payload['conditional'] = conditional
    payload['identity_digest'] = sha256_json(payload)
    return payload


def require_certifiable(identity):
    """Separate the two questions: is the identity consistent, and is it allowed to certify?"""
    if identity.get('identity_consistent') is not True:
        raise Refusal('IDENTITY_NOT_CONSISTENT', {'digest': identity.get('identity_digest')})
    codes = [c['code'] for c in identity.get('conditional') or ()]
    if 'G2_BUDGET_NOT_QUALIFIED' in codes:
        raise Refusal('G2_BUDGET_NOT_QUALIFIED', {'identity_digest': identity.get('identity_digest')})
    if 'CALLBACK_ADVANCES_UNWITNESSED' in codes:
        raise Refusal('CALLBACK_ADVANCES_UNWITNESSED',
                      {'identity_digest': identity.get('identity_digest')})
    if 'TIME_UNIT_CONVERSION_NOT_BUDGETED' in codes:
        raise Refusal('TIME_UNIT_CONVERSION_NOT_BUDGETED', {})
    return True


# =============================================================================================
# the snapshot identity (the isolated replacement for snapshot_state_identity)
# =============================================================================================
def state_identity_v2(snap):
    """Bit-exact identity over the snapshot's physical state.

    Same coverage as the production function, but every time/load field is encoded losslessly, the
    dtype is recorded, `None` is a sentinel, and negative zero is refused instead of normalised.
    """
    rows = []
    for vid in sorted(snap['vehicles']):
        v = snap['vehicles'][vid]
        rows.append({
            'vehicle_id': enc_int(int(vid)),
            'status': str(getattr(v, 'status', None)),
            'current_node': enc_int(int(getattr(v, 'current_node', 0) or 0)),
            'ready_time': enc_time(getattr(v, 'ready_time', None), 'ready_time'),
            'current_load': enc_time(getattr(v, 'current_load', None), 'current_load',
                                     allow_null=True),
            'committed_next': (NULL if getattr(v, 'committed_next', None) is None
                               else enc_int(int(v.committed_next))),
            'committed_finish': enc_time(getattr(v, 'committed_finish', None), 'committed_finish',
                                         allow_null=True),
            'return_finish': enc_time(getattr(v, 'return_finish', None), 'return_finish',
                                      allow_null=True),
            'mutable_suffix': [enc_int(int(c)) for c in (getattr(v, 'mutable_suffix', None) or ())],
            'cc_identity': getattr(v, 'cc_identity', None),
        })
    payload = {
        'vehicles': rows,
        'served_mask_true': [int(i) for i in range(len(snap['served_mask']))
                             if bool(snap['served_mask'][i])],
        'plan_identity': snap.get('plan_identity'),
        'version': IMPLEMENTATION_VERSION,
    }
    return sha256_json(payload)


def verify_state_unchanged_v2(before_snap, after_snap):
    """Post-certification drift check.  Any change is a refusal, with the reason named."""
    b = state_identity_v2(before_snap)
    a = state_identity_v2(after_snap)
    if a == b:
        return {'verdict': 'STATE_IDENTITY_STABLE', 'before': b, 'after': a}
    per_vehicle = []
    for vid in sorted(set(before_snap['vehicles']) | set(after_snap['vehicles'])):
        vb = before_snap['vehicles'].get(vid)
        va = after_snap['vehicles'].get(vid)
        if vb is None or va is None:
            per_vehicle.append({'vehicle': vid, 'why': 'vehicle appeared or vanished'})
            continue
        hb = state_identity_v2({'vehicles': {vid: vb},
                                'served_mask': before_snap['served_mask'],
                                'plan_identity': before_snap.get('plan_identity')})
        ha = state_identity_v2({'vehicles': {vid: va},
                                'served_mask': after_snap['served_mask'],
                                'plan_identity': after_snap.get('plan_identity')})
        if hb != ha:
            per_vehicle.append({'vehicle': vid, 'before': hb, 'after': ha})
    raise Refusal('POST_CERTIFICATION_STATE_CHANGED',
                  {'before': b, 'after': a, 'per_vehicle': per_vehicle})


def require_same_contract(identity, contract_hash):
    """A different contract (e.g. a changed unit scale) may never inherit this identity."""
    if identity['binding']['contract_hash'] != contract_hash:
        raise Refusal('CONTRACT_IDENTITY_MISMATCH',
                      {'identity': identity['binding']['contract_hash'],
                       'asked': contract_hash})
    return True


def require_same_instance_plan(identity, *, instance_id, served_prefix, committed_next):
    """Cross-instance reuse and prefix drift must fail closed."""
    if identity['execution']['instance_id'] != enc_int(int(instance_id)):
        raise Refusal('CROSS_INSTANCE_CONFUSION',
                      {'identity': identity['execution']['instance_id'],
                       'asked': enc_int(int(instance_id))})
    if identity['execution']['served_prefix'] != [enc_int(int(c)) for c in served_prefix]:
        raise Refusal('EXECUTED_PREFIX_CHANGED', {'identity_prefix':
                                                 identity['execution']['served_prefix'][:8]})
    if identity['execution']['committed_next'] != [enc_int(int(c)) for c in committed_next]:
        raise Refusal('EXECUTED_PREFIX_CHANGED', {'committed_next':
                                                 identity['execution']['committed_next'][:8]})
    return True
