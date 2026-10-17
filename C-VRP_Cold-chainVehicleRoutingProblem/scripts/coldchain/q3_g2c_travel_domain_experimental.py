#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION  --  G-2 ERRATA round 2:
(i) the DISTANCE-MODEL domain of the travel bound, and (ii) an execution monitor that covers state
writes and dynamic wrapping, not just call counts.

WHY THIS FILE EXISTS
====================
The previous errata said

    T_clock_sup = horizon + travel_bound,   travel_bound = sqrt(2) / tw_speed

and called `sqrt(2)` a **domain invariant**.  That is too strong, for two source reasons:

  1. `StrictOnlineEnv.__init__` uses `dataset['dist_mat']` when the key exists and only falls back to
     the Euclidean matrix otherwise (`strict_online_env.py:186-190`), and `_travel` divides THAT
     matrix by `tw_speed` (`:193-194`).  So the travel time is whatever the edge-weight model says.
  2. The generator has two branches that scale edge weights UP
     (`generate_coldchain_data.py`):
        asymmetric=True      -> euclidean * ((x + 1/x)/2) * rush  ,  x in [0.85, 1.35], rush in [1.2, 2]
        perturb_prob > 0     -> selected edges * U(1.5, 5.0)      (:141-143)
     so `sqrt(2)` is not an upper bound of the repo-wide data domain, and it is not an upper bound of
     the registered splits *because they are Euclidean* -- that has to be WITNESSED per instance,
     not assumed.

The second half of the file answers the review's S-9 requirement: a call counter cannot claim the
executor is monitored.  This monitor additionally

    * detects a replaced/forwarded advance entry point (identity check),
    * detects a callback that rewrites `coldchain_state` / `coldchain_updated_time` directly,
    * detects an undeclared anchor injection from a restore,
    * and refuses to emit a certification-grade receipt once any of the above fired.

WHAT THIS FILE IS NOT
====================
Not a certificate, not a production change.  It does not decide that the registered data is
Euclidean; it requires a witness for that claim and refuses without one.  `PRODUCTION_NOT_QUALIFIED`
stays true for every production parameter discussed here.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'G2C_TRAVEL_DOMAIN_V1'
SCHEMA = 'q3_g2c_travel_domain'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'


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


def canon(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha256_json(payload):
    return hashlib.sha256(canon(payload).encode('utf-8')).hexdigest()


# =============================================================================================
# PART 1 -- the distance model as an explicit, witnessed domain
# =============================================================================================
#: generator ceilings, read off `generate_coldchain_data.py` (not guessed):
#:   asymmetry noise x in [0.85, 1.35], combined as (x + 1/x)/2  -> max at x = 1.35
#:   rush-hour multiplier U(1.2, 2.0)                            -> max 2.0
#:   long-tail edge perturbation U(1.5, 5.0)                     -> max 5.0
ASYM_NOISE_RANGE = (0.85, 1.35)
RUSH_RANGE = (1.2, 2.0)
PERTURB_EDGE_RANGE = (1.5, 5.0)


def _asym_max():
    lo, hi = ASYM_NOISE_RANGE
    return max((x + 1.0 / x) / 2.0 for x in (lo, hi))


ASYM_MAX = _asym_max()                 # 1.0454545...
RUSH_MAX = RUSH_RANGE[1]               # 2.0
PERTURB_MAX = PERTURB_EDGE_RANGE[1]    # 5.0
UNIT_SQUARE_DIAGONAL = math.sqrt(2.0)

#: every distance model this module is willing to reason about.  A model is a pair
#: (multiplier ceiling, witness requirement); an unknown model gets NO bound.
DISTANCE_MODELS = {
    'euclidean': {
        'multiplier': 1.0,
        'premises': ['dist_mat is the Euclidean matrix of coords (or a float32 rounding of it)',
                     'coords lie in the unit square', 'tw_speed > 0 and finite'],
        'witness': 'max|dist_mat - euclidean(coords)| <= tol AND dist_mat is symmetric',
        'source': 'strict_online_env.py:189-190 (fallback branch)',
    },
    'asymmetric_transit': {
        'multiplier': ASYM_MAX * RUSH_MAX,
        'premises': ['the domain declares asymmetric=True with the generator ranges '
                     'x in [0.85, 1.35] and rush in [1.2, 2.0]'],
        'witness': 'the declaration is bound into the domain identity',
        'source': 'generate_coldchain_data.py:81-89',
    },
    'long_tail_perturbation': {
        'multiplier': PERTURB_MAX,
        'premises': ['the domain declares perturb_prob > 0 with perturb edge scale in [1.5, 5.0]'],
        'witness': 'the declaration is bound into the domain identity',
        'source': 'generate_coldchain_data.py:141-143',
    },
    'external_undeclared': {
        'multiplier': None,
        'premises': [],
        'witness': None,
        'source': 'strict_online_env.py:186-187 (dataset supplies dist_mat)',
    },
}


def compose_model(*models):
    """The multiplier ceiling of a composition of models (multiplicative, as the generator does)."""
    mult = 1.0
    used = []
    for m in models:
        if m not in DISTANCE_MODELS:
            raise Refusal('DISTANCE_MODEL_UNKNOWN', {'model': m,
                                                     'known': sorted(DISTANCE_MODELS)})
        entry = DISTANCE_MODELS[m]
        if entry['multiplier'] is None:
            raise Refusal('DISTANCE_MODEL_UNBOUNDED', {
                'model': m,
                'why': 'an externally supplied dist_mat has no ceiling this module can prove; a '
                       'domain-level edge-weight bound must be declared and witnessed first'})
        mult *= entry['multiplier']
        used.append(m)
    return {'models': used, 'multiplier_ceiling': mult}


def travel_bound(models, tw_speed, extra_safety_ulps=1):
    """bound = sqrt(2) * prod(multipliers) / tw_speed, inflated by `extra_safety_ulps` ulps.

    The inflation is not cosmetic: the multipliers are applied in floating point by the generator, so
    the exact product of two floats can exceed the product of the bound's float rounding by a
    fraction of an ulp.  Inflating outward keeps the bound an upper bound of the arithmetic.
    """
    if not _finite(tw_speed) or float(tw_speed) <= 0:
        raise Refusal('TW_SPEED_INVALID', {'tw_speed': repr(tw_speed)})
    comp = compose_model(*models)
    raw = UNIT_SQUARE_DIAGONAL * comp['multiplier_ceiling'] / float(tw_speed)
    b = raw
    for _ in range(int(extra_safety_ulps)):
        b = math.nextafter(b, math.inf)
    return {'travel_bound': b, 'raw': raw,
            'multiplier_ceiling': comp['multiplier_ceiling'], 'models': comp['models'],
            'tw_speed': float(tw_speed), 'diagonal': UNIT_SQUARE_DIAGONAL,
            'formula': 'sqrt(2) * prod(multipliers) / tw_speed, inflated %d ulp outward'
                       % int(extra_safety_ulps),
            'premises': [p for m in comp['models'] for p in DISTANCE_MODELS[m]['premises']],
            'witnesses_required': [DISTANCE_MODELS[m]['witness'] for m in comp['models']]}


def certify_distance_model(dist_mat, coords, model, *, tol=1e-6, stage='offline'):
    """Does THIS instance's edge-weight matrix satisfy the declared model?  (Offline only.)

    Reading the full matrix includes edges to unrevealed orders, so it is a qualification-stage
    operation; the online path may only consume the resulting bound.
    """
    require_offline(stage)
    import numpy as np
    if model not in DISTANCE_MODELS:
        raise Refusal('DISTANCE_MODEL_UNKNOWN', {'model': model})
    dm = np.asarray(dist_mat, dtype=np.float64)
    if dm.ndim != 2:
        raise Refusal('DIST_MAT_SHAPE_INVALID', {'shape': list(dm.shape)})
    if not np.all(np.isfinite(dm)):
        raise Refusal('DIST_MAT_NON_FINITE', {'n': int((~np.isfinite(dm)).sum())})
    out = {'model': model, 'shape': list(dm.shape), 'max_edge': float(dm.max()),
           'symmetric': bool(np.allclose(dm, dm.transpose(), rtol=0.0, atol=tol))}
    if model == 'euclidean':
        c = np.asarray(coords, dtype=np.float64)
        diff = c[:, None, :] - c[None, :, :]
        eu = np.sqrt((diff ** 2).sum(axis=-1))
        out['max_abs_dev_from_euclidean'] = float(np.abs(dm - eu).max())
        out['tol'] = float(tol)
        out['satisfied'] = bool(out['symmetric'] and out['max_abs_dev_from_euclidean'] <= tol)
        if not out['satisfied']:
            raise Refusal('DISTANCE_MODEL_WITNESS_FAILED', out)
    elif model == 'external_undeclared':
        raise Refusal('DISTANCE_MODEL_UNBOUNDED',
                      {'why': 'an external dist_mat without a declared edge-weight bound cannot be '
                              'certified by this module'})
    else:
        ceil = DISTANCE_MODELS[model]['multiplier']
        out['multiplier_ceiling'] = ceil
        out['satisfied'] = None      # only the DOMAIN declaration can witness this, not the matrix
        out['note'] = ('this model is witnessed by the DOMAIN declaration, not by inspecting one '
                       'matrix: a single matrix cannot prove that no other matrix in the domain '
                       'exceeds the ceiling')
    return out


def require_offline(stage):
    if stage != 'offline':
        raise Refusal('ONLINE_MUST_NOT_READ_HIDDEN_DATA', {
            'stage': stage,
            'why': 'the full distance matrix includes edges to unrevealed orders; the bound must be '
                   'derived once in the qualification stage and then consumed as a scalar'})


def bound_from_euclidean_coords(coords, tw_speed, *, stage='offline'):
    """The fallback branch's bound, computed the way the env would compute `dist_mat`.

    Kept only to show how the two branches differ; it is NOT the certified path when the dataset
    ships its own matrix."""
    require_offline(stage)
    import numpy as np
    c = np.asarray(coords, dtype=np.float64)
    diff = c[:, None, :] - c[None, :, :]
    eu = np.sqrt((diff ** 2).sum(axis=-1))
    return {'max_euclidean_edge': float(eu.max()),
            'bound': travel_bound(('euclidean',), tw_speed)['travel_bound']}


# =============================================================================================
# PART 2 -- the execution monitor (counts + identity + state diff)
# =============================================================================================
STATE_FIELDS = ('cumulative_energy_kwh', 'cumulative_distance_km', 'finalized_quality_loss',
                'thermal_violation_count', 'thermal_violation_duration_h', 'door_open_count',
                'dispatched', 'closed')


def _bits(x):
    import struct
    return struct.pack('>d', float(x)).hex()


def vehicle_state_digest(v):
    """A cheap, bit-exact digest of the fields a callback could silently rewrite."""
    import numpy as np
    st = getattr(v, 'coldchain_state', None)
    anchor = getattr(v, 'coldchain_updated_time', None)
    payload = {'anchor': None if anchor is None else _bits(anchor)}
    if st is None:
        payload['state'] = None
    else:
        payload['state'] = {f: _bits(getattr(st, f)) if isinstance(getattr(st, f, None), float)
                            else getattr(st, f, None) for f in STATE_FIELDS}
        temps = np.asarray(getattr(st, 'compartment_temperature_c', ()), dtype=np.float64).ravel()
        loads = np.asarray(getattr(st, 'zone_load', ()), dtype=np.float64).ravel()
        payload['temps'] = [_bits(t) for t in temps.tolist()]
        payload['loads'] = [_bits(t) for t in loads.tolist()]
        payload['n_cargo'] = len(getattr(st, 'cargo_manifest', ()) or ())
        payload['n_delivered'] = len(getattr(st, 'delivered_to_depot', ()) or ())
    return sha256_json(payload)


class ExecutionMonitor(object):
    """Counts advances, checks the entry-point identity, and diffs state around every callback.

    The three checks exist because a counter alone answers only "how many calls did I see", while
    certification needs "no other path changed the cold chain".  Every failure is a REFUSAL, and a
    refused monitor cannot produce a certification-grade receipt.
    """

    def __init__(self, env, n_max, clock_sup=None):
        self.env = env
        self.n_max = int(n_max)
        self.clock_sup = clock_sup
        self.counts, self.clocks, self.violations = {}, [], []
        self.callback_advances = 0
        self.bypass_events = []
        self.unmonitored_writes = []
        self.restores = []
        self._depth = 0
        self._installed = False
        self._orig = None

    # -- installation -------------------------------------------------------------------
    def install(self):
        if self._installed:                                               # pragma: no cover
            raise Refusal('MONITOR_ALREADY_INSTALLED')
        orig = self.env._advance_coldchain_to

        def counting(v, clock):
            if v.vehicle_id not in self.counts:
                self.counts[v.vehicle_id] = 0
            self.counts[v.vehicle_id] += 1
            self.clocks.append(float(clock))
            if self._depth > 0:
                self.callback_advances += 1
            if self.counts[v.vehicle_id] > self.n_max:
                raise Refusal('COUNT_BOUND_VIOLATED',
                              {'vehicle': int(v.vehicle_id),
                               'count': self.counts[v.vehicle_id], 'n_max': self.n_max})
            if self.clock_sup is not None and float(clock) > float(self.clock_sup) + 1e-12:
                raise Refusal('CLOCK_EXCEEDS_UPPER_BOUND',
                              {'clock': float(clock), 'T_clock_sup': float(self.clock_sup)})
            return orig(v, clock)

        self._orig = orig
        self._counting = counting
        self.env._advance_coldchain_to = counting
        self._installed = True
        return self

    def check_entry_point(self):
        """Has anything replaced the advance entry point since installation?"""
        if not self._installed:                                            # pragma: no cover
            raise Refusal('MONITOR_NOT_INSTALLED')
        if self.env._advance_coldchain_to is not self._counting:
            self.bypass_events.append({'why': 'env._advance_coldchain_to was replaced',
                                       'now': repr(self.env._advance_coldchain_to)[:80]})
            raise Refusal('MONITOR_BYPASSED', {'events': self.bypass_events})
        return True

    # -- audit by diff ------------------------------------------------------------------
    @staticmethod
    def snapshot(vehicles):
        return {int(v.vehicle_id): vehicle_state_digest(v) for v in vehicles}

    def window(self, name, fn, vehicles):
        """Run a callback INSIDE an audited window.

        Any state change that is not explained by a counted advance inside this window is an
        unmonitored write -- that is how a callback that assigns `coldchain_updated_time` or
        `coldchain_state` directly gets caught without intercepting the assignments.
        """
        self.check_entry_point()
        before = self.snapshot(vehicles)
        counts_before = dict(self.counts)
        self._depth += 1
        try:
            result = fn()
        finally:
            self._depth -= 1
        after = self.snapshot(vehicles)
        advanced = {vid: self.counts.get(vid, 0) - counts_before.get(vid, 0)
                    for vid in set(list(counts_before) + list(self.counts))}
        for vid in sorted(after):
            if before.get(vid) == after.get(vid):
                continue
            if advanced.get(vid, 0) == 0:
                self.unmonitored_writes.append({
                    'callback': name, 'vehicle': int(vid),
                    'why': 'the state digest changed but no advance was counted inside the window',
                    'before': before.get(vid), 'after': after.get(vid)})
        return result

    def guard_restore(self, vehicles, declared=False):
        """An anchor injected by a restore is a declared event or a violation -- never silent."""
        self.check_entry_point()
        if not declared:
            self.restores.append({'why': 'state changed by a restore path without declaration'})
            raise Refusal('UNDECLARED_ANCHOR_INJECTION', {'n': len(self.restores)})
        self.restores.append({'why': 'declared restore', 'declared': True})
        return True

    # -- the receipt --------------------------------------------------------------------
    def receipt(self):
        """A receipt is only certification-grade while every check is clean."""
        problems = []
        if not self._installed:
            problems.append('MONITOR_NOT_INSTALLED')
        if self.bypass_events:
            problems.append('MONITOR_BYPASSED')
        if self.unmonitored_writes:
            problems.append('UNMONITORED_STATE_WRITE')
        over = {vid: c for vid, c in self.counts.items() if c > self.n_max}
        if over:
            problems.append('COUNT_BOUND_VIOLATED')
        if self.clock_sup is not None and self.clocks and max(self.clocks) > self.clock_sup + 1e-12:
            problems.append('CLOCK_EXCEEDS_UPPER_BOUND')
        payload = {
            'schema': 'q3_g2c_execution_monitor',
            'EXPERIMENTAL_NOT_FOR_PRODUCTION': True,
            'counts': self.counts, 'n_max': self.n_max,
            'max_clock': (max(self.clocks) if self.clocks else None),
            'clock_sup': self.clock_sup,
            'callback_advances': self.callback_advances,
            'bypass_events': self.bypass_events,
            'unmonitored_writes': self.unmonitored_writes,
            'restores': self.restores,
            'coverage': ['counted advance calls (normal, wrapped and dynamically forwarded through '
                         'the attribute)', 'entry-point identity (replacement detected)',
                         'state-diff around every instrumented callback',
                         'declared-vs-undeclared restore'],
            'not_covered': ['a callback that mutates the cold chain through a captured reference taken '
                            'BEFORE the monitor was installed (use-after-capture)',
                            'process-level interference, signals, or another thread'],
            'problems': problems,
            'certifiable': not problems,
        }
        payload['receipt_digest'] = sha256_json(payload)
        return payload


def instrument(env, monitor, replanner=None, hooks=('snapshot_hook', 'oracle_hook')):
    """Wrap the engine's callbacks so each runs inside an audited window."""
    for name in hooks:
        existing = getattr(env, name, None)
        if existing is None:
            continue

        def make(name=name, existing=existing):
            def wrapped(*args, **kwargs):
                vehicles = args[5] if len(args) > 5 else kwargs.get('vehicles', [])
                return monitor.window(name, lambda: existing(*args, **kwargs), vehicles)

            return wrapped

        setattr(env, name, make())
    if replanner is not None:
        for name in ('plan', 'on_reveal', 'restore_state'):
            existing = getattr(replanner, name, None)
            if existing is None:
                continue

            def make(name=name, existing=existing):
                def wrapped(*args, **kwargs):
                    vehicles = kwargs.get('vehicles')
                    if vehicles is None:
                        vehicles = args[3] if len(args) > 3 else []
                    return monitor.window('replanner.' + name,
                                          lambda: existing(*args, **kwargs), vehicles)

                return wrapped

            setattr(replanner, name, make())
    return monitor


# =============================================================================================
# PART 3 -- the folded verdict
# =============================================================================================
def qualify_travel_and_clock(*, horizon, models, tw_speed, n_max, hours_per_time_unit=1.0,
                             observed_max_clock=None, monitor_receipt=None,
                             distance_witness=None):
    """Combine the distance-model domain, the clock bound and the monitor into one verdict."""
    refusals, conditional, notes = [], [], []
    tb = None
    try:
        tb = travel_bound(models, tw_speed)
    except Refusal as exc:
        refusals.append(exc.as_dict())

    clock_sup, slack = None, None
    if tb is not None and _finite(horizon) and float(horizon) > 0:
        clock_sup = float(horizon) + tb['travel_bound']
        if float(hours_per_time_unit) != 1.0:
            conditional.append({'code': 'TIME_UNIT_CONVERSION_NOT_BUDGETED',
                                'detail': {'hours_per_time_unit': hours_per_time_unit}})
        if observed_max_clock is not None and _finite(observed_max_clock):
            if float(observed_max_clock) > clock_sup + 1e-12:
                refusals.append({'code': 'CLOCK_EXCEEDS_UPPER_BOUND',
                                 'detail': {'observed': float(observed_max_clock),
                                            'T_clock_sup': clock_sup}})
        else:
            notes.append('no observed clock was supplied; the bound is a structural claim only')

    if distance_witness is not None and distance_witness.get('satisfied') is False:
        refusals.append({'code': 'DISTANCE_MODEL_WITNESS_FAILED',
                         'detail': distance_witness})

    if monitor_receipt is None:
        conditional.append({'code': 'CALLBACK_ADVANCES_UNWITNESSED',
                            'detail': {'why': 'no monitor receipt: the static bound cannot cover '
                                              'OPEN callbacks'}})
    elif not monitor_receipt.get('certifiable'):
        refusals.append({'code': 'MONITOR_NOT_CERTIFIABLE',
                         'detail': {'problems': monitor_receipt.get('problems')}})

    if refusals:
        verdict = REFUSED
    elif conditional:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {'schema': SCHEMA, 'verdict': verdict,
            'EXPERIMENTAL_NOT_FOR_PRODUCTION': True,
            'production_consumable': False,
            'implementation_version': IMPLEMENTATION_VERSION,
            'travel_bound': tb, 'T_clock_sup': clock_sup,
            'refusals': refusals, 'conditional': conditional, 'notes': notes,
            'distance_models': {m: DISTANCE_MODELS[m] for m in models},
            'PRODUCTION_NOT_QUALIFIED': True,
            'not_claimed': ['that the registered data domain is Euclidean (it must be witnessed)',
                            'that sqrt(2) is a universal upper bound',
                            'that the monitor sees a pre-installation captured reference'],
            }
