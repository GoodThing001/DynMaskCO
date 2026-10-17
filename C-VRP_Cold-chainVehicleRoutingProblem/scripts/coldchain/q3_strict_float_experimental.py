#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 STRICT floating-point bounds (exact rational arithmetic).

WHY THIS FILE EXISTS
====================
`q3_thermal_interval_experimental.py` rounds an upper bound UP by a fixed four ulps
(`math.nextafter` x4) but never proves that four ulps covers the error.  This module replaces that
unproven hop with arithmetic that is *closed under proof*:

  * every arithmetic step that touches a temperature is carried out in `fractions.Fraction`, i.e.
    exact rational arithmetic -- no rounding error at all;
  * the ONLY non-algebraic function in the executor (`math.exp`) is replaced by a
    **rigorous rational enclosure** `E_lo <= exp(-x) <= E_hi` built from a truncated Taylor sum plus
    a geometric tail bound.  That enclosure is valid for the TRUE exponential: it assumes NOTHING
    about the platform's libm.

TWO SEPARATE OBJECTS, WHICH THE PRE-FIX MODULE CONFLATES
========================================================
  R-level ("real dynamics")   what the closed-form heat model says:
                              Phi_dt(T0) = A + (T0-A)exp(-h dt) - c dt,  psi = max(tau, Phi).
                              Claims about this object need NO floating-point assumption.
  F-level ("executor floats") what `coldchain_state.py` actually stores in
                              `compartment_temperature_c`.  Claims about this object need one extra
                              input: a bound on how far the EXECUTOR's `math.exp` is from the true
                              exponential.  That quantity is called `exp_rel_error` (kappa) here and
                              it can NOT be derived from IEEE-754 (exp is not correctly rounded by
                              standard).  Every F-level function refuses when kappa is None, so no
                              caller can silently obtain an F-level claim without stating the
                              assumption it rests on.

WHAT IS PROVED HERE (derivations in the accompanying report, sections A2/A3)
===========================================================================
  L-A  exp(-x) enclosure : for x >= 0, K >= 1, K + 1 > x:
                           L_K = sum_{k<=K} x^k/k!,  tail(exp(x)) <= t_{K+1}/(1 - x/(K+1)),
                           hence 1/(L_K + tail) <= exp(-x) <= 1/L_K.  Exact rationals only.
  L-B  roundoff          : round-to-nearest gives |fl(v) - v| <= u*|v| with u = 2**-53; every basic
                           operation of the executor is round-to-nearest, so each `+-*/` is covered
                           by `_round_enclosure`.
  L-C  monotonicity      : round-to-nearest is monotone in its argument, so the whole executor step
                           (subtract, multiply, exp, add, subtract, max, add) is non-decreasing in
                           t0.  Hence exec(T) <= exec(U) for T <= U, which is what propagates a bound.
  L-D  P1 soundness      : with the same frozen constants and the same expression order the
                           certifier reproduces the executor's step bit-for-bit, so an outward shift
                           only has to cover certifier/executor DIVERGENCE, not rounding; under
                           bit-identical replay the shift may be zero.
  L-E  P2 unsoundness    : `tau + N*delta` rounded once and shifted 4 ulp is NOT an upper bound on
                           the executor float state, because the executor performs N separate
                           roundings.  `count_bound_strict` is the corrected bound and the executable
                           counterexample lives in `scripts/tests/test_q3_strict_float_bounds.py`.

This module is NOT wired into any certifier, cannot emit a production receipt
(`production_consumable` is hard-coded False) and does not modify any frozen file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_STRICT_FLOAT_EXPERIMENTAL_V1'

#: unit roundoff for IEEE-754 binary64 round-to-nearest (2**-53)
UNIT_ROUNDOFF = Fraction(1, 2 ** 53)

#: below this magnitude the relative roundoff model is replaced by a fixed absolute floor
#: (the subnormal range cannot honour a relative bound)
_REL_FLOOR = Fraction(1, 2 ** 900)
_ABS_FLOOR = Fraction(1, 2 ** 1000)

#: Taylor order floor.  The order is raised with the exponent so that the RELATIVE width of the
#: enclosure stays below ~1e-16 for every exponent of the declared domain (h*dt <= 4.8) and well
#: beyond it; see the report table in section A2.
_TAYLOR_TERMS = 32
#: beyond this many terms the rational enclosure is abandoned for the (valid but loose) 1/(1+x) form
_TAYLOR_MAX_X = 96
_TAYLOR_MAX_TERMS = 400


class Refusal(Exception):
    """Raised instead of returning a numerically meaningless answer.  Deterministic, fail-closed."""

    def __init__(self, code, detail=None):
        self.code = str(code)
        self.detail = dict(detail or {})
        super(Refusal, self).__init__('%s %s' % (self.code, self.detail))


# --------------------------------------------------------------------------------------------
# rigour primitives
# --------------------------------------------------------------------------------------------
def _round_enclosure(lo, hi, u=UNIT_ROUNDOFF):
    """Enclosure of fl(v) for any real v in [lo, hi] (round-to-nearest, lemma L-B)."""
    m = max(abs(lo), abs(hi))
    if m < _REL_FLOOR:
        return lo - _ABS_FLOOR, hi + _ABS_FLOOR
    slack = u * m
    return lo - slack, hi + slack


def _need_fraction(x, field):
    if not isinstance(x, Fraction):
        raise Refusal('NOT_AN_EXACT_RATIONAL', {'field': field, 'type': type(x).__name__})
    return x


def _need_finite_float(x, field):
    if isinstance(x, Fraction):
        raise Refusal('EXACT_RATIONAL_ON_FLOAT_PATH',
                      {'field': field, 'why': 'the F-level (executor) path takes the executor own '
                                              'float inputs; an exact rational duration must be '
                                              'converted by the caller so the rounding is explicit'})
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise Refusal('NON_NUMERIC_INPUT', {'field': field, 'value': repr(x)})
    v = float(x)
    if v != v or v in (float('inf'), float('-inf')):
        raise Refusal('NON_FINITE_INPUT', {'field': field, 'value': repr(x)})
    return v


def exp_neg_enclosure(x, terms=None):
    """Rigorous [lo, hi] with lo <= exp(-x) <= hi for exact rational x >= 0.

    Lemma L-A.  `terms` is the Taylor order K; the geometric tail bound needs K + 1 > x, so K is
    raised automatically.  Beyond `_TAYLOR_MAX_X` the (still valid, but loose) bounds (0, 1/(1+x))
    are returned, since exp(x) >= 1 + x implies exp(-x) <= 1/(1+x).
    """
    _need_fraction(x, 'x')
    if x < 0:
        raise Refusal('NEGATIVE_EXPONENT', {'x': str(x)})
    if x == 0:
        return Fraction(1), Fraction(1)
    k_terms = _TAYLOR_TERMS if terms is None else int(terms)
    if k_terms < 1:
        raise Refusal('BAD_TAYLOR_ORDER', {'terms': k_terms})
    if terms is None:
        # raise the order with the exponent so the relative width stays below double resolution
        try:
            k_terms = max(k_terms, 4 * int(math.ceil(float(x))) + 16)
        except (OverflowError, ValueError):                               # pragma: no cover
            k_terms = _TAYLOR_MAX_TERMS + 1
        if x >= _TAYLOR_MAX_X or k_terms > _TAYLOR_MAX_TERMS:
            return Fraction(0), Fraction(1, 1) / (1 + x)
    if x >= k_terms + 1:
        try:
            need = int(math.ceil(float(x))) + 8
        except (OverflowError, ValueError):                               # pragma: no cover
            need = _TAYLOR_MAX_TERMS + 1
        if need > _TAYLOR_MAX_TERMS:
            return Fraction(0), Fraction(1, 1) / (1 + x)
        k_terms = max(need, k_terms)
    term = Fraction(1)
    lower_exp = Fraction(1)
    for k in range(1, k_terms + 1):
        term = term * x / k
        lower_exp += term
    t_next = term * x / (k_terms + 1)
    r = x / (k_terms + 1)
    upper_exp = lower_exp + t_next / (1 - r)                              # r < 1 by construction
    if lower_exp <= 0:                                                    # pragma: no cover
        raise Refusal('INTERNAL_TAYLOR_DEGENERATE', {'x': str(x)})
    return Fraction(1, 1) / upper_exp, Fraction(1, 1) / lower_exp


def float_ceil(q):
    """Smallest binary64 >= the exact rational q (needed to resume a float recursion)."""
    _need_fraction(q, 'q')
    try:
        f = float(q)
    except OverflowError:                                                 # pragma: no cover
        return float('inf')
    if f == float('inf'):
        return f
    return math.nextafter(f, math.inf) if Fraction(f) < q else f


def float_floor(q):
    """Largest binary64 <= the exact rational q."""
    _need_fraction(q, 'q')
    try:
        f = float(q)
    except OverflowError:                                                 # pragma: no cover
        return float('-inf')
    if f == float('-inf'):
        return f
    return math.nextafter(f, -math.inf) if Fraction(f) > q else f


def ulp_gap(lo, hi):
    """(hi - lo) expressed in ulps of hi.  Reporting helper only -- never used in a verdict."""
    ref = abs(float(hi))
    step = math.ulp(ref) if ref else 5e-324
    return float(hi - lo) / step


# --------------------------------------------------------------------------------------------
# exact contract constants
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ZoneThermalExact:
    """One thermal zone with every contract constant available as an exact rational."""

    zone: int
    ambient_c: float
    target_c: float
    hard_low_c: float
    hard_high_c: float
    heat_transfer_per_hour: float
    cooling_rate_c_per_hour: float
    door_heat_kwh: float
    thermal_capacity_kwh_per_c: float
    contract_hash: str

    @property
    def A(self):
        return Fraction(self.ambient_c)

    @property
    def tau(self):
        return Fraction(self.target_c)

    @property
    def h(self):
        return Fraction(self.heat_transfer_per_hour)

    @property
    def c(self):
        return Fraction(self.cooling_rate_c_per_hour)

    @property
    def heat(self):
        return Fraction(self.door_heat_kwh)

    @property
    def capacity(self):
        return Fraction(self.thermal_capacity_kwh_per_c)

    @property
    def delta_real(self):
        """The exact rational door_heat / thermal_capacity (NOT the float quotient)."""
        return self.heat / self.capacity

    @property
    def delta_float(self):
        """The float the executor actually adds: fl(heat / capacity)."""
        return self.door_heat_kwh / self.thermal_capacity_kwh_per_c

    @property
    def drift_margin_c_per_hour(self):
        """c - h*(A - tau)^+ ; must be > 0 for P3/P3' to apply (D3)."""
        return self.c - self.h * max(Fraction(0), self.A - self.tau)

    @property
    def return_time_h(self):
        """Delta* = delta / margin, the worst-case time needed to fall back to tau (P3/P3')."""
        m = self.drift_margin_c_per_hour
        if m <= 0:
            raise Refusal('DRIFT_CONDITION_VIOLATED', {'zone': self.zone, 'margin': str(m)})
        return self.delta_real / m


def zone_thermals_exact(contract):
    """Read the thermal constants from a contract object; keep the contract hash for identity."""
    th = getattr(contract, 'thermal', None)
    if th is None:
        raise Refusal('CONTRACT_HAS_NO_THERMAL_BLOCK')
    zones = getattr(th, 'supported_temp_classes', None)
    if not zones:
        raise Refusal('CONTRACT_HAS_NO_TEMP_CLASSES')
    chash = getattr(contract, 'contract_hash', None)
    out = []
    for z in range(len(zones)):
        low, high = th.hard_temperature_bounds_c[z]
        out.append(ZoneThermalExact(
            zone=z, ambient_c=float(th.ambient_temperature_c),
            target_c=float(th.target_temperature_c[z]),
            hard_low_c=float(low), hard_high_c=float(high),
            heat_transfer_per_hour=float(th.heat_transfer_per_hour[z]),
            cooling_rate_c_per_hour=float(th.cooling_rate_c_per_hour[z]),
            door_heat_kwh=float(th.door_heat_kwh[z]),
            thermal_capacity_kwh_per_c=float(th.thermal_capacity_kwh_per_c[z]),
            contract_hash=chash))
    return tuple(out)


def require_contract_identity(contract, contract_identity):
    """The pre-fix module accepted ANY string as the contract identity; here it must be the hash."""
    chash = getattr(contract, 'contract_hash', None)
    if not contract_identity:
        raise Refusal('MISSING_CONTRACT_IDENTITY')
    if not chash:
        raise Refusal('CONTRACT_HAS_NO_HASH')
    if str(contract_identity) != str(chash):
        raise Refusal('CONTRACT_IDENTITY_MISMATCH',
                      {'declared': str(contract_identity), 'contract_hash': str(chash)})
    return str(chash)


# --------------------------------------------------------------------------------------------
# R-level: the REAL closed form (no floating-point assumption anywhere)
# --------------------------------------------------------------------------------------------
def real_phi_enclosure(z, t0, dt_h):
    """Enclosure of the REAL Phi_dt(T0) = A + (T0-A)exp(-h dt) - c dt."""
    _need_fraction(t0, 't0')
    _need_fraction(dt_h, 'dt_h')
    if dt_h < 0:
        raise Refusal('NEGATIVE_SEGMENT_DURATION', {'dt_h': str(dt_h)})
    e_lo, e_hi = exp_neg_enclosure(z.h * dt_h)
    d = t0 - z.A
    p_lo, p_hi = (d * e_lo, d * e_hi) if d >= 0 else (d * e_hi, d * e_lo)
    cool = z.c * dt_h
    return z.A + p_lo - cool, z.A + p_hi - cool


def real_psi_enclosure(z, t0, dt_h, eps=1e-9):
    """Enclosure of the REAL psi_dt(T0) = max(tau, Phi_dt(T0)).

    The executor short-circuits to `t0` when dt <= eps (`coldchain_state.py:361-362`); that branch is
    part of the declared semantics and is reproduced here rather than treated as a rounding detail.
    """
    if dt_h <= Fraction(eps):
        return t0, t0
    lo, hi = real_phi_enclosure(z, t0, dt_h)
    return max(z.tau, lo), max(z.tau, hi)


def real_segment_enclosure(z, t0, dt_h, opens, eps=1e-9):
    """Enclosure of the REAL segment-end temperature, including the door increment."""
    lo, hi = real_psi_enclosure(z, t0, dt_h, eps=eps)
    if not opens:
        return lo, hi
    return lo + z.delta_real, hi + z.delta_real


def real_u_recursion(z, segments, eps=1e-9):
    """P1 on the REAL model: U_0 = tau, U_n = psi_{dt_n}(U_{n-1}) + I_n*delta  =>  T_n <= U_n.

    Nothing here depends on floating point: the state is an exact rational and the exponential is
    bracketed by `exp_neg_enclosure`.  This is the strongest statement available with NO exp-error
    assumption, and it is a statement about the MODEL, not about the executor's stored floats.
    """
    hi = z.tau
    out = [hi]
    for i, seg in enumerate(segments):
        dt_h, opens = segment_pair(seg, 'segments[%d]' % i)
        _lo, hi = real_segment_enclosure(z, hi, dt_h, opens, eps=eps)
        out.append(hi)
    return out


def exp_free_worst_case(z):
    """Segment length that a certificate would need with NO knowledge of exp beyond 0 < exp(-x) <= 1.

    With the exponential unconstrained in (0, 1] the segment end can reach A, so a segment can only
    be certified non-warming when dt >= (A - tau)/c.  Reported so that the price of refusing to
    assume anything about libm is visible (for the default contract it is 3.5 / 4.2 / 5.375 h,
    i.e. orders of magnitude above any real service segment).
    """
    if z.A <= z.tau:
        return Fraction(0)
    return (z.A - z.tau) / z.c


# --------------------------------------------------------------------------------------------
# F-level: the EXECUTOR's float value -- requires an exp error bound
# --------------------------------------------------------------------------------------------
def executor_segment_enclosure(z, t0, dt_h, opens, exp_rel_error, eps=1e-9):
    """Rigorous enclosure of the value the executor stores after one segment, for one zone.

    `t0` / `dt_h` are the executor's own floats; `exp_rel_error` (kappa) is the declared relative
    error of the platform's `math.exp`:  |fl_exp(x) - exp(x)| <= kappa * exp(x).
    Refuses when kappa is None: no F-level claim exists without it.

    The expression order is reproduced exactly as in `coldchain_state.py:366-382` and `:413-430`.
    """
    if exp_rel_error is None:
        raise Refusal('EXP_ERROR_BOUND_NOT_DECLARED',
                      {'why': 'an upper bound on the platform exp error is required for any '
                              'statement about the executor float state; IEEE-754 does not '
                              'specify math.exp'})
    kappa = Fraction(exp_rel_error)
    if kappa < 0 or kappa >= 1:
        raise Refusal('EXP_ERROR_BOUND_OUT_OF_RANGE', {'kappa': str(kappa)})
    t0 = _need_finite_float(t0, 't0')
    dt_h = _need_finite_float(dt_h, 'dt_h')
    if dt_h < 0:
        raise Refusal('NEGATIVE_SEGMENT_DURATION', {'dt_h': repr(dt_h)})
    if dt_h <= eps:
        lo = hi = Fraction(t0)                                            # exact, kappa-independent
    else:
        arg = -(z.heat_transfer_per_hour * dt_h)                          # fl(-(h*dt)); sign flip exact
        if arg != arg or not math.isfinite(arg):
            raise Refusal('NON_FINITE_EXP_ARGUMENT', {'arg': repr(arg)})
        # the executor evaluates exp(arg); the enclosure is parameterised by x = -arg >= 0 and the
        # negation is exact, so no rounding is introduced by flipping the sign.
        e_lo, e_hi = exp_neg_enclosure(Fraction(-arg))
        e_lo, e_hi = e_lo * (1 - kappa), e_hi * (1 + kappa)
        d = Fraction(t0 - z.ambient_c)                                    # fl(t0 - A), exactly known
        prod_lo, prod_hi = (d * e_lo, d * e_hi) if d >= 0 else (d * e_hi, d * e_lo)
        p_lo, p_hi = _round_enclosure(prod_lo, prod_hi)                    # fl(d * E)
        n_lo, n_hi = _round_enclosure(z.A + p_lo, z.A + p_hi)              # fl(A + p)
        cdt = z.c * Fraction(dt_h)
        cdt_lo, cdt_hi = _round_enclosure(cdt, cdt)                        # fl(c * dt)
        cd_lo, cd_hi = _round_enclosure(n_lo - cdt_hi, n_hi - cdt_lo)      # fl(n - cdt)
        lo, hi = max(z.tau, cd_lo), max(z.tau, cd_hi)                      # max(tau, .)
    if not opens:
        return lo, hi
    d_lo, d_hi = _round_enclosure(z.delta_real, z.delta_real)              # fl(heat / capacity)
    return _round_enclosure(lo + d_lo, hi + d_hi)                          # fl(updated + delta)


def executor_u_recursion(z, segments, exp_rel_error, eps=1e-9):
    """P1 on the EXECUTOR floats: a rigorous upper bound of `compartment_temperature_c[z]`.

    L-C (monotonicity) turns the per-step enclosure into a propagation: the bound is rounded UP to a
    float before being fed back, so `exec(actual) <= exec(bound)` applies at every step.
    """
    hi = z.tau
    bounds = [hi]
    trace = [z.target_c]
    for i, seg in enumerate(segments):
        dt_h, opens = segment_pair(seg, 'segments[%d]' % i)
        _lo, bnd = executor_segment_enclosure(z, float_ceil(hi), float(dt_h), opens,
                                             exp_rel_error, eps=eps)
        hi = bnd
        bounds.append(hi)
        trace.append(float_ceil(hi))
    top = max(bounds)
    return {'u_sequence_float': trace, 'u_max': top, 'u_max_float': float_ceil(top)}


def count_bound_strict(z, n_doors):
    """P2 corrected: a VALID upper bound on the executor float state after `n_doors` purely-door
    segments (zero-length, i.e. no cooling), stated without any exp assumption.

    Lemma L-E.  Each door adds the same float delta_f = fl(heat/capacity) and each addition rounds by
    at most u relative to the RESULT MAGNITUDE, so `v <- (v + delta) + u*|v + delta|` is an inductive
    upper bound.  The absolute value matters: a zone whose target is below zero (target = -18 C)
    would be pushed the WRONG WAY by a bare `*(1+u)` multiplication, which is why that form is not
    used.  The pre-fix `door_count_bound` rounded ONCE and shifted 4 ulp, which is BELOW the executor
    for N >= 20 (see `scripts/tests/test_q3_strict_float_bounds.py`, control SFC-06).
    """
    if isinstance(n_doors, bool) or int(n_doors) != n_doors or n_doors < 0:
        raise Refusal('BAD_DOOR_COUNT', {'n_doors': repr(n_doors)})
    v = z.tau
    delta = Fraction(z.delta_float)
    for _ in range(int(n_doors)):
        nxt = v + delta
        v = nxt + UNIT_ROUNDOFF * abs(nxt)
    return v


def count_bound_closed_form(z, n_doors):
    """Closed-form twin of `count_bound_strict` (looser, and cheaper for a pure count argument)."""
    if isinstance(n_doors, bool) or int(n_doors) != n_doors or n_doors < 0:
        raise Refusal('BAD_DOOR_COUNT', {'n_doors': repr(n_doors)})
    n = int(n_doors)
    base = z.tau + n * Fraction(z.delta_float)
    return base + n * UNIT_ROUNDOFF * abs(base) + Fraction(n * (n - 1), 2) * UNIT_ROUNDOFF ** 2


# --------------------------------------------------------------------------------------------
# diagnosing the pre-fix 4-ulp guard
# --------------------------------------------------------------------------------------------
def plain_advance_float(z, t0, dt_h, eps=1e-9):
    """The executor's own arithmetic (identical to the pre-fix module's value before the shift)."""
    t0 = _need_finite_float(t0, 't0')
    dt_h = _need_finite_float(dt_h, 'dt_h')
    if dt_h <= eps:
        return t0
    natural = z.ambient_c + (t0 - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * dt_h)
    cooled = natural - z.cooling_rate_c_per_hour * dt_h
    return max(z.target_c, cooled)


def pre_fix_guard_value(z, t0, dt_h, ulp_steps=4, eps=1e-9):
    """Value returned by the pre-fix `advance_upward`, reproduced here for comparison only."""
    v = plain_advance_float(z, t0, dt_h, eps=eps)
    for _ in range(int(ulp_steps)):
        v = math.nextafter(v, math.inf)
    return v


def required_exp_error_budget(z, t0, dt_h, ulp_steps=4, iters=48):
    """Largest kappa for which a fixed `ulp_steps`-ulp upward shift is still a valid F-level bound.

    The pre-fix module shifts its bound up by `ulp_steps` ulps of its own value.  That can only be
    justified while kappa <= the value returned here, so this number is the assumption the shift
    makes silently.  Returned as an exact rational (or None when no exp is involved).
    """
    t0 = _need_finite_float(t0, 't0')
    dt_h = _need_finite_float(dt_h, 'dt_h')
    if dt_h <= 1e-9:
        return None                                                       # no exp involved at all
    guard = Fraction(pre_fix_guard_value(z, t0, dt_h, ulp_steps))
    if executor_segment_enclosure(z, t0, dt_h, False, Fraction(1, 2))[1] <= guard:
        return Fraction(1, 2)                                             # guard is generous
    lo, hi = Fraction(0), Fraction(1, 2)
    for _ in range(int(iters)):
        mid = (lo + hi) / 2
        if executor_segment_enclosure(z, t0, dt_h, False, mid)[1] <= guard:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------------------------
# input validation helpers (the pre-fix module raised uncaught TypeError/ValueError instead)
# --------------------------------------------------------------------------------------------
def segment_pair(seg, where):
    """Validate one (duration_h, opens_this_zone) entry.  Deterministic refusal, no raw exception."""
    try:
        dt_h, opens = seg
    except Exception:                                                     # noqa: BLE001
        raise Refusal('MALFORMED_SEGMENT', {'where': where, 'value': repr(seg),
                                            'expected': '(duration_h, opens_this_zone)'})
    if isinstance(dt_h, bool) or not isinstance(dt_h, (int, float, Fraction)):
        raise Refusal('NON_NUMERIC_SEGMENT_DURATION', {'where': where, 'value': repr(dt_h)})
    if isinstance(dt_h, Fraction):
        dt_exact = dt_h
        if dt_exact < 0:
            raise Refusal('NEGATIVE_SEGMENT_DURATION', {'where': where, 'dt_h': str(dt_exact)})
    else:
        dt_f = _need_finite_float(dt_h, where + '.dt_h')
        if dt_f < 0:
            raise Refusal('NEGATIVE_SEGMENT_DURATION', {'where': where, 'dt_h': repr(dt_f)})
        dt_exact = Fraction(dt_f)
    return dt_exact, bool(opens)


def require_segments(segments, where='segments'):
    """A segment list must be a finite sequence of well-formed pairs, otherwise REFUSED."""
    if segments is None:
        raise Refusal('MISSING_SEGMENT_DATA', {'where': where})
    if isinstance(segments, (str, bytes, dict)):
        raise Refusal('MALFORMED_SEGMENT_CONTAINER', {'where': where,
                                                     'type': type(segments).__name__})
    try:
        seq = list(segments)
    except Exception:                                                     # noqa: BLE001
        raise Refusal('MALFORMED_SEGMENT_CONTAINER', {'where': where,
                                                     'type': type(segments).__name__})
    out = []
    for i, seg in enumerate(seq):
        out.append(segment_pair(seg, '%s[%d]' % (where, i)))
    return out


def segment_duration_bounds(depart, end):
    """(exact real duration, executor float duration) for two finite floats.

    The executor computes `fl(end - depart)`.  For P3/P3' the dangerous direction is a SHORT segment,
    so the exact rational difference is returned (it is <= the rounded value whenever the rounding
    went up), together with the float the executor actually uses.
    """
    d_f = _need_finite_float(depart, 'depart')
    e_f = _need_finite_float(end, 'end')
    exact = Fraction(e_f) - Fraction(d_f)
    if exact < 0:
        raise Refusal('NEGATIVE_SEGMENT_DURATION', {'depart': d_f, 'end': e_f})
    return exact, float(e_f - d_f)
