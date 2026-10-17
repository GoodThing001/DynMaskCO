#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 S1: COMPLETE-ROUTE partition-independent bracket.

WHAT THIS CLOSES (Q3-1 and Q3-4 of `M1_L2最小可行性与成本裁决建议_2026-10-17.md` §4.2)
=====================================================================================
The pre-existing bracket (`q3_float_replay_qualification_experimental.py`) and P5
(`q3_event_segmentation_experimental.py`) are statements about ONE door-less advance and about a
RECORDED event stream.  Neither answers the decision-time question that L2 needs:

    given a candidate plan for the whole fleet, is the CONTINUOUS temperature provably inside the
    band [tau_z, tau_z + delta_z] at every door event, using NOTHING that the decider cannot see --
    in particular no future reveal clock and no future event partition?

This module answers exactly that for a complete route (frozen/committed prefix + candidate suffix +
WAIT + depot return + a mid-tour entry state) and writes "upper end above the threshold => reject"
as a decidable predicate.

THE TWO FACTS THAT MAKE IT POSSIBLE
===================================
S1-L1  the door clocks are determined by the PLAN.  `_commit` computes
       `depart = max(ready_time, clock)`, `arrive = depart + travel`, `service_start =
       max(arrive, tw_start)`, `finish = service_start + service_time`, and `_plan_and_commit` pops
       the next suffix element at the SAME clock the previous service completed
       (`strict_online_env.py:650-655`).  So within one vehicle the whole door-clock sequence is a
       function of (frozen prefix, plan, instance arrays).  The PARTITION of a door-less advance is
       NOT: it is cut at every global event clock, and reveal clocks are future information.  That
       is exactly why the bracket -- not the partition -- has to be propagated.
       LIMIT, stated up front: a reveal inside a committed leg wipes `mutable_suffix`
       (`strict_online_env.py:528`), so the plan is then not executed as submitted.  This is a
       CONDITIONAL PLAN CERTIFICATE ("if this plan is executed as submitted"), never a trajectory
       certificate.  Reachability of the plan is not claimed.

S1-L2  the bracket ends are computable without knowing the partition AND without any libm
       assumption.  For ANY partition of a door-less interval of total D (lemmas BR-1 / L-F / L-G,
       plus "flooring never lowers a value"):

           max(tau, Phi_raw_D(T0)) <= floored composition(T0) <= max(tau, F_D(T0))

       with `Phi_raw` the coarsest (one-piece) real advance and `F` the ODE flow.  Both ends are
       monotone in T0, so propagating the two ends separately through the door sequence is valid.
       The only transcendental is `exp`, enclosed exactly by
       `q3_strict_float_experimental.exp_neg_enclosure` (lemma L-A).  The DECISION
       (`sup_upper <= tau + delta`) is therefore an exact rational comparison and rests on nothing
       about the platform's `math.exp`.

THE THRESHOLD, RIGOROUSLY
=========================
`Theta_z` is the unique t with F_t(tau+delta) = tau, i.e. (1/h)*ln((tau+delta-T_inf)/(tau-T_inf)).
The pre-existing P5 computed it with `math.log` (unproven).  Here it is an exact rational enclosure
from the atanh series of ln (positive terms + geometric tail bound), so `gap >= Theta_z` is decided
exactly:
    gap >= Theta_hi  =>  provably back on the floor (collapse; ZERO exp evaluations)
    gap <  Theta_lo  =>  provably not back on the floor (evaluate the exact bracket)
    otherwise        =>  evaluate the exact bracket (conservative)
For the default contract every Theta is about 3.2-3.3 minutes and every gap is at least the later
stop's service time, so a normal plan costs zero transcendental evaluations.

WHAT THIS MODULE DOES NOT DO
============================
  * no F-level claim: the stored doubles are NOT shown to lie inside the real bracket.  That is
    B-3 / Q3-S3 and is OUT OF SCOPE this round;
  * no intra-segment profile (the executor defines none): the bound is stated at door events, and
    between doors the value is below its entry value;
  * nothing about TW, capacity, energy budget or quality;
  * not wired into any certifier; `production_consumable` is hard-coded False.

VERDICT LADDER (identical vocabulary to `q3_plan_layer_experimental`, never silently promoted)
=============================================================================================
  REFUSED           identity / domain / malformed input / unrevealed served node / commitment
                    mismatch / empty (inverted) bracket / inconsistent frozen prefix / non-finite
  NOT_CERTIFIABLE   some zone's door-event upper end exceeds tau_z + delta_z, or the entry state
                    already violates a hard temperature bound
  DIAGNOSTIC_ONLY   the arithmetic finished but a binding needed to turn it into a CLAIM is absent
                    (entry state unbound / commitment unbound / source identity unbound)
  PROVED_CONDITIONAL every binding in place and every zone's band holds at every door event
"""
from __future__ import annotations

from fractions import Fraction as F

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_S1_ROUTE_BRACKET_EXPERIMENTAL_V1'
SCHEMA = 'q3_s1_route_bracket'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: The propositions this module is allowed to rely on.  Theorem B / B1 / B2s are NOT here.
PROPOSITIONS_PROVED = (
    'S1-L1_DOOR_CLOCKS_DETERMINED_BY_THE_PLAN',
    'BR-1_REFINEMENT_RAISES (Phi_a(Phi_b(T)) - Phi_{a+b}(T) = c*b*(1-e^{-h*a}) >= 0)',
    'L-F_STEP_DOMINATED_BY_FLOW (Phi_Delta(T) <= F_Delta(T))',
    'L-G_FLOW_IS_A_PARTITION_INDEPENDENT_UPPER_BOUND (also with the max(tau,.) floor)',
    'FLOORING_NEVER_LOWERS_A_VALUE (max(tau, .) is monotone)',
    'BOTH_ENDS_ARE_MONOTONE_IN_T0 (so two-end propagation is valid)',
    'L-A_RATIONAL_EXP_ENCLOSURE (q3_strict_float_experimental.exp_neg_enclosure)',
    'RATIONAL_LN_ENCLOSURE (atanh series, positive terms + geometric tail bound)',
)
PROPOSITIONS_FORBIDDEN = ('THEOREM_B_BAND_ANY_SEQUENCE', 'B1_INTRA_SEGMENT_COROLLARY',
                          'B2_SUFFICIENCY', 'BITWISE_REPLAY_CERTIFIES_A_FUTURE_SUFFIX')

#: Copy of the engine's reveal tolerance (`strict_online_env.py:490/514/538`).  This is a mirror of
#: an implementation detail, NOT a ruling on the strict non-anticipation semantics: that question is
#: S-E / C2-L5 and is deliberately not decided here.  The receipt reports what a 0.0 tolerance would
#: additionally have refused (`strict_tolerance_would_refuse`).
DEFAULT_REVEAL_TOLERANCE = 1e-6

_LN_TERMS_MIN = 4
_LN_TERMS_MAX = 256
_LN_WIDTH_TARGET = F(1, 2 ** 96)


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


def _frac(x, field):
    """Exact rational of a finite float/int input.  Never silently accepts nan/inf."""
    if isinstance(x, F):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise SF.Refusal('NON_NUMERIC_INPUT', {'field': field, 'value': repr(x)})
    if not _finite(x):
        raise SF.Refusal('NON_FINITE_INPUT', {'field': field, 'value': repr(x)})
    return F(x)


# ---------------------------------------------------------------------------------------------
# rigorous ln (atanh series) -- no libm, no exp
# ---------------------------------------------------------------------------------------------
def rational_ln_enclosure(K, terms=None, width_target=_LN_WIDTH_TARGET):
    """Rigorous [lo, hi] with lo <= ln(K) <= hi for an exact rational K > 0.

    atanh series: with u = (K-1)/(K+1) in (-1, 1),
        ln K = 2 * sum_{j>=0} u^(2j+1)/(2j+1)
    Every term carries the sign of u, so the partial sum is a lower bound for u > 0 (and an upper
    bound for u < 0), and the remainder is bounded by the geometric tail
        |tail_J| <= |u|^(2J+1) / ((2J+1) * (1 - u^2)).
    Exact rationals only; `math` is never called.
    """
    if not isinstance(K, F):
        raise SF.Refusal('NOT_AN_EXACT_RATIONAL', {'field': 'K', 'type': type(K).__name__})
    if K <= 0:
        raise SF.Refusal('NONPOSITIVE_LOG_ARGUMENT', {'K': str(K)})
    if K == 1:
        return F(0), F(0)
    negate = K < 1
    kk = (1 / K) if negate else K
    u = (kk - 1) / (kk + 1)
    if u <= 0:                                                            # pragma: no cover
        raise SF.Refusal('INTERNAL_LN_DEGENERATE', {'K': str(K)})
    one_minus = 1 - u * u
    if terms is not None:
        j_terms = max(1, int(terms))
        power = u
        partial = F(0)
        for j in range(j_terms):
            power = u if j == 0 else power * u * u
            partial += power / (2 * j + 1)
        tail = power * u * u / ((2 * j_terms + 1) * one_minus)
    else:
        power = u
        partial = F(0)
        j = 0
        while True:
            power = u if j == 0 else power * u * u
            partial += power / (2 * j + 1)
            j += 1
            tail = power * u * u / ((2 * j + 1) * one_minus)
            if 2 * tail <= width_target or j >= _LN_TERMS_MAX:
                break
    lo, hi = 2 * partial, 2 * (partial + tail)
    return (-hi, -lo) if negate else (lo, hi)


# ---------------------------------------------------------------------------------------------
# per-zone rigorous thresholds
# ---------------------------------------------------------------------------------------------
def zone_thresholds_exact(contract):
    """Per-zone exact constants, the rigorous Theta enclosure, band / hard predicates.

    Refuses anything outside the declared domain: a non-positive heat transfer coefficient makes
    the ODE flow degenerate, and `margin <= 0` makes the band unrecoverable once left.
    """
    if contract is None:
        raise SF.Refusal('CONTRACT_MISSING')
    zones = SF.zone_thermals_exact(contract)
    out, n_ln = {}, 0
    for z in zones:
        if not (z.heat_transfer_per_hour > 0):
            raise SF.Refusal('NONPOSITIVE_HEAT_TRANSFER_OUTSIDE_DOMAIN',
                             {'zone': z.zone, 'h': repr(z.heat_transfer_per_hour)})
        margin = z.drift_margin_c_per_hour
        if margin <= 0:
            raise SF.Refusal('DRIFT_CONDITION_VIOLATED',
                             {'zone': z.zone, 'margin': str(margin),
                              'why': 'the band can only be re-entered when the drift margin is > 0'})
        t_inf = z.A - z.c / z.h
        if not (t_inf < z.tau):
            raise SF.Refusal('T_INFINITY_NOT_BELOW_TARGET',
                             {'zone': z.zone, 't_inf': str(t_inf), 'tau': str(z.tau)})
        delta = z.delta_real
        if delta < 0:                                                     # pragma: no cover
            raise SF.Refusal('NEGATIVE_DOOR_INCREMENT', {'zone': z.zone, 'delta': str(delta)})
        if delta == 0:
            theta_lo = theta_hi = F(0)                                    # ln(1) == 0, exact
            n_ln_z = 0
        else:
            k = (z.tau + delta - t_inf) / (z.tau - t_inf)
            lo, hi = rational_ln_enclosure(k)
            theta_lo, theta_hi = lo / z.h, hi / z.h
            n_ln_z = 1
        n_ln += n_ln_z
        out[z.zone] = {
            'zone': z.zone,
            'tau': z.tau, 'delta': delta, 'delta_float': F(z.delta_float),
            'A': z.A, 'h': z.h, 'c': z.c, 't_infinity': t_inf,
            'drift_margin': margin,
            'band_low': z.tau, 'band_high': z.tau + delta,
            'hard_low': F(z.hard_low_c), 'hard_high': F(z.hard_high_c),
            'theta_lo': theta_lo, 'theta_hi': theta_hi,
            'theta_float_math_log': _theta_via_math_log(z, delta, t_inf),
            'n_ln_enclosures': n_ln_z,
            'return_time_h': z.return_time_h,
        }
    return out, n_ln


def _theta_via_math_log(z, delta, t_inf):
    """The PRE-EXISTING (unproven) value, reported only so the difference is visible.  NEVER used
    in a verdict: every decision in this module is an exact rational comparison."""
    import math
    num = float(z.tau + delta - t_inf)
    den = float(z.tau - t_inf)
    if num <= 0 or den <= 0:                                              # pragma: no cover
        return None
    return math.log(num / den) / float(z.h)


# ---------------------------------------------------------------------------------------------
# instance view (read-only, validated)
# ---------------------------------------------------------------------------------------------
class InstanceArrays(object):
    """The only instance data the certificate reads.  Everything is validated on the way in."""

    def __init__(self, raw, n_zones):
        if raw is None:
            raise SF.Refusal('INSTANCE_MISSING')
        self.raw = raw
        self.service_time = self._list(raw, 'service_time')
        self.tw_start = self._list(raw, 'tw_start')
        self.temp_class = self._list(raw, 'temp_class')
        self.reveal_time = self._list(raw, 'reveal_time')
        self.n_nodes = len(self.service_time)
        if self.n_nodes < 2:
            raise SF.Refusal('INSTANCE_TOO_SMALL', {'n_nodes': self.n_nodes})
        for name, arr in (('tw_start', self.tw_start), ('temp_class', self.temp_class),
                          ('reveal_time', self.reveal_time)):
            if len(arr) != self.n_nodes:
                raise SF.Refusal('INSTANCE_ROW_LENGTH_MISMATCH',
                                 {'field': name, 'len': len(arr), 'n_nodes': self.n_nodes})
        for i in range(self.n_nodes):
            for name, arr in (('service_time', self.service_time), ('tw_start', self.tw_start),
                              ('reveal_time', self.reveal_time)):
                if not _finite(arr[i]):
                    raise SF.Refusal('NON_FINITE_INSTANCE_FIELD',
                                     {'field': name, 'node': i, 'value': repr(arr[i])})
            if i != 0 and (int(self.temp_class[i]) < 0 or int(self.temp_class[i]) >= n_zones):
                raise SF.Refusal('TEMP_CLASS_OUT_OF_RANGE',
                                 {'node': i, 'temp_class': self.temp_class[i],
                                  'n_zones': n_zones})
        self.dist_mat = raw.get('dist_mat')
        self.coords = raw.get('coords')
        self.tw_speed = float(raw.get('tw_speed', 1.0))
        if not _finite(self.tw_speed) or self.tw_speed <= 0:
            raise SF.Refusal('NONPOSITIVE_TIME_SCALE', {'tw_speed': repr(self.tw_speed)})
        self.depot_is_clean = float(self.service_time[0]) == 0.0

    @staticmethod
    def _list(raw, key):
        if key not in raw:
            raise SF.Refusal('INSTANCE_MISSING_FIELDS', {'missing': [key]})
        return [float(x) for x in raw[key]]

    def travel_fn(self):
        """Default travel model: the engine's own `dist_mat[i][j] / tw_speed`
        (`strict_online_env.py:193-194`), falling back to Euclidean distance over `coords`."""
        if self.dist_mat is not None:
            dm, sp = self.dist_mat, self.tw_speed

            def _fn(a, b):
                try:
                    v = float(dm[int(a)][int(b)])
                except Exception:                                         # noqa: BLE001
                    raise SF.Refusal('TRAVEL_LOOKUP_FAILED', {'a': repr(a), 'b': repr(b)})
                if not _finite(v) or v < 0:
                    raise SF.Refusal('NON_FINITE_TRAVEL', {'a': repr(a), 'b': repr(b),
                                                           'value': repr(v)})
                return v / sp
            return _fn
        if self.coords is not None:
            import math
            co = self.coords

            def _fn(a, b):
                ax, ay = float(co[int(a)][0]), float(co[int(a)][1])
                bx, by = float(co[int(b)][0]), float(co[int(b)][1])
                return math.hypot(ax - bx, ay - by) / self.tw_speed
            return _fn
        raise SF.Refusal('NO_TRAVEL_MODEL',
                         {'why': 'the instance has neither dist_mat nor coords'})

    def revealed_at(self, clock, tol):
        return {i for i in range(self.n_nodes) if self.reveal_time[i] <= float(clock) + float(tol)}


# ---------------------------------------------------------------------------------------------
# the door schedule of ONE vehicle (frozen prefix + candidate suffix)
# ---------------------------------------------------------------------------------------------
def validate_suffix(route, inst, where='suffix'):
    """A candidate suffix must be node ids; 0 may only terminate the route (depot return)."""
    if route is None:
        raise SF.Refusal('NULL_ROUTE', {'where': where})
    if isinstance(route, (str, bytes, dict)):
        raise SF.Refusal('MALFORMED_ROUTE', {'where': where, 'type': type(route).__name__})
    seq = list(route)
    out = []
    for pos, n in enumerate(seq):
        q = _frac(n, '%s[%d]' % (where, pos))
        if q.denominator != 1:
            raise SF.Refusal('NON_INTEGER_STOP', {'where': where, 'pos': pos, 'node': str(q)})
        n = int(q)
        if n < 0 or n >= inst.n_nodes:
            raise SF.Refusal('STOP_OUT_OF_RANGE', {'where': where, 'node': n,
                                                   'n_nodes': inst.n_nodes})
        if n == 0 and pos != len(seq) - 1:
            raise SF.Refusal('DEPOT_IN_STOP_LIST',
                             {'where': where, 'pos': pos,
                              'why': 'node 0 is the depot; it may only terminate the route'})
        out.append(n)
    return out


def door_schedule(*, inst, clock, current_node, ready_time, prefix, suffix, travel_fn=None):
    """The ordered door list of one vehicle from its frozen prefix and its candidate suffix.

    Returns `{'stops', 'n_suffix_stops', 'n_prefix_stops', 'first_depart', 'last_finish',
    'return_leg', 'return_arrival'}`.  Each stop is
    `{'node','zone','arrive','service_start','finish','source'}`.
    """
    clock = _frac(clock, 'clock')
    ready_time = _frac(ready_time, 'ready_time')
    current_node = int(current_node)
    if current_node < 0 or current_node >= inst.n_nodes:
        raise SF.Refusal('CURRENT_NODE_OUT_OF_RANGE', {'current_node': current_node})
    travel = travel_fn or inst.travel_fn()
    stops = []
    node, depart = current_node, None
    for k, rec in enumerate(prefix or ()):
        where = 'prefix[%d]' % k
        if not isinstance(rec, dict) or 'node' not in rec or 'finish' not in rec:
            raise SF.Refusal('MALFORMED_PREFIX_RECORD',
                             {'where': where, 'keys': sorted(dict(rec or {}))})
        n = int(rec['node'])
        if n == 0:
            raise SF.Refusal('DEPOT_IN_PREFIX', {'where': where, 'node': n})
        if n <= 0 or n >= inst.n_nodes:
            raise SF.Refusal('STOP_OUT_OF_RANGE', {'where': where, 'node': n,
                                                   'n_nodes': inst.n_nodes})
        arrive = _frac(rec.get('arrive', rec['finish']), where + '.arrive')
        finish = _frac(rec['finish'], where + '.finish')
        if finish < arrive:
            raise SF.Refusal('PREFIX_FINISH_BEFORE_ARRIVE',
                             {'node': n, 'arrive': str(arrive), 'finish': str(finish)})
        svc = _frac(inst.service_time[n], 'service_time[%d]' % n)
        tws = _frac(inst.tw_start[n], 'tw_start[%d]' % n)
        want = max(arrive, tws) + svc
        if finish != want:
            raise SF.Refusal('PREFIX_CLOCK_INCONSISTENT',
                             {'node': n, 'declared_finish': str(finish),
                              'recomputed_finish': str(want),
                              'why': 'finish = max(arrive, tw_start) + service_time is the only '
                                     'clock the executor can produce for a committed leg'})
        if finish < clock:
            raise SF.Refusal('PREFIX_FINISH_BEFORE_CLOCK',
                             {'node': n, 'finish': str(finish), 'clock': str(clock)})
        stops.append({'node': n, 'zone': int(inst.temp_class[n]), 'arrive': arrive,
                      'service_start': max(arrive, tws), 'finish': finish,
                      'source': 'frozen_prefix'})
        node, depart = n, finish
    n_prefix = len(stops)
    if depart is None:
        depart = max(ready_time, clock)
    first_depart = depart
    n_suffix = 0
    for n in suffix:
        if n == 0:
            break
        svc = _frac(inst.service_time[n], 'service_time[%d]' % n)
        if svc <= 0:
            raise SF.Refusal('SERVED_ORDER_WITH_ZERO_SERVICE_TIME',
                             {'node': n, 'service_time': repr(inst.service_time[n]),
                              'why': 'a served ORDER node with zero service time cannot be told '
                                     'apart from the depot placeholder'})
        leg = _frac(travel(node, n), 'travel[%d->%d]' % (node, n))
        if leg < 0:                                                       # pragma: no cover
            raise SF.Refusal('NEGATIVE_TRAVEL', {'from': node, 'to': n, 'travel': str(leg)})
        arrive = depart + leg
        tws = _frac(inst.tw_start[n], 'tw_start[%d]' % n)
        start = max(arrive, tws)
        finish = start + svc
        stops.append({'node': n, 'zone': int(inst.temp_class[n]), 'depart': depart,
                      'arrive': arrive, 'service_start': start, 'finish': finish,
                      'source': 'candidate_suffix'})
        node, depart = n, finish
        n_suffix += 1
    ret_leg = _frac(travel(node, 0), 'travel[%d->0]' % node)
    return {'stops': stops, 'n_suffix_stops': n_suffix, 'n_prefix_stops': n_prefix,
            'last_node': node, 'last_finish': depart, 'return_leg': ret_leg,
            'return_arrival': depart + ret_leg, 'first_depart': first_depart}


# ---------------------------------------------------------------------------------------------
# the bracket propagation
# ---------------------------------------------------------------------------------------------
def exp_enclosure(x):
    """Enclosure of exp(-x) for exact rational x >= 0, via the frozen rigour primitive.

    Returns `(mode, lo, hi)` with mode in {'exact','taylor','coarse'}.  `coarse` is beyond the
    Taylor domain and uses (0, 1/(1+x)), which is still a valid enclosure, just loose.
    """
    if not isinstance(x, F):
        raise SF.Refusal('NOT_AN_EXACT_RATIONAL', {'field': 'x', 'type': type(x).__name__})
    if x < 0:
        raise SF.Refusal('NEGATIVE_EXPONENT', {'x': str(x)})
    if x == 0:
        return 'exact', F(1), F(1)
    lo, hi = SF.exp_neg_enclosure(x)
    if lo == 0 and hi == 1 / (1 + x):
        return 'coarse', lo, hi
    return 'taylor', lo, hi


def propagate_zone(*, zth, entry_lo, entry_hi, gaps):
    """Propagate `[entry_lo, entry_hi]` through a gap list.  Exact rationals throughout.

    `gaps` is a list of dicts `{'delta': Fraction, 'opens_door': bool, 'stop_index': int,
    'node': int}` in chronological order.  A gap that is provably long enough to return the zone to
    the floor costs NO transcendental evaluation; every other gap costs exactly ONE.
    """
    tau, delta, band_high = zth['tau'], zth['delta'], zth['band_high']
    lo, hi = entry_lo, entry_hi
    if lo > hi:
        raise SF.Refusal('EMPTY_BRACKET',
                         {'zone': zth['zone'], 'lower': str(lo), 'upper': str(hi),
                          'why': 'an inverted interval is refused, never silently swapped'})
    rows, n_exp, n_loose, n_unresolved = [], 0, 0, 0
    sup = hi
    for g in gaps:
        d = g['delta']
        if d < 0:
            raise SF.Refusal('NEGATIVE_GAP', {'zone': zth['zone'], 'delta': str(d)})
        row = {'stop_index': g.get('stop_index'), 'node': g.get('node'),
               'delta': d, 'opens_door': bool(g['opens_door']),
               'entry_lower': lo, 'entry_upper': hi, 'n_exp': 0,
               'collapse': None, 'exp_mode': None}
        if hi <= tau:
            row['collapse'] = 'already_at_floor'
            lo = hi = tau
        elif hi <= band_high and d >= zth['theta_hi']:
            row['collapse'] = 'gap_at_least_theta_hi'
            lo = hi = tau
        else:
            mode, _e_lo, e_hi = exp_enclosure(zth['h'] * d)
            n_exp += 1
            row['n_exp'] = 1
            row['exp_mode'] = mode
            if mode == 'coarse':
                n_loose += 1
            ti = zth['t_infinity']
            if hi > ti and (hi - ti) * e_hi <= (tau - ti):
                row['collapse'] = 'proved_return'
                lo = hi = tau
            else:
                row['collapse'] = 'none'
                n_unresolved += 1
                hi_new = ti + (hi - ti) * e_hi
                lo_new = zth['A'] + (lo - zth['A']) * e_hi - zth['c'] * d
                lo = lo_new if lo_new > tau else tau
                hi = hi_new if hi_new > tau else tau
        if lo > hi:
            raise SF.Refusal('EMPTY_BRACKET',
                             {'zone': zth['zone'], 'stage': 'propagate',
                              'lower': str(lo), 'upper': str(hi)})
        # the bracket ENDS BEFORE THE DOOR: this is the pair a replay must land inside
        row['lower_before_door'] = lo
        row['upper_before_door'] = hi
        if g['opens_door']:
            lo += delta
            hi += delta
            if lo > hi:                                                   # pragma: no cover
                raise SF.Refusal('EMPTY_BRACKET',
                                 {'zone': zth['zone'], 'stage': 'door',
                                  'lower': str(lo), 'upper': str(hi)})
        row['lower_after'] = lo
        row['upper_after'] = hi
        if hi > sup:
            sup = hi
        rows.append(row)
    return {'zone': zth['zone'], 'gaps': rows, 'lower': lo, 'upper': hi, 'sup_upper': sup,
            'n_exp': n_exp, 'n_short_gaps': n_exp, 'n_loose_enclosures': n_loose,
            'n_unresolved_gaps': n_unresolved,
            'band_ok': sup <= band_high,
            'band_excess': (sup - band_high) if sup > band_high else F(0),
            'hard_ok': sup <= zth['hard_high'],
            'hard_excess': (sup - zth['hard_high']) if sup > zth['hard_high'] else F(0)}


# ---------------------------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------------------------
def check_route_certificate(*, contract, contract_identity, instance, clock, plans_by_vehicle,
                            entry_state_by_vehicle, committed_order_ids=None, revealed_nodes=None,
                            state_identity=None, expected_state_identity=None,
                            source_identity=None, active_zone_mask=None, travel_fn=None,
                            reveal_tolerance=DEFAULT_REVEAL_TOLERANCE, thresholds=None):
    """Certify the continuous thermal band over the COMPLETE route of the whole fleet.

    Keyword-only and fully explicit: there are no defaults that could hide a protocol leak.  See
    the module docstring for exactly what is and is not claimed.
    """
    refusals, notes = [], []
    exp_counter = {'n': 0}

    # ---- 0 identity ------------------------------------------------------------------------
    try:
        contract_hash = SF.require_contract_identity(contract, contract_identity)
    except SF.Refusal as exc:
        refusals.append({'code': exc.code, 'detail': exc.detail})
        contract_hash = None
    if state_identity is None or expected_state_identity is None:
        notes.append('STATE_IDENTITY_UNBOUND')
    elif str(state_identity) != str(expected_state_identity):
        refusals.append({'code': 'STATE_IDENTITY_MISMATCH',
                         'detail': {'supplied': str(state_identity),
                                    'expected': str(expected_state_identity)}})
    if source_identity is None:
        notes.append('SOURCE_IDENTITY_UNBOUND')

    # ---- 1 domain + thresholds -------------------------------------------------------------
    try:
        if thresholds is None:
            zth_all, n_ln = zone_thresholds_exact(contract)
        else:
            zth_all = {int(k): v for k, v in thresholds.items()}
            n_ln = 0            # thresholds are REUSED, so no ln enclosure is evaluated this call
    except SF.Refusal as exc:
        return _receipt(REFUSED, [{'code': exc.code, 'detail': exc.detail}], notes, {},
                        {'clock': None}, contract_hash, {}, {})
    n_zones = len(zth_all)
    mask = ((True,) * n_zones if active_zone_mask is None
            else tuple(bool(x) for x in active_zone_mask))
    if len(mask) != n_zones:
        refusals.append({'code': 'ZONE_MASK_LENGTH_MISMATCH',
                         'detail': {'mask': len(mask), 'zones': n_zones}})
    elif not all(mask):
        refusals.append({'code': 'ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN',
                         'detail': {'mask': list(mask),
                                    'why': 'the certificate is proved only with every zone active'}})
    try:
        clock_f = _frac(clock, 'clock')
        inst = InstanceArrays(instance, n_zones)
    except SF.Refusal as exc:
        refusals.append({'code': exc.code, 'detail': exc.detail})
        return _receipt(REFUSED, refusals, notes, {}, {'clock': None}, contract_hash, {}, {})

    # ---- 2 plan shape, visibility, commitment ----------------------------------------------
    if not isinstance(plans_by_vehicle, dict) or not plans_by_vehicle:
        refusals.append({'code': 'MISSING_PLAN',
                         'detail': {'type': type(plans_by_vehicle).__name__}})
        plans_by_vehicle = {}
    schedules, covered, duplicated = {}, set(), []
    for vid in sorted(plans_by_vehicle):
        rec = plans_by_vehicle[vid]
        if not isinstance(rec, dict):
            refusals.append({'code': 'MALFORMED_PLAN_RECORD', 'detail': {'vehicle': str(vid)}})
            continue
        try:
            suffix = validate_suffix(rec.get('suffix', ()), inst, 'suffix[%s]' % vid)
            sch = door_schedule(inst=inst, clock=clock_f,
                                current_node=rec.get('current_node', 0),
                                ready_time=rec.get('ready_time', clock_f),
                                prefix=rec.get('prefix', ()), suffix=suffix,
                                travel_fn=travel_fn)
        except SF.Refusal as exc:
            refusals.append({'code': exc.code, 'detail': dict(exc.detail, vehicle=str(vid))})
            continue
        for s in sch['stops']:
            n = int(s['node'])
            if n in covered:
                duplicated.append(n)
            covered.add(n)
        sch['suffix'] = suffix
        schedules[vid] = sch
    if duplicated:
        refusals.append({'code': 'ORDER_SERVED_TWICE',
                         'detail': {'nodes': sorted(set(duplicated))}})
    visible = (set(int(x) for x in revealed_nodes) if revealed_nodes is not None
               else inst.revealed_at(clock_f, reveal_tolerance))
    strict_visible = (None if revealed_nodes is not None
                      else inst.revealed_at(clock_f, 0.0))
    hidden = sorted(covered - visible)
    if hidden:
        refusals.append({'code': 'SERVES_UNREVEALED_NODE',
                         'detail': {'nodes': hidden[:20], 'n_hidden': len(hidden),
                                    'tolerance': float(reveal_tolerance),
                                    'why': 'strict-online: a served node must already be revealed '
                                           'at the decision clock'}})
    if committed_order_ids is None:
        notes.append('COMMITMENT_UNBOUND')
    else:
        committed = set(int(x) for x in committed_order_ids)
        if committed != covered:
            refusals.append({'code': 'COMMITTED_SET_MISMATCH',
                             'detail': {'missing_from_plan': sorted(committed - covered)[:20],
                                        'planned_but_not_committed':
                                            sorted(covered - committed)[:20]}})
    if not inst.depot_is_clean:
        refusals.append({'code': 'DEPOT_PROFILE_UNEXPECTED',
                         'detail': {'service_time': inst.service_time[0]}})

    # ---- 3 entry state intervals -----------------------------------------------------------
    entry, entry_missing = {}, []
    if not isinstance(entry_state_by_vehicle, dict):
        refusals.append({'code': 'MALFORMED_ENTRY_STATE',
                         'detail': {'type': type(entry_state_by_vehicle).__name__}})
        entry_state_by_vehicle = {}
    for vid in sorted(schedules):
        ent = entry_state_by_vehicle.get(vid)
        if ent is None:
            entry_missing.append(str(vid))
            continue
        try:
            lo_t = tuple(_frac(x, 'entry.lower') for x in ent[0])
            hi_t = tuple(_frac(x, 'entry.upper') for x in ent[1])
        except Exception as exc:                                          # noqa: BLE001
            refusals.append({'code': getattr(exc, 'code', 'MALFORMED_ENTRY_STATE'),
                             'detail': dict(getattr(exc, 'detail', {}), vehicle=str(vid))})
            continue
        if len(lo_t) != n_zones or len(hi_t) != n_zones:
            refusals.append({'code': 'ENTRY_STATE_LENGTH_MISMATCH',
                             'detail': {'vehicle': str(vid), 'len': [len(lo_t), len(hi_t)],
                                        'zones': n_zones}})
            continue
        entry[vid] = (lo_t, hi_t)
    if entry_missing:
        notes.append('ENTRY_STATE_UNBOUND')

    # ---- 4 per-zone gap lists over the WHOLE fleet -----------------------------------------
    per_zone = {z: {'zone': z, 'gaps': [], 'n_doors': 0, 'n_gaps': 0,
                    'contributing_vehicles': []} for z in sorted(zth_all)}
    per_vehicle = {}
    n_stops_total = 0
    for vid in sorted(schedules):
        sch = schedules[vid]
        stops = sch['stops']
        n_stops_total += len(stops)
        per_vehicle[vid] = {'n_stops': len(stops), 'n_suffix_stops': sch['n_suffix_stops'],
                            'n_prefix_stops': sch['n_prefix_stops'],
                            'first_depart': sch['first_depart'],
                            'last_finish': sch['last_finish'],
                            'return_arrival': sch['return_arrival'],
                            'stops': [{'node': s['node'], 'zone': s['zone'],
                                       'arrive': s['arrive'], 'finish': s['finish'],
                                       'service_time': F(inst.service_time[s['node']]),
                                       'source': s['source']} for s in stops]}
        for idx, s in enumerate(stops):
            per_zone[s['zone']]['n_doors'] += 1
            per_zone[s['zone']]['contributing_vehicles'].append(
                {'vehicle': str(vid), 'stop_index': idx, 'node': s['node'],
                 'finish': s['finish']})
        for z in sorted(zth_all):
            prev, prev_is_entry = clock_f, True
            for idx, s in enumerate(stops):
                if s['zone'] != z:
                    continue
                per_zone[z]['gaps'].append({'vehicle': vid, 'stop_index': idx,
                                            'node': s['node'],
                                            'delta': s['finish'] - prev,
                                            'opens_door': True, 'is_entry_gap': prev_is_entry})
                prev, prev_is_entry = s['finish'], False
    for info in per_zone.values():
        info['n_gaps'] = len(info['gaps'])

    # ---- 5 propagate (per vehicle: each compartment is independent) ------------------------
    for z, info in sorted(per_zone.items()):
        zth = zth_all[z]
        rows_by_vehicle, sup = {}, None
        for vid in sorted(schedules):
            if vid not in entry:
                continue
            e_lo, e_hi = entry[vid][0][z], entry[vid][1][z]
            gaps = [{'delta': g['delta'], 'opens_door': True, 'stop_index': g['stop_index'],
                     'node': g['node']} for g in info['gaps'] if g['vehicle'] == vid]
            if not gaps:
                prop = {'zone': z, 'gaps': [], 'n_exp': 0, 'n_short_gaps': 0,
                        'n_loose_enclosures': 0, 'n_unresolved_gaps': 0,
                        'lower': e_lo, 'upper': e_hi, 'sup_upper': e_hi,
                        'band_ok': (e_lo <= e_hi and e_hi <= zth['band_high']),
                        'band_excess': (max(F(0), e_hi - zth['band_high'])
                                        if e_lo <= e_hi else F(0)),
                        'hard_ok': e_hi <= zth['hard_high'],
                        'hard_excess': max(F(0), e_hi - zth['hard_high'])}
                if e_lo > e_hi:
                    refusals.append({'code': 'EMPTY_BRACKET',
                                     'detail': {'vehicle': str(vid), 'zone': z,
                                                'lower': str(e_lo), 'upper': str(e_hi)}})
            else:
                try:
                    prop = propagate_zone(zth=zth, entry_lo=e_lo, entry_hi=e_hi, gaps=gaps)
                except SF.Refusal as exc:
                    refusals.append({'code': exc.code,
                                     'detail': dict(exc.detail, vehicle=str(vid),
                                                    where='propagate')})
                    prop = {'zone': z, 'gaps': [], 'n_exp': 0, 'n_short_gaps': 0,
                            'n_loose_enclosures': 0, 'n_unresolved_gaps': 0,
                            'lower': e_lo, 'upper': e_hi, 'sup_upper': e_hi,
                            'band_ok': False, 'band_excess': F(0), 'hard_ok': True,
                            'hard_excess': F(0)}
            rows_by_vehicle[str(vid)] = prop
            exp_counter['n'] += prop['n_exp']
            if sup is None or prop['sup_upper'] > sup:
                sup = prop['sup_upper']
        if sup is None:
            sup = zth['band_low']
        info.update({'per_vehicle': rows_by_vehicle, 'sup_upper': sup,
                     'band_ok': sup <= zth['band_high'],
                     'band_excess': max(F(0), sup - zth['band_high']),
                     'hard_ok': sup <= zth['hard_high'],
                     'hard_excess': max(F(0), sup - zth['hard_high']),
                     'n_exp': sum(r['n_exp'] for r in rows_by_vehicle.values()),
                     'n_short_gaps': sum(r['n_short_gaps'] for r in rows_by_vehicle.values()),
                     'n_loose_enclosures': sum(r['n_loose_enclosures']
                                               for r in rows_by_vehicle.values()),
                     'theta_lo': zth['theta_lo'], 'theta_hi': zth['theta_hi'],
                     'band_high': zth['band_high'], 'hard_high': zth['hard_high'],
                     'hard_low': zth['hard_low']})

    # ---- 6 inverted entry intervals / hard-bound sanity ------------------------------------
    entry_out_of_hard = []
    for vid in sorted(entry):
        for z in sorted(zth_all):
            lo, hi = entry[vid][0][z], entry[vid][1][z]
            if lo > hi:
                # already refused above (EMPTY_BRACKET); no hard-bound verdict is meaningful here
                continue
            zth = zth_all[z]
            if hi > zth['hard_high'] or lo < zth['hard_low']:
                entry_out_of_hard.append({'vehicle': str(vid), 'zone': z,
                                          'lower': str(lo), 'upper': str(hi),
                                          'hard': [str(zth['hard_low']), str(zth['hard_high'])]})

    # ---- 7 verdict -------------------------------------------------------------------------
    bad_zones = sorted(z for z, i in per_zone.items() if not i['band_ok'])
    if refusals:
        verdict = REFUSED
    elif bad_zones or entry_out_of_hard:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    summary = {
        'n_vehicles': len(schedules), 'n_stops': n_stops_total,
        'n_doors': sum(i['n_doors'] for i in per_zone.values()),
        'n_gaps': sum(i['n_gaps'] for i in per_zone.values()),
        'n_short_gaps': sum(i['n_short_gaps'] for i in per_zone.values()),
        'n_exp': sum(i['n_exp'] for i in per_zone.values()),
        'n_exp_internal_counter': exp_counter['n'],
        'n_loose_enclosures': sum(i['n_loose_enclosures'] for i in per_zone.values()),
        'n_ln_enclosures': n_ln,
        'n_ln_enclosures_required_by_contract': sum(int(v.get('n_ln_enclosures') or 0)
                                                    for v in zth_all.values()),
        'thresholds_supplied': thresholds is not None,
        'bad_zones': bad_zones,
        'entry_out_of_hard': entry_out_of_hard,
        'band_ok_all': not bad_zones,
        'hard_ok_all': not [z for z, i in per_zone.items() if not i['hard_ok']],
        'max_band_excess_c': max([float(i['band_excess']) for i in per_zone.values()] or [0.0]),
        'n_suffix_stops': sum(v['n_suffix_stops'] for v in per_vehicle.values()),
        'n_prefix_stops': sum(v['n_prefix_stops'] for v in per_vehicle.values()),
        'strict_tolerance_would_refuse': (None if strict_visible is None
                                          else sorted(covered - strict_visible)),
        'reveal_tolerance': float(reveal_tolerance),
        'n_covered_orders': len(covered),
    }
    return _receipt(verdict, refusals, notes, per_zone, {'clock': float(clock_f)},
                    contract_hash, per_vehicle, summary, thresholds=zth_all)


def _zone_row(z, i):
    return {'zone': z,
            'sup_upper_c': (None if i.get('sup_upper') is None else float(i['sup_upper'])),
            'sup_upper_exact': (None if i.get('sup_upper') is None else str(i['sup_upper'])),
            'band_high_c': (None if i.get('band_high') is None else float(i['band_high'])),
            'band_ok': i.get('band_ok'),
            'band_excess_c': float(i.get('band_excess') or 0.0),
            'hard_high_c': (None if i.get('hard_high') is None else float(i['hard_high'])),
            'hard_ok': i.get('hard_ok'),
            'n_doors': i.get('n_doors'), 'n_gaps': i.get('n_gaps'),
            'n_short_gaps': i.get('n_short_gaps'), 'n_exp': i.get('n_exp'),
            'theta_lo_h': (None if i.get('theta_lo') is None else float(i['theta_lo'])),
            'theta_hi_h': (None if i.get('theta_hi') is None else float(i['theta_hi']))}


def _receipt(verdict, refusals, notes, per_zone, ctx, contract_hash, per_vehicle, summary,
             thresholds=None):
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'verdict': verdict,
        'experimental_verdict': verdict,
        'contract_hash': contract_hash,
        'context': ctx,
        'refusals': refusals,
        'binding_notes': list(notes),
        'per_zone': {str(z): _zone_row(z, i) for z, i in sorted((per_zone or {}).items())},
        'per_zone_detail': {str(z): i for z, i in sorted((per_zone or {}).items())},
        'per_vehicle': per_vehicle or {},
        'thresholds': {str(z): {'tau': float(v['tau']), 'delta': float(v['delta']),
                                'delta_exact': str(v['delta']),
                                'band_low': float(v['band_low']),
                                'band_high': float(v['band_high']),
                                'hard_low': float(v['hard_low']),
                                'hard_high': float(v['hard_high']),
                                't_infinity': float(v['t_infinity']),
                                'drift_margin': float(v['drift_margin']),
                                'theta_lo_h': float(v['theta_lo']),
                                'theta_hi_h': float(v['theta_hi']),
                                'theta_interval_width_h': float(v['theta_hi'] - v['theta_lo']),
                                'theta_float_math_log': v.get('theta_float_math_log'),
                                'n_ln_enclosures': v.get('n_ln_enclosures')}
                        for z, v in sorted((thresholds or {}).items())},
        'summary': summary or {},
        'propositions_used': list(PROPOSITIONS_PROVED),
        'propositions_forbidden': list(PROPOSITIONS_FORBIDDEN),
        'refuted_proposition_used': False,
        'claim': ('S1-L1 + BR-1 + L-F + L-G: for EVERY admissible partition of every door-less '
                  'interval of this plan, the REAL executor trajectory stays inside the propagated '
                  'bracket at every door event, so every zone stays in [tau_z, tau_z + delta_z] at '
                  'every door event (and below its entry value between doors).'),
        'not_claimed': [
            'F-LEVEL: that the DOUBLE the executor stores lies inside this real-valued bracket. '
            'That step needs a declared exp error bound kappa or bit-identical replay and is '
            'Q3-S3 / B-3 -- explicitly OUT OF SCOPE this round',
            'the intra-segment (continuous) profile: the executor defines no intra-segment '
            'trajectory, so the bound is stated at door events and at segment boundaries',
            'that the plan is executed as submitted: a reveal inside a committed leg wipes '
            'mutable_suffix (strict_online_env.py:528), so this is a CONDITIONAL PLAN CERTIFICATE',
            'feasibility, profitability or installability: TW, capacity, energy budget and quality '
            'are outside this certificate',
            'any instance, node count, contract or data family not checked here',
        ],
    }


def to_jsonable(obj):
    """JSON-safe rendering that keeps exact rationals readable as strings."""
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
