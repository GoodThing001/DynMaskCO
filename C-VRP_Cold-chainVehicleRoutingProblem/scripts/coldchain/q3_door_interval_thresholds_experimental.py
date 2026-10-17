#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 door-interval thresholds, ERRATA-CORRECTED.

TWO CORRECTIONS TO THE PREVIOUS ROUND (both raised by the user, both verified here)
===================================================================================
ERR-1  **`Delta*` used on a TOTAL door interval is NOT unsound.**  The previous round wrote that
       "Delta* or T_single on the total interval is unsound".  Only the second half is true.

           Theta_z = (1/h_z) * ln(1 + h_z * Delta*_z)        with Delta*_z = delta_z / m_z
           m_z     = c_z - h_z*(A - tau_z)  > 0

       so `ln(1+x) <= x` gives  **Theta_z <= Delta*_z** directly.  Requiring the interval to be at
       least `Delta*` is therefore CONSERVATIVE and SAFE, just slightly loose.  What is unsound is
       `T_single` (the return time of ONE unsplit piece), because a refined partition is warmer.

           T_single_z  <  Theta_z  <=  Delta*_z           (strict at the left, provable)

       Moreover `Delta*` is a RATIONAL function of the contract constants: no `exp`, no `log`, hence
       no libm assumption at all.  It is the better object for an auditable implementation.

ERR-2  **`L-F` is false for `0 < Delta <= _EPS`.**  The previous round stated it for every `Delta >= 0`.
       The executor short-circuits `duration_h <= _EPS` to the IDENTITY (`coldchain_state.py:361-362`),
       and the identity sits ABOVE the flow for any state above the ODE fixed point, so

           executor(Delta) - F_Delta(T0) = +6.0e-10  at Delta = 1e-9, T0 = tau + delta/2, zone 0.

       Corrected scope:  L-F holds for `Delta > _EPS` only; for `Delta <= _EPS` the map is the
       identity and consumes NO cooling time.  Consequently the upper bound is on the **effective
       cooling duration** `Delta_eff = sum of the pieces with Delta > _EPS`, not on the wall clock.

THE ENVIRONMENT DOES NOT LOSE COOLING (and where it could)
==========================================================
`_advance_coldchain_to` (`strict_online_env.py:209-230`) returns EARLY when `end <= start + 1e-9`
and does NOT move `coldchain_updated_time`.  The skipped span is therefore not lost: the next real
advance starts from the same anchor and covers it.  The only way time can be lost is a skipped
advance at the END of an inter-door interval, and that loss is bounded by the skip threshold:

    Delta_eff  >=  (clock_n - clock_{n-1})  -  _SKIP_SLACK          (_SKIP_SLACK ~ 1e-9 h)

`TRUSTED_ANCHOR_SEMANTICS` below lists the eight conditions this rests on; if any of them fails the
checker must return NOT_CERTIFIABLE rather than substituting the wall clock for the effective
duration.

THE CERTIFICATE CONDITION (exact rational, log-free, kappa-free)
===============================================================
    For each zone z and each pair of consecutive door events on that zone of the same vehicle:
        (clock_n - clock_{n-1}) - _SKIP_SLACK   >=   Delta*_z          [exact Fraction comparison]

Every term is either an exact double (the two clocks, from the recorded event stream) or an exact
rational (Delta*_z from the contract constants).  The comparison is done in `fractions.Fraction`, so
there is no rounding, no tolerance, no `math.exp`/`math.log`, and no epsilon to sign.  The sufficient
condition is `T_single <= Theta <= Delta*`, so using `Delta*` is the most conservative of the three and
needs no transcendental.

WHAT THIS IS NOT
================
Not a production certificate (`production_consumable` False), not a continuous-temperature proof
(the intra-segment profile has no executor model), not a reachability theorem, and not a statement
about stored floats (that still needs the bracket or S3).
"""
from __future__ import annotations

import math
from fractions import Fraction

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_DOOR_INTERVAL_THRESHOLDS_V1'
SCHEMA = 'q3_door_interval_thresholds'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: `_EPS` of the executor, taken from the frozen module rather than retyped.
EPS_EXACT = Fraction(1e-9)
#: A skipped `_advance_coldchain_to` call costs at most the skip threshold.  The threshold is
#: `fl(start + 1e-9)`, whose excess over `1e-9` is below one ulp of 24 (3.6e-15), so 2*_EPS is a
#: provable over-estimate and is still five orders of magnitude below any Delta*.
_SKIP_SLACK = 2 * EPS_EXACT

#: The eight facts that make "wall clock >= effective cooling" true for THIS environment.  If any of
#: them is false for the executor under audit, the checker refuses.
TRUSTED_ANCHOR_SEMANTICS = (
    ('A-1', 'strict_online_env.py:209-230',
     'the only way the environment advances a door-less interval is _advance_coldchain_to, which '
     'calls transition_segment with served_customer=None and a POSITIVE duration, or returns early.'),
    ('A-2', 'strict_online_env.py:215-220',
     'the early return happens when end <= start + 1e-9, and it does NOT update '
     'coldchain_updated_time, so the skipped span is covered by the next real advance.'),
    ('A-3', 'strict_online_env.py:230, 332',
     'coldchain_updated_time is set to `end` on a real advance and to `clock` after a door, so the '
     'anchors tile the inter-door interval contiguously.'),
    ('A-4', 'strict_online_env.py:318-331',
     'the door is applied with depart_time == arrival_time == service_finish == clock, i.e. duration '
     'exactly 0, which is the identity in _advance_temperatures plus the door increment.'),
    ('A-5', 'strict_online_env.py:386-402',
     'the return COMPLETION call is also duration 0; the cooling of the return TRIP happened in the '
     'advance calls that preceded it, so "the return leg does not cool" would be wrong.'),
    ('A-6', 'strict_online_env.py:488-510',
     'no other code path calls transition_segment with a duration strictly between 0 and 1e-9.'),
    ('A-7', 'coldchain_state.py:208-211',
     'duration_h is time_to_hours(end - depart) and time_to_hours refuses negative durations, so no '
     'advance can move the clock backwards.'),
    ('A-8', 'coldchain_state.py:361-362',
     'duration_h <= _EPS is the identity, so a piece with 0 < duration <= _EPS would consume no time; '
     'A-1/A-2/A-6 are what exclude such a piece from the environment path.'),
)


# ---------------------------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------------------------
def zone_thresholds_exact(contract, zone_index):
    """All three thresholds for one zone, with the two rational ones as exact Fractions."""
    zone = [z for z in SF.zone_thermals_exact(contract) if z.zone == int(zone_index)][0]
    h, c = zone.h, zone.c
    tau, A = zone.tau, zone.A
    delta = Fraction(zone.door_heat_kwh) / Fraction(zone.thermal_capacity_kwh_per_c)
    margin = c - h * max(Fraction(0), A - tau)
    if margin <= 0:
        raise SF.Refusal('DRIFT_CONDITION_VIOLATED', {'zone': zone.zone})
    delta_star = delta / margin                          # EXACT, rational, no transcendental
    t_inf = A - c / h                                    # exact rational
    theta = (1.0 / float(h)) * math.log(1.0 + float(h * delta_star))   # needs log -> float
    return {
        'zone': zone.zone, 'h': h, 'c': c, 'tau': tau, 'A': A,
        'delta_exact': delta, 'margin_exact': margin, 't_inf_exact': t_inf,
        'delta_star_exact': delta_star,
        'delta_star_float_naive': float(delta_star),
        'delta_star_float_up': float_ceil_exact(delta_star),
        'delta_star_float_down': float_floor_exact(delta_star),
        'naive_float_is_safe_upper_bound': Fraction(float(delta_star)) >= delta_star,
        'theta_float': theta,
        'theta_le_delta_star': Fraction(theta) <= delta_star,
        'ln1p_identity_holds': abs(theta - (1.0 / float(h)) * math.log(
            1.0 + float(h * delta_star))) < 1e-15,
        'theta_min': theta * 60.0, 'delta_star_min': float(delta_star) * 60.0,
    }


def float_ceil_exact(q):
    """Smallest binary64 >= the exact rational q (provable outward rounding, no transcendental)."""
    f = float(q)
    while Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def float_floor_exact(q):
    f = float(q)
    while Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def _finite(x):
    """Local, because the strict-float module exposes only the private `_need_finite_float`."""
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


def t_single_float(contract, zone_index, iters=200):
    """Return time of ONE unsplit executor piece (numeric; DIAGNOSTIC, never used in a verdict)."""
    import coldchain_state as cs
    z = [x for x in SF.zone_thermals_exact(contract) if x.zone == int(zone_index)][0]
    t0 = z.target_c + float(z.delta_real)

    def floored(total):
        temps = [float(t) for t in contract.thermal.target_temperature_c]
        temps[z.zone] = t0
        temps, _e = cs._advance_temperatures(tuple(temps), float(total), (True,) * len(temps),
                                             contract)
        return temps[z.zone]

    lo, hi = 0.0, 1.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if floored(mid) > z.target_c:
            lo = mid
        else:
            hi = mid
    return hi


def ordering_report(contract):
    """`T_single <= Theta <= Delta*` per zone, with the margin at each inequality."""
    zones = SF.zone_thermals_exact(contract)
    rows = []
    for z in zones:
        th = zone_thresholds_exact(contract, z.zone)
        ts = t_single_float(contract, z.zone)
        rows.append({'zone': z.zone, 't_single_h': ts, 'theta_h': th['theta_float'],
                     'delta_star_h': float(th['delta_star_exact']),
                     't_single_le_theta': ts <= th['theta_float'],
                     'theta_le_delta_star': Fraction(th['theta_float']) <= th['delta_star_exact'],
                     'theta_minus_t_single_h': th['theta_float'] - ts,
                     'delta_star_minus_theta_h': float(th['delta_star_exact'] - Fraction(
                         th['theta_float']))})
    return {'rows': rows,
            'theorem': 'T_single < Theta <= Delta*, because Phi_dt <= F_dt (L-F, dt > _EPS) gives '
                       'T_single <= Theta, and ln(1+x) <= x gives Theta <= Delta*',
            'consequence': 'requiring an inter-door interval >= Delta* is SUFFICIENT and conservative; '
                           'requiring it >= T_single is NOT'}


# ---------------------------------------------------------------------------------------------
# effective cooling duration
# ---------------------------------------------------------------------------------------------
def effective_cooling_bounds(clocks):
    """Exact (wall, effective_lower, slack) for one inter-door interval given its clock samples.

    `clocks` is the ordered list of clock values at which the environment touched this vehicle
    inside the interval (the two door clocks plus any intermediate event clocks).  Only the wall
    length is available to a checker that sees just the recorded door events, hence the caller may
    pass `[clock_prev, clock_next]` alone; the returned lower bound is then the honest one.
    """
    if len(clocks) < 2:
        raise SF.Refusal('INTERVAL_NEEDS_TWO_CLOCKS', {'n': len(clocks)})
    ex = [Fraction(float(c)) for c in clocks]
    for a, b in zip(ex, ex[1:]):
        if b < a:
            raise SF.Refusal('CLOCK_MOVED_BACKWARDS', {'a': float(a), 'b': float(b)})
    wall = ex[-1] - ex[0]
    # the anchor only stops at a real advance; a skipped tail costs at most the skip threshold
    eff_lower = wall - _SKIP_SLACK
    return {'wall_exact': wall, 'effective_lower_exact': eff_lower,
            'slack_exact': _SKIP_SLACK, 'wall_h': float(wall),
            'effective_lower_h': float(eff_lower)}


def check_door_intervals_exact(*, contract, contract_identity, event_stream, state_identity=None,
                               expected_state_identity=None, anchor_semantics_assumed=True,
                               mode='delta_star'):
    """Exact-rational inter-door check.  Log-free, kappa-free, fail-closed.

    `mode='delta_star'` (default, the auditable one) compares against the rational Delta*;
    `mode='theta'` compares against the float Theta and is reported as DIAGNOSTIC_ONLY because it
    needs `math.log`, whose error is not specified by any standard.
    """
    import coldchain_state as cs
    refusals, notes = [], []
    try:
        cid = SF.require_contract_identity(contract, contract_identity)
    except SF.Refusal as exc:
        cid = None
        refusals.append({'code': exc.code, 'detail': exc.detail})
    if state_identity is None or expected_state_identity is None:
        notes.append('STATE_IDENTITY_UNBOUND')
    elif str(state_identity) != str(expected_state_identity):
        refusals.append({'code': 'STATE_IDENTITY_MISMATCH',
                         'detail': {'supplied': str(state_identity),
                                    'expected': str(expected_state_identity)}})
    if not anchor_semantics_assumed:
        refusals.append({'code': 'ANCHOR_SEMANTICS_NOT_ESTABLISHED',
                         'detail': {'why': 'without TRUSTED_ANCHOR_SEMANTICS the wall clock is not a '
                                           'lower bound for the effective cooling duration; the '
                                           'checker refuses instead of substituting it'}})
    if mode not in ('delta_star', 'theta'):
        refusals.append({'code': 'UNKNOWN_MODE', 'detail': {'mode': mode}})
    if not isinstance(event_stream, dict) or not event_stream:
        refusals.append({'code': 'MISSING_EVENT_STREAM'})
        event_stream = {}
    per_zone, per_vehicle = {}, {}
    for z in SF.zone_thermals_exact(contract):
        th = zone_thresholds_exact(contract, z.zone)
        need = th['delta_star_exact'] if mode == 'delta_star' else Fraction(th['theta_float'])
        need_float = Fraction(th['theta_float']) if mode == 'theta' else th['delta_star_exact']
        per_zone[z.zone] = {'threshold_used': mode, 'threshold_exact': str(need),
                            'threshold_h': float(need), 'threshold_min': float(need) * 60.0,
                            'theta_h': th['theta_float'], 'theta_min': th['theta_min'],
                            'delta_star_h': float(th['delta_star_exact']),
                            't_single_h': t_single_float(contract, z.zone),
                            'intervals': [], 'n_doors': 0, 'n_intervals': 0,
                            'min_interval_h': None, 'min_slack_exact': None,
                            'violations': []}
        for vid in sorted(event_stream):
            rows = event_stream[vid]
            if not isinstance(rows, (list, tuple)):
                refusals.append({'code': 'MALFORMED_VEHICLE_STREAM',
                                 'detail': {'vehicle': str(vid)}})
                continue
            prev = None
            for i, r in enumerate(rows):
                clk = r.get('door_clock')
                zc = r.get('zone')
                ok = _finite(clk) and zc is not None
                if not ok:
                    refusals.append({'code': 'MALFORMED_EVENT',
                                     'detail': {'vehicle': str(vid), 'index': i}})
                    continue
                if int(zc) != z.zone:
                    continue
                per_zone[z.zone]['n_doors'] += 1
                if prev is not None:
                    try:
                        b = effective_cooling_bounds([prev, clk])
                    except SF.Refusal as exc:
                        refusals.append({'code': exc.code, 'detail': exc.detail})
                        prev = clk
                        continue
                    slack = b['effective_lower_exact'] - need
                    row = {'vehicle': str(vid), 'from_clock': float(prev),
                           'to_clock': float(clk),
                           'wall_h': b['wall_h'],
                           'effective_lower_h': b['effective_lower_h'],
                           'threshold_h': float(need),
                           'slack_exact': str(slack), 'slack_h': float(slack),
                           'ok': slack >= 0}
                    per_zone[z.zone]['intervals'].append(row)
                    per_zone[z.zone]['n_intervals'] += 1
                    if per_zone[z.zone]['min_interval_h'] is None or \
                            b['wall_h'] < per_zone[z.zone]['min_interval_h']:
                        per_zone[z.zone]['min_interval_h'] = b['wall_h']
                    if per_zone[z.zone]['min_slack_exact'] is None or \
                            slack < Fraction(per_zone[z.zone]['min_slack_exact']):
                        per_zone[z.zone]['min_slack_exact'] = slack
                    if not row['ok']:
                        per_zone[z.zone]['violations'].append(row)
                prev = clk
            per_vehicle['%s' % vid] = {'zone%d' % z.zone: per_zone[z.zone]['n_doors']}
    bad = [k for k, v in per_zone.items() if v['violations']]
    if refusals:
        verdict = REFUSED
    elif bad:
        verdict = NOT_CERTIFIABLE
    elif mode == 'theta' or notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'verdict': verdict, 'experimental_verdict': verdict, 'contract_hash': cid,
        'refusals': refusals, 'binding_notes': notes,
        'per_zone': {str(k): v for k, v in sorted(per_zone.items())},
        'summary': {'n_intervals': sum(v['n_intervals'] for v in per_zone.values()),
                    'bad_zones': bad,
                    'threshold_used': mode,
                    'min_slack_h': min([float(v['min_slack_exact'])
                                        for v in per_zone.values()
                                        if v['min_slack_exact'] is not None] or [None]),
                    'skip_slack_h': float(_SKIP_SLACK),
                    'transcendental_used': (mode == 'theta')},
        'anchor_semantics': [{'id': a, 'where': b, 'fact': c} for a, b, c in TRUSTED_ANCHOR_SEMANTICS],
        'claim': ("for every zone, the EFFECTIVE cooling duration between consecutive door events is "
                  "at least Delta*_z, hence the following door starts at exactly tau_z and ends at "
                  "exactly tau_z + delta_z"),
        'not_claimed': [
            'that the stored FLOATS equal the real values (this is an R-level statement)',
            'the intra-segment (continuous) profile -- the executor has no intra-segment model',
            'reachability over instances / policies / service-time families not audited',
            'that wall clock may replace the effective duration when TRUSTED_ANCHOR_SEMANTICS fails'],
    }


# ---------------------------------------------------------------------------------------------
# Task 2: the seven segmentation cases, each with its own statement
# ---------------------------------------------------------------------------------------------
def segmentation_case_ledger(contract):
    """The seven cases the user asked to be handled separately, with the honest statement for each."""
    import coldchain_state as cs
    z = [x for x in SF.zone_thermals_exact(contract) if x.zone == 1][0]
    th = zone_thresholds_exact(contract, 1)
    temps = [float(t) for t in contract.thermal.target_temperature_c]
    temps[1] = z.target_c + float(z.delta_real)

    def advance(state_temps, dur):
        out, _e = cs._advance_temperatures(tuple(state_temps), float(dur), (True,) * len(temps),
                                           contract)
        return list(out)

    top = temps[1]
    c1 = advance(temps, 1e-12)
    c2 = advance(temps, 0.9 * float(EPS_EXACT))
    c3 = advance(temps, 2 * float(EPS_EXACT))
    c4 = advance(temps, 0.01)
    c5 = advance(advance(temps, 0.01), 0.01)
    c6 = advance(temps, 0.02)
    door = cs._apply_door_event(tuple(temps), (True,) * 3, 1, contract)[0][1]
    return {'zone': 1, 'start_c': top,
            'cases': [
                {'id': 'C1 zero-length door (Delta = 0)', 'delta_h': 0.0,
                 'executor_after_c': door, 'cools': door < top,
                 'statement': 'the door call is the identity in _advance_temperatures plus delta; it '
                              'cools by exactly 0 and is NOT a cooling segment',
                 'handled_by': 'A-4; the interval check never counts a door as cooling'},
                {'id': 'C2 sub-epsilon positive duration (0 < Delta <= _EPS)', 'delta_h': 1e-12,
                 'executor_after_c': c1[1], 'cools': c1[1] < top,
                 'statement': 'the executor treats it as the identity (no cooling), which is ABOVE '
                              'the continuous flow: L-F does not hold here',
                 'handled_by': 'A-1/A-2/A-6 exclude such a piece from the environment path; if a '
                               'caller produced one, that time would be lost and the wall clock would '
                               'no longer bound the effective duration'},
                {'id': 'C3 duration just below _EPS', 'delta_h': 0.9 * float(EPS_EXACT),
                 'executor_after_c': c2[1], 'cools': c2[1] < top,
                 'statement': 'same as C2 (identity)', 'handled_by': '_EPS branch'},
                {'id': 'C4 duration just above _EPS', 'delta_h': 2 * float(EPS_EXACT),
                 'executor_after_c': c3[1], 'cools': c3[1] < top,
                 'statement': 'the closed form is used again and L-F recovers',
                 'handled_by': 'the normal branch'},
                {'id': 'C5 many door-less splits of one interval', 'delta_h': 0.02,
                 'executor_after_c': c5[1], 'cools': c5[1] < top,
                 'warmer_than_one_piece': c5[1] > c6[1],
                 'statement': 'splitting is WARMER than one piece (BR-1), so the wall interval is a '
                              'valid bound only through the flow (L-G), not through one piece',
                 'handled_by': 'Delta* >= Theta (ERR-1)'},
                {'id': 'C6 one piece of the same total duration', 'delta_h': 0.02,
                 'executor_after_c': c6[1], 'cools': c6[1] < top,
                 'statement': 'the coldest admissible response', 'handled_by': 'lower end of BRACKET'},
                {'id': 'C7 consecutive doors on the same vehicle', 'delta_h': None,
                 'executor_after_c': None, 'cools': None,
                 'statement': 'the interval between two doors is what the check measures; a zero '
                              'interval leaves the zone at tau + 2*delta',
                 'handled_by': 'the checker requires interval >= Delta* and reports zero intervals '
                               'as violations'},
            ],
            'theoretical_note': {
                'theta_le_delta_star': th['theta_le_delta_star'],
                'delta_star_h': float(th['delta_star_exact']),
                'theta_h': th['theta_float'],
                't_single_h': t_single_float(contract, 1)}}


# ---------------------------------------------------------------------------------------------
# Task 4: what the M1-b replay does and does not imply
# ---------------------------------------------------------------------------------------------
def m1b_replay_scope():
    """Static facts about `replay_thermal_envelope` and the composite gate (read-only)."""
    return {
        'replay_segmentation': 'one transition_segment per service, duration = service_finish - '
                               'depart, door applied at its end (m1b_thermal_gate.py:276-296)',
        'environment_segmentation': 'a door-less advance at every event clock plus a zero-duration '
                                    'door (strict_online_env.py:209-230, 318-331)',
        'direction_of_the_difference': 'the replay splits LESS, therefore it is COOLER than the '
                                       'environment at the same service completion (BR-1)',
        'magnitude_bound': 'the bracket width, measured at <= 1.0e-03 C on the tested grid',
        'per_zone_max_status': 'the replay envelope is a SAMPLED envelope; it may not be upgraded to '
                               'a partition-independent continuous upper bound',
        'composite_gate_status': 'THERMAL_CONTINUOUS_OK is ALWAYS False on the production class '
                                 '(m1b_q2_certifier.py:106-110, 183-191, 375), so composite_pass is '
                                 'fail-closed today: there is NO false pass on the production path',
        'residual': 'if a continuous-level proof is ever installed, the sampled level must not be the '
                    'upper bound it is compared against; the ODE end of the bracket is the correct '
                    'object',
    }
