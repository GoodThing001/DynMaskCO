#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 EVENT-SEGMENTATION AUDIT and the P5 inter-door condition.

WHY THIS FILE EXISTS (the defect this module answers)
=====================================================
The previous round proved P3' ("only DOOR segments need to be at least Delta* long") and a plan-layer
checker that measured DOOR-SEGMENT DURATION as `service_finish - depart`.  That is the segmentation of
`m1b_thermal_gate.replay_thermal_envelope`, **not** the segmentation of the real environment.

The real environment (`scripts/simulation/strict_online_env.py`) advances the cold chain with
`_advance_coldchain_to(v, clock)` (`:209-230`) as a **door-less** segment covering travel + waiting +
service, and only THEN, when a service completes, calls `transition_segment(..., depart_time=clock,
arrival_time=clock, service_finish=clock, served_customer=node, ...)` (`:318-331`).  The door
therefore fires inside a segment whose duration is **exactly zero**.

Consequences (all of them are facts about the source, recorded in `ENV_SEGMENTATION_FACTS` below):
  1. `duration_h = 0 <= _EPS` so `_advance_temperatures` short-circuits to the identity
     (`coldchain_state.py:361-362`); the door simply adds `delta` to whatever the door-less advance
     left behind.
  2. Hence the door segment's duration is 0 for EVERY door, so P3' read literally against the
     environment's own segment sequence declares EVERY plan NOT_CERTIFIABLE.  P3' is not wrong as a
     theorem; it is attached to the wrong object.
  3. `service_time >= Delta*` is NOT the right premise.  The right object is the **clock interval
     between two consecutive door events on the same zone of the same vehicle**.

THE CORRECTED CONDITION (P5, proved below)
==========================================
  L-F  single step : for Delta >= 0 and any T,  Phi_Delta(T) <= F_Delta(T),
                     where Phi is the executor's closed form and F is the exact flow of
                     dT/dt = h(A-T) - c  (henceforth "the ODE flow").
                     Phi_Delta - F_Delta = (c/h)*[(1 - e^{-h Delta}) - h Delta] <= 0  because
                     1 - e^{-x} <= x for every x >= 0.
  L-G  partitioning: for ANY partition Delta_1 + ... + Delta_k = Delta_total,
                     Phi_{Delta_1} o ... o Phi_{Delta_k} (T) <= F_{Delta_total}(T),
                     and the same holds with the max(tau, .) floor.  Proof: each step is dominated by
                     the flow (L-F) and both are monotone in T, so the composition telescopes:
                     Phi_1(Phi_2(x)) <= F_1(Phi_2(x)) <= F_1(F_2(x)) = F_{1+2}(x), etc.
                     => **the ODE flow is a partition-INDEPENDENT upper bound** over every way the
                     environment may split the advance.
  P5   inter-door: let Theta_z be the unique t with F_t(tau_z + delta_z) = tau_z, i.e.
                     Theta_z = (1/h_z) * ln((tau_z + delta_z - T_inf_z) / (tau_z - T_inf_z)),
                     T_inf_z = A - c_z/h_z.
                   If the state at dispatch is tau_z and every clock interval between two
                   consecutive door events on zone z of the same vehicle is at least Theta_z, then
                   every door starts at exactly tau_z and ends at exactly tau_z + delta_z: the band
                   [tau_z, tau_z + delta_z] holds at every door event and accumulation is impossible.
                   **The conclusion does not depend on how the environment partitions the interval.**

WHY THE OLD FORMULA IS NOT JUST IMPRECISE BUT UNSOUND FOR THIS OBJECT
=====================================================================
`T_single` (the return time of ONE unsplit segment) is strictly SMALLER than Theta_z (measured:
0.054271 / 0.052458 / 0.041851 h against 0.055249 / 0.053127 / 0.042147 h).  So for a total interval
in the window `T_single <= Delta_total < Theta_z` the unsplit formula says "already back on the
floor" while a split composition is still above the floor, and the next door then lands ABOVE
tau + delta.  The window is narrow (3.5 / 2.4 / 1.1 seconds) but it exists, which is exactly why the
certificate has to name the segmentation it is talking about.

WHAT THIS MODULE DOES NOT DO
============================
  * it does not modify, instrument or re-run the environment: it consumes the environment's own
    recorded service events (`VehicleTrace.services`), read-only;
  * it makes no claim about the intra-segment (continuous) profile -- the executor defines none;
  * it emits no production receipt (`production_consumable` hard-coded False).
"""
from __future__ import annotations

import math
from fractions import Fraction

import q3_strict_float_experimental as SF

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_EVENT_SEGMENTATION_EXPERIMENTAL_V1'
SCHEMA = 'q3_event_segmentation_experimental'

REFUSED = 'REFUSED'
NOT_CERTIFIABLE = 'NOT_CERTIFIABLE'
DIAGNOSTIC_ONLY = 'DIAGNOSTIC_ONLY'
PROVED_CONDITIONAL = 'PROVED_CONDITIONAL'

#: Every entry is a fact about `scripts/simulation/strict_online_env.py` / `coldchain_state.py`,
#: with the line numbers as of the frozen copy.  Re-verify them before trusting this module.
ENV_SEGMENTATION_FACTS = (
    ('F-1', 'scripts/simulation/strict_online_env.py:209-230',
     '_advance_coldchain_to(v, clock) advances the cold chain from coldchain_updated_time to clock '
     'with served_customer=None, i.e. as a DOOR-LESS segment whose duration is travel + waiting + '
     'service.  It returns early when end <= start + 1e-9 (a no-op, not a split).'),
    ('F-2', 'scripts/simulation/strict_online_env.py:318-331',
     'The door is applied in a SEPARATE transition_segment call with '
     'depart_time == arrival_time == service_finish == clock, so its duration_h is exactly 0.'),
    ('F-3', 'scripts/coldchain/coldchain_state.py:361-362',
     'duration_h <= _EPS short-circuits _advance_temperatures to the identity, so the door segment '
     'contributes exactly +delta and no cooling.'),
    ('F-4', 'scripts/simulation/strict_online_env.py:332',
     'the door call sets coldchain_updated_time = clock, so the next door-less advance starts exactly '
     'at the door clock; consecutive door intervals therefore partition the elapsed clock time.'),
    ('F-5', 'scripts/simulation/strict_online_env.py:488-510',
     'the next event clock is min(next reveal, booking horizon, every committed_finish / return '
     'finish), and _advance_fleet is called once per iteration, so a service door fires at a clock '
     'EQUAL to its committed_finish (no earlier candidate can be smaller without being the minimum).'),
    ('F-6', 'scripts/simulation/strict_online_env.py:266, 287-289, 370',
     'depart = max(ready_time, clock); arrive = depart + travel; service_start = max(arrive, '
     'tw_start); finish = service_start + service_time, and ready_time is set to committed_finish '
     'after a service completes (:370), hence finish_n - clock_{n-1} >= service_time_n.'),
    ('F-7', 'scripts/simulation/strict_online_env.py:386-402',
     'the depot return is ALSO a zero-duration transition_segment (return_to_depot=True) and opens '
     'no door; it therefore contributes no cooling at all and is irrelevant to the band condition.'),
    ('F-8', 'scripts/simulation/d3_geo_adapter.py / m1b_thermal_gate.py:276-296',
     'the M1-b thermal replay uses ONE segment per service with duration service_finish - depart. '
     'That is a different partition of the same elapsed time, so it is a different trajectory '
     '(L-G direction: the replay is COOLER, i.e. optimistically biased).'),
)


# ---------------------------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------------------------
def t_infinity(z):
    """The ODE fixed point A - c/h (NOT the fixed point of the executor's update map)."""
    return z.ambient_c - z.cooling_rate_c_per_hour / z.heat_transfer_per_hour


def ode_flow(z, t0, total_h):
    """Exact solution of dT/dt = h(A - T) - c from `t0` over `total_h` hours."""
    ti = t_infinity(z)
    return ti + (float(t0) - ti) * math.exp(-z.heat_transfer_per_hour * float(total_h))


def theta_ode(z):
    """P5 threshold: the ODE return time from tau+delta back to tau (exact closed form)."""
    ti = t_infinity(z)
    num = z.target_c + float(z.delta_real) - ti
    den = z.target_c - ti
    if den <= 0 or num <= 0:
        raise SF.Refusal('ODE_THRESHOLD_UNDEFINED', {'zone': z.zone, 'num': num, 'den': den})
    return math.log(num / den) / z.heat_transfer_per_hour


def delta_star_closed(z):
    """The pre-fix Delta* = delta / (c - h*(A-tau)^+)  -- a LINEARISED lower bound on the return time.

    Kept for comparison only: it is a bound on the SLOPE-derived return time of ONE segment, not a
    bound for a partitioned advance.
    """
    m = z.c - z.h * max(Fraction(0), z.A - z.tau)
    if m <= 0:
        raise SF.Refusal('DRIFT_CONDITION_VIOLATED', {'zone': z.zone})
    return float(z.delta_real / m)


def t_single(contract, z, iters=200):
    """Return time of ONE unsplit executor segment from tau+delta (numeric; DIAGNOSTIC only).

    This is the quantity the previous round's plan-layer check effectively used.  It is strictly
    smaller than `theta_ode(z)`, which is why applying it to a *total* inter-door interval is
    unsound.
    """
    import coldchain_state as cs
    t0 = z.target_c + float(z.delta_real)

    def floored(total):
        temps = [float(t) for t in contract.thermal.target_temperature_c]
        temps[z.zone] = t0
        temps, _e = cs._advance_temperatures(tuple(temps), float(total), (True,) * len(temps),
                                             contract)
        return temps[z.zone]

    lo, hi = 0.0, 1.0
    if floored(hi) > z.target_c:                                          # pragma: no cover
        raise SF.Refusal('SINGLE_SEGMENT_NEVER_RETURNS', {'zone': z.zone})
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if floored(mid) > z.target_c:
            lo = mid
        else:
            hi = mid
    return hi


def thresholds(contract):
    """Per-zone P5 thresholds plus the comparison numbers the report quotes."""
    out = {}
    for z in SF.zone_thermals_exact(contract):
        out[z.zone] = {
            't_infinity': t_infinity(z),
            'theta_ode_h': theta_ode(z),
            'delta_star_closed_h': delta_star_closed(z),
            't_single_h': t_single(contract, z),
            'unsound_window_h': theta_ode(z) - t_single(contract, z),
            'theta_min': theta_ode(z) * 60.0,
            'tau': z.target_c,
            'delta': float(z.delta_real),
        }
    return out


def _finite(x):
    try:
        v = float(x)
    except Exception:                                                     # noqa: BLE001
        return False
    return v == v and v not in (float('inf'), float('-inf'))


# ---------------------------------------------------------------------------------------------
# the audit: consume the environment's own recorded service events
# ---------------------------------------------------------------------------------------------
def event_stream_from_traces(traces, env, inst_idx=0):
    """Extract, per vehicle, the ordered door events from `VehicleTrace.services` (read-only).

    `ServiceRecord.service_finish` IS the door clock (fact F-5), so no instrumentation is needed and
    the environment is not touched.
    """
    out = {}
    for tr in traces:
        rows = []
        for rec in getattr(tr, 'services', ()):
            node = int(rec.node)
            rows.append({'node': node,
                         'door_clock': float(rec.service_finish),
                         'zone': int(env.temp_class[inst_idx, node]),
                         'service_time_h': float(env.service_time[inst_idx, node]),
                         'arrive': float(rec.arrival_time),
                         'depart': float(rec.depart_time),
                         'temperature_after_c': [float(x) for x in
                                                 rec.coldchain_after.compartment_temperature_c],
                         'temperature_before_c': [float(x) for x in
                                                  rec.coldchain_before.compartment_temperature_c]})
        if rows:
            out[int(tr.vehicle_id)] = rows
    return out


def check_event_stream(*, contract, contract_identity, event_stream, state_identity=None,
                       expected_state_identity=None, dispatch_state_is_target=True,
                       active_zone_mask=None):
    """Check P5 on a recorded event stream.  Read-only, fail-closed, non-production.

    The conditions mirror `check_plan_layer` but on the CORRECT object:
      * door segment duration is 0 by construction (F-2/F-3) -> the pre-fix P3 segment condition is
        reported as INAPPLICABLE rather than silently applied to the wrong quantity;
      * per zone, every consecutive door interval must be >= Theta_z (P5);
      * the provable lower bound `interval >= service_time(later pickup)` (F-6) is evaluated too, so
        the receipt says whether the service times alone already discharge the condition;
      * same-clock or backwards door pairs are NOT_CERTIFIABLE / REFUSED.
    """
    refusals, notes, per_zone, per_vehicle = [], [], {}, {}
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
    zones = SF.zone_thermals_exact(contract) if contract is not None else ()
    n_zones = len(zones)
    mask = ((True,) * n_zones if active_zone_mask is None
            else tuple(bool(x) for x in active_zone_mask))
    if len(mask) != n_zones:
        refusals.append({'code': 'ZONE_MASK_LENGTH_MISMATCH',
                         'detail': {'mask': len(mask), 'zones': n_zones}})
    elif not all(mask):
        refusals.append({'code': 'ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN',
                         'detail': {'mask': list(mask)}})
    if not dispatch_state_is_target:
        refusals.append({'code': 'DISPATCH_STATE_NOT_TARGET',
                         'detail': {'why': 'the induction base of P5 needs T = tau at dispatch '
                                           '(coldchain_state.py:201-207 -> dispatch_vehicle)'}})
    th = thresholds(contract) if contract else {}
    for z in zones:
        per_zone[z.zone] = {'theta_ode_h': th[z.zone]['theta_ode_h'],
                            'theta_ode_min': th[z.zone]['theta_min'],
                            'delta_star_closed_h': th[z.zone]['delta_star_closed_h'],
                            't_single_h': th[z.zone]['t_single_h'],
                            'unsound_window_h': th[z.zone]['unsound_window_h'],
                            'intervals': [], 'n_doors': 0, 'n_intervals': 0,
                            'min_interval_h': None, 'min_interval_min': None,
                            'all_intervals_at_least_theta': None,
                            'short_intervals': []}

    if not isinstance(event_stream, dict) or not event_stream:
        refusals.append({'code': 'MISSING_EVENT_STREAM'})
        event_stream = {}
    for vid in sorted(event_stream):
        rows = event_stream[vid]
        if not isinstance(rows, (list, tuple)) or not rows:
            refusals.append({'code': 'MALFORMED_VEHICLE_STREAM', 'detail': {'vehicle': str(vid)}})
            continue
        clocks, prev = [], None
        for i, r in enumerate(rows):
            clk = r.get('door_clock')
            svc = r.get('service_time_h')
            zc = r.get('zone')
            if not _finite(clk) or not _finite(svc) or zc is None:
                refusals.append({'code': 'MALFORMED_EVENT',
                                 'detail': {'vehicle': str(vid), 'index': i}})
                continue
            if int(zc) < 0 or int(zc) >= n_zones:
                refusals.append({'code': 'TEMP_CLASS_OUT_OF_RANGE',
                                 'detail': {'vehicle': str(vid), 'temp_class': zc}})
                continue
            if prev is not None and clk < prev - 1e-9:
                refusals.append({'code': 'CLOCK_MOVED_BACKWARDS',
                                 'detail': {'vehicle': str(vid), 'index': i,
                                            'prev': prev, 'now': float(clk)}})
            prev = float(clk)
            clocks.append({'index': i, 'node': int(r.get('node', -1)), 'zone': int(zc),
                           'clock': float(clk), 'service_time_h': float(svc)})
        by_zone = {}
        for c in clocks:
            by_zone.setdefault(c['zone'], []).append(c)
        veh_rows = {}
        for zc, seq in sorted(by_zone.items()):
            info = per_zone[zc]
            info['n_doors'] += len(seq)
            gaps = []
            for a, b in zip(seq, seq[1:]):
                gap = b['clock'] - a['clock']
                need = info['theta_ode_h']
                gaps.append({'from_node': a['node'], 'to_node': b['node'],
                             'from_clock': a['clock'], 'to_clock': b['clock'],
                             'gap_h': gap, 'gap_min': gap * 60.0,
                             'theta_h': need,
                             'gap_at_least_theta': gap >= need,
                             'later_service_time_h': b['service_time_h'],
                             'gap_at_least_later_service_time':
                                 gap >= b['service_time_h'] - 1e-12,
                             'gap_minus_theta_h': gap - need})
            if gaps:
                info['intervals'].extend(gaps)
                info['n_intervals'] += len(gaps)
                short = [g for g in gaps if not g['gap_at_least_theta']]
                info['short_intervals'].extend(short)
                mn = min(g['gap_h'] for g in gaps)
                info['min_interval_h'] = (mn if info['min_interval_h'] is None
                                          else min(info['min_interval_h'], mn))
                info['min_interval_min'] = info['min_interval_h'] * 60.0
            veh_rows['zone%d' % zc] = {'n_doors': len(seq), 'n_intervals': len(gaps),
                                       'first_clock': seq[0]['clock'] if seq else None,
                                       'last_clock': seq[-1]['clock'] if seq else None,
                                       'min_gap_h': min([g['gap_h'] for g in gaps]) if gaps else None}
        per_vehicle[vid] = veh_rows
    for zc, info in per_zone.items():
        info['all_intervals_at_least_theta'] = not info['short_intervals']
        info['service_times_imply_theta'] = all(
            g['gap_at_least_later_service_time'] for g in info['intervals'])
    bad = [zc for zc, i in per_zone.items() if i['short_intervals']]
    total_intervals = sum(i['n_intervals'] for i in per_zone.values())
    if refusals:
        verdict = REFUSED
    elif bad:
        verdict = NOT_CERTIFIABLE
    elif notes:
        verdict = DIAGNOSTIC_ONLY
    else:
        verdict = PROVED_CONDITIONAL
    return {
        'schema': SCHEMA,
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'verdict': verdict,
        'experimental_verdict': verdict,
        'contract_hash': contract_hash,
        'refusals': refusals,
        'binding_notes': list(notes),
        'per_zone': {str(k): v for k, v in sorted(per_zone.items())},
        'per_vehicle': {str(k): v for k, v in sorted(per_vehicle.items())},
        'summary': {'n_vehicles': len(per_vehicle),
                    'n_door_events': sum(i['n_doors'] for i in per_zone.values()),
                    'n_intervals': total_intervals,
                    'bad_zones': bad,
                    'min_interval_h': min([i['min_interval_h'] for i in per_zone.values()
                                           if i['min_interval_h'] is not None] or [None]),
                    'min_theta_h': min([i['theta_ode_h'] for i in per_zone.values()] or [None]),
                    'door_segment_duration_h': 0.0,
                    'pre_fix_p3_segment_condition_applicable': False},
        'proposition': ("P5: between two consecutive door events on the same zone the elapsed clock "
                        "interval >= Theta_z; the executor's partition of that interval is "
                        "irrelevant (L-G)"),
        'claim': ('every door event on every zone starts at exactly tau_z and ends at exactly '
                  'tau_z + delta_z, so the band [tau_z, tau_z + delta_z] holds at every door event'),
        'not_claimed': [
            'that the stored FLOATS equal the real values: this is an R-level statement (see '
            'q3_strict_float_experimental for the F-level transfer, which needs kappa or S3)',
            'that the intra-segment (continuous) profile is inside the bounds -- the executor defines '
            'no intra-segment trajectory at all',
            'that the plan is feasible, profitable or installable',
            'that the result transfers to instances, node counts or data families not checked'],
        'env_segmentation_facts': [{'id': a, 'where': b, 'fact': c}
                                   for a, b, c in ENV_SEGMENTATION_FACTS],
    }


# ---------------------------------------------------------------------------------------------
# diagnostics that show WHY the old mapping cannot be reused
# ---------------------------------------------------------------------------------------------
def split_divergence(contract, zone_index, total_h, pieces):
    """Executor states after 1 vs `pieces` splits of the same total door-less duration."""
    import coldchain_state as cs
    z = [x for x in SF.zone_thermals_exact(contract) if x.zone == int(zone_index)][0]
    t0 = z.target_c + float(z.delta_real)

    def run(parts):
        temps = [float(t) for t in contract.thermal.target_temperature_c]
        temps[z.zone] = t0
        for p in parts:
            temps, _e = cs._advance_temperatures(tuple(temps), float(p), (True,) * len(temps),
                                                contract)
        return temps[z.zone]

    whole = run([total_h])
    split = run([total_h / pieces] * pieces)
    return {'zone': z.zone, 'total_h': float(total_h), 'pieces': int(pieces),
            'single_c': whole, 'split_c': split,
            'split_minus_single_c': split - whole,
            'tau': z.target_c, 'tau_plus_delta': z.target_c + float(z.delta_real),
            'ode_upper_c': ode_flow(z, t0, total_h),
            'reading': 'split >= single: the environment (which splits at event clocks) is WARMER '
                       'than an unsplit replay of the same plan'}


def old_p3_on_env_segments(contract):
    """What P3' says when it is applied to the environment's OWN segment sequence.

    The door segment duration is 0 (F-2/F-3), so every door fails `Delta >= Delta*_z`, i.e. the plan
    is NOT_CERTIFIABLE regardless of its service times.  This is the concrete form of the mismatch.
    """
    zones = SF.zone_thermals_exact(contract)
    rows = []
    for z in zones:
        need = float(z.return_time_h)
        rows.append({'zone': z.zone, 'door_segment_duration_h': 0.0,
                     'delta_star_closed_h': need,
                     'predicate_duration_ge_delta_star': 0.0 >= need,
                     'zero_duration_is_below_eps': True,
                     'consequence': 'NOT_CERTIFIABLE for every door, whatever the service times'})
    return {'rows': rows,
            'verdict_if_p3_applied_literally': NOT_CERTIFIABLE,
            'why': 'the environment applies the door in a zero-duration segment; the cooling happens '
                   'in the door-less advance that precedes it (F-1..F-3)'}
