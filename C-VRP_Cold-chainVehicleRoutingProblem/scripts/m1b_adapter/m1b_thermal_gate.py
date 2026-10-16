#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-6: the G7 THERMAL HARD GATE by state-anchored REPLAY.

Why: `action_contract.certify_route:179-183` checks only that the temperature CLASS is supported
and never inspects `hard_temperature_bounds_c`; `coldchain_state._thermal_violations:433-447`
only COUNTS violations.  The real verdict lives in the v1 evaluator plus `hard_gate`, i.e. at the
TERMINAL layer, not before submission.  This module supplies the missing PRE-SUBMIT check.

Method: start from the trusted vehicle's CURRENT cold-chain state and replay the candidate suffix
through the contract's REAL thermal dynamics (`coldchain_state.transition_segment`), recording the
temperature ENVELOPE AT EVERY SEGMENT BOUNDARY and after every door event -- not merely the
endpoint.  That directly addresses the "endpoint only" hazard: because `_apply_door_event` warms a
zone instantaneously and the following segment cools it, a peak can in principle occur strictly
inside a segment, so the check evaluates the post-door value as well as both segment endpoints.

Fail-closed on: missing cold-chain state, time-order violation, incompatible contract, any
non-finite value, and any EXISTING accumulated thermal violation on the vehicle.

HONEST LIMIT (kept as an explicit qualification gap, not hidden): the replay samples the
temperature at event boundaries and immediately after each door event.  Between two consecutive
events the code applies cooling linearly and heat transfer exponentially, and `updated =
max(target, ...)`, so the composed trajectory between two sampled points is not re-sampled
continuously.  The module reports `continuous_time_hard_constraint: NOT_VERIFIED_CONTINUOUSLY`
and states the reason, rather than claiming a continuous-time guarantee.
"""
from __future__ import annotations

THERMAL_HARD_OK = 'THERMAL_HARD_OK'
THERMAL_BOUNDS_VIOLATED = 'THERMAL_BOUNDS_VIOLATED'
THERMAL_REPLAY_STATE_MISSING = 'THERMAL_REPLAY_STATE_MISSING'
THERMAL_REPLAY_TIME_ORDER_INVALID = 'THERMAL_REPLAY_TIME_ORDER_INVALID'
THERMAL_PRIOR_VIOLATIONS_PRESENT = 'THERMAL_PRIOR_VIOLATIONS_PRESENT'
THERMAL_CONTRACT_INCOMPATIBLE = 'THERMAL_CONTRACT_INCOMPATIBLE'
THERMAL_REPLAY_INCOMPLETE = 'THERMAL_REPLAY_INCOMPLETE'
THERMAL_NUMERIC_INVALID = 'THERMAL_NUMERIC_INVALID'

CONTINUOUS_NOT_VERIFIED = 'NOT_VERIFIED_CONTINUOUSLY'


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


def replay_thermal_envelope(snapshot, *, contract, inst_idx, clock, travel_fn, distance_fn,
                            temp_class_of, quantity_of, service_time_of,
                            active_zone_mask=None):
    """Replay ONE vehicle's remaining trajectory and return its temperature envelope.

    Returns (envelope, verdict, detail).  `envelope` contains per-zone min/max over every sampled
    point, the sampled points themselves, and the number of door events.
    """
    import coldchain_state as CS
    from coldchain_state import transition_segment

    if snapshot is None:
        return None, THERMAL_REPLAY_STATE_MISSING, {'field': 'snapshot'}
    if contract is None:
        return None, THERMAL_CONTRACT_INCOMPATIBLE, {'field': 'contract'}
    th = getattr(contract, 'thermal', None)
    if th is None:
        return None, THERMAL_CONTRACT_INCOMPATIBLE, {'field': 'contract.thermal'}
    bounds = getattr(th, 'hard_temperature_bounds_c', None)
    if not bounds:
        return None, THERMAL_CONTRACT_INCOMPATIBLE, {'field': 'thermal.hard_temperature_bounds_c'}
    zones = len(th.supported_temp_classes)
    mask = tuple(bool(x) for x in (active_zone_mask if active_zone_mask is not None
                                   else (True,) * zones))
    if len(mask) != zones:
        return None, THERMAL_CONTRACT_INCOMPATIBLE, {'why': 'active_zone_mask length mismatch'}

    state = snapshot.coldchain_state
    if state is None:
        return None, THERMAL_REPLAY_STATE_MISSING, {
            'vehicle': snapshot.vehicle_id,
            'why': 'the vehicle has no cold-chain state; it was never dispatched. Temperature '
                   'hard feasibility cannot be asserted for a state that does not exist.'}
    if bool(getattr(state, 'closed', False)):
        return None, THERMAL_REPLAY_STATE_MISSING, {
            'vehicle': snapshot.vehicle_id, 'why': 'cold-chain state already closed'}
    prior_count = int(getattr(state, 'thermal_violation_count', 0))
    prior_dur = float(getattr(state, 'thermal_violation_duration_h', 0.0))
    if prior_count > 0 or prior_dur > 1e-9:
        # A replay of the REMAINING trajectory cannot exonerate violations already accumulated.
        return None, THERMAL_PRIOR_VIOLATIONS_PRESENT, {
            'vehicle': snapshot.vehicle_id, 'thermal_violation_count': prior_count,
            'thermal_violation_duration_h': prior_dur,
            'why': 'RESTARTING a replay from a state that already carries violations would '
                   'silently launder them; the cumulative record is part of the verdict'}

    mins = [float('inf')] * zones
    maxs = [float('-inf')] * zones
    samples = []

    def _observe(tag, when, st):
        nonlocal mins, maxs
        for z, T in enumerate(st.compartment_temperature_c):
            if not _finite(T):
                raise ValueError('non-finite temperature zone %d' % z)
            mins[z] = min(mins[z], float(T))
            maxs[z] = max(maxs[z], float(T))
        samples.append({'tag': tag, 'time': float(when),
                        'temps': [float(x) for x in st.compartment_temperature_c]})

    plan = snapshot.plan if snapshot.plan is not None else None
    suffix = [int(x) for x in (plan.suffix if plan is not None else snapshot.mutable_suffix)]
    suffix = [x for x in suffix if x != 0]
    node = int(snapshot.current_node)
    t = float(max(snapshot.ready_time, clock))
    _observe('initial', t, state)

    n_doors = 0
    # PRE-BOARDED LOTS.  The replay covers the trajectory from the CURRENT decision point onward,
    # so lots already in the manifest were acquired earlier and carry their own recorded
    # `pickup_finish_time`.  A lot whose recorded pickup is at or after the replay start cannot
    # have come from the past, so the state and the plan are inconsistent -- refuse rather than
    # invent an acquisition time for it.
    for lot in (getattr(state, 'cargo_manifest', ()) or ()):
        pf = getattr(lot, 'pickup_finish_time', None)
        if pf is None or not _finite(pf):
            return None, THERMAL_REPLAY_INCOMPLETE, {
                'vehicle': snapshot.vehicle_id, 'order_id': getattr(lot, 'order_id', None),
                'why': 'pre-boarded lot without a finite pickup_finish_time'}
        if float(pf) > t + 1e-9:
            return None, THERMAL_REPLAY_INCOMPLETE, {
                'vehicle': snapshot.vehicle_id, 'order_id': getattr(lot, 'order_id', None),
                'why': 'a pre-boarded lot records a pickup LATER than the replay start, so the '
                       'state and the plan are inconsistent', 'pickup_finish_time': float(pf),
                'replay_start': t}
    try:
        for nxt in suffix:
            if int(nxt) <= 0:
                return None, THERMAL_REPLAY_INCOMPLETE, {'why': 'non-customer node in suffix',
                                                         'node': int(nxt)}
            travel = float(travel_fn(node, int(nxt)))
            arrive = t + travel
            if not _finite(arrive) or arrive < t - 1e-9:
                return None, THERMAL_REPLAY_TIME_ORDER_INVALID, {
                    'vehicle': snapshot.vehicle_id, 'depart': t, 'arrive': arrive}
            st = float(service_time_of(int(nxt)))
            finish = max(arrive, 0.0) + st
            if finish < arrive - 1e-9:
                return None, THERMAL_REPLAY_TIME_ORDER_INVALID, {
                    'vehicle': snapshot.vehicle_id, 'arrive': arrive, 'finish': finish}
            try:
                state, _m = transition_segment(
                    state, t, arrive, finish, int(nxt), mask, contract,
                    order_quantity=float(quantity_of(int(nxt))),
                    order_temp_class=int(temp_class_of(int(nxt))),
                    initial_quality=float(snapshot.initial_quality_of(int(nxt))),
                    segment_distance_units=float(distance_fn(node, int(nxt))))
            except Exception as exc:                                    # noqa: BLE001
                return None, THERMAL_REPLAY_INCOMPLETE, {
                    'vehicle': snapshot.vehicle_id, 'node': int(nxt),
                    'exception': '%s: %s' % (type(exc).__name__, str(exc)[:150])}
            n_doors += 1
            # The door event warms the opened zone; sampling here is what catches an
            # INTERIOR peak that an endpoint-only check would miss.
            _observe('after_door@%d' % int(nxt), finish, state)
            t, node = finish, int(nxt)
        # return leg
        travel_home = float(travel_fn(node, 0))
        arrive_home = t + travel_home
        if not _finite(arrive_home):
            return None, THERMAL_NUMERIC_INVALID, {'return_arrival': repr(arrive_home)}
        state, _m = transition_segment(
            state, t, arrive_home, arrive_home, None, mask, contract,
            return_to_depot=True, segment_distance_units=float(distance_fn(node, 0)))
        _observe('return_arrival', arrive_home, state)
    except ValueError as exc:                                            # noqa: BLE001
        return None, THERMAL_REPLAY_TIME_ORDER_INVALID, {
            'vehicle': snapshot.vehicle_id,
            'exception': '%s: %s' % (type(exc).__name__, str(exc)[:150])}

    violations = []
    for z in range(zones):
        lo, hi = bounds[z]
        if mins[z] < float(lo) - 1e-9 or maxs[z] > float(hi) + 1e-9:
            violations.append({'zone': z, 'bounds': [float(lo), float(hi)],
                               'observed_min': mins[z], 'observed_max': maxs[z]})
    env = {'vehicle': snapshot.vehicle_id, 'per_zone_min': mins, 'per_zone_max': maxs,
           'bounds': [[float(b[0]), float(b[1])] for b in bounds],
           'n_door_events': n_doors, 'samples': samples,
           'continuous_time_hard_constraint': CONTINUOUS_NOT_VERIFIED,
           'continuous_time_reason': (
               'sampled at every segment endpoint and immediately after every door event; the '
               'composed intra-segment trajectory (linear cooling plus exponential heat transfer, '
               'floored at target) is not resampled continuously, so no continuous-time guarantee '
               'is claimed here'),
           'post_replay_thermal_violation_count': int(getattr(state, 'thermal_violation_count', 0)),
           'post_replay_thermal_violation_duration_h': float(
               getattr(state, 'thermal_violation_duration_h', 0.0))}
    detail = {'envelope': env, 'n_violations': len(violations), 'violations': violations}
    if violations:
        return env, THERMAL_BOUNDS_VIOLATED, detail
    return env, THERMAL_HARD_OK, detail
