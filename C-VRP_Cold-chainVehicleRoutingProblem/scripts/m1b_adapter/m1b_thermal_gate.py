#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix: the G7 thermal gate, rebuilt on the SHARED time replay and split into two
CERTAINTY LEVELS.

Fixes (Q2-R): C1 (a sampled result could be read as a continuous hard certificate) and the
timing mismatch with Omega11.

CERTAINTY LEVELS (the C1 fix)
    THERMAL_SAMPLED_OK                   the SAMPLED envelope is inside hard_temperature_bounds_c
    THERMAL_CONTINUOUS_OK                the intra-segment continuous extremum is PROVEN in bounds
    THERMAL_CONTINUOUS_NOT_ESTABLISHED   not established (this round's honest standing)

The two are SEPARATE top-level fields.  The composite requires THERMAL_CONTINUOUS_OK, so with no
analytic extremum proof the composite is FAIL-CLOSED while the sampled diagnostic remains
available for engineering purposes and is NEVER named a complete installability certificate.

TIMING: the replay drives the temperature through `m1b_timing.replay_timeline`, i.e. the SAME
wait/service semantics as Omega11 and `action_contract.certify_route`.  The audited version had
its own advance (no `tw_start` wait), so the two gates could disagree about one trajectory.

HONEST LIMIT, stated rather than hidden: `THERMAL_CONTINUOUS_OK` is set only when
`continuous_extremum_proof` is supplied AND returns proven for every zone.  This module does NOT
itself implement that proof, and its mathematical applicability domain would have to be frozen
and independently audited first.
"""
from __future__ import annotations

import m1b_timing as TM

THERMAL_SAMPLED_OK = 'THERMAL_SAMPLED_OK'
THERMAL_BOUNDS_VIOLATED = 'THERMAL_BOUNDS_VIOLATED'
THERMAL_REPLAY_STATE_MISSING = 'THERMAL_REPLAY_STATE_MISSING'
THERMAL_REPLAY_TIME_ORDER_INVALID = 'THERMAL_REPLAY_TIME_ORDER_INVALID'
THERMAL_PRIOR_VIOLATIONS_PRESENT = 'THERMAL_PRIOR_VIOLATIONS_PRESENT'
THERMAL_CONTRACT_INCOMPATIBLE = 'THERMAL_CONTRACT_INCOMPATIBLE'
THERMAL_REPLAY_INCOMPLETE = 'THERMAL_REPLAY_INCOMPLETE'
THERMAL_NUMERIC_INVALID = 'THERMAL_NUMERIC_INVALID'

THERMAL_CONTINUOUS_OK = 'THERMAL_CONTINUOUS_OK'
THERMAL_CONTINUOUS_NOT_ESTABLISHED = 'THERMAL_CONTINUOUS_NOT_ESTABLISHED'
THERMAL_CONTINUOUS_VIOLATED = 'THERMAL_CONTINUOUS_VIOLATED'

#: PRE-FIX name.  In the audited version a single `THERMAL_HARD_OK` conflated the sampled and
#: continuous levels, which is exactly the C1 promotion path.  The name is kept ONLY so the
#: historical `test_m1b_q2_certification.py` can be imported; it now denotes the SAMPLED level and
#: must never be read as a continuous-time guarantee.
THERMAL_HARD_OK = THERMAL_SAMPLED_OK

#: PRE-FIX label for the continuous-time caveat.  Kept for import compatibility with the
#: historical test file; `THERMAL_CONTINUOUS_NOT_ESTABLISHED` is the authoritative spelling.
CONTINUOUS_NOT_VERIFIED = THERMAL_CONTINUOUS_NOT_ESTABLISHED

IMPLEMENTATION_VERSION = 'M1B_Q2TRUSTFIX'

#: F-T4: a vehicle for which a thermal replay is not merely unavailable but MEANINGLESS.
THERMAL_NOT_APPLICABLE = 'THERMAL_NOT_APPLICABLE'


def classify_thermal_applicability(snapshot, *, clock, closed_return_history=None):
    """F-T4 (Q2-TrustFix).  Decide whether a thermal replay APPLIES to this vehicle.

    Q2-Fix-T T4-2/T4-3 found that a vehicle which was NEVER DISPATCHED (so it has no cold-chain
    state) made `replay_thermal_envelope` return `THERMAL_REPLAY_STATE_MISSING`, and because the
    certifier required EVERY vehicle to pass, ONE such vehicle refused the whole fleet.  Real
    multi-vehicle days always contain vehicles that are never dispatched, so that was a
    systematic FALSE_REJECT.

    Returns (applicability, verdict, detail) where applicability is True (replay required),
    False (NOT_APPLICABLE) or None (REFUSAL).

    NOT_APPLICABLE requires ALL of: never dispatched AND idle AND at the depot AND no cargo AND
    no unfinished commitment AND no candidate suffix AND no thermal state to keep satisfying.
    Anything holding cargo, a commitment or planned work REFUSES when its state is missing --
    the original "cargo without state must be refused" negative control is preserved.
    """
    if snapshot is None:
        return None, THERMAL_REPLAY_STATE_MISSING, {'field': 'snapshot'}
    hist = dict(closed_return_history or {})
    cc = snapshot.coldchain_state
    status = snapshot.status
    at_depot = int(snapshot.current_node) == 0
    plan = getattr(snapshot, 'plan', None)
    suffix = [int(x) for x in (plan.suffix if plan is not None else snapshot.mutable_suffix)]
    suffix = [x for x in suffix if x != 0]
    has_cargo = bool(getattr(cc, 'cargo_manifest', ()) or ()) if cc is not None else False
    has_commit = snapshot.committed_next is not None or snapshot.committed_finish is not None
    if cc is not None:
        has_cargo = has_cargo or float(getattr(cc, 'total_load', 0.0) or 0.0) > 1e-9
    facts = {'status': status, 'current_node': int(snapshot.current_node),
             'at_depot': at_depot, 'has_cc_state': cc is not None, 'has_cargo': has_cargo,
             'has_commitment': has_commit, 'n_suffix_stops': len(suffix),
             'dispatched': bool(getattr(cc, 'dispatched', False)) if cc is not None else False}

    # ---- refusals first: something must be kept thermally sound but there is no state ----
    if cc is None:
        if has_cargo:
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'the vehicle holds cargo but has no cold-chain state'}
        if has_commit:
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'the vehicle has an unfinished commitment but no cold-chain state'}
        if suffix or int(snapshot.current_node) != 0:
            # a candidate that dispatches an un-dispatched vehicle is NOT not-applicable
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'the candidate plan sends an un-dispatched vehicle out, so a '
                                'thermal replay IS required but no state exists'}
        if status in ('closed',):
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'a closed vehicle with no cold-chain state cannot be audited'}
        if status in ('returning', 'committed'):
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'an in-flight vehicle must have a cold-chain state'}
        if status == 'idle' and at_depot:
            return False, THERMAL_NOT_APPLICABLE, {
                **facts, 'why': 'never dispatched, idle at the depot, no cargo, no commitment, '
                                'no candidate work and no thermal state to keep satisfying'}
        return None, THERMAL_REPLAY_STATE_MISSING, {
            **facts, 'why': 'no cold-chain state and the not-applicable conditions are not all met'}

    # ---- state present ----
    if status == 'closed':
        if has_cargo:
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'a closed vehicle still holding cargo cannot be exonerated'}
        if has_commit or suffix:
            return None, THERMAL_REPLAY_STATE_MISSING, {
                **facts, 'why': 'a closed vehicle with unfinished work is inconsistent'}
        if snapshot.vehicle_id not in hist:
            # recorded rather than silently passed: the historical return is not auditable here
            return False, THERMAL_NOT_APPLICABLE, {
                **facts, 'why': 'closed vehicle with no pending work: no thermal replay applies; '
                                'its historical return is audited by the Omega11 closed-history '
                                'gate, not by a replay'}
        return False, THERMAL_NOT_APPLICABLE, {
            **facts, 'why': 'closed vehicle with no pending work; history audited elsewhere'}
    if int(snapshot.current_node) == 0 and not suffix and not has_cargo and not has_commit:
        return False, THERMAL_NOT_APPLICABLE, {
            **facts, 'why': 'nothing in cargo and nothing planned, so there is no remaining '
                            'trajectory to replay'}
    return True, 'THERMAL_REPLAY_REQUIRED', facts


def _finite(x):
    try:
        f = float(x)
    except Exception:                                                   # noqa: BLE001
        return False
    return f == f and f not in (float('inf'), float('-inf'))


def replay_thermal_envelope(snapshot, *, contract, inst_idx, clock, travel_fn, distance_fn,
                            temp_class_of, quantity_of, service_time_of, active_zone_mask=None,
                            stops_override=None, start_node_override=None,
                            depart_override=None, initial_quality_of=None,
                            continuous_extremum_proof=None, allow_test_only_proof=False,
                            wait_aware=True):
    """Replay a vehicle's remaining trajectory and report its temperature envelope.

    Returns (envelope, verdict, detail).  `verdict` is the SAMPLED verdict; the envelope carries
    `continuous_time_hard_constraint` with one of the THERMAL_CONTINUOUS_* values.
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
            'why': 'the vehicle has no cold-chain state; it was never dispatched, so temperature '
                   'hard feasibility cannot be asserted for a state that does not exist'}
    if bool(getattr(state, 'closed', False)):
        return None, THERMAL_REPLAY_STATE_MISSING, {
            'vehicle': snapshot.vehicle_id, 'why': 'cold-chain state already closed'}
    prior_count = int(getattr(state, 'thermal_violation_count', 0))
    prior_dur = float(getattr(state, 'thermal_violation_duration_h', 0.0))
    if prior_count > 0 or prior_dur > 1e-9:
        return None, THERMAL_PRIOR_VIOLATIONS_PRESENT, {
            'vehicle': snapshot.vehicle_id, 'thermal_violation_count': prior_count,
            'thermal_violation_duration_h': prior_dur,
            'why': 'restarting from a state that already carries violations would launder them'}

    plan = getattr(snapshot, 'plan', None)
    if stops_override is not None:
        stops = [int(x) for x in stops_override if int(x) != 0]
    else:
        stops = [int(x) for x in (plan.suffix if plan is not None else snapshot.mutable_suffix)]
        stops = [x for x in stops if x != 0]
    node = int(snapshot.current_node if start_node_override is None else start_node_override)
    depart = float(max(snapshot.ready_time, clock) if depart_override is None
                   else depart_override)

    # F-T2 (Q2-TrustFix).  The legacy auto-wrap is GONE.  The pre-fix version accepted a bare
    # `service_time_of` callable and synthesised `tw_start = 0.0` for every named node, which is
    # the same optimistic reconstruction T2-1 recorded on the Omega11 gate.  A caller must now
    # supply a real per-instance time-input object; anything else REFUSES.
    times = service_time_of
    if not (hasattr(times, 'tw_start_of') and hasattr(times, 'service_time_of')):
        return None, THERMAL_REPLAY_INCOMPLETE, {
            'vehicle': snapshot.vehicle_id,
            'why': 'a per-instance time-input object is required (needs tw_start_of and '
                   'service_time_of); a bare callable cannot express a time-window start and is '
                   'therefore refused rather than wrapped with tw_start = 0',
            'implementation_version': IMPLEMENTATION_VERSION}

    # ---- shared timeline: identical wait/service semantics to Omega11 and certify_route ----
    timeline, tverd, tdet = TM.replay_timeline(
        depart=depart, clock=clock, start_node=node, stops=stops, return_to_depot=True,
        travel_fn=travel_fn, times=times)
    if tverd != TM.TIMING_OK:
        return None, (THERMAL_REPLAY_TIME_ORDER_INVALID if tverd in
                      (TM.TIMING_TIME_ORDER_INVALID, TM.TIMING_ANCHOR_MISMATCH)
                      else THERMAL_REPLAY_INCOMPLETE), {'vehicle': snapshot.vehicle_id,
                                                        'timing_verdict': tverd, 'detail': tdet}

    mins = [float('inf')] * zones
    maxs = [float('-inf')] * zones
    samples, segment_durations = [], []

    def _observe(tag, when, st):
        for z, T in enumerate(st.compartment_temperature_c):
            if not _finite(T):
                raise ValueError('non-finite temperature zone %d' % z)
            mins[z] = min(mins[z], float(T))
            maxs[z] = max(maxs[z], float(T))
        samples.append({'tag': tag, 'time': float(when),
                        'temps': [float(x) for x in st.compartment_temperature_c]})

    for lot in (getattr(state, 'cargo_manifest', ()) or ()):
        pf = getattr(lot, 'pickup_finish_time', None)
        if pf is None or not _finite(pf):
            return None, THERMAL_REPLAY_INCOMPLETE, {
                'vehicle': snapshot.vehicle_id, 'order_id': getattr(lot, 'order_id', None),
                'why': 'pre-boarded lot without a finite pickup_finish_time'}
        if float(pf) > depart + 1e-9:
            return None, THERMAL_REPLAY_INCOMPLETE, {
                'vehicle': snapshot.vehicle_id, 'order_id': getattr(lot, 'order_id', None),
                'why': 'a pre-boarded lot records a pickup LATER than the replay start',
                'pickup_finish_time': float(pf), 'replay_start': depart}

    _observe('initial', depart, state)

    def _iq_of(n):
        if initial_quality_of is not None:
            return float(initial_quality_of(int(n)))
        v = getattr(snapshot, 'initial_quality_of', None)
        return float(v(int(n))) if callable(v) else 1.0

    n_doors = 0
    try:
        prev_t = depart
        for ev in timeline['events']:
            if ev['tag'] != 'service':
                continue
            nxt = int(ev['node'])
            dur = TM._finite(ev['service_finish']) and (ev['service_finish'] - ev['depart'])
            if dur is None or dur < -1e-9:
                return None, THERMAL_REPLAY_TIME_ORDER_INVALID, {
                    'vehicle': snapshot.vehicle_id, 'event': ev}
            state, _m = transition_segment(
                state, ev['depart'], ev['arrive'], ev['service_finish'], nxt, mask, contract,
                order_quantity=float(quantity_of(int(nxt))),
                order_temp_class=int(temp_class_of(int(nxt))),
                initial_quality=_iq_of(nxt),
                segment_distance_units=float(distance_fn(node, nxt)))
            n_doors += 1
            # THE interior-peak sample: the door event warms the opened zone instantaneously, so
            # the peak occurs AFTER the door, not at the segment endpoint.
            _observe('after_door@%d' % nxt, ev['service_finish'], state)
            segment_durations.append(float(ev['service_finish']) - float(ev['depart']))
            prev_t = float(ev['service_finish'])
            node = nxt
        ret = timeline.get('return_arrival')
        if ret is None:
            return None, THERMAL_REPLAY_INCOMPLETE, {'why': 'no return leg in the timeline'}
        state, _m = transition_segment(
            state, prev_t, ret, ret, None, mask, contract, return_to_depot=True,
            segment_distance_units=float(distance_fn(node, 0)))
        _observe('return_arrival', ret, state)
        segment_durations.append(float(ret) - float(prev_t))
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

    # ---- continuous level ----------------------------------------------------------------
    # F-T1 (Q2-TrustFix).  The proof branch additionally requires an EXPLICIT opt-in flag, so a
    # production caller cannot reach THERMAL_CONTINUOUS_OK even if it manages to pass a callable
    # through this low-level entry point.  Q2-Fix-T T1-1 was exactly that: an arbitrary
    # `lambda **k: (True, {})` yielded CONTINUOUS_OK and composite_pass = True.
    if violations:
        cont_state, cont_detail = THERMAL_CONTINUOUS_VIOLATED, {
            'why': 'the sampled envelope already breaches the hard bounds'}
    elif continuous_extremum_proof is None or not allow_test_only_proof:
        cont_state, cont_detail = THERMAL_CONTINUOUS_NOT_ESTABLISHED, {
            'why': 'no intra-segment continuous extremum proof is REGISTERED; an injected '
                   'callable is not accepted as one. The sampled envelope is NOT promoted to a '
                   'continuous-time guarantee.',
            'proof_hook_offered': continuous_extremum_proof is not None,
            'proof_hook_accepted': bool(continuous_extremum_proof is not None
                                        and allow_test_only_proof),
            'what_would_be_needed': 'a frozen applicability domain plus an independently audited '
                                    'analytic extremum proof for the composed intra-segment '
                                    'trajectory (linear cooling, exponential heat transfer, '
                                    'floored at target)'}
    else:
        try:
            proven, pdetail = continuous_extremum_proof(
                contract=contract, bounds=bounds, samples=samples,
                segment_durations=segment_durations, active_zone_mask=mask)
        except Exception as exc:                                        # noqa: BLE001
            proven, pdetail = False, {'exception': type(exc).__name__}
        cont_state = THERMAL_CONTINUOUS_OK if proven is True else THERMAL_CONTINUOUS_NOT_ESTABLISHED
        cont_detail = {'proof_detail': pdetail}
    env = {'vehicle': snapshot.vehicle_id, 'per_zone_min': mins, 'per_zone_max': maxs,
           'bounds': [[float(b[0]), float(b[1])] for b in bounds],
           'n_door_events': n_doors, 'samples': samples,
           'segment_durations': segment_durations,
           'timeline_events': timeline['events'],
           'continuous_time_hard_constraint': cont_state,
           'continuous_time_detail': cont_detail,
           'post_replay_thermal_violation_count': int(getattr(state,
                                                              'thermal_violation_count', 0)),
           'post_replay_thermal_violation_duration_h': float(
               getattr(state, 'thermal_violation_duration_h', 0.0)),
           'implementation_version': IMPLEMENTATION_VERSION}
    detail = {'envelope': env, 'n_violations': len(violations), 'violations': violations,
              'sampled_level': THERMAL_SAMPLED_OK if not violations else THERMAL_BOUNDS_VIOLATED,
              'continuous_level': cont_state}
    if violations:
        return env, THERMAL_BOUNDS_VIOLATED, detail
    return env, THERMAL_SAMPLED_OK, detail
