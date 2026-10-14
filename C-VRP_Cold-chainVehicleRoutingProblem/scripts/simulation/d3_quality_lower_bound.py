#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-QC: candidate ANALYTIC quality lower bound + its falsification tools (READ-ONLY).

Authorised scope (user ruling 2026-10-14): DESIGN + AUDIT ONLY.  Nothing here installs, changes
`_commit`, relaxes D8, or claims a proof.  The verdict vocabulary is
PROVEN / FALSIFIED / NOT_ESTABLISHED and a search count may NEVER upgrade it to PROVEN.

Candidate bound (user's proposed structure, made concrete against the v1 physics):

    Q_i^LB = Q_i^start * exp( -k_i(T_i^max) * tau_i^max )

with, for lot i of temperature class c:
  * `T_i^max = max(hard_temperature_bounds_c[c][1], thermal.ambient_temperature_c)`
        -- the zone's hard upper bound, widened by the ambient temperature because the
           NON-PRECOOLED branch rates against `contract.thermal.ambient_temperature_c`
           (coldchain_state.py:398-402), which can lie above the zone bounds;
  * `tau_i^max = tw_end[0] - t_now`
        -- a lot in cargo now, or loaded at any time >= now, must be delivered at the depot
           return, and a return after the depot's `tw_end` is a hard failure
           (`depot_return_violation`), so the remaining in-cargo time cannot exceed this;
  * `Q_i^start = quality_remaining` for a lot already in `cargo_manifest`, else `initial_quality`
        -- an order not yet picked up does not decay before it is loaded.

ASSUMPTIONS (each one falsifiable, see `assumptions()` and the acceptance suite):
  A1 every v1 quality update is `q <- min(q, q*exp(-k*dt))` with no other factor
     (coldchain_state.py:404-405).
  A2 on the admissible set Omega, the temperature actually driving the rate never exceeds
     `T_i^max`: the rate uses either the segment average of the lot's own zone or the ambient
     temperature, and `temperature_hard_feasible` (coldchain_evaluator.py:358) is part of Omega,
     so segment temperatures stay inside the hard bounds.
  A3 `k(c, T)` is non-decreasing in T (Arrhenius, coldchain_state.py:128-143).
  A4 the in-cargo elapsed time from `t_now` to delivery never exceeds `tau_i^max` on Omega.
  A5 `Omega` is DEFINED as the set of executions that satisfy the hard gates (thermal, TW,
     capacity, depot return, commitment).  The bound is therefore CONDITIONAL on those gates: if
     a gate is relaxed, `T^max` / `tau^max` must be re-derived -- relaxing a gate does NOT
     preserve this bound.

KNOWN WAYS THE BOUND CAN BE UNSOUND (must be falsified, not assumed away):
  F-a A door event warms the zone by `door_heat/thermal_capacity` (coldchain_state.py) and the
      warmed value might not be inspected against the hard bounds before a later segment rates
      against it.
  F-b `_force_return_all` at the horizon (`strict_online_env.py:658-664`) returns vehicles after
      `tw_end[0]`; such a trajectory leaves Omega, but if it were counted as admissible the
      `tau^max` argument collapses.
  F-c the depot TW of `tw_end[0]` is genuinely binding (it is checked by the terminal evaluator as
      `depot_return_violation`), but `tw_speed`/`travel_mat` asymmetries could make the actual
      arrival later than any bound computed from `dist_mat`.
  F-d cross-zone interaction: the rate uses the lot's OWN class zone; if a future model version
      couples zones, A2 must be re-derived.
"""
from __future__ import annotations

import math

import numpy as np

PROVEN = 'PROVEN'
FALSIFIED = 'FALSIFIED'
NOT_ESTABLISHED = 'NOT_ESTABLISHED'

BOUND_SCHEMA = 'quality_lower_bound_v1'

# LEMON -- a lemma that a TIGHTER design would have relied on, and that the 1-D sweep FALSIFIED.
# Recorded as a constant so no later document can quietly resurrect it.
FALSIFIED_LEMMAS = {
    'L-departure-monotone': {
        'claim': ('a later admissible departure never yields HIGHER delivered quality, so the worst '
                  'case over Omega-for-a-fixed-plan is the latest admissible departure'),
        'status': 'FALSIFIED',
        'measured': ('1-D admissible-departure sweep on a real decision point: the last grid point '
                     '(delay ~5.99 h) delivered quality 0.997234 while delays 0..~5.2 h all '
                     'delivered 0.997133 -- quality INCREASED with the delay'),
        'why': ('delaying the departure can remove a wait so the vehicle arrives exactly at '
                '`tw_start`; because service_start = max(arrive, tw_start), the downstream schedule '
                'stops moving while pre-departure idle time disappears, SHORTENING the in-cargo '
                'time of later-picked lots'),
        'consequence': ('the tight "worst case = latest feasible departure" design MAY NOT be used; '
                        'the static T/tau analytic bound is the surviving candidate'),
    },
}
# The bound is only claimed for this quality model; the v2 absolute-sensory model needs its own
# derivation (user ruling: a v1 qualification must NOT be reused for v2).
APPLIES_TO = 'QualityConfig (v1, first-order, ratio thresholds)'


class Falsification(object):
    """Collects falsification evidence.  `searches` NEVER upgrades the verdict."""

    def __init__(self):
        self.checks = []
        self.counterexamples = []
        self.searches = 0

    def check(self, cid, description, passed, detail=None):
        self.checks.append({'id': cid, 'description': description, 'passed': bool(passed),
                            'detail': detail})
        if not passed:
            self.counterexamples.append({'id': cid, 'detail': detail})
        return passed

    def verdict(self):
        """FALSIFIED if any counterexample exists; otherwise NOT_ESTABLISHED.

        There is deliberately NO path to PROVEN from here: a bounded search plus passing checks on
        sampled trajectories is not a proof of an inequality over the whole admissible set.
        """
        if self.counterexamples:
            return FALSIFIED, [c['id'] for c in self.counterexamples]
        return NOT_ESTABLISHED, []


def omega_definition():
    """The formal admissible-trajectory set Omega, as data (user deliverable 1)."""
    return {
        'name': 'Omega_admissible_d3m0_v1',
        'fixed_by_production_configuration': {
            'active_zone_mask': '(True,)*zones -- strict_online_env._coldchain_active_mask() '
                                'never disables a zone in this configuration, so the uncooled '
                                'drift branch is out of Omega',
            'quality_model': 'v1 QualityConfig (first-order, per-class Arrhenius rate, ratio '
                             'thresholds)',
            'precool_branch': 'lot.precooled = thermal.precooled and temp_class >= '
                              'precool_min_temp_class (coldchain_state.py:261-262); both branches '
                              'are IN Omega and A2 covers both via the ambient widening',
        },
        'membership_conditions': {
            'thermal': 'temperature_hard_feasible == True (no segment temperature outside '
                       'hard_temperature_bounds_c; coldchain_evaluator.py:358)',
            'time_windows': 'every arrival <= tw_end; time windows are only widened by waiting at '
                            'tw_start, never by departing before the previous service finished',
            'capacity': 'cumulative load <= shared_vehicle_capacity at every pickup',
            'depot_return': 'return arrival <= tw_end[0] (else depot_return_violation)',
            'commitment': 'a committed leg is never revoked; an order is picked exactly once',
            'ordering': 'waiting is only "do not dispatch" (mutable_suffix == []) and may delay a '
                        'later departure, but may not reorder the plan suffix; the plan suffix '
                        'order is preserved in Omega',
            'clock': 'time never moves backwards (_advance_coldchain_to raises otherwise)',
            'horizon': 'the run stops at the instance horizon; a trajectory that needs '
                       '_force_return_all to return after tw_end[0] is NOT in Omega (see F-b)',
        },
        'degrees_of_freedom_within_Omega': [
            'the number and length of WAIT stretches before dispatch (bounded by booking_horizon)',
            'the exact departure time per leg inside the gap left by the previous service finish '
            'and the next tw_end',
            'the realised door events (one per pickup) and the number of remaining pickups',
            'the realised cooling duty, hence the actual per-segment zone temperatures',
        ],
        'not_in_Omega': ['thermal violations', 'arrivals after tw_end', 'capacity overflow',
                         'depot return after tw_end[0]', 'revoking a commitment',
                         'reordering the plan suffix', 'disabling an active zone'],
    }


def assumptions():
    """The five assumptions the candidate bound rests on, each with its falsification test."""
    return [
        {'id': 'A1', 'claim': 'every v1 quality update is q <- min(q, q*exp(-k*dt)); no other '
                              'factor or additive term',
         'evidence_anchor': 'coldchain_state.py:404-405',
         'falsified_by': 'an update path that multiplies/divides/adds something else'},
        {'id': 'A2', 'claim': "the temperature driving the rate never exceeds "
                              "max(hard_upper[class], ambient) on Omega",
         'evidence_anchor': 'coldchain_state.py:398-402 (rate temperature), '
                            'coldchain_evaluator.py:358 (hard feasibility)',
         'falsified_by': 'instrumenting the rate temperature on real runs and finding a value '
                         'above the bound (see F-a door warming)'},
        {'id': 'A3', 'claim': 'k(class, T) is non-decreasing in T',
         'evidence_anchor': 'coldchain_state.py:128-143 (Arrhenius, positive activation/R)',
         'falsified_by': 'a non-monotone rate over the admissible temperature range'},
        {'id': 'A4', 'claim': 'the in-cargo time from now to delivery never exceeds '
                              'tw_end[0] - now on Omega',
         'evidence_anchor': 'depot_return_violation in the terminal gate; '
                            'certify_route return check',
         'falsified_by': 'an Omega-member trajectory delivering after tw_end[0] (see F-b/F-c)'},
        {'id': 'A5', 'claim': 'Omega is DEFINED by the hard gates, so the bound is conditional on '
                              'them',
         'evidence_anchor': 'hard_gate.hard_vector_from_outcome / outcome_protocol_error',
         'falsified_by': 'relaxing a hard gate without re-deriving T^max / tau^max -- a '
                         'qualification change, not an empirical one'},
    ]


def ambient_branch_reachable(contract, dataset=None):
    """Is the NON-PRECOOLED (ambient-rate) branch reachable under this configuration?

    A lot is precooled iff `thermal.precooled and temp_class >= precool_min_temp_class`
    (coldchain_state.py:261-262).  When that holds for EVERY temperature class present, the
    ambient branch cannot be taken and the ambient widening of `T^max` is unnecessary -- which
    matters a lot here: with ambient = 25 C the widening lifts the class-2 bound from -12 C to
    +25 C and makes the analytic bound far looser than it needs to be.
    """
    th = contract.thermal
    if not bool(getattr(th, 'precooled', True)):
        return True, {'why': 'thermal.precooled is False, so every lot takes the ambient branch'}
    floor = int(getattr(th, 'precool_min_temp_class', 0))
    classes = [int(c) for c in th.supported_temp_classes]
    if dataset is not None and 'temp_class' in dataset:
        classes = sorted({int(x) for x in np.asarray(dataset['temp_class']).ravel() if int(x) >= 0})
    cold = [c for c in classes if c < floor]
    return bool(cold), {'precooled_flag': True, 'precool_min_temp_class': floor,
                        'classes_below_floor': sorted(cold),
                        'why': ('no class falls below precool_min_temp_class, so every lot is '
                                'precooled and the ambient branch is UNREACHABLE'
                                if not cold else
                                'classes below the precool floor take the ambient branch')}


def temperature_upper_bound(contract, temp_class, ambient_reachable=True):
    """`max(zone hard upper bound, ambient)` -- the ambient term is applied ONLY when the
    non-precooled branch is actually reachable (see `ambient_branch_reachable`)."""
    th = contract.thermal
    hard_hi = float(th.hard_temperature_bounds_c[int(temp_class)][1])
    ambient = float(th.ambient_temperature_c)
    bound = max(hard_hi, ambient) if ambient_reachable else hard_hi
    return bound, {'hard_upper': hard_hi, 'ambient': ambient,
                   'ambient_widening_applied': bool(ambient_reachable and ambient > hard_hi),
                   'reason': ('ambient branch reachable' if ambient_reachable
                              else 'ambient branch unreachable: every lot is precooled')}


def remaining_time_upper_bound(env, inst_idx, clock):
    """`tw_end[0] - clock` (depot closing time), clamped at 0."""
    depot_tw_end = float(env.tw_end[inst_idx, 0])
    return max(0.0, depot_tw_end - float(clock)), {'depot_tw_end': depot_tw_end,
                                                   'clock': float(clock)}


def rate_upper_bound(contract, temp_class, t_max):
    from coldchain_state import quality_rate_per_hour
    return float(quality_rate_per_hour(int(temp_class), float(t_max), contract))


def quality_lower_bound(contract, temp_class, q_start, tau_max, t_max=None):
    """`q_start * exp(-k(T^max) * tau^max)`, with the v1 per-class Arrhenius rate."""
    if t_max is None:
        t_max, _ = temperature_upper_bound(contract, temp_class)
    k = rate_upper_bound(contract, temp_class, t_max)
    q_lb = float(q_start) * math.exp(-k * float(tau_max))
    thr = float(contract.quality.unsalable_quality_threshold[int(temp_class)])
    ratio_lb = (q_lb / float(q_start)) if float(q_start) > 0 else None
    return {'q_lower_bound': q_lb, 'ratio_lower_bound': ratio_lb, 'threshold': thr,
            'salable_lower_bound': (None if ratio_lb is None else bool(ratio_lb >= thr)),
            'k_upper_per_hour': k, 'T_max_used': float(t_max), 'tau_max_used': float(tau_max)}


def bound_for_state(env, inst_idx, clock, vehicles, plan, contract):
    """Per-order lower bound from the CURRENT strict-online state (no future information).

    Loaded lots integrate from their real `quality_remaining`; orders not yet picked up are
    bounded from `initial_quality` at their (unknown, >= now) loading time -- which is exactly why
    `tau^max` is measured from `t_now` rather than from a predicted pickup.
    """
    tau_max, tau_detail = remaining_time_upper_bound(env, inst_idx, clock)
    amb_reachable, amb_detail = ambient_branch_reachable(contract, env.dataset)
    rows, per_lot = [], {}
    for v in vehicles:
        cc = v.coldchain_state
        if cc is None:
            continue
        for lot in getattr(cc, 'cargo_manifest', ()):
            t_max, t_detail = temperature_upper_bound(contract, lot.temp_class,
                                                     ambient_reachable=amb_reachable)
            lb = quality_lower_bound(contract, lot.temp_class, lot.quality_remaining, tau_max,
                                    t_max)
            lb.update({'order_id': int(lot.order_id), 'source': 'in_cargo',
                       'q_start': float(lot.quality_remaining), 'temp_class': int(lot.temp_class),
                       'initial_quality': float(lot.initial_quality),
                       'temperature_detail': t_detail, 'tau_detail': tau_detail,
                       'vehicle_id': int(v.vehicle_id)})
            rows.append(lb)
            per_lot[int(lot.order_id)] = lb
    planned = sorted({int(x) for p in plan.values() for x in p if int(x) > 0})
    for o in planned:
        if o in per_lot:
            continue
        c = int(env.dataset['temp_class'][inst_idx, o])
        iq = 1.0
        t_max, t_detail = temperature_upper_bound(contract, c,
                                                 ambient_reachable=amb_reachable)
        lb = quality_lower_bound(contract, c, iq, tau_max, t_max)
        lb.update({'order_id': int(o), 'source': 'not_yet_loaded', 'q_start': iq,
                   'temp_class': c, 'initial_quality': iq, 'temperature_detail': t_detail,
                   'tau_detail': tau_detail, 'vehicle_id': None})
        rows.append(lb)
        per_lot[int(o)] = lb
    unsalable_lb = sorted(r['order_id'] for r in rows if r['salable_lower_bound'] is False)
    return {
        'bound_schema': BOUND_SCHEMA, 'applies_to': APPLIES_TO,
        'verdict': NOT_ESTABLISHED,
        'verdict_reason': ('a candidate analytic bound with stated assumptions is NOT a proof; '
                           'verdicts are PROVEN / FALSIFIED / NOT_ESTABLISHED and a search count '
                           'may never upgrade this to PROVEN'),
        'tau_max': tau_max, 'tau_detail': tau_detail,
        'ambient_branch_reachable': amb_reachable,
        'ambient_branch_detail': amb_detail,
        'rows': rows, 'per_order': per_lot,
        'unsalable_lower_bound_orders': unsalable_lb,
        'all_salable_lower_bound': not unsalable_lb,
        'd8_impact': ('NONE: installable stays False. A passing lower bound on a sample is not the '
                      'proven inequality Q_cert,i <= Q_i(omega) for all omega in Omega.'),
    }


# ---------------------------------------------------------------------------
# falsification tools
# ---------------------------------------------------------------------------
class RateTemperatureRecorder(object):
    """Instruments the ACTUAL temperature each v1 quality update rates against (assumption A2).

    Patches `coldchain_state._advance_manifest_quality` by module-global name (the name
    `transition_segment` resolves at call time), so the recorded temperatures are the ones the
    production physics really used.  Read-only: the wrapper calls the original.
    """

    def __init__(self, contract, ambient_reachable=True):
        self.contract = contract
        self.ambient_reachable = bool(ambient_reachable)
        self.samples = []
        self._orig = None

    def __enter__(self):
        import coldchain_state as CS
        self._orig = CS._advance_manifest_quality
        recorder = self

        def wrapper(manifest, temperature_before, temperature_after, duration_h, contract):
            if duration_h > 0:
                for lot in manifest:
                    c = int(lot.temp_class)
                    precooled = bool(getattr(lot, 'precooled', True))
                    used = (0.5 * (float(temperature_before[c]) + float(temperature_after[c]))
                            if precooled
                            else float(contract.thermal.ambient_temperature_c))
                    hi, detail = temperature_upper_bound(
                        contract, c, ambient_reachable=recorder.ambient_reachable)
                    recorder.samples.append({'temp_class': c, 'precooled': precooled,
                                             'rate_temperature_c': used, 'bound_c': hi,
                                             'hard_upper': detail['hard_upper'],
                                             'ambient': detail['ambient'],
                                             'duration_h': float(duration_h)})
            return recorder._orig(manifest, temperature_before, temperature_after, duration_h,
                                  contract)

        CS._advance_manifest_quality = wrapper
        return self

    def __exit__(self, *exc):
        import coldchain_state as CS
        CS._advance_manifest_quality = self._orig
        return False

    def violations(self):
        return [s for s in self.samples if s['rate_temperature_c'] > s['bound_c'] + 1e-9]


def falsify_bound_against_delivered(bound_rows, delivered_records):
    """Necessary condition on real trajectories: `delivered_quality >= q_lower_bound`.

    A single violation FALSIFIES the candidate bound.  Agreement is evidence only.
    """
    by_order = {int(r['order_id']): r for r in bound_rows}
    violations = []
    for rec in delivered_records:
        oid = int(rec.get('order_id'))
        lb = by_order.get(oid)
        if lb is None:
            continue
        if float(rec['delivered_quality']) < lb['q_lower_bound'] - 1e-12:
            violations.append({'order_id': oid, 'delivered_quality': float(rec['delivered_quality']),
                               'q_lower_bound': lb['q_lower_bound'],
                               'gap': lb['q_lower_bound'] - float(rec['delivered_quality']),
                               'source': lb['source'], 'temp_class': lb['temp_class']})
    return violations


def search_schedules_for_binding_lot(env, inst_idx, clock, vehicles, contract, order_id,
                                     served_mask=None, grid=9, dataset=None):
    """Adversarial-but-FEASIBLE 1-D departure sweep for the binding lot's own vehicle.

    WHY 1-D: within this production machinery the admissible schedule of a FIXED plan is
    determined once the vehicle's departure is chosen -- the env never waits mid-route (waiting is
    "do not dispatch", `mutable_suffix == []`), and once committed a leg proceeds.  So the
    schedule degrees of freedom for a fixed plan reduce to the DEPARTURE time, and the sweep below
    covers its admissible range: `delay` from 0 up to the largest delay that keeps every arrival
    <= tw_end and the depot return <= tw_end[0].

    The realised service times and the realised delivered quality of the binding lot are recorded
    for every feasible grid point, so the sweep can be audited and can FALSIFY the monotonicity
    the tightened bound would rely on.  It can never prove anything.
    """
    from action_contract import build_vehicle_plans
    from coldchain_state import create_vehicle_state, dispatch_vehicle, transition_segment
    base_plans = build_vehicle_plans(env, inst_idx, vehicles)
    vid = next((v for v, p in base_plans.items() if order_id in [int(x) for x in p.suffix]), None)
    if vid is None:
        return {'searched': False, 'why': 'binding order is not in any plan suffix'}
    p = base_plans[vid]
    speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
    twe = env.tw_end[inst_idx]
    tws = env.tw_start[inst_idx]
    st_ = env.service_time[inst_idx]
    dist = env.dist_mat[inst_idx]
    zones = (True,) * len(contract.thermal.supported_temp_classes)

    def _replay(v, delay):
        """Replay the whole route with `delay` added to the departure; returns (feasible, rec)."""
        state = (v.coldchain_state if v.coldchain_state is not None
                 else dispatch_vehicle(create_vehicle_state(contract), contract))
        cur = int(p.anchor_node)
        t = float(p.anchor_time) + float(delay)
        times = []
        for o in [int(x) for x in p.suffix]:
            d = float(dist[cur, o])
            depart = max(t, float(env.reveal_time[inst_idx, o]))
            arrive = depart + d / speed
            if arrive > float(twe[o]) + 1e-6:
                return False, 'arrival_after_tw_end', times, None
            sstart = max(arrive, float(tws[o]))
            sfinish = sstart + float(st_[o])
            times.append({'customer': o, 'arrive': arrive, 'service_start': sstart})
            state, _ = transition_segment(
                state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
                served_customer=o, active_zone_mask=zones, contract=contract,
                order_quantity=float(env.demands[inst_idx, o]),
                order_temp_class=int(env.dataset['temp_class'][inst_idx, o]),
                initial_quality=1.0, segment_distance_units=d)
            cur, t = o, sfinish
        d_back = float(dist[cur, 0])
        arr_back = t + d_back / speed
        if arr_back > float(twe[0]) + 1e-6:
            return False, 'return_after_depot_tw_end', times, None
        state, _ = transition_segment(
            state, depart_time=t, arrival_time=arr_back, service_finish=arr_back,
            served_customer=None, active_zone_mask=zones, contract=contract,
            return_to_depot=True, segment_distance_units=d_back)
        rec = next((r for r in getattr(state, 'delivered_to_depot', ())
                    if int(r.order_id) == int(order_id)), None)
        return True, None, times, rec

    v = next(x for x in vehicles if int(x.vehicle_id) == int(vid))
    # largest admissible delay by bisection on feasibility
    lo, hi = 0.0, 48.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok, _why, _t, _r = _replay(v, mid)
        if ok:
            lo = mid
        else:
            hi = mid
    max_delay = lo
    rows, best = [], None
    for g in range(grid):
        frac = (g / float(grid - 1)) if grid > 1 else 0.0
        delay = frac * max_delay
        ok, why, times, rec = _replay(v, delay)
        rows.append({'grid': g, 'delay': round(delay, 9), 'feasible': bool(ok), 'why': why,
                     'first_service_start': (times[0]['service_start'] if times else None),
                     'binding_lot_service_start': next(
                         (t['service_start'] for t in times if t['customer'] == int(order_id)),
                         None),
                     'return_arrival': None,
                     'delivered_quality': (None if rec is None else float(rec.delivered_quality)),
                     'salable': (None if rec is None else
                                 (None if rec.salable is None else bool(rec.salable)))})
        if ok and rec is not None and (best is None or float(rec.delivered_quality) < best[0]):
            best = (float(rec.delivered_quality), g)
    feas = [r for r in rows if r['feasible'] and r['delivered_quality'] is not None]
    quals = [r['delivered_quality'] for r in feas]
    monotone = all(b <= a + 1e-12 for a, b in zip(quals, quals[1:])) if len(quals) > 1 else None
    return {'searched': True, 'vehicle_id': int(vid), 'order_id': int(order_id),
            'max_admissible_delay': max_delay, 'n_feasible': len(feas),
            'n_infeasible': len(rows) - len(feas),
            'min_delivered_quality': (best[0] if best else None),
            'distinct_minima': sorted({round(q, 12) for q in quals}),
            'quality_monotone_nonincreasing_in_delay': monotone,
            'rows': rows,
            'note': ('1-D admissible-departure sweep for a FIXED plan; the realised service times '
                     'and qualities are recorded; a bounded sweep can falsify but never prove')}
