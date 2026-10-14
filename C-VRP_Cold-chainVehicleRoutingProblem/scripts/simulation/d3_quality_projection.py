#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-Q / D-B3A-2: READ-ONLY quality projection (LIMITED implementation).

Authorised scope (user ruling 2026-10-14): implement a read-only projection that REUSES the existing
`transition_segment` physics; do NOT claim conservative certification.  No write-back, no `_commit`
change, no D8 bypass.

What this is and is NOT
-----------------------
It is a PROJECTION: given the current cold-chain state, a candidate route and the contract, replay
the same physics the environment would use and read the resulting per-lot delivered quality.
It is NOT a conservative certificate.  The user rejected the inference "quality is monotone
non-increasing, therefore serving every customer at tw_end gives the worst-case quality":
  * pushing every customer to tw_end can violate later time windows, leaving the admissible set;
  * longer waiting changes compartment temperature / cooling mode, so the worst temperature history
    cannot be inferred from monotone time decay alone;
  * lots enter the cold chain at different times, so maximising one customer's service time does
    not simultaneously maximise every lot's loss.
Two monotonicity lemmas DO hold (quality is non-increasing in elapsed time at fixed temperature,
and non-increasing in temperature at fixed duration), but they do not compose into a bound over the
schedule set.  Every result therefore carries `conservativeness` and the label stays
`CONSERVATIVENESS_NOT_ESTABLISHED` until a testable inequality Q_cert,i <= Q_i(omega) for all
omega in Omega_admissible is PROVEN.  Nothing here may be upgraded to "conservative".

Why the semantics cannot diverge from the environment
-----------------------------------------------------
The return leg is replayed with `return_to_depot=True`, and `transition_segment` itself builds the
`DeliveredCargoRecord`s through `_delivery_record` (coldchain_state.py:450-469), including the v1
rule `salable = quality_remaining / initial_quality >= unsalable_quality_threshold[temp_class]`.
So the projection READS the environment's own records instead of re-implementing the threshold.

Contract-version qualification (paper risk): this line runs v1 `QualityConfig`.  The A-v1 main task
states a fish-salability hard constraint under the v2 absolute sensory metric.  They are different
physics and different metric semantics, so a v1 pass here does NOT transfer to v2.
"""
from __future__ import annotations

import numpy as np

CONSERVATIVENESS_NOT_ESTABLISHED = 'CONSERVATIVENESS_NOT_ESTABLISHED'
PROJECTED_SATISFIED_UNPROVEN = 'PROJECTED_SATISFIED_UNPROVEN'
PROJECTED_FAILED = 'PROJECTED_FAILED'
SCHEDULE_NOMINAL = 'nominal'
SCHEDULE_LATEST_ADMISSIBLE = 'latest_admissible'


def _contract_version(contract):
    q = getattr(contract, 'quality', None)
    name = type(q).__name__
    schema = getattr(contract, 'schema_version', None)
    return {'quality_config_class': name, 'contract_schema_version': schema,
            'is_v2': name == 'QualityConfigV2',
            'note': ('v1 uses ratio thresholds (unsalable_quality_threshold); v2 uses an absolute '
                     'sensory score. A v1 result does NOT transfer to v2.')}


def project_plan_quality(env, inst_idx, clock, vehicles, plan, contract, *,
                         schedule=SCHEDULE_NOMINAL, served_mask=None,
                         active_zone_mask=None, wait_until=None):
    """Replay ONE vehicle's committed+planned load to the depot and read the delivered quality.

    `plan` is `{vid: [customers]}` (not-yet-executed orders, NO trailing 0).
    Returns a dict per vehicle; `conservativeness` is always the NOT_ESTABLISHED label.

    Schedule semantics:
      * `nominal`  -- exactly the schedule `c0_completion_energy` uses: depart = max(t, reveal_time),
                      arrive = depart + d/speed, service_start = max(arrive, tw_start).
      * `latest_admissible` -- a STRESS scenario: arrival is pushed to `tw_end` (the latest arrival
                      the TW admits), then propagated.  The constructed schedule is CHECKED for
                      admissibility and reported; an inadmissible stress schedule is NOT a bound.
    """
    from coldchain_state import (create_vehicle_state, dispatch_vehicle, transition_segment)

    if active_zone_mask is None:
        classes = contract.thermal.supported_temp_classes
        active_zone_mask = (True,) * len(classes)
    speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
    reveal = env.reveal_time[inst_idx]
    tws = env.tw_start[inst_idx]
    twe = env.tw_end[inst_idx]
    st_ = env.service_time[inst_idx]
    dem = env.demands[inst_idx]
    dist = env.dist_mat[inst_idx]
    tc = env.dataset['temp_class'][inst_idx]

    results = {}
    for v in vehicles:
        route = [int(x) for x in plan.get(int(v.vehicle_id), []) if int(x) > 0]
        cc = v.coldchain_state
        if cc is not None and cc.closed:
            results[int(v.vehicle_id)] = {
                'route': route, 'closed': True,
                'delivered': [r.__dict__ if hasattr(r, '__dict__') else str(r)
                              for r in cc.delivered_to_depot],
                'reason': 'vehicle already closed: nothing to project'}
            continue
        # committed leg first, exactly once (mirrors certify_plan:176-181)
        committed = v.committed_next if v.committed_next not in (None, 0) else None
        full = ([int(committed)] if committed is not None else []) + \
               [o for o in route if o != committed]
        if cc is not None:
            state = cc
            cur = int(v.current_node)
            t = float(v.ready_time) if v.status in ('idle', 'ready') else float(v.ready_time)
            t = max(t, float(clock))
        else:
            state = dispatch_vehicle(create_vehicle_state(contract), contract)
            cur = 0
            t = float(clock)
        if not full and cur == 0:
            results[int(v.vehicle_id)] = {
                'route': route, 'delivered': [], 'salable_all': True, 'reason': 'empty route',
                'schedule': {}, 'stress_schedule_admissible': None}
            continue

        schedule_rows, admissible, violations = [], True, []
        for o in full:
            d = float(dist[cur, o])
            if schedule == SCHEDULE_NOMINAL:
                depart = max(t, float(reveal[o]))
                arrive = depart + d / speed
                if arrive > float(twe[o]) + 1e-6:
                    violations.append({'customer': int(o), 'violation': 'arrival_after_tw_end',
                                       'arrive': arrive, 'tw_end': float(twe[o])})
                    admissible = False
                sstart = max(arrive, float(tws[o]))
            else:
                arrive_latest = float(twe[o])
                depart = arrive_latest - d / speed
                depart = max(depart, t)                      # cannot depart before we are ready
                arrive = depart + d / speed
                if arrive > float(twe[o]) + 1e-6:
                    violations.append({'customer': int(o), 'violation': 'latest_arrival_unreachable',
                                       'arrive': arrive, 'tw_end': float(twe[o])})
                    admissible = False
                sstart = max(arrive, float(tws[o]))
            sfinish = sstart + float(st_[o])
            iq = (float(contract.quality.initial_value[int(tc[o])])
                  if hasattr(contract.quality, 'initial_value') else 1.0)
            schedule_rows.append({'customer': int(o), 'depart': depart, 'arrive': arrive,
                                  'service_start': sstart, 'service_finish': sfinish})
            state, _ = transition_segment(
                state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
                served_customer=int(o), active_zone_mask=active_zone_mask, contract=contract,
                order_quantity=float(dem[o]), order_temp_class=int(tc[o]),
                initial_quality=iq, segment_distance_units=d)
            cur, t = int(o), sfinish
        wu = wait_until if wait_until is not None else env.booking_horizon
        if wu is not None and wu > t + 1e-9:
            state, _ = transition_segment(
                state, depart_time=t, arrival_time=wu, service_finish=wu,
                served_customer=None, active_zone_mask=active_zone_mask, contract=contract,
                segment_distance_units=0.0)
            t = wu
        d_back = float(dist[cur, 0])
        arrive_back = t + d_back / speed
        if arrive_back > float(twe[0]) + 1e-6:
            violations.append({'customer': 0, 'violation': 'return_after_depot_tw_end',
                               'arrive': arrive_back, 'tw_end': float(twe[0])})
            admissible = False
        state, _ = transition_segment(
            state, depart_time=t, arrival_time=arrive_back, service_finish=arrive_back,
            served_customer=None, active_zone_mask=active_zone_mask, contract=contract,
            return_to_depot=True, segment_distance_units=d_back)

        records = []
        for r in getattr(state, 'delivered_to_depot', ()):
            records.append({
                'order_id': int(r.order_id), 'temp_class': int(r.temp_class),
                'initial_quality': float(r.initial_quality),
                'delivered_quality': float(r.delivered_quality),
                'ratio': (float(r.delivered_quality) / float(r.initial_quality)
                          if float(r.initial_quality) > 0 else None),
                'threshold': float(contract.quality.unsalable_quality_threshold[int(r.temp_class)]),
                'salable': (None if r.salable is None else bool(r.salable)),
                'return_arrival_time': float(r.return_arrival_time),
            })
        still_loaded = [int(lot.order_id) for lot in getattr(state, 'cargo_manifest', ())]
        unsalable = sorted(r['order_id'] for r in records if r['salable'] is False)
        results[int(v.vehicle_id)] = {
            'route': route, 'committed_leg': (None if committed is None else int(committed)),
            'schedule_kind': schedule, 'schedule': schedule_rows,
            'stress_schedule_admissible': (admissible if schedule != SCHEDULE_NOMINAL else None),
            'admissibility_violations': violations,
            'delivered': records, 'still_loaded_after_return': still_loaded,
            'unsalable_orders': unsalable, 'salable_all': not unsalable and not still_loaded,
            'return_arrival_time': arrive_back,
        }

    all_records = [r for v in results.values() for r in v.get('delivered', [])]
    unsalable = sorted(r['order_id'] for v in results.values() for r in v.get('unsalable_orders', []))
    still_loaded = sorted(o for v in results.values() for o in v.get('still_loaded_after_return', []))
    inadmissible = sorted(vid for vid, v in results.items()
                          if v.get('stress_schedule_admissible') is False)
    if unsalable or still_loaded:
        projection_state = PROJECTED_FAILED
    elif schedule == SCHEDULE_LATEST_ADMISSIBLE and inadmissible:
        projection_state = CONSERVATIVENESS_NOT_ESTABLISHED
    else:
        projection_state = PROJECTED_SATISFIED_UNPROVEN
    return {
        'projection_schema': 'quality_projection_v1',
        'schedule_kind': schedule,
        'conservativeness': CONSERVATIVENESS_NOT_ESTABLISHED,
        'conservativeness_reason': (
            'not a bound: the user rejected the inference that pushing every customer to tw_end '
            'yields the worst-case quality (inadmissible stress schedules, schedule-dependent '
            'compartment temperature, and lots entering the cold chain at different times). '
            'A testable inequality Q_cert,i <= Q_i(omega) for all omega in Omega_admissible is '
            'required before any conservative label may be used.'),
        'projection_state': projection_state,
        'applicable_contract_version': _contract_version(contract),
        'per_vehicle': results,
        'n_delivered_records': len(all_records),
        'unsalable_orders': unsalable,
        'still_loaded_after_return': still_loaded,
        'inadmissible_stress_vehicles': inadmissible,
        'installable_impact': 'NONE: this projection does not change installable; D8 stays blocked',
    }
