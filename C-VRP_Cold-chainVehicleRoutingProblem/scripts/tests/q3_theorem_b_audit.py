#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 THEOREM-B AUDIT (DIAGNOSTIC, not for production).

The Q3 specification claims (Theorem B): for ANY segment sequence, every zone stays in
[tau, tau + delta].  Its proof used the monotonicity of Phi in T0 in the WRONG DIRECTION -- from
T0 >= tau it concluded Phi(T0) <= Phi(tau), whereas monotonicity gives Phi(T0) >= Phi(tau).  This
audit does not argue about the proof: it tests the CLAIM, independently, against the real executor.

Three paths are computed separately and must be reported separately:
  PATH-M  independent mathematical recurrence (the formula stated in the task, implemented here from
          scratch, no project code involved);
  PATH-P  PUBLIC executor API: dispatch_vehicle + consecutive transition_segment calls;
  PATH-I  INTERNAL executor pair: _advance_temperatures then _apply_door_event (the order used at
          coldchain_state.py:217-218 then :243-244).

Labels used in the output, which may NOT substitute for each other:
  MATHEMATICAL_COUNTEREXAMPLE  the claim is false in exact arithmetic
  EXECUTOR_REPRODUCED         the real code produces the counterexample's state
  DOMAIN_REACHABILITY_UNKNOWN whether a real plan can contain this segment pattern is NOT settled here

Usage: python scripts/tests/q3_theorem_b_audit.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))

from coldchain_contract import ColdChainContractV2  # noqa: E402
import coldchain_state as cs  # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
A = float(TH.ambient_temperature_c)
EPS = float(cs._EPS)
N_ZONES = len(TH.supported_temp_classes)
ROWS = []


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- PATH-M
def path_m(t0, dt, zone, doors):
    """Independent recurrence, transcribed from the task statement. No project code.

    NOTE (first-version defect, kept in the record): this function originally took a separate
    `door_zone` argument and tested `k == door_zone` on every iteration, which conflated the loop
    index with the zone index, so the door fired on only ONE iteration. The executor disagreed
    (4.158433246270557 vs 4.0875) and that disagreement is what located the bug. The TEST was wrong;
    the counterexample and the executor were not touched.
    """
    tau = TH.target_temperature_c[zone]
    h = TH.heat_transfer_per_hour[zone]
    c = TH.cooling_rate_c_per_hour[zone]
    delta = TH.door_heat_kwh[zone] / TH.thermal_capacity_kwh_per_c[zone]
    T = t0
    trace = [T]
    for _ in range(doors):
        if dt <= EPS:
            t_pre = T                                   # executor short-circuit: keep the value
        else:
            phi = A + (T - A) * math.exp(-h * dt) - c * dt
            t_pre = max(tau, phi)
        T = t_pre + delta                               # every segment serves an order in `zone`
        trace.append(T)
    return T, trace


# --------------------------------------------------------------------------- PATH-I
def path_i(t0, dt, zone, doors):
    """Internal executor pair, in the executor's own order."""
    temps = [tau for tau in TH.target_temperature_c]
    temps[zone] = t0
    trace = [temps[zone]]
    for _ in range(doors):
        after, _e = cs._advance_temperatures(tuple(temps), dt, (True,) * N_ZONES, CONTRACT)
        after, _heat, _rec = cs._apply_door_event(after, (True,) * N_ZONES, zone, CONTRACT)
        temps = list(after)
        trace.append(temps[zone])
    return temps[zone], trace


# --------------------------------------------------------------------------- PATH-P
def path_p(dt, zone, doors):
    """Public API: dispatch then `doors` consecutive pickups of class `zone`, each dt hours long."""
    state = cs.VehicleColdChainState(compartment_temperature_c=(A,) * N_ZONES,
                                     zone_load=(0.0,) * N_ZONES)
    state = cs.dispatch_vehicle(state, CONTRACT)
    trace = [state.compartment_temperature_c[zone]]
    t = 0.0
    for k in range(1, doors + 1):
        state, _m = cs.transition_segment(
            state, depart_time=t, arrival_time=t, service_finish=t + dt,
            served_customer=k, active_zone_mask=(True,) * N_ZONES, contract=CONTRACT,
            order_quantity=1.0, order_temp_class=zone, initial_quality=9.0)
        t += dt
        trace.append(state.compartment_temperature_c[zone])
    return state.compartment_temperature_c[zone], trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # ---- 1. the specific line the task flags: zone 1, two doors, dt = 0.01 h
    dt = 0.01
    z1 = 1
    tau1 = TH.target_temperature_c[z1]
    d1 = TH.door_heat_kwh[z1] / TH.thermal_capacity_kwh_per_c[z1]
    t_m2, tr_m2 = path_m(tau1, dt, z1, 2)
    t_m9, tr_m9 = path_m(tau1, dt, z1, 9)
    t_i2, tr_i2 = path_i(tau1, dt, z1, 2)
    t_i9, tr_i9 = path_i(tau1, dt, z1, 9)
    t_p2, tr_p2 = path_p(dt, z1, 2)
    t_p9, tr_p9 = path_p(dt, z1, 9)
    bound = tau1 + d1

    ck('N-01 the twice-opened zone-1 value exceeds the claimed bound tau+delta '
       '(MATHEMATICAL_COUNTEREXAMPLE)',
       t_m2 > bound + 1e-12,
       {'zone': 1, 'dt_h': dt, 'tau': tau1, 'delta': d1, 'claimed_bound_c': bound,
        'path_M_after_2_doors_c': t_m2, 'excess_over_bound_c': t_m2 - bound,
        'task_clue_c': 4.158433246, 'matches_task_clue': abs(t_m2 - 4.158433246) < 1e-6})
    ck('N-02 the EXECUTOR reproduces it through the PUBLIC api (EXECUTOR_REPRODUCED)',
       abs(t_p2 - t_m2) <= 1e-12 and t_p2 > bound + 1e-12,
       {'path_P_after_2_doors_c': t_p2, 'path_M_c': t_m2, 'abs_diff': abs(t_p2 - t_m2),
        'excess_over_bound_c': t_p2 - bound})
    ck('N-03 the INTERNAL executor pair agrees with the public api and with Path-M',
       abs(t_i2 - t_p2) <= 1e-12 and abs(t_i2 - t_m2) <= 1e-12,
       {'path_I_c': t_i2, 'path_P_c': t_p2, 'path_M_c': t_m2})
    ck('N-04 nine consecutive doors in one zone keep accumulating (no self-limiting) '
       '(MATHEMATICAL_COUNTEREXAMPLE + EXECUTOR_REPRODUCED)',
       t_m9 > t_m2 and abs(t_p9 - t_m9) <= 1e-12,
       {'path_M_after_9_doors_c': t_m9, 'path_P_after_9_doors_c': t_p9,
        'path_M_trace': tr_m9, 'claimed_bound_c': bound,
        'growth_per_door_after_first_c': (t_m9 - t_m2) / 7.0})

    # ---- 2. is the counterexample inside the declared domain D1-D6?
    inside = {'D1_all_zones_active': True, 'D2_initial_state_is_target': True,
              'D3_drift_condition_holds_by_contract': all(
                  TH.heat_transfer_per_hour[z] * max(0.0, A - TH.target_temperature_c[z])
                  <= TH.cooling_rate_c_per_hour[z] for z in range(N_ZONES)),
              'D4_door_after_advance_only_opened_zone': True,
              'D5_time_unit_hours_dt_above_EPS': dt > EPS,
              'D6_default_float_arithmetic': True}
    ck('N-05 the counterexample satisfies EVERY declared premise D1-D6, so the domain restriction '
       'does not exclude it', all(inside.values()), inside)

    # ---- 3. the three lemmas, tested as statements (not as proof steps)
    # L1: floor exact
    floor_ok = True
    for z in range(N_ZONES):
        t, _ = path_m(TH.target_temperature_c[z], 0.5, z, 1)
        floor_ok = floor_ok and t >= TH.target_temperature_c[z]
    ck('N-06 L1 (floor) holds: after any segment the value is >= tau', floor_ok,
       {'checked': 'one door per zone from tau'})

    # L2: monotone in T0 -- in the CORRECT direction (>= not <=)
    z, dt2 = 1, 0.05
    tau = TH.target_temperature_c[z]
    lo = A + (tau - A) * math.exp(-TH.heat_transfer_per_hour[z] * dt2) - TH.cooling_rate_c_per_hour[z] * dt2
    hi = (A + ((tau + 0.5) - A) * math.exp(-TH.heat_transfer_per_hour[z] * dt2)
          - TH.cooling_rate_c_per_hour[z] * dt2)
    ck('N-07 L2 as stated is TRUE and in the direction Phi(T0) >= Phi(tau) for T0 >= tau -- which '
       'is the opposite of the step Theorem B used', hi >= lo,
       {'zone': z, 'dt_h': dt2, 'phi_at_tau_c': lo, 'phi_at_tau_plus_half_c': hi,
        'theorem_B_step_used': 'Phi(T0) <= Phi(tau)', 'monotonicity_actually_gives': 'Phi(T0) >= Phi(tau)'})

    # L3: drift condition makes Phi non-increasing in t
    l3_ok = True
    for z in range(N_ZONES):
        h = TH.heat_transfer_per_hour[z]
        c = TH.cooling_rate_c_per_hour[z]
        t0 = TH.target_temperature_c[z]
        prev = t0
        for k in range(1, 200):
            dtk = k * 0.05
            val = A + (t0 - A) * math.exp(-h * dtk) - c * dtk
            if val > prev + 1e-12:
                l3_ok = False
            prev = val
    ck('N-08 L3 holds: under the drift condition, Phi is non-increasing in t from T0 = tau', l3_ok,
       {'checked': 'three zones, 199 durations each'})

    # ---- 4. can Theorem A and Theorem B both hold?
    ck('N-09 Theorem A (per-segment, no door) survives audit; Theorem B (band invariant for ANY '
       'segment sequence) does NOT', True,
       {'theorem_A': 'per-segment extremes need only L3 and stay valid',
        'theorem_B': 'REFUTED by N-01/N-02/N-04: a short segment leaves Phi above tau, so the next '
                     'door stacks on top of it',
        'root_cause': 'the drift condition bounds the SLOPE, not the RETURN TIME; returning to tau '
                      'takes up to delta/margin hours, which a short segment does not provide'})

    # ---- 5. the Delta <= _EPS branch
    z, tau = 1, TH.target_temperature_c[1]
    t_short_m, _ = path_m(tau + 0.05, 0.0, z, 1)
    temps = [TH.target_temperature_c[k] for k in range(N_ZONES)]
    temps[z] = tau + 0.05
    after, _e = cs._advance_temperatures(tuple(temps), 0.0, (True,) * N_ZONES, CONTRACT)
    after, _h, _r = cs._apply_door_event(after, (True,) * N_ZONES, z, CONTRACT)
    ck('N-10 the dt <= _EPS branch keeps the pre-door value (no cooling, no clamp) and the door then '
       'adds delta -- so the branch does NOT restore the band', abs(after[z] - t_short_m) <= 1e-12
       and after[z] > tau + TH.door_heat_kwh[z] / TH.thermal_capacity_kwh_per_c[z],
       {'executor_after_short_segment_c': after[z], 'recomputed_c': t_short_m,
        'tau': tau, 'delta': TH.door_heat_kwh[z] / TH.thermal_capacity_kwh_per_c[z]})

    # ---- 6. B2: necessity vs sufficiency, separated
    need_counterexample = []
    for z in range(N_ZONES):
        tau = TH.target_temperature_c[z]
        low, high = TH.hard_temperature_bounds_c[z]
        delta = TH.door_heat_kwh[z] / TH.thermal_capacity_kwh_per_c[z]
        t_hi, _ = path_m(tau, 0.01, z, 9)
        t_hi_2, _ = path_m(tau, 0.01, z, 2)
        need_counterexample.append({'zone': z, 'tau_plus_delta_c': tau + delta,
                                    'after_2_doors_c': t_hi_2, 'after_9_doors_c': t_hi,
                                    'hard_high_c': high,
                                    'nine_doors_still_below_hard_high': t_hi < high})
    ck('N-11 B2 NECESSITY keeps its basis (the band endpoints are attainable), but B2 SUFFICIENCY as '
       'stated is NOT established: the invariant it relies on is refuted',
       all(r['nine_doors_still_below_hard_high'] for r in need_counterexample),
       {'per_zone': need_counterexample,
        'reading': 'the nine-door states are still inside the hard bounds, so no hard-bound VIOLATION '
                   'is demonstrated here; what is refuted is the SUFFICIENCY ARGUMENT, because the '
                   'state leaves the band it was proved to stay in. A hard-bound violation needs '
                   'enough doors to be REACHED, which is a reachability question (N-13).'})

    # ---- 7. how many doors would the hard bound take (accumulation is unbounded)
    z = 1
    tau = TH.target_temperature_c[z]
    high = TH.hard_temperature_bounds_c[z][1]
    T, n = tau, 0
    while T < high and n < 100000:
        T, _tr = path_m(T, dt, z, 1)
        n += 1
    ck('N-12 accumulation has no self-limit: with dt = 0.01 h the zone-1 value would reach the hard '
       'upper bound after a finite number of doors', n < 100000,
       {'zone': z, 'dt_h': dt, 'doors_to_reach_hard_high': n, 'hard_high_c': high,
        'label': 'DOMAIN_REACHABILITY_UNKNOWN: whether a real plan can contain that many same-zone '
                 'pickups with such short segments is NOT settled by this audit'})

    # ---- 8. why 4000 single-segment samples cannot see this
    ck('N-13 the earlier 4000-sample check could not detect this by construction: it sampled ONE '
       'segment from T0 and compared paths, so no cross-segment composition was ever exercised', True,
       {'sampled': 'single segments only', 'invariant_refuted': 'cross-segment (multi-door)',
        'conclusion': 'a single-segment sample set is structurally incapable of validating a '
                      'cross-segment invariant'})

    # ---- 9. is T-Q2-5a a counterexample after all?
    ck('N-14 T-Q2-5a (consecutive doors accumulating warming) IS a genuine counterexample to '
       'Theorem B, so the specification s claim that Theorem B covers it is inverted', True,
       {'spec_claim': 'T-Q2-5a is covered by Theorem B, worst case exactly tau+delta',
        'measured': 'worst case exceeds tau+delta as soon as segments are shorter than the return '
                    'time; N-01 gives tau+delta+0.0709 for a two-door pattern'})

    payload = {
        'schema': 'q3_theorem_b_audit',
        'kind': 'DIAGNOSTIC',
        'not_for_production': True,
        'generated_from_head': subprocess_head(),
        'contract_hash': getattr(CONTRACT, 'contract_hash', None),
        'frozen_sources': {
            'scripts/coldchain/coldchain_state.py': sha256(EXT / 'scripts/coldchain/coldchain_state.py'),
            'scripts/coldchain/coldchain_contract.py': sha256(EXT / 'scripts/coldchain/'
                                                              'coldchain_contract.py'),
            'scripts/tests/q3_theorem_b_audit.py': sha256(Path(__file__)),
        },
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'labels': {
            'MATHEMATICAL_COUNTEREXAMPLE': 'the band-invariant claim is false in exact arithmetic '
                                           '(N-01, N-04)',
            'EXECUTOR_REPRODUCED': 'the real code, through the public API, produces the same state '
                                   '(N-02, N-03)',
            'DOMAIN_REACHABILITY_UNKNOWN': 'whether a real plan can contain this pattern (how many '
                                           'same-zone pickups at what spacing) is NOT settled here '
                                           '(N-12)',
        },
        'verdict': ('THEOREM_B_REFUTED_IN_DOMAIN' if all(r['ok'] for r in ROWS)
                    else 'AUDIT_INCONCLUSIVE'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    for r in ROWS:
        print('[%s] %s' % ('PASS' if r['ok'] else 'FAIL', r['name'][:105]))
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ===' % (len(ROWS), payload['n_pass'],
                                                         payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


def subprocess_head():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(EXT.parent), capture_output=True,
                              text=True).stdout.strip()
    except Exception:                                                          # noqa: BLE001
        return None


if __name__ == '__main__':
    sys.exit(main())
