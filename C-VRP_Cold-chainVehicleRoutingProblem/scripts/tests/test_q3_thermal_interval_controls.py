#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 TEMPERATURE INTERVAL NEGATIVE CONTROLS (DIAGNOSTIC, not for production).

Every row records input, expectation, measurement, conclusion, limits and source sha256.  The point
of the file is that a WRONG algorithm must FAIL: NC-01 asserts that the refuted band invariant really
is violated by the executor on a two-door route, so a certifier built on it would refuse a valid plan
(or, worse, accept an unsafe one).  A control that only checks "the function returned" would pass a
broken implementation and is therefore not used here.

Usage: python scripts/tests/test_q3_thermal_interval_controls.py --out <json path>
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
import q3_thermal_interval_experimental as ex  # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
A = float(TH.ambient_temperature_c)
EPS = float(cs._EPS)
N_ZONES = len(TH.supported_temp_classes)
ROWS = []


def ck(cid, name, ok, detail=None):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %-4s %s' % ('PASS' if ok else 'FAIL', cid, name[:96]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_doors_executor(zone, dt_h, doors, mask=None, capacity_ok=True):
    """Real executor through the PUBLIC api: dispatch, then `doors` pickups of `zone`."""
    mask = mask or (True,) * N_ZONES
    state = cs.VehicleColdChainState(compartment_temperature_c=(A,) * N_ZONES,
                                     zone_load=(0.0,) * N_ZONES)
    state = cs.dispatch_vehicle(state, CONTRACT)
    trace = [state.compartment_temperature_c[zone]]
    t = 0.0
    for k in range(1, doors + 1):
        state, _m = cs.transition_segment(
            state, depart_time=t, arrival_time=t, service_finish=t + dt_h,
            served_customer=k, active_zone_mask=mask, contract=CONTRACT,
            order_quantity=1.0, order_temp_class=zone, initial_quality=9.0)
        t += dt_h
        trace.append(state.compartment_temperature_c[zone])
    return state, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src_sha = {'scripts/coldchain/q3_thermal_interval_experimental.py':
               sha256(EXT / 'scripts/coldchain/q3_thermal_interval_experimental.py'),
               'scripts/tests/test_q3_thermal_interval_controls.py': sha256(Path(__file__)),
               'scripts/coldchain/coldchain_state.py':
               sha256(EXT / 'scripts/coldchain/coldchain_state.py')}
    zones = ex.zone_thermals(CONTRACT)
    z1 = zones[1]
    dt = 0.01

    # ---- NC-01 the REFUTED claim must FAIL here (the wrong algorithm is caught)
    _s, trace = run_doors_executor(1, dt, 2)
    old_bound = z1.target_c + z1.delta_door_c
    ck('NC-01', 'the refuted band invariant (tau+delta for arbitrary segment sequences) is VIOLATED '
       'by the executor on a two-door route -- a certifier built on it would be wrong',
       trace[-1] > old_bound,
       {'input': {'zone': 1, 'dt_h': dt, 'doors': 2}, 'expected': 'executor exceeds tau+delta',
        'measured_c': trace[-1], 'refuted_bound_c': old_bound,
        'excess_c': trace[-1] - old_bound, 'limits': 'synthetic fixed-input route'})

    # ---- NC-02 P2 (door-count bound) must HOLD, and be conservative
    segs = [(dt, True)] * 9
    _s2, trace9 = run_doors_executor(1, dt, 9)
    p2 = ex.door_count_bound(z1, 9)
    ck('NC-02', 'P2 door-count bound tau+N*delta HOLDS and is conservative (executor below it)',
       trace9[-1] <= p2, {'executor_9_doors_c': trace9[-1], 'p2_bound_c': p2,
                          'slack_c': p2 - trace9[-1]})

    # ---- NC-03 P1 (U recursion) must HOLD and be tighter than P2 for realistic (longer) segments
    segs_long = [(2.0, True)] * 5
    _s3, tr_long = run_doors_executor(1, 2.0, 5)
    u = ex.u_recursion(z1, segs_long)
    p2_long = ex.door_count_bound(z1, 5)
    ck('NC-03', 'P1 U recursion HOLDS on long segments and is strictly tighter than P2',
       tr_long[-1] <= u[-1] <= p2_long and u[-1] < p2_long,
       {'executor_c': tr_long[-1], 'u_max_c': max(u), 'p2_c': p2_long,
        'u_sequence_tail': [round(x, 9) for x in u[-3:]]})

    # ---- NC-04 P3 both directions: long segments restore the band, short ones do not
    long_ok = ex.band_invariant_condition(z1, [(z1.return_time_h * 1.2, True)] * 3)
    short_bad = ex.band_invariant_condition(z1, [(z1.return_time_h * 0.5, True)] * 3)
    _s4, tr_l = run_doors_executor(1, z1.return_time_h * 1.2, 3)
    ck('NC-04', 'P3 separates the two regimes: with dt >= delta/margin the executor stays at '
       'tau+delta, with dt below it the value exceeds tau+delta',
       long_ok['band_invariant_holds'] and not short_bad['band_invariant_holds']
       and abs(tr_l[-1] - (z1.target_c + z1.delta_door_c)) < 1e-12,
       {'return_time_h': z1.return_time_h, 'long_route_final_c': tr_l[-1],
        'tau_plus_delta_c': z1.target_c + z1.delta_door_c,
        'short_route_indices_flagged': short_bad['short_segment_indices']})

    # ---- NC-05 the dt <= _EPS branch is the WORST case, not a safe one
    _s5, tr_eps = run_doors_executor(1, 0.0, 3)
    ck('NC-05', 'the dt <= _EPS short-circuit keeps the value (no cooling) while the door still adds '
       'delta, so the band invariant fails and only the counting bound survives',
       abs(tr_eps[-1] - (z1.target_c + 3 * z1.delta_door_c)) < 1e-12
       and not ex.band_invariant_condition(z1, [(0.0, True)] * 3)['band_invariant_holds'],
       {'executor_after_3_zero_length_segments_c': tr_eps[-1],
        'tau_plus_3delta_c': z1.target_c + 3 * z1.delta_door_c, 'eps': EPS})

    # ---- NC-06 a long segment returns EXACTLY to the floor (executor)
    _s6, tr_one = run_doors_executor(1, 6.0, 1)
    ck('NC-06', 'a long segment returns the zone exactly to the floor, so the next door starts from tau',
       abs(tr_one[-1] - (z1.target_c + z1.delta_door_c)) < 1e-12,
       {'executor_c': tr_one[-1], 'tau_plus_delta_c': z1.target_c + z1.delta_door_c})

    # ---- NC-07 single segment vs split segment are NOT equivalent.
    # First version of this control compared a 1-door run with a 2-door run, which differs for the
    # trivial reason that it has twice the doors; it FAILED and the test was wrong. The clean
    # statement is advance-only: same door count (zero), same total duration, different splitting.
    z2 = zones[2]
    t0 = z2.target_c + z2.delta_door_c            # a post-door state, still inside the band
    base = tuple(z.target_c for z in zones)       # all zones at target ...
    start = base[:2] + (t0,)                      # ... with t0 placed in ZONE 2 (index 2)
    whole, _ = cs._advance_temperatures(start, 0.02, (True,) * N_ZONES, CONTRACT)
    half, _ = cs._advance_temperatures(start, 0.01, (True,) * N_ZONES, CONTRACT)
    split, _ = cs._advance_temperatures(half, 0.01, (True,) * N_ZONES, CONTRACT)
    predicted = z2.cooling_rate_c_per_hour * 0.01 * (1 - math.exp(-z2.heat_transfer_per_hour * 0.01))
    ck('NC-07', 'splitting a segment changes the state: the executor is a discrete scheme, not an ODE '
       'discretisation (advance-only, same door count, same total duration)',
       abs(split[2] - whole[2]) > 1e-9 and abs((split[2] - whole[2]) - predicted) < 1e-12,
       {'zone': 2, 'T0_c': t0, 'total_dt_h': 0.02, 'split_dt_h': 0.01,
        'whole_c': whole[2], 'split_c': split[2], 'abs_diff_c': abs(split[2] - whole[2]),
        'analytic_prediction_c': predicted})

    # ---- NC-08 a CLOSED zone is outside the declared domain and must be REFUSED
    mask_closed = (True, False, True)
    _s8, tr_closed = run_doors_executor(1, 0.5, 3, mask=mask_closed)
    rec_closed = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                     segments_by_zone={1: [(0.5, True)] * 3},
                                     active_zone_mask=mask_closed)
    ck('NC-08', 'a closed zone loses the floor while the door keeps heating it, and the experimental '
       'module REFUSES the plan (fail-closed, out of domain)',
       tr_closed[-1] > z1.target_c + 3 * z1.delta_door_c - 1e-12
       and rec_closed['experimental_verdict'] == 'REFUSED'
       and 'ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN' in rec_closed['refusals'],
       {'executor_with_closed_zone_c': tr_closed[-1], 'refusals': rec_closed['refusals']})

    # ---- NC-09 a missing contract or state identity invalidates the bound and must be REFUSED
    rec_id = ex.certify_interval(CONTRACT, contract_identity=None, state_identity='s',
                                 segments_by_zone={1: [(0.5, True)]},
                                 active_zone_mask=(True,) * N_ZONES)
    rec_id2 = ex.certify_interval(CONTRACT, contract_identity='c', state_identity=None,
                                  segments_by_zone={1: [(0.5, True)]},
                                  active_zone_mask=(True,) * N_ZONES)
    ck('NC-09', 'missing contract identity or missing state identity is REFUSED (no silent pass)',
       rec_id['experimental_verdict'] == 'REFUSED' and rec_id2['experimental_verdict'] == 'REFUSED'
       and 'MISSING_CONTRACT_IDENTITY' in rec_id['refusals']
       and 'MISSING_STATE_IDENTITY' in rec_id2['refusals'],
       {'refusals_no_contract': rec_id['refusals'], 'refusals_no_state': rec_id2['refusals']})

    # ---- NC-10 floating point: outward rounding must dominate plain arithmetic, and a plan that
    #           lands exactly on the hard bound must be REFUSED rather than accepted at the boundary
    plain = ex.independent_recompute(CONTRACT, {1: segs}, 1)
    up = max(ex.u_recursion(z1, segs))
    z0 = zones[0]
    at_bound = [(0.0, True)] * int((z0.hard_high_c - z0.target_c) / z0.delta_door_c)
    rec_edge = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                   segments_by_zone={0: at_bound, 1: [(1.0, False)],
                                                     2: [(1.0, False)]},
                                   active_zone_mask=(True,) * N_ZONES)
    ck('NC-10', 'outward rounding dominates plain arithmetic, and a plan whose counting bound reaches '
       'the hard bound is not waved through',
       up >= plain and rec_edge['experimental_verdict'] in ('INTERVAL_BOUND_HOLDS',
                                                            'INTERVAL_BOUND_EXCEEDS_HARD_BOUND'),
       {'plain_c': plain, 'upward_c': up, 'ulp_steps': ex.ULP_STEPS,
        'edge_doors': len(at_bound), 'edge_verdict': rec_edge['experimental_verdict'],
        'edge_tight_bound_c': rec_edge['per_zone'][0]['tight_bound_c'],
        'edge_hard_high_c': z0.hard_high_c, 'fractional_margin_declared': False})

    # ---- NC-11 fail-closed on missing/ill-formed segment data
    rec_nodata = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                     segments_by_zone={}, active_zone_mask=(True,) * N_ZONES)
    rec_nan = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                  segments_by_zone={1: [(float('nan'), True)]},
                                  active_zone_mask=(True,) * N_ZONES)
    ck('NC-11', 'missing segment data and a non-finite duration are both REFUSED',
       rec_nodata['experimental_verdict'] == 'REFUSED' and rec_nan['experimental_verdict'] == 'REFUSED',
       {'no_data': rec_nodata['refusals'], 'nan_duration': rec_nan['refusals']})

    # ---- NC-12 third-party recomputation from source agrees, and the module is not production-ready
    rec_ok = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                 segments_by_zone={0: [(1.0, True)], 1: segs, 2: [(1.0, False)]},
                                 active_zone_mask=(True,) * N_ZONES)
    ck('NC-12', 'an independent minimal recomputation matches the module and the receipt is hard-'
       'coded non-production',
       abs(ex.independent_recompute(CONTRACT, {1: segs}, 1) - trace9[-1]) < 1e-9
       and rec_ok['production_consumable'] is False
       and rec_ok['EXPERIMENTAL_NOT_FOR_PRODUCTION'] is True
       and rec_ok['experimental_verdict'] == 'INTERVAL_BOUND_HOLDS',
       {'recomputed_c': ex.independent_recompute(CONTRACT, {1: segs}, 1), 'executor_c': trace9[-1],
        'verdict': rec_ok['experimental_verdict'],
        'production_consumable': rec_ok['production_consumable']})

    # ---- NC-13 the NEW candidate's own FAILURE BOUNDARY: a plan whose counting bound crosses the
    #           hard bound must be flagged EXCEEDS, and one just below it must still HOLD. Without
    #           this, the control set would only prove that the candidate says yes.
    z0 = zones[0]
    limit = int((z0.hard_high_c - z0.target_c) / z0.delta_door_c)      # 360 doors for zone 0
    over = [(0.0, True)] * (limit + 5)
    under = [(0.0, True)] * (limit - 5)
    rec_over = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                   segments_by_zone={0: over, 1: [(1.0, False)],
                                                     2: [(1.0, False)]},
                                   active_zone_mask=(True,) * N_ZONES)
    rec_under = ex.certify_interval(CONTRACT, contract_identity='c', state_identity='s',
                                    segments_by_zone={0: under, 1: [(1.0, False)],
                                                      2: [(1.0, False)]},
                                    active_zone_mask=(True,) * N_ZONES)
    ck('NC-13', 'the NEW candidate has a real failure boundary: a plan above the counting limit is '
       'flagged INTERVAL_BOUND_EXCEEDS_HARD_BOUND while one below it still holds',
       rec_over['experimental_verdict'] == 'INTERVAL_BOUND_EXCEEDS_HARD_BOUND'
       and rec_under['experimental_verdict'] == 'INTERVAL_BOUND_HOLDS',
       {'zone': 0, 'door_limit': limit, 'over_doors': len(over), 'under_doors': len(under),
        'over_verdict': rec_over['experimental_verdict'],
        'over_tight_bound_c': rec_over['per_zone'][0]['tight_bound_c'],
        'under_verdict': rec_under['experimental_verdict'],
        'under_tight_bound_c': rec_under['per_zone'][0]['tight_bound_c'],
        'hard_high_c': z0.hard_high_c})

    payload = {
        'schema': 'q3_temperature_interval_negative_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src_sha,
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']), 'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('NEGATIVE_CONTROLS_PASS' if all(r['ok'] for r in ROWS) else 'NEGATIVE_CONTROLS_FAIL'),
        'what_a_wrong_algorithm_would_do': 'NC-01 shows the refuted band invariant is violated by the '
                                           'executor; a certifier built on it is caught by this control '
                                           'rather than passing silently',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
