#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 PLAN-LAYER REACHABILITY CONTROLS (DIAGNOSTIC, not for production).

Two things are checked, and they must not be confused:

  (i)  P3' as a MATHEMATICAL statement: with the drift condition, a segment that opens no door on zone
       z cannot push zone z out of [tau, tau+delta], so the minimum-segment-length condition only has
       to hold on the DOOR segments of that zone.  PL-10/PL-11/PL-13 exercise this against the real
       executor, including the `Delta <= _EPS` branch.
  (ii) the plan-layer CHECKER: it must refuse identity/reveal/commitment/malformed input, and it must
       distinguish PROVED_CONDITIONAL from DIAGNOSTIC_ONLY from NOT_CERTIFIABLE.  PL-01..PL-09 and
       PL-12 do this on synthetic instances and, read-only, on the frozen instance files.

The two-door and nine-door counterexamples to the OLD Theorem B are preserved (PL-14): nothing here
restores the refuted band invariant.

Usage: python scripts/tests/test_q3_plan_layer_controls.py --out <json path>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))

from coldchain_contract import ColdChainContractV2                        # noqa: E402
import coldchain_state as cs                                             # noqa: E402
import q3_plan_layer_experimental as PL                                  # noqa: E402
import q3_strict_float_experimental as SF                                # noqa: E402
import q3_thermal_interval_experimental as PRE                           # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
N_ZONES = len(TH.supported_temp_classes)
ZONES = SF.zone_thermals_exact(CONTRACT)
ROWS = []
LIMITS = ('synthetic instances and READ-ONLY frozen-instance census; no plan is installed, no '
          'environment is rolled out, no model runs; a passing row is a DIAGNOSTIC')
INSTANCE_GLOB = ('data/m0_scale/*.npz',
                 'data/archive/baseline_v1_buggy/50_node/test/*.npz',
                 'data/archive/baseline_v1_buggy/100_node/test/*c1*.npz')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-6s %s' % ('PASS' if ok else 'FAIL', cid, name[:96]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------- test contract
def exact_test_contract():
    """A contract whose constants are all exactly representable and give Delta* = 1.0 h per zone.

    h = 0.25, c = (2.0, 5.5, 11.0), A = 25, tau = (18, 4, -18)  =>  margin = c - h*(A-tau) = 0.25
    door_heat = 0.125, capacity = 0.5  =>  delta = 0.25  =>  Delta* = 0.25/0.25 = 1.0 h exactly.
    This is what makes the "segment exactly at the threshold" control an exact equality test rather
    than a test within one ulp of it.
    """
    thermal = dataclasses.replace(
        TH, heat_transfer_per_hour=(0.25, 0.25, 0.25),
        cooling_rate_c_per_hour=(2.0, 5.5, 11.0),
        door_heat_kwh=(0.125, 0.125, 0.125),
        thermal_capacity_kwh_per_c=(0.5, 0.5, 0.5))
    return dataclasses.replace(CONTRACT, thermal=thermal)


# ------------------------------------------------------------------------------- synthetic data
def make_instance(service_times, temp_classes, reveal_times=None, spacing=0.01):
    """Tiny synthetic instance in the frozen NPZ layout (node 0 = depot)."""
    n = len(service_times)
    coords = np.zeros((n, 2), dtype=np.float64)
    coords[0] = (0.5, 0.5)
    for i in range(1, n):
        coords[i] = (0.5 + spacing * i, 0.5)
    return {
        'coords': coords,
        'demands': np.array([0] + [1] * (n - 1)),
        'tw_start': np.zeros(n), 'tw_end': np.full(n, 100.0),
        'service_time': np.array(service_times, dtype=np.float64),
        'temp_class': np.array(temp_classes, dtype=np.int64),
        'reveal_time': np.array(reveal_times if reveal_times is not None else [0.0] * n),
    }


def check(contract, instance, *, clock=0.0, plan, committed=None, revealed=None,
          state_identity='S', expected_state_identity='S', entry=None, contract_identity=None):
    return PL.check_plan_layer(
        contract=contract,
        contract_identity=(contract.contract_hash if contract_identity is None else contract_identity),
        instance=instance, clock=clock, plan_by_vehicle=plan,
        committed_order_ids=committed, revealed_nodes=revealed,
        state_identity=state_identity, expected_state_identity=expected_state_identity,
        initial_temperature_by_vehicle=entry if entry is not None else
        {'v0': tuple(float(t) for t in TH.target_temperature_c)})


# ------------------------------------------------------------------------------- executor probe
def executor_trace(contract, steps, zone_of=None, start_temps=None):
    """Run the real executor over `steps = [(duration_h, kind), ...]`.

    kind: an int zone  -> a pickup segment (door on that zone)
          None         -> a depot-return segment (no door); it CLOSES the vehicle, so it must be last
          'travel'     -> a travel-only segment (no door, no return, state stays open)
    """
    state = cs.VehicleColdChainState(
        compartment_temperature_c=(tuple(start_temps) if start_temps is not None
                                   else (float(TH.ambient_temperature_c),) * N_ZONES),
        zone_load=(0.0,) * N_ZONES)
    state = cs.dispatch_vehicle(state, contract)
    t = 0.0
    out = [tuple(state.compartment_temperature_c)]
    for k, (dur, kind) in enumerate(steps):
        if kind is None:
            state, _m = cs.transition_segment(
                state, depart_time=t, arrival_time=t + dur, service_finish=t + dur,
                served_customer=None, active_zone_mask=(True,) * N_ZONES, contract=contract,
                return_to_depot=True)
        elif kind == 'travel':
            state, _m = cs.transition_segment(
                state, depart_time=t, arrival_time=t + dur, service_finish=t + dur,
                served_customer=None, active_zone_mask=(True,) * N_ZONES, contract=contract)
        else:
            state, _m = cs.transition_segment(
                state, depart_time=t, arrival_time=t, service_finish=t + dur,
                served_customer=k + 1, active_zone_mask=(True,) * N_ZONES, contract=contract,
                order_quantity=1.0, order_temp_class=int(kind), initial_quality=9.0)
        t += dur
        out.append(tuple(state.compartment_temperature_c))
    return out


def band_report(contract, trace):
    zones = SF.zone_thermals_exact(contract)
    bad = []
    for step, temps in enumerate(trace):
        for z in zones:
            lo, hi = float(z.target_c), float(z.target_c + z.delta_real)
            if not (lo - 1e-12 <= temps[z.zone] <= hi + 1e-12):
                bad.append({'step': step, 'zone': z.zone, 't': temps[z.zone], 'band': [lo, hi]})
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-instances-per-file', type=int, default=None)
    args = ap.parse_args()
    src = {
        'scripts/coldchain/q3_plan_layer_experimental.py':
            sha256(EXT / 'scripts/coldchain/q3_plan_layer_experimental.py'),
        'scripts/coldchain/q3_strict_float_experimental.py':
            sha256(EXT / 'scripts/coldchain/q3_strict_float_experimental.py'),
        'scripts/coldchain/coldchain_state.py':
            sha256(EXT / 'scripts/coldchain/coldchain_state.py'),
        'scripts/coldchain/coldchain_contract.py':
            sha256(EXT / 'scripts/coldchain/coldchain_contract.py'),
        'scripts/tests/test_q3_plan_layer_controls.py': sha256(Path(__file__)),
    }
    TC = exact_test_contract()
    TCZ = SF.zone_thermals_exact(TC)
    delta_star = {z.zone: float(z.return_time_h) for z in TCZ}
    entry = {'v0': tuple(float(t) for t in TC.thermal.target_temperature_c)}

    # ============================================================== PL-01 positive control
    inst = make_instance([0.0, 2.0, 2.0], [1, 0, 1], spacing=0.01)
    rec = check(TC, inst, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry)
    ck('PL-01', 'a plan whose every door segment is long enough is PROVED_CONDITIONAL when the '
       'contract, state and commitment bindings are all present',
       rec['verdict'] == PL.PROVED_CONDITIONAL and not rec['refusals'],
       {'verdict': rec['verdict'], 'binding_notes': rec['binding_notes'],
        'delta_star_h': delta_star,
        'per_zone': {k: {kk: v[kk] for kk in ('n_door_segments', 'min_door_segment_h', 'slack_h',
                                              'all_door_segments_long_enough')}
                     for k, v in rec['per_zone'].items()},
        'summary': rec['summary']},
       conclusion='the checker reaches the positive verdict only when every binding is supplied')

    # ============================================================== PL-02 zero service time
    inst0 = make_instance([0.0, 2.0, 0.0], [1, 0, 1], spacing=0.01)
    rec0 = check(TC, inst0, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry)
    codes0 = [r['code'] for r in rec0['refusals']]
    ck('PL-02', 'a served ORDER node with service_time = 0 is REFUSED (the P4 premise is broken), '
       'and the refusal is not confused with the depot',
       rec0['verdict'] == PL.REFUSED and 'SERVED_ORDER_WITH_ZERO_SERVICE_TIME' in codes0,
       {'verdict': rec0['verdict'], 'refusal_codes': codes0, 'refusals': rec0['refusals'],
        'note': 'zero service time is legitimate only for node 0, which is checked separately'},
       conclusion='the outstanding "are the zero entries really the depot?" item is now a '
                  'mechanical check instead of an assumption')

    # ============================================================== PL-03 depot / placeholder
    inst_dep = make_instance([0.0, 2.0, 2.0], [1, 0, 1], spacing=0.01)
    rec_dep = check(TC, inst_dep, plan={'v0': [1, 0, 2]}, committed={1, 2}, entry=entry)
    ck('PL-03', 'node 0 inside a stop list is REFUSED rather than treated as a served order',
       rec_dep['verdict'] == PL.REFUSED
       and 'DEPOT_IN_STOP_LIST' in [r['code'] for r in rec_dep['refusals']],
       {'verdict': rec_dep['verdict'], 'refusals': rec_dep['refusals']},
       conclusion='depot / placeholder identity is checked from the instance, not assumed')

    # ============================================================== PL-04 exact threshold
    inst_eq = make_instance([0.0, delta_star[0], 2.0], [1, 0, 1], spacing=0.0)
    rec_eq = check(TC, inst_eq, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry)
    seg = None
    for row in rec_eq['per_zone'][str(0)]['door_segments']:
        if row['node'] == 1:
            seg = row
    ck('PL-04', 'a door segment whose duration is EXACTLY Delta* passes (`>=`, not `>`)',
       rec_eq['verdict'] == PL.PROVED_CONDITIONAL and seg is not None and seg['ok']
       and Fraction(seg['segment_h']) == Fraction(delta_star[0]),
       {'verdict': rec_eq['verdict'], 'segment': seg, 'delta_star_h': delta_star[0],
        'equality_is_exact': seg is not None and Fraction(seg['segment_h'])
        == Fraction(TCZ[0].return_time_h),
        'why_exact': 'the test contract has exactly representable constants, so Delta* = 1.0 h is an '
                     'exact rational and the boundary is a true equality'},
       conclusion='the threshold is an inclusive boundary and it is decided in exact arithmetic')

    # ============================================================== PL-05 just below threshold
    inst_below = make_instance([0.0, math.nextafter(delta_star[0], 0.0), 2.0], [1, 0, 1],
                               spacing=0.0)
    rec_below = check(TC, inst_below, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry)
    ck('PL-05', 'a door segment one ulp BELOW Delta* makes the plan NOT_CERTIFIABLE',
       rec_below['verdict'] == PL.NOT_CERTIFIABLE
       and rec_below['summary']['short_door_segments'],
       {'verdict': rec_below['verdict'], 'short': rec_below['summary']['short_door_segments'],
        'not_claimed': 'NOT_CERTIFIABLE is not an infeasibility statement: it says this '
                       'certification route is insufficient'},
       conclusion='the checker has a real failure boundary, not just a positive path')

    # ============================================================== PL-06 commitment mismatch
    rec_com = check(TC, inst, plan={'v0': [1, 2]}, committed={1}, entry=entry)
    rec_com2 = check(TC, inst, plan={'v0': [1]}, committed={1, 2}, entry=entry)
    ck('PL-06', 'a plan that serves an unaccepted order, or omits an accepted one, is REFUSED in '
       'both directions',
       rec_com['verdict'] == PL.REFUSED and rec_com2['verdict'] == PL.REFUSED
       and 'COMMITTED_SET_MISMATCH' in [r['code'] for r in rec_com['refusals']]
       and 'COMMITTED_SET_MISMATCH' in [r['code'] for r in rec_com2['refusals']],
       {'extra_service': rec_com['refusals'], 'missing_service': rec_com2['refusals']},
       conclusion='accepted-and-served is checked exactly, not one-sidedly')

    # ============================================================== PL-07 hidden future
    inst_hidden = make_instance([0.0, 2.0, 2.0], [1, 0, 1], reveal_times=[0.0, 0.0, 5.0],
                                spacing=0.01)
    rec_hid = check(TC, inst_hidden, clock=0.0, plan={'v0': [1, 2]}, committed={1, 2},
                    entry=entry)
    ck('PL-07', 'a plan that serves a node not yet revealed at the clock is REFUSED (strict-online, '
       'no hidden future order is read)',
       rec_hid['verdict'] == PL.REFUSED
       and 'SERVES_UNREVEALED_NODE' in [r['code'] for r in rec_hid['refusals']],
       {'verdict': rec_hid['verdict'], 'refusals': rec_hid['refusals'],
        'clock': 0.0, 'reveal_time_of_node_2': 5.0},
       conclusion='the checker cannot be used to certify a plan that peeks into the future')

    # ============================================================== PL-08 cross-vehicle state
    rec_state = check(TC, inst, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry,
                      state_identity='STATE_OF_ANOTHER_VEHICLE',
                      expected_state_identity='EXPECTED_STATE')
    ck('PL-08', 'a state identity that does not match the asserted one is REFUSED (no mixing of '
       'state between vehicles or snapshots)',
       rec_state['verdict'] == PL.REFUSED
       and 'STATE_IDENTITY_MISMATCH' in [r['code'] for r in rec_state['refusals']],
       {'verdict': rec_state['verdict'], 'refusals': rec_state['refusals']},
       conclusion='the receipt is bound to a state digest, so a foreign state cannot be certified')

    # ============================================================== PL-09 contract change
    rec_contract = check(TC, inst, plan={'v0': [1, 2]}, committed={1, 2}, entry=entry,
                         contract_identity=CONTRACT.contract_hash)
    rec_ds = {float(z.return_time_h) for z in ZONES}
    ck('PL-09', 'a contract identity that is not the hash of the contract actually used is REFUSED, '
       'and a contract change moves Delta* so the old conclusion cannot be reused',
       rec_contract['verdict'] == PL.REFUSED
       and 'CONTRACT_IDENTITY_MISMATCH' in [r['code'] for r in rec_contract['refusals']]
       and abs(delta_star[0] - float(ZONES[0].return_time_h)) > 0.1,
       {'verdict': rec_contract['verdict'], 'refusals': rec_contract['refusals'],
        'delta_star_test_contract_h': delta_star,
        'delta_star_default_contract_h': {z.zone: float(z.return_time_h) for z in ZONES},
        'default_contract_delta_star_distinct': len(rec_ds) > 0},
       conclusion='Delta* is derived from the contract, so any parameter change invalidates the '
                  'previous conclusion automatically')

    # ============================================================== PL-10 short NO-door segment
    # a long door segment (returns to tau, then +delta), then a 3-second depot return (no door at all)
    steps_nodoor = [(2.0, 1), (0.000833, None)]
    tr_nodoor = executor_trace(TC, steps_nodoor)
    bad_nodoor = band_report(TC, tr_nodoor)
    inst_r = make_instance([0.0, 2.0], [1, 0], spacing=0.01)   # return leg = 0.02 h << Delta*
    rec_r = check(TC, inst_r, plan={'v0': [1]}, committed={1}, entry=entry)
    ck('PL-10', 'a segment that opens NO door is unconstrained: both a 3-second depot-return segment '
       'and a 0.02 h return leg leave the band intact, so the relaxed proposition certifies plans '
       'the pre-fix P3 would have refused',
       not bad_nodoor and rec_r['verdict'] == PL.PROVED_CONDITIONAL
       and rec_r['summary']['n_short_but_doorless'] > 0,
       {'executor_band_violations': bad_nodoor,
        'trace': [[round(x, 12) for x in t] for t in tr_nodoor],
        'return_segment_h': 0.02, 'delta_star_h': delta_star,
        'verdict': rec_r['verdict'],
        'pre_fix_P3_would_have_refused': True,
        'n_short_but_doorless': rec_r['summary']['n_short_but_doorless']},
       conclusion="P3' is a genuine strengthening of the certification condition, not a weakening "
                  'of safety')

    # ============================================================== PL-11 short DOOR segment
    steps_door = [(0.01, 1), (0.01, 1)]
    tr_door = executor_trace(TC, steps_door)
    bad_door = band_report(TC, tr_door)
    ck('PL-11', 'the SAME short duration is unsafe when the segment opens a door: two 36-second door '
       'segments push the zone out of the band immediately',
       bool(bad_door) and bad_door[0]['zone'] == 1 and bad_door[0]['t'] > bad_door[0]['band'][1],
       {'executor_band_violations': bad_door,
        'trace': [[round(x, 12) for x in t] for t in tr_door],
        'segment_h': 0.01, 'delta_star_h': delta_star[1],
        'why_one_door_is_not_enough': 'the first door starts at tau, so Phi_dt(tau) <= tau and the '
                                      'value lands exactly on tau+delta; the excursion only builds '
                                      'from a value that is ALREADY above tau'},
       conclusion='the relaxed condition is exactly the door/no-door distinction, verified on the '
                  'real executor')

    # ============================================================== PL-12 frozen instance census
    paths = []
    for g in INSTANCE_GLOB:
        paths.extend(sorted(EXT.glob(g)))
    census_rows, files = [], []
    all_depot_only, all_above = True, True
    min_pos = None
    for p in paths:
        raw = dict(np.load(p, allow_pickle=True))
        n_inst = int(np.asarray(raw['service_time']).shape[0])
        limit = n_inst if args.max_instances_per_file is None else min(n_inst,
                                                                     args.max_instances_per_file)
        file_min = None
        non_depot_zero = 0
        for i in range(limit):
            one = {k: (np.asarray(v)[i] if np.asarray(v).ndim >= 2 and np.asarray(v).shape[0] == n_inst
                       else v) for k, v in raw.items()}
            view = PL.InstanceView(one, inst_idx=None)
            zeros = view.zero_service_nodes()
            if zeros != [0]:
                non_depot_zero += 1
            pos = [s for s in view.service_time if s > 0]
            if pos:
                file_min = min(pos) if file_min is None else min(file_min, min(pos))
        rows = {'file': str(p.relative_to(EXT)), 'n_instances': limit,
                'non_depot_zero_service_instances': non_depot_zero,
                'min_positive_service_time_h': file_min,
                'min_positive_service_time_min': None if file_min is None else file_min * 60.0,
                'sha256': sha256(p)}
        census_rows.append(rows)
        files.append(rows['file'])
        all_depot_only = all_depot_only and non_depot_zero == 0
        if file_min is not None:
            min_pos = file_min if min_pos is None else min(min_pos, file_min)
    worst = max(ZONES, key=lambda z: z.return_time_h)
    all_above = min_pos is not None and Fraction(min_pos) >= Fraction(worst.return_time_h)
    ck('PL-12', 'READ-ONLY census of the frozen instance files: every zero service_time entry is the '
       'depot (node 0), and the smallest positive service time is far above the largest Delta*',
       all_depot_only and all_above,
       {'n_files': len(paths), 'files': files,
        'all_zero_service_is_depot_only': all_depot_only,
        'global_min_positive_service_time_h': min_pos,
        'global_min_positive_service_time_min': None if min_pos is None else min_pos * 60.0,
        'max_delta_star_h': float(worst.return_time_h),
        'max_delta_star_min': float(worst.return_time_h) * 60.0,
        'margin_factor': None if min_pos is None else min_pos / float(worst.return_time_h),
        'per_file': census_rows,
        'label': 'DIAGNOSTIC_ONLY (finite sample of frozen files; not a proof of reachability)',
        'scope_limit': 'these files are 50/100-node C1/R1/RC1 with EDoD 0.2/0.5/0.8; other node '
                       'counts and data families are NOT covered'},
       conclusion='the P4 premise holds on every probed instance and the previously outstanding '
                  '"is the zero really the depot?" question is answered: yes, in all probed files')

    # ============================================================== PL-13 P3' vs executor
    steps_mixed = []
    for _ in range(6):
        steps_mixed.append((1.2, 1))         # long door segment -> returns to tau, then +delta
        steps_mixed.append((0.0005, 'travel'))   # 1.8-second travel-only segment, opens nothing
    tr_mixed = executor_trace(TC, steps_mixed)
    bad_mixed = band_report(TC, tr_mixed)
    tr_eps_nodoor = executor_trace(TC, [(2.0, 1), (1e-12, 'travel'), (2.0, 1), (0.0, None)])
    bad_eps_nodoor = band_report(TC, tr_eps_nodoor)
    tr_eps_door = executor_trace(TC, [(2.0, 1), (0.0, 1)])
    bad_eps_door = band_report(TC, tr_eps_door)
    ck('PL-13', "interleaving arbitrarily short NO-DOOR segments (including the Delta <= _EPS branch) "
       'between long door segments keeps the executor inside the band, while the same branch on a '
       'DOOR segment leaves it -- which is exactly the split the proof requires',
       not bad_mixed and not bad_eps_nodoor and bool(bad_eps_door),
       {'n_steps': len(steps_mixed), 'band_violations_mixed': bad_mixed,
        'eps_branch_no_door_violations': bad_eps_nodoor,
        'eps_branch_door_violations': bad_eps_door,
        'trace_tail': [[round(x, 12) for x in t] for t in tr_mixed[-4:]],
        'label': 'EXECUTOR_REPRODUCED (fixed synthetic sequences)'},
       conclusion='the proof and the executor agree on the relaxed condition, including the '
                  'Delta <= _EPS branch in both directions')

    # ============================================================== PL-14 refuted claim preserved
    two_door = executor_trace(CONTRACT, [(0.01, 1), (0.01, 1)])
    nine_door = executor_trace(CONTRACT, [(0.01, 1)] * 9)
    z1 = ZONES[1]
    bound = z1.target_c + float(z1.delta_real)
    ck('PL-14', 'the two-door and nine-door counterexamples to the OLD Theorem B are preserved: a '
       'short DOOR segment sequence still leaves the band, so nothing here restores the refuted claim',
       two_door[-1][1] > bound and nine_door[-1][1] > two_door[-1][1] > bound,
       {'zone1_tau_plus_delta_c': bound, 'after_2_doors_c': two_door[-1][1],
        'after_9_doors_c': nine_door[-1][1],
        'module_guard': {'refuted_proposition_used': False,
                         'forbidden': list(PL.FORBIDDEN_PROPOSITIONS)},
        'label': 'MATHEMATICAL_COUNTEREXAMPLE + EXECUTOR_REPRODUCED'},
       conclusion='the refutation stands; the checker is not allowed to use Theorem B, B1 or B2 '
                  'sufficiency and records that in every receipt')

    payload = {
        'schema': 'q3_plan_layer_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'test_contract_hash': TC.contract_hash,
        'test_contract_delta_star_h': delta_star,
        'default_contract_delta_star_h': {z.zone: float(z.return_time_h) for z in ZONES},
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('PLAN_LAYER_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'PLAN_LAYER_CONTROLS_FAIL'),
        'propositions_used': list(PL.ALLOWED_PROPOSITIONS),
        'propositions_forbidden': list(PL.FORBIDDEN_PROPOSITIONS),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
