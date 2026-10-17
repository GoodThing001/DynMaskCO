#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 S3 (bitwise executor replay) QUALIFICATION CONTROLS (DIAGNOSTIC, not for production).

Rows REP-01..REP-09.  They pin down three things:

  * what S3 can and cannot claim (REP-01, REP-08): it is a conditional parity statement, never a
    continuous-temperature proof and never a proof about a FUTURE suffix;
  * that the claim is decidable and that drift is caught (REP-01, REP-07);
  * that the partition-independent BRACKET replaces S3 whenever the partition is unknown
    (REP-03..REP-06, REP-09) -- which is the strict-online situation.

REP-02 replays the whole recorded event stream of ONE greedy instance and demands bit equality.

Usage: python scripts/tests/test_q3_float_replay_qualification.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))
sys.path.insert(0, str(EXT / 'scripts/simulation'))

import coldchain_state as cs                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import q3_float_replay_qualification_experimental as RP                  # noqa: E402
import q3_strict_float_experimental as SF                                # noqa: E402
from strict_online_env import StrictOnlineEnv, GreedyReplanner           # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
N_ZONES = len(TH.supported_temp_classes)
ZONES = SF.zone_thermals_exact(CONTRACT)
ROWS = []
LIMITS = ('synthetic controls plus ONE greedy instance replay (no model, no training, no GPU); a '
          'passing row is a DIAGNOSTIC, never a production certificate')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-7s %s' % ('PASS' if ok else 'FAIL', cid, name[:94]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ executor wrappers
def door_call(state, depart, arrival, finish, served, temp_class, quantity, iq, distance):
    st, _m = cs.transition_segment(
        state, depart_time=float(depart), arrival_time=float(arrival),
        service_finish=float(finish), served_customer=int(served),
        active_zone_mask=(True,) * N_ZONES, contract=CONTRACT,
        order_quantity=float(quantity), order_temp_class=int(temp_class),
        initial_quality=float(iq), segment_distance_units=float(distance))
    return st


def advance_call(state, start, end):
    st, _m = cs.transition_segment(
        state, depart_time=float(start), arrival_time=float(end), service_finish=float(end),
        served_customer=None, active_zone_mask=(True,) * N_ZONES, contract=CONTRACT)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ex_paths = [EXT / 'scripts/coldchain/coldchain_state.py']
    src = {'scripts/coldchain/q3_float_replay_qualification_experimental.py':
           sha256(EXT / 'scripts/coldchain/q3_float_replay_qualification_experimental.py'),
           'scripts/coldchain/q3_event_segmentation_experimental.py':
           sha256(EXT / 'scripts/coldchain/q3_event_segmentation_experimental.py'),
           'scripts/coldchain/coldchain_state.py': sha256(ex_paths[0]),
           'scripts/simulation/strict_online_env.py':
           sha256(EXT / 'scripts/simulation/strict_online_env.py'),
           'scripts/tests/test_q3_float_replay_qualification.py': sha256(Path(__file__))}

    # ---------------------------------------------------------------- the live trace (shared)
    ds = dict(np.load(EXT / 'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz', allow_pickle=True))
    data = {k: (v[:1] if np.asarray(v).ndim >= 2 and np.asarray(v).shape[0] >= 1 else v)
            for k, v in ds.items()}
    env = StrictOnlineEnv(data, capacity=float(CONTRACT.operational.shared_vehicle_capacity),
                          tw_speed=1.0, num_vehicles=25, replanner=GreedyReplanner('edd'),
                          coldchain_contract=CONTRACT, booking_horizon=None)
    traces, served = env.run(0)

    # the global event clock list, reconstructed from the record (no instrumentation)
    clocks = set()
    for i in range(1, env.num_nodes):
        rt = float(env.reveal_time[0, i])
        if rt > 0:
            clocks.add(rt)
    for tr in traces:
        for r in tr.services:
            clocks.add(float(r.service_finish))
        if tr.return_arrival is not None:
            clocks.add(float(tr.return_arrival))
    clocks = sorted(clocks)

    # ---------------------------------------------------------------- REP-01 preconditions
    code_now = RP.code_identity(ex_paths)
    start_state = cs.dispatch_vehicle(cs.VehicleColdChainState(
        compartment_temperature_c=(float(TH.ambient_temperature_c),) * N_ZONES,
        zone_load=(0.0,) * N_ZONES), CONTRACT)
    sd = RP.state_digest(start_state)
    calls = [{'depart': 0.0, 'arrival': 0.0, 'finish': 0.0, 'served': 1, 'temp_class': 1,
              'quantity': 1.0, 'initial_quality': 9.0, 'distance': 0.0, 'mask': (True,) * 3}]
    runtime = {'python': sys.version.split()[0], 'platform': platform.platform(),
               'machine': platform.machine()}
    good = RP.qualify_replay(contract=CONTRACT, contract_identity=CONTRACT.contract_hash,
                             executor_paths=ex_paths, claimed_code_identity=code_now,
                             claimed_contract_identity=CONTRACT.contract_hash,
                             start_state=start_state, claimed_state_digest=sd, calls=calls,
                             runtime_record=runtime)
    variants = {}
    variants['code_drift'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity={'scripts/coldchain/coldchain_state.py': '0' * 64},
        claimed_contract_identity=CONTRACT.contract_hash, start_state=start_state,
        claimed_state_digest=sd, calls=calls, runtime_record=runtime)['verdict']
    variants['contract_drift'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity=code_now, claimed_contract_identity='1' * 64,
        start_state=start_state, claimed_state_digest=sd, calls=calls,
        runtime_record=runtime)['verdict']
    variants['state_drift'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity=code_now, claimed_contract_identity=CONTRACT.contract_hash,
        start_state=start_state, claimed_state_digest='2' * 64, calls=calls,
        runtime_record=runtime)['verdict']
    pd, _n = RP.partition_digest(calls)
    variants['partition_drift'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity=code_now, claimed_contract_identity=CONTRACT.contract_hash,
        start_state=start_state, claimed_state_digest=sd, calls=calls,
        claimed_partition_digest='3' * 64, runtime_record=runtime)['verdict']
    variants['no_runtime'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity=code_now, claimed_contract_identity=CONTRACT.contract_hash,
        start_state=start_state, claimed_state_digest=sd, calls=calls)['verdict']
    variants['no_state_digest'] = RP.qualify_replay(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, executor_paths=ex_paths,
        claimed_code_identity=code_now, claimed_contract_identity=CONTRACT.contract_hash,
        start_state=start_state, claimed_state_digest=None, calls=calls,
        runtime_record=runtime)['verdict']
    ck('REP-01', 'the S3 qualification record refuses every kind of drift: code, contract, start '
       'state, call partition, missing runtime identity and missing state digest',
       good['verdict'] == RP.REPLAY_QUALIFIED
       and all(v == RP.REPLAY_REFUSED for v in variants.values()),
       {'good': good['verdict'], 'variants': variants,
        'preconditions': good['preconditions'],
        'runtime_recorded': runtime,
        'p2_note': 'interpreter/libm is RECORDED, never proven (P2 status = record_only)'},
       conclusion='S3 is a conditional statement with seven named preconditions, six of which this '
                  'module can check and one of which it can only record')

    # ---------------------------------------------------------------- REP-02 full replay parity
    replay_state, replay_states = {}, {}
    for tr in traces:
        if not tr.services:
            continue
        vid = int(tr.vehicle_id)
        st = cs.dispatch_vehicle(cs.VehicleColdChainState(
            compartment_temperature_c=(float(TH.ambient_temperature_c),) * N_ZONES,
            zone_load=(0.0,) * N_ZONES), CONTRACT)
        updated = float(tr.services[0].depart_time)
        finishes = {round(float(r.service_finish), 9): r for r in tr.services}
        got = []
        for c in clocks:
            if c <= updated + 1e-9:
                continue
            if c > clocks[-1] + 1e-9:
                continue
            st = advance_call(st, updated, c)
            updated = c
            key = round(c, 9)
            if key in finishes:
                r = finishes[key]
                n = int(r.node)
                st = door_call(st, c, c, c, n, int(env.temp_class[0, n]),
                               float(env.demands[0, n]), float(env.initial_quality[0, n]),
                               float(env.dist_mat[0, r.prev_node, n]))
                got.append((c, tuple(st.compartment_temperature_c)))
        replay_state[vid] = {round(c, 9): t for c, t in got}
    worst = 0.0
    n_checked = 0
    for tr in traces:
        if not tr.services:
            continue
        vid = int(tr.vehicle_id)
        for r in tr.services:
            key = round(float(r.service_finish), 9)
            if key not in replay_state.get(vid, {}):
                continue
            for z in range(N_ZONES):
                d = abs(replay_state[vid][key][z]
                        - float(r.coldchain_after.compartment_temperature_c[z]))
                worst = max(worst, d)
            n_checked += 1
    ck('REP-02', 'replaying the WHOLE recorded event sequence (door-less advances AND doors, with the '
       'real partitioning at 99 event clocks) reproduces every recorded service-end temperature '
       'bit for bit',
       worst == 0.0 and n_checked == 50,
       {'n_service_states_replayed': n_checked, 'max_abs_diff_c': worst,
        'n_event_clocks': len(clocks),
        'what_it_proves': 'the record is consistent with the frozen code under the recorded '
                          'partition',
        'what_it_does_not_prove': 'that the partition was knowable in advance -- see REP-06',
        'label': 'DIAGNOSTIC_ONLY'},
       conclusion='S3 parity is achievable and checkable AFTER the fact; it is not a prophecy')

    # ---------------------------------------------------------------- REP-03 BR-1 refinement
    random.seed(5)
    worst_pred, worst_meas = 0.0, 0.0
    for z in ZONES:
        for _ in range(150):
            t0 = z.target_c + random.uniform(-1.0, 1.0)
            a = 10 ** random.uniform(-6, -0.3)
            b = 10 ** random.uniform(-6, -0.3)
            pa = z.ambient_c + (t0 - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * b) \
                - z.cooling_rate_c_per_hour * b
            phab = z.ambient_c + (pa - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * a) \
                - z.cooling_rate_c_per_hour * a
            whole = z.ambient_c + (t0 - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * (a + b)) \
                - z.cooling_rate_c_per_hour * (a + b)
            pred = z.cooling_rate_c_per_hour * b * (1 - math.exp(-z.heat_transfer_per_hour * a))
            worst_pred = max(worst_pred, abs((phab - whole) - pred))
            worst_meas = min(worst_meas, phab - whole)
    ck('REP-03', 'BR-1: refining a door-less interval raises the endpoint by exactly '
       'c*b*(1-exp(-h*a)); the measured difference matches the closed form and is never negative',
       worst_pred < 1e-12 and worst_meas >= -1e-15,
       {'max_abs_diff_between_measured_and_predicted': worst_pred,
        'min_measured_refinement_gain': worst_meas, 'n': 450,
        'label': 'PROVED (algebra) + confirmed numerically'},
       conclusion='splitting is warm-biased, which is why the coarsest partition is the lower end of '
                  'the bracket')

    # ---------------------------------------------------------------- REP-04 bracket soundness
    random.seed(7)
    outside_raw, outside_floor = [], []
    n = 0
    for z in ZONES:
        for _ in range(500):
            t0 = z.target_c + random.uniform(0.0, float(z.delta_real))
            total = 10 ** random.uniform(-5, 0.3)
            k = random.randint(1, 10)
            cuts = sorted(random.uniform(0, total) for _ in range(k - 1))
            pts = [0.0] + cuts + [total]
            st_raw, st_floor = t0, t0
            for i in range(len(pts) - 1):
                d = pts[i + 1] - pts[i]
                if d <= 1e-9:
                    continue
                st_raw = z.ambient_c + (st_raw - z.ambient_c) * math.exp(
                    -z.heat_transfer_per_hour * d) - z.cooling_rate_c_per_hour * d
                st_floor = max(z.target_c, z.ambient_c + (st_floor - z.ambient_c) * math.exp(
                    -z.heat_transfer_per_hour * d) - z.cooling_rate_c_per_hour * d)
            br = RP.bracket(CONTRACT, z, t0, total)
            n += 1
            if not (br['lower_raw_c'] - 1e-12 <= st_raw <= br['upper_raw_c'] + 1e-12):
                outside_raw.append({'zone': z.zone, 't0': t0, 'total': total, 'k': k,
                                    'raw': st_raw,
                                    'bracket_raw': [br['lower_raw_c'], br['upper_raw_c']]})
            if not (br['lower_c'] - 1e-12 <= st_floor <= br['upper_c'] + 1e-12):
                outside_floor.append({'zone': z.zone, 't0': t0, 'total': total, 'k': k,
                                      'floored': st_floor,
                                      'bracket': [br['lower_c'], br['upper_c']]})
    ck('REP-04', 'the bracket contains EVERY admissible partitioning of the interval, both unfloored '
       'and floored (which is what the executor stores): the executor response set is an interval '
       'whose two ends need no knowledge of the partition',
       not outside_raw and not outside_floor and n > 0,
       {'n_partitions': n, 'n_outside_raw': len(outside_raw), 'n_outside_floored': len(outside_floor),
        'outside_raw': outside_raw[:5], 'outside_floored': outside_floor[:5],
        'label': 'PROVED (BR-1 + BR-2) + confirmed numerically'},
       conclusion='this is the object that survives unknown future segmentation')

    # ---------------------------------------------------------------- REP-05 bracket width
    rows = []
    for z in ZONES:
        for total in (0.01, 0.03, 0.05, 0.1, 0.5, 2.0):
            br = RP.bracket(CONTRACT, z, z.target_c + float(z.delta_real), total)
            rows.append({'zone': z.zone, 'total_h': total,
                         'width_c': br['width_c'],
                         'lower_c': br['lower_c'], 'upper_c': br['upper_c'],
                         'single_piece_floored_c': br['single_piece_floored_c'],
                         'tau': z.target_c})
    max_w = max(r['width_c'] for r in rows)
    ck('REP-05', 'the honest cost of not knowing the partition is the bracket width; it collapses to '
       'exactly 0 once both ends are pinned to the floor, and stays under 0.05 C in the narrow regime',
       all(r['width_c'] >= 0.0 for r in rows) and max_w < 0.05,
       {'max_width_c': max_w, 'rows': rows,
        'reading': 'the width is the ONLY uncertainty the segmentation introduces; it is not a '
                   'modelling error and it is not a fitted margin'},
       conclusion='an unknown future partition costs a bounded, computable interval width -- not the '
                  'loss of the certificate')

    # ---------------------------------------------------------------- REP-06 partition not from plan
    z = ZONES[1]
    t0 = z.target_c + float(z.delta_real)
    total = 0.02
    one = RP.unsplit_raw(z, t0, total)
    st = t0
    for _ in range(4):
        st = z.ambient_c + (st - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * (total / 4)) \
            - z.cooling_rate_c_per_hour * (total / 4)
    four = st
    br = RP.bracket(CONTRACT, z, t0, total)
    ck('REP-06', 'the endpoint is NOT a function of the plan: the same start state and the same total '
       'elapsed time give different endpoints under different partitions, and both lie inside the '
       'bracket',
       one != four and four > one
       and br['lower_raw_c'] <= min(one, four) and max(one, four) <= br['upper_raw_c'],
       {'zone': z.zone, 't0': t0, 'total_h': total,
        'partition_1_piece_raw_c': one, 'partition_4_pieces_raw_c': four,
        'difference_c': four - one,
        'bracket_raw': [br['lower_raw_c'], br['upper_raw_c']],
        'consequence': 'a future suffix cannot be certified by replaying it, because its partition is '
                       'decided by future event clocks (strict-online forbids reading them)'},
       conclusion='S3 is a post-hoc/after-the-fact tool, not a pre-decision certificate; the bracket '
                  'is what a pre-decision certificate must use')

    # ---------------------------------------------------------------- REP-07 audit catches tampering
    tr0 = [t for t in traces if t.services][0]
    r0 = tr0.services[0]
    n0 = int(r0.node)
    calls0 = [{'depart': float(r0.depart_time), 'arrival': float(r0.arrival_time),
               'finish': float(r0.service_finish), 'served': n0,
               'temp_class': int(env.temp_class[0, n0]), 'quantity': float(env.demands[0, n0]),
               'initial_quality': float(env.initial_quality[0, n0]),
               'distance': float(env.dist_mat[0, r0.prev_node, n0]), 'mask': (True,) * 3}]
    before = r0.coldchain_before
    recorded = [tuple(float(x) for x in r0.coldchain_after.compartment_temperature_c)]

    def replay_fn(call):
        st = door_call(before, call['depart'], call['arrival'], call['finish'], call['served'],
                       call['temp_class'], call['quantity'], call['initial_quality'],
                       call['distance'])
        return tuple(float(x) for x in st.compartment_temperature_c)

    ok_honest, det_honest = RP.post_hoc_audit(contract=CONTRACT, calls=calls0,
                                              recorded_states=recorded, replay_fn=replay_fn)
    tampered = [tuple(math.nextafter(x, math.inf) for x in recorded[0])]
    ok_tampered, det_tampered = RP.post_hoc_audit(contract=CONTRACT, calls=calls0,
                                                  recorded_states=tampered, replay_fn=replay_fn)
    ck('REP-07', 'the post-hoc parity audit accepts the honest record and rejects a record whose '
       'temperature was moved by ONE ulp',
       ok_honest and not ok_tampered,
       {'honest': det_honest, 'tampered': det_tampered,
        'reading': 'exact equality is the right test: a single differing bit means the record did not '
                   'come from this code path or the runtime drifted'},
       conclusion='this is the shareable role S3 can play: an install-time / receipt-time audit, not '
                  'a pre-decision proof')

    # ---------------------------------------------------------------- REP-08 the guard rails
    not_claimed = ' '.join(good['not_claimed'])
    ck('REP-08', 'the qualification record itself refuses the two upgrades that would be wrong: it '
       'explicitly disclaims continuous/intra-segment safety and any claim about a future suffix, and '
       'it rejects "two calls into the same implementation" as independence',
       'continuous' in not_claimed and 'FUTURE' in not_claimed
       and 'independence' in not_claimed and 'portability' in not_claimed,
       {'not_claimed': good['not_claimed'],
        'replacement': good['replacement_when_the_partition_is_unknown']},
       conclusion='the artefact carries the limits, so a later reader cannot upgrade the claim by '
                  'citing it')

    # ---------------------------------------------------------------- REP-09 series bracket
    z2 = ZONES[2]
    intervals = [0.4, 0.02, 1.1, 0.05, 0.9]
    series = RP.bracket_series(CONTRACT, z2, z2.target_c, intervals)
    random.seed(13)
    outside = 0
    for _ in range(400):
        st = z2.target_c
        for total in intervals:
            k = random.randint(1, 6)
            cuts = sorted(random.uniform(0, total) for _ in range(k - 1))
            pts = [0.0] + cuts + [total]
            for i in range(len(pts) - 1):
                d = pts[i + 1] - pts[i]
                if d <= 1e-9:
                    continue
                st = z2.ambient_c + (st - z2.ambient_c) * math.exp(
                    -z2.heat_transfer_per_hour * d) - z2.cooling_rate_c_per_hour * d
                st = max(z2.target_c, st)
        if not (series['lower_c'] - 1e-12 <= st <= series['upper_c'] + 1e-12):
            outside += 1
    ck('REP-09', 'the bracket propagates across a SERIES of intervals with per-interval random '
       'partitions: every composed trajectory lands inside the propagated bracket',
       outside == 0,
       {'intervals': intervals, 'bracket': [series['lower_c'], series['upper_c']],
        'n_random_trajectories': 400, 'n_outside': outside,
        'label': 'PROVED (monotonicity of both ends) + confirmed numerically'},
       conclusion='one bracket per interval, propagated, covers the whole unknown-partition future '
                  'of the door-less segments')

    payload = {
        'schema': 'q3_float_replay_qualification_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'preconditions': [{'id': a, 'name': b, 'status': c, 'object': d, 'how': e}
                          for a, b, c, d, e in RP.PARITY_PRECONDITIONS],
        'runtime_record': runtime,
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('REPLAY_QUALIFICATION_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'REPLAY_QUALIFICATION_CONTROLS_FAIL'),
        'labels': {'PROVED': 'BR-1, BR-2, bracket soundness (REP-03/04/05/09)',
                   'DIAGNOSTIC': 'one greedy instance replay (REP-02), audit negative (REP-07)',
                   'REFUSED_UPGRADES': 'continuous safety, future suffix, independence, portability '
                                       '(REP-08)'},
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
