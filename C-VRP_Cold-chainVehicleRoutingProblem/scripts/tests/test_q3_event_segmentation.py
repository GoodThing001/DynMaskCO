#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 EVENT-SEGMENTATION CONTROLS (DIAGNOSTIC, not for production).

Rows SEG-01..SEG-12.  The set exists to make three things checkable rather than arguable:

  * SEG-01 reads `strict_online_env.py` and confirms TEXTUALLY that the door is applied in a
    zero-duration `transition_segment` call after a door-less advance (the premise of the whole
    round);
  * SEG-05 exhibits the window in which the pre-fix single-segment formula and the real split
    composition disagree, i.e. the exact reason the old mapping cannot be reused;
  * SEG-07 runs ONE instance with the greedy baseline (no model, no training) and audits the
    environment's own recorded event stream with P5.

Usage: python scripts/tests/test_q3_event_segmentation.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))
sys.path.insert(0, str(EXT / 'scripts/simulation'))

import coldchain_state as cs                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import q3_event_segmentation_experimental as ES                          # noqa: E402
import q3_strict_float_experimental as SF                                # noqa: E402
from strict_online_env import StrictOnlineEnv, GreedyReplanner           # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
ZONES = SF.zone_thermals_exact(CONTRACT)
N_ZONES = len(ZONES)
ROWS = []
LIMITS = ('synthetic controls plus ONE greedy instance trace (no model, no training, no GPU); read '
          'only; a passing row is a DIAGNOSTIC, never a gain or a production certificate')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-7s %s' % ('PASS' if ok else 'FAIL', cid, name[:94]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    env_src = EXT / 'scripts/simulation/strict_online_env.py'
    state_src = EXT / 'scripts/coldchain/coldchain_state.py'
    src = {'scripts/coldchain/q3_event_segmentation_experimental.py':
           sha256(EXT / 'scripts/coldchain/q3_event_segmentation_experimental.py'),
           'scripts/simulation/strict_online_env.py': sha256(env_src),
           'scripts/coldchain/coldchain_state.py': sha256(state_src),
           'scripts/coldchain/coldchain_contract.py':
           sha256(EXT / 'scripts/coldchain/coldchain_contract.py'),
           'scripts/tests/test_q3_event_segmentation.py': sha256(Path(__file__))}
    th = ES.thresholds(CONTRACT)

    # ============================================================ SEG-01 the source premise
    text = env_src.read_text(encoding='utf-8')
    # locate the door call inside _advance_fleet
    m = re.search(r'v\.coldchain_state, event_metrics = transition_segment\((.*?)\n\s*\)\n',
                  text, re.S)
    door_call = m.group(1) if m else ''
    adv = re.search(r'def _advance_coldchain_to.*?\n\n', text, re.S)
    adv_text = adv.group(0) if adv else ''
    door_zero = (re.search(r'depart_time=float\(clock\)', door_call) is not None
                 and re.search(r'arrival_time=float\(clock\)', door_call) is not None
                 and re.search(r'service_finish=float\(clock\)', door_call) is not None
                 and re.search(r'served_customer=int\(node\)', door_call) is not None)
    advance_nodoor = (re.search(r'served_customer=None', adv_text) is not None
                      and re.search(r'transition_segment\(', adv_text) is not None)
    eps_short = 'if duration_h <= _EPS:' in state_src.read_text(encoding='utf-8')
    ck('SEG-01', 'the environment applies the door in a transition_segment call whose three time '
       'arguments are all `clock` (duration 0), after a door-less advance; the executor short-'
       'circuits duration <= _EPS to the identity',
       door_zero and advance_nodoor and eps_short,
       {'door_call_excerpt': door_call.strip()[:420],
        'advance_is_doorless': advance_nodoor,
        'eps_shortcircuit_present': eps_short,
        'door_segment_duration_h': 0.0,
        'where': {'door': 'strict_online_env.py:318-331', 'advance': ':209-230',
                  'shortcircuit': 'coldchain_state.py:361-362'}},
       conclusion='the user-reported call chain is confirmed from the source text, not from a '
                  'summary; the door segment duration is 0 for every door')

    # ============================================================ SEG-02 theta closed form
    rows = []
    ok2 = True
    for z in ZONES:
        # independent numeric root of the exact ODE flow
        lo, hi = 1e-9, 10.0
        f = lambda t: ES.ode_flow(z, z.target_c + float(z.delta_real), t) - z.target_c
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        err = abs(hi - th[z.zone]['theta_ode_h'])
        ok2 = ok2 and err < 1e-10
        rows.append({'zone': z.zone, 'theta_closed_h': th[z.zone]['theta_ode_h'],
                     'theta_numeric_root_h': hi, 'abs_diff': err})
    ck('SEG-02', 'the closed-form Theta_z agrees with an independent bisection of the exact ODE flow',
       ok2, {'rows': rows},
       conclusion='Theta is a well-defined threshold, not a fitted constant')

    # ============================================================ SEG-03 L-F single step
    random.seed(11)
    worst = 0.0
    worst_predicted = 0.0
    worst_above_floor = 0.0
    n_above_floor = 0
    for z in ZONES:
        for _ in range(200):
            t0 = z.target_c + random.uniform(-1.0, 2.0)
            d = 10 ** random.uniform(-9, 0.5)
            phi = z.ambient_c + (t0 - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * d) \
                - z.cooling_rate_c_per_hour * d
            gap = phi - ES.ode_flow(z, t0, d)
            worst = max(worst, gap)
            # numerically stable form of the algebraic prediction: (c/h)[(1-e^{-h d}) - h d],
            # which is <= 0 for every d >= 0 because 1 - e^{-x} <= x.  expm1 keeps the small-d case
            # out of the cancellation regime that made the first version of this control look wrong.
            x = z.heat_transfer_per_hour * d
            pred = (z.cooling_rate_c_per_hour / z.heat_transfer_per_hour) * (
                -math.expm1(-x) - x)
            worst_predicted = max(worst_predicted, pred)
            if abs(pred) > 1e-12:            # only where the gap is above the comparison roundoff
                n_above_floor += 1
                worst_above_floor = max(worst_above_floor, gap)
    ck('SEG-03', 'L-F: the executor single step is dominated by the ODE flow '
       '(Phi_dt(T) <= F_dt(T)); the closed-form gap (c/h)[(1-e^{-h dt}) - h dt] is never positive, '
       'and the measured gap is never positive above the comparison roundoff floor',
       worst_predicted <= 0.0 and worst_above_floor <= 0.0 and worst <= 1e-13,
       {'max_predicted_gap': worst_predicted,
        'max_measured_gap_above_roundoff_floor': worst_above_floor,
        'n_measurable': n_above_floor,
        'max_measured_gap_including_roundoff': worst,
        'roundoff_floor': 'the direct comparison subtracts two numbers of magnitude ~|T0|, so gaps '
                          'below ~1e-13 are not resolvable; the algebraic form is used there',
        'algebra': 'Phi - F = (c/h)*((1 - e^{-h dt}) - h dt) <= 0 because 1 - e^{-x} <= x',
        'label': 'PROVED (algebra) + confirmed numerically'},
       conclusion='the executor under-shoots the exact ODE flow at every single step')

    # ============================================================ SEG-04 L-G partitions
    random.seed(3)

    def compose_raw(z, t0, parts):
        t = t0
        for d in parts:
            if d <= 1e-9:
                continue
            t = z.ambient_c + (t - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * float(d)) \
                - z.cooling_rate_c_per_hour * float(d)
        return t

    worst_raw, worst_floor = 0.0, 0.0
    n = 0
    for z in ZONES:
        for _ in range(400):
            t0 = z.target_c + random.uniform(0.0, float(z.delta_real))
            total = 10 ** random.uniform(-4, 0.5)
            k = random.randint(1, 12)
            cuts = sorted(random.uniform(0, total) for _ in range(k - 1))
            pts = [0.0] + cuts + [total]
            parts = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
            raw = compose_raw(z, t0, parts)
            ub = ES.ode_flow(z, t0, total)
            n += 1
            worst_raw = max(worst_raw, raw - ub)
            worst_floor = max(worst_floor, max(z.target_c, raw) - max(z.target_c, ub))
    ck('SEG-04', 'L-G: for RANDOM partitions of the same total door-less duration the composition '
       'never exceeds the ODE flow, and the floored composition never exceeds max(tau, flow)',
       worst_raw <= 0.0 and worst_floor <= 0.0,
       {'n_partitions': n, 'max_raw_minus_flow': worst_raw,
        'max_floored_minus_max_tau_flow': worst_floor,
        'label': 'PROVED (monotone telescoping) + confirmed numerically'},
       conclusion='the ODE flow is a partition-independent upper bound, which is what makes P5 '
                  'checkable without knowing how the environment splits the advance')

    # ============================================================ SEG-05 the unsound window
    rows = []
    ok5 = True
    for z in ZONES:
        ts = th[z.zone]['t_single_h']
        theta = th[z.zone]['theta_ode_h']
        total = ts + 0.2 * (theta - ts)
        div = ES.split_divergence(CONTRACT, z.zone, total, 8)
        single_at_floor = div['single_c'] <= z.target_c + 1e-15
        split_above = div['split_c'] > z.target_c
        ok5 = ok5 and single_at_floor and split_above
        rows.append({'zone': z.zone, 'total_h': total,
                     't_single_h': ts, 'theta_h': theta,
                     'single_c': div['single_c'], 'split8_c': div['split_c'],
                     'ode_upper_c': div['ode_upper_c'],
                     'single_says_on_floor': single_at_floor,
                     'split_says_still_above_floor': split_above,
                     'window_h': theta - ts, 'window_s': (theta - ts) * 3600.0})
    ck('SEG-05', 'the pre-fix single-segment formula is NOT sound for the environment: on a total '
       'interval between T_single and Theta it reports "back on the floor" while the split '
       'composition is still above it, so the next door would exceed tau+delta',
       ok5, {'rows': rows,
             'reading': 'the same plan, the same total elapsed time, two segmentations, two '
                        'different answers -- this is why a certificate must name its segmentation'},
       conclusion='the old Delta*/service-time mapping cannot be transferred to the environment '
                  'unchanged')

    # ============================================================ SEG-06 old P3 on env segments
    diag = ES.old_p3_on_env_segments(CONTRACT)
    ck('SEG-06', "P3' applied LITERALLY to the environment's own segment sequence declares every "
       'plan NOT_CERTIFIABLE, because the door segment duration is 0 for every door',
       diag['verdict_if_p3_applied_literally'] == ES.NOT_CERTIFIABLE
       and all(not r['predicate_duration_ge_delta_star'] for r in diag['rows']),
       {'rows': diag['rows'], 'why': diag['why'],
        'what_this_means': "P3' is not refuted as a theorem; it is attached to the wrong object"},
       conclusion='the previous round\'s plan-layer verdict is about a model timeline, not about '
                  'the environment; it must not be read as an environmental certificate')

    # ============================================================ SEG-07 live event-stream audit
    ds = dict(np.load(EXT / 'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz', allow_pickle=True))
    data = {k: (v[:1] if np.asarray(v).ndim >= 2 and np.asarray(v).shape[0] >= 1 else v)
            for k, v in ds.items()}
    env = StrictOnlineEnv(data, capacity=float(CONTRACT.operational.shared_vehicle_capacity),
                          tw_speed=1.0, num_vehicles=25, replanner=GreedyReplanner('edd'),
                          coldchain_contract=CONTRACT, booking_horizon=None)
    traces, served = env.run(0)
    stream = ES.event_stream_from_traces(traces, env, 0)
    rec = ES.check_event_stream(contract=CONTRACT, contract_identity=CONTRACT.contract_hash,
                                event_stream=stream, state_identity='S', expected_state_identity='S')
    # independent re-derivation of the door clocks and of the splitting
    event_clocks = set()
    for i in range(1, env.num_nodes):
        if float(env.reveal_time[0, i]) > 0:
            event_clocks.add(float(env.reveal_time[0, i]))
    for tr in traces:
        for r in tr.services:
            event_clocks.add(float(r.service_finish))
        if tr.return_arrival is not None:
            event_clocks.add(float(tr.return_arrival))
    event_clocks = sorted(event_clocks)
    n_split_pairs = 0
    for vid, rows_v in stream.items():
        for a, b in zip(rows_v, rows_v[1:]):
            inner = [c for c in event_clocks if a['door_clock'] + 1e-12 < c < b['door_clock'] - 1e-12]
            if inner:
                n_split_pairs += 1
    ck('SEG-07', 'P5 audit of the environment\'s own recorded event stream: every inter-door interval '
       'on the same zone is far above Theta, and the door-less advance really is split at '
       'intermediate event clocks',
       rec['verdict'] == ES.PROVED_CONDITIONAL and rec['summary']['n_intervals'] > 0
       and rec['summary']['min_interval_h'] > rec['summary']['min_theta_h']
       and n_split_pairs > 0,
       {'verdict': rec['verdict'], 'served': int(served.sum()) - 1,
        'summary': rec['summary'], 'per_zone': {k: {'n_doors': v['n_doors'],
                                                    'n_intervals': v['n_intervals'],
                                                    'min_interval_min': v['min_interval_min'],
                                                    'theta_ode_min': v['theta_ode_min'],
                                                    'all_intervals_at_least_theta':
                                                        v['all_intervals_at_least_theta'],
                                                    'service_times_imply_theta':
                                                        v['service_times_imply_theta']}
                                                for k, v in rec['per_zone'].items()},
        'n_pairs_with_intermediate_event_clocks': n_split_pairs,
        'what_it_proves': 'on THIS instance and baseline the recorded stream satisfies P5 with a '
                          'large margin, AND the advance is genuinely partitioned',
        'what_it_does_not_prove': 'nothing about other instances, policies, vehicle counts or '
                                  'service-time families',
        'label': 'DIAGNOSTIC_ONLY'},
       conclusion='the corrected condition is satisfied with a wide margin on the probed trace, but '
                  'that is an observation, not a proof of reachability')

    # ============================================================ SEG-08 the gap lower bound
    ok8 = True
    viol = []
    for z in ZONES:
        info = rec['per_zone'][str(z.zone)]
        for g in info['intervals']:
            if not g['gap_at_least_later_service_time']:
                ok8 = False
                viol.append(g)
    ck('SEG-08', 'the provable lower bound holds on the trace: every inter-door interval is at least '
       'the service time of the LATER pickup (fact F-6), which is why the 9 min data margin '
       'transfers to the correct object',
       ok8 and rec['summary']['n_intervals'] > 0,
       {'violations': viol,
        'mechanism': 'finish_n = max(arrive_n, tw_start_n) + service_n, depart_n >= clock_{n-1}, '
                     'ready_time = finish_{n-1} => finish_n - clock_{n-1} >= service_n',
        'scope': 'proved for THIS environment code path; a different dispatcher would need its own '
                 'argument'},
       conclusion='the empirical "service_time >= Theta" margin is meaningful, and its justification '
                  'is now attached to the inter-door interval rather than to a door segment')

    # ============================================================ SEG-09 malformed streams
    good = {0: [{'node': 1, 'door_clock': 0.0, 'zone': 0, 'service_time_h': 0.25},
                {'node': 2, 'door_clock': 1.0, 'zone': 0, 'service_time_h': 0.25}]}
    same_clock = {0: [{'node': 1, 'door_clock': 0.5, 'zone': 0, 'service_time_h': 0.25},
                      {'node': 2, 'door_clock': 0.5, 'zone': 0, 'service_time_h': 0.25}]}
    short = {0: [{'node': 1, 'door_clock': 0.0, 'zone': 0, 'service_time_h': 0.25},
                 {'node': 2, 'door_clock': 0.005, 'zone': 0, 'service_time_h': 0.25}]}
    backwards = {0: [{'node': 1, 'door_clock': 1.0, 'zone': 0, 'service_time_h': 0.25},
                     {'node': 2, 'door_clock': 0.5, 'zone': 0, 'service_time_h': 0.25}]}
    nanrow = {0: [{'node': 1, 'door_clock': float('nan'), 'zone': 0, 'service_time_h': 0.25}]}
    codes = {}
    for name, st, kw in (('good', good, {'state_identity': 'S', 'expected_state_identity': 'S'}),
                         ('same_clock', same_clock, {'state_identity': 'S',
                                                     'expected_state_identity': 'S'}),
                         ('short_interval', short, {'state_identity': 'S',
                                                    'expected_state_identity': 'S'}),
                         ('backwards', backwards, {'state_identity': 'S',
                                                   'expected_state_identity': 'S'}),
                         ('non_finite', nanrow, {'state_identity': 'S',
                                                 'expected_state_identity': 'S'}),
                         ('no_state_identity', good, {}),
                         ('state_mismatch', good, {'state_identity': 'A',
                                                   'expected_state_identity': 'B'}),
                         ('bad_contract', good, {'state_identity': 'S',
                                                 'expected_state_identity': 'S',
                                                 'contract_identity': 'x' * 64}),
                         ('not_at_target', good, {'state_identity': 'S',
                                                  'expected_state_identity': 'S',
                                                  'dispatch_state_is_target': False}),
                         ('inactive_zone', good, {'state_identity': 'S',
                                                  'expected_state_identity': 'S',
                                                  'active_zone_mask': (True, False, True)})):
        r = ES.check_event_stream(contract=CONTRACT,
                                  contract_identity=kw.pop('contract_identity',
                                                           CONTRACT.contract_hash),
                                  event_stream=st, **kw)
        codes[name] = (r['verdict'], [x['code'] for x in r['refusals']])
    ck('SEG-09', 'the event-stream checker is fail-closed and has a real failure boundary: same-clock '
       'doors and short intervals are NOT_CERTIFIABLE, backwards clocks / non-finite data / identity '
       'mismatch / wrong domain are REFUSED',
       codes['good'][0] == ES.PROVED_CONDITIONAL
       and codes['no_state_identity'][0] == ES.DIAGNOSTIC_ONLY
       and codes['same_clock'][0] == ES.NOT_CERTIFIABLE
       and codes['short_interval'][0] == ES.NOT_CERTIFIABLE
       and codes['backwards'][0] == ES.REFUSED
       and codes['non_finite'][0] == ES.REFUSED
       and codes['state_mismatch'][0] == ES.REFUSED
       and codes['bad_contract'][0] == ES.REFUSED
       and codes['not_at_target'][0] == ES.REFUSED
       and codes['inactive_zone'][0] == ES.REFUSED,
       {'verdicts': codes},
       conclusion='the four-level ladder is exercised in both directions')

    # ============================================================ SEG-10 the door adds no cooling
    z1 = ZONES[1]
    # NB: dispatch_vehicle sets T = tau, so the state must be built by the executor itself rather
    # than hand-written (the first version of this control set T by hand and then dispatched over it,
    # which silently reset the demonstration).
    st = cs.dispatch_vehicle(cs.VehicleColdChainState(
        compartment_temperature_c=(float(TH.ambient_temperature_c),) * N_ZONES,
        zone_load=(0.0,) * N_ZONES), CONTRACT)
    after_first = cs.transition_segment(
        st, depart_time=0.0, arrival_time=0.0, service_finish=0.0, served_customer=1,
        active_zone_mask=(True,) * N_ZONES, contract=CONTRACT, order_quantity=1.0,
        order_temp_class=1, initial_quality=9.0)[0]
    after_second = cs.transition_segment(
        after_first, depart_time=0.0, arrival_time=0.0, service_finish=0.0, served_customer=2,
        active_zone_mask=(True,) * N_ZONES, contract=CONTRACT, order_quantity=1.0,
        order_temp_class=1, initial_quality=9.0)[0]
    one = float(after_first.compartment_temperature_c[1])
    two = float(after_second.compartment_temperature_c[1])
    ck('SEG-10', 'two zero-duration doors with a ZERO interval leave the zone at tau+2delta: the door '
       'segment contributes no cooling at all, so only the door-less advance between doors can bring '
       'the zone back',
       abs(one - (z1.target_c + float(z1.delta_real))) < 1e-12
       and abs(two - (z1.target_c + 2 * float(z1.delta_real))) < 1e-12
       and two > z1.target_c + float(z1.delta_real),
       {'after_first_door_c': one, 'after_second_door_c': two,
        'tau_plus_delta': z1.target_c + float(z1.delta_real),
        'tau_plus_2delta': z1.target_c + 2 * float(z1.delta_real),
        'reading': 'this is the same-clock door pair that SEG-09 marks NOT_CERTIFIABLE, and it is why '
                   'the interval between doors -- not any segment duration -- is the quantity that '
                   'matters'},
       conclusion='the door and the cooling live in different segments; conflating them is the defect '
                  'this round fixes')

    # ============================================================ SEG-11 traces vs replay
    worst_gap = 0.0
    for tr in traces:
        if not tr.services:
            continue
        st_r = cs.dispatch_vehicle(cs.VehicleColdChainState(
            compartment_temperature_c=(float(TH.ambient_temperature_c),) * N_ZONES,
            zone_load=(0.0,) * N_ZONES), CONTRACT)
        for rec_v in tr.services:
            n = int(rec_v.node)
            st_r, _m = cs.transition_segment(
                st_r, depart_time=float(rec_v.depart_time), arrival_time=float(rec_v.arrival_time),
                service_finish=float(rec_v.service_finish), served_customer=n,
                active_zone_mask=(True,) * N_ZONES, contract=CONTRACT,
                order_quantity=float(env.demands[0, n]),
                order_temp_class=int(env.temp_class[0, n]),
                initial_quality=float(env.initial_quality[0, n]),
                segment_distance_units=float(env.dist_mat[0, rec_v.prev_node, n]))
            worst_gap = max(worst_gap, max(
                abs(float(st_r.compartment_temperature_c[z]) -
                    float(rec_v.coldchain_after.compartment_temperature_c[z])) for z in range(3)))
    ck('SEG-11', 'on the probed trace the unsplit M1-b replay happens to agree with the environment '
       'bit-for-bit at every service completion, because every interval is long enough that the floor '
       'absorbs the difference -- so agreement here is NOT evidence that the segmentations coincide',
       worst_gap == 0.0,
       {'max_abs_diff_c': worst_gap,
        'why_it_matches': 'once the composition reaches tau the floor pins it there, so any further '
                          'partition is invisible; SEG-05 shows where it stops being invisible',
        'reading': 'a matching outcome is not a matching semantics',
        'label': 'DIAGNOSTIC_ONLY'},
       conclusion='the replay is a different trajectory in principle; it matched in this one trace by '
                  'an absorbing-floor accident, which must not be generalised')

    # ============================================================ SEG-12 split happens
    ck('SEG-12', 'the door-less advance IS partitioned in the real environment: intermediate event '
       'clocks exist between consecutive doors, so the partition cannot be assumed to be one piece',
       n_split_pairs > 0,
       {'n_pairs_with_intermediate_event_clocks': n_split_pairs,
        'n_event_clocks': len(event_clocks),
        'consequence': 'a certificate has to be partition-independent (L-G) or it is not a '
                       'certificate about this environment'},
       conclusion='this is the structural reason S3 (bitwise replay) needs the event clock sequence, '
                  'not just the plan')

    payload = {
        'schema': 'q3_event_segmentation_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'thresholds': {str(k): v for k, v in th.items()},
        'env_segmentation_facts': [{'id': a, 'where': b, 'fact': c}
                                   for a, b, c in ES.ENV_SEGMENTATION_FACTS],
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('EVENT_SEGMENTATION_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'EVENT_SEGMENTATION_CONTROLS_FAIL'),
        'labels': {'PROVED': 'L-F, L-G, P5, F-6 mechanism (SEG-02/03/04/08/10)',
                   'FACT': 'source line evidence (SEG-01)',
                   'DIAGNOSTIC': 'one greedy instance trace (SEG-07/11/12)',
                   'UNKNOWN': 'reachability over other instances/policies/service-time families'},
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
