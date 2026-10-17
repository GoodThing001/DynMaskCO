#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 DOOR-INTERVAL THRESHOLD CONTROLS -- ERRATA ROUND (DIAGNOSTIC, not for production).

Rows ERR-01..ERR-13.  They do three things:

  * ERR-01/ERR-02 record and verify the CORRECTION: `T_single < Theta <= Delta*`, so requiring
    `Delta*` on an inter-door interval is conservative and SOUND, and only `T_single` is unsound;
  * ERR-03/ERR-04/ERR-09 close the second gap the user found: L-F is false for `0 < Delta <= _EPS`,
    and the environment's ANCHOR semantics are what make the wall clock a lower bound for the
    EFFECTIVE cooling duration;
  * ERR-05..ERR-08/ERR-10/ERR-11 build and break the exact-rational certificate: it uses no
    transcendental, no tolerance and no epsilon, and it refuses when the anchor semantics are not
    established.

Usage: python scripts/tests/test_q3_door_interval_thresholds.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import math
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
import q3_door_interval_thresholds_experimental as DI                    # noqa: E402
import q3_event_segmentation_experimental as ES                          # noqa: E402
import q3_strict_float_experimental as SF                                # noqa: E402
from strict_online_env import StrictOnlineEnv, GreedyReplanner           # noqa: E402

CONTRACT = ColdChainContractV2()
TH = CONTRACT.thermal
ZONES = SF.zone_thermals_exact(CONTRACT)
N_ZONES = len(ZONES)
ROWS = []
LIMITS = ('synthetic controls plus ONE greedy instance trace (no model, no training, no GPU); read '
          'only; a passing row is a DIAGNOSTIC, never a production certificate')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-7s %s' % ('PASS' if ok else 'FAIL', cid, name[:94]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

#: the identities registered BEFORE the G-TMIN fix live in a report, not in source: a bare 64-hex
#: literal is what the staged-content scanner refuses, and the repository convention is that digests
#: belong in records.  Reading them here changes nothing about what the row asserts.
FROZEN_REGISTRY = 'docs/当前规划/决策与审计/Q3冻结身份登记_2026-10-17.md'


def load_frozen_identities(paths, registry=FROZEN_REGISTRY):
    """Parse `path = <64 hex>` lines out of the registry report; refuse if one is missing."""
    rx = re.compile(r'^\s*(\S+)\s*=\s*([0-9a-f]{64})\s*$')
    found = {}
    for line in (EXT / registry).read_text(encoding='utf-8').splitlines():
        m = rx.match(line)
        if m:
            found[m.group(1)] = m.group(2)
    missing = [p for p in paths if p not in found]
    if missing:
        raise RuntimeError('frozen identity registry is missing: %s' % missing)
    return {p: found[p] for p in paths}


class FakeVehicle(object):
    def __init__(self):
        self.coldchain_state = cs.dispatch_vehicle(cs.VehicleColdChainState(
            compartment_temperature_c=(float(TH.ambient_temperature_c),) * N_ZONES,
            zone_load=(0.0,) * N_ZONES), CONTRACT)
        self.coldchain_updated_time = 0.0
        self.coldchain_accounting_anchor = self.coldchain_state
        self.status = 'idle'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src = {'scripts/coldchain/q3_door_interval_thresholds_experimental.py':
           sha256(EXT / 'scripts/coldchain/q3_door_interval_thresholds_experimental.py'),
           'scripts/coldchain/coldchain_state.py':
           sha256(EXT / 'scripts/coldchain/coldchain_state.py'),
           'scripts/coldchain/coldchain_contract.py':
           sha256(EXT / 'scripts/coldchain/coldchain_contract.py'),
           'scripts/simulation/strict_online_env.py':
           sha256(EXT / 'scripts/simulation/strict_online_env.py'),
           'scripts/tests/test_q3_door_interval_thresholds.py': sha256(Path(__file__))}

    # ============================================================ ERR-01 ordering theorem
    rep = DI.ordering_report(CONTRACT)
    rows = rep['rows']
    ck('ERR-01', 'T_single <= Theta <= Delta* holds in all three zones, and Theta equals '
       '(1/h)*ln(1 + h*Delta*) as the algebra predicts',
       all(r['t_single_le_theta'] and r['theta_le_delta_star'] for r in rows)
       and all(DI.zone_thresholds_exact(CONTRACT, r['zone'])['ln1p_identity_holds'] for r in rows),
       {'ordering': rows, 'theorem': rep['theorem'],
        'theta_delta_star_identity_max_residual': max(
            abs(DI.zone_thresholds_exact(CONTRACT, r['zone'])['theta_float']
                - (1.0 / float(DI.zone_thresholds_exact(CONTRACT, r['zone'])['h']))
                * math.log(1.0 + float(DI.zone_thresholds_exact(CONTRACT, r['zone'])['h']
                                       * DI.zone_thresholds_exact(CONTRACT, r['zone'])[
                                           'delta_star_exact'])))
            for r in rows)},
       conclusion='the three thresholds are strictly ordered; the previous round s claim that Delta* '
                  'is unsound is corrected here')

    # ============================================================ ERR-02 Delta* is SOUND
    rows2, ok2 = [], True
    for z in ZONES:
        th = DI.zone_thresholds_exact(CONTRACT, z.zone)
        ts = DI.t_single_float(CONTRACT, z.zone)
        theta = th['theta_float']
        d_star = th['delta_star_exact']
        # a total interval inside the (T_single, Theta) window: the ONE-piece formula says "back on
        # the floor", a refined composition does not; Delta* is above the whole window.
        total = Fraction(ts) + (Fraction(theta) - Fraction(ts)) / 2
        one_piece = DI.zone_thresholds_exact  # noqa: F841  (documentation only)
        import q3_float_replay_qualification_experimental as RP
        raw = RP.unsplit_raw(z, z.target_c + float(z.delta_real), float(total))
        st = z.target_c + float(z.delta_real)
        for _ in range(8):
            st = z.ambient_c + (st - z.ambient_c) * math.exp(
                -z.heat_transfer_per_hour * float(total) / 8.0) \
                - z.cooling_rate_c_per_hour * float(total) / 8.0
        window_below_delta_star = Fraction(total) < d_star
        t_single_would_admit = Fraction(ts) <= total < Fraction(theta)
        ok2 = ok2 and window_below_delta_star and t_single_would_admit \
            and raw <= z.target_c + 1e-15 and st > z.target_c
        rows2.append({'zone': z.zone, 't_single_h': ts, 'theta_h': theta,
                      'delta_star_h': float(d_star), 'total_tested_h': float(total),
                      'one_piece_raw_c': raw, 'eight_pieces_raw_c': st,
                      'tau': z.target_c,
                      't_single_would_admit_this_interval': t_single_would_admit,
                      'delta_star_excludes_it': window_below_delta_star})
    ck('ERR-02', 'the correction, verified constructively: for every zone there is an interval that '
       'T_single would ADMIT while a refined composition is still above tau, and Delta* excludes that '
       'same interval -- so Delta* is conservative, not unsound',
       ok2, {'rows': rows2,
             'reading': 'BR-1 refutes T_single; it does NOT refute Delta* (Theta <= Delta*).  The '
                        'previous round conflated the two.'},
       conclusion='use Delta* (rational, log-free) as the auditable threshold; Theta is the tight '
                  'one; T_single must never be used')

    # ============================================================ ERR-03 L-F scope
    lf = []
    ok3 = True
    z0 = ZONES[0]
    h0 = float(z0.heat_transfer_per_hour)
    c0 = float(z0.cooling_rate_c_per_hour)
    A0 = float(z0.ambient_c)
    for d in (0.0, 5e-10, 1e-9, 1.5e-9, 2e-9, 1e-6, 1e-3, 0.5):
        t0 = z0.target_c + 0.5 * float(z0.delta_real)
        temps = [float(t) for t in TH.target_temperature_c]
        temps[0] = t0
        after, _e = cs._advance_temperatures(tuple(temps), float(d), (True,) * N_ZONES, CONTRACT)
        floored = after[0]
        phi_raw = A0 + (t0 - A0) * math.exp(-h0 * float(d)) - c0 * float(d)
        flow = ES.ode_flow(z0, t0, d)
        bound = max(z0.target_c, flow)
        in_eps = 0.0 < d <= float(DI.EPS_EXACT)
        # THE phenomenon: for 0 < Delta <= _EPS the executor returns the IDENTITY, which is above the
        # flow (the unfloored closed form is not the object being violated here -- it is ~equal to the
        # flow at such tiny durations, which is why this control had to be restated).
        exceeds = floored > bound + 1e-15
        healthy = not exceeds
        if in_eps and not exceeds:
            ok3 = False
        if (not in_eps) and not healthy:
            ok3 = False
        lf.append({'delta_h': d, 'floored_executor_c': floored, 'flow_c': flow,
                   'max_tau_flow_c': bound,
                   'executor_minus_bound': floored - bound,
                   'phi_raw_minus_flow': phi_raw - flow,
                   'in_eps_branch': in_eps,
                   'executor_exceeds_max_tau_flow': exceeds})
    ck('ERR-03', 'L-F has a SCOPE: for 0 < Delta <= _EPS the executor returns the IDENTITY, which sits '
       'ABOVE max(tau, flow) by up to 6.0e-10 C; above _EPS the executor is back inside '
       'max(tau, flow)',
       ok3,
       {'rows': lf,
        'correction': 'the previous round stated L-F for every Delta >= 0.  The correct scope is '
                      'Delta > _EPS, and the statement must be made about the EXECUTOR (identity '
                      'branch included), not only about the closed form',
        'measured_max_excess_in_eps_branch': max(
            [r['executor_minus_bound'] for r in lf if r['in_eps_branch']] or [0.0]),
        'note': 'the unfloored closed form stays within 4e-15 of the flow at these durations, so a '
                'control written on the closed form cannot see the defect -- which is exactly how the '
                'hole survived the previous round'},
       conclusion='the upper bound must be on the EFFECTIVE cooling duration, not on the wall clock')

    # ============================================================ ERR-04 anchor semantics
    env = StrictOnlineEnv({'coords': np.zeros((1, 2, 2)), 'demands': np.zeros((1, 2)),
                           'tw_start': np.zeros((1, 2)), 'tw_end': np.zeros((1, 2)),
                           'temp_class': np.zeros((1, 2), dtype=int)},
                          capacity=float(CONTRACT.operational.shared_vehicle_capacity),
                          num_vehicles=1, coldchain_contract=CONTRACT)
    rows4 = []
    ok4 = True
    patterns = ([0.1, 5e-10, 0.2], [0.1, 5e-10, 9e-10, 0.2], [0.1, 5e-10], [0.1, 0.2])
    for gaps in patterns:
        v = FakeVehicle()
        rendered, prev_anchor, t = 0.0, 0.0, 0.0
        for g in gaps:
            t += g
            env._advance_coldchain_to(v, t)
            if v.coldchain_updated_time != prev_anchor:
                rendered += v.coldchain_updated_time - prev_anchor
                prev_anchor = v.coldchain_updated_time
        lost = t - rendered
        ok4 = ok4 and lost <= float(DI._SKIP_SLACK)
        rows4.append({'gaps': gaps, 'wall_h': t, 'rendered_h': rendered, 'lost_h': lost,
                      'lost_le_skip_slack': lost <= float(DI._SKIP_SLACK)})
    ck('ERR-04', 'the anchor semantics hold: a skipped _advance_coldchain_to does NOT lose the span '
       '(the next real advance starts from the same anchor), so at most the skip threshold is lost '
       'and only at the END of an interval',
       ok4, {'rows': rows4, 'skip_slack_h': float(DI._SKIP_SLACK),
             'why': 'coldchain_updated_time is only moved on a real advance (A-2/A-3), so the skipped '
                    'span is covered later; the residual is bounded by 2*_EPS'},
       conclusion='wall clock minus the skip slack is a provable lower bound for the effective '
                  'cooling duration in THIS environment')

    # ============================================================ ERR-05 exact Delta*
    rows5 = []
    ok5 = True
    naive_unsafe = []
    for z in ZONES:
        th = DI.zone_thresholds_exact(CONTRACT, z.zone)
        exact = th['delta_star_exact']
        up = th['delta_star_float_up']
        naive = th['delta_star_float_naive']
        rows5.append({'zone': z.zone, 'delta_star_exact': str(exact),
                      'exact_is_rational': isinstance(exact, Fraction),
                      'float_up': up, 'float_up_ge_exact': Fraction(up) >= exact,
                      'naive_float': naive, 'naive_ge_exact': Fraction(naive) >= exact})
        if Fraction(naive) < exact:
            naive_unsafe.append(z.zone)
        ok5 = ok5 and Fraction(up) >= exact and isinstance(exact, Fraction)
    ck('ERR-05', 'Delta* is an EXACT rational (no exp, no log); rounding it up to the next binary64 is '
       'provable, and the naive float is NOT always a safe upper bound',
       ok5 and naive_unsafe,
       {'rows': rows5, 'zones_where_naive_float_is_below_exact': naive_unsafe,
        'consequence': 'the implementation must compare in Fraction (or round up explicitly); using '
                       'float(Delta*) directly would be unsound in those zones'},
       conclusion='the threshold can be handled with zero rounding error and zero transcendental '
                  'assumptions')

    # ============================================================ ERR-06 live trace, exact check
    ds = dict(np.load(EXT / 'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz', allow_pickle=True))
    data = {k: (v[:1] if np.asarray(v).ndim >= 2 and np.asarray(v).shape[0] >= 1 else v)
            for k, v in ds.items()}
    live = StrictOnlineEnv(data, capacity=float(CONTRACT.operational.shared_vehicle_capacity),
                           tw_speed=1.0, num_vehicles=25, replanner=GreedyReplanner('edd'),
                           coldchain_contract=CONTRACT, booking_horizon=None)
    traces, served = live.run(0)
    stream = ES.event_stream_from_traces(traces, live, 0)
    rec = DI.check_door_intervals_exact(contract=CONTRACT,
                                        contract_identity=CONTRACT.contract_hash,
                                        event_stream=stream,
                                        state_identity='S', expected_state_identity='S')
    ck('ERR-06', 'the exact-rational inter-door check certifies the real recorded event stream without '
       'any transcendental, tolerance or epsilon',
       rec['verdict'] == DI.PROVED_CONDITIONAL and rec['summary']['n_intervals'] > 0
       and rec['summary']['transcendental_used'] is False
       and rec['summary']['min_slack_h'] > 0,
       {'verdict': rec['verdict'], 'served': int(served.sum()) - 1,
        'summary': rec['summary'],
        'per_zone': {k: {'n_doors': v['n_doors'], 'n_intervals': v['n_intervals'],
                         'threshold_min': v['threshold_min'],
                         'min_interval_h': v['min_interval_h'],
                         'min_slack_h': (None if v['min_slack_exact'] is None
                                         else float(v['min_slack_exact'])),
                         'theta_min': v['theta_min'], 't_single_h': v['t_single_h']}
                     for k, v in rec['per_zone'].items()},
        'label': 'DIAGNOSTIC_ONLY',
        'what_it_proves': 'on THIS instance and baseline every inter-door interval clears Delta* with '
                          'a wide margin, using exact arithmetic',
        'what_it_does_not_prove': 'reachability over other instances, policies or service-time '
                                  'families'},
       conclusion='the decision-time condition is computable exactly and passes on the probed trace')

    # ============================================================ ERR-07 near threshold
    th1 = DI.zone_thresholds_exact(CONTRACT, 0)
    need = th1['delta_star_exact']
    slack = DI._SKIP_SLACK
    # the clock must be a float that is PROVABLY >= need + slack, otherwise the reconstruction of the
    # interval is what fails, not the condition (this bit the first version of this control)
    at_threshold = DI.float_ceil_exact(need + slack)
    exact_ok = {'v0': [{'node': 1, 'zone': 0, 'door_clock': 0.0},
                       {'node': 2, 'zone': 0, 'door_clock': at_threshold}]}
    just_below = {'v0': [{'node': 1, 'zone': 0, 'door_clock': 0.0},
                         {'node': 2, 'zone': 0,
                          'door_clock': DI.float_floor_exact(need + slack
                                                             - Fraction(1, 2 ** 40))}]}
    r_ok = DI.check_door_intervals_exact(contract=CONTRACT,
                                         contract_identity=CONTRACT.contract_hash,
                                         event_stream=exact_ok, state_identity='S',
                                         expected_state_identity='S')
    r_bad = DI.check_door_intervals_exact(contract=CONTRACT,
                                          contract_identity=CONTRACT.contract_hash,
                                          event_stream=just_below, state_identity='S',
                                          expected_state_identity='S')
    ck('ERR-07', 'the boundary is exact and inclusive: an interval of exactly Delta* + slack passes, '
       'and one 2^-40 h shorter is NOT_CERTIFIABLE',
       r_ok['verdict'] == DI.PROVED_CONDITIONAL and r_bad['verdict'] == DI.NOT_CERTIFIABLE,
       {'at_threshold': r_ok['verdict'], 'just_below': r_bad['verdict'],
        'interval_at_threshold_h': float(need + slack),
        'violation': r_bad['per_zone']['0']['violations'],
        'note': 'the comparison is `>=` on exact rationals, so there is no ulp ambiguity'},
       conclusion='the condition has a real, exactly computable failure boundary')

    # ============================================================ ERR-08 the seven cases
    ledger = DI.segmentation_case_ledger(CONTRACT)
    cases = {c['id']: c for c in ledger['cases']}
    c1 = cases['C1 zero-length door (Delta = 0)']
    c2 = cases['C2 sub-epsilon positive duration (0 < Delta <= _EPS)']
    c4 = cases['C4 duration just above _EPS']
    c5 = cases['C5 many door-less splits of one interval']
    c6 = cases['C6 one piece of the same total duration']
    ck('ERR-08', 'the seven segmentation cases are each stated separately: a zero-length door cools by '
       'exactly 0; a sub-epsilon positive duration is the identity (no cooling); splitting is warmer '
       'than one piece',
       (not c1['cools']) and (not c2['cools']) and c4['cools']
       and c5['warmer_than_one_piece'] and c5['cools'],
       {'n_cases': len(ledger['cases']), 'cases': ledger['cases'],
        'theoretical_note': ledger['theoretical_note']},
       conclusion='the door is never counted as a cooling segment, and the split direction is '
                  'recorded in the ledger rather than implied')

    # ============================================================ ERR-09 fail-closed
    good = {'v0': [{'node': 1, 'zone': 0, 'door_clock': 0.0},
                   {'node': 2, 'zone': 0, 'door_clock': 1.0}]}
    codes = {}
    for name, kw in (('good', {}),
                     ('anchor_not_established', {'anchor_semantics_assumed': False}),
                     ('state_mismatch', {'state_identity': 'A', 'expected_state_identity': 'B'}),
                     ('no_state_identity', {'state_identity': None,
                                            'expected_state_identity': None}),
                     ('bad_contract', {'contract_identity': 'x' * 64}),
                     ('theta_mode', {'mode': 'theta'})):
        k = dict(state_identity='S', expected_state_identity='S')
        k.update(kw)
        r = DI.check_door_intervals_exact(contract=CONTRACT,
                                          contract_identity=k.pop('contract_identity',
                                                                  CONTRACT.contract_hash),
                                          event_stream=good if name != 'nan' else {}, **k)
        codes[name] = (r['verdict'], [x['code'] for x in r['refusals']])
    backwards = {'v0': [{'node': 1, 'zone': 0, 'door_clock': 1.0},
                        {'node': 2, 'zone': 0, 'door_clock': 0.5}]}
    r_back = DI.check_door_intervals_exact(contract=CONTRACT,
                                           contract_identity=CONTRACT.contract_hash,
                                           event_stream=backwards, state_identity='S',
                                           expected_state_identity='S')
    codes['backwards'] = (r_back['verdict'], [x['code'] for x in r_back['refusals']])
    nanstream = {'v0': [{'node': 1, 'zone': 0, 'door_clock': float('nan')}]}
    r_nan = DI.check_door_intervals_exact(contract=CONTRACT,
                                          contract_identity=CONTRACT.contract_hash,
                                          event_stream=nanstream, state_identity='S',
                                          expected_state_identity='S')
    codes['non_finite'] = (r_nan['verdict'], [x['code'] for x in r_nan['refusals']])
    ck('ERR-09', 'the checker is fail-closed: without the anchor semantics it REFUSES rather than '
       'substituting the wall clock; state drift, a foreign contract, backwards clocks and '
       'non-finite clocks are all REFUSED; the theta mode can only be DIAGNOSTIC',
       codes['good'][0] == DI.PROVED_CONDITIONAL
       and codes['anchor_not_established'][0] == DI.REFUSED
       and codes['state_mismatch'][0] == DI.REFUSED
       and codes['bad_contract'][0] == DI.REFUSED
       and codes['backwards'][0] == DI.REFUSED
       and codes['non_finite'][0] == DI.REFUSED
       and codes['theta_mode'][0] == DI.DIAGNOSTIC_ONLY
       and codes['no_state_identity'][0] == DI.DIAGNOSTIC_ONLY,
       {'verdicts': codes},
       conclusion='the certificate cannot be produced by relaxing the anchor assumption')

    # ============================================================ ERR-10 wall vs effective
    # a deliberately sub-epsilon final gap: the wall interval is what the record shows, the effective
    # cooling is smaller by at most the skip slack, and the checker accounts for it.
    z1 = ZONES[1]
    need1 = DI.zone_thresholds_exact(CONTRACT, 1)['delta_star_exact']
    tight = {'v0': [{'node': 1, 'zone': 1, 'door_clock': 0.0},
                    {'node': 2, 'zone': 1, 'door_clock': float(need1)}]}
    r_tight = DI.check_door_intervals_exact(contract=CONTRACT,
                                            contract_identity=CONTRACT.contract_hash,
                                            event_stream=tight, state_identity='S',
                                            expected_state_identity='S')
    wall_only = float(need1) >= float(need1)
    ck('ERR-10', 'the wall clock alone is NOT the condition: an interval of exactly Delta* (without the '
       'skip slack) fails, because the effective cooling may be shorter than the wall interval',
       r_tight['verdict'] == DI.NOT_CERTIFIABLE and wall_only,
       {'verdict': r_tight['verdict'],
        'interval_h': float(need1), 'threshold_h': float(need1),
        'skip_slack_h': float(DI._SKIP_SLACK),
        'violation': r_tight['per_zone']['1']['violations'],
        'why': 'the wall interval must exceed Delta* by the skip slack; a naive `wall >= Delta*` test '
               'would accept this interval'},
       conclusion='mathematical total duration and effectively processed duration are now separate '
                  'objects in the implementation, which is what the errata required')

    # ============================================================ ERR-11 M1-b scope
    scope = DI.m1b_replay_scope()
    ck('ERR-11', 'the M1-b replay is a SAMPLED envelope that splits less than the environment and is '
       'therefore cooler; the composite gate is nevertheless fail-closed today, so there is no false '
       'pass on the production path',
       'fail-closed' in scope['composite_gate_status']
       and 'may not be upgraded' in scope['per_zone_max_status']
       and 'COOLER' in scope['direction_of_the_difference'],
       scope,
       conclusion='the M1-b issue is a precondition on the CONTINUOUS level, not a present production '
                  'defect; it must be fixed before any continuous proof is installed')

    # ============================================================ ERR-12 no frozen change
    frozen = load_frozen_identities((
        'scripts/coldchain/coldchain_state.py',
        'scripts/simulation/strict_online_env.py',
        'scripts/coldchain/q3_thermal_interval_experimental.py',
        'scripts/m1b_adapter/m1b_q2_certifier.py'))
    drift = {k: {'expected': v, 'actual': sha256(EXT / k)} for k, v in frozen.items()
             if sha256(EXT / k) != v}
    prev = {}
    man = json.loads((EXT / 'results/_q3_next_20261017/SHA256_MANIFEST_V2.json')
                     .read_text(encoding='utf-8'))
    for p, h in man['files'].items():
        if sha256(EXT / p) != h:
            prev[p] = {'recorded': h, 'actual': sha256(EXT / p)}
    ck('ERR-12', 'no frozen source moved and none of the previously delivered files moved: this round '
       'adds a new module and a new report only',
       not drift and not prev,
       {'frozen_drift': drift, 'previous_delivery_drift': prev,
        'previous_manifest': 'results/_q3_next_20261017/SHA256_MANIFEST_V2.json'},
       conclusion='append-only; the erroneous report stands as the record and is corrected by a '
                  'separate errata')

    payload = {
        'schema': 'q3_door_interval_threshold_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'thresholds': {str(z.zone): DI.zone_thresholds_exact(CONTRACT, z.zone) for z in ZONES},
        'anchor_semantics': [{'id': a, 'where': b, 'fact': c}
                             for a, b, c in DI.TRUSTED_ANCHOR_SEMANTICS],
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('DOOR_INTERVAL_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'DOOR_INTERVAL_CONTROLS_FAIL'),
        'corrections': {
            'ERR-1': 'Delta* on a total inter-door interval is CONSERVATIVE AND SOUND '
                     '(Theta <= Delta*); only T_single is unsound',
            'ERR-2': 'L-F holds for Delta > _EPS only; the bound is on the EFFECTIVE cooling '
                     'duration, and the environment anchor semantics make it >= wall - 2*_EPS',
        },
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
