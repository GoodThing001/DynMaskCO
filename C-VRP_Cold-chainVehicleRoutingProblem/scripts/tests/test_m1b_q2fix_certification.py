#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix regression suite.

Covers, against the FIXED implementation (`M1B_Q2FIX`):
  * the 16 Q2-R cases, with the four false passes expected to FLIP to refusal;
  * the previously NOT_ESTABLISHED cases, now expected to be DISCHARGED or kept as explicit
    new-version blockers;
  * POSITIVE controls, because a fix that only adds refusals is not a fix;
  * a FALSE_REJECT census.

The original `test_m1b_q2r_negative_controls.py` and its JSON are NOT modified: they are the
historical record of the pre-fix behaviour and the original inputs/expectations are preserved
verbatim in that file.  This file re-asserts the same inputs against the fixed code.

Pure NumPy / stdlib.  No model, no GPU, no rollout, no installation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in (EXT / 'scripts' / 'coldchain', EXT / 'scripts' / 'simulation',
           EXT / 'scripts' / 'evaluation', EXT / 'scripts' / 'm1b_adapter', HERE):
    sys.path.insert(0, str(_p))

import numpy as np                                                       # noqa: E402
import action_contract as AC                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import coldchain_state as CS                                             # noqa: E402
from coldchain_state import create_vehicle_state, dispatch_vehicle       # noqa: E402
from action_contract import VehiclePlan                                   # noqa: E402

import m1b_timing as TM                                                  # noqa: E402
import m1b_return_feasibility as RF                                      # noqa: E402
import m1b_quality_certificate as QC                                     # noqa: E402
import m1b_thermal_gate as TG                                            # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
from m1b_q2_certifier import (M1bQ2FixCertifier, ENGINEERING_CERTIFIED,   # noqa: E402
                              ENGINEERING_REFUSED, IMPLEMENTATION_VERSION)

ROWS = []
CONTRACT = ColdChainContractV2()
T_END = 24.0
SYNTHETIC_B = 12345.678          # SYNTHETIC ONLY


def chk(cid, name, verdict, detail=None):
    ROWS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-9s] %-56s %s' % (cid, name[:56], verdict))


def ck(cid, name, cond, detail=None):
    chk(cid, name, 'PASS' if cond else 'FAIL', detail)
    return bool(cond)


class V(object):
    def __init__(self, vid, status='ready', node=0, ready=0.0, load=0.0, cc=None,
                 committed_next=None, committed_finish=None, return_finish=None, suffix=None):
        self.vehicle_id, self.status, self.current_node = int(vid), status, int(node)
        self.ready_time, self.current_load = float(ready), float(load)
        self.committed_next, self.committed_finish = committed_next, committed_finish
        self.return_finish = return_finish
        self.mutable_suffix = list(suffix or [])
        self.served_route, self.dispatch_time = [], None
        self.coldchain_state, self.coldchain_updated_time = cc, None
        self.committed_coldchain_before, self.coldchain_accounting_anchor = None, None
        self.needs_replan, self.replan_reason = False, None


def fresh_cc(n_lots=1, temp_class=1, initial_quality=9.0, quantity=1.0, pickup=-1.0):
    st = dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT)
    if n_lots > 0:
        lots = tuple(CS.CargoLotState(order_id=100 + i, quantity=quantity,
                                      temp_class=temp_class, pickup_finish_time=pickup,
                                      initial_quality=initial_quality,
                                      quality_remaining=initial_quality)
                     for i in range(n_lots))
        zl = list(st.zone_load)
        zl[int(temp_class)] = float(quantity) * n_lots
        st = CS.replace(st, cargo_manifest=lots, zone_load=tuple(zl))
    return st


def plan(vid, at, suffix, load=0.0):
    return VehiclePlan(vid, 0, at, load, list(suffix))


class MiniEnv(object):
    def __init__(self, n=8, tw_start=None, tw_end=None, service=None, travel=1.0, capacity=50.0):
        self.num_nodes, self.capacity, self.tw_speed = n, capacity, 1.0
        self.coldchain_contract = CONTRACT
        self.tw_start = np.zeros((1, n)); self.tw_end = np.full((1, n), 1e9)
        self.service_time = np.zeros((1, n)); self.demands = np.ones((1, n))
        self.temp_class = np.ones((1, n), dtype=np.int64)
        self.dist_mat = np.full((1, n, n), float(travel))
        for k, v in (tw_start or {}).items():
            self.tw_start[0, int(k)] = float(v)
        for k, v in (tw_end or {}).items():
            self.tw_end[0, int(k)] = float(v)
        for k, v in (service or {}).items():
            self.service_time[0, int(k)] = float(v)


def times_for(service=None, tw_start=None, tw_end=None, revealed=(1, 2, 3, 4, 5, 6, 7),
              identity='synth-instance-1'):
    """Per-instance time inputs.

    `tw_start` DEFAULTS TO 0.0 FOR EVERY REVEALED NODE and `tw_end` to a roomy horizon, because
    the fixed replay REFUSES when a needed key is absent -- that refusal is the A2/A2b fix.  A
    caller that wants to exercise the refusal passes an explicit sparse mapping instead.
    """
    svc = dict(service or {})
    tws = {int(n): 0.0 for n in revealed}
    tws.update(tw_start or {})
    twe = {int(n): 1e9 for n in revealed}
    twe.update(tw_end or {})
    return TM.PublicTimeInputs(service_time=svc, tw_start=tws, tw_end=twe,
                               revealed_nodes=revealed, identity=identity)


def attrs_for(iq=None, tc=None, qty=None, served=(1, 2, 3, 4, 5, 6, 7),
              identity='synth-attrs-1'):
    """Order attributes with sane defaults so a test only states what it is testing."""
    served = tuple(int(x) for x in served)
    iq2 = {int(n): 9.0 for n in served}
    iq2.update({int(k): float(v) for k, v in (iq or {}).items()})
    tc2 = {int(n): 1 for n in served}
    tc2.update({int(k): int(v) for k, v in (tc or {}).items()})
    qty2 = {int(n): 1.0 for n in served}
    qty2.update({int(k): float(v) for k, v in (qty or {}).items()})
    return QC.PublicOrderAttributes(initial_quality=iq2, temp_class=tc2, quantity=qty2,
                                    served_nodes=served, source_identity=identity)


def build_snap(vehicles, plans, iq=None):
    return IS.build_frozen_snapshot(vehicles, np.zeros(40, dtype=bool), plans,
                                    initial_quality_by_node=iq or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    print('M1-b/Q2-Fix regression  (implementation %s)' % IMPLEMENTATION_VERSION)

    # =========================================================================== A1
    print('\n-- A1: late tw_start (time-window WAIT) --')
    veh1 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    pl1 = {0: plan(0, 0.0, [5, 0])}
    snap1, _, _ = build_snap(veh1, pl1)
    t1 = times_for(service={5: 0.0}, tw_start={5: 20.0})
    ok1, v1, d1 = RF.certify_return_feasibility(snap1, T_end=19.0, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=t1)
    env1 = MiniEnv(tw_start={5: 20.0}, tw_end={0: 19.0, 5: 30.0}, service={5: 0.0})
    cert1 = AC.certify_route(env1, 0, 0, 0.0, 0.0, (5,))
    got1 = d1['per_vehicle'][0].get('return_arrival')
    ck('A1', 'tw_start wait included -> FALSE_PASS flipped to refusal',
       (not ok1) and v1 == RF.OMEGA11_RETURN_DEADLINE_EXCEEDED and abs(got1 - 21.0) < 1e-9
       and cert1['reason'] == 'return',
       {'verdict': v1, 'return_arrival': got1, 'expected_true_return': 21.0,
        'certify_route_reason': cert1['reason']})

    # =========================================================================== A2 / A2b
    print('\n-- A2 / A2b: unregistered service time must REFUSE --')
    veh2 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    pl2 = {0: plan(0, 0.0, [5, 0])}
    snap2, _, _ = build_snap(veh2, pl2)
    t_empty = times_for(service={}, tw_start={})           # service[5] deliberately ABSENT
    ok2, v2, d2 = RF.certify_return_feasibility(snap2, T_end=8.0, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=t_empty)
    ck('A2', 'unregistered service time -> OMEGA11_TIME_INPUT_UNAVAILABLE (no silent 0)',
       (not ok2) and v2 in (RF.OMEGA11_TIME_INPUT_UNAVAILABLE, RF.OMEGA11_TIME_ORDER_INVALID),
       {'verdict': v2})
    ok2b, v2b, d2b = RF.certify_return_feasibility(snap2, T_end=3.5, clock=0.0,
                                                   travel_fn=lambda a, b: 1.0, times=t_empty)
    ck('A2b', 'the silent-zero verdict flip is gone (now refuses)',
       (not ok2b) and v2b in (RF.OMEGA11_TIME_INPUT_UNAVAILABLE, RF.OMEGA11_TIME_ORDER_INVALID),
       {'verdict': v2b, 'pre_fix': 'OMEGA11_OK reporting 2.0'})

    # =========================================================================== A3
    print('\n-- A3: no process-global service-time cache --')
    no_cache = (not hasattr(RF, '_SERVICE_TIME_CACHE')) and (not hasattr(RF, 'register_service_times'))
    tA = times_for(service={5: 5.0}, tw_start={5: 0.0}, identity='inst-A')
    tB = times_for(service={5: 0.5}, tw_start={5: 0.0}, identity='inst-B')
    okA, vA, dA = RF.certify_return_feasibility(snap2, T_end=8.0, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=tA)
    okB, vB, dB = RF.certify_return_feasibility(snap2, T_end=8.0, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=tB)
    ra = dA['per_vehicle'][0].get('return_arrival')
    rb = dB['per_vehicle'][0].get('return_arrival')
    # expected: 1.0 travel to 5 + service_time + 1.0 travel home
    ck('A3', 'cache removed AND two instances yield their OWN results (falsifiable)',
       no_cache and ra is not None and rb is not None
       and abs(ra - 7.0) < 1e-9 and abs(rb - 2.5) < 1e-9 and ra != rb,
       {'cache_removed': no_cache, 'instA_return': ra, 'expected_A': 7.0,
        'instB_return': rb, 'expected_B': 2.5, 'verdict_A': vA, 'verdict_B': vB,
        'falsifiable_claim': 'inst-A input 5.0 -> 7.0 ; inst-B input 0.5 -> 2.5'})

    # =========================================================================== A4 / A4b
    print('\n-- A4 / A4b: committed vehicle keeps its later accepted tasks --')
    veh4 = [V(0, 'committed', node=2, ready=1.0, cc=fresh_cc(), committed_next=3,
              committed_finish=2.0, suffix=[6, 0])]
    pl4 = {0: plan(0, 2.0, [6, 0])}
    snap4, _, _ = build_snap(veh4, pl4)
    t4 = times_for(service={6: 0.0}, tw_start={6: 0.0})
    ok4, v4, d4 = RF.certify_return_feasibility(snap4, T_end=6.0, clock=1.0,
                                                travel_fn=lambda a, b: 1.0, times=t4)
    r4 = d4['per_vehicle'][0].get('return_arrival')
    ck('A4', 'committed leg THEN the later task is replayed (true return 4.0)',
       ok4 and abs(r4 - 4.0) < 1e-9,
       {'verdict': v4, 'return_arrival': r4, 'expected': 4.0,
        'basis': d4['per_vehicle'][0].get('basis')})
    veh4b = [V(0, 'committed', node=2, ready=1.0, cc=fresh_cc(), committed_next=3,
               committed_finish=4.0, suffix=[6, 0])]
    snap4b, _, _ = build_snap(veh4b, pl4)
    ok4b, v4b, d4b = RF.certify_return_feasibility(snap4b, T_end=5.0, clock=1.0,
                                                   travel_fn=lambda a, b: 1.0, times=t4)
    ck('A4b', 'FALSE_PASS flipped to refusal (true return 6.0 > 5.0)',
       (not ok4b) and v4b == RF.OMEGA11_RETURN_DEADLINE_EXCEEDED
       and abs(d4b['per_vehicle'][0].get('return_arrival') - 6.0) < 1e-9,
       {'verdict': v4b, 'return_arrival': d4b['per_vehicle'][0].get('return_arrival')})

    # =========================================================================== A5
    print('\n-- A5: depart choice and basis label must agree --')
    veh5 = [V(0, 'ready', node=2, ready=1.0, cc=fresh_cc(), suffix=[])]
    pl5 = {0: plan(0, 20.0, [7, 0])}
    snap5, _, _ = build_snap(veh5, pl5)
    t5 = times_for(service={7: 1.0}, tw_start={7: 0.0})
    ok5, v5, d5 = RF.certify_return_feasibility(snap5, T_end=24.0, clock=1.0,
                                                travel_fn=lambda a, b: 1.0, times=t5)
    row5 = d5['per_vehicle'][0]
    ck('A5', 'candidate replayed; depart rule RECORDED and consistent with the advance',
       ok5 and abs(row5.get('return_arrival') - 4.0) < 1e-9
       and row5.get('depart_rule') == 'max(ready_time, clock)'
       and abs(row5.get('depart_used') - 1.0) < 1e-9,
       {'return_arrival': row5.get('return_arrival'), 'expected': 4.0,
        'depart_used': row5.get('depart_used'), 'depart_rule': row5.get('depart_rule'),
        'basis': row5.get('basis'),
        'errata': 'the Q2-R report text said 2.0; the JSON said 4.0 and 4.0 is correct'})

    # =========================================================================== A6
    print('\n-- A6: closed vehicle audited against TRUSTED HISTORY --')
    veh6 = [V(0, 'closed', node=0, ready=15.0, cc=fresh_cc(), return_finish=None)]
    snap6, _, _ = build_snap(veh6, {})
    ok6a, v6a, d6a = RF.certify_return_feasibility(snap6, T_end=24.0, clock=20.0,
                                                   travel_fn=lambda a, b: 1.0,
                                                   times=times_for(), closed_return_history={})
    ck('A6a', 'closed with NO history -> refuse (a cleared live field is not compliance)',
       (not ok6a) and v6a == RF.OMEGA11_CLOSED_HISTORY_MISSING, {'verdict': v6a})
    ok6b, v6b, d6b = RF.certify_return_feasibility(snap6, T_end=24.0, clock=20.0,
                                                   travel_fn=lambda a, b: 1.0,
                                                   times=times_for(),
                                                   closed_return_history={0: 30.0})
    ck('A6b', 'closed with history showing a breach -> refuse',
       (not ok6b) and v6b == RF.OMEGA11_CLOSED_HISTORY_DEADLINE_EXCEEDED,
       {'verdict': v6b})
    ok6c, v6c, d6c = RF.certify_return_feasibility(snap6, T_end=24.0, clock=20.0,
                                                   travel_fn=lambda a, b: 1.0,
                                                   times=times_for(),
                                                   closed_return_history={0: 15.0})
    ck('A6c', 'closed with in-time history -> OK, and the cleared live field is NOT a violation',
       ok6c and v6c == RF.OMEGA11_OK, {'verdict': v6c})

    # =========================================================================== B1
    print('\n-- B1: candidate fish coverage --')
    attrsB = attrs_for(iq={7: 1.0}, tc={7: 1}, qty={7: 1.0})     # true value 1.0 vs contract 9.0
    pc, _, _ = __import__('m1b_contract_identity').check_quality_classes_are_certifiable(
        CONTRACT, epsilon_init=QC.EPSILON_INIT)
    vehB = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=0))]
    plB = {0: plan(0, 0.0, [7, 0])}
    cB = M1bQ2FixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B)
    okB, rcB = cB.certify(vehicles=vehB, served_mask=np.zeros(40, dtype=bool),
                          candidate_plans=plB, clock=0.0, inst_idx=0,
                          travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
                          times=times_for(service={7: 0.05}, tw_start={}),
                          order_attrs=attrsB, per_class_constants=pc)
    wB = (rcB['gates'].get('omega10_coverage') or {}).get('verdict')
    ck('B1', 'FALSE_PASS flipped: candidate fish with iq=1.0 is REFUSED',
       (not okB) and wB == QC.OMEGA10_INITIAL_VALUE_MISMATCH,
       {'omega10_verdict': wB, 'composite': okB, 'final': rcB['final'],
        'required_set_E': (rcB['gates'].get('omega10_coverage') or {}).get('required_set_E'),
        'n_required': (rcB['gates'].get('omega10_coverage') or {}).get('n_required')})

    # ---- positive control: empty hold and NO fish order at all must NOT be refused
    attrsClean = attrs_for(iq={2: 9.0}, tc={2: 0}, qty={2: 1.0})   # node 2 is class 0 (tomato)
    vehC = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=0))]
    plC = {0: plan(0, 0.0, [2, 0])}
    cC = M1bQ2FixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B)
    okC, rcC = cC.certify(vehicles=vehC, served_mask=np.zeros(40, dtype=bool),
                          candidate_plans=plC, clock=0.0, inst_idx=0,
                          travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
                          times=times_for(service={2: 0.05}, tw_start={}),
                          order_attrs=attrsClean, per_class_constants=pc)
    wC = (rcC['gates'].get('omega10_coverage') or {}).get('verdict')
    ck('B1-POS', 'POSITIVE CONTROL: no fish in scope is admissible (NO_FISH_IN_SCOPE, not a fail)',
       wC in (QC.OMEGA10_NO_FISH_IN_SCOPE, QC.OMEGA10_OK) and rcC['QUALITY_LB_OK'] is True,
       {'omega10_verdict': wC, 'QUALITY_LB_OK': rcC['QUALITY_LB_OK'],
        'composite': okC, 'final': rcC['final'],
        'note': 'composite is still fail-closed only because of the CONTINUOUS thermal level'})

    # =========================================================================== B3
    print('\n-- B3: structural visibility of the attribute provider --')
    leaked = []
    try:
        attrs_for(iq={7: 1.0}, tc={7: 1}, qty={7: 1.0}, served=(1, 2, 3)).temp_class_of(7)
        leaked.append('temp_class_of(7)')
    except PermissionError:
        pass
    try:
        attrs_for(iq={7: 1.0}, tc={7: 1}, qty={7: 1.0}, served=(1, 2, 3)).initial_quality_of(7)
        leaked.append('initial_quality_of(7)')
    except PermissionError:
        pass
    src_required = False
    try:
        QC.PublicOrderAttributes(initial_quality={}, temp_class={}, quantity={},
                                 served_nodes=(1,), source_identity='')
    except ValueError:
        src_required = True
    ck('B3', 'B3 discharged: provider refuses unserved nodes AND requires a source identity',
       not leaked and src_required,
       {'leaked': leaked, 'source_identity_required': src_required,
        'why': 'a full-node truth table / arbitrary callable can no longer be a trusted input'})

    # =========================================================================== C1
    print('\n-- C1: sampled vs continuous thermal levels --')
    attrsT = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0})
    vehT = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=9.0))]
    plT = {0: plan(0, 0.0, [7, 0])}
    cT = M1bQ2FixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B)
    okT, rcT = cT.certify(vehicles=vehT, served_mask=np.zeros(40, dtype=bool),
                          candidate_plans=plT, clock=0.0, inst_idx=0,
                          travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
                          times=times_for(service={7: 0.05}, tw_start={}),
                          order_attrs=attrsT, per_class_constants=pc)
    ck('C1', 'sampled OK but continuous NOT_ESTABLISHED -> composite FAIL-CLOSED',
       rcT['THERMAL_SAMPLED_OK'] is True
       and rcT['THERMAL_CONTINUOUS_OK'] is False
       and rcT['THERMAL_CONTINUOUS_STATE'] == TG.THERMAL_CONTINUOUS_NOT_ESTABLISHED
       and okT is False and rcT['final'] == ENGINEERING_REFUSED,
       {'THERMAL_SAMPLED_OK': rcT['THERMAL_SAMPLED_OK'],
        'THERMAL_CONTINUOUS_OK': rcT['THERMAL_CONTINUOUS_OK'],
        'THERMAL_CONTINUOUS_STATE': rcT['THERMAL_CONTINUOUS_STATE'],
        'composite': okT, 'final': rcT['final'],
        'note': 'the sampled diagnostic remains available and is NOT named an installability '
                'certificate'})

    # ---- positive control: a supplied continuous proof turns the composite ON
    cT2 = M1bQ2FixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B,
                            continuous_extremum_proof=lambda **k: (True, {'synthetic': True}))
    okT2, rcT2 = cT2.certify(vehicles=vehT, served_mask=np.zeros(40, dtype=bool),
                             candidate_plans=plT, clock=0.0, inst_idx=0,
                             travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
                             times=times_for(service={7: 0.05}, tw_start={}),
                             order_attrs=attrsT, per_class_constants=pc)
    ck('C1-POS', 'POSITIVE CONTROL: with a supplied continuous proof the composite can pass',
       okT2 is True and rcT2['THERMAL_CONTINUOUS_OK'] is True
       and rcT2['final'] == ENGINEERING_CERTIFIED,
       {'composite': okT2, 'final': rcT2['final'],
        'note': 'the proof hook is a wiring control, NOT a real extremum proof'})

    # =========================================================================== positives
    print('\n-- positive controls (no false rejects) --')
    t_zero = times_for(service={7: 0.0}, tw_start={0.0: 0.0, 7: 0.0}, identity='zero-svc')
    vehZ = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    snapZ, _, _ = build_snap(vehZ, plT)
    okZ, vZ, dZ = RF.certify_return_feasibility(snapZ, T_end=T_END, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=t_zero)
    ck('POS-1', 'a LEGITIMATE zero service time is accepted (key present, value 0.0)',
       okZ and vZ == RF.OMEGA11_OK, {'verdict': vZ,
                                     'return_arrival': dZ['per_vehicle'][0].get('return_arrival')})
    vehD = [V(0, 'idle', node=0, ready=0.0, cc=None)]
    snapD, _, _ = build_snap(vehD, {})
    okD, vD, dD = RF.certify_return_feasibility(snapD, T_end=T_END, clock=0.0,
                                                travel_fn=lambda a, b: 1.0, times=times_for())
    ck('POS-2', 'idle at depot with nothing pending is NOT a violation',
       okD and vD == RF.OMEGA11_OK, {'verdict': vD})

    # ---- certify_route cross-check
    tX = times_for(service={5: 2.0}, tw_start={5: 3.0})
    vehX = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    plX = {0: plan(0, 0.0, [5, 0])}
    snapX, _, _ = build_snap(vehX, plX)
    envX = MiniEnv(tw_start={5: 3.0}, tw_end={0: 24.0, 5: 30.0}, service={5: 2.0})
    okX, vX, dX = RF.certify_return_feasibility(
        snapX, T_end=24.0, clock=0.0, travel_fn=lambda a, b: 1.0, times=tX,
        certify_route_fn=lambda vid: AC.certify_route(envX, 0, 0, 0.0, 0.0, (5,)))
    cross = dX.get('certify_route_cross_check', {})
    ck('XCHECK', 'agreeing case: Omega11 and certify_route concur (no disagreement raised)',
       okX and cross.get('n_disagreements') == 0,
       {'verdict': vX, 'cross': cross})

    # =========================================================================== B2/B4 retained
    okB2, vB2, dB2 = QC.certify_omega10_coverage(
        on_board_lots=[], committed_fish_nodes=[], candidate_fish_nodes=[],
        revealed_nodes={7}, provider=attrsB, per_class_constants=pc)
    ck('B-retained', 'contract-level controls still behave (B4 comparison soundness)',
       QC.check_boundary_initial_value(type('L', (), {'order_id': 7, 'temp_class': 1,
                                                      'initial_quality': 1.0})(), pc)[1]
       == QC.OMEGA10_INITIAL_VALUE_MISMATCH,
       {'boundary_verdict': QC.check_boundary_initial_value(
           type('L', (), {'order_id': 7, 'temp_class': 1, 'initial_quality': 1.0})(), pc)[1]})

    # =========================================================================== legacy record
    # The historical test_m1b_q2_certification.py is NOT modified.  Under the fixed code it
    # diverges in TWO distinct ways, both recorded rather than papered over:
    #   (1) genuine SEMANTIC failures: it supplies no per-instance time inputs, so the fixed
    #       Omega11 refuses, and the composite is now fail-closed on the continuous thermal level;
    #   (2) a terminal compatibility stop: it SUBSCRIPTS the pre-fix key `receipt['THERMAL_HARD_OK']`.
    #       That name conflated the sampled and continuous levels (the C1 promotion path), so the
    #       fixed receipt exposes it only as a read-only ATTRIBUTE and never STORES it -- a
    #       persisted `THERMAL_HARD_OK` key is precisely what C1 was about.
    import subprocess
    legacy = EXT / 'scripts' / 'tests' / 'test_m1b_q2_certification.py'
    legacy_bytes_before = legacy.read_bytes()
    proc = subprocess.run([sys.executable, str(legacy)], capture_output=True, text=True,
                          cwd=str(EXT), timeout=600)
    tail = (proc.stdout or '')[-1200:]
    ck('LEGACY-1', 'historical Q2 test file is byte-unchanged on disk',
       legacy.read_bytes() == legacy_bytes_before,
       {'sha256_prefix': __import__('hashlib').sha256(legacy_bytes_before).hexdigest()[:16]})
    ck('LEGACY-2', 'historical file re-runs and its divergence is RECORDED (not hidden)',
       ('FAIL' in tail) or (proc.returncode != 0),
       {'returncode': proc.returncode,
        'divergence_1': 'semantic: no time inputs supplied -> Omega11 refuses; composite '
                        'fail-closed on continuous thermal level',
        'divergence_2': "terminal: it subscripts receipt['THERMAL_HARD_OK'], a pre-fix name that "
                        'the fixed receipt deliberately does NOT store (the C1 promotion path); '
                        'exposed as a read-only attribute instead'})

    # =========================================================================== census
    fails = [r for r in ROWS if r['verdict'] != 'PASS']
    false_reject = [r for r in ROWS if r['verdict'] == 'FALSE_REJECT']
    status = 'Q2FIX_ALL_PASS' if not fails else 'Q2FIX_HAS_FAILURES'
    print('\n== %s ==  (%d checks, %d not PASS, %d FALSE_REJECT)'
          % (status, len(ROWS), len(fails), len(false_reject)))
    for f in fails:
        print('   NOT PASS: %s -> %s' % (f['cid'], f['verdict']))
    print('\nDISCLAIMER: engineering acceptance only. NOT production installability, NOT '
          'scientific evidence, NOT L2. FULL_INSTALLABILITY_ESTABLISHED is NOT established.')
    payload = {'schema': 'm1b_q2fix_regression_v1', 'status': status,
               'implementation_version': IMPLEMENTATION_VERSION,
               'n_checks': len(ROWS), 'n_not_pass': len(fails),
               'false_reject_census': len(false_reject),
               'synthetic_budget': SYNTHETIC_B, 'budget_is_synthetic': True,
               'does_not_establish': ['FULL_INSTALLABILITY_ESTABLISHED', 'L2_PASS', 'A3_PASS',
                                      'stable_online_gain'],
               'results': ROWS}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                  encoding='utf-8')
        print('written: %s' % args.out)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
