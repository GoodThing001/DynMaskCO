#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-R: TARGETED NEGATIVE-CONTROL AUDIT of the M1-b/Q2 certification chain.

This file exists to FALSIFY the Q2 implementation, not to flatter it.  It is a diagnostic
suite: every case reports the real return value, the source line responsible, and a
classification of FALSE_PASS / FALSE_REJECT / NOT_ESTABLISHED.

Task A  Omega11 time-feasibility negative controls (6 cases).
Task B  Omega10 coverage negative controls.
Task C  certification-chain completeness audit.

Pure NumPy / stdlib.  No model, no GPU, no rollout, no installation, no frozen-source edit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in (EXT / 'scripts' / 'coldchain', EXT / 'scripts' / 'simulation',
           EXT / 'scripts' / 'evaluation', EXT / 'scripts' / 'm1b_adapter',
           EXT / 'scripts' / 'm1a_adapter', HERE):
    sys.path.insert(0, str(_p))

import numpy as np                                                       # noqa: E402
import action_contract as AC                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import coldchain_state as CS                                             # noqa: E402
from coldchain_state import create_vehicle_state, dispatch_vehicle       # noqa: E402
from action_contract import VehiclePlan                                   # noqa: E402

import m1b_return_feasibility as RF                                      # noqa: E402
import m1b_quality_certificate as QC                                     # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
from m1b_q2_certifier import M1bQ2Certifier, ENGINEERING_CERTIFIED       # noqa: E402

ROWS = []
CONTRACT = ColdChainContractV2()
T_END = 24.0
SYNTHETIC_B = 12345.678          # SYNTHETIC ONLY (interface / budget-refusal semantics)


def row(cid, name, inp, expected, actual, cls, root, line):
    ROWS.append({'cid': cid, 'name': name, 'INPUT': inp, 'EXPECTED': expected,
                 'ACTUAL': actual, 'CLASS': cls, 'ROOT_CAUSE': root, 'SOURCE_LINE': line})
    print('  [%-7s] %-46s %s' % (cid, name[:46], cls))


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


# ---------------------------------------------------------------- a REAL env for certify_route
class MiniEnv(object):
    """Minimal env exposing exactly what certify_route reads, so the two implementations can be
    compared on IDENTICAL time data."""

    def __init__(self, n=8, tw_start=None, tw_end=None, service=None, travel=1.0, capacity=50.0):
        self.num_nodes = n
        self.capacity = capacity
        self.tw_speed = 1.0
        self.coldchain_contract = CONTRACT
        self.tw_start = np.zeros((1, n), dtype=np.float64)
        self.tw_end = np.full((1, n), 1e9, dtype=np.float64)
        self.service_time = np.zeros((1, n), dtype=np.float64)
        self.demands = np.ones((1, n), dtype=np.float64)
        self.temp_class = np.ones((1, n), dtype=np.int64)
        # certify_route reads dist_mat and divides by tw_speed, so a uniform `travel` matrix with
        # tw_speed = 1.0 makes every leg take exactly `travel` time units.
        self.dist_mat = np.full((1, n, n), float(travel), dtype=np.float64)
        for k, v in (tw_start or {}).items():
            self.tw_start[0, int(k)] = float(v)
        for k, v in (tw_end or {}).items():
            self.tw_end[0, int(k)] = float(v)
        for k, v in (service or {}).items():
            self.service_time[0, int(k)] = float(v)


def build_snap(vehicles, plans, iq=None):
    snap, v, d = IS.build_frozen_snapshot(vehicles, np.zeros(40, dtype=bool), plans,
                                          initial_quality_by_node=iq or {})
    return snap, v, d


# ==============================================================================================
def task_A():
    print('\n===== TASK A: Omega11 time-feasibility negative controls =====')
    RF.register_service_times({})          # clear any residual cache at the start

    # ---------------------------------------------------------------- A1 late tw_start waiting
    # A customer whose window opens at 20.0.  Travel 1.0 each leg, service 0.0.
    # certify_route: arrive 2.0, start max(2.0, 20.0) = 20.0, finish 20.0, return 21.0  -> OK
    # Make T_end 19.0 so the TRUE return (21.0) exceeds it while a tw_start-ignoring replay that
    # starts at 0.0 would compute 2.0 and pass.
    env1 = MiniEnv(tw_start={5: 20.0}, tw_end={0: 19.0, 5: 30.0}, service={5: 0.0})
    veh1 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    pl1 = {0: plan(0, 0.0, [5, 0])}
    snap1, _, _ = build_snap(veh1, pl1)
    ok11, v11, d11 = RF.certify_return_feasibility(snap1, T_end=19.0, clock=0.0,
                                                  travel_fn=lambda a, b: 1.0)
    cert = AC.certify_route(env1, 0, 0, 0.0, 0.0, (5,))
    got = d11['per_vehicle'][0].get('return_arrival')
    true_ret = 21.0
    row('A1', 'late tw_start (wait) -> true return beyond T_end',
        {'tw_start[5]': 20.0, 'T_end': 19.0, 'travel': 1.0, 'service': 0.0},
        'OMEGA11_RETURN_DEADLINE_EXCEEDED (true return %.1f > 19.0+1e-6)' % true_ret,
        {'omega11_verdict': v11, 'omega11_return_arrival': got,
         'certify_route_feasible': cert['feasible'], 'certify_route_reason': cert['reason'],
         'certify_route_return_slack': cert['return_slack']},
        'FALSE_PASS' if ok11 else 'OK',
        'Omega11 computes t = max(ready_time, clock) and NEVER applies '
        'start = max(arrive, tw_start); it therefore omits the waiting time that '
        'certify_route includes (action_contract.py:187).',
        'm1b_return_feasibility.py:_replay (t = float(max(v.ready_time, clock)))')

    # ---------------------------------------------------------------- A2 missing service time
    # Real service time 2.0 at the served customer; the cache is NOT registered for it.
    env2 = MiniEnv(tw_end={0: 8.0, 5: 30.0}, service={5: 2.0})
    veh2 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    pl2 = {0: plan(0, 0.0, [5, 0])}
    snap2, _, _ = build_snap(veh2, pl2)
    RF.register_service_times({})           # deliberately EMPTY -> default 0.0
    ok2, v2, d2 = RF.certify_return_feasibility(snap2, T_end=8.0, clock=0.0,
                                                travel_fn=lambda a, b: 1.0)
    cert2 = AC.certify_route(env2, 0, 0, 0.0, 0.0, (5,))
    row('A2', 'service time unregistered -> cache default 0 silently used',
        {'service_time[5]': 2.0, 'registered_service_times': {}, 'T_end': 8.0, 'travel': 1.0},
        'refuse (service time unavailable) OR return 4.0 > 8.0? true return = 1+2+1 = 4.0',
        {'omega11_verdict': v2, 'omega11_return_arrival': d2['per_vehicle'][0].get('return_arrival'),
         'certify_route_feasible': cert2['feasible'],
         'note': 'with T_end=8.0 both agree by luck; the DEFECT is that 0.0 is used silently '
                 'and there is no refusal when the service time is unknown'},
        'NOT_ESTABLISHED',
        '_SERVICE_TIME_CACHE.get(node, 0.0) defaults to ZERO service duration with no '
        'provenance check, while certify_route reads env.service_time (action_contract.py:196).',
        'm1b_return_feasibility.py:_service_time (return float(_SERVICE_TIME_CACHE.get(node, 0.0)))')

    # A2b: make the silent default decisive
    env2b = MiniEnv(tw_end={0: 3.5, 5: 30.0}, service={5: 2.0})
    veh2b = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    snap2b, _, _ = build_snap(veh2b, pl2)
    RF.register_service_times({})
    ok2b, v2b, d2b = RF.certify_return_feasibility(snap2b, T_end=3.5, clock=0.0,
                                                   travel_fn=lambda a, b: 1.0)
    cert2b = AC.certify_route(env2b, 0, 0, 0.0, 0.0, (5,))
    row('A2b', 'silent zero service time flips the verdict',
        {'service_time[5]': 2.0, 'registered': {}, 'T_end': 3.5, 'travel': 1.0},
        'refuse; true return 4.0 > 3.5+1e-6',
        {'omega11_verdict': v2b, 'omega11_return_arrival': d2b['per_vehicle'][0].get('return_arrival'),
         'certify_route_feasible': cert2b['feasible'], 'certify_route_reason': cert2b['reason']},
        'FALSE_PASS' if ok2b else 'OK',
        'the zero default makes the replay SHORTER than reality, which can only produce '
        'optimistic (false-pass) verdicts.',
        'm1b_return_feasibility.py:_service_time')

    # ---------------------------------------------------------------- A3 cross-instance residue
    RF.register_service_times({5: 5.0})     # instance 1 registers a LONG service time
    veh3 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    snap3, _, _ = build_snap(veh3, pl2)
    ok3a, v3a, d3a = RF.certify_return_feasibility(snap3, T_end=8.0, clock=0.0,
                                                   travel_fn=lambda a, b: 1.0)
    RF.register_service_times({5: 0.5})     # instance 2 registers a SHORT one
    ok3b, v3b, d3b = RF.certify_return_feasibility(snap3, T_end=8.0, clock=0.0,
                                                   travel_fn=lambda a, b: 1.0)
    row('A3', 'cross-instance service-time residue in the module-global cache',
        {'instance1_service[5]': 5.0, 'instance2_service[5]': 0.5, 'T_end': 8.0},
        'each run uses its OWN instance service times',
        {'run1_return': d3a['per_vehicle'][0].get('return_arrival'),
         'run2_return': d3b['per_vehicle'][0].get('return_arrival'),
         'run1_verdict': v3a, 'run2_verdict': v3b},
        'NOT_ESTABLISHED' if d3a['per_vehicle'][0].get('return_arrival')
        != d3b['per_vehicle'][0].get('return_arrival') else 'OK',
        'the cache is MODULE-GLOBAL and keyed by node only, so a stale entry from a previous '
        'instance is reused until the caller re-registers; nothing binds the table to an '
        'instance identity, so a forgotten re-registration silently reuses another instance.',
        'm1b_return_feasibility.py:_SERVICE_TIME_CACHE (module-level dict) + register_service_times')

    # ---------------------------------------------------------------- A4 committed + later tasks
    # A committed vehicle whose committed leg finishes in time, but which ALSO carries an
    # already-accepted subsequent obligation in its mutable suffix.
    veh4 = [V(0, 'committed', node=2, ready=1.0, cc=fresh_cc(),
              committed_next=3, committed_finish=2.0, suffix=[6, 0])]
    pl4 = {0: plan(0, 2.0, [6, 0])}
    snap4, _, _ = build_snap(veh4, pl4)
    RF.register_service_times({6: 0.0})
    ok4, v4, d4 = RF.certify_return_feasibility(snap4, T_end=6.0, clock=1.0,
                                                travel_fn=lambda a, b: 1.0)
    row('A4', 'committed vehicle also holding a later accepted task in its suffix',
        {'status': 'committed', 'committed_next': 3, 'committed_finish': 2.0,
         'mutable_suffix': [6, 0], 'T_end': 6.0, 'travel': 1.0},
        'the audit must account for BOTH the committed leg AND the later suffix task; '
        'true return = 2.0 (committed finish) + 1.0 (3->6) + 1.0 (6->depot) = 4.0',
        {'omega11_verdict': v4, 'omega11_return_arrival': d4['per_vehicle'][0].get('return_arrival'),
         'basis': d4['per_vehicle'][0].get('basis')},
        'NOT_ESTABLISHED',
        'the committed branch computes ONLY committed_finish + travel(committed_next -> depot) '
        'and NEVER inspects the suffix, so a later accepted task on the same vehicle is omitted '
        'from the return computation.',
        'm1b_return_feasibility.py: elif v.status == "committed" branch')

    # A4b: make the omission decisive
    veh4b = [V(0, 'committed', node=2, ready=1.0, cc=fresh_cc(),
               committed_next=3, committed_finish=4.0, suffix=[6, 0])]
    snap4b, _, _ = build_snap(veh4b, pl4)
    ok4b, v4b, d4b = RF.certify_return_feasibility(snap4b, T_end=5.0, clock=1.0,
                                                   travel_fn=lambda a, b: 1.0)
    row('A4b', 'committed omission flips the verdict',
        {'committed_finish': 4.0, 'committed_next': 3, 'suffix': [6, 0], 'T_end': 5.0},
        'refuse; true return = 4.0 + 1.0 + 1.0 = 6.0 > 5.0+1e-6',
        {'omega11_verdict': v4b, 'omega11_return_arrival': d4b['per_vehicle'][0].get('return_arrival'),
         'basis': d4b['per_vehicle'][0].get('basis')},
        'FALSE_PASS' if ok4b else 'OK',
        'same omission as A4: only the committed leg is timed.',
        'm1b_return_feasibility.py: committed branch')

    # ---------------------------------------------------------------- A5 WAIT can return now
    # At clock 1.0 a WAITing vehicle CAN reach the depot in time; but the plan permits waiting for
    # a later customer, and once that customer is served the return exceeds the deadline.  The
    # gate must judge the plan it is certifying, not merely the earliest possible return.
    veh5 = [V(0, 'ready', node=2, ready=1.0, cc=fresh_cc(), suffix=[])]
    pl5 = {0: plan(0, 20.0, [7, 0])}       # the candidate plan DOES continue to customer 7
    snap5, _, _ = build_snap(veh5, pl5)
    RF.register_service_times({7: 1.0})
    ok5, v5, d5 = RF.certify_return_feasibility(snap5, T_end=24.0, clock=1.0,
                                                travel_fn=lambda a, b: 1.0)
    row('A5', 'WAIT judged as "return now" while the certified plan continues',
        {'suffix_in_vehicle': [], 'candidate_plan_suffix': [7, 0], 'clock': 1.0,
         'service[7]': 1.0, 'T_end': 24.0},
        'must time the CANDIDATE plan it certifies: 20.0 (anchor) + 1 (2->7) + 1 (svc) '
        '+ 1 (7->depot) = 23.0 <= 24.0, so the true verdict is OK here',
        {'omega11_verdict': v5, 'omega11_return_arrival': d5['per_vehicle'][0].get('return_arrival'),
         'basis': d5['per_vehicle'][0].get('basis'),
         'n_suffix_stops': d5['per_vehicle'][0].get('n_suffix_stops')},
        'NOT_ESTABLISHED',
        'the WAIT path is entered only when the candidate suffix is EMPTY, and then the vehicle '
        'is judged by "return now".  This specific fixture shows the gate reporting a return of '
        '2.0 (immediate) while the certified candidate plan actually continues to customer 7; '
        'the two numbers describe different plans.',
        'm1b_return_feasibility.py: else-branch (suffix = plan.suffix or mutable_suffix)')

    # ---------------------------------------------------------------- A6 closed vehicle audit
    veh6 = [V(0, 'closed', node=0, ready=15.0, cc=fresh_cc(), return_finish=30.0)]
    snap6, _, _ = build_snap(veh6, {})
    ok6, v6, d6 = RF.certify_return_feasibility(snap6, T_end=24.0, clock=20.0,
                                                travel_fn=lambda a, b: 1.0)
    row('A6', 'closed vehicle with a recorded return LATER than T_end',
        {'status': 'closed', 'return_finish': 30.0, 'T_end': 24.0},
        'a closed vehicle whose recorded return exceeded the deadline must be auditable; '
        'the current branch treats closed as no-pending and never examines return_finish',
        {'omega11_verdict': v6, 'row': d6['per_vehicle'][0]},
        'NOT_ESTABLISHED',
        'the closed branch sets pending = False and skips the deadline comparison entirely, so a '
        'historical deadline breach on a closed vehicle is invisible to this gate.',
        'm1b_return_feasibility.py: if v.status == "closed" branch')


# ==============================================================================================
def _mini_env_for_thermal():
    class E(object):
        pass
    return E()


def task_B():
    print('\n===== TASK B: Omega10 coverage negative controls =====')
    pc, pcv, _ = __import__('m1b_contract_identity').check_quality_classes_are_certifiable(
        CONTRACT, epsilon_init=QC.EPSILON_INIT)

    # Current manifest has NO fish; the candidate plan references a REVEALED fish order whose
    # TRUE initial quality is 1.0 (the Q1 logical counterexample), while the contract says 9.0.
    TCLASS = {7: 1}
    QTY = {7: 1.0}
    IQ = {7: 1.0}                      # the true, wrong value
    SVC = {7: 0.05}
    TRAVEL = {}

    def tfn(a, b):
        return 1.0

    def dfn(a, b):
        return 10.0

    veh = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=0))]
    pl = {0: plan(0, 0.0, [7, 0])}

    def run(lot_provider=None):
        c = M1bQ2Certifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B)
        return c.certify(vehicles=veh, served_mask=np.zeros(40, dtype=bool),
                         candidate_plans=pl, clock=0.0, inst_idx=0, travel_fn=tfn,
                         distance_fn=dfn, temp_class_of=lambda n: TCLASS.get(int(n), 1),
                         quantity_of=lambda n: QTY.get(int(n), 1.0),
                         service_time_of=lambda n: SVC.get(int(n), 0.5),
                         initial_quality_by_node=IQ, lot_provider=lot_provider)

    ok_def, rc_def = run(None)
    w10d = (rc_def['gates'].get('omega10_initialization') or {}).get('verdict')
    row('B1', 'DEFAULT lot_provider: candidate fish order never checked',
        {'manifest_fish': 0, 'candidate_plan': [7, 0], 'temp_class[7]': 1,
         'true initial_quality[7]': 1.0, 'contract initial_value[1]': 9.0},
        'the fish order carried by the certified candidate plan must be checked; its true 1.0 '
        'differs from 9.0 by 8.0 >> eps_init, so Omega10 must REFUSE',
        {'omega10_verdict': w10d, 'composite': ok_def, 'final': rc_def['final'],
         'QUALITY_LB_OK': rc_def['QUALITY_LB_OK'], 'n_checked':
             (rc_def['gates']['omega10_initialization']['detail'] or {}).get('n_checked')},
        'FALSE_PASS' if (ok_def and w10d == QC.OMEGA10_NO_RELEVANT_LOTS) else 'NOT_ESTABLISHED',
        'the default lot provider harvests only cargo-manifest lots of vehicles that have a plan, '
        'so when nothing is on board it returns NO lots; check_omega10_initialization then returns '
        'NO_RELEVANT_LOTS with ok=True, and the orchestrator treats ok=True as a pass.',
        'm1b_q2_certifier.py:_default_lot_provider; '
        'm1b_quality_certificate.py:check_omega10_initialization (checked == 0 -> ok True)')

    # explicit provider that DOES include the candidate order
    class FakeOrder(object):
        order_id = 7
        temp_class = 1
        initial_quality = 1.0

    ok_exp, rc_exp = run(lambda snap: [FakeOrder()])
    w10e = (rc_exp['gates'].get('omega10_initialization') or {}).get('verdict')
    row('B2', 'EXPLICIT lot_provider including the candidate order',
        {'provider': 'returns the candidate fish order with iq=1.0'},
        'OMEGA10_INITIAL_VALUE_MISMATCH and composite REFUSED',
        {'omega10_verdict': w10e, 'composite': ok_exp, 'final': rc_exp['final'],
         'QUALITY_LB_OK': rc_exp['QUALITY_LB_OK'],
         'quality_impl_call': (rc_exp['gates'].get('quality_impl_call') or {}).get('verdict')},
        'FALSE_PASS' if ok_exp else 'OK',
        'the gate itself is correct when it is GIVEN the lots; the hole is that the DEFAULT '
        'provider decides what is relevant, and the orchestrator cannot tell "nothing to check" '
        'from "checked and fine".',
        'm1b_quality_certificate.py:OMEGA10_NO_RELEVANT_LOTS (ok=True)')

    # B3: interface exposure of future information
    exposure = []
    for name in ('initial_quality_by_node', 'temp_class_of', 'quantity_of'):
        exposure.append((name, 'caller-supplied mapping covering ALL nodes, including '
                               'unrevealed ones, and passed into the snapshot'))
    row('B3', 'interfaces that can carry unrevealed-order information',
        {'initial_quality_by_node': IQ, 'temp_class_of': 'lambda over all nodes',
         'quantity_of': 'lambda over all nodes'},
        'no interface used by certification should accept unrevealed-order data',
        {'interfaces': exposure,
         'snapshot_carries_iq_table': True},
        'NOT_ESTABLISHED',
        'initial_quality_by_node is a PUBLIC table but the adapter accepts it for ALL nodes; '
        'temp_class_of/quantity_of are arbitrary callables over all nodes. Nothing structurally '
        'prevents a caller from passing a table that only happens to be complete because it holds '
        'FUTURE values, and the thermal replay calls them for suffix nodes only -- so the risk is '
        'a caller error rather than an enforced boundary.',
        'm1b_immutable_snapshot.py:build_frozen_snapshot(initial_quality_by_node=...)')

    # B4: a lot whose class is fish but which sits in a manifest while the class is out of scope
    class FakeOutOfScope(object):
        order_id = 200
        temp_class = 1
        initial_quality = 1.0
    ok4, v4, d4 = QC.check_omega10_initialization([FakeOutOfScope()], pc)
    row('B4', 'fish lot with a WRONG initial value is refused (gate soundness)',
        {'temp_class': 1, 'initial_quality': 1.0, 'contract initial_value': 9.0},
        'OMEGA10_INITIAL_VALUE_MISMATCH',
        {'verdict': v4, 'ok': ok4},
        'OK' if v4 == QC.OMEGA10_INITIAL_VALUE_MISMATCH else 'FALSE_PASS',
        'control: the gate is correct when exercised; this isolates the coverage hole to the '
        'provider/orchestrator layer rather than the comparison itself.',
        'm1b_quality_certificate.py:check_omega10_initialization')


def task_C():
    print('\n===== TASK C: certification-chain completeness audit =====')
    import subprocess
    src = (EXT / 'scripts' / 'm1b_adapter' / 'm1b_thermal_gate.py').read_text(encoding='utf-8')
    cert = (EXT / 'scripts' / 'm1b_adapter' / 'm1b_q2_certifier.py').read_text(encoding='utf-8')
    row('C1', 'can NOT_VERIFIED_CONTINUOUSLY still yield THERMAL_HARD_OK=True?',
        {'thermal_gate': 'sets continuous_time_hard_constraint = NOT_VERIFIED_CONTINUOUSLY in the '
                         'envelope but returns THERMAL_HARD_OK when bounds are met'},
        'a consumer must not be able to read THERMAL_HARD_OK as a CONTINUOUS-time guarantee',
        {'thermal_gate_returns_hard_ok': "return env, THERMAL_HARD_OK" in src,
         'certifier_sets_field_true': "receipt['THERMAL_HARD_OK'] = bool(thermal_ok)" in cert,
         'composite_uses_field': "t_ok = receipt['THERMAL_HARD_OK'] is True" in cert,
         'continuous_label_only_in_detail': "CONTINUOUS_NOT_VERIFIED" in src},
        'PROMOTION_PATH_EXISTS',
        'THERMAL_HARD_OK means "the SAMPLED envelope is inside the hard bounds". The '
        'continuous-time caveat lives only inside the per-vehicle detail; the top-level boolean '
        'and the composite do not carry it. A downstream reader that inspects only '
        'THERMAL_HARD_OK/composite_pass cannot distinguish sampled from continuous.',
        'm1b_thermal_gate.py:return env, THERMAL_HARD_OK; m1b_q2_certifier.py:composite_pass')

    row('C2', 'does the composite bind a single shared state identity with the installer?',
        {'M1-a': 'composer/install boundary lives in d3_geo_adapter.run_repair_transaction '
                 '(no certifier -> refusal)',
         'M1-b': 'm1b receipt carries state_identity_before/after'},
        'certification and installation should share ONE state identity',
        {'m1b_receipt_has_state_identity': 'state_identity_before' in cert,
         'any_production_caller_of_m1b': False,
         'm1b_wired_into_install_path': False},
        'NOT_ESTABLISHED',
        'no production path calls the M1-b certifier, and the M1-b receipt is not consumed by '
        'run_repair_transaction; the two identity notions are not yet joined.',
        'm1b_q2_certifier.py (receipt only); d3_geo_adapter.py:708 run_repair_transaction')

    row('C3', 'is the C0 energy check part of the M1-b composite?',
        {'M1-b gates': ['contract_identity', 'quality_class_scope', 'snapshot', 'aliasing',
                        'omega11_return_feasibility', 'omega10_initialization',
                        'thermal_replay', 'quality_lb', 'state_unchanged']},
        'the composite should state explicitly which checks it does and does not include',
        {'certify_plan_in_composite': False,
         'compose_certification_in_composite': False,
         'm1a_s1_s17_still_separate': True},
        'NOT_ESTABLISHED',
        'the M1-b composite covers contract/quality/thermal/return/state but NOT the M1-a '
        'legality+repair+partition+route set and NOT certify_plan. The two suites are '
        'complementary; composite_pass=True must NOT be read as covering the M1-a checks.',
        'm1b_q2_certifier.py:certify() gate list')

    # C4: what composite_pass=True actually proved, measured
    SVC = {7: 0.05}
    veh = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=9.0))]
    pl = {0: plan(0, 0.0, [7, 0])}
    c = M1bQ2Certifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B)
    ok, rc = c.certify(vehicles=veh, served_mask=np.zeros(40, dtype=bool), candidate_plans=pl,
                       clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0,
                       distance_fn=lambda a, b: 10.0, temp_class_of=lambda n: 1,
                       quantity_of=lambda n: 1.0, service_time_of=lambda n: SVC.get(int(n), 0.5),
                       initial_quality_by_node={7: 9.0})
    row('C4', 'what composite_pass=True actually establishes (measured)',
        {'one vehicle, one on-board fish lot, candidate plan [7, 0]'},
        'a POSITIVE engineering case, so the receipt can be read literally',
        {'composite': ok, 'gates': sorted(rc['gates'].keys()),
         'QUALITY_LB_OK': rc['QUALITY_LB_OK'], 'THERMAL_HARD_OK': rc['THERMAL_HARD_OK'],
         'disclaimer_present': 'engineering acceptance only' in (rc.get('disclaimer') or '')},
        'OK' if ok else 'UNEXPECTED',
        'composite_pass=True means: the contract is v2, the snapshot is isolated and unchanged, '
        'the sampled return is inside T_end, the checked lots match the contract initial value, '
        'the SAMPLED temperature envelope is inside the hard bounds, and the analytic quality '
        'lower bound is above threshold. It does NOT mean the M1-a legality/repair/partition/route '
        'checks passed, does NOT include certify_plan, does NOT cover unrevealed or '
        'candidate-only orders, and does NOT establish continuous-time temperature feasibility.',
        'm1b_q2_certifier.py:certify() composite')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    print('M1-b/Q2-R targeted negative-control audit')
    task_A()
    task_B()
    task_C()
    n_false_pass = sum(1 for r in ROWS if r['CLASS'] == 'FALSE_PASS')
    n_ne = sum(1 for r in ROWS if r['CLASS'] == 'NOT_ESTABLISHED')
    print('\n== SUMMARY: %d cases, %d FALSE_PASS, %d NOT_ESTABLISHED =='
          % (len(ROWS), n_false_pass, n_ne))
    for r in ROWS:
        if r['CLASS'] in ('FALSE_PASS', 'NOT_ESTABLISHED', 'PROMOTION_PATH_EXISTS'):
            print('   %-7s %s' % (r['cid'], r['CLASS']))
    payload = {'schema': 'm1b_q2r_negative_control_audit_v1',
               'n_cases': len(ROWS), 'n_false_pass': n_false_pass, 'n_not_established': n_ne,
               'n_false_reject': sum(1 for r in ROWS if r['CLASS'] == 'FALSE_REJECT'),
               'synthetic_only': True, 'results': ROWS}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                  encoding='utf-8')
        print('written: %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
