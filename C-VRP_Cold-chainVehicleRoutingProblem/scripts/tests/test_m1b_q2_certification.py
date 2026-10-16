#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2 regression suite: T-Q2-1 .. T-Q2-11.

The specification was frozen BEFORE this file was written, in
`docs/当前规划/决策与审计/M1b_Q2_测试规格冻结_2026-10-16.md`.  Every expected verdict here is
traceable to that document.

T-1..T-5 from the task text are implemented as T-Q2-1..T-Q2-5, because the repository already
uses the bare `T-G<n>` namespace and the spec forbids overwriting existing numbers.

PURE NumPy / stdlib.  No GPU, no model, no checkpoint, no rollout, no installation.
"""
from __future__ import annotations

import argparse
import copy
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
from coldchain_contract import ColdChainContractV2                       # noqa: E402
import coldchain_state as CS                                             # noqa: E402
from coldchain_state import (create_vehicle_state, dispatch_vehicle,      # noqa: E402
                             transition_segment, CargoLotState)
from action_contract import VehiclePlan                                   # noqa: E402

import m1b_contract_identity as CI                                       # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
import m1b_quality_certificate as QC                                     # noqa: E402
import m1b_return_feasibility as RF                                      # noqa: E402
import m1b_thermal_gate as TG                                            # noqa: E402
from m1b_q2_certifier import (M1bQ2Certifier, ENGINEERING_CERTIFIED,      # noqa: E402
                              ENGINEERING_REFUSED)

ROWS = []
CONTRACT = ColdChainContractV2()
T_END = 24.0
SYNTHETIC_B = 12345.678          # SYNTHETIC ONLY -- interface/budget-refusal semantics.
                                 # MUST NOT be registered as a formal L2 budget.


def chk(cid, name, verdict, detail=None):
    ROWS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-9s] %-56s %s' % (cid, name[:56], verdict))


def ck(cid, name, cond, detail=None):
    chk(cid, name, 'PASS' if cond else 'FAIL', detail)
    return bool(cond)


# ==============================================================================================
# Fixtures (synthetic; no real data, no model)
# ==============================================================================================
class V(object):
    """A synthetic VehicleState with the fields certification reads."""

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
    """A dispatched cold-chain state, optionally carrying lots already on board.

    `pickup=-1.0` models a lot acquired strictly BEFORE the current decision point, which is the
    only way an on-board lot can exist at a decision point.  The thermal replay refuses a
    pre-boarded lot whose recorded pickup is at or after the replay start, because such a state
    is internally inconsistent.
    """
    st = dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT)
    if n_lots > 0:
        lots = tuple(CargoLotState(order_id=100 + i, quantity=quantity, temp_class=temp_class,
                                   pickup_finish_time=pickup, initial_quality=initial_quality,
                                   quality_remaining=initial_quality)
                     for i in range(n_lots))
        # `zone_load` MUST equal the manifest quantities per zone: the engine itself enforces
        # this in _validate_state, so a fixture that adds manifest lots without updating zone_load
        # is malformed and is correctly refused by transition_segment.
        zl = list(st.zone_load)
        zl[int(temp_class)] = float(quantity) * n_lots
        st = CS.replace(st, cargo_manifest=lots, zone_load=tuple(zl))
    return st


def plan(vid, anchor_time, suffix, anchor_load=0.0, anchor_node=0):
    return VehiclePlan(vid, anchor_node, anchor_time, anchor_load, list(suffix))


TRAVEL = {}          # (a,b) -> hours
DIST = {}            # (a,b) -> km
SVC = {}             # node -> service hours
TCLASS = {}          # node -> temp class
QTY = {}             # node -> demand
IQ0 = {}             # node -> public initial quality


def reset_public():
    TRAVEL.clear(); DIST.clear(); SVC.clear(); TCLASS.clear(); QTY.clear(); IQ0.clear()


def tfn(a, b):
    return float(TRAVEL.get((int(a), int(b)), 1.0))


def dfn(a, b):
    return float(DIST.get((int(a), int(b)), 10.0))


def make_certifier(**kw):
    kw.setdefault('contract', CONTRACT)
    kw.setdefault('T_end', T_END)
    kw.setdefault('budget', SYNTHETIC_B)
    return M1bQ2Certifier(**kw)


def run_case(vehicles, plans, clock=0.0, **kw):
    return make_certifier(**kw).certify(
        vehicles=vehicles, served_mask=np.zeros(60, dtype=bool), candidate_plans=plans,
        clock=clock, inst_idx=0, travel_fn=tfn, distance_fn=dfn,
        temp_class_of=lambda n: TCLASS.get(int(n), 1),
        quantity_of=lambda n: QTY.get(int(n), 1.0),
        service_time_of=lambda n: SVC.get(int(n), 0.5),
        initial_quality_by_node=IQ0)


def verdict_of(rc, gate):
    return (rc.get('gates', {}).get(gate) or {}).get('verdict')


# ==============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    reset_public()

    # ---------------------------------------------------------------- T-Q2-6 contract identity
    ident, v, d = CI.validate_contract_for_quality_certificate(CONTRACT)
    ck('T-Q2-6a', 'v2 contract resolves with class/schema/is_v2/hash',
       v == CI.CONTRACT_IDENTITY_OK and ident['quality_config_class'] == 'QualityConfigV2'
       and ident['contract_schema_version'] == 'coldchain-contract-v2'
       and ident['is_v2'] is True and isinstance(ident['contract_hash'], str), {'v': v, 'id': ident})

    from coldchain_contract import default_pilot_contract
    _, v1v, v1d = CI.validate_contract_for_quality_certificate(default_pilot_contract())
    ck('T-Q2-6b', 'v1 contract REFUSED (no silent v1 fallback)',
       v1v == CI.CONTRACT_NOT_V2, {'verdict': v1v})

    _, hv, hd = CI.validate_contract_for_quality_certificate(CONTRACT,
                                                             registered_contract_hash='deadbeef')
    ck('T-Q2-6c', 'contract_hash mismatch REFUSED', hv == CI.CONTRACT_HASH_MISMATCH,
       {'verdict': hv})

    class NoHash(object):
        quality = CONTRACT.quality
        schema_version = 'coldchain-contract-v2'
    _, nv, nd = CI.validate_contract_for_quality_certificate(NoHash())
    ck('T-Q2-6d', 'missing contract_hash REFUSED (not defaulted)',
       nv == CI.CONTRACT_HASH_MISSING, {'verdict': nv})

    # AST: getattr(contract,'version',None) must not appear in the m1b adapter
    import ast as _ast
    offenders = []
    for p in sorted((EXT / 'scripts' / 'm1b_adapter').glob('*.py')):
        src = p.read_text(encoding='utf-8')
        tree = _ast.parse(src)
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == 'getattr':
                if len(n.args) >= 2 and isinstance(n.args[1], _ast.Constant) and \
                        n.args[1].value == 'version':
                    offenders.append('%s:%s' % (p.name, getattr(n, 'lineno', '?')))
            if isinstance(n, _ast.Attribute) and n.attr == 'version':
                offenders.append('%s:%s(attr)' % (p.name, getattr(n, 'lineno', '?')))
    ck('T-Q2-6e', "getattr(contract,'version') / .version ABSENT from the m1b adapter (AST)",
       not offenders, {'offenders': offenders})

    # ---------------------------------------------------------------- T-Q2-7 snapshots
    cc = fresh_cc(n_lots=1, initial_quality=9.0)
    veh = [V(0, 'ready', node=5, ready=1.0, cc=cc)]
    pl = {0: plan(0, 1.0, [7, 0], anchor_load=2.5)}
    snap, sv, sd = IS.build_frozen_snapshot(veh, np.zeros(60, dtype=bool), pl)
    ck('T-Q2-7-setup', 'snapshot builds', sv == IS.SNAPSHOT_OK, {'v': sv})
    if sv == IS.SNAPSHOT_OK:
        before = IS.snapshot_state_identity(snap)
        veh[0].current_node = 9                 # mutate the LIVE object
        veh[0].mutable_suffix.append(11)
        pl[0].suffix.append(13)
        after = IS.snapshot_state_identity(snap)
        ck('T-Q2-7a/7b', 'live mutation does NOT change the snapshot',
           before == after, {'before': before[:16], 'after': after[:16]})
        ck('T-Q2-7d', 'alias detection: snapshot shares no mutable object', 
           IS.detect_aliasing(snap, veh, pl)[0] is True)

    a1 = IS.plan_identity_v2({0: plan(0, 1.0, [7], anchor_load=1.0)})
    a2 = IS.plan_identity_v2({0: plan(0, 1.0, [7], anchor_load=2.0)})
    a3 = IS.plan_identity_v2({0: plan(0, 1.0, [7], anchor_load=1.0)})
    ck('T-Q2-7c', 'anchor_load IS part of plan identity (G4)',
       a1 != a2 and a1 == a3, {'l1': a1[:16], 'l2': a2[:16]})

    veh2 = [V(0, 'ready', node=5, ready=1.0, cc=fresh_cc())]
    pl2 = {0: plan(0, 1.0, [7, 0])}
    snap2, _, _ = IS.build_frozen_snapshot(veh2, np.zeros(60, dtype=bool), pl2)
    veh2[0].current_node = 3
    unch, uv, ud = IS.verify_state_unchanged(snap2, veh2, np.zeros(60, dtype=bool), pl2)
    ck('T-Q2-7e', 'state changed during certification -> REFUSED',
       unch is False and uv == IS.STATE_CHANGED_DURING_CERTIFICATION, {'v': uv})

    # ---------------------------------------------------------------- T-Q2-3 horizon formula
    H, hv2, _hd = QC.horizon_hours(24.0, 0.0, 1.0)
    ck('T-Q2-3a', 'H_max = (24.0 + 1e-6 - 0) * 1.0', abs(H - 24.000001) < 1e-12, {'H': H})
    H2, _, _ = QC.horizon_hours(24.0, 0.0, 2.0)
    ck('T-Q2-3b', 'unit factor 2 -> H_max = 48.000002 (recomputed, NOT halved)',
       abs(H2 - 48.000002) < 1e-12, {'H': H2})
    Ht, _, _ = QC.horizon_hours(24.0, 2.0, 1.0)
    ck('T-Q2-3f', 'non-zero t_min reduces the span by exactly 2.0',
       abs(Ht - 22.000001) < 1e-12, {'H': Ht})
    Hu, uv3, _ = QC.horizon_hours(24.0, 0.0, 0.0)
    ck('T-Q2-3g', 'hours_per_time_unit <= 0 REFUSED',
       Hu is None and uv3 == QC.QUALITY_TIME_UNIT_INVALID, {'v': uv3})

    pc, pcv, pcd = CI.check_quality_classes_are_certifiable(CONTRACT, epsilon_init=QC.EPSILON_INIT)
    ck('T-Q2-3-setup', 'quality classes resolve (0 out of scope, 1/2 in scope)',
       pcv == CI.CONTRACT_IDENTITY_OK and pc[0]['in_scope'] is False
       and pc[1]['in_scope'] and pc[2]['in_scope'], {'v': pcv})
    for Hh, want, sal in ((200.0, 5.234000, True), (230.0, 4.669100, False)):
        res, qv, _ = QC.certify_quality_lb(pc, T_end=Hh - QC.EPSILON_RETURN, t_min=0.0,
                                           hours_per_time_unit=1.0)
        got = res[1]['q_lb']
        ck('T-Q2-3c[%g]' % Hh, 'zone1 Q_LB=%.6f salable=%s' % (want, sal),
           abs(got - want) < 1e-6 and res[1]['salable'] is sal and (qv == QC.QUALITY_LB_OK) == sal,
           {'q_lb': got, 'salable': res[1]['salable'], 'verdict': qv})
    for Hh, want, sal in ((14000.0, 5.108000, True), (15000.0, 4.830000, False)):
        res, qv, _ = QC.certify_quality_lb(pc, T_end=Hh - QC.EPSILON_RETURN, t_min=0.0,
                                           hours_per_time_unit=1.0)
        got = res[2]['q_lb']
        ck('T-Q2-3d[%g]' % Hh, 'zone2 Q_LB=%.6f salable=%s' % (want, sal),
           abs(got - want) < 1e-6 and res[2]['salable'] is sal, {'q_lb': got, 'salable': res[2]['salable']})
    res, _, _ = QC.certify_quality_lb(pc, T_end=24.0 - QC.EPSILON_RETURN, t_min=0.0,
                                      hours_per_time_unit=1.0)
    ck('T-Q2-3e', 'H_crit documented (zone1 212.426978279, zone2 14388.489212230)',
       abs(res[1]['H_crit'] - 212.426978279) < 1e-6
       and abs(res[2]['H_crit'] - 14388.489212230) < 1e-5,
       {'z1': res[1]['H_crit'], 'z2': res[2]['H_crit']})
    ck('T-Q2-3-tomato', 'tomato class 0 is OUT OF SCOPE with salable=None (never unsalable)',
       res[0]['in_scope'] is False and res[0]['salable'] is None
       and res[0]['claim'] == 'NOT_CERTIFIED_BY_THIS_BOUND', {'c0': res[0]})

    # ---------------------------------------------------------------- T-Q2-1 Omega10
    good = CargoLotState(order_id=1, quantity=1.0, temp_class=1, pickup_finish_time=0.0,
                         initial_quality=9.0, quality_remaining=9.0)
    # NOTE ON THE BOUNDARY CASE: 9.0 - 1e-9 is not exactly representable, so the computed
    # absolute deviation at the exact epsilon is ~1.00000000058e-9 in float64 -- i.e. the exact
    # boundary falls on the far side of the comparison.  The boundary is therefore tested with a
    # margin strictly inside it (0.9e-9 accepted, 2e-9 refused); asserting the EXACT epsilon would
    # be testing float representation, not the gate.
    edge_in = CS.replace(good, initial_quality=9.0 - 0.9e-9, quality_remaining=9.0)
    edge_out = CS.replace(good, initial_quality=9.0 - 2e-9, quality_remaining=9.0)
    bad_q = CS.replace(good, initial_quality=1.0, quality_remaining=1.0)
    ok, vv, dd = QC.check_omega10_initialization([good], pc)
    ck('T-Q2-1a', 'exact initial value -> OMEGA10_OK', ok and vv == QC.OMEGA10_OK, {'v': vv})
    ok, vv, dd = QC.check_omega10_initialization([edge_in], pc)
    ck('T-Q2-1b', 'deviation 0.9e-9, strictly inside eps_init -> accepted',
       ok and vv == QC.OMEGA10_OK, {'v': vv, 'dev': 0.9e-9,
                                    'note': 'exact-epsilon asserted as a documentation fact, not '
                                            'as an assertion, because 9.0-1e-9 is not exactly '
                                            'representable'})
    ok, vv, dd = QC.check_omega10_initialization([edge_out], pc)
    ck('T-Q2-1c', 'deviation 2e-9 -> OMEGA10_INITIAL_VALUE_MISMATCH',
       (not ok) and vv == QC.OMEGA10_INITIAL_VALUE_MISMATCH, {'v': vv})
    ok, vv, dd = QC.check_omega10_initialization([bad_q], pc)
    ck('T-Q2-1d', 'the Q1 logical counterexample (iq=1.0) is REFUSED',
       (not ok) and vv == QC.OMEGA10_INITIAL_VALUE_MISMATCH, {'v': vv})
    ok, vv, dd = QC.check_omega10_initialization([], pc)
    ck('T-Q2-1-empty', 'no in-scope lots -> OMEGA10_NO_RELEVANT_LOTS (not a positive claim)',
       ok and vv == QC.OMEGA10_NO_RELEVANT_LOTS, {'v': vv})
    ok, vv, dd = QC.check_omega10_initialization([good], pc)
    ck('T-Q2-1f', 'conservative Q0_safe = 9.0 - 1e-9 is used', 
       abs(dd['q0_safe_used'].get(1, 0) - (9.0 - 1e-9)) < 1e-15, {'q0_safe': dd['q0_safe_used']})

    # T-Q2-1e: the gate must NOT read unrevealed orders -- enforced structurally by lot_provider
    reset_public()
    IQ0[3] = 9.0
    IQ0[4] = 1.0                      # an unrevealed order with a WRONG value
    veh3 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=9.0))]
    pl3 = {0: plan(0, 0.0, [3, 0])}
    okc, rc = run_case(veh3, pl3)
    w10 = verdict_of(rc, 'omega10_initialization')
    decl = set((rc['gates']['omega10_initialization']['detail'] or {}).get('q0_safe_used', {}))
    ck('T-Q2-1e', 'unrevealed order 4 (bad iq) is NOT scanned; gate reads manifest lots only',
       w10 in (QC.OMEGA10_OK, QC.OMEGA10_NO_RELEVANT_LOTS), {'verdict': w10})

    # ---------------------------------------------------------------- T-Q2-2 Omega11
    reset_public()
    veh4 = [V(0, 'ready', node=5, ready=1.0, cc=fresh_cc())]
    pl4 = {0: plan(0, 1.0, [7, 0])}
    sn4, _, _ = IS.build_frozen_snapshot(veh4, np.zeros(60, dtype=bool), pl4)
    ok11, v11, d11 = RF.certify_return_feasibility(sn4, T_end=T_END, clock=0.0, travel_fn=tfn)
    ck('T-Q2-2a', 'in-time return -> OMEGA11_OK', ok11 and v11 == RF.OMEGA11_OK,
       {'v': v11, 'ret': d11['per_vehicle'][0].get('return_arrival')})

    veh5 = [V(0, 'ready', node=5, ready=23.0, cc=fresh_cc())]
    sn5, _, _ = IS.build_frozen_snapshot(veh5, np.zeros(60, dtype=bool), pl4)
    ok11b, v11b, d11b = RF.certify_return_feasibility(sn5, T_end=T_END, clock=23.0, travel_fn=tfn)
    ck('T-Q2-2b', 'late return -> OMEGA11_RETURN_DEADLINE_EXCEEDED',
       (not ok11b) and v11b == RF.OMEGA11_RETURN_DEADLINE_EXCEEDED, {'v': v11b})

    veh6 = [V(0, 'returning', node=5, ready=1.0, cc=fresh_cc(), return_finish=10.0)]
    sn6, _, _ = IS.build_frozen_snapshot(veh6, np.zeros(60, dtype=bool), {})
    ok11c, v11c, d11c = RF.certify_return_feasibility(sn6, T_end=T_END, clock=2.0, travel_fn=tfn)
    ck('T-Q2-2c', 'in-flight leg uses the RECORDED return_finish (not recomputed)',
       ok11c and d11c['per_vehicle'][0]['return_arrival'] == 10.0, {'v': v11c})

    veh7 = [V(0, 'ready', node=5, ready=1.0, cc=fresh_cc(), suffix=[])]
    sn7, _, _ = IS.build_frozen_snapshot(veh7, np.zeros(60, dtype=bool), {0: plan(0, 1.0, [])})
    ok11d, v11d, d11d = RF.certify_return_feasibility(sn7, T_end=T_END, clock=1.0, travel_fn=tfn)
    ck('T-Q2-2d', 'WAIT (empty suffix away from depot) is judged as "must still return"',
       ok11d and d11d['per_vehicle'][0]['pending'] is True
       and d11d['per_vehicle'][0]['return_arrival'] > 1.0, {'v': v11d})

    veh8 = [V(0, 'ready', node=5, ready=31.0, cc=fresh_cc())]
    sn8, _, _ = IS.build_frozen_snapshot(veh8, np.zeros(60, dtype=bool), {0: plan(0, 31.0, [0])})
    ok11e, v11e, _ = RF.certify_return_feasibility(sn8, T_end=T_END, clock=31.0, travel_fn=tfn)
    ck('T-Q2-2f', 'clock beyond horizon -> OMEGA11_CLOCK_BEYOND_HORIZON (no fabricated cert)',
       (not ok11e) and v11e == RF.OMEGA11_CLOCK_BEYOND_HORIZON, {'v': v11e})

    # ---------------------------------------------------------------- T-Q2-5 thermal gate
    reset_public()
    SVC[7] = 0.1
    veh9 = [V(0, 'ready', node=0, ready=0.0, cc=dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT))]
    pl9 = {0: plan(0, 0.0, [7, 0])}
    sn9, _, _ = IS.build_frozen_snapshot(veh9, np.zeros(60, dtype=bool), pl9, initial_quality_by_node=IQ0)
    env, tv, td = TG.replay_thermal_envelope(
        sn9['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0, travel_fn=tfn,
        distance_fn=dfn, temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: SVC.get(int(n), 0.5))
    ck('T-Q2-5-setup', 'thermal replay runs and samples the door event',
       tv == TG.THERMAL_HARD_OK and env['n_door_events'] == 1, {'v': tv, 'doors': env.get('n_door_events')})
    ck('T-Q2-5b', 'post-door sample present (interior peak is sampled, not only endpoints)',
       any(s['tag'].startswith('after_door') for s in env['samples']), {'tags': [s['tag'] for s in env['samples']]})
    ck('T-Q2-5d', 'continuous-time constraint reported as NOT verified (honest gap)',
       env['continuous_time_hard_constraint'] == TG.CONTINUOUS_NOT_VERIFIED,
       {'label': env['continuous_time_hard_constraint']})

    veh10 = [V(0, 'ready', node=0, ready=0.0, cc=None)]
    sn10, _, _ = IS.build_frozen_snapshot(veh10, np.zeros(60, dtype=bool), pl9)
    _, tv2, _ = TG.replay_thermal_envelope(
        sn10['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0, travel_fn=tfn,
        distance_fn=dfn, temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: 0.5)
    ck('T-Q2-5c', 'missing cold-chain state -> THERMAL_REPLAY_STATE_MISSING (fail-closed)',
       tv2 == TG.THERMAL_REPLAY_STATE_MISSING, {'v': tv2})

    cc_bad = CS.replace(fresh_cc(), thermal_violation_count=3, thermal_violation_duration_h=0.5)
    veh11 = [V(0, 'ready', node=0, ready=0.0, cc=cc_bad)]
    sn11, _, _ = IS.build_frozen_snapshot(veh11, np.zeros(60, dtype=bool), pl9)
    _, tv3, _ = TG.replay_thermal_envelope(
        sn11['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0, travel_fn=tfn,
        distance_fn=dfn, temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: 0.5)
    ck('T-Q2-5f', 'pre-existing accumulated violations -> REFUSED (not laundered by replay)',
       tv3 == TG.THERMAL_PRIOR_VIOLATIONS_PRESENT, {'v': tv3})

    # T-Q2-5a: consecutive doors with no cooling gap -- catches cumulative warming
    reset_public()
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        SVC[n] = 0.01
    veh12 = [V(0, 'ready', node=0, ready=0.0,
               cc=dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT))]
    pl12 = {0: plan(0, 0.0, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0])}
    sn12, _, _ = IS.build_frozen_snapshot(veh12, np.zeros(60, dtype=bool), pl12)
    env12, tv12, td12 = TG.replay_thermal_envelope(
        sn12['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0, travel_fn=tfn,
        distance_fn=dfn, temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: SVC.get(int(n), 0.5))
    zmax = env12['per_zone_max'][1]
    ck('T-Q2-5a', 'consecutive doors accumulate warming and are observed (zone1 max > target)',
       tv12 == TG.THERMAL_HARD_OK and zmax > 4.0,
       {'zone1_max': zmax, 'n_doors': env12['n_door_events'],
        'cumulative_over_target': zmax - 4.0})

    # T-Q2-5e: time order violation
    veh13 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    sn13, _, _ = IS.build_frozen_snapshot(veh13, np.zeros(60, dtype=bool), pl9)
    TRAVEL[(0, 7)] = -5.0
    _, tv4, _ = TG.replay_thermal_envelope(
        sn13['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0, travel_fn=tfn,
        distance_fn=dfn, temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: 0.5)
    TRAVEL.clear()
    ck('T-Q2-5e', 'negative travel (time-order violation) -> fail-closed',
       tv4 in (TG.THERMAL_REPLAY_TIME_ORDER_INVALID, TG.THERMAL_REPLAY_INCOMPLETE), {'v': tv4})

    # ---------------------------------------------------------------- T-Q2-4 non-substitution
    reset_public()
    SVC[7] = 0.05
    IQ0[7] = 9.0
    TCLASS[7] = 1
    QTY[7] = 1.0
    vehA = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=9.0))]
    plA = {0: plan(0, 0.0, [7, 0])}
    okA, rcA = run_case(vehA, plA)
    ck('T-Q2-4c', 'both gates pass -> engineering certified',
       okA and rcA['final'] == ENGINEERING_CERTIFIED
       and rcA['QUALITY_LB_OK'] is True and rcA['THERMAL_HARD_OK'] is True,
       {'final': rcA['final'], 'Q': rcA['QUALITY_LB_OK'], 'T': rcA['THERMAL_HARD_OK']})

    # 4a: quality OK but thermal NOT OK -- tighten the hard bounds so they cannot be satisfied
    class Tight(ColdChainContractV2):
        pass
    tight = ColdChainContractV2()
    import dataclasses as _dc
    new_th = _dc.replace(tight.thermal, hard_temperature_bounds_c=((-50.0, -40.0),) * 3)
    tight = _dc.replace(tight, thermal=new_th)
    vehB = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=9.0))]
    okB, rcB = make_certifier(contract=tight).certify(
        vehicles=vehB, served_mask=np.zeros(60, dtype=bool), candidate_plans=plA, clock=0.0,
        inst_idx=0, travel_fn=tfn, distance_fn=dfn, temp_class_of=lambda n: TCLASS.get(int(n), 1),
        quantity_of=lambda n: 1.0, service_time_of=lambda n: SVC.get(int(n), 0.5),
        initial_quality_by_node=IQ0)
    ck('T-Q2-4a', 'quality TRUE + thermal FALSE -> composite REFUSED (gates independent)',
       (not okB) and rcB['QUALITY_LB_OK'] is True and rcB['THERMAL_HARD_OK'] is False
       and rcB['final'] == ENGINEERING_REFUSED,
       {'Q': rcB['QUALITY_LB_OK'], 'T': rcB['THERMAL_HARD_OK'], 'final': rcB['final']})

    # 4b: thermal OK but quality NOT OK -- a manifest lot whose iq is wrong AND whose class the
    # certificate can still evaluate.  The Omega10 gate fires first, which is the stronger result,
    # so this case asserts that the composite is refused and that QUALITY_LB_OK is False.
    vehC = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, initial_quality=1.0))]
    okC, rcC = run_case(vehC, plA)
    ck('T-Q2-4b', 'bad initial value -> refused before quality stage; QUALITY_LB_OK not True',
       (not okC) and rcC['QUALITY_LB_OK'] is not True
       and verdict_of(rcC, 'quality_impl_call') == 'NOT_CALLED_OMEGA10_REFUSED',
       {'Q': rcC['QUALITY_LB_OK'], 'omega10': verdict_of(rcC, 'omega10_initialization')})

    # 4d: unknown counts as failure -- supply no vehicles at all
    okD, rcD = run_case([], {})
    ck('T-Q2-4d', 'no vehicles / no evidence -> refused; unknown is not a pass',
       (not okD) and rcD['THERMAL_HARD_OK'] is False, {'T': rcD['THERMAL_HARD_OK']})

    # ---------------------------------------------------------------- T-Q2-8 future isolation
    from m1a_certification_adapter import MinimalVisibilityView
    class _E(object):
        reveal_time = np.array([[0.0, 1.0, 2.0]])
        num_nodes = 3
    view = MinimalVisibilityView(clock=1.0, inst_idx=0, served_mask=np.zeros(3, dtype=bool),
                                 visible_mask=[True, True, False], visible_order_ids=[1],
                                 visible_order_attrs={}, vehicle_cc_state=None,
                                 contract_version='x', contract_hash='y')
    leaked = []
    for f in ('reveal_time', 'env'):
        try:
            view.get(f); leaked.append(f)
        except PermissionError:
            pass
    ck('T-Q2-8a', 'reveal_time / env access raises PermissionError', not leaked, {'leaked': leaked})
    try:
        view.get('something_not_allowed'); bad8 = True
    except PermissionError:
        bad8 = False
    ck('T-Q2-8b', 'non-allowlisted field raises PermissionError', not bad8)

    # ---------------------------------------------------------------- T-Q2-9 fail-closed types
    bad_inputs = [None, float('nan'), float('inf'), 'x', [1]]
    all_refused = True
    for b in bad_inputs:
        try:
            okl, vv, _ = QC.check_omega10_initialization(
                [type('L', (), {'temp_class': 1, 'initial_quality': b, 'order_id': 1})()], pc)
            if okl:
                all_refused = False
        except Exception:                                                # noqa: BLE001
            all_refused = False
    ck('T-Q2-9', 'None/NaN/inf/str/list as initial_quality -> all refused',
       all_refused, {'inputs': [repr(b) for b in bad_inputs]})
    Hn, vn, _ = QC.horizon_hours(float('nan'), 0.0, 1.0)
    ck('T-Q2-9b', 'NaN T_end -> QUALITY_NUMERIC_INVALID',
       Hn is None and vn == QC.QUALITY_NUMERIC_INVALID, {'v': vn})

    # ---------------------------------------------------------------- T-Q2-10 synthetic B
    ck('T-Q2-10', 'synthetic B is never registered as a formal budget',
       rcA['budget_is_synthetic'] is True,
       {'budget': SYNTHETIC_B, 'note': 'interface/budget-refusal semantics only'})

    # ---------------------------------------------------------------- summary
    fails = [r for r in ROWS if r['verdict'] != 'PASS']
    status = 'T_Q2_ALL_PASS' if not fails else 'T_Q2_HAS_FAILURES'
    print('\n== %s ==  (%d checks, %d not PASS)' % (status, len(ROWS), len(fails)))
    for f in fails:
        print('   NOT PASS: %s %s -> %s' % (f['cid'], f['name'][:50], f['verdict']))
    print('\nDISCLAIMER: engineering acceptance only. NOT production installability, NOT '
          'scientific evidence, NOT L2. FULL_INSTALLABILITY_ESTABLISHED is NOT established.')
    payload = {'schema': 'm1b_q2_regression_v1', 'status': status, 'n_checks': len(ROWS),
               'n_not_pass': len(fails),
               'synthetic_budget': SYNTHETIC_B, 'budget_is_synthetic': True,
               'does_not_establish': ['FULL_INSTALLABILITY_ESTABLISHED', 'L2_PASS', 'A3_PASS',
                                      'stable_online_gain'],
               'results': ROWS}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
        print('written: %s' % args.out)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
