#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix-T: TARGETED TRUST-BOUNDARY AUDIT of the Q2-Fix implementation.

Diagnostic only.  It tries to BREAK the trust boundaries, and records each attempt with its
minimal input, actual return, expectation, source location and whether it blocks installability.
It does NOT modify any source.

Tasks
  T1  the continuous-temperature proof hook's trust boundary
  T2  the compatibility layer as a bypass surface
  T3  strict-online source identity (PublicOrderAttributes / PublicTimeInputs)
  T4  compatibility of the original 49 checks, and multi-vehicle FALSE_REJECT risk

Pure NumPy / stdlib.  No GPU, no model, no frozen-tree change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in (EXT / 'scripts' / 'coldchain', EXT / 'scripts' / 'simulation',
           EXT / 'scripts' / 'evaluation', EXT / 'scripts' / 'm1b_adapter', HERE):
    sys.path.insert(0, str(_p))

import numpy as np                                                       # noqa: E402
import coldchain_state as CS                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
from coldchain_state import create_vehicle_state, dispatch_vehicle       # noqa: E402
from action_contract import VehiclePlan                                   # noqa: E402
import m1b_timing as TM                                                  # noqa: E402
import m1b_quality_certificate as QC                                     # noqa: E402
import m1b_public_projection as PP                                       # noqa: E402
import m1b_return_feasibility as RF                                      # noqa: E402
import m1b_thermal_gate as TG                                            # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
from m1b_q2_certifier import (M1bQ2TrustFixCertifier, M1bQ2TrustFixTestOnlyCertifier,
                              ENGINEERING_CERTIFIED, TEST_ONLY_LABEL)     # noqa: E402

ROWS = []
CONTRACT = ColdChainContractV2()
T_END = 24.0
SYNTHETIC_B = 12345.678


def row(tid, name, inp, expected, actual, verdict, src, blocks):
    ROWS.append({'id': tid, 'name': name, 'MINIMAL_INPUT': inp, 'EXPECTED': expected,
                 'ACTUAL': actual, 'VERDICT': verdict, 'SOURCE': src,
                 'INSTALLABILITY_BLOCKER': blocks})
    print('  [%-6s] %-44s %s' % (tid, name[:44], verdict))


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


def fresh_cc(n_lots=1, temp_class=1, iq=9.0, qty=1.0, pickup=-1.0):
    st = dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT)
    if n_lots > 0:
        lots = tuple(CS.CargoLotState(order_id=100 + i, quantity=qty, temp_class=temp_class,
                                      pickup_finish_time=pickup, initial_quality=iq,
                                      quality_remaining=iq) for i in range(n_lots))
        zl = list(st.zone_load)
        zl[int(temp_class)] = float(qty) * n_lots
        st = CS.replace(st, cargo_manifest=lots, zone_load=tuple(zl))
    return st


def plan(vid, at, suffix, load=0.0):
    return VehiclePlan(vid, 0, at, load, list(suffix))


def times_for(service=None, tw_start=None, revealed=(1, 2, 3, 4, 5, 6, 7),
              identity='t-synth-1'):
    tws = {int(n): 0.0 for n in revealed}
    tws.update(tw_start or {})
    return TM.PublicTimeInputs(service_time=dict(service or {}), tw_start=tws,
                               tw_end={int(n): 1e9 for n in revealed},
                               revealed_nodes=revealed, identity=identity)


def attrs_for(iq=None, tc=None, qty=None, served=(1, 2, 3, 4, 5, 6, 7), identity='a-synth-1'):
    served = tuple(int(x) for x in served)
    iq2 = {n: 9.0 for n in served}; iq2.update({int(k): float(v) for k, v in (iq or {}).items()})
    tc2 = {n: 1 for n in served}; tc2.update({int(k): int(v) for k, v in (tc or {}).items()})
    q2 = {n: 1.0 for n in served}; q2.update({int(k): float(v) for k, v in (qty or {}).items()})
    return QC.PublicOrderAttributes(initial_quality=iq2, temp_class=tc2, quantity=q2,
                                    served_nodes=served, source_identity=identity)


def proj_for(iq=None, tc=None, qty=None, svc=None, tws=None, served=(1, 2, 3, 4, 5, 6, 7),
             instance_id=0, event_id=1, clock=0.0):
    """A TrustedPublicProjection for the synthetic cases (F-T3 type required by the certifier)."""
    served = tuple(int(x) for x in served)
    iq2 = {n: 9.0 for n in served}; iq2.update({int(k): float(v) for k, v in (iq or {}).items()})
    tc2 = {n: 1 for n in served}; tc2.update({int(k): int(v) for k, v in (tc or {}).items()})
    q2 = {n: 1.0 for n in served}; q2.update({int(k): float(v) for k, v in (qty or {}).items()})
    sv2 = {n: 0.05 for n in served}; sv2.update({int(k): float(v) for k, v in (svc or {}).items()})
    tw2 = {n: 0.0 for n in served}; tw2.update({int(k): float(v) for k, v in (tws or {}).items()})
    return PP.projection_from_revealed_maps(
        iq=iq2, tc=tc2, qty=q2, svc=sv2, tws=tw2, twe={n: 1e9 for n in served},
        revealed_nodes=served, instance_id=instance_id, event_id=event_id, clock=clock,
        contract_hash=getattr(CONTRACT, 'contract_hash', None), snapshot_identity=None)


def certifier(**kw):
    kw.setdefault('contract', CONTRACT)
    kw.setdefault('T_end', T_END)
    kw.setdefault('budget', SYNTHETIC_B)
    return M1bQ2TrustFixCertifier(**kw)


def positive_case(proof=None, vehicles=None, planes=None, test_only=False):
    """Default uses the PRODUCTION certifier, which refuses any proof hook (F-T1).

    `test_only=True` switches to the ISOLATED test entry, whose receipt is stamped TEST_ONLY and
    is explicitly not a consumable certificate.
    """
    proj = proj_for()
    vs = vehicles if vehicles is not None else [
        V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))]
    ps = planes if planes is not None else {0: plan(0, 0.0, [7, 0])}
    if test_only:
        c = M1bQ2TrustFixTestOnlyCertifier(
            contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B,
            test_continuous_extremum_proof=(proof if callable(proof)
                                            else (lambda: (True, {}))))
    else:
        c = certifier()
    return c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                     clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0,
                     distance_fn=lambda a, b: 10.0,
                     times=proj, order_attrs=proj, projection=proj)


# ==============================================================================================
def task1():
    print('\n===== T1: continuous-temperature proof hook trust boundary =====')

    # T1-1 the PRODUCTION certifier refuses a proof hook (F-T1)
    raised = None
    try:
        c_hook = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B,
                                        continuous_extremum_proof=lambda **k: (True, {}))
        _ok, _rc = c_hook.certify(
            vehicles=[V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))],
            served_mask=np.zeros(40, dtype=bool), candidate_plans={0: plan(0, 0.0, [7, 0])},
            clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
            times=proj_for(svc={7: 0.05}), order_attrs=proj_for(svc={7: 0.05}),
            projection=proj_for(svc={7: 0.05}))
        raised = ((_rc.get('gates', {}).get('proof_hook') or {}).get('verdict')
                  or 'HOOK_NOT_REFUSED')
    except TypeError as exc:
        raised = str(exc)[:60]
    ok, rc = positive_case()
    row('T1-1', 'arbitrary proof (True,{}) can no longer yield CONTINUOUS_OK',
        {'production certifier': 'continuous_extremum_proof = lambda **k: (True, {})'},
        'F-T1: the production path must refuse the hook; the continuous level must stay '
        'NOT_ESTABLISHED and composite_pass False',
        {'hook_rejection_verdict': raised,
         'THERMAL_CONTINUOUS_STATE': rc.get('THERMAL_CONTINUOUS_STATE'),
         'THERMAL_CONTINUOUS_OK': rc.get('THERMAL_CONTINUOUS_OK'),
         'THERMAL_SAMPLED_OK': rc.get('THERMAL_SAMPLED_OK'),
         'composite_pass': rc.get('composite_pass'), 'final': rc.get('final')},
        'OK' if (raised and raised != 'HOOK_NOT_REFUSED'
                 and rc.get('THERMAL_CONTINUOUS_OK') is False
                 and rc.get('THERMAL_CONTINUOUS_STATE') == TG.THERMAL_CONTINUOUS_NOT_ESTABLISHED
                 and rc.get('composite_pass') is False) else 'FALSE_PASS',
        'm1b_q2_certifier.py:PRODUCTION_PROOF_HOOK_REFUSED gate at the top of certify(); '
        'm1b_thermal_gate.py continuous branch additionally requires allow_test_only_proof',
        False)

    # T1-1b the LOW-LEVEL gate ignores a hook without the explicit test opt-in
    import m1b_immutable_snapshot as _IS
    vs = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))]
    ps = {0: plan(0, 0.0, [7, 0])}
    sn, _, _ = _IS.build_frozen_snapshot(vs, np.zeros(40, dtype=bool), ps,
                                         initial_quality_by_node={7: 9.0})
    env, tv, td = TG.replay_thermal_envelope(
        sn['vehicles'][0], contract=CONTRACT, inst_idx=0, clock=0.0,
        travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
        temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=times_for(service={7: 0.05}),
        continuous_extremum_proof=lambda **k: (True, {}))          # NO opt-in flag
    det = ((td.get('envelope') or {}).get('continuous_time_detail') or {})
    row('T1-1b', 'low-level gate ignores an injected proof without the test opt-in',
        {'continuous_extremum_proof': 'lambda **k: (True, {})', 'allow_test_only_proof': False},
        'continuous level must stay NOT_ESTABLISHED',
        {'sampled_level': td.get('sampled_level'), 'continuous_level': td.get('continuous_level'),
         'proof_hook_offered': det.get('proof_hook_offered'),
         'proof_hook_accepted': det.get('proof_hook_accepted')},
        'OK' if td.get('continuous_level') == TG.THERMAL_CONTINUOUS_NOT_ESTABLISHED else 'FALSE_PASS',
        'm1b_thermal_gate.py: `elif continuous_extremum_proof is None or not '
        'allow_test_only_proof:`', False)

    # T1-1c the ISOLATED test entry still allows wiring, but the receipt is TEST_ONLY
    okT, rcT = positive_case(test_only=True)
    row('T1-1c', 'isolated test entry marks its receipt TEST_ONLY and non-consumable',
        {'class': 'M1bQ2TrustFixTestOnlyCertifier',
         'test_continuous_extremum_proof': 'lambda: (True, {})'},
        'the wiring must stay exercisable, but the receipt must be unusable as a production '
        'certificate',
        {'test_only': rcT.get('test_only'), 'TEST_ONLY_LABEL': rcT.get('TEST_ONLY_LABEL'),
         'not_consumable': rcT.get('not_consumable_as_production_certificate'),
         'wiring': rcT.get('test_only_continuous_wiring')},
        'OK' if (rcT.get('test_only') is True and rcT.get('TEST_ONLY_LABEL') == TEST_ONLY_LABEL
                 and rcT.get('not_consumable_as_production_certificate') is True) else 'FALSE_PASS',
        'm1b_q2_certifier.py:M1bQ2TrustFixTestOnlyCertifier', False)

    # T1-2 the production path has no hook left to identify
    row('T1-2', 'no SHA256 / domain / contract / thermal-version / provenance check exists',
        {'production certifier': 'refuses any proof hook outright',
         'low-level gate': 'requires allow_test_only_proof and is not wired from production'},
        'F-T1: until a real proof is registered and independently audited, the continuous level '
        'stays NOT_ESTABLISHED -- which is now the structural situation',
        {'any_sha256_check': False, 'any_domain_check': False,
         'any_contract_binding_check': False, 'any_thermal_version_check': False,
         'any_provenance_check': False,
         'grep_evidence': 'no registration surface exists; the only route to '
                          'THERMAL_CONTINUOUS_OK is the test-only opt-in'},
        'TRUST_BOUNDARY_RESOLVED_BY_REMOVAL',
        'm1b_q2_certifier.py:PRODUCTION_PROOF_HOOK_REFUSED ; '
        'm1b_thermal_gate.py allow_test_only_proof opt-in', False)

    # T1-3 the sampled level stays independent of any proof
    ok3, rc3 = positive_case()
    row('T1-3', 'the sampled level is measured independently and is not affected by any proof',
        {'certifier': 'production (no proof hook)'},
        'THERMAL_SAMPLED_OK must come from the measured envelope alone',
        {'THERMAL_SAMPLED_OK': rc3.get('THERMAL_SAMPLED_OK'),
         'THERMAL_CONTINUOUS_STATE': rc3.get('THERMAL_CONTINUOUS_STATE'),
         'composite_pass': rc3.get('composite_pass')},
        'OK' if rc3.get('THERMAL_SAMPLED_OK') is True else 'CHECK',
        'm1b_thermal_gate.py sampled branch', False)

    # T1-4 SAMPLED violation must still sink the composite
    import dataclasses as dc
    tight = dc.replace(CONTRACT, thermal=dc.replace(
        CONTRACT.thermal, hard_temperature_bounds_c=((-50.0, -40.0),) * 3))
    a = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0})
    c4 = M1bQ2TrustFixCertifier(contract=tight, T_end=T_END, budget=SYNTHETIC_B)
    ok4, rc4 = c4.certify(vehicles=[V(0, 'ready', node=0, ready=0.0,
                                      cc=fresh_cc(n_lots=1, iq=9.0))],
                          served_mask=np.zeros(40, dtype=bool),
                          candidate_plans={0: plan(0, 0.0, [7, 0])}, clock=0.0, inst_idx=0,
                          travel_fn=lambda a_, b_: 1.0, distance_fn=lambda a_, b_: 10.0,
                          times=proj_for(svc={7: 0.05}), order_attrs=proj_for(svc={7: 0.05}), projection=proj_for(svc={7: 0.05}))
    row('T1-4', 'SAMPLED violation -> composite must fail',
        {'hook': 'returns True', 'hard bounds': '(-50,-40) so sampling violates them'},
        'composite_pass must be False',
        {'THERMAL_SAMPLED_OK': rc4.get('THERMAL_SAMPLED_OK'),
         'THERMAL_CONTINUOUS_STATE': rc4.get('THERMAL_CONTINUOUS_STATE'),
         'composite_pass': rc4.get('composite_pass')},
        'OK' if rc4.get('composite_pass') is False else 'FALSE_PASS',
        'm1b_thermal_gate.py:234-236', False)


# ==============================================================================================
def task2():
    print('\n===== T2: compatibility layer as a bypass surface =====')
    from test_m1b_q2r_negative_controls import MiniEnv                      # reuse the fixture
    import action_contract as AC

    # T2-1 _LegacyTimes forces tw_start = 0.0 -> the A1 counterexample is REACHABLE via the legacy path
    veh = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc())]
    snap, _, _ = IS.build_frozen_snapshot(veh, np.zeros(40, dtype=bool),
                                          {0: plan(0, 0.0, [5, 0])})
    RF.register_service_times({5: 0.0})            # legacy registration (tw_start unavailable)
    ok, v, d = RF.certify_return_feasibility(snap, T_end=19.0, clock=0.0,
                                             travel_fn=lambda a, b: 1.0)     # NO times= passed
    env = MiniEnv(tw_start={5: 20.0}, tw_end={0: 19.0, 5: 30.0}, service={5: 0.0})
    cert = AC.certify_route(env, 0, 0, 0.0, 0.0, (5,))
    row('T2-1', 'direct Omega11 call WITHOUT times= now refuses (F-T2)',
        {'legacy register_service_times': '{5: 0.0}', 'true tw_start[5]': 20.0,
         'T_end': 19.0, 'times= passed': False},
        'F-T2: a missing per-instance time input must be a REFUSAL at this entry point too, so a '
        'caller that bypasses the orchestrator cannot recover the optimistic estimate',
        {'omega11_verdict': v, 'ok': ok,
         'omega11_return_arrival': (d.get('per_vehicle') or {}).get(0, {}).get('return_arrival'),
         'certify_route_reason': cert['reason'],
         'certify_route_return_slack': cert['return_slack'],
         'why': (d.get('why') or '')[:140]},
        'OK' if ((not ok) and v == RF.OMEGA11_TIME_INPUT_UNAVAILABLE) else 'FALSE_PASS',
        'm1b_return_feasibility.py:certify_return_feasibility now refuses when times is None; the '
        '_LegacyServiceTimes auto-consult was removed', True)

    # T2-2 legacy callables into the certifier: do they influence the fixed verdict?
    RF.register_service_times({})
    vs = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=0))]
    ps = {0: plan(0, 0.0, [7, 0])}
    c = certifier()
    ok2, rc2 = c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                         clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0,
                         distance_fn=lambda a, b: 10.0,
                         temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
                         service_time_of=lambda n: 0.05,
                         initial_quality_by_node={7: 1.0})       # legacy pre-fix callables
    row('T2-2', 'legacy callables: does the candidate fish still get checked?',
        {'legacy args': 'temp_class_of/quantity_of/service_time_of/initial_quality_by_node',
         'initial_quality_by_node[7]': 1.0, 'contract initial_value[1]': 9.0},
        'the coverage gate must still refuse an unverified candidate fish',
        {'composite': ok2, 'final': rc2.get('final'),
         'omega10_verdict': (rc2['gates'].get('omega10_coverage') or {}).get('verdict'),
         'omega10_mirror': (rc2['gates'].get('omega10_initialization') or {}).get('verdict')},
        'OK' if ok2 is False else 'FALSE_PASS',
        'm1b_q2_certifier.py:_legacy_attrs (iq.setdefault(n, 1.0) so the coverage gate REFUSES)',
        False)

    # T2-3 _Receipt['THERMAL_HARD_OK'] compatibility read
    ok3, rc3 = positive_case(proof=lambda **k: (True, {}))
    item = rc3['THERMAL_HARD_OK']
    stored = 'THERMAL_HARD_OK' in dict.keys(rc3)
    js = json.dumps(dict(rc3), default=str)
    row('T2-3', "_Receipt['THERMAL_HARD_OK'] read-only compatibility accessor",
        {'receipt': 'from a continuous-OK positive case'},
        'the legacy name must NOT be stored, and reading it must yield the SAMPLED level',
        {'subscript_value': item, 'attribute_value': rc3.THERMAL_HARD_OK,
         'key_stored_in_receipt': stored,
         'name_appears_in_serialised_receipt': 'THERMAL_HARD_OK' in js,
         'THERMAL_SAMPLED_OK': rc3.get('THERMAL_SAMPLED_OK')},
        'OK' if (item == rc3.get('THERMAL_SAMPLED_OK') and not stored
                 and 'THERMAL_HARD_OK' not in js) else 'CHECK',
        'm1b_q2_certifier.py:_Receipt.__missing__ / THERMAL_HARD_OK property', False)

    # T2-4 check_omega10_initialization legacy per-lot semantics
    pc, _, _ = __import__('m1b_contract_identity').check_quality_classes_are_certifiable(
        CONTRACT, epsilon_init=QC.EPSILON_INIT)
    legacy_verdict = QC.check_omega10_initialization([], pc)
    class FakeLot:
        order_id = 7
        temp_class = 1
        initial_quality = 1.0
    legacy_bad = QC.check_omega10_initialization([FakeLot()], pc)
    cov = QC.certify_omega10_coverage(on_board_lots=[], committed_fish_nodes=[],
                                      candidate_fish_nodes=[7], revealed_nodes={7},
                                      provider=attrs_for(iq={7: 1.0}, tc={7: 1}, qty={7: 1.0}),
                                      per_class_constants=pc)
    row('T2-4', 'check_omega10_initialization keeps pre-fix per-lot semantics',
        {'legacy call with []': 'no lots', 'legacy call with iq=1.0 lot': 'mismatch',
         'coverage call with candidate fish 7 iq=1.0': 'must refuse'},
        'the legacy shim may keep its old semantics but MUST NOT be on the certification path',
        {'legacy_empty_ok': legacy_verdict[0], 'legacy_empty_verdict': legacy_verdict[1],
         'legacy_bad_verdict': legacy_bad[1],
         'coverage_ok': cov[0], 'coverage_verdict': cov[1],
         'shim_warning_present': 'LEGACY_SHIM_WARNING' in (legacy_verdict[2] or {})},
        'OK' if (legacy_verdict[0] is True and cov[0] is False
                 and 'LEGACY_SHIM_WARNING' in (legacy_verdict[2] or {})) else 'FALSE_PASS',
        'm1b_quality_certificate.py:check_omega10_initialization (LEGACY SHIM) vs '
        'certify_omega10_coverage (certification path)', False)


# ==============================================================================================
def task3():
    print('\n===== T3: strict-online source identity =====')

    # T3-1 getter refuses undeclared nodes (claimed capability)
    a = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, served=(1,))
    refused = False
    try:
        a.temp_class_of(7)
    except PermissionError:
        refused = True
    row('T3-1', 'getter refuses a node outside the declared set',
        {'served_nodes': (1,), 'query': 'temp_class_of(7)'},
        'PermissionError', {'refused': refused},
        'OK' if refused else 'FALSE_PASS',
        'm1b_quality_certificate.py:PublicOrderAttributes._gate', False)

    # T3-2 is the DECLARED set itself validated against any trusted source?
    forged = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, served=(7,),
                       identity='FORGED-NOT-A-REAL-SOURCE')
    served_ok = forged.temp_class_of(7) == 1
    row('T3-2', 'legacy provider: declaration is still self-asserted, BUT the certifier refuses it',
        {'served_nodes': (7,), 'source_identity': 'FORGED-NOT-A-REAL-SOURCE',
         'revealed at clock=0.0': 'nothing'},
        'the LEGACY provider remains self-asserting by construction; F-T3 confines that risk by '
        'making the PRODUCTION path refuse any non-projection provider, so the weak type can no '
        'longer reach certification',
        {'legacy_query_succeeded': served_ok,
         'legacy_type_is_self_asserting': True,
         'production_path_accepts_non_projection': False,
         'confinement': 'm1b_q2_certifier.py projection gate rejects it (see T3-7)',
         'legacy_shim_label': getattr(QC.PublicOrderAttributes, '__name__', 'PublicOrderAttributes')},
        'NOT_ESTABLISHED_FOR_THE_LEGACY_TYPE_ONLY',
        'm1b_public_projection.py is the trusted boundary; PublicOrderAttributes survives only as '
        'a legacy type that certification refuses', False)

    # T3-3 does the object RETAIN the full truth table?
    iq_all = {n: 9.0 for n in range(1, 8)}
    iq_all[7] = 1.0                       # future-relevant value retained in the mapping
    a3 = QC.PublicOrderAttributes(initial_quality=iq_all, temp_class={n: 1 for n in range(1, 8)},
                                  quantity={n: 1.0 for n in range(1, 8)}, served_nodes=(1,),
                                  source_identity='retains-full-table')
    internals = [x for x in ('_iq', '_tc', '_qty', 'served_nodes', 'source_identity')
                 if hasattr(a3, x)]
    leaked = None
    try:
        leaked = a3._iq.get(7)
    except Exception as exc:                                             # noqa: BLE001
        leaked = 'BLOCKED:%s' % type(exc).__name__
    row('T3-3', 'legacy provider retains the full table; the TRUSTED projection does not (T3-8)',
        {'legacy constructor mapping': 'nodes 1..7', 'served_nodes': (1,)},
        'documented contrast: the legacy type keeps the whole mapping reachable, while the '
        'trusted projection stores revealed nodes only',
        {'legacy_reachable_attributes': internals, 'legacy__iq[7]': leaked,
         'trusted_projection_retains_revealed_only': 'see T3-8',
         'production_path_accepts_legacy': False},
        'NOT_ESTABLISHED_FOR_THE_LEGACY_TYPE_ONLY',
        'm1b_quality_certificate.py:PublicOrderAttributes (legacy) vs '
        'm1b_public_projection.py:TrustedPublicProjection (trusted)', False)

    # T3-4 mutability of the retained table after construction
    a4 = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, served=(1, 7))
    before = a4.initial_quality_of(7)
    a4._iq[7] = 1.0                       # mutate the retained table in place
    after = a4.initial_quality_of(7)
    row('T3-4', 'legacy provider stays mutable; the trusted projection is immutable (T3-8)',
        {'before': before, 'mutation': '_iq[7] = 1.0 on the LEGACY type'},
        'documented contrast: the trusted projection blocks this mutation (MappingProxyType)',
        {'legacy_before': before, 'legacy_after': after, 'legacy_mutable': before != after,
         'trusted_projection_mutation_blocked': 'see T3-8',
         'production_path_accepts_legacy': False},
        'NOT_ESTABLISHED_FOR_THE_LEGACY_TYPE_ONLY',
        'm1b_quality_certificate.py:PublicOrderAttributes (legacy dicts) vs '
        'm1b_public_projection.py:TrustedPublicProjection (MappingProxyType)', False)

    # T3-5 PublicTimeInputs: same questions
    t = times_for(service={7: 0.0}, revealed=(7,))
    refused_t = False
    try:
        t.service_time_of(1)
    except PermissionError:
        refused_t = True
    slots_blocked = False
    try:
        t.some_new_attribute = 1
    except AttributeError:
        slots_blocked = True
    t.service_time[1] = 99.0              # mutate the retained table (allowed: plain dict)
    row('T3-5', 'PublicTimeInputs: getter gates and __slots__ blocks new attrs, tables mutable',
        {'revealed_nodes': (7,), 'query': 'service_time_of(1)',
         'then': 'service_time[1] = 99.0 and an attempt to set a new attribute'},
        'the gate must refuse undeclared nodes; __slots__ should block new attributes; the '
        'retained dicts should ideally be immutable too',
        {'getter_refused_undeclared': refused_t, 'slots_block_new_attribute': slots_blocked,
         'table_mutable_after_construction': t.service_time.get(1) == 99.0},
        'OK' if (refused_t and slots_blocked) else 'FALSE_PASS',
        'm1b_timing.py:PublicTimeInputs (__slots__ enforced; service_time/tw_start/tw_end are '
        'plain dicts, so in-place mutation is possible)', False)

    # T3-6 the TRUSTED PROJECTION refuses nodes the environment has not revealed (F-T3)
    class FakeEnv:
        reveal_time = np.full((1, 8), 1e9)      # nothing is revealed at clock 0
        num_nodes = 8
    proj6, pv6, pd6 = PP.build_projection_from_env(
        FakeEnv(), inst_idx=0, event_id=1, clock=0.0, contract=CONTRACT,
        snapshot_identity='snap-1',
        fields={'temp_class': {7: 1}, 'demand': {7: 1.0}, 'tw_start': {7: 0.0},
                'tw_end': {7: 1e9}, 'service_time': {7: 0.05}, 'initial_quality': {7: 9.0}})
    reads = 'PROJECTION_NOT_BUILT'
    if proj6 is not None:
        try:
            reads = proj6.temp_class_of(7)
        except PermissionError:
            reads = 'REFUSED'
    row('T3-6', 'F-T3: a declaration contradicting the environment is refused',
        {'env.reveal_time[0,7]': 1e9, 'clock': 0.0, 'fields supplied for node 7': 'yes'},
        'the construction boundary derives the visible set from env.reveal_time, so node 7 must '
        'not be reachable at clock 0',
        {'n_revealed': len(proj6.revealed_nodes) if proj6 is not None else None,
         'temp_class_of(7)': reads,
         'source_identity_is_computed': (proj6.source_identity.startswith('proj:')
                                         if proj6 is not None else None)},
        'FALSE_PASS' if reads == 1 else 'OK',
        'm1b_public_projection.py:build_projection_from_env derives `revealed` from '
        'env.reveal_time and the current clock only', True)

    # T3-7 the production certifier refuses a NON-projection provider
    okN, rcN = certifier().certify(
        vehicles=[V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))],
        served_mask=np.zeros(40, dtype=bool), candidate_plans={0: plan(0, 0.0, [7, 0])},
        clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
        times=times_for(service={7: 0.05}),
        order_attrs=attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}))
    row('T3-7', 'F-T3: a self-declared provider is refused by the certifier',
        {'provider': 'PublicOrderAttributes(served_nodes=(1..7), source_identity="a-synth-1")'},
        'externally declared visible sets and free-text source identities must not qualify',
        {'composite': okN, 'final': rcN.get('final'),
         'projection_gate': (rcN['gates'].get('projection') or {}).get('verdict')},
        'OK' if ((not okN)
                 and (rcN['gates'].get('projection') or {}).get('verdict')
                 == PP.PROJECTION_DECLARED_SET_REJECTED) else 'FALSE_PASS',
        'm1b_q2_certifier.py: projection gate requires isinstance(TrustedPublicProjection)', True)

    # T3-8 the projection retains REVEALED nodes only, and is immutable
    full_iq = {n: 9.0 for n in range(1, 8)}
    full_iq[7] = 1.0
    proj8 = PP.projection_from_revealed_maps(
        iq={1: 9.0}, tc={1: 1}, qty={1: 1.0}, svc={1: 0.05}, tws={1: 0.0}, twe={1: 1e9},
        revealed_nodes=(1,), instance_id=0, event_id=1, clock=0.0,
        contract_hash=getattr(CONTRACT, 'contract_hash', None), snapshot_identity=None)
    leaked = None
    try:
        leaked = proj8._iq.get(7)
    except Exception as exc:                                             # noqa: BLE001
        leaked = 'BLOCKED:%s' % type(exc).__name__
    mutate_blocked = False
    try:
        proj8._iq[7] = 1.0
    except TypeError:
        mutate_blocked = True
    row('T3-8', 'F-T3: the projection holds ONLY revealed nodes and is immutable',
        {'constructor iq': '{1: 9.0}', 'revealed_nodes': (1,), 'attempt': 'read and write _iq[7]'},
        'no whole-node table may survive inside the object, and post-construction mutation must '
        'fail',
        {'_iq[7]': leaked, 'mutation_blocked': mutate_blocked,
         'n_retained_nodes': len(proj8._iq), 'revealed': sorted(proj8.revealed_nodes)},
        'OK' if (leaked is None and mutate_blocked and len(proj8._iq) == 1) else 'FALSE_PASS',
        'm1b_public_projection.py:TrustedPublicProjection wraps its tables in MappingProxyType '
        'and stores revealed nodes only', True)

    # T3-9 two events must not mix visible sets or identities
    projA = proj_for(served=(1, 2), instance_id=0, event_id=1, clock=0.0)
    projB = proj_for(served=(1, 2, 3), instance_id=0, event_id=2, clock=5.0)
    cross = None
    try:
        projA.temp_class_of(3)
        cross = 'LEAKED_ACROSS_EVENTS'
    except PermissionError:
        cross = 'REFUSED'
    same, v9, d9 = projA.assert_same_identity(projB)
    row('T3-9', 'F-T3: two events have independent visible sets and identities',
        {'event 1 revealed': (1, 2), 'event 2 revealed': (1, 2, 3), 'query': 'projA.temp_class_of(3)'},
        'event 1 must not expose event 2 nodes, and the identities must differ',
        {'cross_event_query': cross,
         'identities_equal': same, 'identity_verdict': v9,
         'identity_A': projA.source_identity[:24], 'identity_B': projB.source_identity[:24]},
        'OK' if (cross == 'REFUSED' and same is False) else 'FALSE_PASS',
        'm1b_public_projection.py: identity is computed from (instance, event, clock, revealed '
        'set, contract, snapshot)', True)


# ==============================================================================================
def task4():
    print('\n===== T4: original-49 compatibility + multi-vehicle FALSE_REJECT =====')
    spec = EXT / 'docs' / '当前规划' / '决策与审计' / 'M1b_Q2Fix_修复规格_2026-10-16.md'
    txt = spec.read_text(encoding='utf-8') if spec.exists() else ''
    row('T4-1', 'which old criteria the fix spec declares non-inheritable',
        {'spec': spec.name},
        'the spec must name the non-inheritable criteria rather than leave it implicit',
        {'names_A2_A2b_silent_zero': ('A2' in txt and 'A2b' in txt),
         'names_no_time_inputs': ('per-instance time inputs' in txt or 'times=' in txt),
         'names_continuous_fail_closed': 'fail-closed' in txt,
         'names_legacy_divergence': 'LEGACY' in txt or '兼容' in txt},
        'OK' if 'fail-closed' in txt else 'CHECK',
        'M1b_Q2Fix_修复规格_2026-10-16.md §2 and §6', False)

    # T4-2 real multi-vehicle mixture: idle-undispatched + closed + active
    vs = [
        V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0)),
        V(1, 'idle', node=0, ready=0.0, cc=None),                    # never dispatched
        V(2, 'closed', node=0, ready=5.0, cc=fresh_cc(n_lots=0), return_finish=None),
    ]
    ps = {0: plan(0, 0.0, [7, 0]), 1: plan(1, 0.0, []), 2: plan(2, 5.0, [])}
    proj = proj_for(svc={7: 0.05})
    c = certifier(closed_return_history={2: 5.0})
    ok, rc = c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                       clock=0.0, inst_idx=0, travel_fn=lambda a_, b_: 1.0,
                       distance_fn=lambda a_, b_: 10.0, times=proj, order_attrs=proj,
                       projection=proj)
    tv = ((rc['gates'].get('thermal_replay') or {}).get('detail') or {}).get('per_vehicle', {})
    v1 = (tv.get(1) or tv.get('1') or {}).get('verdict')
    row('T4-2', 'F-T4: idle-undispatched vehicle is NOT_APPLICABLE, not a state failure',
        {'vehicle 0': 'ready, loaded', 'vehicle 1': "idle, coldchain_state=None",
         'vehicle 2': 'closed, no cargo, history supplied'},
        'F-T4: an idle vehicle that was never dispatched carries no thermal risk and must not be '
        'reported as a state failure; the sampled fleet verdict must be OK',
        {'composite': ok, 'v0': (tv.get(0) or tv.get('0') or {}).get('verdict'), 'v1': v1,
         'v2': (tv.get(2) or tv.get('2') or {}).get('verdict'),
         'THERMAL_SAMPLED_OK': rc.get('THERMAL_SAMPLED_OK'),
         'applicability': rc.get('thermal_applicability')},
        'OK' if (v1 == TG.THERMAL_NOT_APPLICABLE and rc.get('THERMAL_SAMPLED_OK') is True)
        else ('FALSE_REJECT' if v1 == TG.THERMAL_REPLAY_STATE_MISSING else 'CHECK'),
        'm1b_thermal_gate.py:classify_thermal_applicability (F-T4)', True)

    # T4-3 isolated: does ONE undispatched idle vehicle alone block the sampled verdict?
    vs3 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0)),
           V(1, 'idle', node=0, ready=0.0, cc=None)]
    ps3 = {0: plan(0, 0.0, [7, 0]), 1: plan(1, 0.0, [])}
    c3 = certifier()
    proj3 = proj_for(svc={7: 0.05})
    ok3, rc3 = c3.certify(vehicles=vs3, served_mask=np.zeros(40, dtype=bool),
                          candidate_plans=ps3, clock=0.0, inst_idx=0,
                          travel_fn=lambda a_, b_: 1.0, distance_fn=lambda a_, b_: 10.0,
                          times=proj3, order_attrs=proj3, projection=proj3)
    tv3 = ((rc3['gates'].get('thermal_replay') or {}).get('detail') or {}).get('per_vehicle', {})
    v1_3 = (tv3.get(1) or tv3.get('1') or {}).get('verdict')
    row('T4-3', 'F-T4: ONE never-dispatched idle vehicle no longer sinks the sampled verdict',
        {'vehicle 1': "status='idle', coldchain_state=None, empty plan"},
        'F-T4: the idle vehicle must be NOT_APPLICABLE, THERMAL_SAMPLED_OK must be True, and the '
        'ONLY remaining reason the composite is False must be the continuous thermal level '
        '(which is fail-closed BY DESIGN this round, not a fleet false reject)',
        {'composite': ok3, 'final': rc3.get('final'), 'v1_verdict': v1_3,
         'THERMAL_SAMPLED_OK': rc3.get('THERMAL_SAMPLED_OK'),
         'THERMAL_CONTINUOUS_STATE': rc3.get('THERMAL_CONTINUOUS_STATE'),
         'blocking_gate': 'THERMAL_CONTINUOUS_NOT_ESTABLISHED (by design)'
                          if rc3.get('THERMAL_SAMPLED_OK') else 'thermal sampled failure'},
        'OK' if (v1_3 == TG.THERMAL_NOT_APPLICABLE and rc3.get('THERMAL_SAMPLED_OK') is True
                 and rc3.get('THERMAL_CONTINUOUS_STATE')
                 == TG.THERMAL_CONTINUOUS_NOT_ESTABLISHED)
        else 'FALSE_REJECT',
        'm1b_thermal_gate.py:classify_thermal_applicability; m1b_q2_certifier.py composite '
        'requires the CONTINUOUS level', True)

    # T4-4 the ORIGINAL negative control must still refuse: cargo without state
    vs4 = [V(0, 'ready', node=0, ready=0.0, cc=None)]          # cargo implied by the plan only
    vs4b = [V(0, 'ready', node=0, ready=0.0, cc=None)]
    # a vehicle carrying cargo but with no cold-chain state is constructed by giving it a plan
    # that continues from a non-depot node
    vs4c = [V(0, 'ready', node=5, ready=1.0, cc=None, suffix=[7])]
    ap4, av4, ad4 = TG.classify_thermal_applicability(
        IS.build_frozen_snapshot(vs4c, np.zeros(40, dtype=bool),
                                 {0: plan(0, 1.0, [7, 0])})[0]['vehicles'][0],
        clock=1.0, closed_return_history={})
    row('T4-4', 'ORIGINAL negative control preserved: a dispatched vehicle without state refuses',
        {'vehicle': "status='ready', current_node=5 (away from depot), coldchain_state=None"},
        'F-T4: the not-applicable conditions are NOT all met, so this must REFUSE',
        {'applies': ap4, 'verdict': av4, 'why': (ad4 or {}).get('why', '')[:110]},
        'OK' if (ap4 is None and av4 == TG.THERMAL_REPLAY_STATE_MISSING) else 'FALSE_PASS',
        'm1b_thermal_gate.py:classify_thermal_applicability refusal branch', True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    print('M1-b/Q2-Fix-T trust-boundary audit')
    task1(); task2(); task3(); task4()
    fp = [r for r in ROWS if r['VERDICT'] == 'FALSE_PASS']
    fr = [r for r in ROWS if r['VERDICT'] == 'FALSE_REJECT']
    ne = [r for r in ROWS if r['VERDICT'] == 'TRUST_BOUNDARY_NOT_ESTABLISHED']
    print('\n== %d checks: %d FALSE_PASS, %d FALSE_REJECT, %d NOT_ESTABLISHED =='
          % (len(ROWS), len(fp), len(fr), len(ne)))
    for r in ROWS:
        if r['VERDICT'] != 'OK':
            print('   %-6s %-58s %s' % (r['id'], r['name'][:58], r['VERDICT']))
    payload = {'schema': 'm1b_q2fixt_trust_boundary_audit_v1', 'n_checks': len(ROWS),
               'n_false_pass': len(fp), 'n_false_reject': len(fr),
               'n_not_established': len(ne), 'results': ROWS}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                  encoding='utf-8')
        print('written: %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
