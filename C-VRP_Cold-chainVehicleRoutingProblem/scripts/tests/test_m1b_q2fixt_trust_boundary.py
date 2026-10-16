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
import m1b_return_feasibility as RF                                      # noqa: E402
import m1b_thermal_gate as TG                                            # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
from m1b_q2_certifier import M1bQ2FixCertifier, ENGINEERING_CERTIFIED     # noqa: E402

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


def certifier(**kw):
    kw.setdefault('contract', CONTRACT)
    kw.setdefault('T_end', T_END)
    kw.setdefault('budget', SYNTHETIC_B)
    return M1bQ2FixCertifier(**kw)


def positive_case(proof=None, vehicles=None, planes=None):
    a = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0})
    vs = vehicles if vehicles is not None else [
        V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))]
    ps = planes if planes is not None else {0: plan(0, 0.0, [7, 0])}
    c = certifier(continuous_extremum_proof=proof)
    return c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                     clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0,
                     distance_fn=lambda a, b: 10.0,
                     times=times_for(service={7: 0.05}), order_attrs=a)


# ==============================================================================================
def task1():
    print('\n===== T1: continuous-temperature proof hook trust boundary =====')

    # T1-1 an arbitrary callable that simply returns (True, {})
    ok, rc = positive_case(proof=lambda **k: (True, {}))
    row('T1-1', 'arbitrary proof returning (True,{}) yields CONTINUOUS_OK',
        {'continuous_extremum_proof': 'lambda **k: (True, {})'},
        'a proof that carries no identity, domain or contract binding should NOT be able to '
        'assert continuous-time feasibility',
        {'THERMAL_CONTINUOUS_STATE': rc.get('THERMAL_CONTINUOUS_STATE'),
         'THERMAL_CONTINUOUS_OK': rc.get('THERMAL_CONTINUOUS_OK'),
         'composite_pass': rc.get('composite_pass'), 'final': rc.get('final')},
        'FALSE_PASS' if (rc.get('THERMAL_CONTINUOUS_OK') is True
                         and rc.get('composite_pass') is True) else 'NOT_ESTABLISHED',
        'm1b_thermal_gate.py:251 cont_state = THERMAL_CONTINUOUS_OK if proven is True else ...',
        True)

    # T1-2 what does the hook actually receive / is anything verified about it
    seen = {}

    def spy(**k):
        seen.update({kk: (type(vv).__name__ if not isinstance(vv, (int, float, str, list))
                          else vv) for kk, vv in k.items()})
        seen['_kwargs'] = sorted(k.keys())
        return True, {}
    positive_case(proof=spy)
    row('T1-2', 'the hook receives contract/bounds/samples/durations but is NOT identified',
        {'hook': 'spy capturing kwargs', 'received_kwargs': seen.get('_kwargs')},
        'supplying the contract is NOT the same as VERIFYING that the proof was built for it',
        {'received_kwargs': seen.get('_kwargs'),
         'any_sha256_check': False, 'any_domain_check': False,
         'any_contract_binding_check': False, 'any_thermal_version_check': False,
         'any_provenance_check': False,
         'grep_evidence': 'the certifier stores self.continuous_extremum_proof and forwards it '
                          'verbatim (m1b_q2_certifier.py:106, :278); the gate only tests '
                          '`proven is True` (m1b_thermal_gate.py:251)'},
        'TRUST_BOUNDARY_NOT_ESTABLISHED',
        'm1b_q2_certifier.py:106,278 ; m1b_thermal_gate.py:246-252', True)

    # T1-3 a hook that always lies even with contradicting samples
    ok3, rc3 = positive_case(proof=lambda **k: (True, {'says': 'ok'}))
    row('T1-3', 'a lying hook overrides contradicting sampled evidence?',
        {'hook': 'returns True unconditionally', 'sampled envelope': 'measured in the same run'},
        'the sampled level is measured independently, so a lying hook must not be able to turn a '
        'SAMPLED violation into a pass',
        {'THERMAL_SAMPLED_OK': rc3.get('THERMAL_SAMPLED_OK'),
         'THERMAL_CONTINUOUS_STATE': rc3.get('THERMAL_CONTINUOUS_STATE'),
         'composite_pass': rc3.get('composite_pass')},
        'OK' if rc3.get('THERMAL_SAMPLED_OK') is True else 'CHECK',
        'm1b_thermal_gate.py:234-236 (violations short-circuit the continuous branch)', False)

    # T1-4 a lying hook when the sampled envelope IS violated
    import dataclasses as dc
    tight = dc.replace(CONTRACT, thermal=dc.replace(
        CONTRACT.thermal, hard_temperature_bounds_c=((-50.0, -40.0),) * 3))
    a = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0})
    c4 = M1bQ2FixCertifier(contract=tight, T_end=T_END, budget=SYNTHETIC_B,
                           continuous_extremum_proof=lambda **k: (True, {}))
    ok4, rc4 = c4.certify(vehicles=[V(0, 'ready', node=0, ready=0.0,
                                      cc=fresh_cc(n_lots=1, iq=9.0))],
                          served_mask=np.zeros(40, dtype=bool),
                          candidate_plans={0: plan(0, 0.0, [7, 0])}, clock=0.0, inst_idx=0,
                          travel_fn=lambda a_, b_: 1.0, distance_fn=lambda a_, b_: 10.0,
                          times=times_for(service={7: 0.05}), order_attrs=a)
    row('T1-4', 'lying hook + SAMPLED violation -> composite must still fail',
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
    row('T2-1', '_LegacyTimes tw_start=0.0 re-opens the A1 late-window case',
        {'legacy register_service_times': '{5: 0.0}', 'true tw_start[5]': 20.0,
         'T_end': 19.0, 'times= passed': False},
        'the legacy path must not be able to certify a plan that certify_route rejects',
        {'omega11_verdict': v, 'omega11_return_arrival': d['per_vehicle'][0].get('return_arrival'),
         'certify_route_reason': cert['reason'],
         'certify_route_return_slack': cert['return_slack'],
         'time_input_identity': d.get('time_input_identity'),
         'REACHES_FIXED_COMPOSITE': False,
         'why_not': 'the certifier refuses at the time_inputs gate when `times` is None '
                    '(OMEGA11_TIME_INPUT_UNAVAILABLE), so the legacy fallback cannot reach '
                    'composite_pass; but any caller that calls certify_return_feasibility '
                    'DIRECTLY without times= silently gets the optimistic replay'},
        'FALSE_PASS' if (ok and cert['reason'] == 'return') else 'CHECK',
        'm1b_return_feasibility.py:_LegacyServiceTimes.register (tw_start={n: 0.0}) and the '
        'times-is-None fallback in certify_return_feasibility; '
        'm1b_q2_certifier.py blocks the same fallback at the time_inputs gate', True)

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
    row('T3-2', 'declared set + source identity are SELF-ASSERTED (no trusted validation)',
        {'served_nodes': (7,), 'source_identity': 'FORGED-NOT-A-REAL-SOURCE',
         'revealed at clock=0.0': 'nothing (reveal_time not consulted)'},
        'a declaration that reveals node 7 at clock 0 without any trusted source should not '
        'by itself authorise reading node 7',
        {'query_succeeded': served_ok,
         'any_registry_or_reveal_check': False,
         'identity_is_free_text': True},
        'TRUST_BOUNDARY_NOT_ESTABLISHED',
        'm1b_quality_certificate.py:PublicOrderAttributes.__init__ stores served_nodes and '
        'source_identity verbatim; nothing cross-checks them against reveal_time or a registry',
        True)

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
    row('T3-3', 'the full truth table is RETAINED inside the object',
        {'constructor mapping': 'nodes 1..7 with node 7 = 1.0', 'served_nodes': (1,)},
        'the gating getter may refuse, but the raw mapping is still reachable as an attribute',
        {'reachable_attributes': internals, '_iq[7]': leaked,
         'gate_blocks_public_getter': True},
        'TRUST_BOUNDARY_NOT_ESTABLISHED',
        'm1b_quality_certificate.py:PublicOrderAttributes keeps _iq/_tc/_qty as plain dicts with '
        'no __slots__ privacy; __slots__ is declared but the dicts themselves are readable and '
        'mutable', False)

    # T3-4 mutability of the retained table after construction
    a4 = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, served=(1, 7))
    before = a4.initial_quality_of(7)
    a4._iq[7] = 1.0                       # mutate the retained table in place
    after = a4.initial_quality_of(7)
    row('T3-4', 'the retained table is MUTABLE after construction',
        {'before': before, 'mutation': '_iq[7] = 1.0'},
        'a certificate input should be immutable once constructed',
        {'before': before, 'after': after, 'changed': before != after},
        'TRUST_BOUNDARY_NOT_ESTABLISHED' if before != after else 'OK',
        'm1b_quality_certificate.py:PublicOrderAttributes.__slots__ lists _iq/_tc/_qty as dicts',
        False)

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

    # T3-6 can a caller declare reveal-time truth that contradicts the environment?
    class FakeEnv:
        reveal_time = np.full((1, 8), 1e9)      # nothing is revealed at clock 0
        num_nodes = 8
    a6 = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, served=(7,),
                   identity='claims-node-7-visible-at-clock-0')
    reads = None
    try:
        reads = a6.temp_class_of(7)
    except PermissionError:
        reads = 'REFUSED'
    row('T3-6', 'a declaration contradicting the environment is not cross-checked',
        {'env.reveal_time[0,7]': 1e9, 'clock': 0.0, 'provider served_nodes': (7,)},
        'the provider should refuse node 7 when the environment says it is unrevealed',
        {'temp_class_of(7)': reads,
         'provider_consults_env': False,
         'consequence': 'a caller can declare any node visible and read its true attributes '
                        'through this provider'},
        'TRUST_BOUNDARY_NOT_ESTABLISHED',
        'm1b_quality_certificate.py:PublicOrderAttributes has no environment/reveal binding',
        True)


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
    a = attrs_for(iq={7: 9.0}, tc={7: 1}, qty={7: 1.0})
    c = certifier(closed_return_history={2: 5.0})
    ok, rc = c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                       clock=0.0, inst_idx=0, travel_fn=lambda a_, b_: 1.0,
                       distance_fn=lambda a_, b_: 10.0, times=times_for(service={7: 0.05}),
                       order_attrs=a)
    tv = ((rc['gates'].get('thermal_replay') or {}).get('detail') or {}).get('per_vehicle', {})
    row('T4-2', 'idle-undispatched vehicle in a multi-vehicle state',
        {'vehicle 1': "status='idle', coldchain_state=None"},
        'an idle vehicle that was never dispatched has no thermal state; whether that should '
        'block the WHOLE certification or only that vehicle is a design question',
        {'composite': ok, 'v1_thermal_verdict': (tv.get(1) or tv.get('1') or {}).get('verdict'),
         'v0_thermal_verdict': (tv.get(0) or tv.get('0') or {}).get('verdict'),
         'v2_thermal_verdict': (tv.get(2) or tv.get('2') or {}).get('verdict')},
        'FALSE_REJECT' if (ok is False and (tv.get(1) or tv.get('1') or {}
                                            ).get('verdict') == TG.THERMAL_REPLAY_STATE_MISSING)
        else 'CHECK',
        'm1b_thermal_gate.py:replay_thermal_envelope returns THERMAL_REPLAY_STATE_MISSING when '
        'coldchain_state is None; the certifier requires ALL vehicles to be SAMPLED_OK', True)

    # T4-3 isolated: does ONE undispatched idle vehicle alone block certification?
    vs3 = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0)),
           V(1, 'idle', node=0, ready=0.0, cc=None)]
    ps3 = {0: plan(0, 0.0, [7, 0]), 1: plan(1, 0.0, [])}
    c3 = certifier()
    ok3, rc3 = c3.certify(vehicles=vs3, served_mask=np.zeros(40, dtype=bool),
                          candidate_plans=ps3, clock=0.0, inst_idx=0,
                          travel_fn=lambda a_, b_: 1.0, distance_fn=lambda a_, b_: 10.0,
                          times=times_for(service={7: 0.05}), order_attrs=a)
    tv3 = ((rc3['gates'].get('thermal_replay') or {}).get('detail') or {}).get('per_vehicle', {})
    only = [k for k in (tv3.get(1), tv3.get('1')) if k]
    row('T4-3', 'ONE never-dispatched idle vehicle blocks the whole composite',
        {'vehicle 1': "status='idle', coldchain_state=None, empty plan"},
        'an idle vehicle with nothing on board and no thermal state carries no thermal risk; '
        'refusing ALL certification because of it is a systematic FALSE_REJECT in realistic '
        'multi-vehicle days where some vehicles are never dispatched',
        {'composite': ok3, 'final': rc3.get('final'),
         'v1_verdict': (only[0] if only else {}).get('verdict'),
         'reason': 'THERMAL_REPLAY_STATE_MISSING for the idle vehicle'},
        'FALSE_REJECT' if ok3 is False else 'OK',
        'm1b_thermal_gate.py:replay_thermal_envelope (state is None -> STATE_MISSING); '
        'm1b_q2_certifier.py requires every vehicle to be THERMAL_SAMPLED_OK', True)


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
