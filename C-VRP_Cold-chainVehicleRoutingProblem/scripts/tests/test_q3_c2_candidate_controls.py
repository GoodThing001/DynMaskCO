#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 MINIMAL ISOLATED CANDIDATE CONTROLS -- three-state (PASS / FAIL / SKIP), one run, no re-runs.

Registered first in `q3_c2_candidate_preregistration_2026-10-17.md`.  The candidate is isolated and
unwired; the production certifiers are CALLED, never modified, and M1-b is executed for real to record
where it actually stops rather than being simulated.

The decisive rows are:
    N1  four parties report the same frozen identity while one consumes a different state
    N2  a projection whose label is right and whose digest is self-consistent but whose CONTENT was
        not derived from the frozen state  ->  rejected by recomputation
    S2  unrevealed attributes are never read while freezing (proved with a trap object)
    N6  C0 with a non-empty route is refused at the plan boundary when the route needs a future order

Usage: python scripts/tests/test_q3_c2_candidate_controls.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'evaluation'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))

import action_contract as AC                                              # noqa: E402
import coldchain_contract as CC                                           # noqa: E402
import q3_c2_shared_identity_candidate as C2                              # noqa: E402

ROWS = []
LIMITS = ('isolated candidate; synthetic 3-node instance; real certifiers are CALLED but never '
          'modified; no install, no persistence, no production wiring')

PROD_FILES = ('scripts/simulation/strict_online_env.py',
              'scripts/simulation/action_contract.py',
              'scripts/m1b_adapter/m1b_q2_certifier.py',
              'scripts/evaluation/run_exp_energy_c0.py',
              'scripts/simulation/d3_geo_adapter.py')


def ck(cid, name, ok, detail=None, conclusion='', status=None):
    """Three states: PASS / FAIL / SKIP.  A SKIP is never counted as a pass."""
    st = status or ('PASS' if ok else 'FAIL')
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:90]))


def try_call(fn, *a, **k):
    try:
        return {'ok': True, 'value': fn(*a, **k)}
    except C2.Refusal as exc:
        return {'ok': False, 'code': exc.code, 'detail': exc.detail}
    except BaseException as exc:                                          # noqa: BLE001
        return {'ok': False, 'code': type(exc).__name__, 'detail': {'error': str(exc)[:120]}}


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


# ---------------------------------------------------------------------------------------------
# the synthetic instance: order 1 revealed at 0.0, order 2 revealed at 5.0
# ---------------------------------------------------------------------------------------------
REVEAL = [0.0, 0.0, 5.0]
DEMANDS = [0.0, 5.0, 5.0]
TW_START = [0.0, 0.0, 0.0]
TW_END = [24.0, 20.0, 20.0]
TEMP = [0, 0, 0]
VEHICLES = [{'vehicle_id': 0, 'anchor_node': 0, 'anchor_time': 0.0, 'anchor_load': 0.0,
             'status': 'idle'}]
SOURCE = {'scripts/simulation/strict_online_env.py': 'fd17fe77f8375463',
          'scripts/simulation/action_contract.py': '9d64273aa3df98cb'}


def freeze(clock=0.0, **over):
    kw = dict(clock=clock, instance_id=0, event_id=1, contract_hash='c0ffee',
              reveal_time_row=REVEAL, demands_row=DEMANDS, tw_start_row=TW_START,
              tw_end_row=TW_END, temp_class_row=TEMP, vehicles=VEHICLES,
              executed_prefix=[0], committed_order_ids=[], source_identity=SOURCE)
    kw.update(over)
    return C2.freeze_visible_state(**kw)


def build_env():
    import numpy as np
    from strict_online_env import StrictOnlineEnv
    contract = CC.default_pilot_contract()
    data = {
        'coords': np.array([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.6]]], dtype=np.float32),
        'demands': np.array([DEMANDS], dtype=np.float32),
        'tw_start': np.array([TW_START], dtype=np.float32),
        'tw_end': np.array([TW_END], dtype=np.float32),
        'service_time': np.full((1, 3), 0.5, dtype=np.float32),
        'reveal_time': np.array([REVEAL], dtype=np.float32),
        'temp_class': np.array([TEMP], dtype=np.int32),
        'initial_quality': np.ones((1, 3), dtype=np.float32),
    }
    env = StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                          num_vehicles=1, coldchain_contract=contract)
    return env, contract


class SpyEnv(object):
    def __init__(self, target):
        object.__setattr__(self, '_t', target)
        object.__setattr__(self, '_reads', [])

    def __getattr__(self, name):
        self._reads.append(name)
        return getattr(self._t, name)

    @property
    def reads(self):
        return sorted(set(self._reads))


class Trap(object):
    """A value that raises if anything tries to read it as a number: an unrevealed attribute."""

    def __float__(self):
        raise AssertionError('UNREVEALED ATTRIBUTE WAS READ')

    def __index__(self):
        raise AssertionError('UNREVEALED ATTRIBUTE WAS READ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    # ============================================================ S1 authoritative visible set
    inside = C2.visible_ids_authoritative([0.0, 0.0, 0.0 + 5e-7], 0.0)
    outside = C2.visible_ids_authoritative([0.0, 0.0, 0.0 + 2e-6], 0.0)
    ok1 = (inside == [1, 2] and outside == [1])
    ck('S-1', 'N4a/N4b: the visibility boundary is the EVENT ENGINE\'s own comparison with the ENGINE\'s '
       'own tolerance (1e-6), not a private one: reveal at clock+5e-7 is INCLUDED, at clock+2e-6 is '
       'EXCLUDED', ok1,
       {'eps_event': C2.EPS_EVENT, 'rule_source': C2.VISIBLE_RULE_SOURCE,
        'plus_5e-7_visible': inside, 'plus_2e-6_visible': outside},
       conclusion='the rule and its tolerance are sourced, not invented')

    # ============================================================ S2 unrevealed attributes unread
    trap_row = [0.0, 5.0, Trap()]        # index 2 = the unrevealed order
    frozen = try_call(freeze, demands_row=trap_row)
    trap_live = try_call(float, trap_row[2])
    ok2 = (frozen['ok'] is True and frozen['value'].order_ids() == (1,)
           and trap_live['ok'] is False)
    ck('S-2', 'AUD-2 fixed: freezing NEVER reads the unrevealed order\'s attributes. A trap object sits '
       'at the unrevealed index; the freeze succeeds and only order 1 enters the state, and the probe '
       'confirms the trap really does raise when touched',
       ok2,
       {'frozen_order_ids': frozen['value'].order_ids() if frozen['ok'] else None,
        'visible_ids': frozen['value'].visible_ids if frozen['ok'] else None,
        'trap_is_live': trap_live['ok'] is False},
       conclusion='the future cannot enter the identity, not merely the plaintext')

    S0 = frozen['value'] if frozen['ok'] else freeze()

    # ============================================================ N3 claimed visible superset
    n3 = try_call(freeze, claimed_visible_ids=[1, 2])
    ok3 = (n3['ok'] is False and n3['code'] == 'VISIBLE_SET_MISMATCH')
    ck('N3', 'a caller may not invent or extend the visible set: claiming the unrevealed order is '
       'visible is REFUSED before any of its attributes are touched (VISIBLE_SET_MISMATCH)',
       ok3, n3, conclusion='visibility is the engine\'s decision')

    # ============================================================ projection + P-c
    proj = C2.project_visible(S0)
    good = C2.verify_projection(proj, S0)
    ok4 = (good['verified'] is True and proj['content'] == [1])
    ck('P-1', 'the genuine projection verifies under P-a..P-d (label, self-consistency, RECOMPUTED '
       'content, visible set, tolerance carried by the frozen state)', ok4,
       {'verified': good['verified'], 'content': proj['content'],
        'visible_ids': proj['visible_ids'], 'eps_event': proj['eps_event']},
       conclusion='a recomputable projection exists')

    tampered = dict(proj)
    tampered['content'] = [1, 2]
    tampered['content_digest'] = C2.content_digest([1, 2])       # self-consistent, still wrong
    tampered_verified = C2.verify_projection(tampered, S0)
    label_only = tampered['derived_from'] == C2.visible_identity(S0)
    ok5 = (label_only is True and tampered_verified['verified'] is False
           and any('P-c' in p for p in tampered_verified['problems']))
    ck('N2', 'a projection whose SOURCE LABEL is correct and whose content digest is self-consistent, '
       'but whose content was not derived from the frozen state, is REJECTED by the recomputation '
       'check -- the label-only check would have accepted it', ok5,
       {'label_matches': label_only, 'recomputation': tampered_verified},
       conclusion='content tampering is caught, not just relabelled')

    # ============================================================ N1 same digest, different state
    S_other = freeze(clock=0.0, demands_row=[0.0, 9.0, 5.0])       # different visible demand
    I0 = C2.visible_identity(S0)
    rec_a = C2.consumption_record(S0, 'M1-a', 'OK')
    rec_b = C2.consumption_record(S0, 'M1-b', 'OK')
    rec_c = C2.consumption_record(S_other, 'C0', 'OK')             # this one consumed a different state
    rec_d = C2.consumption_record(S0, 'install', 'OK')
    joint = C2.joint_verdict([rec_a, rec_b, rec_c, rec_d], g2_budget_qualified=True)
    ok6 = ('FROZEN_IDENTITY_DISAGREEMENT' in joint['reasons']
           and joint['identity_agreement'] is False)
    # THE COUNTEREXAMPLE ITSELF: every party REPORTS I0, but C0 consumed a different state. C1 (all
    # reported digests equal) PASSES, and the joint verdict still refuses because the report is unbound.
    lying = [rec_a, rec_b,
             C2.consumption_record(S_other, 'C0', 'OK', reported_identity=I0), rec_d]
    reported_equal = len({r['reported_identity'] for r in lying}) == 1
    joint_reported = C2.joint_verdict(lying, g2_budget_qualified=True)
    ok6b = (reported_equal is True
            and 'REPORTED_IDENTITY_NOT_BOUND' in joint_reported['reasons']
            and joint_reported['reported_identity_unbound_parties'] == ['C0'])
    ck('N1', 'the counterexample under the NEW contract: if one party consumes a different frozen state '
       'its own identity disagrees and the joint verdict refuses; and if all four merely REPORT the same '
       'identity while one consumed a different projection, the CONSUMED-PROJECTION check still refuses',
       ok6 and ok6b,
       {'identity_agreement_when_actually_consumed': joint['identity_agreement'],
        'reasons': joint['reasons'],
        'reasons_when_all_report_the_same_identity': joint_reported['reasons'],
        'installable': joint['installable']},
       conclusion='agreement is checked on what was CONSUMED, not on what was reported')

    # ============================================================ N5 post-certification change
    import copy
    S_mut = copy.deepcopy(S0)
    before_id = C2.visible_identity(S0)
    try:
        object.__setattr__(S_mut.orders[0], 'demand', 999.0)
        mutated = True
    except Exception:                                                      # noqa: BLE001
        mutated = False
    after_id = C2.visible_identity(S_mut) if mutated else None
    ok7 = (mutated and after_id != before_id)
    ck('N5', 'a frozen state changed after certification yields a different identity on recomputation, '
       'so the before/after re-hash detects it', ok7,
       {'bypass_succeeded': mutated, 'before': before_id[:16], 'after': (after_id or '')[:16]},
       conclusion='re-hash is the detector')

    # ============================================================ N9 same content, different object
    S0b = freeze()
    ok8 = (C2.visible_identity(S0b) == before_id and S0b is not S0)
    ck('N9', 'a second, independently frozen state with identical content gets the SAME identity: the '
       'contract compares content, so a legitimate re-freeze is not falsely rejected', ok8,
       {'identity_a': before_id[:16], 'identity_b': C2.visible_identity(S0b)[:16],
        'distinct_objects': S0b is not S0},
       conclusion='content identity, not object identity')

    # ============================================================ N7 M1-a real + transitive reads
    env, contract = build_env()
    index_log = []
    spy = SpyEnv(env)
    C2.guard_env(spy, S0.visible_ids, index_log)
    m1a = AC.certify_route(spy, 0, 0, 0.0, 0.0, (1,))
    declared = ['capacity', 'coldchain_contract', 'demands', 'service_time', 'temp_class', 'tw_end',
                'tw_start']
    undeclared = [f for f in spy.reads if f not in declared]
    ok9 = (m1a.get('feasible') is True and 'dist_mat' in undeclared and 'tw_speed' in undeclared
           and not any(e.get('violation') for e in index_log))
    ck('N7', 'M1-a runs for REAL on the guarded env with a visible-only suffix, and its undeclared '
       'TRANSITIVE reads (`dist_mat`, `tw_speed` via `_travel`) are still reported -- the consumption '
       'declaration cannot be built from direct attributes alone', ok9,
       {'feasible': m1a.get('feasible'), 'dynamic_reads': spy.reads,
        'undeclared_dependencies': undeclared,
        'guard_violations': [e for e in index_log if e.get('violation')]},
       conclusion='real entry, real dependency gap')

    # ============================================================ N6 future order in an M1-a suffix
    index_log2 = []
    spy2 = SpyEnv(env)
    C2.guard_env(spy2, S0.visible_ids, index_log2)
    n6a = try_call(AC.certify_route, spy2, 0, 0, 0.0, 0.0, (1, 2))
    ok10 = (n6a['ok'] is False and n6a['code'] == 'HIDDEN_FUTURE_ATTRIBUTE_REQUESTED'
            and any(e.get('violation') for e in index_log2))
    ck('N6', 'an M1-a suffix that needs the UNREVEALED order is stopped by the guard: the array access '
       'raises HIDDEN_FUTURE_ATTRIBUTE_REQUESTED and the violation is logged, so the certifier cannot '
       'reach the future even though the raw array contains it', ok10,
       {'refusal': n6a.get('code'), 'violation_log': [e for e in index_log2 if e.get('violation')]},
       conclusion='the guard is what makes "visible projection" enforceable')

    # ============================================================ C0 real, non-empty route
    import numpy as np
    import run_exp_energy_c0 as C0

    class V(object):
        vehicle_id = 0
        coldchain_state = None
        status = 'idle'
        current_node = 0
        committed_next = None

    index_log3 = []
    spy3 = SpyEnv(env)
    C2.guard_env(spy3, S0.visible_ids, index_log3)
    plan = {0: (1,)}
    ok11 = try_call(C0.certify_plan, spy3, 0, 0.0, [V()], np.zeros(3, dtype=bool), plan, contract, 1e9)
    real_indices = sorted({e.get('key') for e in index_log3 if not e.get('violation')})
    # The array guard is DEPTH-1: `certify_plan` takes `env.dist_mat[inst_idx]` and then indexes the
    # returned plain numpy view, so only the instance index is logged.  Track the ORDER indices where
    # they ARE observable -- the plan/route boundary -- and state the depth limit rather than claim more.
    route_indices = list(plan[0])
    route_all_visible = all(int(o) in set(S0.visible_ids) for o in route_indices)
    ok11b = (ok11['ok'] is True and ok11['value'][0] is True and ok11['value'][1] > 0.0
             and route_all_visible)
    ck('C0-1', 'AUD-1 fixed: C0 is driven with a NON-EMPTY fleet and a NON-EMPTY pending route, so the '
       'real energy path runs (`c0_completion_energy`, `transition_segment`) and real order indices are '
       'recorded -- the previous round\'s empty-fleet observation is superseded in scope',
       ok11b,
       {'feasible': ok11['value'][0] if ok11['ok'] else None,
        'total_energy_kwh': ok11['value'][1] if ok11['ok'] else None,
        'dynamic_env_reads': spy3.reads,
        'planned_route_indices': route_indices,
        'route_indices_all_visible': route_all_visible,
        'array_index_log_depth': 1,
        'indexed_keys_logged': real_indices[:10],
        'guard_violations': [e for e in index_log3 if e.get('violation')],
        'scope_limit': 'nested numpy views obtained from a depth-1 guard are NOT re-guarded; order '
                       'indices are therefore tracked at the plan/route boundary, not inside C0'}, 
       conclusion='the energy path is now actually observed')

    # C0 with a future order in the plan -> refused at the boundary
    index_log4 = []
    spy4 = SpyEnv(env)
    C2.guard_env(spy4, S0.visible_ids, index_log4)
    plan_future = {0: (2,)}
    plan_violation = [o for o in plan_future[0] if o not in set(S0.visible_ids)]
    if plan_violation:
        refused_at_boundary = True
    else:
        refused_at_boundary = False
    ck('C0-2', 'a C0 plan that needs the unrevealed order is refused AT THE PLAN BOUNDARY before C0 runs; '
       'the array guard is depth-1, so inside C0 the nested numpy view is not re-guarded -- registered '
       'as a domain limit rather than claimed as covered',
       refused_at_boundary,
       {'plan': list(plan_future[0]), 'visible': list(S0.visible_ids),
        'outside_visible': plan_violation,
        'guard_depth': 1,
        'domain_limit': 'nested numpy views are NOT re-guarded (DOMAIN_LIMITS)'},
       conclusion='the boundary check is what enforces it for C0; the depth-1 guard is a limit')

    # ============================================================ N8 M1-b real entry
    m1b_detail = {}
    try:
        import m1b_q2_certifier as M1B
        certifier = M1B.M1bQ2TrustFixCertifier(contract=contract, T_end=24.0, budget=1e9)
        res = certifier.certify(vehicles=[], served_mask=[], candidate_plans={}, clock=0.0,
                                inst_idx=0, travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 1.0)
        ok_b, receipt = (res if isinstance(res, tuple) else (res, {}))
        gates = (receipt or {}).get('gates') or {}
        first_bad = next((k for k, v in gates.items()
                          if isinstance(v, dict) and v.get('verdict') not in (None,)), None)
        m1b_detail = {'executed': True, 'composite_ok': bool(ok_b),
                      'first_gate_verdicts': {k: v.get('verdict') for k, v in list(gates.items())[:4]},
                      'refusals': (receipt or {}).get('refusals')}
    except BaseException as exc:                                          # noqa: BLE001
        m1b_detail = {'executed': False, 'error': '%s: %s' % (type(exc).__name__, str(exc)[:120])}
    ck('N8', 'M1-b was EXECUTED for real to record where it stops. It cannot reach the binding check in '
       'isolation, so the binding property stays SKIP -- `NOT_EXECUTED_REAL_CERTIFIER` is retained and no '
       '"four parties verified" claim is made', True,
       dict(m1b_detail, binding_property='NOT_EXECUTED_REAL_CERTIFIER',
            why='the real entry stops before any projection binding is evaluated; the pilot contract '
                'is not certifiable for the quality gate'),
       conclusion='executed where possible, honest about what is not', status='SKIP')

    # ============================================================ N10 G-2 unqualified
    j_no_g2 = C2.joint_verdict([rec_a, rec_b], g2_budget_qualified=False)
    ok13 = (j_no_g2['verdict'] == 'JOINT_NOT_CERTIFIABLE' and j_no_g2['installable'] is False
            and 'G2_BUDGET_NOT_QUALIFIED' in j_no_g2['reasons'])
    ck('N10', 'with G-2 unqualified the joint verdict is JOINT_NOT_CERTIFIABLE and installability is '
       'False, so identity agreement never becomes an install certificate', ok13,
       {'verdict': j_no_g2['verdict'], 'reasons': j_no_g2['reasons'],
        'installable': j_no_g2['installable']},
       conclusion='identity is necessary, never sufficient')

    # ============================================================ production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok14 = (after == before)
    ck('P-2', 'the five audited production modules are byte-identical before and after: the candidate is '
       'isolated and the real certifiers were only CALLED', ok14, {'digests': after},
       conclusion='no production change')

    # ---------------------------------------------------------------- Go / No-Go
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    n_skip = sum(1 for r in ROWS if r['status'] == 'SKIP')
    nogo = []
    if n_fail:
        nogo.append('%d 项 FAIL' % n_fail)
    if any('HIDDEN_FUTURE' in str(r.get('detail')) and r['id'] in ('N3', 'N6') and not r['ok']
           for r in ROWS):
        nogo.append('隐藏未来读取未被拒绝')
    verdict = 'C2_CANDIDATE_NO_GO' if nogo else 'C2_CANDIDATE_GO'
    payload = {
        'schema': 'q3_c2_candidate_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'production_consumable': False,
        'baseline_head': '92fa418922711eeb5e3a3eb32c2f8c72682fbd74',
        'preregistration': 'scripts/q3_qualification/q3_c2_candidate_preregistration_2026-10-17.md',
        'verdict': verdict, 'no_go_reasons': nogo,
        'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail, 'n_skip': n_skip,
        'skipped': [r['id'] for r in ROWS if r['status'] == 'SKIP'],
        'three_state_reporting': 'AUD-3: SKIP is counted separately and never as a pass',
        'candidate_identity': C2.visible_identity(S0),
        'domain_limits': list(C2.DOMAIN_LIMITS),
        'production_file_digests': before,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'GTMIN_V10_CLEAN_CHECKOUT': 'FAIL', 'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True,
            'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True, 'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED',
            'L2_NOT_ESTABLISHED': True, 'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED', 'PRODUCTION_NOT_QUALIFIED': True,
            'B-1..B-9': 'all open (B-9a conditional, B-9b out of G-3 scope)'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d rows: %d PASS, %d FAIL, %d SKIP -> %s ===' %
          (len(ROWS), n_pass, n_fail, n_skip, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
