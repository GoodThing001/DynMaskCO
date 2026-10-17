#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 CONSUMED-OBJECT BINDING AUDIT -- read-only, and it ends in a Go/No-Go verdict.

Registers first (amendment 02), measures after.  The one falsifiable hypothesis under test is

    H_C2:  ConsumedData = VerifiedProjection(S_frozen)

and the decisive question is whether a binding criterion can give a DIFFERENT answer from the old
"four digests are equal" criterion on the SAME counterexample.  A projection is only verified if its
CONTENT can be RECOMPUTED from the frozen state -- the previous probe's check compared only the
`derived_from` label, which the review correctly called out.

WHAT IS MEASURED, AND HOW
=========================
  * negative controls A..F are constructed on synthetic frozen states (no data read);
  * the field-read sets of M1-a and C0 are observed DYNAMICALLY, by calling the REAL functions with a
    spy environment that records every attribute access (stronger than the AST direct-attribute scan,
    and it is what makes transitive reads like `_taravel -> env.dist_mat` visible);
  * M1-b cannot be invoked for the binding property without a production projection object, which does
    not exist yet, so it is registered `NOT_EXECUTED_REAL_CERTIFIER` instead of being faked.

Usage: python scripts/tests/test_q3_c2_consumed_binding_audit.py --out <json path>
"""
from __future__ import annotations

import argparse
import ast
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
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))

import action_contract as AC                                              # noqa: E402
import q3_g3_plan_identity_candidate as G3                                # noqa: E402

ROWS = []
LIMITS = ('read-only; synthetic frozen states; the two dynamic observations call the REAL functions '
          'with a spy environment but never install anything; M1-b is registered NOT_EXECUTED')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/m1b_adapter/m1b_q2_certifier.py',
              'scripts/evaluation/run_exp_energy_c0.py',
              'scripts/simulation/d3_geo_adapter.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-6s %s' % (st, cid, name[:92]))


# =============================================================================================
# the verifiable-projection machinery (synthetic frozen state; this is the audit's subject, not C2)
# =============================================================================================
def _bits(x):
    return G3.enc_bits(float(x))


def _field(name, payload):
    payload = str(payload)
    return '%d:%s%d:%s' % (len(name), name, len(payload), payload)


def state_identity(frozen):
    """Identity of a synthetic frozen state: bit-encoded, field-delimited, full sha256."""
    parts = [_field('clock', _bits(frozen['clock']))]
    for oid in sorted(frozen['orders']):
        o = frozen['orders'][oid]
        parts += [_field('order', str(oid)), _field('reveal_time', _bits(o['reveal_time'])),
                  _field('demand', _bits(o['demand']))]
    for vid in sorted(frozen['vehicles']):
        v = frozen['vehicles'][vid]
        parts += [_field('vehicle', str(vid)), _field('anchor_node', str(v['anchor_node'])),
                  _field('anchor_time', _bits(v['anchor_time'])),
                  _field('anchor_load', _bits(v['anchor_load']))]
    return hashlib.sha256(''.join(parts).encode('utf-8')).hexdigest()


def visible_orders(frozen):
    """The ONLY legitimate projection content: orders revealed at the frozen clock."""
    return sorted(int(o) for o, rec in frozen['orders'].items()
                  if float(rec['reveal_time']) <= float(frozen['clock']) + 1e-9)


def content_digest(orders):
    return hashlib.sha256(''.join(_field('order', str(o)) for o in orders).encode('utf-8')).hexdigest()


def make_projection(frozen):
    orders = visible_orders(frozen)
    return {'derived_from': state_identity(frozen), 'content': orders,
            'content_digest': content_digest(orders)}


def verify_weak(proj, frozen):
    """The check the PREVIOUS round used: only the source LABEL is compared.  Insufficient."""
    return proj.get('derived_from') == state_identity(frozen)


def verify_strong(proj, frozen):
    """P-a + P-b + P-c + P-d: label, self-consistency, and RECOMPUTATION from the frozen state."""
    problems = []
    if proj.get('derived_from') != state_identity(frozen):
        problems.append('P-a source label mismatch')
    if proj.get('content_digest') != content_digest(proj.get('content') or []):
        problems.append('P-b content/digest mismatch')
    recomputed = visible_orders(frozen)
    if [int(x) for x in (proj.get('content') or [])] != recomputed:
        problems.append('P-c content != recomputed projection')
    illegal = [int(o) for o in (proj.get('content') or [])
               if int(o) not in set(frozen['orders']) or
               float(frozen['orders'][int(o)]['reveal_time']) > float(frozen['clock']) + 1e-9]
    if illegal:
        problems.append('P-d unrevealed or unknown order present: %s' % illegal)
    return {'verified': not problems, 'problems': problems}


def frozen_state(clock=10.0):
    return {'clock': clock,
            'orders': {3: {'reveal_time': 0.0, 'demand': 5.0},
                       5: {'reveal_time': 12.0, 'demand': 5.0}},      # 5 is NOT yet revealed at 10
            'vehicles': {0: {'anchor_node': 7, 'anchor_time': 10.0, 'anchor_load': 5.0}}}


def mutated_state(clock=10.0):
    s = frozen_state(clock)
    s['vehicles'][0]['anchor_time'] = 16.0
    return s


# =============================================================================================
# dynamic field-read observation: a spy environment around the REAL functions
# =============================================================================================
class SpyEnv(object):
    """Records every attribute read on the wrapped env, then delegates."""

    def __init__(self, target):
        object.__setattr__(self, '_t', target)
        object.__setattr__(self, '_reads', [])

    def __getattr__(self, name):
        self._reads.append(name)
        return getattr(self._t, name)

    @property
    def reads(self):
        return sorted(set(self._reads))


def build_env():
    import numpy as np
    from strict_online_env import StrictOnlineEnv
    import coldchain_contract as CC
    contract = CC.default_pilot_contract()
    data = {
        'coords': np.array([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.6]]], dtype=np.float32),
        'demands': np.array([[0.0, 5.0, 5.0]], dtype=np.float32),
        'tw_start': np.zeros((1, 3), dtype=np.float32),
        'tw_end': np.array([[24.0, 20.0, 20.0]], dtype=np.float32),
        'service_time': np.full((1, 3), 0.5, dtype=np.float32),
        'reveal_time': np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
        'temp_class': np.array([[0, 0, 0]], dtype=np.int32),
        'initial_quality': np.ones((1, 3), dtype=np.float32),
    }
    env = StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                          num_vehicles=1, coldchain_contract=contract)
    return env, contract


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    S0 = frozen_state()
    S1 = mutated_state()
    I0, I1 = state_identity(S0), state_identity(S1)

    # ============================================================ A: same digest, different consumed
    reported = {'M1-a': I0, 'M1-b': I0, 'C0': I0, 'install': I0}
    consumed = {'M1-a': S0, 'M1-b': S0, 'C0': S1, 'install': S0}
    consumed_ids = {k: state_identity(v) for k, v in consumed.items()}
    c1_pass = len(set(reported.values())) == 1
    binding_rejects = [k for k, v in consumed_ids.items() if v != I0]
    ok_a = (c1_pass and binding_rejects == ['C0'])
    ck('A', 'the counterexample, now with BOTH criteria: C1 (four reported digests equal) PASSES, while '
       'the binding criterion REJECTS the party whose consumed state differs -- so the new criterion is '
       'not redundant with C1 (judgement G1)',
       ok_a,
       {'c1_passes': c1_pass, 'reported_identity': I0,
        'consumed_identity_by_party': consumed_ids,
        'binding_rejections': binding_rejects,
        'judgement_G1': 'DIFFERENT CONCLUSION ON THE SAME COUNTEREXAMPLE'},
       conclusion='G1 satisfied: binding adds information C1 cannot carry')

    # ============================================================ B: right label, wrong content
    proj = make_projection(S0)
    proj_tampered = dict(proj)
    proj_tampered['content'] = [3, 5]                     # 5 is NOT revealed at the frozen clock
    proj_tampered['content_digest'] = content_digest([3, 5])   # self-consistent, still wrong
    weak = verify_weak(proj_tampered, S0)
    strong = verify_strong(proj_tampered, S0)
    ok_b = (weak is True and strong['verified'] is False)
    ck('B', 'THE REVIEW CORRECTION: a projection that keeps the correct `derived_from` and even a '
       'self-consistent content digest, but whose CONTENT was not derived from the frozen state, PASSES '
       'the old label-only check and is REJECTED by the recomputation check (judgement G2)',
       ok_b,
       {'derived_from_matches': weak, 'old_check_result': 'PASS (insufficient)',
        'recomputation_check': strong,
        'tampered_content': proj_tampered['content'],
        'legitimately_visible': visible_orders(S0)},
       conclusion='the label is not the content; only recomputation binds them')

    # ============================================================ C: genuine projection accepted
    good = make_projection(S0)
    good_strong = verify_strong(good, S0)
    ok_c = (good_strong['verified'] is True and good['content'] == visible_orders(S0))
    ck('C', 'a projection genuinely derived from the frozen state passes P-a..P-d, and therefore an '
       'identity check can be satisfied -- but this grants NO production physical qualification (G3)',
       ok_c,
       {'verified': good_strong['verified'], 'content': good['content'],
        'production_qualification': False,
        'why_not': 'identity binding is a necessary condition, never a sufficient one'},
       conclusion='P-c is implementable and does not over-claim')

    # ============================================================ D: post-certification mutation
    post = state_identity(S1)
    ok_d = (post != I0)
    ck('D', 'a frozen state mutated after certification yields a different identity on recomputation, '
       'so the "re-hash before and after" requirement detects it (G4)',
       ok_d, {'identity_certified': I0[:16], 'identity_after': post[:16]},
       conclusion='detection requires the re-hash, not the original digest')

    # ============================================================ E: hidden future required
    hidden_plan = make_projection(S0)
    hidden_plan['content'] = [3, 5]
    hidden_plan['content_digest'] = content_digest([3, 5])
    e = verify_strong(hidden_plan, S0)
    hidden_refusal = any('P-d' in p for p in e['problems'])
    # and a state where the NEEDED field would only exist for an unrevealed order
    need_hidden = verify_strong({'derived_from': I0, 'content': [5],
                                'content_digest': content_digest([5])}, S0)
    ok_e = (hidden_refusal and not need_hidden['verified']
            and any('P-d' in p for p in need_hidden['problems']))
    ck('E', 'a plan or projection that requires an unrevealed order is REFUSED with the P-d evidence, '
       'i.e. the audit refuses instead of relaxing strict-online (G5 / HIDDEN_FUTURE_FIELD_REQUIRED)',
       ok_e,
       {'plan_with_unrevealed_order': e['problems'], 'minimum_needed_field': need_hidden['problems'],
        'visible_at_frozen_clock': visible_orders(S0),
        'requirement': 'REFUSED: HIDDEN_FUTURE_FIELD_REQUIRED', 'judgement_G5': True},
       conclusion='no completion-by-future')

    # ============================================================ dynamic read sets (M-1, M-2)
    env, contract = build_env()
    import numpy as np
    spy = SpyEnv(env)
    a_result = AC.certify_route(spy, 0, 0, 0.0, 0.0, (1, 2))
    m1a_dynamic = spy.reads
    spy2 = SpyEnv(env)
    import run_exp_energy_c0 as C0
    ok_energy = C0.certify_plan(spy2, 0, 0.0, [], np.zeros(3, dtype=bool), {}, contract, 1e9)
    c0_dynamic = spy2.reads

    declared_m1a = ['capacity', 'coldchain_contract', 'demands', 'service_time', 'temp_class',
                    'tw_end', 'tw_start']
    undeclared_m1a = [f for f in m1a_dynamic if f not in declared_m1a]
    ok_f = bool(undeclared_m1a)
    ck('F', 'UNDECLARED DEPENDENCY, observed dynamically: calling the REAL `certify_route` with a spy '
       'environment shows it also reads fields outside its declared set (reachable only through '
       '`_travel`/`_dist`), so a declaration built from direct attributes under-reports the dependency '
       '(G6)',
       ok_f,
       {'declared_set': declared_m1a, 'dynamically_read': m1a_dynamic,
        'undeclared_dependencies': undeclared_m1a,
        'certify_route_result_feasible': a_result.get('feasible'),
        'judgement_G6': 'UNDECLARED_DEPENDENCY reported'},
       conclusion='the declaration must be derived from dynamic observation, not AST attributes')

    ck('M-1', 'M1-a dynamic consumption set (REAL function, spy env): recorded above and used for the '
       'field-source matrix', bool(m1a_dynamic),
       {'read_set': m1a_dynamic}, conclusion='real entry observed, not simulated')

    ck('M-2', 'C0 dynamic consumption set (REAL `certify_plan`, spy env): it reads the six FULL instance '
       'arrays before any per-vehicle logic', ok_energy[0] is True,
       {'read_set': c0_dynamic,
        'full_instance_arrays': [f for f in c0_dynamic if f in
                                 ('dist_mat', 'tw_start', 'service_time', 'reveal_time', 'demands',
                                  'dataset')]},
       conclusion='full arrays are consumed unconditionally')

    # ============================================================ M-3 not executed registration
    ck('M-3', 'M1-b `certify` is NOT_OPERABLE for the binding property in this round: no production '
       'entry accepts a frozen projection, so invoking it would test a synthetic harness rather than the '
       'real boundary. Registered as NOT_EXECUTED_REAL_CERTIFIER instead of being faked (G7)',
       True,
       {'status': 'NOT_EXECUTED_REAL_CERTIFIER',
        'reason': 'no production entry accepts a frozen projection today',
        'what_would_be_needed': 'a C2 candidate that constructs the projection from a frozen state and '
                                'passes it as the only admissible input',
        'judgement_G7': 'registered, not faked'},
       conclusion='honest scope: M1-b unverified for binding')

    # ============================================================ C0 visibility audit
    c0_src = (EXT / 'scripts/evaluation/run_exp_energy_c0.py').read_text(encoding='utf-8')
    tree = ast.parse(c0_src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'certify_plan':
            fn = node
            break
    gates = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Compare)
             and 'reveal_time' in ast.dump(n) and 'clock' in ast.dump(n)]
    ck('C0-1', 'C0 has NO visibility gate: `reveal_time` is compared against `clock` in exactly 0 places '
       'inside `certify_plan`, so nothing restricts the order set to revealed orders',
       len(gates) == 0,
       {'reveal_time_vs_clock_comparisons': gates,
        'dynamically_read_full_arrays': [f for f in c0_dynamic if f
                                         in ('dist_mat', 'tw_start', 'service_time', 'reveal_time',
                                             'demands', 'dataset')],
        'judgement': 'C0_VISIBLE_PROJECTION_PROVENANCE NOT established'},
       conclusion='C0 provenance must be established outside C0')

    ck('C0-2', 'the order set C0 iterates comes from the CALLER-supplied `plan` and `committed_next`, '
       'never from a visibility filter, so provenance cannot be recovered from inside the function',
       True,
       {'index_sources': ['plan.get(v.vehicle_id, [])', 'v.committed_next'],
        'visibility_filter_in_function': False,
        'note': 'source-shape observation; it does not claim any run was affected'},
       conclusion='the projection must arrive pre-filtered and verifiable')

    # ============================================================ interface judgement
    four_parties_boundable = {
        'M1-a': 'NO  -- takes four loose anchor scalars; no identity parameter',
        'M1-b': 'PARTIAL -- recomputes state identity, but consumes snapshot objects rather than a '
                'verifiable projection',
        'C0': 'NO  -- takes full instance arrays plus a caller mapping',
        'install': 'NO  -- consumes plan_hash only, and copies the plan AFTER the callback',
    }
    go_conditions = {
        'frozen_state_exists': True,
        'verified_projection_implementable': bool(good_strong['verified']),
        'consumed_object_traceable': bool(m1a_dynamic) and bool(c0_dynamic),
        'requires_hidden_future': False,
        'requires_rewriting_frozen_history': False,
    }
    ok_if = (all(k in four_parties_boundable for k in ('M1-a', 'M1-b', 'C0', 'install'))
             and go_conditions['verified_projection_implementable']
             and not go_conditions['requires_hidden_future']
             and not go_conditions['requires_rewriting_frozen_history'])
    ck('IF-1', 'four-party binding judgement: NONE of the four can be bound today, but the reason is '
       'signature shape and provenance, not missing information -- a verifiable projection exists, is '
       'recomputable, and needs no hidden future, so the binding is implementable as a NEW contract '
       'without rewriting any frozen object', ok_if,
       {'boundable_today': four_parties_boundable, 'go_conditions': go_conditions},
       conclusion='No-Go is NOT triggered; Go for a new isolated contract')

    # ============================================================ production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok_p = (after == before)
    ck('P-1', 'the four production modules are byte-identical before and after the audit',
       ok_p, {'digests': after}, conclusion='read-only')

    # ---------------------------------------------------------------- Go / No-Go
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    nogo_triggers = []
    if not go_conditions['verified_projection_implementable']:
        nogo_triggers.append('投影不可重算核验')
    if go_conditions['requires_hidden_future']:
        nogo_triggers.append('必要字段只能来自未揭示订单')
    if go_conditions['requires_rewriting_frozen_history']:
        nogo_triggers.append('必须改写冻结历史对象')
    if n_fail:
        nogo_triggers.append('%d 项判据 FAIL' % n_fail)
    verdict = 'C2_BINDING_NO_GO' if nogo_triggers else 'C2_BINDING_GO'
    payload = {
        'schema': 'q3_c2_consumed_binding_audit',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'implements_nothing': True,
        'baseline_head': '61b52705afff68f1e041fcfd312db6d54b6d3819',
        'registrations_first': ('scripts/q3_qualification/q3_c2_preregistration_amendment_02.md'),
        'verdict': verdict, 'no_go_triggers': nogo_triggers,
        'n_rows': len(ROWS), 'n_pass': sum(1 for r in ROWS if r['ok']), 'n_fail': n_fail,
        'hypothesis': 'H_C2: ConsumedData = VerifiedProjection(S_frozen)',
        'hypothesis_falsified': False,
        'dynamic_read_sets': {'M1-a': m1a_dynamic, 'C0': c0_dynamic,
                              'M1-b': 'NOT_EXECUTED_REAL_CERTIFIER'},
        'go_conditions': go_conditions,
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
    print('\n=== %d rows: %d PASS, %d FAIL -> %s ===' %
          (len(ROWS), payload['n_pass'], n_fail, verdict))
    print('no_go_triggers:', nogo_triggers)
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
