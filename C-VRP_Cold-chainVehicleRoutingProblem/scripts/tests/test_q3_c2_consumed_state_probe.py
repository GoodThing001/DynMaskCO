#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 CONSUMED-STATE DIAGNOSTIC -- read-only.  Constructs the counterexample the review asked for.

The question is NOT "can four parties compute the same digest" (they can, trivially).  It is:

    can four parties report the same digest while one of them actually COMPUTES OVER A DIFFERENT
    STATE?

If yes, then the C2 primary control (C1: four equal digests, and all four move together on a one-ulp
change) is satisfied while the property it was supposed to establish is false.  That is the
`CONSUMED_STATE_BOUND_TO_FROZEN_IDENTITY` gap registered as Q-C2-1 in
`q3_c2_preregistration_amendment_01.md`, and this file measures it instead of arguing it.

It also measures the two secondary gaps registered there:
  Q-C2-2  the interface audit's direct-attribute statistics under-report TRANSITIVE reads
          (`_travel` -> env.dist_mat / env.tw_speed is the concrete case)
  Q-C2-3  C0's `certify_plan` takes FULL instance arrays and contains no visibility gate

Nothing here implements C2, wires anything, or modifies a production module.

Usage: python scripts/tests/test_q3_c2_consumed_state_probe.py --out <json path>
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
sys.path.insert(0, str(EXT / 'scripts' / 'q3_qualification'))
sys.path.insert(0, str(EXT / 'scripts' / 'tests'))

import action_contract as AC                                              # noqa: E402
import q3_g3_plan_identity_candidate as G3                                # noqa: E402

ROWS = []
LIMITS = ('read-only; synthetic state objects; the counterexample is constructed by the probe and is '
          'NOT evidence of a production incident; no C2 implementation')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/m1b_adapter/m1b_q2_certifier.py',
              'scripts/evaluation/run_exp_energy_c0.py',
              'scripts/simulation/d3_geo_adapter.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-6s %s' % (st, cid, name[:92]))


def identity_of(plan_mapping):
    """The identity function under discussion: the audited G-3 frozen-plan identity."""
    return G3.plan_identity(G3.deep_freeze_plan(plan_mapping))


def state(t=24.0, load=12.5, suffix=(3, 5)):
    return {0: AC.VehiclePlan(0, 7, t, load, tuple(suffix))}


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def source_has_gate(path, func_name, needle='reveal_time'):
    """Does `func_name` compare reveal_time against a clock to RESTRICT an order set?"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            fn = node
            break
    if fn is None:
        return {'found': False}
    uses, compares_clock = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == needle:
            uses.append(node.lineno)
        if isinstance(node, ast.Compare):
            dumped = ast.dump(node)
            if needle in dumped and 'clock' in dumped:
                compares_clock.append(node.lineno)
    return {'found': True, 'uses': uses, 'compares_against_clock': compares_clock,
            'has_visibility_gate': bool(compares_clock)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES}

    S0 = state()
    S0_MUTATED = state(t=30.0, load=25.0)          # what a party that read a different env would see
    I0 = identity_of(S0)
    I1 = identity_of(S0_MUTATED)

    # ============================================================ Q1-N1 the core counterexample
    # Four parties, each of which REPORTS the identity computed at freeze time and CONSUMES a state.
    reported = {'M1-a': I0, 'M1-b': I0, 'C0': I0, 'install': I0}
    consumed = {'M1-a': S0, 'M1-b': S0, 'C0': S0_MUTATED, 'install': S0}
    consumed_ids = {k: identity_of(v) for k, v in consumed.items()}
    c1_passes = len(set(reported.values())) == 1
    consumed_all_equal = len(set(consumed_ids.values())) == 1
    ok1 = (c1_passes and not consumed_all_equal)
    ck('Q1-N1', 'THE COUNTEREXAMPLE: four parties report byte-identical identities (so the C1 primary '
       'control PASSES) while one of them actually consumes a different state (so the consumed states '
       'are NOT all equal). No hash collision and no digest tampering is needed -- the identity simply '
       'does not bind the object that is used',
       ok1,
       {'reported_identical': c1_passes, 'reported': I0,
        'consumed_identity_by_party': consumed_ids,
        'consumed_all_equal': consumed_all_equal,
        'divergent_party': 'C0',
        'divergent_consumed_identity': consumed_ids['C0'],
        'ulps_of_separation_in_the_divergent_field': 'anchor_time 24.0 -> 30.0 (a material change)',
        'required_by_amendment_01': 'CONSUMED_STATE_BOUND_TO_FROZEN_IDENTITY (Q-C2-1)'},
       conclusion='C1 is necessary and NOT sufficient; Q-C2-1 is a separate requirement')

    # ============================================================ Q1-N2 authorized projection
    # The legitimate variant: a projection that CARRIES the frozen identity it was derived from can be
    # independently checked.  A bare mutated state cannot.
    projection = {'derived_from': I0, 'orders': [3, 5]}
    def check_projection(proj, frozen_plan):
        return proj.get('derived_from') == identity_of(frozen_plan)
    proj_ok = check_projection(projection, S0)
    bare_ok = check_projection({'derived_from': I0, 'orders': [3, 5]}, S0_MUTATED)
    ok2 = (proj_ok is True and bare_ok is False)
    ck('Q1-N2', 'the legitimate variant works: a projection that carries the identity it was derived '
       'from can be independently checked (passes for the true state, FAILS for the mutated one), while '
       'a bare state carries nothing to check. So Q-C2-1 is implementable, not just a wish',
       ok2,
       {'projection_from_true_state': proj_ok, 'projection_claimed_over_mutated_state': bare_ok,
        'reading': 'the binding must travel WITH the object, not alongside it'},
       conclusion='a bound projection is the admissible form of "pass a digest"')

    # ============================================================ Q1-N3 post-hoc mutation
    frozen = G3.deep_freeze_plan(S0)
    pre = G3.plan_identity(frozen)
    try:
        object.__setattr__(frozen.plans[0], 'anchor_time', 30.0)
        bypassed = True
    except Exception:                                                      # noqa: BLE001
        bypassed = False
    post = G3.plan_identity(frozen)
    ok3 = (bypassed and post != pre)
    ck('Q1-N3', 'a post-freeze mutation is invisible to a party that only stored the old digest, and '
       'VISIBLE only if the party re-hashes the frozen body; so requirement (c) of Q-C2-1 (re-hash '
       'before and after) is what makes it detectable',
       ok3,
       {'bypass_succeeded': bypassed, 'identity_before': pre[:16], 'identity_after': post[:16],
        'detectable_without_rehash': False, 'detectable_with_rehash': post != pre},
       conclusion='the re-hash is the detector, not the freeze')

    # ============================================================ Q2-N1 transitive reads
    import test_q3_c2_interface_audit as AUD
    direct = AUD.analyse(EXT / 'scripts/simulation/action_contract.py', 'certify_route')
    travel = AUD.analyse(EXT / 'scripts/simulation/action_contract.py', '_travel')
    dist = AUD.analyse(EXT / 'scripts/simulation/action_contract.py', '_dist')
    indirect = sorted(set(travel.get('state_fields_read', [])) | set(dist.get('state_fields_read', [])))
    missed = sorted(set(indirect) - set(direct.get('state_fields_read', [])))
    ok4 = ('dist_mat' in indirect and 'dist_mat' not in direct.get('state_fields_read', []))
    ck('Q2-N1', 'the interface audit under-reports TRANSITIVE reads: `certify_route` reaches '
       '`env.dist_mat` / `env.tw_speed` only through `_travel`/`_dist`, and those names are absent from '
       'its direct-attribute set -- so the "fields read" figure is a lower bound, exactly as the review '
       'concluded',
       ok4,
       {'certify_route_direct_fields': direct.get('state_fields_read'),
        'travel_fields': travel.get('state_fields_read'),
        'dist_fields': dist.get('state_fields_read'),
        'transitive_only': missed,
        'consequence': 'a layer that declares only its direct reads under-declares what its conclusion '
                       'depends on (requirement d of Q-C2-1 / Q-C2-2)'},
       conclusion='direct statistics are a lower bound, not a dependency proof')

    # ============================================================ Q2-N2 inference scope label
    audit_out = EXT / 'results/_q3_next_20261017/c2/interface_audit.json'
    verdict_name = None
    if audit_out.exists():
        verdict_name = json.loads(audit_out.read_text(encoding='utf-8')).get('verdict')
    ok5 = (verdict_name == 'C2_INTERFACE_AUDIT_OBSERVED')
    ck('Q2-N2', 'the audit artifact is labelled OBSERVED, not PROVEN, so its 7/7 cannot be read as a '
       'complete data-dependency proof; the scope limitation is registered in amendment 01 section 2',
       ok5,
       {'artifact_verdict': verdict_name, 'required_label': 'C2_INTERFACE_AUDIT_OBSERVED',
        'registered_limit': 'STATE_FIELDS / RECOMPUTE_CALLS name-list analysis, no helper recursion'},
       conclusion='the label already carries the limit; the errata states it explicitly')

    # ============================================================ Q3-N1 C0 takes full arrays
    c0_src = (EXT / 'scripts/evaluation/run_exp_energy_c0.py').read_text(encoding='utf-8')
    full_array_reads = [ln.strip() for ln in c0_src.splitlines()
                        if 'env.' in ln and '[inst_idx]' in ln and '=' in ln][:8]
    gate = source_has_gate(EXT / 'scripts/evaluation/run_exp_energy_c0.py', 'certify_plan')
    ok6 = (bool(full_array_reads) and gate.get('has_visibility_gate') is False)
    ck('Q3-N1', 'C0 `certify_plan` takes FULL instance arrays and contains NO visibility gate: '
       '`reveal_time` is read and passed downstream for scheduling, but never compared against `clock` '
       'to restrict the order set, so nothing in the function enforces "only revealed orders"',
       ok6,
       {'full_array_reads': full_array_reads,
        'reveal_time_uses': gate.get('uses'),
        'comparisons_against_clock': gate.get('compares_against_clock'),
        'has_visibility_gate': gate.get('has_visibility_gate'),
        'required_by_amendment_01': 'C0_VISIBLE_PROJECTION_PROVENANCE (Q-C2-3)'},
       conclusion='C0 provenance is unproven at the source level')

    # ============================================================ Q3-N2 indexed set not gated
    # the order set comes from plan/vehicles (caller), never from a visibility filter
    ok7 = ('served_mask' in c0_src and 'reveal_time' in c0_src
           and '[o for o in plan.get' in c0_src)
    ck('Q3-N2', 'the indexed order set is taken from the CALLER-supplied `plan` (and `committed_next`), '
       'not from a visibility filter over `reveal_time`, so a caller can point C0 at orders outside the '
       'visible set and the function has no way to notice',
       ok7,
       {'index_source': 'plan.get(v.vehicle_id, []) and v.committed_next',
        'visibility_filter_present': False,
        'reading': 'this is a source-shape observation, NOT a claim that any caller does it'},
       conclusion='provenance must be established outside C0')

    # ============================================================ production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok8 = (after == before)
    ck('P-1', 'the four production modules are byte-identical before and after the probe',
       ok8, {'digests': after}, conclusion='read-only')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'C2_CONSUMED_STATE_PROBE_FAIL' if n_fail else 'C2_CONSUMED_STATE_GAP_CONFIRMED'
    payload = {
        'schema': 'q3_c2_consumed_state_probe',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'implements_nothing': True,
        'baseline_head': 'fe18e070cb57b53ab2b942450b3613b45a02cf5a',
        'registrations_first': 'scripts/q3_qualification/q3_c2_preregistration_amendment_01.md',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'core_finding': {
            'statement': ('four parties can report identical identities while one consumes a different '
                          'state, so the C1 primary control alone does not establish shared identity'),
            'reported_identity': I0,
            'consumed_identities': consumed_ids,
            'requires': 'CONSUMED_STATE_BOUND_TO_FROZEN_IDENTITY',
        },
        'secondary_findings': {
            'transitive_reads_undercounted': missed,
            'c0_visibility_gate_present': gate.get('has_visibility_gate'),
        },
        'does_not_claim': ['that any production incident occurred',
                           'that C2 is unimplementable',
                           'that C0 leaks future information in any existing run'],
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
    print('\n=== %d probe rows: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
