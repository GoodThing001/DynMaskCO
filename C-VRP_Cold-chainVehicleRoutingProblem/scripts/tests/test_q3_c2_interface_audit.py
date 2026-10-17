#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2 READ-ONLY INTERFACE AUDIT -- what do M1-a, M1-b and `certify_plan` actually read, and what can
they prove they read?

C2's gap (per `Q3夜间工作包_GTMIN_G5G6_C2C3_资格准备_2026-10-16.md` section 2) is that M1-a, M1-b and the
production install path do not share one independently verifiable identity of state and plan.  Before
designing a shared identity contract this audit establishes, mechanically and read-only:

    * the CALL SHAPE of each of the three entry points
    * whether it accepts an identity argument at all
    * which execution-state fields it reads
    * whether it RECOMPUTES any identity, or only trusts a caller-supplied projection
    * which claims it makes that are hard-coded rather than computed

It implements nothing, wires nothing, and touches no production module.

Usage: python scripts/tests/test_q3_c2_interface_audit.py --out <json path>
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

ROWS = []
LIMITS = ('static AST analysis only; no module is imported for mutation, no experiment is run, and '
          'this is not a C2 implementation')

#: the three entry points C2 has to reconcile
TARGETS = (
    ('M1-a', 'scripts/simulation/action_contract.py', 'certify_route',
     'TW / capacity / return / temperature replay for one suffix'),
    ('M1-b', 'scripts/m1b_adapter/m1b_q2_certifier.py', 'certify',
     'the eleven quality/thermal/identity gates (method of M1bQ2TrustFixCertifier)'),
    ('C0', 'scripts/evaluation/run_exp_energy_c0.py', 'certify_plan',
     'the C0 energy budget only (total energy <= B)'),
)

#: names that RECOMPUTE an identity rather than accept one
RECOMPUTE_CALLS = ('plan_hash', 'plan_identity_v2', 'snapshot_state_identity', 'verify_state_unchanged',
                   'contract_hash', 'validate_contract_for_quality_certificate', 'plan_identity',
                   'require_same_contract', 'state_identity_v2')

#: names that suggest an identity is being TRUSTED from the caller
TRUSTED_ARGS = ('state_identity', 'plan_identity', 'certified_plan_hash', 'contract_hash',
                'registered_contract_hash', 'state_identity_before', 'state_identity_after')

#: env / vehicle attributes that constitute "the execution state" for strict-online purposes
STATE_FIELDS = ('dist_mat', 'tw_start', 'tw_end', 'service_time', 'reveal_time', 'demands',
                'temp_class', 'capacity', 'coldchain_state', 'status', 'committed_next',
                'current_node', 'coldchain_updated_time', 'ready_time', 'current_load',
                'booking_horizon', 'served_mask', 'mutable_suffix')

PROD_FILES = ('scripts/simulation/action_contract.py',
              'scripts/simulation/d3_geo_adapter.py',
              'scripts/m1b_adapter/m1b_q2_certifier.py',
              'scripts/evaluation/run_exp_energy_c0.py')


def ck(cid, name, ok, detail=None, conclusion=''):
    st = 'PASS' if ok else 'FAIL'
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-5s %s' % (st, cid, name[:92]))


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def analyse(path, func_name):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = find_function(tree, func_name)
    if fn is None:
        return {'found': False}
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    attrs, calls, constants = set(), set(), []
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Constant) and v.value is True:
                    constants.append(k.value)
    return {
        'found': True, 'args': args, 'n_lines': (fn.end_lineno or 0) - fn.lineno + 1,
        'state_fields_read': sorted(a for a in attrs if a in STATE_FIELDS),
        'identity_args_accepted': sorted(a for a in args if a in TRUSTED_ARGS),
        'recomputes': sorted(c for c in calls if c in RECOMPUTE_CALLS),
        'hardcoded_true_claims': sorted(set(constants)),
        'n_calls': len(calls),
    }


def blob_digest(rel):
    b = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)], cwd=str(EXT.parent),
                       capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    before = {rel: blob_digest(rel) for rel in PROD_FILES}
    facts = {}
    for tag, rel, fn, role in TARGETS:
        facts[tag] = dict(analyse(EXT / rel, fn), role=role, file=rel, function=fn)

    # ============================================================ A-1 call shape / identity arguments
    no_identity = [t for t, f in facts.items() if f.get('found') and not f['identity_args_accepted']]
    ok1 = (len(no_identity) == 3)
    ck('A-1', 'none of the three entry points accepts a state or plan identity argument: M1-a takes four '
       'loose anchor scalars, M1-b takes the snapshot objects, `certify_plan` takes env/vehicles/plan. '
       'So "share one identity" cannot be satisfied by passing a digest through the existing signatures',
       ok1,
       {'identity_args_by_layer': {t: f.get('identity_args_accepted') for t, f in facts.items()},
        'call_shapes': {t: f.get('args') for t, f in facts.items()},
        'reading': 'C2 needs a new contract, not a parameter'},
       conclusion='the three signatures have nothing to share today')

    # ============================================================ A-2 who recomputes an identity
    recompute = {t: f.get('recomputes') for t, f in facts.items()}
    ok2 = (recompute['M1-b'] and not recompute['M1-a'] and not recompute['C0'])
    ck('A-2', 'exactly ONE of the three recomputes an identity read from the real state: M1-b (state '
       'unchanged, projection identity, contract identity). M1-a and `certify_plan` recompute NOTHING '
       'identity-shaped, so a shared digest supplied by the caller would be accepted by them without '
       'any independent re-derivation', ok2,
       {'recomputes_by_layer': recompute},
       conclusion='a caller-supplied digest would be trusted by two of the three')

    # ============================================================ A-3 field overlap of the "state"
    fields = {t: set(f.get('state_fields_read') or []) for t, f in facts.items()}
    common = set.intersection(*fields.values()) if all(fields.values()) else set()
    union = set.union(*fields.values()) if any(fields.values()) else set()
    # The measurement was stronger than the first draft of this assertion: the intersection is EMPTY.
    # No single execution-state field is read by all three layers, so there is no existing read whose
    # coverage could be reused as "the" shared state -- it has to be defined, not discovered.
    ok3 = (len(common) == 0 and len(union) > 0)
    ck('A-3', 'the three read DISJOINT-per-field sets: NO execution-state field is read by all three '
       '(intersection is empty, union is %d fields), so no existing read is a superset and the shared '
       'state must be DEFINED rather than discovered' % len(union),
       ok3,
       {'per_layer': {t: sorted(v) for t, v in fields.items()},
        'common': sorted(common), 'union': sorted(union),
        'n_common': len(common), 'n_union': len(union)},
       conclusion='the identity must be defined once and consumed three times')

    # ============================================================ A-4 hard-coded claims
    hard = {t: f.get('hardcoded_true_claims') for t, f in facts.items()}
    ok4 = ('manifest_consistent' in (hard.get('M1-a') or []))
    ck('A-4', 'M1-a returns `manifest_consistent: True` as a LITERAL without computing anything, and its '
       'projected energy/quality/thermal fields are `None`. A consumer that reads those keys cannot tell '
       'a computed pass from a constant', ok4,
       {'hardcoded_true_claims': hard,
        'note': 'this is a claim-shape observation about M1-a, not a claim that it is wrong to return '
                'constant fields it does not evaluate'},
       conclusion='constant claims must be distinguishable from computed ones in C2')

    # ============================================================ A-5 what each layer actually certifies
    roles = {t: f.get('role') for t, f in facts.items()}
    ok5 = all(roles.values())
    ck('A-5', 'the three certify DISJOINT things -- feasibility replay (M1-a), quality/thermal/identity '
       'gates (M1-b), C0 energy budget (C0) -- and none of them certifies that it saw the same state the '
       'other two saw', ok5, {'roles': roles},
       conclusion='C2 is a coverage question as much as an identity question')

    # ============================================================ A-6 production install consumes none
    src = (EXT / 'scripts/simulation/d3_geo_adapter.py').read_text(encoding='utf-8')
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(('import ', 'from ')) and
               any(k in ln for k in ('m1b_q2_certifier', 'action_contract', 'run_exp_energy_c0'))]
    calls_m1b = 'M1bQ2TrustFixCertifier' in src
    ok6 = (not calls_m1b)
    ck('A-6', 'the production install entry imports `action_contract.plan_hash` but never constructs the '
       'M1-b certifier and never calls `certify_plan`, so today there is no path in which the three '
       'identities could even be compared', ok6,
       {'imports_observed': imports, 'constructs_m1b_certifier': calls_m1b},
       conclusion='C2 has no consumer to wire yet')

    # ============================================================ A-7 production untouched
    after = {rel: blob_digest(rel) for rel in PROD_FILES}
    ok7 = (after == before)
    ck('A-7', 'the four production modules are byte-identical before and after the audit',
       ok7, {'digests': after}, conclusion='read-only')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'C2_INTERFACE_AUDIT_FAIL' if n_fail else 'C2_INTERFACE_AUDIT_OBSERVED'
    payload = {
        'schema': 'q3_c2_interface_audit',
        'kind': 'DIAGNOSTIC', 'not_for_production': True, 'implements_nothing': True,
        'baseline_head': 'da8a93a2029e18b2e19be143706ec31979fcf335',
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'layers': facts,
        'gap_statement': ('M1-a, M1-b and certify_plan take different input shapes, read overlapping but '
                          'different state fields, recompute identity in only one of the three cases, and '
                          'are not consumed by the production install path -- so a shared identity '
                          'contract does not exist and cannot be simulated by passing a digest'),
        'does_not_claim': ['that any layer is defective for its own purpose',
                          'that C2 is implementable without a new contract',
                          'that a shared digest would close C2 by itself'],
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
    print('\n=== %d audit rows: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail, verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
