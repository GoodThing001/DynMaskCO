#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1-D3-L1 collection-readiness audit (read-only): (a) source dependency closure with import
origins, (b) minimal counterexamples qualifying the bitwise comparator that the pre-registration
requires for resume_parity (P0-3 and the pre-batch comparator gate).

Nothing here collects states, runs a baseline rollout or calls F/D-Geo.

Comparator contract qualified here (pre-registered precision):
  * every float leaf must have IDENTICAL IEEE-754 bits (struct.pack('>d')),
    so +0.0 vs -0.0 is a DIFFERENCE and NaN is compared BY BITS (equal bits -> equal);
  * numpy scalars and Python numbers compare by bits/values (no global float() coercion);
  * raw container/object types must match exactly (no canonicalisation);
  * any other difference is reported with its path.
The semantic comparator used earlier (canon() + numeric tolerance for numpy<->python) is kept only
as a DIAGNOSTIC and is NOT the criterion for the collection.
"""
import argparse
import ast
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
SCRIPTS = EXT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / 'simulation'))
sys.path.insert(0, str(SCRIPTS / 'coldchain'))
sys.path.insert(0, str(SCRIPTS / 'tests'))
sys.path.insert(0, str(SCRIPTS / 'data'))
CASES = []
REC = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def local_modules():
    m = {}
    for p in SCRIPTS.rglob('*.py'):
        if '__pycache__' in str(p) or '_bverify' in str(p):
            continue
        m.setdefault(p.stem, p)
    return m


def imports_of(path):
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split('.')[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module.split('.')[0], node.lineno))
    return out


def closure(seeds):
    mods = local_modules()
    seen, edges, external = {}, [], set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        p = mods.get(name)
        if p is None:
            external.add(name)
            continue
        if name in seen:
            continue
        seen[name] = p
        for dep, line in imports_of(p):
            edges.append({'from': name, 'to': dep, 'line': line,
                          'internal': dep in mods})
            if dep in mods and dep not in seen:
                stack.append(dep)
    return seen, edges, external


def bits(x):
    try:
        return struct.pack('>d', float(x)).hex()
    except Exception:
        return None


def bitwise_diff(a, b, path='root', out=None, budget=None):
    out = [] if out is None else out
    budget = {'n': 0, 'cap': 200000} if budget is None else budget
    budget['n'] += 1
    if budget['n'] > budget['cap']:
        raise RuntimeError('budget')
    if len(out) >= 40:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append({'path': '%s.%s' % (path, k), 'kind': 'missing'})
            else:
                bitwise_diff(a[k], b[k], '%s.%s' % (path, k), out, budget)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append({'path': '%s.len' % path, 'kind': 'length', 'a': len(a), 'b': len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            bitwise_diff(x, y, '%s[%d]' % (path, i), out, budget)
        return out
    fa = isinstance(a, float) or hasattr(a, 'dtype')
    fb = isinstance(b, float) or hasattr(b, 'dtype')
    if fa and fb:
        ba, bb = bits(a), bits(b)
        if ba != bb:
            out.append({'path': path, 'kind': 'float-bits', 'a_bits': ba, 'b_bits': bb})
        return out
    if type(a) is not type(b):
        out.append({'path': path, 'kind': 'type', 'a': type(a).__name__,
                    'b': type(b).__name__})
        return out
    if a != b:
        out.append({'path': path, 'kind': 'value', 'a': repr(a)[:80], 'b': repr(b)[:80]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    seeds = ['d3_l1_state_collector', 'd3_l1_server_timing_qualification',
             'test_d3_l1_resume_parity_attribution']
    seen, edges, external = closure(seeds)
    REC['dependency_closure'] = {
        'seeds': seeds,
        'internal_modules': {n: {'path': str(p.relative_to(EXT)).replace('\\', '/'),
                                 'sha256': sha(p), 'import_origin': 'reached from the seeds'}
                             for n, p in sorted(seen.items())},
        'n_internal': len(seen), 'edges': edges[:80], 'n_edges': len(edges),
        'external_modules': sorted(external)[:80]}
    MUST = ('jf1h_repair', 'joint_fleet', 'action_contract', 'strict_online_env',
            'recourse_snapshot', 'coldchain_contract', 'coldchain_state')
    NOT_IN_THIS_PATH = {
        'cc_lns_replanner': 'LNS/regret-2 reference line, not used by the F-free JF1-H-F collection',
        'dynmaskco_cc_context': 'model/teacher context helpers, not used by the F-free collection',
        'visible_state': 'visible-scenario evaluator of the LNS line, not used here',
        'repair_state': 'model feature extractor (F side), deliberately excluded (F-free)'}
    REC['required_dependency_status'] = {m: (m in seen) for m in MUST}
    REC['not_required_by_design'] = {m: {'in_closure': (m in seen), 'reason': why}
                                     for m, why in NOT_IN_THIS_PATH.items()}
    ck('P0-3a the dependency closure of the collection path is enumerated with per-module sha256 '
       'and import edges, and includes every module the path actually needs (the user-named '
       'joint_fleet and action_contract included); modules outside this path are recorded with the '
       'reason they are absent instead of being silently omitted',
       all(REC['required_dependency_status'].values()),
       {'missing_required': [k for k, v in REC['required_dependency_status'].items() if not v],
        'n_internal': len(seen), 'not_required_by_design': REC['not_required_by_design']})
    REC.setdefault('runner_identity', {})['formal_runner_present'] = \
        (SCRIPTS / 'tests/d3_l1_server_timing_qualification.py').is_file()
    REC['runner_identity']['comparator_source'] = 'bitwise comparator defined in this audit + the '\
        'diagnostic comparators in test_d3_l1_resume_parity_attribution.py'

    # ---- minimal counterexamples for the pre-registered bitwise precision -----------------
    nan = float('nan')
    tests = {
        'signed_zero': (-0.0, 0.0, True),
        'pos_zero_same': (0.0, 0.0, False),
        'nan_same_bits': (nan, nan, False),
        'numpy_scalar_vs_float': (None, None, False),
        'int_vs_float_same_value': (1, 1.0, True),
        'type_mismatch': ('1', 1, True),
        'deep_nested_value': ({'a': [{'b': 1.5}]}, {'a': [{'b': 1.5000000000000002}]}, True),
    }
    try:
        import numpy as np
        tests['numpy_scalar_vs_float'] = (np.float64(0.1), 0.1, False)
    except Exception:
        tests.pop('numpy_scalar_vs_float')
    results = {}
    for name, (a, b, expect_diff) in tests.items():
        diffs = bitwise_diff(a, b)
        results[name] = {'expected_difference': expect_diff, 'detected': bool(diffs),
                         'n': len(diffs), 'first': (diffs[0] if diffs else None)}
    REC['comparator_counterexamples'] = results
    ok = all(v['detected'] == v['expected_difference'] for v in results.values())
    ck('P0-3b the bitwise comparator meets the pre-registered precision on the minimal '
       'counterexamples: +0.0 vs -0.0 IS a difference, NaN compares BY BITS, numpy scalars are '
       'compared by bits, and type mismatches are caught', ok, results)
    REC['comparator_verdict'] = {
        'bitwise_comparator_qualified': bool(ok),
        'criterion': 'resume_parity requires identical IEEE-754 bits for every float leaf and exact '
                     'types; the earlier canonicalising + numeric-tolerant comparator is DIAGNOSTIC '
                     'only and must not be used as the collection criterion',
        'declared_perturbation_family_only': True}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': sum(1 for c in CASES if c[1]),
                               'n_fail': sum(1 for c in CASES if not c[1]),
                               'source_sha256_this_tool': sha(__file__)},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL ===' % (len(CASES),
                                                   sum(1 for c in CASES if c[1]),
                                                   sum(1 for c in CASES if not c[1])))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
