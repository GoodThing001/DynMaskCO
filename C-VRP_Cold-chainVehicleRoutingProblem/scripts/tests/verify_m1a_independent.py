#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT READ-ONLY VERIFICATION of the M1-a prototype and its S-1..S-17 results.

Deliberately does NOT import `m1a_certification_adapter` or
`test_m1a_synthetic_acceptance`.  It re-derives its conclusions from:
  * the acceptance JSON on disk (recorded verdicts and details),
  * the SOURCE TEXT of the adapter and the test (read only, via `ast` and `re`,
    never executed),
  * byte hashes of the files that must be unchanged.

It is a verifier, not a second implementation: it can falsify the recorded claims, and it
reports disagreement rather than repairing it.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
ADAPTER = EXT / 'scripts' / 'm1a_adapter' / 'm1a_certification_adapter.py'
TESTFILE = EXT / 'scripts' / 'tests' / 'test_m1a_synthetic_acceptance.py'
ENGINE = EXT / 'scripts' / 'simulation' / 'd3_composed_certifier.py'

# The composer as it stood when the M1-a erratum was written.  If this changes, the recorded
# defect analysis must be re-derived rather than assumed.
ENGINE_SHA_AT_ERRATUM = 'b65ddf6545a84b656cd2e9dd541cea266d2d6bdbc1cd56abd76f7780104aea01'

ROWS = []


def chk(cid, name, verdict, detail=None):
    ROWS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-7s] %-52s %s' % (cid, name[:52], verdict))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap_ok = len(sys.argv) > 1
    acceptance = Path(sys.argv[1]) if ap_ok else EXT / 'results' / '_m1a_synth' / 'acceptance.json'

    # ------------------------------------------------------------------ V1 engine identity
    engine_now = sha(ENGINE)
    chk('V1', 'composer unchanged since the erratum was written',
        'ENGINE_UNCHANGED' if engine_now == ENGINE_SHA_AT_ERRATUM else 'ENGINE_CHANGED',
        {'sha256': engine_now, 'expected': ENGINE_SHA_AT_ERRATUM})

    # ------------------------------------------------------------------ V2 defect still present
    src = io.open(ENGINE, encoding='utf-8').read().splitlines()
    if_line = next((i + 1 for i, L in enumerate(src) if re.match(r'^    if legality_ok:\s*$', L)), None)
    else_line = next((i + 1 for i, L in enumerate(src) if re.match(r'^    else:\s*$', L)
                      and i + 1 > (if_line or 0)), None)
    q_branch = next((i + 1 for i, L in enumerate(src)
                     if re.match(r'^    if quality_check_fn is None:\s*$', L)), None)
    q_call = next((i + 1 for i, L in enumerate(src)
                   if 'qok, qdetail = quality_check_fn(' in L), None)
    q_gaps = next((i + 1 for i, L in enumerate(src)
                   if re.match(r'^\s+blocking_gaps = \[\] if qok else \[\]\s*$', L)), None)
    defect = (None not in (if_line, q_branch, q_call) and q_branch > if_line)
    chk('V2', 'composer quality branch is OUTSIDE the legality gate (defect present)',
        'DEFECT_PRESENT' if defect else 'DEFECT_ABSENT_OR_MOVED',
        {'if_legality_ok_line': if_line, 'else_line': else_line,
         'quality_branch_line': q_branch, 'quality_call_line': q_call,
         'blocking_gaps_line': q_gaps,
         'blocking_gaps_text': src[q_gaps - 1].strip() if q_gaps else None,
         'reading': 'quality branch at %s > if legality_ok at %s, so it is NOT gated' % (q_branch, if_line)})

    # ------------------------------------------------------------------ V3 stage-1 arity via AST
    tree = ast.parse(io.open(ADAPTER, encoding='utf-8').read())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id == 'compose':
            calls.append(node)
    stage1_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Call) and isinstance(fn.func, ast.Attribute) and fn.func.attr == '_compose':
            kw = {k.arg: k.value for k in node.keywords}
            qcf = kw.get('quality_check_fn')
            pq = kw.get('project_quality')
            stage1_calls.append({
                'quality_check_fn_is_None_literal': isinstance(qcf, ast.Constant) and qcf.value is None,
                'project_quality_is_False_literal': isinstance(pq, ast.Constant) and pq.value is False,
                'base_plans_kwarg': 'base_plans' in kw})
    chk('V3', 'stage 1 passes quality_check_fn=None and project_quality=False (AST)',
        'STAGE1_CORRECT' if (len(stage1_calls) == 1
                             and stage1_calls[0]['quality_check_fn_is_None_literal']
                             and stage1_calls[0]['project_quality_is_False_literal'])
        else 'STAGE1_WRONG',
        {'n_composer_calls': len(stage1_calls), 'detail': stage1_calls,
         'why_it_matters': 'passing None routes around composer :%s entirely' % q_call})

    # ------------------------------------------------------------------ V4 no truthiness test on qok
    # Scan CODE ONLY.  A regex over raw source is not evidence: the adapter's own comment explains
    # why `if qok:` is forbidden, and a raw-text scan would flag that comment as a violation.
    # Comments and string literals are therefore removed via the AST before matching.
    class _StripDocstrings(ast.NodeTransformer):
        def visit_FunctionDef(self, node):                            # noqa: N802
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
            return self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Module = visit_FunctionDef

    stripped = _StripDocstrings().visit(ast.parse(io.open(ADAPTER, encoding='utf-8').read()))
    ast.fix_missing_locations(stripped)
    code_only = ast.unparse(stripped)          # docstrings and comments are both gone
    bad_patterns = [r'if\s+qok\s*:', r'bool\(\s*qok\s*\)', r'qok\s*==\s*True']
    found = [p for p in bad_patterns if re.search(p, code_only)]
    has_type_check = bool(re.search(r'type\(qok\)\s+is\s+not\s+bool', code_only))
    chk('V4', 'no truthiness test on qok in EXECUTABLE code; "type(qok) is bool" present',
        'T1_HOLDS' if (has_type_check and not found) else 'T1_VIOLATED',
        {'forbidden_patterns_found': found, 'has_type_is_bool': has_type_check,
         'method': 'AST docstring-stripped + ast.unparse (comments excluded)',
         'why_not_raw_regex': 'a raw scan matches the COMMENT that forbids truthiness'})

    # ------------------------------------------------------------------ V5 recorded verdicts
    data = json.loads(acceptance.read_text(encoding='utf-8'))
    res = {r['cid']: r for r in data['results']}
    expected = {
        'S-1': 'PASS', 'S-2': 'FULL_INSTALLABILITY_NOT_AVAILABLE', 'S-3': 'PASS',
        'S-4': 'PASS', 'S-5': 'PASS', 'S-6': 'PASS', 'S-7': 'PASS', 'S-8': 'PASS',
        'S-9': 'PASS', 'S-10': 'PASS', 'S-11': 'PASS', 'S-12': 'PASS', 'S-13': 'PASS',
        'S-14': 'PASS', 'S-15A': 'KNOWN_DEFECT_CONFIRMED',
        'S-15B': 'ADAPTER_LEGALITY_FIRST_HOLDS', 'S-16': 'PASS', 'S-17': 'PASS',
        'ENG': 'ENGINE_UNMODIFIED'}
    mismatch = {k: {'expected': v, 'recorded': res.get(k, {}).get('verdict')}
                for k, v in expected.items() if res.get(k, {}).get('verdict') != v}
    chk('V5', 'all recorded verdicts match the pre-registered expectations',
        'VERDICTS_MATCH' if not mismatch else 'VERDICT_MISMATCH', {'mismatches': mismatch})

    # ------------------------------------------------------------------ V6 the two load-bearing claims
    s15a = res.get('S-15A', {}).get('detail', {})
    s15b = res.get('S-15B', {}).get('detail', {})
    chk('V6', 'defect reproduced AND adapter holds (independent of each other)',
        'BOTH_CONFIRMED' if (s15a.get('calls', 0) >= 1 and s15b.get('adapter_quality_calls') == 0)
        else 'NOT_CONFIRMED',
        {'direct_injection_calls': s15a.get('calls'),
         'adapter_quality_calls': s15b.get('adapter_quality_calls'),
         'note': 'the defect call count must NOT be used as evidence the adapter is fixed'})

    # ------------------------------------------------------------------ V7 status claim scope
    chk('V7', 'declared status is the ENGINEERING status only',
        'STATUS_SCOPE_CORRECT'
        if (data.get('status') == 'M1A_SYNTHETIC_ENGINEERING_ACCEPTED'
            and data.get('is_engineering_status_only') is True
            and 'FULL_INSTALLABILITY_ESTABLISHED' in (data.get('does_not_establish') or []))
        else 'STATUS_SCOPE_WRONG',
        {'status': data.get('status'), 'does_not_establish': data.get('does_not_establish')})

    # ------------------------------------------------------------------ V8 no model / no install
    tsrc = io.open(TESTFILE, encoding='utf-8').read()
    forbidden = {'model forward': r'\.forward\(|model\(', 'install call': r'run_repair_transaction|'
                 r'InstallLedger|install_fn',
                 'ckpt load': r'\.ckpt', 'real rollout': r'env\.run\('}
    hits = {k: bool(re.search(v, tsrc)) for k, v in forbidden.items()}
    # env.run() IS legitimate here: it drives the frozen deterministic baseline to capture
    # decision states, exactly as the existing D3 tests do.  It is not a learned-policy rollout.
    chk('V8', 'no model forward, no install, no checkpoint load',
        'CLEAN' if not hits['model forward'] and not hits['install call'] and not hits['ckpt load']
        else 'FORBIDDEN_REFERENCE',
        {**hits, 'env_run_note': 'env.run() drives the frozen baseline for fixture capture only'})

    # ------------------------------------------------------------------ V9 identities
    chk('V9', 'artifact identities recorded',
        'IDENTITIES_RECORDED',
        {'adapter_sha256': sha(ADAPTER), 'test_sha256': sha(TESTFILE),
         'acceptance_sha256': sha(acceptance), 'composer_sha256': engine_now})

    bad = [r for r in ROWS if r['verdict'] in ('ENGINE_CHANGED', 'DEFECT_ABSENT_OR_MOVED',
                                               'STAGE1_WRONG', 'T1_VIOLATED', 'VERDICT_MISMATCH',
                                               'NOT_CONFIRMED', 'STATUS_SCOPE_WRONG',
                                               'FORBIDDEN_REFERENCE')]
    print('\n== independent verification: %s ==' % ('VERIFIED' if not bad else 'DISAGREEMENT'))
    print('  %d checks, %d disagreement(s)' % (len(ROWS), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
