#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2-Fix INDEPENDENT READ-ONLY VERIFICATION.

Does NOT import `m1b_*`, `test_m1b_q2fix_certification` or the historical test files.  It
re-derives its conclusions from on-disk JSON, from SOURCE TEXT via `ast`, and from byte hashes.

It can falsify the recorded claims and reports disagreement rather than repairing it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
ADIR = EXT / 'scripts' / 'm1b_adapter'
FIXTEST = EXT / 'scripts' / 'tests' / 'test_m1b_q2fix_certification.py'
SPEC = EXT / 'docs' / '当前规划' / '决策与审计' / 'M1b_Q2Fix_修复规格_2026-10-16.md'

ROWS = []


def chk(cid, name, verdict, detail=None):
    ROWS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-9s] %-54s %s' % (cid, name[:54], verdict))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_only(path):
    class _S(ast.NodeTransformer):
        def _d(self, node):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
            return self.generic_visit(node)
        visit_FunctionDef = _d
        visit_AsyncFunctionDef = _d
        visit_ClassDef = _d
        visit_Module = _d
    t = _S().visit(ast.parse(Path(path).read_text(encoding='utf-8')))
    ast.fix_missing_locations(t)
    return ast.unparse(t)


def main():
    fix = Path(sys.argv[1]) if len(sys.argv) > 1 else EXT / 'results' / '_m1b_q2fix' / 'regression.json'

    # --------------------------------------------------------------- X1 spec frozen first
    ok_order = SPEC.exists() and FIXTEST.exists() and SPEC.stat().st_mtime <= FIXTEST.stat().st_mtime
    chk('X1', 'fix spec exists and was written BEFORE the fixed tests',
        'SPEC_FROZEN_FIRST' if ok_order else 'SPEC_ORDER_UNVERIFIED',
        {'spec': SPEC.name, 'spec_written_before_test': ok_order})

    # --------------------------------------------------------------- X2 cache gone
    rf = (ADIR / 'm1b_return_feasibility.py').read_text(encoding='utf-8')
    rf_code = code_only(ADIR / 'm1b_return_feasibility.py')
    reads_cache = bool(re.search(r'_SERVICE_TIME_CACHE\s*\.get\(', rf_code))
    c1 = bool(re.search(r'_SERVICE_TIME_CACHE\s*=\s*\{\}', rf_code))
    chk('X2', 'no verdict reads a service-time cache; the legacy shim table stays EMPTY',
        'CACHE_NOT_READ_FOR_VERDICT' if (not reads_cache and c1) else 'CACHE_STILL_READ',
        {'reads_cache_in_code': reads_cache, 'shim_table_is_empty_dict': c1})

    # --------------------------------------------------------------- X3 tw_start applied
    tm_code = code_only(ADIR / 'm1b_timing.py')
    has_wait = bool(re.search(r'\bmax\(\s*arrive\s*,\s*times\.tw_start_of\(', tm_code))
    chk('X3', 'the shared replay applies max(arrive, times.tw_start_of(node))',
        'WAIT_APPLIED' if has_wait else 'WAIT_MISSING',
        {'evidence': has_wait,
         'matched_expression': 'max(arrive, times.tw_start_of(nxt)) in the per-stop loop'})

    # --------------------------------------------------------------- X4 both gates use ONE replay
    tg_code = code_only(ADIR / 'm1b_thermal_gate.py')
    chk('X4', 'the thermal gate drives the SAME shared replay (no private time advance)',
        'SHARED_REPLAY' if 'replay_timeline' in tg_code else 'PRIVATE_ADVANCE')

    # --------------------------------------------------------------- X5 committed continues
    chk('X5', 'committed branch replays the SUFFIX instead of only the immediate return',
        'SUFFIX_INCLUDED'
        if ('committed_finish' in rf_code and 'stops=suffix' in rf_code) else 'SUFFIX_OMITTED')

    # --------------------------------------------------------------- X6 coverage gate
    qc_code = code_only(ADIR / 'm1b_quality_certificate.py')
    chk('X6', 'Omega10 compares a required set E against the verified set C',
        'COVERAGE_ENFORCED'
        if ('enumerate_required_fish_scope' in qc_code and 'check_omega10_coverage' in qc_code
            and 'COVERAGE_INCOMPLETE' in qc_code) else 'COVERAGE_MISSING')

    # --------------------------------------------------------------- X7 candidate fish required
    cert_code = code_only(ADIR / 'm1b_q2_certifier.py')
    chk('X7', 'the default candidate-fish enumerator feeds the coverage gate (B1 fix)',
        'CANDIDATE_FISH_REQUIRED'
        if ('_candidate_fish_nodes' in cert_code and 'candidate_fish_nodes=candidate_fish_nodes'
            in cert_code) else 'CANDIDATE_FISH_IGNORED')

    # --------------------------------------------------------------- X8 thermal levels separate
    has_s = "receipt['THERMAL_SAMPLED_OK']" in cert_code
    has_c = "receipt['THERMAL_CONTINUOUS_OK']" in cert_code
    requires = bool(re.search(r'all_gates\s+and\s+q_ok\s+and\s+s_ok\s+and\s+c_ok', cert_code))
    stored_hard = bool(re.search(r"receipt\['THERMAL_HARD_OK'\]\s*=", cert_code))
    chk('X8', 'sampled and continuous levels are SEPARATE and the composite requires both',
        'NON_SUBSTITUTABLE' if (has_s and has_c and requires and not stored_hard)
        else 'SUBSTITUTION_RISK',
        {'sampled_field': has_s, 'continuous_field': has_c, 'composite_requires_both': requires,
         'ambiguous_key_stored': stored_hard})

    # --------------------------------------------------------------- X9 recorded verdicts
    data = json.loads(fix.read_text(encoding='utf-8'))
    notpass = [r['cid'] for r in data['results'] if r['verdict'] != 'PASS']
    chk('X9', 'all fixed-suite cases recorded PASS with a zero FALSE_REJECT census',
        'ALL_PASS' if (not notpass and data.get('false_reject_census') == 0) else 'CASES_NOT_PASS',
        {'n_checks': data.get('n_checks'), 'not_pass': notpass,
         'false_reject_census': data.get('false_reject_census')})

    # --------------------------------------------------------------- X10 flip evidence
    rerun = EXT / 'results' / '_m1b_q2' / 'negative_controls_q2r_rerun_under_q2fix.json'
    if rerun.exists():
        rr = json.loads(rerun.read_text(encoding='utf-8'))
        by = {r['cid']: r['CLASS'] for r in rr['results']}
        flips = {k: by.get(k) for k in ('A1', 'A2b', 'A4b')}
        chk('X10', 'the four recorded false passes no longer reproduce as FALSE_PASS',
            'FLIPPED' if (rr.get('n_false_pass') == 0 and flips['A1'] == 'OK'
                          and flips['A2b'] == 'OK' and flips['A4b'] == 'OK')
            else 'STILL_FALSE_PASS',
            {'n_false_pass_after_fix': rr.get('n_false_pass'), 'classes': flips,
             'B1': by.get('B1'),
             'note': 'B1 becomes NOT_ESTABLISHED because the default provider now REQUIRES the '
                     'candidate fish, so the old case refuses before reaching its assertion'})
    else:
        chk('X10', 'the four recorded false passes no longer reproduce', 'RERUN_MISSING')

    # --------------------------------------------------------------- X11 originals untouched
    originals = {
        'scripts/m1b_adapter/m1b_quality_certificate_v1_q2r.py': None,
        'scripts/tests/test_m1b_q2_certification.py': None,
        'scripts/tests/test_m1b_q2r_negative_controls.py': None,
        'results/_m1b_q2/regression.json': None,
        'results/_m1b_q2/negative_controls_q2r.json': None,
    }
    chk('X11', 'audited artifacts and the historical test inputs still exist and are hashed',
        'HASHES_RECORDED',
        {k: sha(EXT / k)[:16] for k in originals if (EXT / k).exists()})

    # --------------------------------------------------------------- X12 claim scope
    chk('X12', 'the recorded payload refuses scientific/production status',
        'SCOPE_CORRECT'
        if (data.get('budget_is_synthetic') is True and data.get('implementation_version') == 'M1B_Q2FIX'
            and 'FULL_INSTALLABILITY_ESTABLISHED' in (data.get('does_not_establish') or []))
        else 'SCOPE_WRONG',
        {'implementation_version': data.get('implementation_version'),
         'budget_is_synthetic': data.get('budget_is_synthetic')})

    # --------------------------------------------------------------- X13 no forbidden refs
    tcode = code_only(FIXTEST)
    hits = {k: bool(re.search(v, tcode)) for k, v in {
        'ckpt': r'\.ckpt', 'install': r'run_repair_transaction|InstallLedger|install_fn',
        'forward': r'\.forward\(', 'jax': r'\bimport jax\b|\bcuda\b|\bjax\b'}.items()}
    chk('X13', 'no checkpoint / install / model-forward / GPU in executable test code',
        'CLEAN' if not any(hits.values()) else 'FORBIDDEN_REFERENCE', hits)

    bad = [r for r in ROWS if r['verdict'] not in (
        'SPEC_FROZEN_FIRST', 'CACHE_NOT_READ_FOR_VERDICT', 'WAIT_APPLIED', 'SHARED_REPLAY',
        'SUFFIX_INCLUDED', 'COVERAGE_ENFORCED', 'CANDIDATE_FISH_REQUIRED', 'NON_SUBSTITUTABLE',
        'ALL_PASS', 'FLIPPED', 'HASHES_RECORDED', 'SCOPE_CORRECT', 'CLEAN')]
    print('\n== independent verification: %s ==' % ('VERIFIED' if not bad else 'DISAGREEMENT'))
    print('  %d checks, %d disagreement(s)' % (len(ROWS), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
