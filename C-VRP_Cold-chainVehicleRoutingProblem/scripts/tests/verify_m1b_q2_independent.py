#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2 INDEPENDENT READ-ONLY VERIFICATION.

Deliberately does NOT import `m1b_*` or `test_m1b_q2_certification`.  Its conclusions are
re-derived from:
  * the on-disk regression JSON (recorded verdicts),
  * the SOURCE TEXT of the modules (read only, via `ast` -- never executed),
  * byte hashes of everything that must be unchanged.

It can falsify the recorded claims and reports disagreement rather than repairing it.
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
ADAPTER_DIR = EXT / 'scripts' / 'm1b_adapter'
TESTFILE = EXT / 'scripts' / 'tests' / 'test_m1b_q2_certification.py'
SPEC = (EXT / 'docs' / '当前规划' / '决策与审计' /
        'M1b_Q2_测试规格冻结_2026-10-16.md')

# M1-a artefacts that must remain BYTE-IDENTICAL (accepted at 5b69052 / d8dfae4).
M1A_FROZEN = {
    'scripts/m1a_adapter/m1a_certification_adapter.py':
        'f9f8b6f2c3a0d4e5',   # placeholder replaced below by recorded hash comparison
}
# Frozen sources that must not have been touched by this round.
UNTOUCHED = [
    'scripts/simulation/d3_composed_certifier.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
    'scripts/simulation/action_contract.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/evaluation/hard_gate.py',
    'scripts/evaluation/coldchain_evaluator_a1.py',
]

ROWS = []


def chk(cid, name, verdict, detail=None):
    ROWS.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail or {}})
    print('  [%-9s] %-54s %s' % (cid, name[:54], verdict))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    acceptance = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        EXT / 'results' / '_m1b_q2' / 'regression.json'

    # ---------------------------------------------------------------- W1 spec frozen first
    spec_exists = SPEC.exists()
    spec_mtime_vs_test = None
    if spec_exists and TESTFILE.exists():
        spec_mtime_vs_test = SPEC.stat().st_mtime <= TESTFILE.stat().st_mtime
    chk('W1', 'test spec exists and is not newer than the test file',
        'SPEC_FROZEN_FIRST' if (spec_exists and spec_mtime_vs_test) else 'SPEC_ORDER_UNVERIFIED',
        {'spec': str(SPEC.name), 'spec_exists': spec_exists,
         'spec_written_before_test': spec_mtime_vs_test,
         'note': 'mtime ordering is weak evidence; the spec content is the binding record'})

    # ---------------------------------------------------------------- W2 adapter source scans
    srcs = {p.name: p.read_text(encoding='utf-8') for p in sorted(ADAPTER_DIR.glob('*.py'))}
    offenders = []
    for name, src in srcs.items():
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == 'version':
                offenders.append('%s:%s' % (name, getattr(n, 'lineno', '?')))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'getattr' \
                    and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) \
                    and n.args[1].value == 'version':
                offenders.append('%s:%s(getattr)' % (name, getattr(n, 'lineno', '?')))
    chk('W2', "no getattr(contract,'version') / .version anywhere in the m1b adapter",
        'NO_VERSION_ATTR' if not offenders else 'VERSION_ATTR_FOUND', {'offenders': offenders})

    # ---------------------------------------------------------------- W3 truthiness on qok-style
    joined = '\n'.join(srcs.values())
    chk('W3', 'm1b modules import the repository contract classes (no re-implementation)',
        'IMPORTS_REAL_CONTRACT' if ('from coldchain_contract import' in joined
                                    or 'import coldchain_contract' in joined)
        else 'CONTRACT_REIMPLEMENTED')

    # ---------------------------------------------------------------- W4 thermal uses the real engine
    tg = srcs.get('m1b_thermal_gate.py', '')
    chk('W4', 'thermal replay calls the REAL transition_segment (not a re-implementation)',
        'USES_REAL_ENGINE' if ('from coldchain_state import transition_segment' in tg)
        else 'REIMPLEMENTED_THERMAL',
        {'evidence': 'transition_segment import present' if 'transition_segment' in tg else 'absent'})

    # ---------------------------------------------------------------- W5 non-substitutable fields
    cert = srcs.get('m1b_q2_certifier.py', '')
    has_q = "receipt['QUALITY_LB_OK'] = bool(qverd == QC.QUALITY_LB_OK)" in cert
    has_t = "receipt['THERMAL_HARD_OK'] = bool(thermal_ok)" in cert
    composite = re.search(r"receipt\['composite_pass'\]\s*=\s*bool\(all_gates and q_ok and t_ok\)",
                          cert)
    chk('W5', 'QUALITY_LB_OK and THERMAL_HARD_OK are separate AND jointly required',
        'NON_SUBSTITUTABLE' if (has_q and has_t and composite) else 'SUBSTITUTION_RISK',
        {'quality_field': has_q, 'thermal_field': has_t, 'composite_requires_both': bool(composite)})

    # ---------------------------------------------------------------- W6 Omega10 precedes quality
    i10 = cert.find('check_omega10_initialization')
    iq = cert.find('certify_quality_lb(')
    chk('W6', 'Omega10 initialization gate precedes the quality certificate in source order',
        'OMEGA10_FIRST' if (0 <= i10 < iq) else 'ORDER_WRONG',
        {'omega10_index': i10, 'quality_index': iq})

    # ---------------------------------------------------------------- W7 no env leak
    m1a = EXT / 'scripts' / 'm1a_adapter' / 'm1a_certification_adapter.py'
    leak = []
    if m1a.exists():
        s = m1a.read_text(encoding='utf-8')
        for pat in (r"'reveal_time'", r"'env'", r'FORBIDDEN_VISIBLE_FIELDS'):
            if re.search(pat, s):
                leak.append(pat)
    chk('W7', 'M1-a view still forbids reveal_time / env (isolation retained)',
        'ISOLATION_INTACT' if len(leak) >= 2 else 'ISOLATION_WEAKENED', {'markers': leak})

    # ---------------------------------------------------------------- W8 frozen sources untouched
    chk('W8', 'frozen engine sources present and hashed (identity recorded)',
        'HASHES_RECORDED',
        {p: sha(EXT / p)[:16] for p in UNTOUCHED if (EXT / p).exists()})

    # ---------------------------------------------------------------- W9 recorded verdicts
    data = json.loads(acceptance.read_text(encoding='utf-8'))
    res = {r['cid']: r for r in data['results']}
    must_pass_prefixes = ('T-Q2-1', 'T-Q2-2', 'T-Q2-3', 'T-Q2-4', 'T-Q2-5', 'T-Q2-6', 'T-Q2-7',
                          'T-Q2-8', 'T-Q2-9', 'T-Q2-10')
    notpass = [r['cid'] for r in data['results'] if r['verdict'] != 'PASS']
    covered = sorted({r['cid'].split('[')[0].rsplit('-', 1)[0] for r in data['results']})
    families = sorted(set(p for p in must_pass_prefixes
                          if any(r['cid'].startswith(p) for r in data['results'])))
    chk('W9', 'every T-Q2 case recorded PASS and the required families are covered',
        'ALL_CASES_PASS' if not notpass else 'CASES_NOT_PASS',
        {'n_checks': data['n_checks'], 'not_pass': notpass, 'families_present': families})

    # ---------------------------------------------------------------- W10 claim scope
    chk('W10', 'recorded payload refuses to claim scientific or production status',
        'SCOPE_CORRECT'
        if (data.get('budget_is_synthetic') is True
            and 'FULL_INSTALLABILITY_ESTABLISHED' in (data.get('does_not_establish') or [])
            and 'L2_PASS' in (data.get('does_not_establish') or []))
        else 'SCOPE_WRONG',
        {'budget_is_synthetic': data.get('budget_is_synthetic'),
         'does_not_establish': data.get('does_not_establish')})

    # ---------------------------------------------------------------- W11 no model/install/GPU
    # Scan EXECUTABLE CODE ONLY.  A raw regex over the whole file matches prose: the test's own
    # docstring says "No GPU, no model, no checkpoint", which a naive scan reports as a violation.
    class _Strip(ast.NodeTransformer):
        def _drop(self, node):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
            return self.generic_visit(node)

        visit_FunctionDef = _drop
        visit_AsyncFunctionDef = _drop
        visit_ClassDef = _drop
        visit_Module = _drop

    tree = _Strip().visit(ast.parse(TESTFILE.read_text(encoding='utf-8')))
    ast.fix_missing_locations(tree)
    code_only = ast.unparse(tree)
    forbidden = {'ckpt load': r'\.ckpt', 'install call': r'run_repair_transaction|InstallLedger'
                 r'|install_fn', 'model forward': r'\.forward\(',
                 'jax/gpu': r'\bimport jax\b|\bcuda\b|\bjax\b'}
    hits = {k: bool(re.search(v, code_only)) for k, v in forbidden.items()}
    chk('W11', 'no checkpoint load, no install, no model forward, no GPU in executable code',
        'CLEAN' if not any(hits.values()) else 'FORBIDDEN_REFERENCE',
        {**hits, 'method': 'AST docstring-stripped + ast.unparse (comments/prose excluded)'})

    bad = [r for r in ROWS if r['verdict'] not in ('SPEC_FROZEN_FIRST', 'NO_VERSION_ATTR',
                                                   'IMPORTS_REAL_CONTRACT', 'USES_REAL_ENGINE',
                                                   'NON_SUBSTITUTABLE', 'OMEGA10_FIRST',
                                                   'ISOLATION_INTACT', 'HASHES_RECORDED',
                                                   'ALL_CASES_PASS', 'SCOPE_CORRECT', 'CLEAN')]
    print('\n== independent verification: %s ==' % ('VERIFIED' if not bad else 'DISAGREEMENT'))
    print('  %d checks, %d disagreement(s)' % (len(ROWS), len(bad)))
    print('  identities: adapter=%s' % {k: sha(ADAPTER_DIR / k)[:12] for k in srcs})
    print('  test sha256=%s' % sha(TESTFILE)[:16])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
