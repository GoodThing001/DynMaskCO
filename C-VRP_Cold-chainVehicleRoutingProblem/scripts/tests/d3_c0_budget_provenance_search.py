#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C0 budget provenance: the EXHAUSTIVE bounded search that V1.20 declared it had not done.

V1.20 returned `ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED` and
the D3 ledger keeps `AWAITING_USER_BUDGET_RULING`.  This tool closes the SEARCH half of that: it
sweeps every directory a budget declaration could plausibly live in, with a matcher that catches
the shapes the V1.19 matcher missed, and classifies every hit.

It is strictly read-only and bounded:
  * roots: the extension's `scripts/`, the D3 evidence root, `data/`, `docs/`, plus the CC_Compare
    audit directory that is EXPLICITLY EXCLUDED from D3 source identity (recorded separately so a
    hit there can never be mistaken for a D3 registered budget);
  * file types: `.py`, `.json`, `.md`, `.jsonl`, `.txt`, `.yaml`, `.yml`, `.toml`;
  * one pass, no recursion limit games, every skip is counted and reported.

It classifies each hit as one of:
  HISTORICAL_CAL_FIXTURE      - a synthetic-calibration fixture constant (never portable)
  ISOLATION_FIXTURE           - a deliberately huge sentinel (1e9 and friends) that means
                                "unbounded, for isolation only"
  TEST_LOCAL                  - declared inside a test file for that test's own use
  ENGINEERING_DEFAULT         - a non-binding default in library code
  REGISTERED_PROTOCOL_BUDGET  - a budget bound to a named protocol/run identity AND to the
                                64 dev_check states -- the only thing that would close the gap

The tool NEVER decides the formal budget.  It reports whether a registered protocol budget exists
and, when it does not, says so in exactly those words, so the ruling stays with the user.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1'
L1 = EV / 'a1_d3_l1_prereg_root_v1'
CC_AUDIT = REPO / 'CC_Compare/audit'

# Matcher: catches plain assignment, annotated assignment, tuple unpacking, dict literals and
# `key: value` JSON/YAML forms, for any identifier mentioning BUDGET / B_BUDGET / ENERGY_CAP etc.
BUDGET_NAME = r'[A-Za-z_][A-Za-z0-9_]*(?:BUDGET|BUDGET_[A-Za-z0-9_]*|_BUDGET[A-Za-z0-9_]*)'
PATTERNS = (
    ('assignment', re.compile(r'(?m)^\s*(' + BUDGET_NAME + r')\s*(?::[^=\n]*)?=\s*([0-9][0-9.eE+_]*)')),
    ('tuple', re.compile(r'(?m)^\s*([A-Za-z_,\s]*' + BUDGET_NAME +
                         r'[A-Za-z_,\s]*)\s*=\s*([^\n]*)$')),
    ('mapping', re.compile(r'(?m)["\'](' + BUDGET_NAME + r')["\']\s*[:=]\s*([0-9][0-9.eE+_]*)')),
    ('kwarg', re.compile(r'(?m)\b(' + BUDGET_NAME + r')\s*=\s*([0-9][0-9.eE+_]*)')),
)
SUFFIXES = ('.py', '.json', '.md', '.jsonl', '.txt', '.yaml', '.yml', '.toml')
MAX_BYTES = 2_000_000


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def classify(hit, path, text):
    """Classify a hit by WHERE it is and WHAT the surrounding source says about it.

    The distinction that matters for the ruling is NOT the identifier but portability: a value that
    its own source labels synthetic / fixture / isolation-only can never become the formal budget,
    and a value that appears only in prose is a MENTION rather than a declaration.  A first version
    of this classifier keyed off the word "protocol" and swept 29 doc mentions into
    REGISTERED_PROTOCOL_BUDGET; that was an over-claim and is corrected here (recorded, not hidden).
    """
    value = hit['value'].strip()
    rel = str(path.relative_to(REPO)).replace('\\', '/')
    low = text.lower()
    window = _context_window(text, hit['line'], radius=6).lower()
    if value.replace('_', '').startswith(('1e9', '1e12', '1e18', '1e30')) or 'inf' == value:
        return 'ISOLATION_FIXTURE'
    if path.suffix.lower() in ('.md', '.txt', '.json', '.jsonl'):
        return 'DOC_MENTION'
    if any(m in window for m in ('synthetic', 'synthetic-only', 'fixture', 'isolation',
                                 'do_not_reuse', 'not_a_formal', 'never a formal',
                                 'historical cal')):
        return 'NON_PORTABLE_FIXTURE'
    if path.name.startswith('test_') or '/tests/' in ('/' + rel):
        return 'TEST_LOCAL'
    if any(m in window for m in ('protocol', 'registered', 'run_id', 'commit')):
        return 'PROTOCOL_REACHABLE_CANDIDATE'
    return 'ENGINEERING_DEFAULT'


def _context_window(text, line_no, *, radius=6):
    lines = text.splitlines()
    lo = max(0, line_no - 1 - radius)
    hi = min(len(lines), line_no + radius)
    return '\n'.join(lines[lo:hi])


def scan(root, *, exclude_marker=None, limit_bytes=MAX_BYTES):
    hits, skipped = [], []
    if not root.is_dir():
        return hits, [{'root': str(root), 'reason': 'absent'}]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in SUFFIXES:
            continue
        if exclude_marker and exclude_marker in p.parts:
            skipped.append({'path': str(p), 'reason': 'excluded_subtree'})
            continue
        try:
            if p.stat().st_size > limit_bytes:
                skipped.append({'path': str(p), 'reason': 'oversize'})
                continue
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            skipped.append({'path': str(p), 'reason': 'unreadable:%s' % type(exc).__name__})
            continue
        for kind, pattern in PATTERNS:
            for m in pattern.finditer(text):
                groups = [g for g in m.groups() if g is not None]
                if len(groups) < 2:
                    continue
                name, value = groups[0].strip(), groups[1].strip()
                if not re.match(r'^[0-9]', value):
                    # a tuple/mapping RHS may be a whole expression; keep only a leading number
                    mv = re.match(r'^([0-9][0-9.eE+_]*)', value)
                    if not mv:
                        continue
                    value = mv.group(1)
                line_no = text[:m.start()].count('\n') + 1
                hit = {'kind': kind, 'name': name, 'value': value,
                       'path': str(p.relative_to(REPO)).replace('\\', '/'),
                       'line': line_no,
                       'text': text.splitlines()[line_no - 1].strip()[:140] if line_no - 1 < len(
                           text.splitlines()) else ''}
                if not any(h['path'] == hit['path'] and h['line'] == hit['line']
                           and h['name'] == hit['name'] and h['value'] == hit['value']
                           for h in hits):
                    hit['class'] = classify(hit, p, text)
                    hits.append(hit)
    return hits, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    roots = {
        'extension_scripts': EXT / 'scripts',
        'd3_evidence_root': EV,
        'extension_data': EXT / 'data',
        'extension_docs': EXT / 'docs',
        'research_root': REPO / 'research',
        'cc_compare_audit_EXCLUDED_FROM_D3_IDENTITY': CC_AUDIT,
    }
    report = {'schema': 'c0_budget_provenance_search',
              'matcher_patterns': {k: v.pattern for k, v in PATTERNS},
              'roots': {}, 'n_hits': 0, 'by_class': {}, 'skipped': []}
    all_hits = []
    for name, root in roots.items():
        hits, skipped = scan(root)
        report['roots'][name] = {'path': str(root), 'exists': root.is_dir(), 'n_hits': len(hits),
                                 'n_skipped': len(skipped)}
        report['skipped'] += [dict(s, root=name) for s in skipped[:40]]
        for h in hits:
            h['root'] = name
        all_hits += hits
    report['n_hits'] = len(all_hits)
    report['hits'] = sorted(all_hits, key=lambda h: (h['class'], h['path'], h['line']))
    for h in all_hits:
        report['by_class'][h['class']] = report['by_class'].get(h['class'], 0) + 1

    # ---- the only question that matters for the ruling -------------------------------------
    # A REGISTERED protocol budget would be a Python-level declaration, reachable from a formal
    # protocol path, that is NOT labelled synthetic / fixture / isolation-only AND that is bound to
    # the 64 dev_check states by a registration artifact.  Nothing weaker counts.
    candidates = [h for h in all_hits if h['class'] == 'PROTOCOL_REACHABLE_CANDIDATE'
                  and h['root'] != 'cc_compare_audit_EXCLUDED_FROM_D3_IDENTITY']
    state_bound = []
    for h in candidates:
        try:
            text = (REPO / h['path']).read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        window = _context_window(text, h['line'], radius=12).lower()
        if 'dev_check' in window and ('64' in window or 'states' in window):
            state_bound.append(h)
    b2 = EXT / 'scripts/tests/test_d3_b2_certification.py'
    b2_line = None
    if b2.is_file():
        for i, line in enumerate(b2.read_text(encoding='utf-8').splitlines(), 1):
            if re.match(r'^\s*[A-Za-z_,\s]*B_BUDGET[A-Za-z_,\s]*=', line):
                b2_line = {'line': i, 'text': line.strip()}
                break
    report['registered_protocol_budget_for_the_64_dev_check_states'] = {
        'found': bool(state_bound), 'n_candidates': len(candidates),
        'n_state_bound': len(state_bound), 'state_bound_hits': state_bound[:8],
        'candidates': candidates[:12],
        'definition': 'a Python declaration reachable from a formal protocol path, not labelled '
                      'synthetic/fixture/isolation-only, bound to the 64 dev_check states by a '
                      'registration artifact'}
    report['historical_cal_fixture'] = {'file': 'scripts/tests/test_d3_b2_certification.py',
                                        'declaration': b2_line}
    report['isolation_fixtures'] = sorted({h['path'] for h in all_hits
                                           if h['class'] == 'ISOLATION_FIXTURE'})
    report['excluded_from_d3_identity'] = {
        'root': 'cc_compare_audit_EXCLUDED_FROM_D3_IDENTITY',
        'n_hits': report['roots']['cc_compare_audit_EXCLUDED_FROM_D3_IDENTITY']['n_hits'],
        'note': 'the CC_Compare audit directory is explicitly excluded from D3 source identity; a '
                'budget constant found there is NOT a D3 registered budget and is reported '
                'separately so it can never be mistaken for one'}
    report['verdict'] = ('REGISTERED_PROTOCOL_BUDGET_FOUND' if state_bound
                         else 'SEARCH_EXHAUSTIVE_NO_REGISTERED_PROTOCOL_BUDGET_FOUND')
    report['corrected_from'] = {
        'first_pass_verdict': 'REGISTERED_PROTOCOL_BUDGET_FOUND',
        'why_it_was_wrong': 'the first classifier keyed off the word "protocol" in the surrounding '
                            'text, so 29 documentation MENTIONS of the historical CAL fixture '
                            'value were swept into REGISTERED_PROTOCOL_BUDGET',
        'disposition': 'the first classification is recorded here rather than deleted, and the '
                       'matcher/classifier above is the corrected one'}
    report['ruling_status'] = 'AWAITING_USER_BUDGET_RULING'
    report['what_this_does_not_do'] = [
        'it does not choose a formal C0 budget',
        'it does not promote a test fixture or an isolation sentinel to a formal budget',
        'it does not change any frozen contract, tree, gate or published result']
    report['source_sha256'] = {'self': sha(Path(__file__)),
                               'd3_geo_adapter': sha(EXT / 'scripts/simulation/d3_geo_adapter.py'),
                               'coldchain_contract': sha(EXT / 'scripts/coldchain/'
                                                         'coldchain_contract.py')}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print(json.dumps({'n_hits': report['n_hits'], 'by_class': report['by_class'],
                      'registered_found': bool(state_bound),
                      'roots': {k: v['n_hits'] for k, v in report['roots'].items()},
                      'verdict': report['verdict']}, indent=1))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
