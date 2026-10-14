#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IMPORT PROOF for a D3 freeze tree (read-only; not a test).

For every module file present in `<tree>/scripts/{simulation,tests}`, import it by stem with the
TREE first on `sys.path` and print the resolved `__file__` plus its sha256.  A row is
`FROM_TREE` only when the imported file really is the copy inside the tree -- that is the proof
that the frozen bytes, not the worktree bytes, are what executed.

The worktree's `scripts/` directory is appended as a fallback so that non-frozen helper
modules (coldchain_*, project_paths, ...) still resolve; those show up as `FROM_WORKTREE`.

Exit code is 0 when every module in the tree resolved FROM_TREE, 1 otherwise (with the failures
printed).  Import failures are reported as `IMPORT_FAILED` and are not silently skipped.

Usage:
  python scripts/tests/d3_fixvalid_import_proof.py --tree <freeze_tree_dir> \
      --worktree-scripts <C-VRP.../scripts> [--json <out.json>]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--worktree-scripts', required=True)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    tree = Path(args.tree).resolve()
    wt_scripts = Path(args.worktree_scripts).resolve()
    tree_sim, tree_tests = tree / 'scripts' / 'simulation', tree / 'scripts' / 'tests'

    rows = []
    for d in (tree_sim, tree_tests):
        if not d.is_dir():
            rows.append({'module': '<missing dir>', 'dir': str(d), 'status': 'MISSING_DIR'})
    for d in (tree_sim, tree_tests):
        if not d.is_dir():
            continue
        for f in sorted(d.glob('*.py')):
            rows.append({'module': f.stem, 'tree_file': str(f),
                         'tree_sha256': sha256_of(f), 'status': None})

    sys.path.insert(0, str(wt_scripts))
    sys.path.insert(0, str(wt_scripts / 'tests'))
    sys.path.insert(0, str(tree_tests))
    sys.path.insert(0, str(tree_sim))

    for row in rows:
        if row.get('status'):
            continue
        name = row['module']
        sys.modules.pop(name, None)
        try:
            mod = __import__(name)
        except BaseException as exc:
            row['status'] = 'IMPORT_FAILED'
            row['error'] = '%s: %s' % (type(exc).__name__, exc)
            continue
        resolved = Path(getattr(mod, '__file__', '<none>')).resolve()
        row['resolved_file'] = str(resolved)
        row['resolved_sha256'] = sha256_of(resolved)
        if resolved == Path(row['tree_file']).resolve():
            row['status'] = 'FROM_TREE'
        elif str(resolved).startswith(str(tree)):
            row['status'] = 'FROM_TREE_UNEXPECTED_PATH'
        elif str(resolved).startswith(str(wt_scripts)):
            row['status'] = 'FROM_WORKTREE'
        else:
            row['status'] = 'FROM_ELSEWHERE'

    n_tree = sum(1 for r in rows if r['status'] == 'FROM_TREE')
    n_expected = sum(1 for r in rows if r['status'] != 'MISSING_DIR')
    constants = {}
    try:
        import d3_geo_adapter as G
        constants = {'d3_geo_adapter.__file__': str(Path(G.__file__).resolve()),
                     'EPS_D_PRIMARY_REL': getattr(G, 'EPS_D_PRIMARY_REL', '<missing>'),
                     'EPS_D_PRIMARY_ABS': getattr(G, 'EPS_D_PRIMARY_ABS', '<missing>'),
                     'TIE_BREAK_LEGAL_ORDER': getattr(G, 'TIE_BREAK_LEGAL_ORDER', '<missing>'),
                     'TIE_BREAK_CANONICAL': getattr(G, 'TIE_BREAK_CANONICAL', '<missing>'),
                     'INSTALL_REQUIRES_CERTIFICATION':
                         getattr(G, 'INSTALL_REQUIRES_CERTIFICATION', '<missing>')}
    except BaseException as exc:
        constants = {'error': '%s: %s' % (type(exc).__name__, exc)}

    print('tree            : %s' % tree)
    print('worktree scripts: %s' % wt_scripts)
    print('-' * 100)
    for r in rows:
        print('%-9s %-34s %s' % (r['status'], r['module'],
                                 r.get('resolved_file', r.get('tree_file', ''))))
    print('-' * 100)
    print('modules imported FROM_TREE: %d / %d' % (n_tree, n_expected))
    print('constants: %s' % json.dumps(constants, sort_keys=True))
    bad = [r for r in rows if r['status'] not in ('FROM_TREE', 'MISSING_DIR')]
    for r in bad:
        print('NOT_FROM_TREE: %s -> %s (%s)' % (r['module'], r['status'],
                                                r.get('error', r.get('resolved_file'))))
    print('VERDICT: %s' % ('ALL_FROM_TREE' if not bad else 'IMPORT_PROOF_FAILED'))

    if args.json:
        Path(args.json).write_text(json.dumps({'tree': str(tree), 'rows': rows,
                                               'n_from_tree': n_tree,
                                               'n_expected': n_expected,
                                               'constants': constants,
                                               'bad': bad,
                                               'verdict': 'ALL_FROM_TREE' if not bad
                                               else 'IMPORT_PROOF_FAILED'},
                                              indent=2, sort_keys=True), encoding='utf-8')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
