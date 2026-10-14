#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner for the D3 A2/A3 FIXTURE-VALIDITY round (2026-10-14). Deterministic, re-runnable.

What it does, in order:
  1. builds a NEW freeze tree `freeze_tree_<newsha>/` from the registered 22-file identity of
     `A1_D3_FREEZE_MANIFEST_7b79ac57.json` PLUS the two new files of this round (the
     fixture-validity suite and the import-proof tool), verifying every copied byte;
  2. runs the IMPORT PROOF against the tree (`__file__` and sha256 per module);
  3. re-runs the three FROZEN suites from the TREE and from the WORKTREE, and the new suite in
     both places, capturing raw stdout/stderr per run;
  4. writes A1_D3_FIXVALID_RECORDS.json with exit codes and PASS/FAIL counts.

No experiment, no metric, no threshold, no F-vs-D-Geo comparison. Existing freeze trees and
`MASKCO_code/` are never written.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
EXT = REPO / 'C-VRP_Cold-chainVehicleRoutingProblem'
RAW = ROOT / 'raw'
PY = Path(r'D:\AA_Py\A_Anaconda\python.exe')
if not PY.is_file():
    PY = Path(sys.executable)

PRIOR = (REPO / 'research/review_3080_20261010/delivery/a1_d3_geo_masked_attribution_root_v1'
         / 'A1_D3_FREEZE_MANIFEST_7b79ac57.json')
NEW_FILES = ['C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/'
             'test_d3_assembly_fixture_validity.py',
             'C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/'
             'd3_fixvalid_import_proof.py']
FROZEN_SUITES = ['test_d3_geo_adapter.py', 'test_d3_adapter_risks.py',
                 'test_d3_assembly_invariants.py']
NEW_SUITE = 'test_d3_assembly_fixture_validity.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel_to_repo(rel):
    """The registered manifest stores EXTENSION-relative paths (`scripts/...`)."""
    if rel.startswith('C-VRP_Cold-chainVehicleRoutingProblem/'):
        return REPO / rel
    return EXT / rel


def run(cmd, out_path, err_path, cwd, env):
    print('RUN %s' % ' '.join(str(c) for c in cmd[1:]))
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, text=True,
                       capture_output=True, encoding='utf-8', errors='replace')
    Path(out_path).write_text(p.stdout, encoding='utf-8')
    Path(err_path).write_text(p.stderr, encoding='utf-8')
    # the frozen suites print "  PASS <name>" / "  FAIL <name>"; the new suite prints "[PASS]".
    # Count only row lines (the previous round counted a summary line that contained FAIL).
    counts = {'pass': len(re.findall(r'(?m)^\s*\[?PASS\]?\s', p.stdout)),
              'fail': len(re.findall(r'(?m)^\s*\[?FAIL\]?\s', p.stdout))}
    return {'cmd': [str(c) for c in cmd], 'exit_code': p.returncode,
            'stdout': str(out_path), 'stderr': str(err_path), 'counts': counts,
            'n_stdout_lines': len(p.stdout.splitlines())}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    files = list(prior['files'])
    new_short = []
    for f in NEW_FILES:
        short = f.split('C-VRP_Cold-chainVehicleRoutingProblem/', 1)[1]
        new_short.append(short)
        if short not in files:
            files.append(short)
    shas = {}
    missing = []
    for rel in files:
        p = rel_to_repo(rel)
        if not p.is_file():
            missing.append(rel)
        else:
            shas[rel] = sha256(p)
    if missing:
        print('MISSING WORKTREE FILES: %s' % missing)
        return 2
    newsha = hashlib.sha256('\n'.join('%s %s' % (k, shas[k])
                                      for k in sorted(shas)).encode()).hexdigest()[:8]
    tree = ROOT / ('freeze_tree_%s' % newsha)
    print('newsha=%s tree=%s' % (newsha, tree))

    tree.mkdir(parents=True, exist_ok=True)
    mismatched = []
    for rel in files:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rel_to_repo(rel), dst)
        if sha256(dst) != shas[rel]:
            mismatched.append(rel)
    (tree / '.gitattributes').write_text('* -text\n', encoding='utf-8')

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True,
                          capture_output=True).stdout.strip()
    manifest = {
        'freeze_tree': 'freeze_tree_%s' % newsha,
        'freeze_tree_derivation':
            "first 8 hex of sha256 over sorted 'relpath sha256' lines",
        'git_commit_at_creation': head,
        'created_from': prior['freeze_tree'] + ' (same 22-file identity) + 2 new files of the '
                        'fixture-validity round; bytes copied from the worktree and verified',
        'historical_trees_modified': False,
        'maskco_code_modified': False,
        'files': files,
        'sha256': shas,
        'n_files': len(files),
        'new_files': NEW_FILES,
        'copy_mismatches': mismatched,
        'env': {'python': str(PY), 'platform': platform.platform(),
                'python_version': platform.python_version()},
    }
    (ROOT / ('A1_D3_FIXVALID_FREEZE_MANIFEST_%s.json' % newsha)).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    (ROOT / 'A1_D3_FIXVALID_IDENTITY.json').write_text(json.dumps({
        'version': 'A1_D3_FIXVALID_%s' % newsha,
        'is_freeze_seal': False,
        'is_method_approval': False,
        'note': 'New-version frozen source identity for the D3 assembly/fixture-validity '
                'round. It records the bytes that were executed; it does NOT approve any '
                'method, threshold, metric or claim.',
        'freeze_tree': 'freeze_tree_%s' % newsha,
        'files': files, 'sha256': shas,
        'imported_from_tree_required': True,
    }, indent=2, sort_keys=True), encoding='utf-8')

    env = dict(os.environ)
    # Tree dirs come FIRST because each suite inserts its own dirs at sys.path[0]; the worktree
    # entries only supply the NON-frozen helpers (coldchain_state, project_paths, ...) whose
    # relative `__file__`-based shims break inside the tree because those files are not part of
    # the registered identity.  Every frozen module must still resolve inside the tree -- that is
    # exactly what the import proof checks.
    fallback = [str(EXT / 'scripts'), str(EXT / 'scripts' / 'tests')]
    fallback += [str(p) for p in sorted((EXT / 'scripts').iterdir())
                 if p.is_dir() and any(p.glob('*.py'))]
    env['PYTHONPATH'] = os.pathsep.join(fallback)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    proof = EXT / 'scripts' / 'tests' / 'd3_fixvalid_import_proof.py'
    runs = {}
    runs['import_proof_tree'] = run(
        [PY, proof, '--tree', tree, '--worktree-scripts', EXT / 'scripts',
         '--json', RAW / 'A1_D3_FIXVALID_import_proof_tree.json'],
        RAW / 'A1_D3_FIXVALID_import_proof_tree_stdout.txt',
        RAW / 'A1_D3_FIXVALID_import_proof_tree_stderr.txt', EXT, env)
    runs['import_proof_worktree'] = run(
        [PY, proof, '--tree', EXT / 'scripts' / '..', '--worktree-scripts', EXT / 'scripts',
         '--json', RAW / 'A1_D3_FIXVALID_import_proof_worktree.json'],
        RAW / 'A1_D3_FIXVALID_import_proof_worktree_stdout.txt',
        RAW / 'A1_D3_FIXVALID_import_proof_worktree_stderr.txt', EXT, env)

    for suite in FROZEN_SUITES + [NEW_SUITE]:
        stem = suite.replace('.py', '')
        extra = ['--json', RAW / ('A1_D3_FIXVALID_rows_tree_%s.json' % stem)] \
            if suite == NEW_SUITE else []
        runs['tree_' + stem] = run(
            [PY, tree / 'scripts' / 'tests' / suite] + extra,
            RAW / ('A1_D3_FIXVALID_%s_tree_stdout.txt' % stem),
            RAW / ('A1_D3_FIXVALID_%s_tree_stderr.txt' % stem), EXT, env)
        extra_w = ['--json', RAW / ('A1_D3_FIXVALID_rows_worktree_%s.json' % stem)] \
            if suite == NEW_SUITE else []
        runs['worktree_' + stem] = run(
            [PY, EXT / 'scripts' / 'tests' / suite] + extra_w,
            RAW / ('A1_D3_FIXVALID_%s_worktree_stdout.txt' % stem),
            RAW / ('A1_D3_FIXVALID_%s_worktree_stderr.txt' % stem), EXT, env)

    rows_file = RAW / ('A1_D3_FIXVALID_rows_tree_%s.json' % NEW_SUITE.replace('.py', ''))
    new_rows = json.loads(rows_file.read_text(encoding='utf-8')) if rows_file.is_file() else None
    records = {'freeze_tree': 'freeze_tree_%s' % newsha,
               'git_commit_at_creation': head,
               'runs': runs, 'new_suite_rows_tree': new_rows,
               'n_files_in_tree': len(files), 'copy_mismatches': mismatched}
    (ROOT / 'A1_D3_FIXVALID_RECORDS.json').write_text(
        json.dumps(records, indent=2, sort_keys=True, default=repr), encoding='utf-8')

    print('\n=== SUMMARY ===')
    for key in sorted(runs):
        r = runs[key]
        print('%-46s exit=%-3s %s' % (key, r['exit_code'], r['counts']))
    print('copy_mismatches=%s' % mismatched)
    if new_rows:
        print('new suite (tree): %d rows, %d PASS, %d FAIL'
              % (new_rows['n_rows'], new_rows['n_pass'], new_rows['n_fail']))
        for row in new_rows['rows']:
            print('  %-4s %s' % ('PASS' if row['ok'] else 'FAIL', row['name']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
