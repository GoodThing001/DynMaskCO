#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Freeze the PRE-FIX `d3_geo_adapter.py` (and the three historical test files) byte-intact.

Why this exists: the Green Phase acceptance criteria require running the SAME invariant checks
against the FROZEN pre-fix version and the new version, so the evidence chain is
    pre-fix violates  ->  minimal fix  ->  post-fix satisfies  ->  legal-input behaviour unchanged
A fix is not allowed to rewrite the artefact it is judged against.

The frozen copy is verified TWO ways:
  1. byte-for-byte against the live worktree file at freeze time;
  2. byte-for-byte against the blob committed at the declared base commit (`git cat-file blob`),
     so the frozen copy is tied to published history, not just to a mutable working tree.

Read-only w.r.t. everything it inspects; it only WRITES into its own root directory.

Run: python research/review_3080_20261010/d3_g0_green_phase_root_v1/freeze_prefix.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# `87c6a5c` = the Red Phase delivery commit.  It is the PRE-FIX state for the production source
# (`d3_geo_adapter.py` is byte-identical to `dbc5971`, sha256 f7480925...), and it is also the
# first commit that contains the Red Phase counterexample harness -- which is exactly why the
# Red Phase suite can be frozen against it and `dbc5971` cannot.
BASE_COMMIT = '87c6a5c'
EXT = 'C-VRP_Cold-chainVehicleRoutingProblem'

PREFIX_FILES = {
    'd3_geo_adapter': '%s/scripts/simulation/d3_geo_adapter.py' % EXT,
    'test_d3_geo_adapter': '%s/scripts/tests/test_d3_geo_adapter.py' % EXT,
    'test_d3_adapter_risks': '%s/scripts/tests/test_d3_adapter_risks.py' % EXT,
    'test_d3_assembly_invariants': '%s/scripts/tests/test_d3_assembly_invariants.py' % EXT,
    'test_d3_g0_red_phase': '%s/scripts/tests/test_d3_g0_red_phase.py' % EXT,
    # NOTE: `test_d3_g0_green_phase.py` is deliberately NOT here -- it does not exist before the
    # fix by definition.  It is a NEW artefact and its identity is recorded in the Green Phase
    # identity file, never retro-frozen as if it were pre-fix evidence.
}


def repo_root():
    p = ROOT
    while p is not None and not (p / 'MASKCO_code').is_dir():
        p = p.parent
    if p is None:
        raise SystemExit('could not locate the repo root (no MASKCO_code/ above %s)' % ROOT)
    return p


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_blob(repo, commit, rel):
    """Bytes of `rel` at `commit`, or None when the path does not exist there."""
    r = subprocess.run(['git', '-C', str(repo), 'cat-file', 'blob', '%s:%s' % (commit, rel)],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main():
    repo = repo_root()
    frozen = ROOT / 'frozen_prefix'
    frozen.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, rel in PREFIX_FILES.items():
        live = repo / rel
        if not live.is_file():
            rows[name] = {'path': rel, 'status': 'ABSENT_IN_WORKTREE'}
            continue
        wt = live.read_bytes()
        blob = git_blob(repo, BASE_COMMIT, rel)
        dest = frozen / (name + '__' + Path(rel).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(wt)
        rows[name] = {
            'path': rel,
            'frozen_copy': str(dest.relative_to(repo)).replace('\\', '/'),
            'n_bytes': len(wt),
            'sha256_worktree_at_freeze': sha256_bytes(wt),
            'sha256_frozen_copy': sha256_bytes(dest.read_bytes()),
            'copy_equals_worktree': dest.read_bytes() == wt,
            'base_commit': BASE_COMMIT,
            'sha256_at_base_commit': sha256_bytes(blob) if blob is not None else None,
            'n_bytes_at_base_commit': len(blob) if blob is not None else None,
            'worktree_equals_base_commit': (blob == wt) if blob is not None else None,
        }
    manifest = {
        'stage': 'D3_G0_Green_Phase / prefix freeze',
        'base_commit': BASE_COMMIT,
        'purpose': ('byte-intact PRE-FIX reference for the differential Green Phase run; a fix '
                    'must not rewrite the artefact it is judged against'),
        'files': rows,
        'all_frozen_copies_verified': all(
            r.get('copy_equals_worktree') and r.get('worktree_equals_base_commit')
            for r in rows.values() if 'copy_equals_worktree' in r),
    }
    out = ROOT / 'PREFIX_FREEZE.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8', newline='\n')
    for name, r in rows.items():
        print('%-30s %s  %s' % (name, r.get('sha256_worktree_at_freeze', '-')[:16],
                                r.get('status', 'frozen')))
    print('\nall_frozen_copies_verified:', manifest['all_frozen_copies_verified'])
    print('wrote', out)
    return 0 if manifest['all_frozen_copies_verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
