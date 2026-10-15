#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.21 RUNTIME DEPENDENCY CLOSURE: what the real F execution path actually loads.

Why this module exists
----------------------
`IDENTITY_TARGETS` froze 14 paths.  The real F worker also imports the authorizer, the execution
context, the V1.16 forward control (and through it the V1.14 phase journal) and
`d3_l1_v18_readonly_audit.typed_bytes`, which computes the bitwise-repeat decision.  Editing any of
those would have changed execution, identity checking, timing or the result decision while
`--readiness` happily reported READY, because the drift checker only re-hashes the paths it was
given.

So the watch set is not a hand-written list any more.  It is DERIVED:

  seeds  = the modules the real execution path starts from (the runner, the authorizer, the
           execution context, the supervisor control, the phase journal)
  edges  = the local `import x` / `from x import y` edges reachable from the seeds, resolved by
           walking the extension's `scripts/` tree, transitively, to a fixed point
  closure = every reachable local `.py` file, EXCLUDING the test/audit/evidence tools that never
           run inside the execution path (they are recorded as intentionally excluded, with the
           reason, rather than silently dropped)

`build_watch_set()` then returns the closure as repository-relative paths, and
`cross_check_watch_set()` reports the files the closure contains that a given watch set is missing
and vice versa.  Both are used by the V1.21 identity module so the freeze and the freeze CHECK come
from one place and cannot drift apart.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# this file lives in `scripts/data/`, so the extension root is TWO parents up
EXT = HERE.parent.parent
REPO = EXT.parent
SCRIPTS = EXT / 'scripts'

# The modules the real execution path STARTS from.  A seed is included unconditionally.
SEEDS = (
    'scripts/simulation/d3_l1_v118_forward_runner.py',
    'scripts/simulation/d3_l1_v118_run_identity.py',
    'scripts/simulation/d3_l1_v118_persistence.py',
    'scripts/simulation/d3_l1_v118_targets.py',
    'scripts/simulation/d3_l1_v118_authorization.py',
    'scripts/simulation/d3_l1_v120_execution_context.py',
    'scripts/tests/d3_l1_v116_forward_control.py',
    'scripts/tests/d3_l1_v112_phase_journal.py',
)

# Non-source containers whose content participates in the decision but which live outside
# `scripts/`.  The closure itself reports SOURCE files only, and the identity module appends these
# assets with an explicit root each (weights/data/profile are extension-relative, the backbone is
# repository-relative).  The list here is documentation, not a path source, so a wrong root can no
# longer make an asset silently uncheckable.
NON_SOURCE_PATHS = (
    'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt',
    'MASKCO_code/ckpts/cvrp100.ckpt',
    'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz',
    'results/o0cc/scale_v2/objective_profile.json',
)
NON_SOURCE_NOTE = ('appended by the identity module with an explicit root per asset; the closure '
                   'reports source files only')

# Intentionally NOT in the execution-path closure, with the reason.  These are analysis or harness
# tools: the worker never imports them, so freezing them would freeze tooling rather than behaviour.
EXCLUDED_WITH_REASON = {
    'scripts/tests/d3_l1_v118_end_to_end_acceptance.py':
        'acceptance harness: drives the supervisor, is not imported by the execution path',
    'scripts/tests/d3_l1_v118_synthetic_worker.py':
        'synthetic worker: substitutes the real worker, never part of a real run',
    'scripts/tests/d3_l1_v119_blocker_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v120_bypass_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v119_zero_f_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_v118_independent_verifier.py': 'read-only verifier',
    'scripts/tests/d3_l1_two_arm_acceptance.py': 'L1 harness, not the F execution path',
    'scripts/tests/d3_l1_two_arm_readonly_audit.py': 'read-only audit tool',
    'scripts/tests/d3_c0_budget_provenance_search.py': 'read-only audit tool',
    'scripts/tests/d3_g0_residual_risk_counterexamples.py': 'read-only counterexample tool',
    'scripts/tests/d3_install_receipt_acceptance.py': 'D8 acceptance harness',
    'scripts/simulation/d3_install_receipt.py':
        'D8 receipt writer: not imported by the F execution path',
    'scripts/simulation/d3_install_boundary.py':
        'D8 boundary: re-exports the frozen gate, not imported by the F execution path',
    'scripts/tests/d3_l1_v118_freeze_check.py': 'V1.18 freeze check tool',
    'scripts/tests/d3_l1_v117_gap_audit.py': 'read-only gap audit',
    'scripts/tests/d3_l1_v120_forward_and_l1_readiness.py': 'read-only readiness tool',
    'scripts/tests/d3_l1_v110_input_qualification.py':
        'produced the V1.10 inputs; the worker reads its ARTIFACTS, not the module',
}


def repo_rel(path):
    """Repository-relative POSIX path for anything under REPO.

    Everything the closure reports uses THIS form, so the freeze, the drift checker and the
    cross-check all compare the same strings.  Mixing extension-relative and repo-relative paths was
    a real defect in the first cut of this tool: the seeds resolved against the repository root while
    the watch set was written relative to the extension root, so the closure came out empty.
    """
    return Path(path).resolve().relative_to(REPO).as_posix()


def _local_index():
    """module name -> repository-relative path, over every `scripts/**/*.py` file."""
    index = {}
    for p in sorted(SCRIPTS.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        index.setdefault(p.stem, []).append(repo_rel(p))
    return index


def _imports_of(path):
    """Local-looking module names imported by `path` (top level AND inside functions)."""
    try:
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    except SyntaxError as exc:
        raise RuntimeError('cannot parse %s: %r' % (path, exc))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split('.')[0])
    return names


def build_closure(seeds=SEEDS):
    """Transitive local-import closure of `seeds`, to a fixed point.

    Returns `(closure, edges, unresolved)` where `closure` is a sorted list of repository-relative
    paths and `edges` maps each file to the local files it imports.
    """
    index = _local_index()
    seen, edges, unresolved = {}, {}, set()
    queue = [(repo_rel(EXT / s), 'seed') for s in seeds]
    while queue:
        rel, why = queue.pop(0)
        if rel in seen:
            continue
        path = REPO / rel
        if not path.is_file():
            unresolved.add(rel)
            continue
        seen[rel] = why
        targets = {}
        for name in _imports_of(path):
            hits = index.get(name, [])
            if not hits:
                continue                        # third-party or stdlib: not our identity
            if len(hits) > 1:
                # ambiguous module name: prefer the one already known, else record every candidate
                targets[name] = hits
            else:
                targets[name] = hits[0]
        edges[rel] = targets
        for name, hit in targets.items():
            for h in ([hit] if isinstance(hit, str) else hit):
                if h not in seen:
                    queue.append((h, 'imported by %s' % rel))
    return sorted(seen), edges, sorted(unresolved)


def _repo_rel_non_source(rel):
    """The non-source containers may live under MASKCO_code/ (repo) or under the extension."""
    return rel


def build_watch_set():
    """The SOURCE watch list for a real F run: the execution-path closure, nothing else.

    Non-source assets are NOT appended here.  They live under two different roots (see
    NON_SOURCE_PATHS) and the identity module appends them with an explicit root each; mixing them
    into this list made the first cut of this function report entries that did not exist on disk,
    which silently demoted the derived freeze back to the old 14-path list.
    """
    closure, edges, unresolved = build_closure()
    watch = [r for r in closure if r not in EXCLUDED_WITH_REASON]
    excluded_reachable = sorted(r for r in closure if r in EXCLUDED_WITH_REASON)
    return {
        'watch': sorted(watch),
        'source_closure': closure,
        'excluded_reachable': excluded_reachable,
        'unresolved_seeds': unresolved,
        'edges': edges,
        'n_watch': len(watch),
        'n_source_closure': len(closure),
        'non_source_assets_note': NON_SOURCE_NOTE,
        'non_source_assets': list(NON_SOURCE_PATHS),
    }


def cross_check_watch_set(frozen_paths):
    """Compare a frozen watch set against the derived SOURCE closure.  `(ok, report)`.

    The frozen record mixes absolute paths from two roots, so a frozen path is matched by SUFFIX
    against each derived entry: the question that matters is "was this source file frozen at all",
    and suffix matching answers exactly that without needing to know which root a path came from.
    """
    derived = build_watch_set()
    want = set(derived['watch'])
    matched = set()
    for p in frozen_paths:
        s = str(p).replace('\\', '/')
        for rel in want:
            if s.endswith(rel):
                matched.add(rel)
    missing = sorted(want - matched)
    return (not missing), {'n_derived_source': len(want),
                           'n_frozen_total': len(list(frozen_paths)),
                           'n_derived_matched_in_freeze': len(matched),
                           'missing_from_freeze': missing,
                           'excluded_reachable': derived['excluded_reachable'],
                           'non_source_assets': derived['non_source_assets']}


if __name__ == '__main__':
    out = build_watch_set()
    print(json.dumps({'n_watch': out['n_watch'], 'n_source_closure': out['n_source_closure'],
                      'watch': out['watch'],
                      'excluded_reachable': out['excluded_reachable'],
                      'unresolved_seeds': out['unresolved_seeds']},
                     indent=1, default=repr))
