#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 DEPENDENCY CLOSURE: what the L1 FORMAL COMPARISON actually loads.

Why a separate closure from the F-forward one
---------------------------------------------
`d3_l1_v121_dependency_closure` derives the closure of the **F forward** path.  The L1 comparison is
a different entry point with different modules: it adds the two-arm protocol, the real-F arm, the
L1 formal runner and the V1.10 state materialisation, and it does NOT load the F-forward runner's
scoring loop.  Reusing the F closure therefore left M7 PASS while
`l1_sources_in_closure == []` -- the gate re-hashed 34 F-only paths and never saw the code that
actually decides the L1 result.

This module seeds from `d3_l1_l1_formal_run.py` and walks the transitive local-import graph to a
fixed point, exactly as the F version does, so the freeze and the freeze CHECK come from one place.

Every seed, every exclusion and every unresolved name is reported, so a missing module shows up as
`unresolved_seeds` / `missing_from_freeze` rather than as a silent gap.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent          # this file lives in scripts/data/, beside the F closure
REPO = EXT.parent
SCRIPTS = EXT / 'scripts'

# The L1 execution path's entry points.  A seed is included unconditionally; the closure then adds
# everything they import, transitively.
SEEDS = (
    'scripts/simulation/d3_l1_l1_formal_run.py',        # the formal run entry (all five modes)
    'scripts/simulation/d3_l1_real_arm.py',             # the REAL MaskCO arm + per-step ledger
    'scripts/simulation/d3_l1_two_arm_protocol.py',     # the shared-input protocol and three verdicts
    'scripts/simulation/d3_geo_adapter.py',             # D-Geo, the mask pool, the install gate
    'scripts/tests/d3_l1_v110_input_qualification.py',  # the accepted state materialisation
    'scripts/simulation/d3_l1_v118_authorization.py',   # credential validation + one-shot consume
    'scripts/simulation/d3_l1_v120_execution_context.py',   # the execution-context guard
    'scripts/tests/d3_l1_v116_forward_control.py',      # the contained supervisor
    'scripts/tests/d3_l1_v112_phase_journal.py',        # the phase journal
    'scripts/simulation/d3_l1_v118_persistence.py',     # new-destination publication + verification
    'scripts/simulation/d3_l1_v118_targets.py',         # the row/target verifier
    'scripts/simulation/d3_l1_v118_run_identity.py',    # the derived watch set
    'scripts/data/d3_l1_v121_dependency_closure.py',    # the F closure (kept, and watched)
    'scripts/data/d3_l1_closure.py',                    # THIS module: it decides the watch set
)

# Analysis / acceptance / evidence tooling that the L1 execution path never imports.  Each entry has
# a reason; an entry that IS reachable is reported as `excluded_reachable` and makes the closure
# unclean, so this list can never hide a real dependency.
EXCLUDED_WITH_REASON = {
    'scripts/tests/d3_l1_run_entry_acceptance.py':
        'acceptance harness: drives the entry from outside, not imported by it',
    'scripts/tests/d3_l1_real_arm_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_l1_signing_package.py': 'produces the signing package; never runs at runtime',
    'scripts/tests/d3_l1_l1_blocker_acceptance.py': 'acceptance harness for the three blockers',
    'scripts/tests/d3_l1_two_arm_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_two_arm_readonly_audit.py': 'independent read-only audit tool',
    'scripts/tests/d3_l1_v118_end_to_end_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v118_synthetic_worker.py': 'synthetic worker used by harnesses only',
    'scripts/tests/d3_l1_v119_zero_f_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_v120_bypass_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v119_blocker_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v118_independent_verifier.py': 'read-only verifier',
    'scripts/tests/d3_l1_v121_freeze_closure_acceptance.py': 'acceptance harness',
    'scripts/tests/d3_l1_v122_real_f_adjudication.py': 'read-only adjudication tool',
    'scripts/tests/d3_l1_v111_real_f_forward.py': 'the consumed V1.11 harness; never imported',
    'scripts/tests/d3_l1_v110_readonly_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_v114_failure_readonly_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_v19_formal_readonly_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_v117_gap_audit.py': 'read-only gap audit',
    'scripts/tests/d3_l1_v120_forward_and_l1_readiness.py': 'read-only readiness tool',
    'scripts/tests/d3_l1_v118_freeze_check.py': 'V1.18 freeze check tool',
    'scripts/tests/d3_l1_collection_readiness_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_contract_provenance_audit.py': 'read-only audit tool',
    'scripts/tests/d3_l1_f_scoring_interface_check.py': 'read-only interface check',
    'scripts/tests/d3_l1_server_timing_qualification.py': 'read-only timing tool',
    'scripts/tests/d3_l1_bitwise_parity.py': 'comparator library used by audits, not by the L1 run',
    'scripts/tests/d3_action_consistency_audit.py': 'read-only audit tool',
    'scripts/tests/d3_c0_budget_provenance_search.py': 'read-only audit tool',
    'scripts/tests/d3_g0_residual_risk_counterexamples.py': 'read-only counterexample tool',
    'scripts/tests/d3_install_receipt_acceptance.py': 'D8 acceptance harness',
    'scripts/simulation/d3_install_receipt.py': 'D8 receipt writer: not on the L1 path',
    'scripts/simulation/d3_install_boundary.py': 'D8 boundary: not on the L1 path',
    'scripts/simulation/d3_l1_v120_execution_context_selftest.py': 'self-test tool',
}

# Non-source assets whose content decides the L1 result.  Recorded with an explicit root each.
NON_SOURCE_ASSETS = (
    ('ext', 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'),
    ('repo', 'MASKCO_code/ckpts/cvrp100.ckpt'),
    ('ext', 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'),
    ('ext', 'results/o0cc/scale_v2/objective_profile.json'),
)
NON_SOURCE_NOTE = ('appended by the L1 identity with an explicit root per asset; the closure reports '
                   'source files only')


def repo_rel(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def _local_index():
    index = {}
    for p in sorted(SCRIPTS.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        index.setdefault(p.stem, []).append(repo_rel(p))
    return index


def _imports_of(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
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
            if hits:
                targets[name] = hits
        edges[rel] = targets
        for name, hit in targets.items():
            for h in hit:
                if h not in seen:
                    queue.append((h, 'imported by %s' % rel))
    return sorted(seen), edges, sorted(unresolved)


def build_watch_set():
    closure, edges, unresolved = build_closure()
    watch = [r for r in closure if r not in EXCLUDED_WITH_REASON]
    excluded_reachable = sorted(r for r in closure if r in EXCLUDED_WITH_REASON)
    return {'watch': sorted(watch), 'source_closure': closure,
            'excluded_reachable': excluded_reachable, 'unresolved_seeds': unresolved,
            'edges': edges, 'n_watch': len(watch), 'n_source_closure': len(closure),
            'non_source_assets': [r for _root, r in NON_SOURCE_ASSETS],
            'non_source_assets_note': NON_SOURCE_NOTE}


def critical_l1_sources(*, form='repo'):
    """The modules WITHOUT which the L1 comparison cannot be what it claims to be.

    M7 requires each of these to be in the frozen watch set by name, so the gate cannot pass on a
    closure that happens to be large but omits the decision-relevant code.

    `form='repo'` returns REPOSITORY-relative paths -- the SAME spelling
    `build_watch_set()['watch']` uses, so membership is a plain string comparison.  `form='ext'`
    returns the extension-relative spelling, which is what a human reads.  Handing back two
    different spellings from one function is how the first cut of this check compared an absolute
    Windows path against a repo-relative suffix and reported every module as missing.
    """
    ext_rel = (
        'scripts/simulation/d3_l1_l1_formal_run.py',
        'scripts/simulation/d3_l1_real_arm.py',
        'scripts/simulation/d3_l1_two_arm_protocol.py',
        'scripts/simulation/d3_geo_adapter.py',
        'scripts/tests/d3_l1_v110_input_qualification.py',
        'scripts/simulation/d3_l1_v118_authorization.py',
        'scripts/simulation/d3_l1_v120_execution_context.py',
        'scripts/tests/d3_l1_v116_forward_control.py',
        'scripts/tests/d3_l1_v112_phase_journal.py',
        'scripts/simulation/d3_l1_v118_persistence.py',
        'scripts/simulation/d3_l1_v118_targets.py',
        'scripts/simulation/d3_l1_v118_run_identity.py',
        'scripts/data/d3_l1_v121_dependency_closure.py',
        'scripts/data/d3_l1_closure.py',
    )
    if form == 'ext':
        return ext_rel
    if form != 'repo':
        raise ValueError("form must be 'repo' or 'ext', got %r" % (form,))
    return tuple(repo_rel(EXT / s) for s in ext_rel)


def cross_check_watch_set(frozen_paths):
    """`(ok, report)`: are ALL critical modules frozen, and does the freeze cover the closure?"""
    derived = build_watch_set()
    want = set(derived['watch'])
    frozen_posix = [str(p).replace('\\', '/') for p in frozen_paths]
    matched = set()
    for s in frozen_posix:
        for rel in want:
            if s.endswith(rel):
                matched.add(rel)
    missing = sorted(want - matched)
    # Membership is decided in ONE repository-relative spelling, so a spelling difference can no
    # longer masquerade as a missing module.
    critical = list(critical_l1_sources())
    critical_missing = [c for c in critical if not any(s.endswith(c) for s in frozen_posix)]
    return (not missing and not critical_missing), {
        'n_derived_source': len(want), 'n_frozen_total': len(frozen_posix),
        'n_derived_matched_in_freeze': len(matched), 'missing_from_freeze': missing,
        'critical_l1_sources': critical,
        'critical_l1_sources_ext_relative': list(critical_l1_sources(form='ext')),
        'critical_missing_from_freeze': critical_missing,
        'excluded_reachable': derived['excluded_reachable'],
        'non_source_assets': derived['non_source_assets']}


if __name__ == '__main__':
    out = build_watch_set()
    print(json.dumps({'n_watch': out['n_watch'], 'n_source_closure': out['n_source_closure'],
                      'watch': out['watch'], 'excluded_reachable': out['excluded_reachable'],
                      'unresolved_seeds': out['unresolved_seeds'],
                      'critical_l1_sources': list(critical_l1_sources())},
                     indent=1, default=repr))
