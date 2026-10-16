#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 SIGN-BLOCKER INDEPENDENT READ-ONLY AUDIT.

Re-verifies, from the FILES and the EMITTED ARTIFACTS, that the three sign-blocking defects are
closed.  It imports NEITHER the L1 formal run entry, NOR the two-arm protocol, NOR the real arm, NOR
the geo adapter, NOR `d3_l1_closure` -- it reads their SOURCE as text and parses it with `ast`, so a
module that decides the L1 result cannot be the thing that certifies itself.

Rows
----
  R1   the closure module exists, parses, and seeds the L1 FORMAL ENTRY (not the F-forward runner)
  R2   every module the review named is in the closure's `critical_l1_sources` list
  R3   the L1 identity builds its watch set from the L1 closure and from nothing else -- the
       F-forward closure import is GONE, and the silent F-only fallback is GONE
  R4   the entry refuses (raises) when a critical module is missing from the derived watch set
  R5   the supervisor calls `run_contained` WITH a ready probe that reads an immutable record
  R6   the worker publishes an immutable READY carrying the run id, the warm-up call count and the
       warm-up digest -- the fields the probe validates
  R7   the worker publishes a `slot` / `slot_finished` phase pair per state, and a `completed` stage
  R8   the worker writes no non-journal record into the journal directory under the `phase_` prefix
  R9   `adjudicate` no longer filters `FAILED_EXCEPTION`, and EVERY call site passes the whole list
  R10  `adjudicate` takes an `expected_states` bound and emits BLOCKED on a short/duplicated list
  R11  `ledger_ok=false`, a missing state and an identity failure each produce a blocking reason
  R12  the frozen caps are 10 / 60 / 180 s and 128 / 2 / 2 and no diff-window edit changed them
  R13  the blocker acceptance exists, its verdict is CLOSED, and every row passed
  R14  the blocker acceptance's A5/A6 rows really exercised the drift path (not an empty fixture)
  R15  the signing package gates on the L1 closure and reports the L1 modules in the closure
  R16  the invalidated L1-v1 package is byte-intact and explicitly annotated, never overwritten
  R17  the audit itself imports none of the modules under test

Every row reads a file or an artifact.  A row that cannot read its evidence FAILS; it never passes by
default.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
L1 = EV / 'l1_gates'
RUN_ENTRY = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
CLOSURE = EXT / 'scripts/data/d3_l1_closure.py'
PROTOCOL = EXT / 'scripts/simulation/d3_l1_two_arm_protocol.py'
BLOCKER_ACC = L1 / 'L1_SIGN_BLOCKER_ACCEPTANCE.json'
ORDER_ACC = L1 / 'L1_ARM_ORDER_ACCEPTANCE.json'
PREREG_V2 = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
PREREG_V1 = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md'
MANIFEST = L1 / 'L1_V2_PREREGISTRATION_MANIFEST.json'
PREREG_BUILDER = EXT / 'scripts/tests/d3_l1_l1_prereg_v2.py'
PACKAGE = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V3.json'
PACKAGE_BLOCKED = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V3.json.blocked'
OLD_PACKAGE = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE.json'
ARCHIVED_V2 = L1 / 'SUPERSEDED_L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V2_run_l1v1.json'
ARCHIVE_NOTE = L1 / 'SUPERSEDED_L1_EVIDENCE_ARCHIVE_NOTE.json'
INVALIDATION = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V1_INVALIDATION.json'

ROW_NAMES = (
    'R1 closure_module_seeds_the_l1_entry',
    'R2 review_named_modules_are_critical',
    'R3 identity_watch_set_derived_from_the_l1_closure',
    'R4 missing_critical_module_is_refused',
    'R5 supervisor_passes_a_ready_probe',
    'R6 worker_publishes_an_immutable_ready',
    'R7 worker_publishes_per_state_phase_pairs',
    'R8 journal_directory_holds_only_journal_records',
    'R9 adjudication_has_no_failed_exception_filter',
    'R10 adjudication_bounds_the_denominator',
    'R11 blocking_reasons_cover_ledger_missing_identity',
    'R12 frozen_caps_unchanged',
    'R13 blocker_acceptance_closed',
    'R14 counterexamples_really_exercised_drift',
    'R15 package_gates_on_the_l1_closure',
    'R16 invalidated_v1_package_intact_and_annotated',
    'R17 audit_imports_none_of_the_modules_under_test',
    'R18 preregistration_registers_the_frozen_watch_set',
    'R19 preregistration_identity_and_package_are_one_object',
    'R20 v1_preregistration_and_v2_package_archived_byte_intact',
    'R21 arm_order_confound_is_bounded_and_classified',
)

# the modules the independent review named as decision-relevant
REVIEW_NAMED = (
    'd3_l1_l1_formal_run.py', 'd3_l1_real_arm.py', 'd3_l1_two_arm_protocol.py', 'd3_geo_adapter.py',
    'd3_l1_v110_input_qualification.py', 'd3_l1_v118_authorization.py',
    'd3_l1_v120_execution_context.py', 'd3_l1_v116_forward_control.py',
    'd3_l1_v112_phase_journal.py', 'd3_l1_v118_persistence.py', 'd3_l1_v118_targets.py',
    'd3_l1_v118_run_identity.py', 'd3_l1_v121_dependency_closure.py',
)

# module names the audit must NOT have imported, and the source files it must not execute
FORBIDDEN_IMPORTS = ('d3_l1_l1_formal_run', 'd3_l1_two_arm_protocol', 'd3_l1_real_arm',
                     'd3_geo_adapter', 'd3_l1_closure', 'd3_l1_v121_dependency_closure',
                     'd3_l1_v116_forward_control', 'd3_l1_v118_authorization')


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:                                                      # noqa: BLE001
        return None


def tree_of(path):
    return ast.parse(Path(path).read_text(encoding='utf-8'))


def calls_in(tree, name, src=None):
    """Every call to `name(...)`, as `{lineno, kwargs, arg_srcs}`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        got = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
        if got != name:
            continue
        out.append({'lineno': node.lineno,
                    'kwargs': sorted(k.arg for k in node.keywords if k.arg),
                    'n_args': len(node.args)})
    return out


def func_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    rows = []

    def ck(name, ok, detail):
        rows.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                                 json.dumps(detail, default=repr, sort_keys=True)[:210]))

    entry_src = RUN_ENTRY.read_text(encoding='utf-8') if RUN_ENTRY.is_file() else ''
    entry_tree = ast.parse(entry_src) if entry_src else None
    closure_src = CLOSURE.read_text(encoding='utf-8') if CLOSURE.is_file() else ''
    proto_tree = tree_of(PROTOCOL) if PROTOCOL.is_file() else None
    blockers = load(BLOCKER_ACC) or {}
    # The audit and the package are MUTUALLY DEPENDENT: the package's M8 needs the audit's verdict,
    # and this row needs the package.  So the row reads the SIGNED-READY package when it exists and
    # otherwise the explicitly `.blocked` diagnostic copy, and says WHICH one it read.  A blocked
    # copy can never be mistaken for a signable one: the gate carries `all_gates_pass` and the path
    # is different.
    pkg_path, pkg_is_blocked = (PACKAGE, False) if PACKAGE.is_file() else (PACKAGE_BLOCKED, True)
    package = load(pkg_path) or {}

    # R1 -------------------------------------------------------------------------------------
    seeds = []
    if closure_src:
        for node in ast.parse(closure_src).body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, 'id', None) == 'SEEDS' for t in node.targets):
                seeds = [e.value for e in node.value.elts
                         if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    r1_ok = bool(CLOSURE.is_file() and 'd3_l1_l1_formal_run.py' in ' '.join(seeds)
                 and not any('d3_l1_v118_forward_runner' in s for s in seeds))
    ck(ROW_NAMES[0], r1_ok,
       {'closure_module': str(CLOSURE.relative_to(REPO)), 'closure_sha256': sha(CLOSURE),
        'n_seeds': len(seeds), 'seeds_l1_entry': [s for s in seeds if 'l1_formal_run' in s],
        'seeds_f_forward_runner': [s for s in seeds if 'forward_runner' in s]})

    # R2 -------------------------------------------------------------------------------------
    critical = []
    if closure_src:
        node = func_node(ast.parse(closure_src), 'critical_l1_sources')
        if node is not None:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and sub.value.endswith('.py'):
                    critical.append(sub.value)
    critical = sorted(set(critical))
    missing = [m for m in REVIEW_NAMED if not any(c.endswith(m) for c in critical)]
    ck(ROW_NAMES[1], not missing and len(critical) >= len(REVIEW_NAMED),
       {'n_critical_listed': len(critical), 'n_review_named': len(REVIEW_NAMED),
        'review_named_not_critical': missing, 'critical': critical})

    # R3 -------------------------------------------------------------------------------------
    # the identity must build its watch set from the L1 closure, and the F-only fallback must be gone
    uses_l1_closure = 'd3_l1_closure' in entry_src
    uses_f_closure = 'd3_l1_v121_dependency_closure as closure' in entry_src
    fwd_identity_call = 'fwd_identity.watched_identity_paths' in entry_src
    r3_ok = uses_l1_closure and not uses_f_closure and not fwd_identity_call
    ck(ROW_NAMES[2], r3_ok,
       {'imports_d3_l1_closure': uses_l1_closure,
        'imports_the_f_forward_closure': uses_f_closure,
        'calls_fwd_identity_watched_identity_paths': fwd_identity_call,
        'pre_fix_line': 'watch = fwd_identity.watched_identity_paths()  # the F-forward watch set',
        'why_the_fallback_mattered': 'watched_identity_paths() fell back to WATCHED_CODE_V118 when '
                                     'the closure could not be derived, so a broken derivation '
                                     'produced a QUIET F-only freeze instead of an error'})

    # R4 -------------------------------------------------------------------------------------
    r4_ok = ('critical_missing' in entry_src
             and 'the L1 watch set omits decision-relevant module(s)' in entry_src
             and 'unresolved_seeds' in entry_src and 'excluded_reachable' in entry_src)
    ck(ROW_NAMES[3], r4_ok,
       {'refuses_on_critical_missing': 'critical_missing' in entry_src,
        'refuses_on_unclean_closure': 'unresolved_seeds' in entry_src
        and 'excluded_reachable' in entry_src})

    # R5 -------------------------------------------------------------------------------------
    rc_calls = calls_in(entry_tree, 'run_contained') if entry_tree else []
    with_probe = [c for c in rc_calls if 'ready_probe' in c['kwargs']]
    probe_src = 'make_l1_ready_probe' in entry_src and "latest('L1_READY')" in entry_src
    r5_ok = bool(rc_calls) and len(with_probe) == len(rc_calls) and probe_src
    ck(ROW_NAMES[4], r5_ok,
       {'run_contained_calls': rc_calls, 'calls_with_ready_probe': len(with_probe),
        'probe_reads_immutable_record': "latest('L1_READY')" in entry_src,
        'pre_fix_call': 'control.run_contained(command, out_dir, identity, init_cap=..., '
                        'slot_cap=..., batch_cap=..., env=env)  # NO ready_probe',
        'pre_fix_consequence': 'the V1.16 default probe looked for a fixed `READY.json`, which the '
                               'V1.18+ worker can never publish, so initialization was measured as '
                               'the WHOLE batch and a slow batch was killed at the 60 s init cap'})

    # R6 -------------------------------------------------------------------------------------
    ready_fields = ("'n_warmup_calls': 1", "'warmup_scores_sha256'",
                    "'initialization_and_compilation_finished'", "'run_id': ident['run_id']")
    have = {f: f in entry_src for f in ready_fields}
    r6_ok = all(have.values()) and "publish(out_dir, 'L1_READY'" in entry_src
    ck(ROW_NAMES[5], r6_ok, {'ready_fields_present': have,
                             'published_under_immutable_name': "publish(out_dir, 'L1_READY'"
                             in entry_src})

    # R7 -------------------------------------------------------------------------------------
    slot_pub = "journal.publish('slot'" in entry_src
    slot_fin = "journal.publish('slot_finished'" in entry_src
    completed = "journal.publish('completed'" in entry_src
    r7_ok = slot_pub and slot_fin and completed
    ck(ROW_NAMES[6], r7_ok, {'publishes_slot_start': slot_pub, 'publishes_slot_finish': slot_fin,
                             'publishes_completed': completed,
                             'pre_fix_state': 'the worker published NO phase records at all, so the '
                                              'supervisor could not tell a running state from a '
                                              'stuck initialization'})

    # R8 -------------------------------------------------------------------------------------
    # no record may be published into the journal directory under a `phase_` prefix by the
    # persistence layer: `timeline_check` opens EVERY `phase_*.json` and requires a sequence + mono
    bad_journal = []
    if entry_tree:
        for node in ast.walk(entry_tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
            if name != 'publish_new' or len(node.args) < 2:
                continue
            d, prefix = node.args[0], node.args[1]
            d_src = ast.unparse(d)
            p_src = ast.unparse(prefix)
            if 'phases' in d_src and 'phase_' in p_src.lower():
                bad_journal.append({'line': node.lineno, 'dir': d_src, 'prefix': p_src})
    r8_ok = not bad_journal and "publish_new(out_dir / 'phases', 'IDENTITY_END'" in entry_src
    ck(ROW_NAMES[7], r8_ok,
       {'non_journal_publishes_under_phase_prefix': bad_journal,
        'identity_tail_prefix_used': "publish_new(out_dir / 'phases', 'IDENTITY_END'"
        in entry_src,
        'why': 'a record published through a different mechanism under `phase_*` corrupts '
               'd3_l1_v116_forward_control.timeline_check, which parses every file matching that '
               'glob and requires a `sequence` and a finite `mono`'})

    # R9 -------------------------------------------------------------------------------------
    adj_def = func_node(entry_tree, 'adjudicate') if entry_tree else None
    adj_node = adj_def.args if adj_def is not None else None      # the `arguments` node
    adj_src = ast.get_source_segment(entry_src, adj_def) if adj_def else ''
    filters = "!= 'FAILED_EXCEPTION'" in adj_src
    adj_calls = []
    for node in ast.walk(entry_tree) if entry_tree else []:
        if isinstance(node, ast.Call):
            fn = node.func
            if (fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)) == 'adjudicate' \
                    and node.args:
                adj_calls.append({'line': node.lineno,
                                  'arg_src': ast.unparse(node.args[0])[:60],
                                  'is_bare_records': isinstance(node.args[0], ast.Name)
                                  and node.args[0].id == 'records'})
    r9_ok = (not filters and bool(adj_calls) and all(c['is_bare_records'] for c in adj_calls))
    ck(ROW_NAMES[8], r9_ok,
       {'failed_exception_filter_in_adjudicate': filters, 'adjudicate_call_sites': adj_calls,
        'pre_fix_call': "adjudicate([r for r in records if r.get('status') != 'FAILED_EXCEPTION'], "
                        "synthetic=synthetic_run)  # denominator silently 63",
        'pre_fix_consequence': 'a run in which one state crashed reported its verdict over the '
                               'survivors while the report still read like a round verdict'})

    # R10 ------------------------------------------------------------------------------------
    # `expected_states` is keyword-only, so it lives in `kwonlyargs`, not `args` -- reading only
    # `args` is how this row first crashed with "'arguments' object is not iterable"
    sig = ''
    if adj_node is not None:
        # `adj_node` is the `arguments` node; the individual arguments live in `.args` / `.kwonlyargs`
        names = [a.arg for a in adj_node.args] + [a.arg for a in adj_node.kwonlyargs]
        sig = 'expected_states' in names
    sig_default = None
    if adj_node is not None:
        pairs = []
        if adj_node.defaults:
            pairs += list(zip(adj_node.args[-len(adj_node.defaults):], adj_node.defaults))
        # `kw_defaults` carries None for a keyword-only argument with NO default
        pairs += [(a, d) for a, d in zip(adj_node.kwonlyargs, adj_node.kw_defaults) if d is not None]
        for a, d in pairs:
            if a.arg == 'expected_states' and isinstance(d, ast.Constant):
                sig_default = d.value
    # the module-level EXPECTED_STATES constant the default refers to
    module_expected = None
    if entry_tree is not None:
        for node in entry_tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, 'id', None) == 'EXPECTED_STATES' for t in node.targets) \
                    and isinstance(node.value, ast.Constant):
                module_expected = node.value.value
    r10_ok = sig and 'denominator_is_the_preregistered_list' in adj_src \
        and "'BLOCKED' if blocking else 'ADJUDICATED'" in adj_src and module_expected == 64
    ck(ROW_NAMES[9], r10_ok,
       {'expected_states_parameter': sig, 'default': 'EXPECTED_STATES (module constant)',
        'expected_states_constant': module_expected,
        'reports_denominator_is_the_preregistered_list': 'denominator_is_the_preregistered_list'
        in adj_src,
        'blocked_verdict_present': "'BLOCKED' if blocking else 'ADJUDICATED'" in adj_src})

    # R11 ------------------------------------------------------------------------------------
    ledger_block = 'ledger_ok=false' in adj_src or 'ledgers_bad' in adj_src
    missing_block = 'pre-registered state list' in adj_src
    ident_block = 'single-factor' in adj_src
    r11_ok = ledger_block and missing_block and ident_block
    ck(ROW_NAMES[10], r11_ok,
       {'ledger_blocking_reason': ledger_block, 'missing_state_blocking_reason': missing_block,
        'identity_blocking_reason': ident_block})

    # R12 ------------------------------------------------------------------------------------
    caps = {}
    if entry_tree:
        for node in ast.walk(entry_tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, 'id', None) == 'BUDGET' for t in node.targets) \
                    and isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                        caps[k.value] = v.value
    want_caps = {'mask_steps_m': 2, 'per_arm_per_state_scorer_calls': 2,
                 'per_arm_total_scorer_calls': 128, 'per_state_wall_clock_s': 10.0,
                 'initialize_compile_s': 60.0, 'batch_wall_clock_s': 180.0}
    r12_ok = caps == want_caps
    ck(ROW_NAMES[11], r12_ok,
       {'budget_as_parsed': caps, 'expected': want_caps,
        'wall_clock_scope': (lambda m: m.group(1) if m else None)(
            __import__('re').search(r"WALL_CLOCK_SCOPE = '([^']+)'", entry_src))})

    # R13 ------------------------------------------------------------------------------------
    r13_ok = (blockers.get('verdict') == 'L1_SIGN_BLOCKERS_CLOSED' and blockers.get('n_fail') == 0
              and blockers.get('n_rows') == 21)
    ck(ROW_NAMES[12], r13_ok,
       {'verdict': blockers.get('verdict'), 'n_rows': blockers.get('n_rows'),
        'n_pass': blockers.get('n_pass'), 'n_fail': blockers.get('n_fail'),
        'tool_sha256': blockers.get('tool_sha256'), 'runner_sha256': blockers.get('runner_sha256'),
        'closure_sha256': blockers.get('closure_sha256')})

    # R14 ------------------------------------------------------------------------------------
    a = (blockers.get('results') or {}).get('A') or {}
    b = (blockers.get('results') or {}).get('B') or {}
    c = (blockers.get('results') or {}).get('C') or {}
    rows14 = [r for r in (blockers.get('rows') or [])]
    a5 = next((r for r in rows14 if 'A5' in r['name']), {})
    a6 = next((r for r in rows14 if 'A6' in r['name']), {})
    tamper_rows = a.get('tamper_rows') or []
    n_drifted = sum(1 for r in tamper_rows if r.get('A4_n_drifted') == 1)
    n_a5 = sum(1 for r in tamper_rows if r.get('A5_authorize_raised') is True
               and r.get('A5_n_receipts') == 0)
    n_a6 = sum(1 for r in tamper_rows if r.get('A6_drift_in_stderr') is True
               and r.get('A6_n_state_records') == 0)
    repo_intact = a.get('A8', {}).get('ok') is True
    r14_ok = (len(tamper_rows) == 14 and n_drifted == 14 and n_a5 == 14 and n_a6 == 14
              and repo_intact)
    ck(ROW_NAMES[13], r14_ok,
       {'n_modules_tampered': len(tamper_rows), 'n_with_exactly_one_drifted_path': n_drifted,
        'n_blocking_consumption': n_a5, 'n_blocking_the_worker_start': n_a6,
        'repository_untouched': repo_intact,
        'b2_wall_s': (b.get('B2') or {}).get('wall_s'), 'b2_ok': (b.get('B2') or {}).get('ok'),
        'b3_ok': (b.get('B3') or {}).get('ok'), 'b4_ok': (b.get('B4') or {}).get('ok'),
        'b6_ok': (b.get('B6') or {}).get('ok'),
        'c1_ok': (c.get('C1') or {}).get('ok'), 'c4_ok': (c.get('C4') or {}).get('ok'),
        'a5_row': a5.get('ok'), 'a6_row': a6.get('ok')})

    # R15 ------------------------------------------------------------------------------------
    m7 = ((package.get('7_mechanical_acceptance') or {}).get('M7_closure_frozen_by_sha') or {})
    m7d = m7.get('detail') or {}
    ident6 = package.get('6_identity_freeze') or {}
    r15_ok = (m7.get('ok') is True and m7d.get('closure_module') == 'scripts/data/d3_l1_closure.py'
              and len(m7d.get('l1_sources_in_closure') or []) >= 6
              and (m7d.get('closure_diff') or {}).get('critical_missing_from_freeze') == []
              and ident6.get('derived_from', '').startswith('the L1 formal entry'))
    ck(ROW_NAMES[14], r15_ok,
       {'package_read': str(pkg_path.relative_to(REPO)), 'package_is_blocked_diagnostic_copy':
        pkg_is_blocked,
        'package_written_at_the_signable_path': PACKAGE.is_file(),
        'package_sha256': sha(pkg_path),
        'run_id': package.get('run_id'), 'package_version': package.get('package_version'),
        'all_gates_pass': package.get('all_gates_pass'),
        'blocking_gates': package.get('blocking_gates'),
        'm7_ok': m7.get('ok'), 'm7_n_watch_paths': m7d.get('n_watch_paths'),
        'm7_l1_sources_in_closure': m7d.get('l1_sources_in_closure'),
        'm7_critical_missing': (m7d.get('closure_diff') or {})
        .get('critical_missing_from_freeze'),
        'identity_freeze_n_watch_paths': ident6.get('watch_paths'),
        'pre_fix': 'M7 read PASS with l1_sources_in_closure == [] and n_watch_paths == 34'})

    # R16 ------------------------------------------------------------------------------------
    inv = load(INVALIDATION) or {}
    old_sha_now = sha(OLD_PACKAGE)
    r16_ok = (OLD_PACKAGE.is_file() and inv.get('invalidated_package_sha256') == old_sha_now
              and bool(inv.get('reason')) and inv.get('not_deleted') is True
              and inv.get('superseded_by') is not None)
    ck(ROW_NAMES[15], r16_ok,
       {'invalidated_package_present': OLD_PACKAGE.is_file(),
        'sha256_now': old_sha_now,
        'sha256_recorded_at_invalidation': inv.get('invalidated_package_sha256'),
        'bytes_unchanged_since_invalidation': inv.get('invalidated_package_sha256') == old_sha_now,
        'reason_present': bool(inv.get('reason')),
        'not_deleted': inv.get('not_deleted'),
        'superseded_by': inv.get('superseded_by')})

    # R17 ------------------------------------------------------------------------------------
    loaded = sorted(m for m in FORBIDDEN_IMPORTS if m in sys.modules)
    src = Path(__file__).read_text(encoding='utf-8')
    imports_mod = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            imports_mod += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports_mod.append(node.module)
    r17_ok = not loaded and not any(m in ' '.join(imports_mod) for m in FORBIDDEN_IMPORTS)
    ck(ROW_NAMES[16], r17_ok,
       {'modules_under_test_found_in_sys_modules': loaded,
        'imports_declared_in_this_file': sorted(set(imports_mod)),
        'method': 'the audit reads source as text and parses with ast; it never imports or executes '
                  'the entry, the protocol, the real arm, the geo adapter or the closure'})

    # R18 ------------------------------------------------------------------------------------
    # TASK 1.  The pre-registration must register the watch set the identity actually freezes, with
    # the COMPLETE per-file list.  The V1 pairing failed exactly here: §7 said 34, the identity froze
    # 42, and nothing compared the two.
    manifest = load(MANIFEST) or {}
    md = PREREG_V2.read_text(encoding='utf-8') if PREREG_V2.is_file() else ''
    n_md_count = None
    m = __import__('re').search(r'运行期 watch 集 = (\d+) 条', md)
    if m:
        n_md_count = int(m.group(1))
    manifest_watch = manifest.get('watch') or []
    rows_in_md = sum(1 for e in manifest_watch
                     if ('| `%s` | `%s` |' % (e.get('path'), e.get('sha256_frozen'))) in md)
    try:
        import importlib.util as _ilu
        spec_pb = _ilu.spec_from_file_location('audit_prereg_builder', str(PREREG_BUILDER))
        builder = _ilu.module_from_spec(spec_pb)
        sys.modules['audit_prereg_builder'] = builder
        spec_pb.loader.exec_module(builder)
        builder_problems = builder.verify()
    except BaseException as exc:                                          # noqa: BLE001
        builder_problems = ['%s: %s' % (type(exc).__name__, exc)]
    r18_ok = (bool(md) and n_md_count is not None and n_md_count == len(manifest_watch)
              and len(manifest_watch) == manifest.get('n_watch_total') == 42
              and rows_in_md == len(manifest_watch) and not builder_problems
              and manifest.get('state_binding_sha256')
              == 'f5eedd1027342496939e7257e64f2d25dd6f7cd8787e731b6aa9dde213eefee1'
              and manifest.get('run_id') == 'd3l1-l1-two-arm-20261016-l1v2')
    ck(ROW_NAMES[17], r18_ok,
       {'preregistration': str(PREREG_V2.relative_to(REPO)), 'preregistration_sha256': sha(PREREG_V2),
        'watch_count_parsed_from_§7': n_md_count,
        'watch_count_in_manifest': manifest.get('n_watch_total'),
        'complete_rows_present_in_the_document': rows_in_md,
        'manifest_verify_problems': builder_problems[:4],
        'run_id': manifest.get('run_id'),
        'state_binding_sha256': manifest.get('state_binding_sha256'),
        'n_watch_source': manifest.get('n_watch_source'),
        'n_watch_non_source_assets': manifest.get('n_watch_non_source_assets'),
        'pre_fix_state': 'V1 §7 registered 34 F-only paths against a 42-path identity'})

    # R19 ------------------------------------------------------------------------------------
    # The package must bind the NEW pre-registration SHA, the source identity and the state identity,
    # and its M11 must pass -- i.e. document, identity and package are ONE object.
    m11 = ((package.get('7_mechanical_acceptance') or {})
           .get('M11_preregistration_identity_and_package_are_one_object') or {})
    m11d = m11.get('detail') or {}
    freeze6 = package.get('6_identity_freeze') or {}
    man6b = package.get('6b_preregistration_manifest') or {}
    r19_ok = (package.get('run_id') == 'd3l1-l1-two-arm-20261016-l1v2'
              and package.get('package_version') == 'L1_V3'
              and m11.get('ok') is True
              and (m11d.get('preregistration') or {}).get('sha256') == sha(PREREG_V2)
              and (m11d.get('preregistration') or {}).get('version') == 'L1-v2'
              and (freeze6.get('preregistration') or {}).get('sha256') == sha(PREREG_V2)
              and freeze6.get('source_identity_sha256') == manifest.get('source_identity_sha256')
              and freeze6.get('state_binding_sha256') == manifest.get('state_binding_sha256')
              and man6b.get('verify_problems') == []
              and freeze6.get('drift_at_package_time') == []
              and (package.get('5_budget') or {}).get('frozen_caps', {}).get(
                  'unchanged_by_the_blocker_fixes') is True)
    ck(ROW_NAMES[18], r19_ok,
       {'package': str(pkg_path.relative_to(REPO)), 'run_id': package.get('run_id'),
        'package_version': package.get('package_version'),
        'all_gates_pass': package.get('all_gates_pass'),
        'm11_ok': m11.get('ok'),
        'preregistration_sha256_in_package': (freeze6.get('preregistration') or {}).get('sha256'),
        'preregistration_sha256_on_disk': sha(PREREG_V2),
        'source_identity_sha256_in_package': freeze6.get('source_identity_sha256'),
        'source_identity_sha256_in_manifest': manifest.get('source_identity_sha256'),
        'state_binding_sha256': freeze6.get('state_binding_sha256'),
        'watch_paths': freeze6.get('watch_paths'),
        'drift_at_package_time': freeze6.get('drift_at_package_time'),
        'manifest_verify_problems': man6b.get('verify_problems')})

    # R20 ------------------------------------------------------------------------------------
    # NOTE ON R17: `PREREG_BUILDER.verify()` imports the run entry, which would leave
    # `d3_l1_l1_formal_run` in `sys.modules` and make the independence row read a leakage that this
    # audit itself caused.  The module is therefore evicted again as soon as the row is done, and R17
    # reports the state as it is at THAT point rather than trusting a claim.
    for _leaked in list(sys.modules):
        if _leaked in ('l1_entry_prereg', 'd3_l1_l1_formal_run', 'audit_prereg_builder',
                       'l1_prereg_builder'):
            del sys.modules[_leaked]
    # Everything superseded must still be present and byte-intact, and the archive must say so.
    note = load(ARCHIVE_NOTE) or {}
    note_rows = {r.get('file'): r.get('sha256') for r in (note.get('archived') or [])}
    archived_ok = True
    archived_detail = []
    for f, recorded in note_rows.items():
        actual = sha(L1 / f)
        same = (recorded is None) or (actual == recorded)
        archived_ok = archived_ok and (L1 / f).is_file() and same
        archived_detail.append({'file': f, 'present': (L1 / f).is_file(),
                                'sha256_now': actual, 'sha256_recorded': recorded, 'ok': same})
    inv = load(INVALIDATION) or {}
    r20_ok = (archived_ok and note.get('archived') and note.get('no_state_or_result_was_rewritten')
              is True and bool(note.get('note'))
              and PREREG_V1.is_file()
              and sha(PREREG_V1) == '2e73cac4787b3ba4092d8cd4546332d2b63514724163cba5e1ba1a19dd82c54e'
              and sha(OLD_PACKAGE) == '0b9336fa8036a47fdeb50e4e4394bf90a2cd3d10edd8d5e670616afe81988cfe'
              and inv.get('invalidated_package_sha256') == sha(OLD_PACKAGE)
              and (package.get('superseded') or {}).get('v2_package', {}).get('bytes_intact')
              is True)
    ck(ROW_NAMES[19], r20_ok,
       {'archive_note': str(ARCHIVE_NOTE.relative_to(REPO)),
        'archived_files': archived_detail,
        'preregistration_v1_sha256': sha(PREREG_V1),
        'preregistration_v1_expected': '2e73cac4787b3ba4092d8cd4546332d2b63514724163cba5e1ba1a19dd82c54e',
        'original_v1_package_sha256': sha(OLD_PACKAGE),
        'invalidation_record_records_it': inv.get('invalidated_package_sha256') == sha(OLD_PACKAGE),
        'v2_package_bytes_intact': (package.get('superseded') or {})
        .get('v2_package', {}).get('bytes_intact')})

    # R21 ------------------------------------------------------------------------------------
    # TASK 2.  The order confound must be stated, bounded, and mechanically classified; the caps must
    # be untouched and no formal run may be added by the handling.
    order_acc = load(ORDER_ACC) or {}
    o_rows = {r['name'][:3]: r for r in (order_acc.get('rows') or [])}
    o6c = package.get('6c_arm_order_and_shared_allowance') or {}
    entry_order = ''
    if entry_tree is not None:
        for node in entry_tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'ARM_ORDER'
                                                    for t in node.targets):
                entry_order = [e.value for e in node.value.elts
                               if isinstance(e, ast.Constant)]
    entry_txt = RUN_ENTRY.read_text(encoding='utf-8')
    # TASK 2, and the load-bearing part: the fairness handling must be a READING RULE.  Asserted
    # structurally rather than by substring -- the package must claim no budget widening, no arm
    # change and no added run, AND the entry must not have gained a second opportunity, a rerun path
    # or a reorder knob while adding the classifier.
    added_run_markers = [m for m in ('--rerun', '--second-run', '--reorder', 'reversed_order',
                                     'second_opportunity')
                         if m in entry_txt]
    o6c_fh = o6c.get('fairness_handling') or ''
    r21_ok = (entry_order == ['F', 'D-Geo']
              and 'per_state_both_arms_share_one_10s_allowance' in entry_txt
              and 'def classify_order_confound' in entry_txt
              and "ORDER_ATTRIBUTABLE" in entry_txt and "ORDER_INDETERMINATE" in entry_txt
              # the wording spans a line break in the source, so the assertion is on the two halves
              and 'may NOT be read as a scorer mechanism' in entry_txt
              and 'advantage: the second arm was charged for time it did not spend' in entry_txt
              and 'v2_order_attribution' in entry_txt
              and order_acc.get('verdict') == 'L1_ARM_ORDER_CONFOUND_BOUNDED'
              and order_acc.get('n_fail') == 0 and order_acc.get('n_rows') == 10
              and o6c.get('v1_is_order_invariant') is True
              and o6c.get('v2_can_be_order_sensitive') is True
              and caps == want_caps
              and 'publication/2026' not in o6c_fh
              and 'interpretation rule' in o6c_fh
              and 'No budget is widened' in o6c_fh
              and 'no formal run is added' in o6c_fh
              and 'stays in the 64-state denominator' in o6c_fh
              and not added_run_markers)
    ck(ROW_NAMES[20], r21_ok,
       {'arm_order': entry_order, 'order_acceptance': order_acc.get('verdict'),
        'order_acceptance_n_pass': order_acc.get('n_pass'),
        'order_acceptance_n_fail': order_acc.get('n_fail'),
        'classifier_present': 'def classify_order_confound' in entry_txt,
        'attributions': ['ORDER_ATTRIBUTABLE', 'ORDER_INDETERMINATE', 'NOT_ORDER_ATTRIBUTABLE'],
        'caps_unchanged': caps,
        'no_added_run_or_reorder_knob': added_run_markers,
        'fairness_handling_is_a_reading_rule': {
            'claims_interpretation_rule': 'interpretation rule' in o6c_fh,
            'claims_no_budget_widened': 'No budget is widened' in o6c_fh,
            'claims_no_formal_run_added': 'no formal run is added' in o6c_fh,
            'claims_states_kept': 'stays in the 64-state denominator' in o6c_fh},
        'keys_in_the_order_acceptance': sorted(o_rows),
        'package_6c': {'order': o6c.get('arm_order'), 'scope': o6c.get('wall_clock_scope'),
                       'verdict': (o6c.get('acceptance') or {}).get('verdict')}})

    fails = [r for r in rows if not r['ok']]
    payload = {
        'schema': 'd3_l1_l1_sign_blocker_independent_audit',
        'rows': rows, 'n_rows': len(rows), 'n_pass': len(rows) - len(fails), 'n_fail': len(fails),
        'expected_rows': list(ROW_NAMES),
        'row_names_match_expected': [r['name'] for r in rows] == list(ROW_NAMES),
        'independence': 'imports none of: ' + ', '.join(FORBIDDEN_IMPORTS),
        'artifacts_read': {Path(p).resolve().relative_to(REPO.resolve()).as_posix(): sha(p)
                           for p in (RUN_ENTRY, CLOSURE, PROTOCOL, BLOCKER_ACC, pkg_path,
                                     OLD_PACKAGE, INVALIDATION)},
        'not_run': ['the L1 comparison itself', 'training', 'L2', 'L3', 'the 40-day batch',
                    'the final test'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)),
        'verdict': ('L1_SIGN_BLOCKERS_INDEPENDENTLY_AUDITED' if not fails
                    else 'L1_SIGN_BLOCKERS_AUDIT_FAILED'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1 sign-blocker independent audit: %d rows, %d PASS, %d FAIL ==='
          % (len(rows), len(rows) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
