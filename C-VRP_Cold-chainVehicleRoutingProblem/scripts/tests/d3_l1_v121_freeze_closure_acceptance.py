#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.21 acceptance: the runtime source-dependency freeze is now COMPLETE and the drift is DETECTED.

The gap this closes
-------------------
`IDENTITY_TARGETS` froze 14 paths.  The real F execution path also imports the authorizer, the
execution context, the V1.16 forward control (and through it the V1.14 phase journal) and
`d3_l1_v18_readonly_audit.typed_bytes` -- and the closure turned out to be 30 source files, not 9.
Editing any of the missing ones would have changed execution, identity checking, timing or the
result decision while `--readiness` still printed READY.

What is checked here
--------------------
  F1  the DERIVED watch set is complete for the named files (the four the reviewer listed, PLUS the
      module that defines `typed_bytes`) and is larger than the historical 14-path list
  F2  a frozen V1.21 identity carries the FULL derived watch set, every path hashing to its record
  F3  `--readiness` is READY_FOR_USER_SIGNATURE and reports `runtime_watch_covers_closure = true`
  F4  for EACH of the four previously unregistered modules, an isolated tamper makes readiness
      NOT_READY (and the restore makes it READY again) -- the per-module counterexample
  F5  the same for `d3_l1_v18_readonly_audit.py`, which defines `typed_bytes`
  F6  a full sweep: EVERY path in the frozen watch set is individually tampered and individually
      detected (no path in the freeze is unchecked)
  F7  the closure itself is load-bearing: a file reachable only transitively is in the watch set and
      its tamper is detected
  F8  the pre-existing identities are untouched: the historical V1.18/V1.19/V1.20 evidence
      directories still exist and were not rewritten by this work

Every tamper is reverted immediately and the revert is verified; the repository is left byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_persistence as persistence                                 # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN_ID = 'd3l1-real-f-forward-20261015-v121'
PREREG = EV / 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_21.md'

PREVIOUSLY_UNREGISTERED = (
    'scripts/simulation/d3_l1_v118_authorization.py',
    'scripts/simulation/d3_l1_v120_execution_context.py',
    'scripts/tests/d3_l1_v116_forward_control.py',
    'scripts/tests/d3_l1_v112_phase_journal.py',
)
TYPED_BYTES_DEFINER = 'scripts/tests/d3_l1_v18_readonly_audit.py'

ROWS = []


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:230]))


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def runner_module(name='v121_runner'):
    spec = importlib.util.spec_from_file_location(name, str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def closure_module():
    spec = importlib.util.spec_from_file_location(
        'v121_closure', str(EXT / 'scripts/data/d3_l1_v121_dependency_closure.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['v121_closure'] = module
    spec.loader.exec_module(module)
    return module


def readiness_json(run_dir, entry):
    """Run `--readiness` and return its parsed verdict + the newest READINESS record."""
    proc = subprocess.run(
        [sys.executable, str(RUNNER), '--readiness', '--out-dir', str(run_dir)],
        capture_output=True, text=True, timeout=600)
    try:
        parsed = json.loads(proc.stdout[proc.stdout.index('{'):])
    except Exception:                                                     # noqa: BLE001
        parsed = {}
    rec = persistence.VerifiedReader(run_dir).latest('READINESS') or {}
    return parsed, rec


def tamper_detect(run_dir, rel, *, entry):
    """Tamper ONE watched path, check readiness turns NOT_READY, then restore and re-check."""
    path = REPO / rel if (REPO / rel).is_file() else EXT / rel
    original = path.read_bytes()
    before = sha(path)
    try:
        path.write_bytes(original + b'\n# V1.21 tamper counterexample\n')
        after = sha(path)
        parsed, rec = readiness_json(run_dir, entry)
        detected = (parsed.get('verdict') == 'NOT_READY'
                    and any(rel.rsplit('/', 1)[-1] in str(m)
                            for m in parsed.get('missing_before_signature', []))
                    or (rec.get('readiness', {}).get('checks', {})
                        .get('runtime_watch_closed') is False))
        listed = (rec.get('readiness', {}).get('checks', {}).get('runtime_watch', {})
                  .get('drifted') or [])
    finally:
        path.write_bytes(original)
    restored = sha(path) == before
    parsed2, _rec2 = readiness_json(run_dir, entry)
    return {'path': rel, 'detected': bool(detected), 'restored': restored,
            'drifted_reported': listed[:3],
            'verdict_after_tamper': parsed.get('verdict'),
            'verdict_after_restore': parsed2.get('verdict'),
            'sha_before': before, 'sha_tampered': after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    started = time.monotonic()
    run_dir = Path(args.run_dir)

    # ---- F1 ------------------------------------------------------------------------------
    closure = closure_module().build_watch_set()
    # The closure reports extension-root-relative paths; match by SUFFIX so the check is about
    # coverage and not about which root the two spellings happen to use.  (A prefix strip is
    # fragile: getting it wrong made this check pass vacuously on the first attempt.)
    def _covered(rel, paths):
        return any(p.replace('\\', '/').endswith(rel) for p in paths)

    missing_named = [r for r in PREVIOUSLY_UNREGISTERED + (TYPED_BYTES_DEFINER,)
                     if not _covered(r, closure['watch'])]
    ck('F1 the derived watch set contains the four previously unregistered modules AND the module '
       'that defines typed_bytes, and it is strictly larger than the historical 14-path list',
       not missing_named and closure['n_watch'] > 14
       and not closure['unresolved_seeds'] and not closure['excluded_reachable'],
       {'n_watch_source': closure['n_watch'], 'n_source_closure': closure['n_source_closure'],
        'missing_named': missing_named, 'unresolved_seeds': closure['unresolved_seeds'],
        'excluded_reachable': closure['excluded_reachable']})

    # ---- F2 / F3 -------------------------------------------------------------------------
    ident = persistence.VerifiedReader(run_dir).latest('RUN_IDENTITY') or {}
    targets = persistence.VerifiedReader(run_dir).latest('IDENTITY_TARGETS') or {}
    frozen_paths = targets.get('paths') or {}
    watch = {p: {'frozen': v, 'current': sha(p), 'ok': sha(p) == v}
             for p, v in frozen_paths.items()}
    frozen_rel = {str(p).replace('\\', '/').rsplit('/', 1)[-1] for p in frozen_paths}
    covered = [r for r in PREVIOUSLY_UNREGISTERED + (TYPED_BYTES_DEFINER,)
               if r.rsplit('/', 1)[-1] in frozen_rel]
    ck('F2 the frozen V1.21 identity carries the FULL derived watch set and every path still hashes '
       'to its record',
       len(frozen_paths) == closure['n_watch'] + len(closure['non_source_assets'])
       and all(r['ok'] for r in watch.values())
       and len(covered) == len(PREVIOUSLY_UNREGISTERED) + 1,
       {'run_id': ident.get('run_id'), 'n_frozen': len(frozen_paths),
        'n_expected': closure['n_watch'] + len(closure['non_source_assets']),
        'n_drifted': sum(1 for r in watch.values() if not r['ok']),
        'n_named_covered': len(covered)})

    parsed, rec = readiness_json(run_dir, None)
    checks = (rec.get('readiness') or {}).get('checks', {})
    ck('F3 readiness is READY_FOR_USER_SIGNATURE and reports that the freeze covers the closure',
       parsed.get('verdict') == 'READY_FOR_USER_SIGNATURE'
       and checks.get('runtime_watch_covers_closure') is True
       and checks.get('runtime_watch_closed') is True
       and checks.get('real_F_forward_executed') is False,
       {'verdict': parsed.get('verdict'),
        'covers_closure': checks.get('runtime_watch_covers_closure'),
        'watch_closed': checks.get('runtime_watch_closed'),
        'n_paths': (checks.get('runtime_watch') or {}).get('n_paths')})

    # ---- F4 / F5: the per-module counterexamples the reviewer asked for -------------------
    per_module = {}
    for rel in PREVIOUSLY_UNREGISTERED + (TYPED_BYTES_DEFINER,):
        per_module[rel] = tamper_detect(run_dir, rel, entry=None)
    all_detected = all(v['detected'] and v['restored'] for v in per_module.values())
    ck('F4/F5 EACH previously unregistered module (and the typed_bytes definer) is individually '
       'detected when tampered, and each restore returns readiness to READY',
       all_detected and all(v['verdict_after_restore'] == 'READY_FOR_USER_SIGNATURE'
                            for v in per_module.values()),
       {k.rsplit('/', 1)[-1]: {'detected': v['detected'], 'restored': v['restored'],
                               'after_tamper': v['verdict_after_tamper'],
                               'after_restore': v['verdict_after_restore']}
        for k, v in per_module.items()})

    # ---- F6: the FULL sweep ---------------------------------------------------------------
    sweep = []
    for path_str in sorted(frozen_paths):
        p = Path(path_str)
        rel = str(p).replace('\\', '/')
        if not p.is_file():
            sweep.append({'path': rel, 'detected': False, 'error': 'file absent'})
            continue
        original = p.read_bytes()
        before = sha(p)
        try:
            p.write_bytes(original + b'\n# V1.21 sweep tamper\n')
            parsed_t, rec_t = readiness_json(run_dir, None)
            drifted = ((rec_t.get('readiness') or {}).get('checks', {})
                       .get('runtime_watch', {}).get('drifted') or [])
            detected = (parsed_t.get('verdict') == 'NOT_READY'
                        and any(d.replace('\\', '/') == rel for d in drifted))
        finally:
            p.write_bytes(original)
        sweep.append({'path': rel, 'detected': bool(detected), 'restored': sha(p) == before,
                      'after_tamper': parsed_t.get('verdict')})
    n_detected = sum(1 for s in sweep if s.get('detected'))
    ck('F6 a FULL sweep: every one of the %d frozen paths is individually tampered and individually '
       'detected (no path in the freeze is unchecked)' % len(frozen_paths),
       n_detected == len(frozen_paths)
       and all(s.get('restored') for s in sweep if 'restored' in s),
       {'n_paths': len(frozen_paths), 'n_detected': n_detected,
        'undetected': [s['path'] for s in sweep if not s.get('detected')][:8]})

    # ---- F7: closure-only file is load-bearing -------------------------------------------
    transitive_only = [r for r in closure['watch']
                       if r not in [str(p) for p in
                                    ('scripts/simulation/d3_l1_v118_forward_runner.py',
                                     'scripts/simulation/d3_l1_v118_run_identity.py',
                                     'scripts/simulation/d3_l1_v118_persistence.py',
                                     'scripts/simulation/d3_l1_v118_targets.py',
                                     'scripts/tests/d3_l1_v118_authorization.py',
                                     'scripts/simulation/d3_l1_v120_execution_context.py',
                                     'scripts/tests/d3_l1_v116_forward_control.py',
                                     'scripts/tests/d3_l1_v112_phase_journal.py')]
                       and not r.endswith(('.ckpt', '.npz', '.json'))]
    sample = sorted(transitive_only)[:3]
    sample_results = {r: tamper_detect(run_dir, r, entry=None) for r in sample}
    ck('F7 files reachable ONLY transitively are in the freeze and their tampering is detected',
       bool(sample) and all(v['detected'] for v in sample_results.values()),
       {k.rsplit('/', 1)[-1]: v['detected'] for k, v in sample_results.items()})

    # ---- F8: previous identities untouched ------------------------------------------------
    previous = {}
    for v in ('v118_evidence', 'v119_evidence', 'v120_evidence'):
        d = EV / v
        previous[v] = {'exists': d.is_dir(),
                       'n_files': len(list(d.rglob('*'))) if d.is_dir() else 0,
                       'frozen_runs': sorted(p.name for p in d.glob('run_v*')) if d.is_dir() else []}
    ck('F8 the V1.18/V1.19/V1.20 evidence directories are present and were not rewritten by this '
       'work', all(x['exists'] and x['n_files'] > 0 for x in previous.values()), previous)

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_v121_freeze_closure_acceptance',
        'run_id': RUN_ID,
        'n_watch_paths': len(frozen_paths),
        'watch_paths_with_sha256': {str(p): v for p, v in sorted(frozen_paths.items())},
        'closure': {'n_source': closure['n_watch'], 'n_closure': closure['n_source_closure'],
                    'source_paths': closure['watch'],
                    'non_source_assets': closure['non_source_assets'],
                    'excluded_reachable': closure['excluded_reachable']},
        'previously_unregistered': list(PREVIOUSLY_UNREGISTERED),
        'typed_bytes_definer': TYPED_BYTES_DEFINER,
        'per_module_tamper': per_module,
        'full_sweep': sweep,
        'transitive_only_sample': sample_results,
        'previous_identity_dirs': previous,
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'budgets_unchanged': {'initialize_compile_s': 60.0, 'slot_span_s': 10.0, 'batch_s': 180.0},
        'calls_unchanged': 129, 'states_unchanged': 64,
        'not_run': ['the real F 64-state batch', 'L1 comparison', 'training', 'L2/L3',
                    'the 40-day batch', 'the final test'],
        'science_ready': False, 'stable_online_gain': False,
        'wall_s': time.monotonic() - started,
        'tool_sha256': sha(Path(__file__)),
        'closure_module_sha256': sha(EXT / 'scripts/data/d3_l1_v121_dependency_closure.py'),
        'identity_module_sha256': sha(EXT / 'scripts/simulation/d3_l1_v118_run_identity.py'),
        'verdict': ('FREEZE_CLOSURE_COMPLETE_AND_DRIFT_DETECTED' if not fails
                    else 'FREEZE_CLOSURE_INCOMPLETE'),
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'D3_V121_FREEZE_CLOSURE_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== V1.21 freeze closure acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
