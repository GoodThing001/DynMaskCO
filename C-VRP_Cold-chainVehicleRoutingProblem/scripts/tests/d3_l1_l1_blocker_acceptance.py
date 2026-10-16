#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 SIGN-BLOCKER ACCEPTANCE: executable counterexamples for the three reviewed defects.

Each row is a FALSIFICATION of the defect, not a restatement of the fix.

A -- SIGNING FREEZE MUST COVER L1'S OWN SOURCES
  A1  the derived watch set contains every decision-relevant L1 module BY NAME
  A2  each of those modules is registered in the frozen identity with a digest that matches disk
  A3  per-module TAMPER: a single edited byte in module M makes M's digest differ from the frozen one
  A4  per-module TAMPER: that drift makes the frozen identity NON-MATCHING (so the credential cannot
      be sealed against it)
  A5  per-module TAMPER: `validate_credential` REFUSES and `authorize` raises BEFORE any consumption
      receipt is written -- i.e. drift blocks authorization CONSUMPTION, not merely a later step
  A6  per-module TAMPER: the guarded worker, given a VALID execution context, still refuses with
      IDENTITY DRIFT and writes ZERO per-state records -- i.e. drift blocks the START of the real
      comparison
  A7  the counterexample is real: with the module restored the same worker reaches its guard PASS

B -- READY AND PER-STATE SUPERVISION
  B1  the supervisor's probe is the immutable-record probe, NOT the legacy fixed-name default
  B2  init done FAST then batch SLOW: a worker that publishes READY at +2 s and then runs 70 s is NOT
      killed for the 60 s init cap, and is not killed either while the batch stays under 180 s
  B3  init SLOW: a worker that never publishes READY is force-killed by the 60 s init cap
  B4  one state exceeding the pre-registered 10 s: the run degrades per the EXISTING semantics (the
      arm fails closed with `timeout` and the state keeps its record) rather than being silently
      truncated
  B5  the 10 s is the allowance for the WHOLE STATE, not per arm: both arms' gates are charged from
      ONE monotonic origin
  B6  per-state start/end records persist and a state completed before a kill keeps its record
  B7  the frozen 60 / 10 / 180 s caps are untouched

C -- THE 64-STATE DENOMINATOR
  C1  a fault injected into ONE state leaves the denominator at 64, with that state REPORTED
  C2  exception / timeout / empty-pool / not-comparable / not-attempted are listed SEPARATELY
  C3  `ledger_ok=false` forces a BLOCKED round and can never be promoted to a formal PASS
  C4  a missing state forces a BLOCKED round (the denominator may not shrink)
  C5  an identity (single-factor) failure forces a BLOCKED round
  C6  the pre-fix filter is gone: the adjudication is computed over ALL records

Every worker here is a STAND-IN that speaks the real READY / phase / caps protocol and never loads
F or D-Geo.  Nothing in this file is evidence about MaskCO, D-Geo, or any scientific question.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

RUNNER = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
PREREG = (REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1'
          '/a1_d3_l1_prereg_root_v1/A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md')
RUN_ID = 'd3l1-l1-two-arm-20261015-l1v1'
ROWS = []


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:200]))


def repo_rel_str(path):
    """Repository-relative, forward-slashed -- the ONE spelling the watch set and closure use."""
    return Path(path).resolve().relative_to(REPO.resolve()).as_posix()


def _twin_ext_of(entry):
    """The twin's extension root, read off the entry module instead of counting `parents`.

    `Path.parents` indexing is how the twin path first resolved to `<twin>/C-.../simulation` instead
    of `<twin>/C-.../scripts/simulation`.  The entry module has already resolved its own roots.
    """
    return Path(entry.EXT)


def _twin_ext_of_entry(runner_path):
    """The twin's extension root from the runner path: `<root>/twin/<EXT_NAME>/scripts/...`."""
    return Path(runner_path).parents[2]


class twin_namespace(object):
    """Bind a twin tree's modules to THEIR OWN names, so a twin load cannot fall through to the repo.

    Every L1 module imports its peers by BARE name (`import d3_l1_closure`, `import
    d3_l1_v118_authorization`, ...).  The harness process already has the REPOSITORY's `scripts/`
    directories on `sys.path` and those modules in `sys.modules`, so a twin entry that mutates a
    twin file would otherwise still import the REPOSITORY's copy of everything it imports -- and the
    counterexample would measure nothing.  That is not a hypothetical: it is why the first run of
    this harness reported `n_drifted == 0` on a file it had just appended a byte to.

    Entering evicts every module that lives under the repository's `scripts/` tree and puts the
    twin's directories at the front of `sys.path`; leaving restores `sys.modules` exactly, so the
    harness's own imported objects are never replaced by a twin's.
    """

    def __init__(self, twin_ext):
        self.twin_ext = Path(twin_ext)
        self._snapshot = None
        self._path = None

    def __enter__(self):
        self._snapshot = dict(sys.modules)
        repo_scripts = str(REPO / 'C-VRP_Cold-chainVehicleRoutingProblem' / 'scripts')
        evicted = []
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if f and str(f).startswith(repo_scripts):
                evicted.append(name)
                del sys.modules[name]
        self.evicted = evicted
        self._path = list(sys.path)
        for sub in ('scripts/simulation', 'scripts/tests', 'scripts/data', 'scripts/models',
                    'scripts/coldchain', 'scripts'):
            p = str(self.twin_ext / sub)
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        return self

    def __exit__(self, *exc):
        sys.modules.clear()
        sys.modules.update(self._snapshot)
        sys.path[:] = self._path
        return False

    def load_entry(self, runner_path, name):
        spec = importlib.util.spec_from_file_location(name, str(runner_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def runner_module(name='l1_blocker_entry'):
    spec = importlib.util.spec_from_file_location(name, str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================================
# a STAND-IN worker that speaks the production protocol without loading F or D-Geo
# =============================================================================================
STANDIN = r'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STAND-IN worker for the L1 supervision acceptance.  Loads no model and scores nothing.

It publishes exactly the records the real worker publishes -- `identity_start`, `initialize`,
`compile`, an immutable `L1_READY`, one `slot`/`slot_finished` pair per pre-registered state, and
`completed` -- so the supervisor's READY probe, per-state deadline and timeline audit are exercised
against the REAL protocol rather than against a mock of the supervisor.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--states-json", required=True)
    ap.add_argument("--entry", required=True,
                    help="path to d3_l1_l1_formal_run.py, used ONLY to locate scripts/ so the "
                         "stand-in can import the production persistence/journal modules")
    ap.add_argument("--ready-sleep-s", type=float, default=0.2)
    ap.add_argument("--slot-sleep-s", type=float, default=0.02)
    ap.add_argument("--slot-sleep-at", type=int, default=-1)
    ap.add_argument("--slot-sleep-long-s", type=float, default=0.0)
    ap.add_argument("--publish-ready", type=int, default=1)
    args = ap.parse_args()
    # the stand-in lives in a scratch directory, so the production modules are located from the
    # entry path rather than from `__file__`
    scripts = Path(args.entry).resolve().parents[1]
    for _p in (scripts, scripts / "simulation", scripts / "tests", scripts / "data"):
        sys.path.insert(0, str(_p))
    import d3_l1_v118_persistence as persistence
    from d3_l1_v112_phase_journal import PhaseJournal

    out = Path(args.run_dir)
    states = json.loads(Path(args.states_json).read_text(encoding="utf-8"))
    journal = PhaseJournal(out / "phases")
    started = time.monotonic()
    journal.publish("identity_start", run_id="standin")
    journal.publish("initialize")
    time.sleep(args.ready_sleep_s)
    journal.publish("compile")
    if args.publish_ready:
        persistence.publish_new(out, "L1_READY",
                                {"run_id": "standin", "mono": time.monotonic(),
                                 "kind": "initialization_and_compilation_finished",
                                 "n_warmup_calls": 1,
                                 "warmup_scores_sha256": hashlib.sha256(b"standin").hexdigest(),
                                 "initialize_and_compile_s": time.monotonic() - started})
    records = []
    for i, s in enumerate(states):
        day, pos = int(s["day_id"]), int(s["position"])
        journal.publish("slot", day_id=day, position=pos)
        if i == args.slot_sleep_at and args.slot_sleep_long_s:
            time.sleep(args.slot_sleep_long_s)
        else:
            time.sleep(args.slot_sleep_s)
        records.append({"day_id": day, "position": pos, "status": "COMPLETED"})
        journal.publish("slot_finished", day_id=day, position=pos, status="COMPLETED")
    persistence.write_once(out / "STANDIN_STATE_RECORDS.jsonl",
                           "\n".join(json.dumps(r) for r in records).encode("utf-8"))
    journal.publish("completed", run_id="standin")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
'''


# =============================================================================================
# A. the freeze covers L1's own sources, and drift blocks consumption AND start
# =============================================================================================
def _build_copy_tree(root):
    """A DISPOSABLE twin of the extension: real source COPIED, bulky assets JUNCTIONED.

    Tampering the live repository to prove tamper-detection would be an unacceptable way to test a
    freeze, so the counterexample runs in a twin tree.  `scripts/` is a real copy (that is the part
    being edited); `data/`, `results/`, `research/` and `MASKCO_code/` are junctions or hardlinks to
    the originals, so the frozen assets keep their real digests while nothing in the repository is
    written to.  The twin is a faithful stand-in: the ONLY difference from the repository is the
    tampered byte the counterexample injects.
    """
    twin_ext = root / 'twin' / 'C-VRP_Cold-chainVehicleRoutingProblem'
    twin_repo = root / 'twin'
    twin_ext.mkdir(parents=True, exist_ok=True)
    shutil.copytree(EXT / 'scripts', twin_ext / 'scripts',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('data', 'results'):
        src = EXT / name
        if src.is_dir():
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(twin_ext / name), str(src)],
                           capture_output=True, text=True, timeout=120)
    # The V1.22 evidence tree and the upstream `MASKCO_code/` package live under the repository root.
    # `MASKCO_code/` is junctioned WHOLE rather than reconstructed: the F scorer imports
    # `training.load_ckpt` out of it, so a backbone-only stand-in fails at model load with
    # `ImportError: cannot import name 'load_ckpt' from 'training'` -- which is a property of the
    # fixture, not of the guard under test.
    for name in ('research', 'MASKCO_code'):
        if (REPO / name).is_dir():
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(twin_repo / name), str(REPO / name)],
                           capture_output=True, text=True, timeout=120)
    return twin_repo, twin_ext / 'scripts' / 'simulation' / 'd3_l1_l1_formal_run.py'


def blocker_a(root, module):
    import d3_l1_closure as closure
    import d3_l1_v118_authorization as AUTH

    _, twin_runner = _build_copy_tree(root)
    if not twin_runner.is_file():
        raise SystemExit('the twin tree was not built: %s' % twin_runner)

    ident = module.build_l1_identity(created_utc='2026-01-01T00:00:00Z')
    posix = {p.replace('\\', '/'): d for p, d in ident['watch'].items()}
    critical = list(closure.critical_l1_sources())

    # A1 ------------------------------------------------------------------------------------
    named = {c: [p for p in posix if p.endswith(c)] for c in critical}
    a1_ok = all(len(v) == 1 for v in named.values())
    ck('A1 the derived L1 watch set contains every decision-relevant L1 module BY NAME',
       a1_ok, {'n_critical': len(critical),
               'n_uniquely_frozen': sum(1 for v in named.values() if len(v) == 1),
               'n_watch_total': len(ident['watch']),
               'by_module': {c.split("/")[-1]: (v[0].split("/")[-1] if v else None)
                             for c, v in named.items()},
               'pre_fix_l1_sources_in_closure': 0,
               'l1_sources_in_closure_now': sum(1 for v in named.values() if v)})

    # A2 ------------------------------------------------------------------------------------
    wrong = {c: {'frozen': posix[[p for p in posix if p.endswith(c)][0]], 'disk': sha(REPO / c)}
             for c in critical}
    a2_bad = {c: v for c, v in wrong.items() if v['frozen'] != v['disk']}
    ck('A2 every critical L1 module is registered in the frozen identity with the digest on disk',
       not a2_bad, {'n_checked': len(critical), 'n_mismatched': len(a2_bad),
                    'mismatched': sorted(a2_bad)})

    repo_before = {c: sha(REPO / c) for c in critical}

    # A3-A7: ONE tampered module at a time, in the TWIN tree ---------------------------------
    # `critical_l1_sources()` speaks REPOSITORY-relative paths (the same spelling the watch set
    # uses), so the extension-relative tail is what gets joined onto the twin's extension root.
    ext_prefix = repo_rel_str(EXT) + '/'
    tamper_rows, a3_ok, a4_ok, a5_ok, a6_ok, a7_ok = [], True, True, True, True, True
    for rel in critical:
        name = rel.split('/')[-1]
        rel_ext = rel[len(ext_prefix):] if rel.startswith(ext_prefix) else rel
        twin_src = root / 'twin' / 'C-VRP_Cold-chainVehicleRoutingProblem' / rel_ext
        original = twin_src.read_bytes()
        row = {'module': name, 'twin_path_exists': twin_src.is_file()}
        try:
            # ORDER MATTERS.  The identity is frozen from the CLEAN twin FIRST -- that is the state a
            # human signs and a credential binds -- and only THEN is the module tampered.  Freezing
            # after tampering would record the tampered digest, make the identity self-consistent, and
            # measure nothing.  That mistake is exactly what the first two runs of this harness made.
            with twin_namespace(_twin_ext_of_entry(twin_runner)) as ns:
                twin = ns.load_entry(twin_runner, 'twin_entry_' + name.replace('.', '_'))
                run_dir = root / ('tamper_run_' + uuid.uuid4().hex)
                twin.freeze_identity(run_dir, RUN_ID)              # <-- CLEAN freeze
                frozen = twin.load_frozen(run_dir)
                row['A4_frozen_watch_is_twin'] = all(
                    str(p).startswith(str(root)) for p in frozen['watch'])
                pre_drift = [p for p, want in frozen['watch'].items() if sha(p) != want]
                row['A4_drift_before_tamper'] = len(pre_drift)

                # seal the credential against the CLEAN frozen identity, i.e. what a signer would
                # approve, and publish nothing yet
                import importlib.util as _ilu
                twin_scripts = Path(twin.EXT) / 'scripts'
                spec_a = _ilu.spec_from_file_location(
                    'twin_auth', str(twin_scripts / 'simulation' / 'd3_l1_v118_authorization.py'))
                twin_auth = _ilu.module_from_spec(spec_a)
                sys.modules['twin_auth'] = twin_auth
                spec_a.loader.exec_module(twin_auth)
                view = twin.authorization_identity_view(frozen)
                targets = twin.authorization_targets_view(frozen)
                cred = twin_auth.seal_template(dict(
                    twin_auth.build_template(view, targets,
                                             preregistration_sha256=twin.sha(PREREG),
                                             output_dir=run_dir),
                    approved_by='BLOCKER-A TAMPER FIXTURE',
                    approved_utc='2026-10-15T00:00:00Z',
                    approval_statement='tamper counterexample fixture; NOT a real human approval'))
                cred_path = root / ('tamper_cred_' + uuid.uuid4().hex + '.json')
                cred_path.write_text(json.dumps(cred, indent=2, sort_keys=True, default=repr),
                                     encoding='utf-8')

            # ---- NOW tamper, after the freeze and the seal ----
            tampered = original + b'\n# L1 BLOCKER-A TAMPER COUNTEREXAMPLE\n'
            twin_src.write_bytes(tampered)
            row['tampered_sha256'] = hashlib.sha256(tampered).hexdigest()
            row['frozen_sha256'] = posix[[p for p in posix if p.endswith(rel)][0]]
            row['A3_digest_differs'] = row['tampered_sha256'] != row['frozen_sha256']
            a3_ok = a3_ok and row['A3_digest_differs']

            with twin_namespace(_twin_ext_of_entry(twin_runner)) as ns2:
                twin = ns2.load_entry(twin_runner, 'twin_check_' + name.replace('.', '_'))
                frozen = twin.load_frozen(run_dir)

                # A4: the identity the credential was SEALED against no longer matches disk
                drift = [p for p, want in frozen['watch'].items() if sha(p) != want]
                row['A4_n_drifted'] = len(drift)
                row['A4_names_tampered_module'] = any(p.endswith(name) for p in drift)
                a4_ok = a4_ok and row['A4_drift_before_tamper'] == 0 \
                    and row['A4_n_drifted'] == 1 and row['A4_names_tampered_module'] \
                    and row['A4_frozen_watch_is_twin']

                # A5: authorization CONSUMPTION of that already-sealed credential is refused, and
                # no receipt is written.  The twin's own authorizer is used, so a fixture in the twin
                # can never reach the repository's credential machinery.
                import importlib.util as _ilu5
                twin_scripts = Path(twin.EXT) / 'scripts'
                spec_a5 = _ilu5.spec_from_file_location(
                    'twin_auth5', str(twin_scripts / 'simulation' / 'd3_l1_v118_authorization.py'))
                twin_auth = _ilu5.module_from_spec(spec_a5)
                sys.modules['twin_auth5'] = twin_auth
                spec_a5.loader.exec_module(twin_auth)
                consumption = root / ('tamper_consumption_' + uuid.uuid4().hex)
                # THE POINT OF A5: the supervisor authorises over the LIVE identity -- the frozen
                # record with every watched digest re-derived from disk -- not over the record alone.
                # Handing `validate_credential` the record is what let a credential be CONSUMED for
                # code that no longer existed; the live view is what makes the drift bite.
                live, live_drift = twin.live_identity_view(frozen)
                row['A5_live_drift_inside_worker_path'] = len(live_drift)
                view = twin.authorization_identity_view(live)
                targets = twin.authorization_targets_view(frozen)
                row['A5_live_watch_differs_from_frozen'] = any(
                    live['watch'].get(p) != frozen['watch'].get(p) for p in frozen['watch'])
                # and, as the counterexample to the fix, show the RECORD-ONLY view does NOT refuse:
                # this is the pre-fix behaviour, recorded rather than described
                try:
                    ok_rec, _det_rec = twin_auth.validate_credential(
                        twin_auth.load_credential(cred_path),
                        identity=twin.authorization_identity_view(frozen),
                        forward_targets=targets, output_dir=run_dir,
                        preregistration_sha256=twin.sha(PREREG))
                    row['A5_record_only_validate_ok_PREFIX_BEHAVIOUR'] = bool(ok_rec)
                except BaseException as exc:                                  # noqa: BLE001
                    row['A5_record_only_validate_ok_PREFIX_BEHAVIOUR'] = '%s: %s' % (
                        type(exc).__name__, exc)
                row['A5_validate_ok'] = None
                row['A5_authorize_raised'] = None
                try:
                    ok, detail = twin_auth.validate_credential(
                        twin_auth.load_credential(cred_path), identity=view,
                        forward_targets=targets, output_dir=run_dir,
                        preregistration_sha256=twin.sha(PREREG))
                    row['A5_validate_ok'] = bool(ok)
                    row['A5_validate_problems'] = (detail.get('problems') or [])[:3]
                    try:
                        twin_auth.authorize(twin_auth.load_credential(cred_path), identity=view,
                                            forward_targets=targets, output_dir=run_dir,
                                            preregistration_path=PREREG, run_dir=run_dir,
                                            consumption_root=consumption)
                        row['A5_authorize_raised'] = False
                    except twin_auth.AuthorizationRefused as exc:
                        row['A5_authorize_raised'] = True
                        row['A5_authorize_reason'] = str(exc)[:160]
                except twin_auth.AuthorizationRefused as exc:
                    row['A5_validate_ok'] = False
                    row['A5_authorize_raised'] = True
                    row['A5_authorize_reason'] = 'validate_credential refused: %s' % str(exc)[:120]
                row['A5_n_receipts'] = len(sorted(
                    consumption.glob('authorization_consumption_*.json'))) \
                    if consumption.is_dir() else 0
                a5_ok = a5_ok and row['A5_live_drift_inside_worker_path'] == 1 \
                    and row['A5_live_watch_differs_from_frozen'] is True \
                    and row['A5_record_only_validate_ok_PREFIX_BEHAVIOUR'] is True \
                    and row['A5_validate_ok'] is False \
                    and row['A5_authorize_raised'] is True and row['A5_n_receipts'] == 0

            # A6: the tamper is applied AFTER the freeze, then a VALID execution context is offered.
            # ORDER: (1) restore the module so the twin is CLEAN, (2) freeze the run directory, (3)
            # build the execution context from that clean freeze, (4) tamper the already-frozen
            # module.  Freezing after tampering would record the tampered digest and the startup
            # check would have nothing to catch.
            start_dir = root / ('tamper_start_' + uuid.uuid4().hex)
            twin_src.write_bytes(original)
            with twin_namespace(_twin_ext_of_entry(twin_runner)) as ns6:
                twin = ns6.load_entry(twin_runner, 'twin_a6_' + name.replace('.', '_'))
                twin.freeze_identity(start_dir, RUN_ID)              # CLEAN freeze
                start_frozen = twin.load_frozen(start_dir)
                row['A6_drift_before_tamper'] = len(
                    [p for p, want in start_frozen['watch'].items() if sha(p) != want])
                ctx_env, ctx_meta = _publish_context(twin, root, start_dir, start_frozen,
                                                     'a6_' + name.replace('.', '_'))
                twin_ext_used = Path(twin.EXT)
            twin_src.write_bytes(tampered)
            row['A6_drift_before_worker'] = len(
                [p for p, want in start_frozen['watch'].items() if sha(p) != want])
            # the subprocess leaves the namespace: it must import the TWIN's tampered module
            proc = subprocess.run([sys.executable, str(twin_runner), '--worker', '--out-dir',
                                   str(start_dir)], capture_output=True, text=True, timeout=600,
                                  env=_twin_env(twin_ext_used, **ctx_env))
            row['A6_returncode'] = proc.returncode
            row['A6_drift_in_stderr'] = 'IDENTITY DRIFT' in proc.stderr
            drift_rec = start_dir / 'L1_IDENTITY_DRIFT.json'
            # the drift record carries full rows ({path, recorded, actual}), not bare strings
            drift_rows_rec = (json.loads(drift_rec.read_text(encoding='utf-8')).get('drifted')
                              or []) if drift_rec.is_file() else []
            row['A6_drift_record_names_module'] = any(
                str(r.get('path') or '').endswith(name) for r in drift_rows_rec)
            row['A6_n_state_records'] = 1 if (start_dir / 'L1_STATE_RECORDS.jsonl').exists() else 0
            row['A6_anything_loaded'] = any((start_dir / n).exists()
                                            for n in ('L1_WARMUP.json', 'L1_READY.json',
                                                      'L1_ADJUDICATION.json'))
            row['A6_context_was_valid'] = bool(ctx_env)
            row['A6_fixture_receipts'] = ctx_meta['receipts']
            a6_ok = a6_ok and row['A6_drift_before_tamper'] == 0 \
                and row['A6_drift_before_worker'] == 1 \
                and proc.returncode != 0 and row['A6_drift_in_stderr'] \
                and row['A6_drift_record_names_module'] and row['A6_n_state_records'] == 0 \
                and not row['A6_anything_loaded'] and row['A6_context_was_valid'] \
                and row['A6_fixture_receipts'] == 1
        finally:
            twin_src.write_bytes(original)
        # A7: with the module RESTORED the same guard reaches its PASS
        if rel == critical[0]:
            restored_dir = root / ('restored_' + uuid.uuid4().hex)
            with twin_namespace(_twin_ext_of_entry(twin_runner)) as ns7:
                twin2 = ns7.load_entry(twin_runner, 'twin_restored')
                twin2.freeze_identity(restored_dir, RUN_ID)
                ctx_env, _meta7 = _publish_context(twin2, root, restored_dir,
                                                   twin2.load_frozen(restored_dir), 'a7')
                twin_ext7 = Path(twin2.EXT)
            proc7 = subprocess.run([sys.executable, str(twin_runner), '--guard-probe', '--out-dir',
                                    str(restored_dir)], capture_output=True, text=True, timeout=300,
                                   env=_twin_env(twin_ext7, **ctx_env))
            a7_ok = proc7.returncode == 0 and '"guard": "PASSED"' in proc7.stdout
            row['A7_restored_guard_passed'] = a7_ok
        tamper_rows.append(row)

    # the repository itself must be untouched by the whole exercise
    repo_after = {c: sha(REPO / c) for c in critical}
    repo_intact = repo_before == repo_after

    ck('A3 PER-MODULE TAMPER: a single appended byte changes each critical module\'s digest',
       a3_ok, {'n_modules': len(critical),
               'n_differing': sum(1 for r in tamper_rows if r.get('A3_digest_differs'))})
    ck('A4 PER-MODULE TAMPER: the drift makes the frozen identity non-matching and names exactly '
       'that module', a4_ok,
       {'n_modules': len(critical),
        'rows': [{'m': r['module'], 'n_drifted': r.get('A4_n_drifted')}
                 for r in tamper_rows][:5]})
    ck('A5 PER-MODULE TAMPER: drift BLOCKS authorization consumption -- validate refuses, authorize '
       'raises, and ZERO consumption receipts are written',
       a5_ok, {'n_modules': len(critical),
               'example': {k: v for k, v in tamper_rows[0].items()
                           if k.startswith('A5')} if tamper_rows else None})
    ck('A6 PER-MODULE TAMPER: drift BLOCKS the start of the real comparison -- with a VALID '
       'execution context the worker still refuses, names the module, and writes no per-state record',
       a6_ok, {'n_modules': len(critical),
               'example': {k: v for k, v in tamper_rows[0].items()
                           if k.startswith('A6')} if tamper_rows else None})
    ck('A7 the counterexample is real, not a broken fixture: with the module RESTORED the guard '
       'PASSES', a7_ok, {'restored_guard_passed': a7_ok})
    ck('A8 the counterexample ran in a DISPOSABLE TWIN tree: every critical repository file is '
       'byte-identical before and after the tamper loop', repo_intact,
       {'n_critical': len(critical),
        'n_changed_in_repository': sum(1 for c in critical if repo_before[c] != repo_after[c]),
        'twin_scripts_is_a_copy': True,
        'twin_assets_are_junctions': ['data', 'results']})

    return {'A1': {'ok': a1_ok, 'critical': critical},
            'A2': {'ok': not a2_bad, 'mismatched': sorted(a2_bad)},
            'A3': {'ok': a3_ok}, 'A4': {'ok': a4_ok}, 'A5': {'ok': a5_ok}, 'A6': {'ok': a6_ok},
            'A7': {'ok': a7_ok}, 'A8': {'ok': repo_intact}, 'tamper_rows': tamper_rows}


def _twin_env(twin_ext, **extra):
    """The environment a TWIN subprocess needs: the twin's `scripts/` ahead of anything else.

    The twin entry inserts the twin's own directories, so the tampered module is the one Python
    imports -- but the harness process itself already has the real `scripts/` on `sys.path` and
    `PYTHONPATH` is inherited, so it must be pinned explicitly or the subprocess would import the
    repository copy and see no drift at all.
    """
    subdirs = ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data')
    pp = os.pathsep.join(str(Path(twin_ext) / d) for d in subdirs)
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = pp + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(extra)
    return env


def _publish_context(entry, root, run_dir, frozen, tag):
    """Publish a VALID execution context for `run_dir` in the TWIN, without touching production.

    `publish_execution_context` refuses to precede a consumption -- the context IS the proof that a
    credential was spent -- so this fixture builds a credential for the twin's own frozen identity,
    spends it into a THROWAWAY receipt directory, and publishes the context from that result.  The
    twin's authorization module is imported explicitly, so a fixture in the twin can never consume
    the repository's identity.
    """
    import importlib.util as _ilu
    # the entry module already resolved its own roots; reading them beats counting `parents`
    twin_scripts = Path(entry.EXT) / 'scripts'                     # <twin>/C-.../scripts
    twin_ext = Path(entry.EXT)
    twin_repo = Path(entry.REPO)
    auth = _ilu.module_from_spec(_ilu.spec_from_file_location(
        'twin_auth_' + tag, str(twin_scripts / 'simulation' / 'd3_l1_v118_authorization.py')))
    sys.modules[auth.__name__] = auth
    auth.__spec__.loader.exec_module(auth)
    execctx = _ilu.module_from_spec(_ilu.spec_from_file_location(
        'twin_ctx_' + tag, str(twin_scripts / 'simulation' / 'd3_l1_v120_execution_context.py')))
    sys.modules[execctx.__name__] = execctx
    execctx.__spec__.loader.exec_module(execctx)

    view = entry.authorization_identity_view(frozen)
    targets = entry.authorization_targets_view(frozen)
    cred = auth.seal_template(dict(
        auth.build_template(view, targets, preregistration_sha256=entry.sha(PREREG),
                            output_dir=Path(run_dir)),
        approved_by='BLOCKER ACCEPTANCE FIXTURE',
        approved_utc='2026-10-15T00:00:00Z',
        approval_statement='blocker acceptance fixture; NOT a real human approval and NOT the '
                           'production credential'))
    cred_path = root / ('ctx_cred_' + tag + '.json')
    cred_path.write_text(json.dumps(cred, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    consumption = root / ('ctx_consumption_' + tag)
    granted = auth.authorize(auth.load_credential(cred_path), identity=view, forward_targets=targets,
                             output_dir=Path(run_dir), preregistration_path=PREREG,
                             run_dir=Path(run_dir), consumption_root=consumption)
    secret = root / ('ctx_secret_' + tag)
    secret.write_bytes(os.urandom(32))
    try:
        os.chmod(secret, 0o600)
    except OSError:
        pass
    os.environ[execctx.SECRET_ENV] = str(secret)
    published = execctx.publish_execution_context(Path(run_dir), authorization_result=granted)
    env = dict(published['env'])
    env['D3_L1_EXECUTION_SECRET_PATH'] = str(secret)
    os.environ.update(env)
    published = execctx.publish_execution_context(Path(run_dir), authorization_result=granted)
    return published['env'], {'twin_ext': str(twin_ext), 'twin_repo': str(twin_repo),
                              'receipts': len(sorted(
                                  consumption.glob('authorization_consumption_*.json')))}


# =============================================================================================
# B. READY + per-state supervision
# =============================================================================================
def _arm_timeout_degrades_fail_closed():
    """The ARM layer of the 10 s rule, proved on a real two-arm run rather than asserted.

    A scorer that overruns its wall clock must make the arm fail CLOSED: `failure_reason='timeout'`,
    the ORIGINAL plan returned, `full_repair_completed=False`, and no certified plan.  This is the
    existing semantics the L1 state record reports as `TIMEOUT`; re-running it here keeps the blocker
    acceptance self-contained instead of citing another harness's row.
    """
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location('two_arm_acc',
                                            str(HERE / 'd3_l1_two_arm_acceptance.py'))
        acc = _ilu.module_from_spec(spec)
        sys.modules['two_arm_acc'] = acc
        spec.loader.exec_module(acc)
        import d3_l1_two_arm_protocol as TA
        import numpy as np
        import time as _time

        def slow(state, legal):
            _time.sleep(0.05)
            return np.zeros(len(legal), dtype=np.float32)

        fx = acc.build_fixture()
        arms, legal_fn, apply_fn = acc.make_arms(fx, f_scores_fn=slow)
        shared = acc.shared_input(fx, legal_fn=legal_fn)
        report = TA.run_two_arms(shared, arms, legal_fn=legal_fn, apply_fn=apply_fn,
                                 wall_clock_limit_s=0.01)
        r = report['results']
        det = {'f_failure': r[TA.ARM_F]['failure_reason'],
               'f_timed_out': r[TA.ARM_F]['diag'].get('timed_out'),
               'f_returned_original': r[TA.ARM_F]['returned_original_plan'],
               'v2_f_completed': report['V2'][TA.ARM_F]['full_repair_completed'],
               'v3_certified': report['V3']['any_certified'],
               'wall_clock_limit_s_used': 0.01,
               'why': 'the wall limit is passed to the run, so the ARM overruns and fails closed; in '
                      'production both this limit and the supervisor cap are the frozen 10 s'}
        ok = (det['f_failure'] == 'timeout' and det['f_timed_out'] is True
              and det['f_returned_original'] is True and det['v2_f_completed'] is False
              and det['v3_certified'] == 'NOT_RUN')
        return ok, det
    except BaseException as exc:                                          # noqa: BLE001
        return False, {'error': '%s: %s' % (type(exc).__name__, exc)}


def blocker_b(root, module, states):
    import d3_l1_v116_forward_control as control

    b_root = root / 'b'
    b_root.mkdir(parents=True, exist_ok=True)
    standin = b_root / 'standin_worker.py'
    standin.write_text(STANDIN, encoding='utf-8')
    states_json = b_root / 'states.json'
    states_json.write_text(json.dumps(states), encoding='utf-8')
    n = len(states)

    def cmd(ready_sleep=0.2, slot_sleep=0.02, publish_ready=1, slot_at=-1, slot_long=0.0):
        out = Path(tempfile.mkdtemp(prefix='standin_run_', dir=str(b_root)))
        (out / 'phases').mkdir(exist_ok=True)
        c = [sys.executable, str(standin), '--out-dir', str(out), '--run-dir', str(out),
             '--states-json', str(states_json), '--entry', str(RUNNER),
             '--ready-sleep-s', str(ready_sleep),
             '--slot-sleep-s', str(slot_sleep), '--publish-ready', str(publish_ready),
             '--slot-sleep-at', str(slot_at), '--slot-sleep-long-s', str(slot_long)]
        return out, c

    def contained(out, c, *, init_cap, slot_cap=10.0, batch_cap=180.0):
        # `run_contained` calls `identity_check(identity)`, and an EMPTY mapping reports `ok=False`,
        # so the fixture must watch something real: the stand-in worker and its state list.
        identity = {str(standin): sha(standin), str(states_json): sha(states_json)}
        return control.run_contained(
            c, out, identity, init_cap=init_cap, slot_cap=slot_cap, batch_cap=batch_cap,
            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
            ready_probe=module.make_l1_ready_probe(out))

    # B1 ------------------------------------------------------------------------------------
    empty = b_root / 'empty_run'
    empty.mkdir(exist_ok=True)
    probe = module.make_l1_ready_probe(empty)
    b1_ok = (probe(empty).get('probe') == 'immutable L1_READY record'
             and probe(empty).get('ready') is False
             and probe(empty).get('reason') == 'absent'
             and control.default_ready_probe(empty) is False)
    ck('B1 the supervisor probe is the IMMUTABLE-record probe bound to one run dir, not the legacy '
       '`READY.json` default (which the V1.18+ worker can never publish)', b1_ok,
       {'probe_report_on_empty_dir': probe(empty), 'legacy_default_ready': False,
        'pre_fix_call': 'control.run_contained(command, out, identity, init_cap=..., slot_cap=..., '
                        'batch_cap=..., env=env)  # no ready_probe'})

    # B2 FAST init (READY at +2 s) then a batch that runs PAST 60 s but stays under 180 s.
    #    slot_sleep x 64 states ~= 67 s of batch after a 2 s initialization, so the wall clock clears
    #    the 60 s initialization cap while the batch is still well inside its own cap.
    out2, c2 = cmd(ready_sleep=2.0, slot_sleep=1.05)
    t0 = time.monotonic()
    res2 = contained(out2, c2, init_cap=60.0)
    wall2 = time.monotonic() - t0
    ready2 = res2.get('ready_state') or {}
    b2_ok = (ready2.get('ready') is True and ready2.get('n_records') == 1
             and res2.get('reason') is None and res2.get('returncode') == 0
             and wall2 > 60.0 and wall2 < 180.0)
    ck('B2 a worker whose READY lands at ~2 s and whose batch then runs past the 60 s initialization '
       'cap (while staying under the 180 s batch cap) is NOT misjudged as an init timeout', b2_ok,
       {'wall_s': round(wall2, 2), 'reason': res2.get('reason'), 'returncode': res2.get('returncode'),
        'ready_state': {k: v for k, v in ready2.items()
                        if k in ('ready', 'probe', 'n_records', 'problems')},
        'n_states': n, 'caps': res2.get('caps'), 'ok': res2.get('ok'),
        'why_this_is_the_counterexample': 'the pre-fix supervisor never saw READY, so it measured '
                                          '"initialization" as this whole 67 s batch and killed it '
                                          'at the 60 s init cap'})

    # B3 SLOW init: READY never lands, so the init cap must really force-kill.
    out3, c3 = cmd(ready_sleep=0.2, publish_ready=0, slot_sleep=0.5)
    t0 = time.monotonic()
    res3 = contained(out3, c3, init_cap=3.0)
    wall3 = time.monotonic() - t0
    b3_ok = (res3.get('reason') == 'initialization_compile_deadline'
             and bool((res3.get('cleanup') or {}).get('cleanup_ok'))
             and (res3.get('cleanup') or {}).get('tree_active_after') == 0
             and res3.get('tree_active_after') == 0 and wall3 < 15.0
             and (res3.get('ready_state') or {}).get('ready') is False)
    ck('B3 a worker that never publishes READY IS force-killed by the initialization cap -- the cap '
       'is still a real cap, not a formality', b3_ok,
       {'wall_s': round(wall3, 2), 'reason': res3.get('reason'), 'cleanup': res3.get('cleanup'),
        'tree_active_after': res3.get('tree_active_after'),
        'ready_state': {k: v for k, v in (res3.get('ready_state') or {}).items()
                        if k in ('ready', 'probe', 'reason')},
        'init_cap_used_s': 3.0,
        'note': 'this fixture SHORTENS the cap to 3 s so the check costs seconds, not a minute; the '
                'production cap stays 60.0 and is asserted unchanged in B7'})

    # B4 -- the two layers of "a state exceeded the pre-registered 10 s", both of them REAL:
    #   B4a the ARM layer: a scorer that overruns its wall clock degrades FAIL-CLOSED to the original
    #       plan with `timeout`, and the state keeps a record.  This is the layer that makes the run
    #       answerable per state.
    #   B4b the SUPERVISOR layer: a worker that is stuck inside a state past the cap has its process
    #       tree terminated and the round stops, with every record published before the strike intact.
    # A statement that "the state degrades" without naming WHICH layer would be exactly the kind of
    # unverifiable summary this acceptance exists to prevent.
    b4a_ok, b4a_det = _arm_timeout_degrades_fail_closed()
    out4, c4 = cmd(ready_sleep=0.2, slot_sleep=0.02, slot_at=5, slot_long=12.0)
    res4 = contained(out4, c4, init_cap=60.0)
    tl4 = res4.get('timeline') or {}
    phases4 = [json.loads(p.read_text(encoding='utf-8'))
               for p in sorted((out4 / 'phases').glob('phase_*.json'))]
    starts4 = [p for p in phases4 if p.get('stage') == 'slot']
    finishes4 = [p for p in phases4 if p.get('stage') == 'slot_finished']
    b4b_ok = (res4.get('reason') == 'slot_span_deadline'
              and (res4.get('cleanup') or {}).get('tree_active_after') == 0
              and len(starts4) >= 6 and len(finishes4) >= 5
              and len(starts4) - len(finishes4) == 1)
    ck('B4 a state exceeding the pre-registered 10 s is handled at BOTH real layers: the arm '
       'degrades fail-closed to the original plan with `timeout`, and the supervisor terminates the '
       'tree so the round stops instead of the overrun passing unnoticed',
       b4a_ok and b4b_ok,
       {'arm_layer_fail_closed': b4a_det,
        'supervisor_layer': {'reason': res4.get('reason'),
                             'tree_active_after': (res4.get('cleanup') or {})
                             .get('tree_active_after'),
                             'slot_starts': len(starts4), 'slot_finishes': len(finishes4),
                             'open_without_finish': len(starts4) - len(finishes4),
                             'records_published_before_strike_survived': len(finishes4) >= 5}})

    # B5 the 10 s bounds the STATE, not each arm
    snapshots, b5_ok = {}, False
    try:
        protocol = module.load_helpers()[2]
        origin = time.monotonic()
        gates = {a: protocol.BudgetGate(max_scorer_calls=128, max_wall_seconds=10.0,
                                        started_mono=origin) for a in protocol.ARMS}
        time.sleep(0.15)
        snapshots = {a: g.snapshot() for a, g in gates.items()}
        # the assertion is on the SHARED ORIGIN and on both arms reporting non-trivial elapsed time
        # from it -- not on a sleep duration, which the first cut compared against a 0.05 s
        # threshold and failed at 0.047 s of measured clock
        b5_ok = (len(gates) == 2 and len({id(g) for g in gates.values()}) == 2
                 and all(g.started == origin for g in gates.values())
                 and all(s['wall_origin_shared_across_arms'] is True for s in snapshots.values())
                 and all(s['wall_seconds'] >= 0.10 for s in snapshots.values())
                 and all(s['max_wall_seconds'] == 10.0 for s in snapshots.values()))
        # and the ALTERNATIVE would be two independent origins: show that they are NOT independent
        independent = protocol.BudgetGate(max_scorer_calls=128, max_wall_seconds=10.0)
        snapshots['__independent_default_origin__'] = dict(
            independent.snapshot(), note='default (no started_mono): a per-gate origin, which is '
                                         'what the L1 run must NOT use for its two arms')
        b5_ok = b5_ok and independent.started != origin
    except BaseException as exc:                                          # noqa: BLE001
        snapshots = {'error': '%s: %s' % (type(exc).__name__, exc)}
    ck('B5 the frozen 10 s bounds the WHOLE STATE, not each arm: both arms\' gates are charged from '
       'ONE monotonic origin, so two 10 s allowances cannot be summed', b5_ok,
       {'arm_gate_snapshots': snapshots, 'scope_constant': module.WALL_CLOCK_SCOPE,
        'what_the_alternative_would_allow': 'F 10 s + D-Geo 10 s = 20 s for one state, while every '
                                            'durable record still declared "10 s per state"'})

    # B6 per-state records persist, and a state completed before a deadline strike keeps its record
    out6, c6 = cmd(ready_sleep=0.2, slot_sleep=0.02, slot_at=n - 1, slot_long=12.0)
    res6 = contained(out6, c6, init_cap=60.0)
    phase_recs = [json.loads(p.read_text(encoding='utf-8'))
                  for p in sorted((out6 / 'phases').glob('phase_*.json'))]
    opens = [p for p in phase_recs if p.get('stage') == 'slot']
    closes = [p for p in phase_recs if p.get('stage') == 'slot_finished']
    # every state opened BEFORE the strike must still have its durable last phase record
    persisted = (out6 / 'phases').is_dir() and len(opens) >= n
    b6_ok = (res6.get('reason') == 'slot_span_deadline' and persisted
             and len(closes) >= n - 1
             and sorted((p['day_id'], p['position']) for p in opens)
             == sorted((int(s['day_id']), int(s['position'])) for s in states))
    ck('B6 per-state start/end phase records persist for EVERY state, and the states completed '
       'before the deadline strike keep their durable records', b6_ok,
       {'slot_starts': len(opens), 'slot_finishes': len(closes), 'n_states': n,
        'reason': res6.get('reason'),
        'first_phase': {k: phase_recs[0].get(k) for k in ('stage', 'sequence')} if phase_recs else None,
        'all_states_opened_in_order': b6_ok})

    # B7 the frozen caps are untouched
    b7_ok = (module.BUDGET['per_state_wall_clock_s'] == 10.0
             and module.BUDGET['initialize_compile_s'] == 60.0
             and module.BUDGET['batch_wall_clock_s'] == 180.0
             and module.BUDGET['per_arm_total_scorer_calls'] == 128
             and module.BUDGET['per_arm_per_state_scorer_calls'] == 2
             and module.BUDGET['mask_steps_m'] == 2)
    ck('B7 the frozen 60 / 10 / 180 s caps and the 128 / 2 / 2 call counts are UNCHANGED by this fix',
       b7_ok, dict(module.BUDGET, wall_clock_scope=module.WALL_CLOCK_SCOPE))

    return {'B1': {'ok': b1_ok}, 'B2': {'ok': b2_ok, 'wall_s': round(wall2, 2)},
            'B3': {'ok': b3_ok, 'wall_s': round(wall3, 2)},
            'B4': {'ok': b4a_ok and b4b_ok, 'arm_layer': b4a_det,
                   'supervisor_layer': res4.get('reason')},
            'B5': {'ok': b5_ok}, 'B6': {'ok': b6_ok}, 'B7': {'ok': b7_ok},
            'standin_worker_note': 'the workers here are STAND-INS that publish exactly the '
                                   'production READY / phase protocol and load no model',
            'b2_supervision': {k: v for k, v in res2.items() if k != 'end_identity'}}


# =============================================================================================
# C. the 64-state denominator
# =============================================================================================
def blocker_c(root, module, states):
    def rec(day, pos, verdict='DIVERGENT', status='COMPLETED', ledger_ok=True, single_factor=True,
            failure=None):
        r = module.blank_state_record(day, pos, state_hash='h%02d%d' % (day, pos),
                                      status=status)
        r['V1'] = {'verdict': verdict}
        r['V2'] = {'both_completed': verdict in ('DIVERGENT', 'IDENTICAL'),
                   'F': {'full_repair_completed': verdict in ('DIVERGENT', 'IDENTICAL')},
                   'D-Geo': {'full_repair_completed': verdict in ('DIVERGENT', 'IDENTICAL')}}
        r['audit'] = {'single_factor_ok': single_factor,
                      'state_identity_agrees_both_arms': single_factor}
        r['ledger_ok'] = ledger_ok
        r['per_arm_failure'] = failure or {'F': None, 'D-Geo': None}
        r['ledger_problems'] = [] if ledger_ok else [{'reason': 'injected ledger fault'}]
        return r

    days, positions = [int(s['day_id']) for s in states], [int(s['position']) for s in states]

    # C1 + C6: inject a fault into ONE state; the denominator must stay 64 and report it
    clean = [rec(days[i], positions[i]) for i in range(len(states))]
    inj = [dict(r) for r in clean]
    k = 7
    inj[k] = module.blank_state_record(days[k], positions[k], status='FAILED_EXCEPTION',
                                       exception_type='InjectedFault',
                                       error='injected fault for the denominator counterexample')
    v_clean = module.adjudicate(clean, synthetic=True)
    v_inj = module.adjudicate(inj, synthetic=True)
    c1_ok = (v_inj['n_states_in_denominator'] == 64
             and v_inj['denominator_is_the_preregistered_list'] is True
             and len(v_inj['exception_states']) == 1
             and v_inj['exception_states'][0]['day_id'] == days[k]
             and v_inj['status'] == 'BLOCKED')
    ck('C1 a fault injected into ONE state leaves the denominator at 64 and REPORTS that state '
       '(the pre-fix code adjudicated over the 63 survivors)', c1_ok,
       {'n_clean': v_clean['n_states_in_denominator'],
        'n_with_one_fault': v_inj['n_states_in_denominator'],
        'exception_states': v_inj['exception_states'],
        'status_with_fault': v_inj['status'],
        'status_clean': v_clean['status'],
        'pre_fix_behaviour': 'adjudicate([r for r in records if r.get("status") != '
                             '"FAILED_EXCEPTION"]) -> denominator 63'})

    # C2 ------------------------------------------------------------------------------------
    mixed = [dict(r) for r in clean]
    mixed[0] = rec(days[0], positions[0], verdict='NOT_APPLICABLE_EMPTY_MASK', status='EXEMPT_EMPTY_MASK')
    mixed[1] = rec(days[1], positions[1], verdict='NOT_COMPARABLE', status='COMPLETED',
                   single_factor=True)
    mixed[2] = rec(days[2], positions[2], status='TIMEOUT', failure={'F': 'timeout',
                                                                    'D-Geo': None})
    mixed[3] = module.blank_state_record(days[3], positions[3], status='NOT_ATTEMPTED_INIT_FAILURE')
    v_mix = module.adjudicate(mixed, synthetic=True)
    c2_ok = (len(v_mix['empty_pool_states']) == 1 and len(v_mix['not_comparable_states']) == 1
             and len(v_mix['timeout_states']) == 1 and len(v_mix['not_attempted_states']) == 1
             and v_mix['n_states_in_denominator'] == 64
             and v_mix['outcome_counts']['EXEMPT_EMPTY_MASK'] == 1
             and v_mix['outcome_counts']['NOT_ATTEMPTED'] == 1)
    ck('C2 empty-pool / not-comparable / timeout / not-attempted states are each listed SEPARATELY '
       'while all 64 stay in the denominator', c2_ok,
       {'denominator': v_mix['n_states_in_denominator'], 'outcome_counts': v_mix['outcome_counts'],
        'V1_counts': v_mix['V1_counts'],
        'separate_lists': {'empty_pool': len(v_mix['empty_pool_states']),
                           'not_comparable': len(v_mix['not_comparable_states']),
                           'timeout': len(v_mix['timeout_states']),
                           'not_attempted': len(v_mix['not_attempted_states']),
                           'exception': len(v_mix['exception_states'])}})

    # C3 ------------------------------------------------------------------------------------
    bad_ledger = [dict(r) for r in clean]
    bad_ledger[9]['ledger_ok'] = False
    bad_ledger[9]['ledger_problems'] = [{'reason': 'injected ledger fault'}]
    v_led = module.adjudicate(bad_ledger, synthetic=True)
    c3_ok = (v_led['status'] == 'BLOCKED' and v_led['all_ledgers_cross_checked'] is False
             and len(v_led['ledgers_not_cross_checked']) == 1
             and any('ledger' in b for b in v_led['blocking_reasons']))
    ck('C3 `ledger_ok=false` on ONE state forces a BLOCKED round: it can never be promoted to a '
       'formal PASS', c3_ok,
       {'status': v_led['status'], 'all_ledgers_cross_checked': v_led['all_ledgers_cross_checked'],
        'blocking_reasons': v_led['blocking_reasons'],
        'ledgers_not_cross_checked': v_led['ledgers_not_cross_checked']})

    # C4 ------------------------------------------------------------------------------------
    short = [dict(r) for r in clean[:-1]]
    v_short = module.adjudicate(short, synthetic=True)
    c4_ok = (v_short['n_states_in_denominator'] == 63
             and v_short['denominator_is_the_preregistered_list'] is False
             and v_short['status'] == 'BLOCKED'
             and any('pre-registered state list' in b for b in v_short['blocking_reasons']))
    ck('C4 a MISSING state forces a BLOCKED round: the denominator may not shrink, and a '
       '63-of-64 report is refused rather than accepted', c4_ok,
       {'denominator': v_short['n_states_in_denominator'], 'expected': v_short['n_states_expected'],
        'status': v_short['status'], 'blocking_reasons': v_short['blocking_reasons']})

    # C5 ------------------------------------------------------------------------------------
    bad_ident = [dict(r) for r in clean]
    bad_ident[11]['audit'] = {'single_factor_ok': False,
                              'state_identity_agrees_both_arms': False}
    v_ident = module.adjudicate(bad_ident, synthetic=True)
    c5_ok = (v_ident['status'] == 'BLOCKED' and len(v_ident['single_factor_failures']) == 1
             and any('single-factor' in b for b in v_ident['blocking_reasons']))
    ck('C5 a single-factor / shared-state identity failure on ONE state forces a BLOCKED round',
       c5_ok, {'status': v_ident['status'],
               'single_factor_failures': v_ident['single_factor_failures'],
               'blocking_reasons': v_ident['blocking_reasons']})

    # C6: the pre-fix filter is gone from the SOURCE ITSELF, and every call site still passes the
    # WHOLE record list
    import ast as _ast
    src = RUNNER.read_text(encoding='utf-8')
    tree = _ast.parse(src)
    calls, filtered_calls = [], []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, _ast.Attribute) else getattr(fn, 'id', None)
        if name != 'adjudicate' or not node.args:
            continue
        first = node.args[0]
        calls.append({'line': node.lineno,
                      'arg_src': (_ast.get_source_segment(src, first) or '').strip()[:90],
                      'is_bare_list': isinstance(first, _ast.Name) and first.id == 'records'})
        if isinstance(first, (_ast.ListComp, _ast.GeneratorExp)):
            filtered_calls.append(node.lineno)
    all_bare = bool(calls) and all(c['is_bare_list'] for c in calls)
    adj_body = ''
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == 'adjudicate':
            adj_body = _ast.get_source_segment(src, node) or ''
    # the only permitted mentions inside `adjudicate` are (a) the docstring that records the defect
    # and (b) an equality COUNT of exception states -- never a comprehension that removes them
    selects_exception = "r.get('status') == 'FAILED_EXCEPTION'" in adj_body
    filters_exception = "r.get('status') != 'FAILED_EXCEPTION'" in adj_body
    c6_ok = (not filters_exception and selects_exception and all_bare and not filtered_calls)
    ck('C6 the pre-fix denominator SHRINK is gone from the SOURCE: no `FAILED_EXCEPTION` filter '
       'remains, and EVERY `adjudicate(...)` call site passes the whole record list', c6_ok,
       {'pre_fix_filter_in_code': filters_exception,
        'exception_states_are_enumerated_not_removed': selects_exception,
        'call_sites': calls,
        'call_sites_passing_a_filtered_list': filtered_calls,
        'pre_fix_source_line': "adjudicate([r for r in records "
                               "if r.get('status') != 'FAILED_EXCEPTION'], synthetic=...)"})

    return {'C1': {'ok': c1_ok}, 'C2': {'ok': c2_ok}, 'C3': {'ok': c3_ok}, 'C4': {'ok': c4_ok},
            'C5': {'ok': c5_ok}, 'C6': {'ok': c6_ok}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    # The scratch root MUST sit on the workspace drive: the twin needs the real 305 MB backbone and a
    # hardlink cannot cross volumes.  A dot-prefixed name keeps it obvious (and out of the way) if a
    # previous run was killed before its cleanup.
    scratch = Path(tempfile.mkdtemp(prefix='.l1_blocker_acc_', dir=str(REPO)))
    root = scratch
    module = runner_module()
    ident = module.build_l1_identity(created_utc='2026-01-01T00:00:00Z')
    states = ident['state_list']['states']
    results = {}
    try:
        results['A'] = blocker_a(root, module)
        results['B'] = blocker_b(root, module, states)
        results['C'] = blocker_c(root, module, states)
        fails = [r for r in ROWS if not r['ok']]
        payload = {
            'schema': 'd3_l1_l1_sign_blocker_acceptance',
            'run_id': RUN_ID,
            'rows': ROWS, 'n_rows': len(ROWS),
            'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
            'results': results,
            'blockers': {
                'A': 'M7 source freeze did not cover L1 itself: the watch set was the F closure, so '
                     'signing shipped l1_sources_in_closure == [] while M7 read PASS',
                'B': 'L1 READY and per-state supervision: run_contained was called with no '
                     'ready_probe and the worker published no READY and no per-state phases',
                'C': 'the formal denominator: FAILED_EXCEPTION states were filtered out of the '
                     'adjudication, silently shrinking 64 to the survivors',
            },
            'scorer': ('NO SCORER AT ALL: the workers in this file are stand-ins that load no model '
                       'and score nothing. This acceptance says nothing about MaskCO, D-Geo, or any '
                       'scientific question.'),
            'not_run': ['the real F-vs-D-Geo comparison', 'training', 'L2', 'L3', 'the 40-day '
                        'batch', 'the final test'],
            'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
            'tool_sha256': sha(Path(__file__)), 'runner_sha256': sha(RUNNER),
            'closure_sha256': sha(EXT / 'scripts/data/d3_l1_closure.py'),
            'verdict': ('L1_SIGN_BLOCKERS_CLOSED' if not fails else 'L1_SIGN_BLOCKERS_OPEN'),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr),
                       encoding='utf-8')
        print('\n=== L1 sign-blocker acceptance: %d rows, %d PASS, %d FAIL ==='
              % (len(ROWS), len(ROWS) - len(fails), len(fails)))
        print('WROTE', out)
        return 0 if not fails else 1
    finally:
        # `rmtree` removes the junctions as LINKS, never recursing into the real trees they point at
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        if scratch.exists():
            print('SCRATCH KEPT', scratch)


if __name__ == '__main__':
    sys.exit(main())
