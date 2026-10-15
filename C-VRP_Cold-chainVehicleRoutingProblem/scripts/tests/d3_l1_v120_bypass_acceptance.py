#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.20 acceptance: the execution-entry bypass is closed.

The bypass under test
---------------------
`main()` accepted `--worker` directly and called `worker()`, which loaded the F checkpoint and called
the scorer WITHOUT any human-authorization check, without one-shot consumption and without the
supervisor's deadline containment.  `--synthetic-command` was a second opening: it substitutes the
worker process, so pointing it at the real worker would have started real F work outside the
authorization gate.

The guard under test
--------------------
The supervisor publishes an EXECUTION CONTEXT only after a human-approved credential has been
validated against the frozen identity and consumed one-shot.  The real worker's first action
verifies that context (HMAC over a private secret, run id/dir binding, the consumption receipt, and
the credential's post-consumption digest).  Everything the worker would load sits behind it.

Scenarios (one attempt each, failures kept):
  N1  `--worker` with NO context                     -> REFUSED, 0 real-F artifacts
  N2  `--worker` with a context env pointing nowhere  -> REFUSED, 0 real-F artifacts
  N3  `--guard-probe` with no context                 -> REFUSED
  N4  hand-written context (attacker-secret HMAC)     -> REFUSED (no consumption receipt exists)
  N5  context EDITED after signing (run_dir changed)  -> REFUSED (HMAC mismatch)
  N6  context edited after signing (digest changed)   -> REFUSED (HMAC mismatch)
  N7  receipt exists but names ANOTHER run            -> REFUSED (run id mismatch)
  N8  credential EDITED after consumption             -> REFUSED (digest changed)
  N9  `--synthetic-command` pointed at the REAL worker, no authorization
                                                      -> REFUSED, 0 real-F artifacts
  N10 `--supervise` with a valid credential AND `--synthetic-command`
                                                      -> REFUSED before launching anything
  P1  the supervisor publishes a context ONLY after consumption, and the guard accepts exactly
      what it produced (proved in a separate process that loads nothing)
  P2  the supervised synthetic path still works end to end (9/9 semantics preserved)

"0 real-F artifacts" is checked by looking for the two records the worker writes immediately after
the guard -- `MODEL_LOAD` and `WARMUP` -- plus any slot row or raw score artifact.
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as AUTH                                       # noqa: E402
import d3_l1_v118_persistence as persistence                                 # noqa: E402
import d3_l1_v120_execution_context as EXECCTX                               # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
SYNTH_WORKER = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V120_PREREG = EV / 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_20.md'
RUN_ID = 'd3l1-real-f-forward-20261015-v120'

ROWS = []


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:235]))


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def runner_module(name='v120_runner'):
    spec = importlib.util.spec_from_file_location(name, str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze_fixture(root, name, run_id=RUN_ID):
    run_dir = Path(root) / name
    runner_module('v120_freeze_%s' % name).freeze_identity(
        run_dir, run_id, now_utc='2026-10-15T00:00:00Z')
    return run_dir


def real_f_artifacts(run_dir):
    """The records that only a worker which got PAST the guard can produce."""
    run_dir = Path(run_dir)
    found = {}
    for prefix, where in (('MODEL_LOAD', run_dir), ('WARMUP', run_dir),
                          ('SLOT_day', run_dir / 'scores'), ('SCORES', run_dir / 'scores'),
                          ('WORKER_RESULT', run_dir), ('WORKER_FAILURE', run_dir),
                          ('FIRST_FAILURE', run_dir)):
        if where.is_dir():
            found[prefix] = len(list(where.glob('%s*.json' % prefix)))
        else:
            found[prefix] = 0
    found['phase_records'] = len(list((run_dir / 'phases').glob('phase_*.json'))) \
        if (run_dir / 'phases').is_dir() else 0
    found['guard_refusals'] = len(list(run_dir.glob('WORKER_GUARD_REFUSED_*.json')))
    return found


def run(cmd, *, env=None, timeout_s=300):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=e)



def guard_refusal_reason(run_dir):
    """The FULL guard refusal, read from the immutable record the worker published.

    Reading stderr would be reading a human-facing tail; the record is the durable evidence, and it
    is what an auditor would read.
    """
    rec = persistence.VerifiedReader(run_dir).latest('WORKER_GUARD_REFUSED') or {}
    return rec.get('reason') or ''


def context_records_on_disk(run_dir):
    """REAL execution contexts only.

    A glob such as `EXECUTION_CONTEXT_[0-9a-f]*.json` is not exclusive: on Windows the filesystem
    match is case-insensitive, so the `E` of `EXECUTION_CONTEXT_EVIDENCE_...` also matches
    `[0-9a-f]` and the evidence record gets miscounted as an issued context.  The nonce is therefore
    validated explicitly rather than trusted to a character class.
    """
    out = []
    for p in Path(run_dir).glob('EXECUTION_CONTEXT_*.json'):
        stem = p.name[len('EXECUTION_CONTEXT_'):-len('.json')]
        if len(stem) == 32 and all(c in '0123456789abcdef' for c in stem):
            out.append(p)
    return out


def worker_cmd(run_dir):
    return [sys.executable, str(RUNNER), '--worker', '--out-dir', str(run_dir)]


def guard_probe_cmd(run_dir):
    return [sys.executable, str(RUNNER), '--guard-probe', '--out-dir', str(run_dir)]


def approve(frozen, *, prereg, consumption_root, output_dir=None):
    """Build a HUMAN-FILLED credential for the fixture and return (path, template)."""
    module = runner_module('v120_approve')
    fr = module.load_frozen(frozen)
    sealed = AUTH.seal_template(AUTH.build_template(
        fr['identity'], fr['forward'], preregistration_sha256=sha(prereg),
        output_dir=output_dir or frozen))
    sealed = dict(sealed, approved_by='USER (synthetic acceptance fixture)',
                  approved_utc='2026-10-15T00:00:00Z',
                  approval_statement='SYNTHETIC ACCEPTANCE FIXTURE -- not a real approval')
    sealed = AUTH.seal_template(sealed)
    path = Path(frozen).parent / 'credential_approved.json'
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    return path, sealed


# =============================================================================================
# negative scenarios
# =============================================================================================
def n1_direct_worker(root):
    run_dir = freeze_fixture(root, 'N1_direct_worker')
    proc = run(worker_cmd(run_dir))
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    ok = (proc.returncode != 0 and 'EXECUTION CONTEXT REFUSED' in proc.stderr
          and art['MODEL_LOAD'] == 0 and art['WARMUP'] == 0 and art['SLOT_day'] == 0
          and art['SCORES'] == 0 and art['guard_refusals'] == 1)
    ck('N1 `--worker` invoked DIRECTLY refuses before loading anything: 0 model loads, 0 warm-ups, '
       '0 slot rows, 0 scores', ok, det)
    return det


def n2_context_env_nowhere(root):
    run_dir = freeze_fixture(root, 'N2_context_nowhere')
    proc = run(worker_cmd(run_dir),
               env={EXECCTX.CONTEXT_ENV: str(Path(root) / 'nope.json')})
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    ok = (proc.returncode != 0 and 'context_env_missing_file' in proc.stderr
          and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0)
    ck('N2 `--worker` with the context variable pointing at a MISSING file refuses', ok, det)
    return det


def n3_guard_probe_no_context(root):
    run_dir = freeze_fixture(root, 'N3_probe_no_context')
    proc = run(guard_probe_cmd(run_dir))
    det = {'returncode': proc.returncode, 'stdout': proc.stdout.strip()[:200]}
    ok = proc.returncode != 0 and 'REFUSED' in proc.stdout and 'no_context_env' in proc.stdout
    ck('N3 the guard probe reports REFUSED without a context', ok, det)
    return det


def n4_handwritten_context(root):
    run_dir = freeze_fixture(root, 'N4_handwritten')
    secret = 'ab' * 32
    secret_path = Path(run_dir) / 'guard' / 'forged_secret.txt'
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(secret, encoding='utf-8')
    os.chmod(secret_path, 0o600)
    context = {'schema': EXECCTX.CONTEXT_SCHEMA, 'run_id': RUN_ID,
               'run_dir': str(Path(run_dir).resolve()), 'supervisor_pid': os.getpid(),
               'nonce': 'f' * 32, 'created_mono': time.monotonic(), 'authorization_age_mono': 0.0,
               'credential_content_sha256': 'c' * 64, 'credential_file_sha256': 'd' * 64,
               'credential_path': str(Path(root) / 'forged_credential.json'),
               'consumption_path': str(Path(root) / 'forged_consumption.json'),
               'consumption_credential_content_sha256': 'c' * 64,
               'secret_path': str(secret_path)}
    context['hmac_sha256'] = EXECCTX.compute_hmac(secret, context)
    context_path = Path(run_dir) / ('EXECUTION_CONTEXT_%s.json' % context['nonce'])
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding='utf-8')
    proc = run(worker_cmd(run_dir),
               env={EXECCTX.CONTEXT_ENV: str(context_path), EXECCTX.SECRET_ENV: str(secret_path)})
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    reason = guard_refusal_reason(run_dir)
    det['guard_reason'] = reason
    ok = (proc.returncode != 0 and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0
          and ('consumption_receipt_missing' in reason
               or 'context_missing_field' in reason))
    ck('N4 a HAND-WRITTEN context with a self-made HMAC is refused: inventing a nonce proves '
       'nothing, because the one-shot consumption receipt cannot be fabricated', ok, det)
    return det


def n5_context_edited_run_dir(root):
    run_dir = freeze_fixture(root, 'N5_edited_rundir')
    context, secret_path = _signed_context(root, run_dir)
    context['run_dir'] = str(Path(root) / 'somewhere_else')
    context_path = Path(run_dir) / ('EXECUTION_CONTEXT_%s.json' % context['nonce'])
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding='utf-8')
    proc = run(worker_cmd(run_dir),
               env={EXECCTX.CONTEXT_ENV: str(context_path), EXECCTX.SECRET_ENV: str(secret_path)})
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    reason = guard_refusal_reason(run_dir)
    det['guard_reason'] = reason
    ok = proc.returncode != 0 and 'hmac_mismatch' in reason and art['MODEL_LOAD'] == 0
    ck('N5 a context EDITED after signing (run_dir changed) is refused by the HMAC', ok, det)
    return det


def n6_context_edited_digest(root):
    run_dir = freeze_fixture(root, 'N6_edited_digest')
    context, secret_path = _signed_context(root, run_dir)
    context['credential_content_sha256'] = 'e' * 64
    context_path = Path(run_dir) / ('EXECUTION_CONTEXT_%s.json' % context['nonce'])
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True), encoding='utf-8')
    proc = run(worker_cmd(run_dir),
               env={EXECCTX.CONTEXT_ENV: str(context_path), EXECCTX.SECRET_ENV: str(secret_path)})
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    reason = guard_refusal_reason(run_dir)
    det['guard_reason'] = reason
    ok = proc.returncode != 0 and 'hmac_mismatch' in reason and art['MODEL_LOAD'] == 0
    ck('N6 a context EDITED after signing (credential digest changed) is refused by the HMAC',
       ok, det)
    return det


def n7_receipt_other_run(root):
    """A genuine-looking consumption receipt that names a DIFFERENT run must not open the gate."""
    run_dir = freeze_fixture(root, 'N7_other_run')
    cons = Path(root) / 'consumption_n7'
    credential, _ = approve(run_dir, prereg=V120_PREREG if V120_PREREG.is_file() else _stub_prereg(root),
                            consumption_root=cons)
    module = runner_module('v120_n7')
    fr = module.load_frozen(run_dir)
    granted = AUTH.authorize(AUTH.load_credential(credential), identity=fr['identity'],
                             forward_targets=fr['forward'], output_dir=run_dir,
                             preregistration_path=V120_PREREG if V120_PREREG.is_file()
                             else _stub_prereg(root), run_dir=run_dir, consumption_root=cons)
    pub = EXECCTX.publish_execution_context(run_dir, authorization_result=granted)
    # rewrite the receipt to claim another run id
    receipt_path = Path(granted['consumption']['path'])
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['run_id'] = 'd3l1-real-f-forward-99999999-v999'
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')
    proc = run(worker_cmd(run_dir), env=pub['env'])
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    reason = guard_refusal_reason(run_dir)
    det['guard_reason'] = reason
    ok = proc.returncode != 0 and 'consumption_run_id_mismatch' in reason \
        and art['MODEL_LOAD'] == 0
    ck('N7 a consumption receipt naming ANOTHER run id is refused', ok, det)
    return det


def n8_credential_edited_after_consumption(root):
    run_dir = freeze_fixture(root, 'N8_cred_edited')
    cons = Path(root) / 'consumption_n8'
    prereg = V120_PREREG if V120_PREREG.is_file() else _stub_prereg(root)
    credential, _ = approve(run_dir, prereg=prereg, consumption_root=cons)
    module = runner_module('v120_n8')
    fr = module.load_frozen(run_dir)
    granted = AUTH.authorize(AUTH.load_credential(credential), identity=fr['identity'],
                             forward_targets=fr['forward'], output_dir=run_dir,
                             preregistration_path=prereg, run_dir=run_dir,
                             consumption_root=cons)
    pub = EXECCTX.publish_execution_context(run_dir, authorization_result=dict(
        granted, credential_path=str(credential)))
    # edit the credential AFTER it was consumed
    edited = json.loads(Path(credential).read_text(encoding='utf-8'))
    edited['approval_statement'] = 'EDITED AFTER CONSUMPTION'
    Path(credential).write_text(json.dumps(edited, indent=2, sort_keys=True), encoding='utf-8')
    proc = run(worker_cmd(run_dir), env=pub['env'])
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stderr_tail': proc.stderr.strip()[-200:],
           'artifacts': art}
    reason = guard_refusal_reason(run_dir)
    det['guard_reason'] = reason
    ok = proc.returncode != 0 and 'credential_digest_changed_after_consumption' in reason \
        and art['MODEL_LOAD'] == 0
    ck('N8 a credential EDITED after consumption is refused (the consumed digest no longer '
       'matches the file)', ok, det)
    return det


def n9_synthetic_command_smuggling(root):
    """`--synthetic-command` pointed at the REAL worker must not produce real F work."""
    run_dir = freeze_fixture(root, 'N9_smuggle')
    cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir),
           '--synthetic-command', sys.executable, str(RUNNER), '--worker',
           '--out-dir', str(run_dir)]
    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    ctx = persistence.VerifiedReader(run_dir).latest('EXECUTION_CONTEXT_EVIDENCE') or {}
    # Strongest form of the check: hand the guard the environment the supervisor itself recorded
    # for this run (it recorded that NO context was issued) and ask it directly.
    probe_env = {EXECCTX.CONTEXT_ENV: str(run_dir / 'EXECUTION_CONTEXT_none.json')}
    probe = run(guard_probe_cmd(run_dir), env=probe_env)
    det = {'returncode': proc.returncode, 'artifacts': art,
           'context_record': {'published': ctx.get('published'), 'real_run': ctx.get('real_run')},
           'context_records_on_disk': len(context_records_on_disk(run_dir)),
           'guard_probe_returncode': probe.returncode,
           'guard_probe_stdout': probe.stdout.strip()[:200]}
    ok = (art['MODEL_LOAD'] == 0 and art['WARMUP'] == 0 and art['SLOT_day'] == 0
          and art['SCORES'] == 0 and probe.returncode != 0
          and '"guard": "REFUSED"' in probe.stdout
          and ctx.get('published') is False)
    ck('N9 smuggling the real worker through `--synthetic-command` yields 0 real-F artifacts, the '
       'supervisor records that NO context was published, and the guard refuses', ok, det)
    return det


def n10_authorization_plus_synthetic(root):
    """A valid credential combined with `--synthetic-command` is refused outright."""
    run_dir = freeze_fixture(root, 'N10_auth_plus_synth')
    cons = Path(root) / 'consumption_n10'
    prereg = V120_PREREG if V120_PREREG.is_file() else _stub_prereg(root)
    credential, _ = approve(run_dir, prereg=prereg, consumption_root=cons)
    cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir),
           '--authorization', str(credential), '--preregistration', str(prereg),
           '--consumption-root', str(cons),
           '--synthetic-command', sys.executable, str(SYNTH_WORKER),
           '--out-dir', str(run_dir), '--mode', 'pass']
    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    refusal = persistence.VerifiedReader(run_dir).latest('AUTHORIZATION_REFUSED') or {}
    det = {'returncode': proc.returncode,
           'refusal_record': {k: refusal.get(k) for k in ('granted', 'refused', 'required')},
           'artifacts': art,
           'consumption_records': len(list(cons.glob('*.json'))) if cons.is_dir() else 0}
    ok = (proc.returncode != 0 and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0
          and 'synthetic-command' in str(refusal.get('refused'))
          and det['consumption_records'] == 0)
    ck('N10 `--supervise` with a real authorization AND `--synthetic-command` is refused BEFORE the '
       'credential is consumed (0 consumption receipts) and before anything is launched', ok, det)
    return det


# =============================================================================================
# positive scenarios
# =============================================================================================
def p1_guard_accepts_what_the_supervisor_produces(root):
    """The supervisor publishes a context ONLY after consumption, and the guard accepts it.

    The acceptance runs in a SEPARATE process that loads nothing, so "the supervisor's context is
    enough" is proven without touching the model.
    """
    run_dir = freeze_fixture(root, 'P1_supervised_context')
    cons = Path(root) / 'consumption_p1'
    prereg = V120_PREREG if V120_PREREG.is_file() else _stub_prereg(root)
    credential, _ = approve(run_dir, prereg=prereg, consumption_root=cons)
    module = runner_module('v120_p1')
    fr = module.load_frozen(run_dir)
    # the context cannot precede the consumption
    premature = None
    try:
        EXECCTX.publish_execution_context(run_dir, authorization_result={'consumption': None})
    except EXECCTX.ExecutionContextRefused as exc:
        premature = str(exc)
    granted = AUTH.authorize(AUTH.load_credential(credential), identity=fr['identity'],
                             forward_targets=fr['forward'], output_dir=run_dir,
                             preregistration_path=prereg, run_dir=run_dir,
                             consumption_root=cons)
    pub = EXECCTX.publish_execution_context(run_dir, authorization_result=dict(
        granted, credential_path=str(credential)))
    proc = run(guard_probe_cmd(run_dir), env=pub['env'])
    det = {'premature_refusal': premature, 'probe_returncode': proc.returncode,
           'probe_stdout': proc.stdout.strip()[:400],
           'secret_mode': oct(os.stat(pub['secret_path']).st_mode & 0o777),
           'secret_mode_is_enforceable_here': os.name != 'nt',
           'secret_created_owner_only_via': 'os.open(O_CREAT|O_EXCL, 0o600)',
           'consumption_exists': Path(granted['consumption']['path']).is_file()}
    ok = (premature is not None and 'without a consumed credential' in premature
          and proc.returncode == 0 and '"guard": "PASSED"' in proc.stdout
          and det['consumption_exists'])
    ck('P1 the guard cannot be published before consumption, and it ACCEPTS exactly the context '
       'the supervisor produces (separate process, nothing loaded)', ok, det)
    return det


def p2_supervised_synthetic_path_still_works(root):
    """The synthetic acceptance path keeps working: no context, no authorization, 64 rows, PASS."""
    run_dir = freeze_fixture(root, 'P2_synthetic_path')
    cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir),
           '--synthetic-command', sys.executable, str(SYNTH_WORKER),
           '--out-dir', str(run_dir), '--mode', 'pass']
    proc = run(cmd)
    sup = persistence.VerifiedReader(run_dir).latest('SUPERVISION_RESULT') or {}
    ctx = len(context_records_on_disk(run_dir))
    det = {'returncode': proc.returncode, 'round_status': (sup.get('round') or {}).get('status'),
           'n_rows_qualified': sup.get('n_rows_qualified'), 'verdict': sup.get('verdict'),
           'execution_context_records': ctx}
    ctx_rec = persistence.VerifiedReader(run_dir).latest('EXECUTION_CONTEXT_EVIDENCE') or {}
    det['context_record_published'] = ctx_rec.get('published')
    ok = (proc.returncode == 0 and det['round_status'] == 'ROUND_PASS'
          and det['n_rows_qualified'] == 64 and ctx == 0
          and ctx_rec.get('published') is False)
    ck('P2 the supervised SYNTHETIC path still works end to end (64 rows, ROUND_PASS) and '
       'publishes no execution context', ok, det)
    return det


def _signed_context(root, run_dir, *, run_id=RUN_ID):
    """A correctly signed context for a run dir, used as the base for the tamper scenarios."""
    secret = 'cd' * 32
    secret_path = Path(run_dir) / 'guard' / 'tamper_secret.txt'
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(secret, encoding='utf-8')
    os.chmod(secret_path, 0o600)
    context = {'schema': EXECCTX.CONTEXT_SCHEMA, 'run_id': run_id,
               'run_dir': str(Path(run_dir).resolve()), 'supervisor_pid': os.getpid(),
               'nonce': hashlib.sha256(str(time.monotonic()).encode()).hexdigest()[:32],
               'created_mono': time.monotonic(), 'authorization_age_mono': 0.0,
               'credential_content_sha256': 'c' * 64, 'credential_file_sha256': 'd' * 64,
               'credential_path': str(Path(root) / 'cred.json'),
               'consumption_path': str(Path(root) / 'consumption.json'),
               'consumption_credential_content_sha256': 'c' * 64,
               'secret_path': str(secret_path)}
    context['hmac_sha256'] = EXECCTX.compute_hmac(secret, context)
    return context, secret_path


def _stub_prereg(root):
    p = Path(root) / 'stub_prereg.md'
    if not p.is_file():
        p.write_text('# stub pre-registration for the acceptance fixture\n', encoding='utf-8')
    return p


SCENARIOS = (
    ('N1 direct --worker', n1_direct_worker),
    ('N2 context env missing file', n2_context_env_nowhere),
    ('N3 guard probe without context', n3_guard_probe_no_context),
    ('N4 hand-written context', n4_handwritten_context),
    ('N5 context edited (run_dir)', n5_context_edited_run_dir),
    ('N6 context edited (digest)', n6_context_edited_digest),
    ('N7 receipt names another run', n7_receipt_other_run),
    ('N8 credential edited after consumption', n8_credential_edited_after_consumption),
    ('N9 synthetic-command smuggling', n9_synthetic_command_smuggling),
    ('N10 authorization plus synthetic-command', n10_authorization_plus_synthetic),
    ('P1 guard accepts the supervised context', p1_guard_accepts_what_the_supervisor_produces),
    ('P2 supervised synthetic path', p2_supervised_synthetic_path_still_works),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='v120_acc_'))
    started = time.monotonic()
    results = {}
    try:
        for name, fn in SCENARIOS:
            t = time.monotonic()
            try:
                results[name] = fn(root)
            except Exception as exc:                                      # noqa: BLE001
                import traceback
                results[name] = {'exception': '%s: %s' % (type(exc).__name__, exc),
                                 'traceback': traceback.format_exc()}
                ck('%s completed without an unhandled exception' % name, False, results[name])
            results[name] = dict(results[name], wall_s=time.monotonic() - t)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for r in ROWS if not r['ok'])
    payload = {
        'schema': 'd3_l1_v120_bypass_acceptance',
        'run_id': RUN_ID,
        'bypass_closed': 'the real F worker now requires an execution context published by the '
                         'supervisor after a human-approved credential was validated and consumed '
                         'one-shot; `--worker` and `--synthetic-command` can no longer start real F '
                         'work',
        'guard_conditions': ['context exists and parses', 'private 0600 secret',
                             'HMAC verifies', 'run id and run dir bound',
                             'one-shot consumption receipt present and for this run',
                             'credential still hashes to the consumed digest'],
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - n_fail, 'n_fail': n_fail,
        'results': results,
        'F_forward_calls_on_real_model': 0, 'F_D_Geo_comparisons': 0, 'training_runs': 0,
        'science_ready': False, 'stable_online_gain': False,
        'not_run': ['the real F 64-state batch', 'any F-vs-D-Geo comparison', 'any training',
                    'L2/L3', 'the 40-day batch', 'the final test'],
        'wall_s': time.monotonic() - started,
        'tool_sha256': sha(Path(__file__)),
        'runner_sha256': sha(RUNNER),
        'execution_context_module_sha256':
            sha(EXT / 'scripts/simulation/d3_l1_v120_execution_context.py'),
        'verdict': ('EXECUTION_ENTRY_BYPASS_CLOSED' if n_fail == 0
                    else 'EXECUTION_ENTRY_BYPASS_STILL_OPEN'),
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'D3_V120_BYPASS_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== V1.20 bypass acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
