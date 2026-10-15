#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.20 EXECUTION CONTEXT: the real F worker may only run inside an AUTHORISED, SUPERVISED run.

The bypass this closes
----------------------
`main()` accepted `--worker` directly:

    if args.worker:
        return worker(args.out_dir)

That path performed NO human-authorization check, NO one-shot consumption, NO frozen-identity gate
and NO deadline supervision -- yet it loaded the F checkpoint and called the scorer.  In other words
the entire authorization apparatus installed for blocker 1 could be skipped by typing one flag.
`--synthetic-command` was a second opening: it substitutes the worker process, so pointing it at the
real worker would have launched real F work under the synthetic harness's (deliberately absent)
authorization.

The guard
---------
The supervisor is the ONLY component that may authorise real F work.  After the credential has been
validated against the frozen identity and CONSUMED, the supervisor publishes an execution context:

    execution context  =  fresh 256-bit secret (a private file, never on a command line)
                       +  one-time nonce
                       +  run_id, run directory, supervisor pid, creation clock
                       +  the CONSUMED credential's content digest and consumption receipt path
                       +  HMAC(secret, canonical payload)

The worker's FIRST action -- before importing JAX, before reading the checkpoint, before calling any
scorer -- is `require_execution_context(...)`.  It refuses unless ALL of the following hold:

  C1  the context file exists and parses
  C2  the secret file exists, is not world-readable, and is the one the HMAC was computed with
  C3  the HMAC verifies (so the context was not hand-written or edited)
  C4  the run id and the run directory are the ones the worker is operating on
  C5  the ONE-SHOT consumption receipt exists on disk, is for this run, and records exactly the
      credential digest the context names
  C6  the credential file named by the receipt still hashes to that digest (it was not edited
      between consumption and execution)

C5/C6 are what make the context unforgeable-by-a-script: a nonce alone proves nothing (anyone can
invent one), but a receipt that only the supervisor's `authorize()` can create -- bound to a
credential a human approved -- cannot be fabricated by the process it gates.  Together they give the
property that matters: **real F work runs only when a human-approved, identity-bound, one-shot-
consumed credential exists for exactly this run**.

`--synthetic-command` stays fully available (the synthetic acceptance harness never calls the guard,
because it never runs real F work), but it can no longer be used to start the real worker: no
authorization context is ever published for a substituted worker, and `require_execution_context`
makes the smuggled real worker refuse before it touches the model.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import sys
import time
from pathlib import Path

CONTEXT_SCHEMA = 'd3_l1_execution_context_v1'
CONTEXT_ENV = 'D3_L1_EXECUTION_CONTEXT'
SECRET_ENV = 'D3_L1_EXECUTION_SECRET_PATH'
CONTEXT_PREFIX = 'EXECUTION_CONTEXT'
SECRET_PREFIX = 'execution_secret'

# Reasons the guard refuses.  Enumerated so a refusal is always attributable and so an audit can
# tell "no attempt was authorised" from "an attempt was made and rejected".
REFUSAL_REASONS = (
    'no_context_env', 'context_env_missing_file', 'context_unreadable', 'context_schema_unknown',
    'context_missing_field', 'secret_env_absent', 'secret_file_missing', 'secret_not_private',
    'hmac_mismatch', 'run_id_mismatch', 'run_dir_mismatch', 'consumption_receipt_missing',
    'consumption_receipt_invalid', 'consumption_run_id_mismatch',
    'consumption_credential_digest_mismatch', 'credential_file_missing',
    'credential_digest_changed_after_consumption', 'supervisor_pid_absent',
)


class ExecutionContextRefused(RuntimeError):
    """The worker was not started inside an authorised, supervised run; nothing may be loaded."""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_payload(context):
    """The HMAC-covered fields, in a fixed order so the signature is reproducible."""
    return json.dumps({k: context.get(k) for k in (
        'schema', 'run_id', 'run_dir', 'supervisor_pid', 'nonce', 'created_mono',
        'credential_content_sha256', 'credential_file_sha256', 'consumption_path',
        'consumption_credential_content_sha256', 'authorization_age_mono')},
        sort_keys=True, default=repr).encode('utf-8')


def compute_hmac(secret_hex, context):
    return hmac.new(bytes.fromhex(secret_hex), canonical_payload(context),
                    hashlib.sha256).hexdigest()


def publish_execution_context(run_dir, *, authorization_result, supervisor_pid=None):
    """SUPERVISOR ONLY.  Called after the credential is validated AND consumed.

    `authorization_result` is what `d3_l1_v118_authorization.authorize` returned, so the context can
    only exist downstream of a real, one-shot consumption.  Returns a dict with the paths and the
    environment variables the worker must be given.
    """
    run_dir = Path(run_dir)
    consumption = (authorization_result or {}).get('consumption') or {}
    if not consumption.get('path') or not (authorization_result or {}).get(
            'credential_content_sha256'):
        raise ExecutionContextRefused(
            'refusing to publish an execution context without a consumed credential: the context '
            'is the proof of consumption, so it cannot precede it')
    if consumption.get('run_id') != Path(run_dir).name and consumption.get('run_dir') != str(
            run_dir.resolve()):
        raise ExecutionContextRefused('the consumption receipt is for a different run directory')
    secret = secrets.token_hex(32)
    secret_dir = run_dir / 'guard'
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret_path = secret_dir / ('%s_%s.txt' % (SECRET_PREFIX, secrets.token_hex(8)))
    # Exclusive create, owner-only.  The secret never appears on a command line or in a record.
    fd = os.open(str(secret_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as fh:
        fh.write(secret)
        fh.flush()
        os.fsync(fh.fileno())
    context = {
        'schema': CONTEXT_SCHEMA,
        'run_id': consumption.get('run_id'),
        'run_dir': str(run_dir.resolve()),
        'supervisor_pid': supervisor_pid if supervisor_pid is not None else os.getpid(),
        'nonce': secrets.token_hex(16),
        'created_mono': time.monotonic(),
        'authorization_age_mono': time.monotonic() - float(consumption.get('consumed_mono') or 0.0),
        'credential_content_sha256': authorization_result['credential_content_sha256'],
        'credential_file_sha256': consumption.get('credential_file_sha256'),
        'credential_path': authorization_result.get('credential_path'),
        'consumption_path': consumption.get('path'),
        'consumption_credential_content_sha256': consumption.get('credential_content_sha256'),
        'secret_path': str(secret_path),
    }
    context['hmac_sha256'] = compute_hmac(secret, context)
    context_path = run_dir / ('%s_%s.json' % (CONTEXT_PREFIX, context['nonce']))
    with open(context_path, 'x', encoding='utf-8') as fh:
        json.dump(context, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return {'context_path': str(context_path), 'secret_path': str(secret_path),
            'nonce': context['nonce'], 'hmac_sha256': context['hmac_sha256'],
            'env': {CONTEXT_ENV: str(context_path), SECRET_ENV: str(secret_path)}}


def _refuse(reason, **detail):
    raise ExecutionContextRefused('%s%s' % (reason, (' :: ' + json.dumps(detail, default=repr))
                                            if detail else ''))


def validate_execution_context(*, run_dir, environ=None, require_consumption=True):
    """WORKER SIDE read-only check.  Returns the validated context, or raises with a reason.

    Nothing is loaded before this returns.  Every failure mode in `REFUSAL_REASONS` is reachable and
    attributable.
    """
    env = os.environ if environ is None else environ
    context_path = env.get(CONTEXT_ENV)
    if not context_path:
        _refuse('no_context_env', env_var=CONTEXT_ENV,
                note='the real F worker may only be started by the supervisor, which publishes an '
                     'authorization context after consuming a human-approved credential')
    if not Path(context_path).is_file():
        _refuse('context_env_missing_file', context_path=context_path)
    try:
        context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    except Exception as exc:                                              # noqa: BLE001
        _refuse('context_unreadable', context_path=context_path, error=repr(exc))
    if context.get('schema') != CONTEXT_SCHEMA:
        _refuse('context_schema_unknown', schema=context.get('schema'))
    for field in ('run_id', 'run_dir', 'supervisor_pid', 'nonce', 'hmac_sha256',
                  'credential_content_sha256', 'consumption_path', 'secret_path'):
        if not context.get(field):
            _refuse('context_missing_field', field=field)
    if not context.get('supervisor_pid'):
        _refuse('supervisor_pid_absent')
    if not context.get('credential_path'):
        _refuse('context_missing_field', field='credential_path',
                note='the context must name the credential file whose digest was consumed')

    secret_path = env.get(SECRET_ENV) or context.get('secret_path')
    if not secret_path:
        _refuse('secret_env_absent', env_var=SECRET_ENV)
    if not Path(secret_path).is_file():
        _refuse('secret_file_missing', secret_path=secret_path)
    # The secret is created with mode 0600, which POSIX honours.  On Windows `os.chmod` cannot
    # express "owner-only" (the stat mode is synthesised from the read-only bit and reports 0o666
    # for every ordinary file), so the mode check is applied ONLY where it is meaningful; on
    # Windows the protection is the filesystem ACL of the run directory plus the fact that the
    # secret is never passed on a command line and never written into any record.
    if os.name != 'nt':
        mode = stat.S_IMODE(os.stat(secret_path).st_mode)
        if mode & 0o077:
            _refuse('secret_not_private', secret_path=secret_path, mode=oct(mode))
    secret = Path(secret_path).read_text(encoding='utf-8').strip()
    if not secret or len(secret) != 64:
        _refuse('secret_file_missing', secret_path=secret_path, note='not a 32-byte hex secret')
    if compute_hmac(secret, context) != context['hmac_sha256']:
        _refuse('hmac_mismatch', context_path=context_path,
                note='the context was hand-written or edited after the supervisor signed it')

    if str(Path(run_dir).resolve()) != str(Path(context['run_dir']).resolve()):
        _refuse('run_dir_mismatch', expected=context['run_dir'], actual=str(Path(run_dir).resolve()))

    # C5/C6: the consumption receipt and the credential it names.
    if require_consumption:
        receipt_path = Path(context['consumption_path'])
        if not receipt_path.is_file():
            _refuse('consumption_receipt_missing', consumption_path=str(receipt_path),
                    note='no one-shot consumption exists, so no human-approved credential was '
                         'spent on this run')
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        except Exception as exc:                                          # noqa: BLE001
            _refuse('consumption_receipt_invalid', error=repr(exc))
        if receipt.get('one_shot') is not True:
            _refuse('consumption_receipt_invalid', note='the receipt is not marked one-shot')
        if receipt.get('run_id') != context['run_id']:
            _refuse('consumption_run_id_mismatch', receipt_run=receipt.get('run_id'),
                    context_run=context['run_id'])
        if receipt.get('credential_content_sha256') != context['credential_content_sha256']:
            _refuse('consumption_credential_digest_mismatch',
                    receipt=receipt.get('credential_content_sha256'),
                    context=context['credential_content_sha256'])

    # C4 (run id) is checked last so the more specific failures are reported first.
    expected_run = None
    try:
        import d3_l1_v118_persistence as persistence
        expected_run = (persistence.VerifiedReader(run_dir).latest('RUN_IDENTITY') or {}).get(
            'run_id')
    except Exception:                                                     # noqa: BLE001
        expected_run = None
    if expected_run is not None and expected_run != context['run_id']:
        _refuse('run_id_mismatch', frozen=expected_run, context=context['run_id'])

    # C6: the credential file must still hash to the digest that was consumed.
    if require_consumption:
        import d3_l1_v118_authorization as authorization_mod
        try:
            credential = authorization_mod.load_credential(context['credential_path'])
        except Exception as exc:                                          # noqa: BLE001
            _refuse('credential_file_missing', path=context['credential_path'], error=repr(exc))
        if authorization_mod.content_digest(credential) != context['credential_content_sha256']:
            _refuse('credential_digest_changed_after_consumption',
                    note='the credential bytes no longer produce the digest that was consumed')
    return context
