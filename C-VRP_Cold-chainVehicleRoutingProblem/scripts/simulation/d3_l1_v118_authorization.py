#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.19 AUTHORIZATION CREDENTIAL: bind a human approval to the EXACT frozen execution identity.

Why the previous check was not a check
--------------------------------------
`supervise()` accepted any parseable JSON, compared an OPTIONAL `authorization_sha256` field against
the file's own digest, and otherwise proceeded.  A file containing `{}` was therefore a valid
"authorization", and a file could not state which run, which weights, which data, which states,
which budgets or which pre-registration it approved.  A digest of the file cannot express
self-binding identity either: it says "these bytes", not "this approval".

What this module requires instead
---------------------------------
A credential is a HUMAN-AUTHORED record that must declare every binding explicitly:

    approval          who approved it and when (`approved_by`, `approved_utc`, non-empty)
    run               `run_id`
    source            the pre-registration's sha256 (the version that was approved)
    weights           F checkpoint sha256 + backbone sha256
    data              dev_check npz sha256
    contract          objective_profile sha256 + profile_hash + effective contract hash
    states            the 64-state binding sha256 + the exact ordered list digest
    calls             warm-up count, per-state count, max forward calls
    budget            initialize / slot / batch caps (seconds)
    output            the run directory this approval is FOR
    scope             what is authorised, and what is explicitly NOT

Validation compares EVERY ONE of those against the run directory's frozen identity.  A mismatch, an
absent field, an empty JSON object, a wrong directory, a tampered file or a second use of the same
credential all fail CLOSED with an attributable reason.  Nothing here decides the budget or grants
authority; it only refuses to proceed without an approval that names precisely what it approves.

Consumption (one-shot, independently auditable)
----------------------------------------------
Consumption lives OUTSIDE the run directory, in `receipts/authorization_consumption_<digest>.json`,
created with mode 'x' (exclusive).  A second attempt to use the same credential finds the record and
REFUSES.  The record stores the run id, the run directory, the credential's content digest, the
consume moment and the consuming pid, so the fact and the time of use are auditable afterwards even
if the run directory is later destroyed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

CREDENTIAL_SCHEMA = 'd3_l1_authorization_credential_v1'
CONSUMPTION_SCHEMA = 'd3_l1_authorization_consumption_v1'

# Fields whose values are compared against the frozen run identity, in a fixed order so the content
# digest is canonical.  `approval` and `scope` are author-supplied and are NOT identity-checked.
BINDING_FIELDS = (
    'run_id', 'preregistration_sha256', 'f_ckpt_sha256', 'backbone_sha256', 'data_sha256',
    'profile_file_sha256', 'profile_hash', 'effective_contract_hash', 'state_binding_sha256',
    'state_order_sha256', 'forward_targets_sha256', 'dependency_identity_sha256',
    'warmup_calls', 'calls_per_state', 'max_forward_calls',
    'budget_initialize_compile_s', 'budget_slot_span_s', 'budget_batch_s', 'output_dir',
)
HUMAN_FIELDS = ('approved_by', 'approved_utc', 'approval_statement')


class AuthorizationRefused(RuntimeError):
    """The credential does not authorize this execution; nothing was started or consumed."""


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


def content_digest(credential):
    """Digest of the credential's BINDING + APPROVAL content, EXCLUDING `credential_sha256`.

    The exclusion is what makes the field meaningful: the stored digest covers the payload, so a
    single edited byte changes the payload digest and the comparison fails.  A digest of the whole
    file including the digest field would be a fixed point that cannot be computed and therefore
    cannot be checked.
    """
    payload = {k: credential.get(k) for k in BINDING_FIELDS}
    payload['approval'] = {k: credential.get(k) for k in HUMAN_FIELDS}
    payload['scope'] = credential.get('scope')
    payload['schema'] = credential.get('schema')
    return _digest(json.dumps(payload, sort_keys=True, default=repr).encode('utf-8'))


def consumption_dir(root=None):
    """Where one-shot consumption records live.  Deliberately NOT inside the run directory.

    The path is resolved to an ABSOLUTE one.  A relative `--consumption-root` is interpreted against
    whatever the process working directory happens to be when `consume()` runs, so a relative path
    written for one directory and used from another made the exclusive create fail with
    FileNotFoundError.  That crash happened BEFORE any F work and left the credential unconsumed, but
    it is still a defect: the one-shot receipt must be findable deterministically for the whole life
    of the run, and "the receipt directory could not be created" must never be conflated with "the
    credential was spent".
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get('D3_AUTH_CONSUMPTION_DIR')
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / 'results' / 'd3_authorization_receipts').resolve()


# ---------------------------------------------------------------------------------------------
# template + validation
# ---------------------------------------------------------------------------------------------
def build_template(identity, forward_targets, *, preregistration_sha256, output_dir, run_id=None):
    """A credential skeleton with every binding filled in and the HUMAN fields left EMPTY.

    This is what the runner can generate: the machine-readable binding.  It is NOT an approval --
    `approved_by` / `approved_utc` / `approval_statement` are empty and validation refuses them, so
    the template cannot be used as if it were signed.
    """
    caps = identity['budgets']['caps']
    sched = identity['call_schedule']
    return {
        'schema': CREDENTIAL_SCHEMA,
        'run_id': run_id or identity['run_id'],
        'preregistration_sha256': preregistration_sha256,
        'f_ckpt_sha256': identity['identities']['f_ckpt']['sha256'],
        'backbone_sha256': identity['identities']['backbone']['sha256'],
        'data_sha256': identity['identities']['dev_check_npz']['sha256'],
        'profile_file_sha256': identity['identities']['profile_file']['sha256'],
        'profile_hash': identity['identities']['profile_file']['profile_hash'],
        'effective_contract_hash': identity['contracts']['effective_contract_hash'],
        'state_binding_sha256': identity['state_binding_sha256'],
        'state_order_sha256': _digest(json.dumps(
            [[s['day_id'], s['position'], s['state_hash']] for s in identity['state_list']['states']],
            sort_keys=True).encode()),
        'forward_targets_sha256': forward_targets['targets_sha256'],
        'dependency_identity_sha256': identity['dependency_identity_sha256'],
        'warmup_calls': sched['warmup_calls'],
        'calls_per_state': sched['calls_per_state'],
        'max_forward_calls': sched['max_forward_calls'],
        'budget_initialize_compile_s': float(caps['initialize_compile_s']),
        'budget_slot_span_s': float(caps['slot_span_s']),
        'budget_batch_s': float(caps['batch_s']),
        'output_dir': str(Path(output_dir).resolve()),
        'approved_by': '',
        'approved_utc': '',
        'approval_statement': '',
        'scope': {
            'authorised': ['one real-F scoring-interface pass over the 64 frozen states',
                           'at most %d forward calls' % sched['max_forward_calls']],
            'not_authorised': ['F-vs-D-Geo comparison', 'training', 'L2/L3', '40-day batch',
                               'final test', 'install into a live state',
                               'any utility or significance claim'],
        },
        'credential_sha256': None,
    }


def seal_template(template):
    """Fill in the content digest.  The human fields stay empty until a human fills them."""
    out = dict(template)
    out['credential_sha256'] = content_digest(out)
    return out


def validate_credential(credential, *, identity, forward_targets, output_dir, preregistration_sha256,
                        require_human=True):
    """Compare the credential against the frozen identity.  Returns `(ok, detail)`; never raises."""
    problems, detail = [], {}
    if not isinstance(credential, dict):
        return False, {'problems': ['credential is not a JSON object']}
    if not credential:
        return False, {'problems': ['credential is an EMPTY JSON object']}
    if credential.get('schema') != CREDENTIAL_SCHEMA:
        problems.append('unexpected schema %r' % credential.get('schema'))

    caps = identity['budgets']['caps']
    sched = identity['call_schedule']
    expected = {
        'run_id': identity['run_id'],
        'preregistration_sha256': preregistration_sha256,
        'f_ckpt_sha256': identity['identities']['f_ckpt']['sha256'],
        'backbone_sha256': identity['identities']['backbone']['sha256'],
        'data_sha256': identity['identities']['dev_check_npz']['sha256'],
        'profile_file_sha256': identity['identities']['profile_file']['sha256'],
        'profile_hash': identity['identities']['profile_file']['profile_hash'],
        'effective_contract_hash': identity['contracts']['effective_contract_hash'],
        'state_binding_sha256': identity['state_binding_sha256'],
        'state_order_sha256': _digest(json.dumps(
            [[s['day_id'], s['position'], s['state_hash']]
             for s in identity['state_list']['states']], sort_keys=True).encode()),
        'forward_targets_sha256': forward_targets['targets_sha256'],
        'dependency_identity_sha256': identity['dependency_identity_sha256'],
        'warmup_calls': sched['warmup_calls'],
        'calls_per_state': sched['calls_per_state'],
        'max_forward_calls': sched['max_forward_calls'],
        'budget_initialize_compile_s': float(caps['initialize_compile_s']),
        'budget_slot_span_s': float(caps['slot_span_s']),
        'budget_batch_s': float(caps['batch_s']),
        'output_dir': str(Path(output_dir).resolve()),
    }
    rows = {}
    for field in BINDING_FIELDS:
        want, got = expected[field], credential.get(field)
        if field in ('budget_initialize_compile_s', 'budget_slot_span_s', 'budget_batch_s'):
            try:
                got = float(got)
            except (TypeError, ValueError):
                got = None
        same = (got == want)
        rows[field] = {'expected': want, 'declared': got, 'ok': same}
        if not same:
            problems.append('%s_mismatch' % field)
    detail['bindings'] = rows

    if require_human:
        missing = [f for f in HUMAN_FIELDS if not str(credential.get(f) or '').strip()]
        detail['missing_human_fields'] = missing
        if missing:
            problems.append('no_human_approval:%s' % ','.join(missing))

    declared_digest = credential.get('credential_sha256')
    recomputed = content_digest(credential)
    detail['content_digest'] = {'declared': declared_digest, 'recomputed': recomputed,
                                'ok': declared_digest == recomputed}
    if not declared_digest:
        problems.append('credential_sha256_absent')
    elif declared_digest != recomputed:
        problems.append('credential_sha256_mismatch:tampered_or_edited_after_sealing')

    detail['problems'] = problems
    return (not problems), detail


def validate_pre_registration(path, declared_sha256):
    """The credential names the pre-registration it approved; that file must still hash to it."""
    actual = sha256_file(path)
    return {'path': str(path), 'declared': declared_sha256, 'actual': actual,
            'ok': bool(actual) and actual == declared_sha256}


def consume(credential, *, run_dir, root=None, consumer_pid=None):
    """One-shot consumption.  A second call with the same credential REFUSES.

    Returns the consumption record.  The record is created with mode 'x', so two concurrent
    attempts cannot both succeed.
    """
    directory = consumption_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    digest = content_digest(credential)
    path = directory / ('authorization_consumption_%s.json' % digest)
    if path.exists():
        existing = {}
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except Exception:                                                 # noqa: BLE001
            pass
        raise AuthorizationRefused(
            'this credential has ALREADY been consumed (one-shot): %s' % json.dumps(
                {k: existing.get(k) for k in ('run_id', 'consumed_utc', 'consumer_pid',
                                              'run_dir')}, default=repr))
    record = {
        'schema': CONSUMPTION_SCHEMA,
        'credential_content_sha256': digest,
        'credential_file_sha256': credential.get('_file_sha256'),
        'run_id': credential.get('run_id'),
        'run_dir': str(Path(run_dir).resolve()),
        'preregistration_sha256': credential.get('preregistration_sha256'),
        'maximum_forward_calls': credential.get('max_forward_calls'),
        'approved_by': credential.get('approved_by'),
        'approved_utc': credential.get('approved_utc'),
        'consumed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'consumed_mono': time.monotonic(),
        'consumer_pid': consumer_pid if consumer_pid is not None else os.getpid(),
        'one_shot': True,
        'note': 'this record is the ONLY permitted evidence of consumption; the credential may not '
                'be used again for any run',
    }
    try:
        with open(path, 'x', encoding='utf-8') as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError:
        raise AuthorizationRefused('a concurrent consumer won the one-shot race for %s' % digest)
    record['path'] = str(path)
    return record


def load_credential(path):
    """Read the credential file and attach its own file digest for the consumption record."""
    p = Path(path)
    if not p.is_file():
        raise AuthorizationRefused('authorization file does not exist: %s' % p)
    raw = p.read_bytes()
    if not raw.strip():
        raise AuthorizationRefused('authorization file is EMPTY: %s' % p)
    try:
        credential = json.loads(raw.decode('utf-8'))
    except Exception as exc:                                              # noqa: BLE001
        raise AuthorizationRefused('authorization file is not valid JSON: %r' % exc)
    if not isinstance(credential, dict) or not credential:
        raise AuthorizationRefused('authorization file is not a non-empty JSON object')
    credential = dict(credential)
    credential['_file_sha256'] = _digest(raw)
    credential['_source_path'] = str(p.resolve())
    return credential


def authorize(credential, *, identity, forward_targets, output_dir, preregistration_path,
              run_dir, consumption_root=None, require_human=True):
    """The full gate: validate against the frozen identity, then consume one-shot.

    Returns `{'ok', 'detail', 'consumption'}`.  Raises `AuthorizationRefused` with the reasons when
    anything does not line up, so a caller cannot proceed on a partial match.
    """
    ok, detail = validate_credential(
        credential, identity=identity, forward_targets=forward_targets, output_dir=output_dir,
        preregistration_sha256=sha256_file(preregistration_path), require_human=require_human)
    pre = validate_pre_registration(preregistration_path,
                                    credential.get('preregistration_sha256'))
    detail['preregistration'] = pre
    if not pre['ok']:
        detail['problems'] = list(detail['problems']) + ['preregistration_sha256_mismatch']
        ok = False
    if not ok:
        raise AuthorizationRefused('authorization refused: %s' % json.dumps(
            detail['problems'], default=repr))
    record = consume(credential, run_dir=run_dir, root=consumption_root)
    return {'ok': True, 'detail': detail, 'consumption': record,
            'credential_path': str(Path(credential.get('_source_path') or '')),
            'credential_content_sha256': content_digest(credential)}
