#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 real-F forward worker on the repaired persistence + supervision path.

This module EXECUTES NOTHING BY ITSELF.  `main()` refuses unless all three of the following hold:

  1. `--supervise` was passed together with an `--authorization` file whose SHA-256 equals the
     value frozen in the signed pre-registration (the same scoped pattern V1.9 used for its single
     consumed authorization), and
  2. the run directory has been materialised by `--freeze-identity` and every registered identity
     still matches the frozen bytes, and
  3. the caller is not in the consumed-run namespace (see `d3_l1_v118_run_identity.validate_run_id`).

What the worker does per state, unchanged from V1.11's fixed schedule
--------------------------------------------------------------------
  warm-up : the FIRST state's input, one compiled forward (call 1 of 129)
  slot i  : the SAME input scored exactly twice (calls 2k, 2k+1), and the two results must be
            bit-identical in dtype, shape and bytes
  every slot record is published to a FRESH destination name and read back to prove durability

What is different from V1.11 (and why it cannot fail the same way)
-----------------------------------------------------------------
  * `STAGE.json`, a fixed destination replaced while the supervisor read it, is GONE.  Progress is
    published as immutable, uniquely named records; the supervisor derives the current stage from
    those records.  There is nothing left to replace, so the WinError 5 race is structurally
    impossible rather than merely less likely.
  * the per-slot score artifact and the per-slot JSON record are both published through
    `publish_new(..., verify=True)`, and a failed verification aborts the slot instead of leaving
    an unverified file behind.
  * the supervisor itself lives in `--supervise`, not in this module's `__main__`, so a worker can
    never accidentally become its own supervisor.

Boundaries: no D-Geo call, no comparison, no training, no install, no rollout, no new state, no
change to any frozen input.  Every one of the 64 states stays in the denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/models',
           'scripts/tests', 'scripts/evaluation'):
    sys.path.insert(0, str(EXT / _p))

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V110_REPORT = EV / 'v110_evidence/INPUT_QUALIFICATION_V110.json'
PHASE_JOURNAL_SRC = EXT / 'scripts/tests/d3_l1_v112_phase_journal.py'
FORWARD_CONTROL_SRC = EXT / 'scripts/tests/d3_l1_v116_forward_control.py'

IDENTITY_START_RECORD = 'IDENTITY_START'
IDENTITY_END_RECORD = 'IDENTITY_END'
SLOT_START_RECORD = 'SLOT_START'
SLOT_END_RECORD = 'SLOT_END'
ROUND_RECORD_PREFIX = 'ROUND'


def _import(name, path):
    """Import a module by file path so the run never depends on sys.path ordering."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers():
    journal = _import('d3_l1_v112_phase_journal_v118', PHASE_JOURNAL_SRC)
    control = _import('d3_l1_v116_forward_control_v118', FORWARD_CONTROL_SRC)
    import d3_l1_v118_persistence as persistence
    import d3_l1_v118_run_identity as identity
    import d3_l1_v118_targets as targets
    return journal, control, persistence, identity, targets


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================================
# identity + freeze
# =============================================================================================
def freeze_identity(run_dir, run_id, *, now_utc, watched_extra=None):
    """Materialise the run identity, the expected-row targets and the identity target set.

    `watched_extra` appends extra absolute paths to the trailing-identity target set.  It exists so
    the acceptance harness can watch a disposable file and mutate it; a production run passes
    nothing and therefore watches exactly the frozen code + weights + data + profile.
    """
    _, _, persistence, identity, targets = load_helpers()
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit('refusing to reuse a non-empty run directory: %s' % run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'scores').mkdir(exist_ok=True)
    ident = identity.build_run_identity(run_id, created_utc=now_utc)
    publish = persistence.publish_new(run_dir, 'RUN_IDENTITY', ident)
    qualification = targets.input_qualification_digests(V110_REPORT)
    fwd = targets.forward_targets(ident, qualification)
    publish_fwd = persistence.publish_new(run_dir, 'FORWARD_TARGETS', fwd)
    watched = identity.watched_identity_paths()
    if watched_extra:
        watched = list(watched) + [Path(p) for p in watched_extra]
    ident_targets = targets.identity_targets(run_id, watched, spec='code+weights+data+profile'
                                             + ('+acceptance_extra' if watched_extra else ''))
    publish_targets = persistence.publish_new(run_dir, 'IDENTITY_TARGETS', ident_targets)
    report = {
        'schema': 'd3_l1_v118_freeze',
        'run_id': run_id, 'run_dir': str(run_dir), 'created_utc': now_utc,
        'identity': publish, 'forward_targets': publish_fwd, 'identity_targets': publish_targets,
        'n_target_rows': fwd['n_rows'], 'targets_sha256': fwd['targets_sha256'],
        'targets_ok': fwd['ok'], 'target_errors': fwd['errors'],
        'budgets': ident['budgets'], 'call_schedule': ident['call_schedule'],
        'identity_match': identity.identity_match_report(ident),
        'watched_paths': [str(p) for p in watched],
        'wrote': ['RUN_IDENTITY_*.json', 'FORWARD_TARGETS_*.json', 'IDENTITY_TARGETS_*.json'],
        'not_run': ['no F forward', 'no D-Geo', 'no comparison', 'no training', 'no install'],
    }
    report['identity_match_ok'] = report['identity_match']['ok']
    persistence.write_once(run_dir / 'FREEZE_REPORT.json', report)
    return report


def load_frozen(run_dir):
    _, _, persistence, identity, targets = load_helpers()
    run_dir = Path(run_dir)
    reader = persistence.VerifiedReader(run_dir)
    ident = reader.latest('RUN_IDENTITY')
    fwd = reader.latest('FORWARD_TARGETS')
    idt = reader.latest('IDENTITY_TARGETS')
    if ident is None or fwd is None or idt is None:
        raise SystemExit('run directory is not frozen: RUN_IDENTITY / FORWARD_TARGETS / '
                         'IDENTITY_TARGETS records are required')
    return {'run_dir': run_dir, 'identity': ident, 'forward': fwd, 'identity_targets': idt,
            'reader': reader, 'persistence': persistence, 'identity_mod': identity,
            'targets_mod': targets}


def _make_ready_probe(out_dir):
    """V1.18 READY semantics: an IMMUTABLE, uniquely named READY record must exist AND verify.

    The worker can never publish a fixed `READY.json` -- replacing a fixed destination name under a
    reader is exactly the V1.11 `WinError 5` failure -- so the supervisor must not look for one
    either.  The record is accepted only when it is retrievable through the verified reader and its
    payload is a well-formed readiness statement carrying the run id, the warm-up call count and the
    warm-up score digest, so "ready" cannot be forged by an empty or truncated file.  The probe is
    bound to ONE run directory, so it can never read another run's READY record.
    """
    import d3_l1_v118_persistence as persistence
    reader = persistence.VerifiedReader(Path(out_dir))

    def probe(_out):
        record = reader.latest('READY')
        if record is None:
            return {'ready': False, 'probe': 'immutable READY record', 'reason': 'absent',
                    'run_dir': str(out_dir)}
        problems = []
        if record.get('kind') != 'initialization_and_compilation_finished':
            problems.append('kind is %r' % record.get('kind'))
        if not record.get('run_id'):
            problems.append('run_id absent')
        if record.get('n_warmup_calls') != 1:
            problems.append('n_warmup_calls is %r' % record.get('n_warmup_calls'))
        if not record.get('warmup_scores_sha256'):
            problems.append('warmup_scores_sha256 absent')
        if not isinstance(record.get('mono'), (int, float)):
            problems.append('mono absent')
        return {'ready': not problems, 'probe': 'immutable READY record',
                'published_name': record.get('published_name'), 'run_id': record.get('run_id'),
                'n_records': reader.count('READY'), 'problems': problems,
                'run_dir': str(out_dir)}
    return probe


def _failure_reason_for(failed_checks):
    """Map the failed checks to the frozen failure vocabulary (first match wins)."""
    for check, reason in (('shape_matches_ordered_legal', 'score_shape'),
                          ('all_finite', 'nan'),
                          ('repeat_exact_type_dtype_shape_bytes', 'score_len'),
                          ('first_argmax_rule', 'score_shape'),
                          ('span_within_slot_cap', 'timeout')):
        if check in failed_checks:
            return reason
    return 'score_shape'


# =============================================================================================
# worker
# =============================================================================================
def worker(out_dir):
    journal, _, persistence, identity_mod, targets = load_helpers()
    out_dir = Path(out_dir)

    # =========================================================================================
    # EXECUTION-CONTEXT GUARD -- the FIRST thing the real worker does.
    # Nothing below this point may run without it: no frozen-identity read, no JAX import, no
    # checkpoint load, no scorer call.  `--worker` invoked directly, or smuggled in through
    # `--synthetic-command`, has no authorization context and therefore refuses here.
    # =========================================================================================
    import d3_l1_v120_execution_context as execctx
    try:
        context = execctx.validate_execution_context(run_dir=out_dir)
    except execctx.ExecutionContextRefused as exc:
        refusal = {'run_id': None, 'stage': 'execution_context_guard', 'refused': True,
                   'reason': str(exc), 'argv': sys.argv, 'pid': os.getpid(),
                   'note': 'the real F worker may only be started by the supervisor, after a '
                           'human-approved credential has been validated and consumed one-shot'}
        try:
            persistence.publish_new(out_dir, 'WORKER_GUARD_REFUSED', refusal)
        except Exception:                                                  # noqa: BLE001
            pass
        print('EXECUTION CONTEXT REFUSED: %s' % exc, file=sys.stderr)
        return 1

    frozen = load_frozen(out_dir)
    ident, fwd, idt = frozen['identity'], frozen['forward'], frozen['identity_targets']
    run_id = ident['run_id']
    phases = out_dir / 'phases'
    journal_obj = journal.PhaseJournal(phases)
    publish = persistence.publish_new
    record = {'schema': 'd3_l1_v118_worker', 'run_id': run_id, 'pid': os.getpid(),
              'argv': sys.argv, 'execution_context_nonce': context.get('nonce'),
              'execution_context_hmac_sha256': context.get('hmac_sha256'),
              'supervisor_pid': context.get('supervisor_pid')}
    started = time.monotonic()
    try:
        # ---- identity, start -------------------------------------------------------------
        for key, rec in identity_mod.identity_match_report(ident)['rows'].items():
            if not rec['ok']:
                raise RuntimeError('registered identity mismatch before any forward: %s' % key)
        for path_str, expected in idt['paths'].items():
            actual = sha256_file(path_str)
            if actual != expected:
                raise RuntimeError('watched identity mismatch before any forward: %s'
                                   % Path(path_str).name)
        publish(phases, 'identity_start', {'stage': 'identity_start', 'run_id': run_id,
                                           'paths': idt['paths']})

        # ---- model ------------------------------------------------------------------------
        journal_obj.publish('initialize')
        import numpy as np
        import jax
        from flax import nnx                                                     # noqa: F401
        from mtrained_replanner import load_trained_model, mtrained_score_fn
        from mpre_trained import partition
        assert jax.default_backend() == 'cpu', 'the F interface run is a CPU-only protocol'
        from d3_l1_v18_readonly_audit import typed_bytes
        model_path = EXT / ident['identities']['f_ckpt']['path']
        backbone_path = REPO / ident['identities']['backbone']['path']
        if sha256_file(model_path) != ident['identities']['f_ckpt']['sha256']:
            raise RuntimeError('F checkpoint digest changed before loading')
        if sha256_file(backbone_path) != ident['identities']['backbone']['sha256']:
            raise RuntimeError('backbone digest changed before loading')
        model, _ = load_trained_model(str(backbone_path), str(model_path), seed=42)
        _gd, train, frozen_params, other = partition(model)
        meta = pickle.loads(model_path.read_bytes())
        trained = jax.tree_util.tree_flatten_with_path(train)[0]
        reference = jax.tree_util.tree_flatten_with_path(meta['train_state'])[0]
        restored_exact = len(trained) == len(reference) and all(
            str(p) == str(q) and typed_bytes(np.asarray(x)) == typed_bytes(np.asarray(y))
            for (p, x), (q, y) in zip(trained, reference))
        if not restored_exact:
            raise RuntimeError('trained leaves were not restored bit-exactly')
        score = mtrained_score_fn(model, Nv_max=int(fwd['Nv_max']), M_max=int(fwd['M_max']))
        model_report = {
            'run_id': run_id, 'n_train_leaves': len(trained), 'restored_exact': restored_exact,
            'n_frozen_leaves': len(jax.tree_util.tree_leaves(frozen_params)),
            'n_other_leaves': len(jax.tree_util.tree_leaves(other)),
            'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
            'f_ckpt_sha256': sha256_file(model_path),
            'backbone_sha256': sha256_file(backbone_path),
            'label': ident['labels']['F'], 'backbone_label': ident['labels']['backbone'],
            'S_TRAIN': float(meta['s_TRAIN']),
            'initialization_wall_s': time.monotonic() - started,
            'Nv_max': int(fwd['Nv_max']), 'M_max': int(fwd['M_max'])}
        publish(out_dir, 'MODEL_LOAD', model_report)

        # ---- warm-up: one compiled forward on the FIRST target ----------------------------
        journal_obj.publish('compile')
        first = fwd['rows'][0]
        payload = pickle.loads((EV / 'v110_evidence' / first['derived_relpath']).read_bytes())
        inputs = payload['original']['inputs']
        if hashlib.sha256(typed_bytes(inputs)).hexdigest() != first['input_sha256']:
            raise RuntimeError('the first state input does not match its pre-qualified digest')
        warm_started = time.monotonic()
        warm = np.asarray(score(inputs))
        warm_report = {'run_id': run_id, 'calls': 1, 'wall_s': time.monotonic() - warm_started,
                       'scores_dtype': warm.dtype.str, 'scores_shape': list(warm.shape),
                       'scores_sha256': hashlib.sha256(warm.tobytes()).hexdigest(),
                       'day_id': first['day_id'], 'position': first['position'],
                       'input_sha256': first['input_sha256']}
        publish(out_dir, 'WARMUP', warm_report)
        ready_record = {'run_id': run_id, 'mono': time.monotonic(),
                        'initialize_and_compile_s': time.monotonic() - started,
                        'kind': 'initialization_and_compilation_finished',
                        'worker_pid': os.getpid(),
                        'warmup_scores_sha256': warm_report['scores_sha256'],
                        'n_warmup_calls': 1}
        publish(out_dir, 'READY', ready_record)

        # ---- the 64 slots, frozen order, all kept in the denominator ----------------------
        # V1.18 pre-registration §5: a slot whose checks FAIL is a TERMINAL condition.  The loop
        # stops immediately, no later state may be scored, and the round is revoked.  The first
        # failing slot's record is published BEFORE the stop so the reason is durable.
        first_failure = None
        for row in fwd['rows']:
            day, pos = int(row['day_id']), int(row['position'])
            slot_started = time.monotonic()
            journal_obj.publish('slot', day_id=day, position=pos)
            try:
                derived = EV / 'v110_evidence' / row['derived_relpath']
                if sha256_file(derived) != row['derived_sha256']:
                    raise RuntimeError('V1.10 derived artifact digest changed')
                payload = pickle.loads(derived.read_bytes())
                inputs = payload['original']['inputs']
                input_sha = hashlib.sha256(typed_bytes(inputs)).hexdigest()
                if input_sha != row['input_sha256']:
                    raise RuntimeError('state input digest differs from the pre-qualified value')
                t0 = time.monotonic()
                a = np.asarray(score(inputs))
                t1 = time.monotonic()
                b = np.asarray(score(inputs))
                t2 = time.monotonic()
                checks = {
                    'shape_matches_ordered_legal': (a.ndim == 1 and b.ndim == 1
                                                    and a.shape[0] == row['n_legal_expected']
                                                    and b.shape[0] == row['n_legal_expected']),
                    'all_finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                    'repeat_exact_type_dtype_shape_bytes': typed_bytes(a) == typed_bytes(b),
                    'span_within_slot_cap': (t2 - slot_started)
                    <= float(ident['budgets']['caps']['slot_span_s']),
                    'first_argmax_rule': bool(a.size > 0 and np.isfinite(a).all()
                                              and int(np.argmax(a))
                                              == int(np.flatnonzero(a == a.max())[0])),
                }
                status = ('FORWARD_CHECKS_PASS' if all(checks.values())
                          else 'FAILED_CHECKS')
                failed_checks = sorted(k for k, v in checks.items() if not v)
                raw_pub = publish(out_dir / 'scores', 'SCORES',
                                  pickle.dumps({'scores_first': a, 'scores_repeat': b,
                                                'input_sha256': input_sha,
                                                'legal_sha256': row['legal_sha256'],
                                                'run_id': run_id,
                                                'label': ident['labels']['F']},
                                               protocol=4),
                                  verify=True)
                slot_record = {
                    'run_id': run_id, 'day_id': day, 'position': pos, 'status': status,
                    'checks': checks, 'failed_checks': failed_checks,
                    'failure_reason': None if status == 'FORWARD_CHECKS_PASS'
                    else _failure_reason_for(failed_checks),
                    'raw_relpath': str(Path(raw_pub['path']).relative_to(out_dir)
                                       .as_posix()),
                    'raw_sha256': raw_pub['sha256'],
                    'input_sha256': input_sha, 'legal_sha256': row['legal_sha256'],
                    'state_hash': row['state_hash'],
                    'scores_dtype': a.dtype.str, 'scores_shape': list(a.shape),
                    'scores_first_hex': a.tobytes().hex(),
                    'scores_repeat_hex': b.tobytes().hex(),
                    'forward_first_wall_s': t1 - t0, 'forward_repeat_wall_s': t2 - t1,
                    'span_wall_s': time.monotonic() - slot_started,
                    'F_calls': 2, 'label': ident['labels']['F'],
                    'published_name': None,
                }
                json_pub = publish(out_dir / 'scores',
                                   'SLOT_day%02d_pos%d' % (day, pos), slot_record)
                slot_record = dict(slot_record, published_name=json_pub['name'])
                journal_obj.publish('slot_finished', day_id=day, position=pos, status=status)
                if status != 'FORWARD_CHECKS_PASS':
                    first_failure = {'day_id': day, 'position': pos, 'status': status,
                                     'failed_checks': failed_checks,
                                     'failure_reason': slot_record['failure_reason'],
                                     'row_index': int(row['index'])}
                    publish(out_dir, 'FIRST_FAILURE',
                            dict(first_failure, run_id=run_id,
                                 policy='terminal: no later state may be scored; the round is '
                                        'revoked by the supervisor'))
                    break
            except Exception as exc:                                  # noqa: BLE001
                slot_record = {'run_id': run_id, 'day_id': day, 'position': pos,
                               'status': 'FAILED_EXCEPTION', 'exception_type': type(exc).__name__,
                               'error': str(exc), 'traceback': traceback.format_exc(),
                               'F_calls': 0, 'span_wall_s': time.monotonic() - slot_started,
                               'input_sha256': None, 'legal_sha256': None,
                               'scores_dtype': None, 'scores_shape': None,
                               'scores_first_hex': None, 'scores_repeat_hex': None}
                publish(out_dir / 'scores', 'SLOT_day%02d_pos%d' % (day, pos), slot_record)
                journal_obj.publish('slot_finished', day_id=day, position=pos, status='FAILED')
                publish(out_dir, 'FIRST_FAILURE',
                        {'run_id': run_id, 'day_id': day, 'position': pos,
                         'status': 'FAILED_EXCEPTION', 'exception_type': type(exc).__name__,
                         'error': str(exc),
                         'policy': 'terminal: no later state may be scored'})
                raise

        if first_failure is not None:
            # The round is over, but the trailing identity record is STILL published: the V1.16
            # revocation needs a tail to adjudicate, and its absence would be reported as a
            # missing tail rather than as the slot failure that actually happened.
            watched_now = {str(p): sha256_file(p) for p in idt['paths']}
            publish(phases, 'identity_end',
                    {'stage': 'identity_end', 'run_id': run_id, 'paths': watched_now,
                     'worker_ended_mono': time.monotonic(),
                     'ended_because': 'slot_failure', 'first_failure': first_failure})
            record.update(ok=False, terminated_on_first_failure=True, first_failure=first_failure,
                          worker_wall_s=time.monotonic() - started)
            publish(out_dir, 'WORKER_RESULT', record)
            return 1

        # ---- trailing identity ------------------------------------------------------------
        watched_now = {str(p): sha256_file(p) for p in idt['paths']}
        end_record = {'stage': 'identity_end', 'run_id': run_id, 'paths': watched_now,
                      'worker_ended_mono': time.monotonic()}
        end_pub = publish(phases, 'identity_end', end_record)
        record.update(ok=True, identity_end=end_pub['name'], worker_wall_s=time.monotonic() - started)
        journal_obj.publish('completed', run_id=run_id)
        publish(out_dir, 'WORKER_RESULT', record)
        return 0
    except BaseException as exc:                                       # noqa: BLE001
        record.update(ok=False, type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        try:
            publish(out_dir, 'WORKER_FAILURE', record)
        except Exception:                                              # noqa: BLE001
            pass
        return 1


# =============================================================================================
# supervisor
# =============================================================================================
def supervise(out_dir, *, authorization, dry_command=None, caps_override=None,
              preregistration=None, consumption_root=None, authorization_record=None):
    """Drive the worker through the V1.16 contained supervisor, then adjudicate the round.

    `dry_command` exists ONLY for the synthetic acceptance harness: it substitutes the worker
    process so the persistence / deadline / kill / revocation semantics can be exercised without
    loading F.  `caps_override` is refused unless `dry_command` is given, so the FROZEN budgets can
    never be relaxed by a real run.

    AUTHORIZATION (blocker 1): a real run requires a credential that declares the run id, the
    pre-registration, the weights, the data, the contract, the 64-state order, the call counts, the
    budgets and the output directory, is compared to the frozen identity field by field, and is
    consumed ONCE with an auditable receipt.  A parseable JSON file is not an authorization.
    """
    journal, control, persistence, identity_mod, targets = load_helpers()
    import d3_l1_v118_authorization as authorization_mod
    out_dir = Path(out_dir)
    frozen = load_frozen(out_dir)
    ident, fwd, idt = frozen['identity'], frozen['forward'], frozen['identity_targets']
    run_id = ident['run_id']
    identity_mod.validate_run_id(run_id)
    if caps_override is not None and dry_command is None:
        raise SystemExit('refusing a budget override for a real F run: the frozen caps are '
                         'binding (only the synthetic acceptance harness may shorten them)')

    authorization_evidence = {'required': dry_command is None, 'granted': False,
                              'credential_path': str(authorization) if authorization else None,
                              'consumption': None, 'detail': None}
    if authorization is not None and dry_command is not None:
        # A real authorization must never be spent on a SUBSTITUTED worker: the substituted process
        # is not the guarded real worker, so the consumption would buy nothing while permanently
        # burning a one-shot credential.  The synthetic harness runs without one.
        authorization_evidence['refused'] = ('refusing to combine an authorization with '
                                             '--synthetic-command')
        persistence.publish_new(out_dir, 'AUTHORIZATION_REFUSED', authorization_evidence)
        raise SystemExit('refusing to combine an authorization with --synthetic-command: a '
                         'one-shot credential must not be spent on a substituted worker')
    if dry_command is None:
        if not authorization:
            raise SystemExit('refusing to run the real F worker without an authorization file')
        if not preregistration:
            raise SystemExit('refusing to run the real F worker without --preregistration: the '
                             'credential binds a pre-registration digest, and that file must be '
                             'present to check it')
        try:
            credential = authorization_mod.load_credential(authorization)
            granted = authorization_mod.authorize(
                credential, identity=ident, forward_targets=fwd, output_dir=out_dir,
                preregistration_path=Path(preregistration), run_dir=out_dir,
                consumption_root=consumption_root)
        except authorization_mod.AuthorizationRefused as exc:
            authorization_evidence['refused'] = str(exc)
            persistence.publish_new(out_dir, 'AUTHORIZATION_REFUSED', authorization_evidence)
            raise SystemExit('AUTHORIZATION REFUSED: %s' % exc)
        authorization_evidence.update(granted=True, detail=granted['detail'],
                                      consumption=granted['consumption'],
                                      credential_path=str(authorization),
                                      credential_content_sha256=granted[
                                          'credential_content_sha256'])
        if authorization_record is not None:
            Path(authorization_record).write_text(
                json.dumps(authorization_evidence, indent=2, sort_keys=True, default=repr),
                encoding='utf-8')
    elif authorization is not None:
        # unreachable: the combination above is refused, kept as a belt-and-braces branch
        authorization_evidence['detail'] = ('ignored: the synthetic harness never authorises '
                                            'a real execution')
    persistence.publish_new(out_dir, 'AUTHORIZATION_EVIDENCE', authorization_evidence)

    watched = {str(p): sha256_file(p) for p in idt['paths']}
    freeze_report = json.loads((out_dir / 'FREEZE_REPORT.json').read_text(encoding='utf-8'))
    identity = {path: digest for path, digest in watched.items()}
    target_by_key = {(int(r['day_id']), int(r['position'])): r for r in fwd['rows']}

    # The environment the worker process is launched with.  For a REAL run it carries the
    # execution-context variables, which the worker's first action verifies: that is what stops
    # `--worker` (or a smuggled real worker behind `--synthetic-command`) from loading the model.
    worker_env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
    execution_context_evidence = {'published': False, 'real_run': dry_command is None}
    if dry_command is None:
        import d3_l1_v120_execution_context as execctx
        try:
            published = execctx.publish_execution_context(out_dir, authorization_result=granted)
        except execctx.ExecutionContextRefused as exc:
            execution_context_evidence['refused'] = str(exc)
            persistence.publish_new(out_dir, 'EXECUTION_CONTEXT_REFUSED',
                                    execution_context_evidence)
            raise SystemExit('EXECUTION CONTEXT REFUSED: %s' % exc)
        worker_env.update(published['env'])
        execution_context_evidence.update(published=True, nonce=published['nonce'],
                                          hmac_sha256=published['hmac_sha256'],
                                          context_path=published['context_path'],
                                          env_vars=sorted(published['env']))
        # A real run may NEVER substitute the worker process: the substituted process would not be
        # the guarded real worker, and `--synthetic-command` exists only for the acceptance harness.
        if dry_command is not None:
            raise SystemExit('refusing to combine an authorization with --synthetic-command')
    else:
        execution_context_evidence['detail'] = (
            'no execution context is published for a substituted worker: the synthetic harness '
            'runs no real F work, and the real worker refuses to start without a context')
    persistence.publish_new(out_dir, 'EXECUTION_CONTEXT_EVIDENCE', execution_context_evidence)

    if dry_command is not None:
        command = list(dry_command)
    else:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--out-dir', str(out_dir)]

    def verify_record(rec, _out):
        key = (rec.get('day_id'), rec.get('position'))
        target = target_by_key.get(key)
        if target is None:
            return False, {'problems': ['row is outside the frozen target set']}
        return targets.verify_slot_record(rec, target)

    caps = dict(ident['budgets']['caps'])
    if caps_override:
        caps.update(caps_override)
        caps['overridden_for_synthetic_acceptance'] = True

    def tail_publisher(end_identity):
        """Publish the supervisor's trailing identity record to a FRESH destination name.

        This is the V1.18 replacement for overwriting `CONTROL_IDENTITY_END.json`: the record is
        immutable, uniquely named and its durability is verified by re-reading it.
        """
        return persistence.publish_new(out_dir, 'CONTROL_IDENTITY_END', end_identity)

    control_result = control.run_contained(
        command, out_dir, identity, init_cap=float(caps['initialize_compile_s']),
        slot_cap=float(caps['slot_span_s']), batch_cap=float(caps['batch_s']),
        env=worker_env,
        tail_publisher=tail_publisher, ready_probe=_make_ready_probe(out_dir))
    tail_path = (control_result.get('tail_publication') or {}).get('path')
    rows = control.reconcile_rows(out_dir, control_result, verify_record=verify_record,
                                  rows_dir='scores', row_glob=targets.SLOT_RECORD_GLOB)
    round_report, rows = control.round_verdict(out_dir, control_result, rows,
                                               expected_rows=int(fwd['expected_rows']),
                                               tail_path=tail_path)
    # V1.18 addition: the round also requires the TAIL IDENTITY payload to cover the frozen paths.
    # The worker's trailing identity record lives in the phase journal (a fresh destination name,
    # never a replaced fixed name), so it is read from there.
    tail_problems = []
    try:
        tail = persistence.VerifiedReader(out_dir / 'phases').latest('identity_end')
    except Exception as exc:                                           # noqa: BLE001
        tail = None
        tail_problems.append('%s: %s' % (type(exc).__name__, exc))
    if tail is None:
        tail_problems.append('no identity_end record was published in the phase journal')
    else:
        ok, detail = targets.verify_identity(tail, {'run_id': run_id, 'paths': watched})
        if not ok:
            tail_problems += detail['problems']
    if tail_problems and round_report['status'] == 'ROUND_PASS':
        round_report = dict(round_report, status='ROUND_REVOKED', revoked=True,
                            reason='TAIL_IDENTITY_PAYLOAD_INVALID',
                            tail_identity_problems=tail_problems)
        persistence.publish_new(out_dir, ROUND_RECORD_PREFIX,
                                round_report)
        rows = [dict(r, qualified=False, revoked=True,
                     status='REVOKED_' + str(r.get('status') or 'UNKNOWN')) for r in rows]
    report = {
        'schema': 'd3_l1_v118_supervision',
        'run_id': run_id,
        'isolation': ident['isolation_claim'],
        'freeze_report_sha256_present': bool(freeze_report),
        'targets_sha256': fwd['targets_sha256'],
        'identity_payload_problems': tail_problems,
        'control': {k: v for k, v in control_result.items() if k != 'timeline'},
        'timeline': control_result.get('timeline'),
        'round': round_report,
        'n_rows_qualified': sum(1 for r in rows if r.get('qualified')),
        'expected_rows': fwd['expected_rows'],
        'rows_status_counts': {s: sum(1 for r in rows if r.get('status') == s)
                               for s in sorted({r.get('status') for r in rows})},
        'forward_calls_declared': fwd['max_forward_calls'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'F_D_Geo_comparisons': 0, 'install_calls': 0, 'training_runs': 0, 'new_rollouts': 0,
        'verdict': ('REAL_F_FORWARD_ENGINEERING_PASS'
                    if round_report['status'] == 'ROUND_PASS'
                    and control_result.get('ok') and not tail_problems
                    else 'REAL_F_FORWARD_ENGINEERING_BLOCKED'),
        'label': ident['labels']['F'],
    }
    persistence.publish_new(out_dir, 'SUPERVISION_RESULT', report)
    persistence.write_once(out_dir / 'ROUND_VERDICT.json',
                           {'status': round_report['status'],
                            'reason': round_report.get('reason'),
                            'n_rows_qualified': report['n_rows_qualified'],
                            'verdict': report['verdict']})
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('control', 'timeline')}, default=repr))
    return 0 if report['verdict'] == 'REAL_F_FORWARD_ENGINEERING_PASS' else 1


def readiness_report(out_dir, *, run_id, acceptance=None, verifier=None, synthetic_execution=None):
    """Compute the signature readiness from the artifacts on disk -- no hand-set flag.

    The verdict is `READY_FOR_USER_SIGNATURE` only when EVERY item in
    `d3_l1_v118_run_identity.compute_missing_items` is satisfied by an artifact.  Each check is
    resolved from a file (or from a verdict passed in explicitly), so a claim written in a document
    can never make this return READY.
    """
    _, _, persistence, identity_mod, _ = load_helpers()
    out_dir = Path(out_dir)
    ident_record = persistence.VerifiedReader(out_dir).latest('RUN_IDENTITY')
    if ident_record is None:
        raise SystemExit('the run directory has no RUN_IDENTITY record')
    extra = {'acceptance': acceptance, 'verifier': verifier,
             'synthetic_execution': synthetic_execution}
    readiness = identity_mod.compute_missing_items(out_dir, ident_record, extra=extra)
    match = identity_mod.identity_match_report(ident_record)
    fwd = persistence.VerifiedReader(out_dir).latest('FORWARD_TARGETS')
    # The full watch list WITH per-file SHA, recomputed live.  Reporting only a count would let a
    # reader believe the freeze covers something it does not; the paths and digests are the evidence
    # a signer actually needs, and the audit re-derives them independently.
    _watch_rec = persistence.VerifiedReader(out_dir).latest('IDENTITY_TARGETS') or {}
    current_watch = {p: {'frozen': want, 'current': sha256_file(p), 'ok': sha256_file(p) == want}
                     for p, want in (_watch_rec.get('paths') or {}).items()}
    report = {
        'schema': 'd3_l1_v118_readiness',
        'run_id': run_id,
        'run_dir': str(out_dir),
        'identity_match': match,
        'targets_ok': bool(fwd and fwd.get('ok')),
        'targets_sha256': (fwd or {}).get('targets_sha256'),
        'n_target_rows': (fwd or {}).get('n_rows'),
        'budgets': ident_record['budgets'],
        'call_schedule': ident_record['call_schedule'],
        'dataset_identity': ident_record['identities']['dev_check_npz']['sha256'],
        'contract': ident_record['contracts']['effective_contract_hash'],
        'dependency_identity_sha256': ident_record['dependency_identity_sha256'],
        'state_binding_sha256': ident_record['state_binding_sha256'],
        'runtime_watch': {
            'basis': 'derived execution-path closure '
                     '(scripts/data/d3_l1_v121_dependency_closure.py)',
            'n_paths': len(current_watch),
            'covers_closure': (readiness.get('checks') or {}).get(
                'runtime_watch_covers_closure'),
            'closure_diff': (readiness.get('checks') or {}).get('runtime_watch_closure_diff'),
            'paths': current_watch,
        },
        'readiness': readiness,
        'verdict': readiness['verdict'],
        'no_execution_authorised_here': True,
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
    }
    persistence.publish_new(out_dir, 'READINESS', report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze-identity', action='store_true')
    ap.add_argument('--supervise', action='store_true')
    ap.add_argument('--worker', action='store_true',
                    help='the real F worker.  Requires an execution context published by the '
                         'supervisor after a human-approved credential was consumed; invoked '
                         'directly it refuses BEFORE loading anything')
    ap.add_argument('--guard-probe', action='store_true',
                    help='run ONLY the execution-context guard and report its verdict, loading '
                         'nothing: this is how a supervised positive path is proven without '
                         'touching the model')
    ap.add_argument('--readiness', action='store_true')
    ap.add_argument('--out-dir')
    ap.add_argument('--run-id')
    ap.add_argument('--authorization')
    ap.add_argument('--preregistration',
                    help='the pre-registration file whose sha256 the credential binds')
    ap.add_argument('--consumption-root',
                    help='directory for one-shot consumption receipts (defaults outside the run '
                         'directory so the fact of use survives the run)')
    ap.add_argument('--authorization-record')
    ap.add_argument('--write-authorization-template',
                    help='write a credential skeleton (HUMAN FIELDS LEFT EMPTY) and exit')
    ap.add_argument('--now-utc')
    # NOTE: `--synthetic-command` is declared with argparse.REMAINDER, so it MUST be the last
    # option on the command line; anything after it belongs to the substituted process.  Putting
    # another flag after it silently hands that flag to the worker instead of the supervisor.
    ap.add_argument('--synthetic-slot-cap', type=float)
    ap.add_argument('--synthetic-command', nargs=argparse.REMAINDER,
                    help='SYNTHETIC ACCEPTANCE ONLY: substitute the worker process '
                         '(must be the LAST option)')
    args = ap.parse_args(argv)
    if not args.out_dir:
        raise SystemExit('--out-dir is required')
    if args.readiness:
        run_id = args.run_id or (load_frozen(args.out_dir)['identity']['run_id'])
        report = readiness_report(args.out_dir, run_id=run_id)
        print(json.dumps({'run_id': report['run_id'], 'verdict': report['verdict'],
                          'missing_before_signature': report['readiness']
                          ['missing_before_signature'],
                          'checks': report['readiness']['checks'],
                          'targets_sha256': report['targets_sha256']}, indent=1, default=repr))
        return 0 if report['verdict'] == 'READY_FOR_USER_SIGNATURE' else 1
    if args.freeze_identity:
        run_id = args.run_id
        if not run_id:
            raise SystemExit('--freeze-identity requires --run-id')
        report = freeze_identity(args.out_dir, run_id,
                                 now_utc=args.now_utc or time.strftime(
                                     '%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        print(json.dumps({'run_id': report['run_id'], 'targets_ok': report['targets_ok'],
                          'identity_match_ok': report['identity_match_ok'],
                          'targets_sha256': report['targets_sha256'],
                          'n_target_rows': report['n_target_rows'],
                          'budgets': report['budgets']}, indent=1, default=repr))
        return 0 if report['targets_ok'] and report['identity_match_ok'] else 1
    if args.write_authorization_template:
        import d3_l1_v118_authorization as authorization_mod
        if not args.preregistration:
            raise SystemExit('--write-authorization-template requires --preregistration')
        frozen_t = load_frozen(args.out_dir)
        template = authorization_mod.build_template(
            frozen_t['identity'], frozen_t['forward'],
            preregistration_sha256=sha256_file(args.preregistration), output_dir=args.out_dir)
        sealed = authorization_mod.seal_template(template)
        dest = Path(args.write_authorization_template)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(sealed, indent=2, sort_keys=True, default=repr),
                        encoding='utf-8')
        print(json.dumps({
            'template': str(dest), 'run_id': sealed['run_id'],
            'credential_sha256': sealed['credential_sha256'],
            'human_fields_left_empty': [f for f in authorization_mod.HUMAN_FIELDS
                                        if not sealed.get(f)],
            'next': 'a HUMAN must fill approved_by / approved_utc / approval_statement and then '
                    're-seal with d3_l1_v118_authorization.content_digest, because an approval '
                    'cannot be generated by the process that consumes it',
            'this_template_alone_is_not_an_authorization': True}, indent=1, default=repr))
        return 1 if any(not sealed.get(f) for f in authorization_mod.HUMAN_FIELDS) else 0
    if args.guard_probe:
        import d3_l1_v120_execution_context as execctx
        try:
            context = execctx.validate_execution_context(run_dir=Path(args.out_dir))
        except execctx.ExecutionContextRefused as exc:
            print(json.dumps({'stage': 'execution_context_guard', 'guard': 'REFUSED',
                              'reason': str(exc), 'argv': sys.argv}, indent=1, default=repr))
            return 1
        print(json.dumps({'stage': 'execution_context_guard', 'guard': 'PASSED',
                          'run_id': context.get('run_id'), 'nonce': context.get('nonce'),
                          'supervisor_pid': context.get('supervisor_pid'),
                          'consumption_path': context.get('consumption_path'),
                          'credential_content_sha256': context.get(
                              'credential_content_sha256'),
                          'loaded_anything': False,
                          'note': 'the guard passed and NOTHING was loaded: no JAX import, no '
                                  'checkpoint read, no scorer call'},
                         indent=1, default=repr))
        return 0
    if args.worker:
        return worker(args.out_dir)
    if args.supervise:
        override = ({'slot_span_s': float(args.synthetic_slot_cap)}
                    if args.synthetic_slot_cap else None)
        return supervise(args.out_dir, authorization=args.authorization,
                         dry_command=args.synthetic_command or None, caps_override=override,
                         preregistration=args.preregistration,
                         consumption_root=args.consumption_root,
                         authorization_record=args.authorization_record)
    raise SystemExit('HOLD: this module executes nothing unless --freeze-identity, --supervise, '
                     '--worker, --readiness or --write-authorization-template is passed '
                     'explicitly')


if __name__ == '__main__':
    sys.exit(main())
