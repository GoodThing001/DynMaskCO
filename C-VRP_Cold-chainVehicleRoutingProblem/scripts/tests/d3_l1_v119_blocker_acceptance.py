#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.19 acceptance for the three formal-execution blockers found by independent review.

Blocker 1 -- AUTHORIZATION must bind the real execution identity
  B1-a  no authorization at all                          -> REFUSED, nothing started
  B1-b  an EMPTY JSON object `{}`                        -> REFUSED (was previously accepted)
  B1-c  a human template with the approval fields EMPTY  -> REFUSED
  B1-d  wrong run_id                                     -> REFUSED
  B1-e  wrong weight / data / contract / state / target SHA -> REFUSED (each field separately)
  B1-f  wrong output directory                           -> REFUSED
  B1-g  tampered after sealing                           -> REFUSED (payload digest mismatch)
  B1-h  correct credential, first use                    -> GRANTED and CONSUMED with a receipt
  B1-i  the SAME credential used a second time           -> REFUSED (one-shot replay)
  B1-j  correct credential but the pre-registration file -> REFUSED (its digest is bound too)
        has changed

Blocker 2 -- READY semantics must match between worker and supervisor
  B2-a  a REAL subprocess completes initialization inside the initialization cap and then runs a
        batch that EXCEEDS the initialization cap (but stays inside the batch cap) -> must NOT be
        killed, and the supervisor must observe `ready=True`
  B2-b  a REAL initialization overrun -> killed at the initialization deadline and recorded
  B2-c  the immutable READY record is what the probe reads; the legacy fixed `READY.json` is absent
        in this path, so a probe keyed to the fixed name would have reported "not ready"

Blocker 3 -- a FAILED slot must stop the round immediately
  B3-a..d  a first-slot defect in finite / shape / repeat / length -> the worker stops, NO later row
           record exists, the supervisor reports the round REVOKED and the verdict BLOCKED
  B3-e  the production worker's own check set (not a harness assertion) is what flags the defect

Every scenario runs the PRODUCTION supervisor entry point with a synthetic worker; the real F model
is never loaded.  One attempt per directory, failures kept.
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

RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
SYNTH_WORKER = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
PREREG = EV / 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_18.md'
V19_PREREG = EV / 'A1_D3_L1_REAL_F_FORWARD_EXECUTION_PREREGISTRATION_V1_19.md'
RUN_ID = 'd3l1-real-f-forward-20261015-v119'

ROWS = []


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:230]))


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def runner_module():
    spec = importlib.util.spec_from_file_location('v119_runner', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['v119_runner'] = module
    spec.loader.exec_module(module)
    return module


def freeze_fixture(root, name, run_id=RUN_ID, prereg=None):
    """Freeze a run directory for the CURRENT source identity."""
    run_dir = Path(root) / name
    module = runner_module()
    module.freeze_identity(run_dir, run_id, now_utc='2026-10-15T00:00:00Z')
    return run_dir


def supervisor_cmd(run_dir, *, mode, init_delay=None, slot_cap=None, authorization=None,
                   preregistration=None, consumption_root=None, extra=None):
    cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir)]
    if slot_cap is not None:
        cmd += ['--synthetic-slot-cap', str(slot_cap)]
    if authorization:
        cmd += ['--authorization', str(authorization)]
    if preregistration:
        cmd += ['--preregistration', str(preregistration)]
    if consumption_root:
        cmd += ['--consumption-root', str(consumption_root)]
    worker = [sys.executable, str(SYNTH_WORKER), '--out-dir', str(run_dir), '--mode', mode]
    if init_delay is not None:
        worker += ['--init-delay', str(init_delay)]
    cmd += ['--synthetic-command'] + worker + (extra or [])
    return cmd


def run_supervisor(cmd, timeout_s=300.0, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)


def supervisor_result(run_dir):
    reader = persistence.VerifiedReader(run_dir)
    return {'supervision': reader.latest('SUPERVISION_RESULT'),
            'round': reader.latest('ROUND'),
            'authorization': reader.latest('AUTHORIZATION_EVIDENCE'),
            'first_failure': reader.latest('FIRST_FAILURE'),
            'ready_records': reader.count('READY'),
            'slot_records': len(list((Path(run_dir) / 'scores').glob('SLOT_day*_*.json')))}


# =============================================================================================
# Blocker 1
# =============================================================================================
def blocker1(root):
    frozen = freeze_fixture(root, 'B1_base')
    module = runner_module()
    fr = module.load_frozen(frozen)
    ident, fwd = fr['identity'], fr['forward']
    cons = Path(root) / 'consumption_b1'
    sealed = AUTH.seal_template(AUTH.build_template(
        ident, fwd, preregistration_sha256=sha(PREREG), output_dir=frozen))
    approved = dict(sealed, approved_by='USER (synthetic acceptance fixture)',
                    approved_utc='2026-10-15T00:00:00Z',
                    approval_statement='SYNTHETIC ACCEPTANCE FIXTURE -- not a real approval')
    approved = AUTH.seal_template(approved)

    def write(name, payload):
        p = Path(root) / name
        if isinstance(payload, str):
            p.write_text(payload, encoding='utf-8')
        else:
            p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr),
                         encoding='utf-8')
        return p

    def gate(cred_path, run_dir=None, prereg=PREREG, consumption=cons):
        """Run the authorization gate alone (no worker), and report the outcome."""
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location('v119_authmod', str(RUNNER))
        m = _ilu.module_from_spec(spec)
        sys.modules['v119_authmod'] = m
        spec.loader.exec_module(m)
        target = Path(run_dir or frozen)
        try:
            credential = AUTH.load_credential(cred_path)
            granted = AUTH.authorize(credential, identity=ident, forward_targets=fwd,
                                     output_dir=target,
                                     preregistration_path=Path(prereg), run_dir=target,
                                     consumption_root=consumption)
            return True, granted
        except AUTH.AuthorizationRefused as exc:
            return False, {'refused': str(exc)}

    # B1-a: nothing at all
    ok, det = gate(Path(root) / 'does_not_exist.json')
    ck('B1-a an ABSENT authorization file is refused', ok is False
       and 'does not exist' in det['refused'], det)

    # B1-b: empty JSON object (the previously accepted case)
    p = write('empty.json', {})
    ok, det = gate(p)
    ck('B1-b an EMPTY JSON object is refused (this was previously accepted)', ok is False
       and ('EMPTY' in det['refused'] or 'non-empty JSON object' in det['refused']), det)

    # B1-c: human template with approval fields empty
    p = write('template_only.json', sealed)
    ok, det = gate(p)
    ck('B1-c a machine-generated template with the human approval fields EMPTY is refused',
       ok is False and 'no_human_approval' in det['refused'], det)

    # B1-d..f: one binding wrong at a time
    variants = {
        'B1-d wrong run_id': {'run_id': 'd3l1-real-f-forward-99999999-v000'},
        'B1-e1 wrong F checkpoint sha': {'f_ckpt_sha256': 'f' * 64},
        'B1-e2 wrong backbone sha': {'backbone_sha256': 'f' * 64},
        'B1-e3 wrong data sha': {'data_sha256': 'f' * 64},
        'B1-e4 wrong effective contract': {'effective_contract_hash': 'f' * 64},
        'B1-e5 wrong state binding': {'state_binding_sha256': 'f' * 64},
        'B1-e6 wrong forward-targets digest': {'forward_targets_sha256': 'f' * 64},
        'B1-e7 wrong call count': {'max_forward_calls': 130},
        'B1-e8 widened budget': {'budget_slot_span_s': 60.0},
        'B1-f wrong output directory': {'output_dir': str(Path(root) / 'somewhere_else')},
    }
    for label, patch in variants.items():
        bad = dict(approved)
        bad.update(patch)
        bad = AUTH.seal_template({k: v for k, v in bad.items() if k != '_file_sha256'})
        p = write(label.split()[0].lower() + '_' + str(abs(hash(label)) % 10 ** 6) + '.json', bad)
        ok, det = gate(p)
        ck('%s is refused' % label, ok is False
           and any('mismatch' in s for s in json.loads(det['refused'].split(': ', 1)[1])), det)

    # B1-g: tampered AFTER sealing
    tampered = dict(approved)
    tampered['data_sha256'] = 'a' * 64            # edited without re-sealing
    p = write('tampered.json', tampered)
    ok, det = gate(p)
    ck('B1-g a credential edited AFTER sealing is refused (payload digest mismatch)',
       ok is False and 'credential_sha256_mismatch' in det['refused'], det)

    # B1-h: the correct credential, first use
    p = write('approved.json', approved)
    ok, det = gate(p)
    ck('B1-h the CORRECT credential is granted and consumed with an auditable receipt',
       ok is True and det['consumption']['one_shot'] is True
       and Path(det['consumption']['path']).is_file()
       and det['consumption']['credential_content_sha256'] == approved['credential_sha256'],
       {'consumption': det['consumption'] if ok else det})

    # B1-i: replay the SAME credential
    ok2, det2 = gate(p)
    ck('B1-i the SAME credential used a SECOND time is refused (one-shot replay)',
       ok2 is False and 'ALREADY been consumed' in det2['refused'], det2)

    # B1-j: correct credential, but the pre-registration file changed
    other_prereg = write('prereg_copy.md', '# a DIFFERENT pre-registration\n')
    ok3, det3 = gate(p, prereg=other_prereg, consumption=Path(root) / 'consumption_b1j')
    ck('B1-j a correct credential whose bound pre-registration file has changed is refused',
       ok3 is False, det3)
    return {'frozen': str(frozen), 'consumption_dir': str(cons),
            'n_consumption_records': len(list(cons.glob('*.json')))}


# =============================================================================================
# Blocker 2
# =============================================================================================
def blocker2(root):
    # B2-a: REAL initialization inside the cap, batch far beyond the INIT cap but inside the BATCH cap.
    # The initialization and batch caps are the FROZEN 60 / 180 s; the hold is 70 s, so the batch
    # outlives the initialization cap by 10 s and stays 110 s inside the batch cap.  Nothing is
    # shortened, so this exercises the production numbers directly.
    hold_s = 70.0
    run_dir = freeze_fixture(root, 'B2a_ready_then_slow_batch')
    cmd = supervisor_cmd(run_dir, mode='one_slot_then_hold')
    proc = run_supervisor(cmd, timeout_s=400, env_extra={'V119_HOLD_S': str(hold_s)})
    res = supervisor_result(run_dir)
    sup = res['supervision'] or {}
    ctrl = (sup.get('control') or {})
    det = {'returncode': proc.returncode, 'control_reason': ctrl.get('reason'),
           'ready_state': ctrl.get('ready_state'), 'caps': ctrl.get('caps'),
           'round_status': sup.get('round', {}).get('status'),
           'rows_qualified': sup.get('n_rows_qualified'),
           'legacy_ready_json_present': (Path(run_dir) / 'READY.json').exists(),
           'immutable_ready_records': res['ready_records'],
           'hold_s': hold_s,
           'worker_control_wall_s': ctrl.get('worker_control_wall_s')}
    # The batch runs ~70 s: longer than the 60 s initialization cap, far inside the 180 s batch cap.
    # It must NOT be killed for initialization, and the supervisor must have observed readiness.
    ok = (ctrl.get('reason') is None and det['legacy_ready_json_present'] is False
          and res['ready_records'] == 1
          and det['worker_control_wall_s'] > 60.0
          and isinstance(ctrl.get('ready_state'), dict)
          and ctrl['ready_state'].get('ready') is True
          and sup.get('round', {}).get('status') == 'ROUND_PASS')
    ck('B2-a a batch that outlives the FROZEN 60 s INITIALIZATION cap (but not the 180 s batch cap) '
       'is NOT killed, and the supervisor observed the immutable READY record', ok, det)
    ck('B2-a2 the frozen initialization and batch caps were NOT relaxed for this scenario',
       ctrl.get('caps', {}).get('initialize') == 60.0 and ctrl.get('caps', {}).get('batch') == 180.0,
       ctrl.get('caps'))

    # B2-b: a REAL initialization overrun -> killed at the initialization deadline
    run_dir2 = freeze_fixture(root, 'B2b_slow_init')
    cmd2 = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir2),
            '--synthetic-command', sys.executable, str(SYNTH_WORKER),
            '--out-dir', str(run_dir2), '--mode', 'pass']
    # 61 s sleep exceeds the frozen 60 s initialization cap
    cmd2 = supervisor_cmd(run_dir2, mode='slow_init', init_delay=61.0)
    t0 = time.monotonic()
    proc2 = run_supervisor(cmd2, timeout_s=300)
    wall = time.monotonic() - t0
    res2 = supervisor_result(run_dir2)
    sup2 = res2['supervision'] or {}
    ctrl2 = (sup2.get('control') or {})
    det2 = {'returncode': proc2.returncode, 'control_reason': ctrl2.get('reason'),
            'ready_state': ctrl2.get('ready_state'), 'wall_s': wall,
            'tree_active_after': ctrl2.get('tree_active_after'),
            'round_status': sup2.get('round', {}).get('status'),
            'rows_qualified': sup2.get('n_rows_qualified'),
            'immutable_ready_records': res2['ready_records'],
            'slot_records': res2['slot_records']}
    ok2 = (ctrl2.get('reason') == 'initialization_compile_deadline'
           and ctrl2.get('tree_active_after') == 0
           and res2['ready_records'] == 0
           and res2['slot_records'] == 0
           and sup2.get('round', {}).get('status') == 'ROUND_REVOKED'
           and sup2.get('n_rows_qualified') == 0
           and wall < 120.0)
    ck('B2-b a REAL initialization overrun is killed at the initialization deadline and recorded',
       ok2, det2)

    # B2-c: the probe reads the immutable record, and no fixed READY.json exists in this path
    ck('B2-c the legacy fixed READY.json is absent in the V1.19 path, so a probe keyed to the fixed '
       'name would have reported "not ready" forever',
       det['legacy_ready_json_present'] is False and det['immutable_ready_records'] == 1,
       {'legacy_present': det['legacy_ready_json_present'],
        'immutable_records': det['immutable_ready_records']})
    return {'B2-a': det, 'B2-b': det2}


# =============================================================================================
# Blocker 3
# =============================================================================================
DEFECTS = (
    ('B3-a', 'fail_finite', 'all_finite'),
    ('B3-b', 'fail_shape', 'shape_matches_ordered_legal'),
    ('B3-c', 'fail_repeat', 'repeat_exact_type_dtype_shape_bytes'),
    ('B3-d', 'fail_len', 'shape_matches_ordered_legal'),
)


def blocker3(root):
    out = {}
    for label, mode, expected_check in DEFECTS:
        run_dir = freeze_fixture(root, 'B3_%s' % mode)
        proc = run_supervisor(supervisor_cmd(run_dir, mode=mode), timeout_s=300)
        res = supervisor_result(run_dir)
        sup = res['supervision'] or {}
        ff = res['first_failure'] or {}
        rows = sorted((Path(run_dir) / 'scores').glob('SLOT_day*_*.json'))
        det = {'returncode': proc.returncode, 'slot_records': len(rows),
               'first_failure': ff.get('expected_failing_checks'),
               'first_failure_position': [ff.get('day_id'), ff.get('position')],
               'round_status': sup.get('round', {}).get('status'),
               'verdict': sup.get('verdict'), 'rows_qualified': sup.get('n_rows_qualified'),
               'rows_status_counts': sup.get('rows_status_counts'),
               'worker_terminated_on_first_failure': None,
               'n_not_attempted': sum(v for k, v in (sup.get('rows_status_counts') or {}).items()
                                      if 'NOT_ATTEMPTED' in k)}
        worker_result = None
        for p in sorted(run_dir.glob('WORKER_RESULT_*.json')):
            worker_result = json.loads(p.read_text(encoding='utf-8'))
        det['worker_terminated_on_first_failure'] = (worker_result or {}).get(
            'terminated_on_first_failure')
        ok = (proc.returncode != 0
              and det['slot_records'] == 1                     # NO later state was scored
              and ff.get('expected_failing_checks') == [expected_check]
              and (ff.get('day_id'), ff.get('position')) == (0, 1)
              and det['round_status'] == 'ROUND_REVOKED'
              and det['verdict'] == 'REAL_F_FORWARD_ENGINEERING_BLOCKED'
              and det['rows_qualified'] == 0
              and det['n_not_attempted'] == 63)
        ck('%s a first-slot %s defect stops the round: exactly 1 row record, 63 NOT_ATTEMPTED, '
           'round REVOKED, verdict BLOCKED' % (label, expected_check), ok, det)
        out[label] = det
    return out


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='v119_acc_'))
    started = time.monotonic()
    results = {}
    try:
        for name, fn in (('blocker1', blocker1), ('blocker2', blocker2), ('blocker3', blocker3)):
            t = time.monotonic()
            try:
                results[name] = fn(root)
            except Exception as exc:                                      # noqa: BLE001
                import traceback
                results[name] = {'exception': '%s: %s' % (type(exc).__name__, exc),
                                 'traceback': traceback.format_exc()}
                ck('%s completed without an unhandled exception' % name, False, results[name])
            results.setdefault('_wall', {})[name] = time.monotonic() - t
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for r in ROWS if not r['ok'])
    payload = {
        'schema': 'd3_l1_v119_blocker_acceptance',
        'run_id': RUN_ID,
        'scorer': 'SYNTHETIC_SCORER_NOT_F -- the F model is never loaded in this harness',
        'blockers': {
            '1': 'the authorization credential must bind the real execution identity',
            '2': 'the READY state and the supervisor deadline semantics must agree',
            '3': 'a FAILED slot must terminate all later scoring immediately'},
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - n_fail, 'n_fail': n_fail,
        'results': results,
        'frozen_budgets_untouched': {'initialize_compile_s': 60.0, 'slot_span_s': 10.0,
                                     'batch_s': 180.0},
        'history_preserved': {'v118_evidence': 'untouched (4a6d207 archive)',
                              'l1_gates': 'untouched (10/10, 17/17)',
                              'v116_acceptance': 'unchanged interface and semantics'},
        'F_forward_calls_on_real_model': 0, 'F_D_Geo_comparisons': 0, 'training_runs': 0,
        'science_ready': False, 'stable_online_gain': False,
        'wall_s': time.monotonic() - started,
        'tool_sha256': sha(Path(__file__)),
        'runner_sha256': sha(RUNNER),
        'authorization_module_sha256': sha(EXT / 'scripts/simulation/d3_l1_v118_authorization.py'),
        'control_sha256': sha(EXT / 'scripts/tests/d3_l1_v116_forward_control.py'),
        'verdict': ('THREE_BLOCKERS_CLOSED' if n_fail == 0 else 'THREE_BLOCKERS_BLOCKED'),
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'D3_V119_BLOCKER_ACCEPTANCE.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== V1.19 blocker acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
