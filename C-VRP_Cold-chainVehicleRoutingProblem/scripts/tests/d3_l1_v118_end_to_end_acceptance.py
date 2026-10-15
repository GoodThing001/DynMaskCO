#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 end-to-end acceptance: drive the REAL worker / supervisor path with a SYNTHETIC scorer.

Requirement being closed
------------------------
"使用合成 scorer 驱动真实 worker/监督器路径，完成 Windows 新目标名持久化、PermissionError、超时、
强杀、尾部身份异常、行数不足及整轮撤销的端到端验收。"

Every scenario below runs the production code: `d3_l1_v118_forward_runner.supervise`, the V1.16
contained supervisor (`run_contained` / `reconcile_rows` / `round_verdict`), the V1.18 verified
new-destination publisher and the V1.18 target verifier.  Only the F model is replaced, because the
real F forward is the single item awaiting signature.  Each scenario gets its OWN frozen run
directory and is attempted exactly ONCE -- a failure is kept as a failure and never re-run.

Scenario matrix (fixed in advance)
----------------------------------
  S1  pass (readers held open)  64 verified rows, readers holding every record open -> ROUND_PASS
  S2  transient PermissionError the production reader meets a temporary sharing denial -> recovers
  S3  persistent read denial    a corrupt record -> HARD error, never silently skipped
  S4  deadline kill             a blocked slot is killed by the real supervisor deadline
  S5  forced kill               the worker is terminated while a slot is open (real job kill)
  S6  tail identity mutation    the trailing identity disagrees with the frozen watched files
  S7  tail record absent        the trailing identity record never appears
  S8  row count short           63 of 64 rows
  S9  bad score record          a row whose score length contradicts the ordered legal length
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_persistence as persistence                              # noqa: E402
import d3_l1_v118_targets as targets                                      # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
SYNTH_WORKER = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
RUN_ID = 'd3l1-real-f-forward-20261015-v118'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(run_dir, run_id=RUN_ID, *, watched_extra=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location('v118_runner_acc', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['v118_runner_acc'] = module
    spec.loader.exec_module(module)
    return module.freeze_identity(run_dir, run_id, now_utc='2026-10-15T00:00:00Z',
                                  watched_extra=watched_extra)


def _runner_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('v118_runner_sup', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['v118_runner_sup'] = module
    spec.loader.exec_module(module)
    return module


def run_supervise(run_dir, *, mode, slot_cap=None, watched=None, extra_env=None, timeout_s=180.0):
    """Drive the PRODUCTION supervisor entry point in a child process.

    The child is `python d3_l1_v118_forward_runner.py --supervise --synthetic-command ...`, i.e. the
    same CLI a real run uses, with only the worker process substituted for the synthetic scorer.
    """
    cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir)]
    if slot_cap is not None:
        cmd += ['--synthetic-slot-cap', str(slot_cap)]
    cmd += ['--synthetic-command', sys.executable, str(SYNTH_WORKER),
            '--out-dir', str(run_dir), '--mode', mode]
    env = dict(os.environ)
    if watched:
        env['V118_WATCHED'] = str(watched)
    if extra_env:
        env.update(extra_env)
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    return proc


def run_supervise_inproc(run_dir, *, mode, slot_cap=None, watched=None):
    """Same production `supervise()` entry point, called in-process (used where the harness itself
    must control the worker's lifetime).  The supervisor command is built the same way."""
    module = _runner_module()
    command = [sys.executable, str(SYNTH_WORKER), '--out-dir', str(run_dir), '--mode', mode]
    env = dict(os.environ)
    if watched:
        env['V118_WATCHED'] = str(watched)
    code = module.supervise(run_dir, authorization=None, dry_command=command,
                            caps_override=({'slot_span_s': float(slot_cap)}
                                           if slot_cap else None))
    return code


def read_result(run_dir):
    reader = persistence.VerifiedReader(run_dir)
    return {'supervision': reader.latest('SUPERVISION_RESULT'),
            'round': reader.latest('ROUND'),
            'round_verdict': json.loads((Path(run_dir) / 'ROUND_VERDICT.json').read_text())
            if (Path(run_dir) / 'ROUND_VERDICT.json').is_file() else None,
            'control': reader.latest('CONTROL_RESULT')}


def scenario_pass(root):
    """S1: 64 verified rows while readers hold every published record open."""
    run_dir = root / 'S1_pass'
    freeze(run_dir)
    held = []
    try:
        for p in sorted(run_dir.rglob('*.json')):
            held.append(open(p, 'rb'))
        proc = run_supervise(run_dir, mode='pass')
        res = read_result(run_dir)
        tail = persistence.VerifiedReader(run_dir).latest('CONTROL_IDENTITY_END')
        return (res['supervision'] is not None
                and res['supervision']['round']['status'] == 'ROUND_PASS'
                and res['supervision']['n_rows_qualified'] == 64
                and res['supervision']['verdict'] == 'REAL_F_FORWARD_ENGINEERING_PASS'
                and not res['supervision']['identity_payload_problems']
                and tail is not None and tail.get('ok') is True
                and res['round_verdict']['status'] == 'ROUND_PASS'
                and proc.returncode == 0), {
            'returncode': proc.returncode,
            'round_status': res['supervision']['round']['status'] if res['supervision'] else None,
            'n_rows_qualified': res['supervision']['n_rows_qualified'] if res['supervision'] else None,
            'tail_ok': (tail or {}).get('ok'),
            'held_readers': len(held),
            'tail_publication_verified': bool(
                (res['supervision'] or {}).get('control', {}).get('tail_publication', {})
                .get('verified')),
            'new_destination_names': sorted(
                {p.name.split('_')[0] for p in run_dir.rglob('*.json')})[:12],
            'stderr_tail': proc.stderr[-400:]}
    finally:
        for fh in held:
            fh.close()


def scenario_transient_permission(_root):
    """S2: the production reader survives a temporary sharing denial and still reads the record."""
    d = Path(tempfile.mkdtemp(prefix='v118_perm_'))
    try:
        pub = persistence.publish_new(d, 'REC', {'value': 1})
        import unittest.mock as mock
        real = persistence.Path.read_bytes
        state = {'n': 0}

        def flaky(self):
            if self.name == Path(pub['path']).name and state['n'] < 3:
                state['n'] += 1
                raise PermissionError('injected transient sharing denial')
            return real(self)

        with mock.patch.object(persistence.Path, 'read_bytes', flaky):
            got = persistence.VerifiedReader(d, wait_s=0.5).latest('REC')
        return (got is not None and got.get('value') == 1 and state['n'] == 3), {
            'injected_denials': state['n'], 'recovered': got is not None,
            'read_diagnostics': persistence.READ_DIAGNOSTICS}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def scenario_persistent_denial(_root):
    """S3: a corrupt record must be a HARD error, never silently treated as 'no record yet'."""
    d = Path(tempfile.mkdtemp(prefix='v118_corrupt_'))
    try:
        persistence.publish_new(d, 'REC', {'value': 1})
        (d / ('REC_%s.json' % ('f' * 32))).write_bytes(b'{ not json at all')
        raised = None
        try:
            persistence.VerifiedReader(d, wait_s=0.05).latest('REC')
        except Exception as exc:                                          # noqa: BLE001
            raised = '%s: %s' % (type(exc).__name__, exc)
        # and through the real supervisor path
        run_dir = _root / 'S3_corrupt'
        freeze(run_dir)
        proc = run_supervise(run_dir, mode='corrupt_phase', timeout_s=90)
        res = read_result(run_dir)
        ctrl = (res['supervision'] or {}).get('control', {})
        return (raised is not None and 'TransientReadError' in raised
                and (res['supervision'] or {}).get('round', {}).get('status')
                in ('ROUND_REVOKED', None)
                and ctrl.get('ok') is False), {
            'reader_error': raised, 'returncode': proc.returncode,
            'control_ok': ctrl.get('ok'), 'control_reason': ctrl.get('reason'),
            'round_status': (res['supervision'] or {}).get('round', {}).get('status')}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def scenario_deadline_kill(root):
    """S4: a slot that blocks past the (synthetic-shortened) slot cap is killed by the supervisor."""
    run_dir = root / 'S4_deadline'
    freeze(run_dir)
    proc = run_supervise(run_dir, mode='block_slot', slot_cap=3.0, timeout_s=120)
    res = read_result(run_dir)
    ctrl = (res['supervision'] or {}).get('control', {})
    round_status = (res['supervision'] or {}).get('round', {}).get('status')
    return (ctrl.get('reason') == 'slot_span_deadline'
            and ctrl.get('tree_active_after') == 0
            and round_status == 'ROUND_REVOKED'
            and (res['supervision'] or {}).get('n_rows_qualified') == 0), {
        'control_reason': ctrl.get('reason'), 'tree_active_after': ctrl.get('tree_active_after'),
        'cleanup': ctrl.get('cleanup'), 'round_status': round_status,
        'statuses': (res['supervision'] or {}).get('rows_status_counts'),
        'worker_control_wall_s': ctrl.get('worker_control_wall_s')}


def scenario_forced_kill(root):
    """S5: a real process-tree kill while a slot is open; the round must be revoked, not passed.

    The kill goes through the SAME `ProcessTree.terminate()` the production supervisor uses
    (TerminateJobObject on Windows), so this is a genuine forced kill of a contained process, not a
    simulated one.  The supervisor's own deadline is set far out (600 s) so that ONLY the external
    kill can end the worker.
    """
    run_dir = root / 'S5_forced_kill'
    freeze(run_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location('v118_runner_kill', str(RUNNER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['v118_runner_kill'] = mod
    spec.loader.exec_module(mod)
    _j, control, _persistence, _i, _t = mod.load_helpers()
    frozen = mod.load_frozen(run_dir)
    watched = {str(p): sha(p) for p in frozen['identity_targets']['paths']}
    command = [sys.executable, str(SYNTH_WORKER), '--out-dir', str(run_dir), '--mode', 'block_slot']
    holder = {}
    real_tree = control.ProcessTree

    class Wrapped(real_tree):
        def __init__(self, process):
            super(Wrapped, self).__init__(process)
            holder['tree'] = self

    def killer():
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
        if holder.get('tree') is not None:
            holder['kill_report'] = holder['tree'].terminate()
            holder['killed'] = True

    control.ProcessTree = Wrapped
    thread = threading.Thread(target=killer, daemon=True)
    try:
        thread.start()
        ctrl = control.run_contained(command, run_dir, watched, init_cap=600., slot_cap=600.,
                                     batch_cap=600., env=dict(os.environ))
    finally:
        control.ProcessTree = real_tree
    rows = control.reconcile_rows(run_dir, ctrl, verify_record=lambda r, o: (False, {}))
    rep, rows = control.round_verdict(run_dir, ctrl, rows, expected_rows=64)
    killed_flag = bool(holder.get('killed'))
    return (killed_flag and rep['status'] == 'ROUND_REVOKED'
            and sum(1 for r in rows if r.get('qualified')) == 0
            and ctrl.get('tree_active_after') == 0), {
        'killed': killed_flag, 'kill_report': holder.get('kill_report'),
        'round_status': rep['status'], 'control_reason': ctrl.get('reason'),
        'cleanup': ctrl.get('cleanup'), 'tree_active_after': ctrl.get('tree_active_after'),
        'statuses': sorted({r.get('status') for r in rows})}


def scenario_tail_mutation(root):
    """S6: the trailing identity record disagrees with the frozen watched files."""
    run_dir = root / 'S6_tail_mutation'
    watched = root / 'WATCHED_IDENTITY.txt'
    watched.write_text('frozen\n', encoding='utf-8')
    freeze(run_dir, watched_extra=[watched])
    proc = run_supervise(run_dir, mode='mutate_watched', watched=watched, timeout_s=120)
    res = read_result(run_dir)
    sup = res['supervision'] or {}
    return (sup.get('round', {}).get('status') == 'ROUND_REVOKED'
            and sup.get('n_rows_qualified') == 0
            and bool(sup.get('identity_payload_problems'))
            and any('WATCHED_IDENTITY' in p for p in sup.get('identity_payload_problems', []))
            and sup.get('verdict') == 'REAL_F_FORWARD_ENGINEERING_BLOCKED'), {
        'returncode': proc.returncode, 'round_status': sup.get('round', {}).get('status'),
        'reason': sup.get('round', {}).get('reason'),
        'identity_payload_problems': sup.get('identity_payload_problems'),
        'control_reason': (sup.get('control') or {}).get('reason'),
        'n_rows_qualified': sup.get('n_rows_qualified')}


def scenario_tail_absent(root):
    """S7: the trailing identity record never appears -> the whole round is revoked."""
    run_dir = root / 'S7_tail_absent'
    freeze(run_dir)
    proc = run_supervise(run_dir, mode='no_tail', timeout_s=120)
    res = read_result(run_dir)
    sup = res['supervision'] or {}
    return (sup.get('round', {}).get('status') == 'ROUND_REVOKED'
            and sup.get('n_rows_qualified') == 0
            and bool(sup.get('identity_payload_problems'))), {
        'returncode': proc.returncode, 'round_status': sup.get('round', {}).get('status'),
        'reason': sup.get('round', {}).get('reason'),
        'identity_payload_problems': sup.get('identity_payload_problems')}


def scenario_rows_short(root):
    """S8: 63 of the 64 expected rows -> ROUND_REVOKED with INCOMPLETE_ROWS."""
    run_dir = root / 'S8_rows_63'
    freeze(run_dir)
    proc = run_supervise(run_dir, mode='rows_63', timeout_s=120)
    res = read_result(run_dir)
    sup = res['supervision'] or {}
    return (sup.get('round', {}).get('status') == 'ROUND_REVOKED'
            and sup.get('round', {}).get('reason') == 'INCOMPLETE_ROWS'
            and sup.get('n_rows_qualified') == 0), {
        'returncode': proc.returncode, 'round_status': sup.get('round', {}).get('status'),
        'reason': sup.get('round', {}).get('reason'),
        'rows_status_counts': sup.get('rows_status_counts')}


def scenario_bad_record(root):
    """S9: a row whose score length contradicts the ordered-legal length is a verification FAIL."""
    run_dir = root / 'S9_bad_record'
    freeze(run_dir)
    proc = run_supervise(run_dir, mode='bad_record', timeout_s=120)
    res = read_result(run_dir)
    sup = res['supervision'] or {}
    counts = sup.get('rows_status_counts') or {}
    # round_verdict downgrades every row to REVOKED_<status>, so the defect is counted on the
    # prefixed name; exactly one row must be the failed one and the other 63 must be intact.
    failed = counts.get('REVOKED_FAILED_ROW_OR_PUBLICATION', 0)
    intact = counts.get('REVOKED_DURABLE_ROW_VERIFIED', 0)
    return (sup.get('round', {}).get('status') == 'ROUND_REVOKED'
            and failed == 1 and intact == 63
            and sup.get('n_rows_qualified') == 0), {
        'returncode': proc.returncode, 'round_status': sup.get('round', {}).get('status'),
        'rows_status_counts': counts, 'n_rows_qualified': sup.get('n_rows_qualified')}


SCENARIOS = (
    ('S1 64 verified rows with readers holding every record open -> ROUND_PASS', scenario_pass),
    ('S2 a transient Windows sharing denial is retried and the record is still read',
     scenario_transient_permission),
    ('S3 a corrupt record is a HARD bounded error on both the reader and the supervisor path',
     scenario_persistent_denial),
    ('S4 a blocked slot is killed by the real supervisor deadline -> round revoked',
     scenario_deadline_kill),
    ('S5 a real forced process-tree kill mid-slot -> round revoked', scenario_forced_kill),
    ('S6 a trailing identity record that disagrees with the frozen files -> round revoked',
     scenario_tail_mutation),
    ('S7 a missing trailing identity record -> round revoked', scenario_tail_absent),
    ('S8 63 of 64 rows -> ROUND_REVOKED (INCOMPLETE_ROWS)', scenario_rows_short),
    ('S9 a row whose score length contradicts the ordered legal length -> verification FAIL',
     scenario_bad_record),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--keep-run-dirs', action='store_true')
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='v118_acc_'))
    rows = []
    try:
        for name, fn in SCENARIOS:
            started = time.monotonic()
            try:
                ok, detail = fn(root)
            except Exception as exc:                                      # noqa: BLE001
                import traceback
                ok, detail = False, {'exception': '%s: %s' % (type(exc).__name__, exc),
                                     'traceback': traceback.format_exc()}
            rows.append({'name': name, 'ok': bool(ok), 'wall_s': time.monotonic() - started,
                         'detail': detail})
            print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                                     json.dumps(detail, default=repr, sort_keys=True)[:260]))
    finally:
        if not args.keep_run_dirs:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for r in rows if not r['ok'])
    payload = {
        'schema': 'd3_l1_v118_synthetic_end_to_end',
        'run_id': RUN_ID,
        'scorer': 'SYNTHETIC_SCORER_NOT_F -- the F model is deliberately not loaded here',
        'what_is_real': ['d3_l1_v118_forward_runner.supervise (the production supervisor entry)',
                         'V1.16 run_contained / reconcile_rows / round_verdict',
                         'V1.18 publish_new(verify=True) new-destination persistence',
                         'V1.18 verify_slot_record / verify_identity target verifier',
                         'the V1.14 phase journal'],
        'what_is_synthetic': ['the F model forward (replaced by a deterministic scorer)',
                              'the score values themselves'],
        'not_a_substitute_for': 'the real F 64-state forward, which remains the single item '
                                'awaiting the user signature',
        'rows': rows, 'n_rows': len(rows), 'n_pass': len(rows) - n_fail, 'n_fail': n_fail,
        'tool_sha256': sha(__file__),
        'runner_sha256': sha(RUNNER),
        'synthetic_worker_sha256': sha(SYNTH_WORKER),
        'persistence_sha256': sha(EXT / 'scripts/simulation/d3_l1_v118_persistence.py'),
        'targets_sha256': sha(EXT / 'scripts/simulation/d3_l1_v118_targets.py'),
        'read_diagnostics': persistence.READ_DIAGNOSTICS,
        'F_forward_calls_on_real_model': 0, 'F_D_Geo_comparisons': 0, 'training_runs': 0,
        'science_ready': False, 'stable_online_gain': False,
        'verdict': ('NEW_DESTINATION_PERSISTENCE_ACCEPTED' if n_fail == 0
                    else 'NEW_DESTINATION_PERSISTENCE_BLOCKED'),
    }
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'SYNTHETIC_END_TO_END.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== V1.18 synthetic end-to-end: %d rows, %d PASS, %d FAIL ==='
          % (len(rows), len(rows) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
