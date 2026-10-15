"""Fixed engineering matrix: real day-0 worker, exact negatives, identity/state machines.

Run once into a fresh directory. This is NOT the 16-day formal collection.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'scripts/simulation'))
import d3_l1_collection_runner as R
import d3_l1_p0c_v18 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args0 = ap.parse_args()
    root = Path(args0.outdir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    def ck(name, ok, detail=None):
        rows.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print(name, bool(ok), flush=True)
    args = SimpleNamespace(outdir=None, identity=R.DEFAULT_IDENTITY, dataset=R.DEFAULT_DATASET,
        runtime_identity=str(root / 'RUNTIME_IDENTITY_V18.json'), slot_cap_s=60.,
        day_rollout_cap_s=120., day_cap_s=120., batch_cap_s=3600., days=1, day_id=0,
        purpose='engineering', run_id='unused', expect_dataset_sha256=R.DEV_CHECK_SHA,
        inject_block_s=0., fault='none')
    V.write_json(R, args.runtime_identity, V.execution_identity(R, args))
    worker_paths = []
    # Each fault is run once against the SAME dev day and its first four reveal slots.
    for fault in ('none', 'type', 'dtype', 'signed_zero', 'nan_payload', 'vehicle_boundary',
                  'restore_error', 'write_error'):
        out = root / ('worker_' + fault)
        run = copy.copy(args)
        run.outdir, run.fault, run.run_id = str(out), fault, 'v18-worker-' + fault
        rc = V.supervise(R, run)
        result = json.loads((out / 'SUPERVISOR_RESULT.json').read_text())
        ledger = R.read_ledger(out / 'STATE_LEDGER.jsonl')
        first = R.slot_state(ledger, 0, 1)
        ck('real_worker:' + fault, (rc == 0 and first[0] == 'PUBLISHED'
            and result['slots_published'] == 4) if fault == 'none' else
           (rc != 0 and first[0] == 'FAILED' and result['slots_published'] == 3
            and result['verdict'] != 'BATCH_COMPLETE'),
           {'exit': rc, 'result': result, 'first_slot': first})
        audit_path = root / ('INDEPENDENT_WORKER_' + fault + '.json')
        cmd = [sys.executable, str(EXT / 'scripts/tests/d3_l1_v18_readonly_audit.py'),
               '--root', str(out), '--runtime-identity', args.runtime_identity, '--out', str(audit_path)]
        audit = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        (root / ('INDEPENDENT_' + fault + '.stdout')).write_text(audit.stdout)
        (root / ('INDEPENDENT_' + fault + '.stderr')).write_text(audit.stderr)
        ck('independent_artifacts:' + fault, audit.returncode == 0,
           {'command': cmd, 'exit': audit.returncode, 'report': str(audit_path)})
        worker_paths.append(out)
    # Strict comparator's public 16-case counterexample set, without its historical perturbation.
    sys.path.insert(0, str(EXT / 'scripts/tests'))
    from d3_l1_bitwise_parity import bitwise_equal
    import numpy as np
    import struct
    n0 = struct.unpack('>d', bytes.fromhex('7ff8000000000000'))[0]
    n1 = struct.unpack('>d', bytes.fromhex('7ff8000000000001'))[0]
    for name, a, b, expected in (
        ('np_python', np.float64(1), 1., 'DIFFERENT'),
        ('signed_zero', 0., -0., 'DIFFERENT'),
        ('nan_same_bits', n0, n0, 'EQUAL'), ('nan_payload', n0, n1, 'DIFFERENT'),
        ('array_dtype', np.array([1], dtype='f4'), np.array([1], dtype='f8'), 'DIFFERENT'),
        ('array_bits', np.array([0.]), np.array([-0.]), 'DIFFERENT'),
        ('budget_exhaustion', 0., 0., 'NOT_VERIFIABLE')):
        got = bitwise_equal(a, b, budget={'n': 0, 'cap': 0}) if name == 'budget_exhaustion' else bitwise_equal(a, b)
        ck('comparator:' + name, got[0] == expected, {'expected': expected, 'actual': got})
    # Identity failures are exercised via the real supervisor entry, before any child is spawned.
    identity = json.loads(Path(args.runtime_identity).read_text())
    for field in ('dataset_sha256', 'profile_file_sha256', 'profile_hash', 'base_contract_hash',
                  'effective_contract_hash', 'runner_sha256', 'historical_identity_sha256',
                  'teacher_manifest_sha256', 'modules', 'parameters', 'missing'):
        bad = copy.deepcopy(identity)
        if field == 'modules':
            bad['modules']['joint_fleet']['sha256'] = '0' * 64
        elif field == 'parameters':
            bad['parameters']['slot_cap_s'] = 61.
        else:
            bad[field] = '0' * 64
        p = root / ('bad_' + field + '.json')
        if field != 'missing':
            V.write_json(R, p, bad)
        run = copy.copy(args)
        run.runtime_identity = str(p)
        run.outdir, run.run_id = str(root / ('identity_' + field)), 'v18-id-' + field
        rc = V.supervise(R, run)
        result = json.loads((Path(run.outdir) / 'SUPERVISOR_RESULT.json').read_text())
        ck('identity_hard_stop:' + field, rc == 2 and result['verdict'] == 'HARD_STOP_IDENTITY'
           and result['workers_started'] == 0, result)
    # Import-origin mismatch and end-only source mismatch are both independently detectable.
    import joint_fleet
    origin = joint_fleet.__spec__.origin
    joint_fleet.__spec__.origin = str(root / 'wrong_origin.py')
    ck('identity:runtime_import_origin', not V.identity_gate(R, args)['ok'])
    joint_fleet.__spec__.origin = origin
    pinned = V.identity_gate(R, args)
    changed = copy.deepcopy(identity)
    changed['modules']['joint_fleet']['sha256'] = '0' * 64
    end_path = root / 'changed_at_end.json'
    V.write_json(R, end_path, changed)
    end_args = copy.copy(args)
    end_args.runtime_identity = str(end_path)
    ck('identity:end_gate_rejects_mismatch', pinned['ok'] and not V.identity_gate(R, end_args, end=True)['ok'])
    # Full source invariance of ordinary rollout, comparing actual PRE-fix retained objects.
    old_path = EXT.parent / ('research/review_3080_20261010/d3_g0_evidence_root_v1/'
        'a1_d3_l1_prereg_root_v1/v18_evidence/before/TYPED_OBJECTS.pkl')
    with open(old_path, 'rb') as fh:
        before = pickle.load(fh)
    pos = worker_paths[0] / 'validation/day00_pos1.pkl'
    with open(pos, 'rb') as fh:
        after = pickle.load(fh)
    ck('ordinary_rollout:unchanged_exactly', bitwise_equal(before['original_traces'], after['original_traces'])[0] == 'EQUAL'
       and bitwise_equal(before['original_served'], after['original_served'])[0] == 'EQUAL')
    # One-pass startup checks: even a fresh run_id cannot recap a started/failed day.
    positive = copy.copy(args)
    positive.outdir, positive.run_id = str(worker_paths[0]), 'v18-attempt-positive-again'
    rc = V.supervise(R, positive)
    ck('one_pass:new_run_no_recapture', all(row['status'] == 'SKIPPED_DAY_ALREADY_STARTED'
        for row in json.loads((worker_paths[0] / 'SUPERVISOR_RESULT.json').read_text())['days']))
    corrupt = root / 'corrupt'
    corrupt.mkdir()
    (corrupt / 'STATE_LEDGER.jsonl').write_text('{broken line}\n')
    run = copy.copy(args)
    run.outdir, run.run_id = str(corrupt), 'v18-corrupt'
    rc = V.supervise(R, run)
    ck('one_pass:corrupt_ledger_blocks_batch', rc != 0 and json.loads(
        (corrupt / 'SUPERVISOR_RESULT.json').read_text())['verdict'] == 'LEDGER_CORRUPT'
        and not list(corrupt.glob('WORKER_DAY*.stdout')))
    ck('slot_state:auxiliary_records_do_not_hide_attempt', R.slot_state([
        {'event': 'ATTEMPT_STARTED', 'day_id': 0, 'position': 1},
        {'event': 'WRITE_PROGRESS', 'day_id': 0, 'position': 1}], 0, 1)[0] == 'ATTEMPT_STARTED')
    # Scope controls: delayed pre-capture start, rollout-end separate from validation, final I/O.
    for scenario in ('long_validation', 'final_persistence'):
        run = copy.copy(args)
        run.purpose, run.days = 'synthetic', 1
        run.slot_cap_s, run.day_rollout_cap_s, run.batch_cap_s = 1.5, .6, 6.
        run.outdir, run.run_id = str(root / scenario), 'v18-' + scenario
        run.runtime_identity = str(root / (scenario + '_identity.json'))
        V.write_json(R, run.runtime_identity, V.execution_identity(R, run))
        def command(day, go):
            return [sys.executable, str(EXT / 'scripts/tests/d3_l1_v18_synthetic_worker.py'),
                    '--outdir', run.outdir, '--go', str(go), '--day', str(day), '--stage',
                    'long_validation' if scenario == 'long_validation' else 'success',
                    '--runtime-identity', run.runtime_identity]
        orig_write = R.atomic_write_bytes
        if scenario == 'final_persistence':
            def slow_write(path, data):
                if Path(path).name == 'SUPERVISOR_RESULT.json':
                    time.sleep(3.1)  # two real writes, exceeds the 6s test batch window
                return orig_write(path, data)
            R.atomic_write_bytes = slow_write
        try:
            rc = V.supervise(R, run, synthetic_command=command)
        finally:
            R.atomic_write_bytes = orig_write
        result = json.loads((Path(run.outdir) / 'SUPERVISOR_RESULT.json').read_text())
        ck('timing:' + scenario, (rc == 0 and result['verdict'] == 'BATCH_COMPLETE')
           if scenario == 'long_validation' else
           (rc != 0 and result['verdict'] == 'batch_deadline_after_persistence'), result)
    for scenario in ('identity_drift', 'supervisor_write_error'):
        run = copy.copy(args)
        run.purpose, run.days = 'synthetic', 1
        run.outdir, run.run_id = str(root / scenario), 'v18-' + scenario
        run.runtime_identity = str(root / (scenario + '_identity.json'))
        V.write_json(R, run.runtime_identity, V.execution_identity(R, run))
        def command(day, go):
            return [sys.executable, str(EXT / 'scripts/tests/d3_l1_v18_synthetic_worker.py'),
                    '--outdir', run.outdir, '--go', str(go), '--day', str(day), '--stage',
                    'identity_drift' if scenario == 'identity_drift' else 'worker_error',
                    '--runtime-identity', run.runtime_identity]
        real_write, real_terminate = R.atomic_write_bytes, V.ProcessTree.terminate
        cleanups = []
        if scenario == 'supervisor_write_error':
            def broken_write(path, data):
                if Path(path).name == 'WORKER_DAY00.json':
                    raise OSError('injected supervisor result disk failure')
                return real_write(path, data)
            def traced_terminate(tree):
                rec = real_terminate(tree)
                cleanups.append(rec)
                return rec
            R.atomic_write_bytes = broken_write
            V.ProcessTree.terminate = traced_terminate
        try:
            rc = V.supervise(R, run, synthetic_command=command)
        finally:
            R.atomic_write_bytes, V.ProcessTree.terminate = real_write, real_terminate
        result = json.loads((Path(run.outdir) / 'SUPERVISOR_RESULT.json').read_text())
        ck('supervisor:' + scenario, rc != 0 and (result['verdict'] == 'HARD_STOP_IDENTITY'
           if scenario == 'identity_drift' else result['verdict'] == 'SUPERVISOR_PERSISTENCE_FAILED'
           and cleanups and all(r['cleanup_ok'] for r in cleanups)),
           {'result': result, 'cleanups': cleanups})
    result = {'rows': rows, 'n_pass': sum(r['ok'] for r in rows),
              'n_fail': sum(not r['ok'] for r in rows), 'argv': sys.argv,
              'classification': 'real_dev_day0_workers_plus_synthetic_counterexamples',
              'formal_slots_collected': 0, 'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    V.write_json(R, root / 'ACCEPTANCE.json', result)
    return int(result['n_fail'] != 0)


if __name__ == '__main__':
    sys.exit(main())
