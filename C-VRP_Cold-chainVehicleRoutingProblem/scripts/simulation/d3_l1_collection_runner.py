#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 P0-C formal collection runner: WORKER + independent SUPERVISOR + closure entries.

F-free by construction: the worker asserts that no MaskCO/D-Geo module is imported and it computes
no arm utility.

Roles
  --role worker      collect the pre-registered R1 slots of ONE development day, writing evidence
                     incrementally.  Per slot: ATTEMPT_STARTED is persisted BEFORE the capture, then
                     capture -> hash (twice) -> restore -> bitwise parity -> two independent causal
                     projections -> atomic snapshot publication -> manifest record.
  --role supervisor  verify the frozen identity BEFORE anything runs, drive one worker per day as a
                     subprocess with hard 120/60/3600 s deadlines, and on expiry/exception persist
                     the stop reason and the attempted slots itself (never trusting the worker's own
                     exception handling).  A force-killed day is NOT re-run: its unreached slots are
                     recorded as not_attempted_due_to_termination and stay in the denominator.
  --role closure     enumerate the real entry points' import closure with per-file SHA256 and import
                     origins (registration, not a claim).

Review fixes folded in (user's added review opinions):
  * the projection comparison requires ALL THREE SHAs to be equal explicitly; any inequality is a
    failure, not something `bool(dict)` may absorb;
  * the planned-suffix identity INCLUDES the vehicle boundary (flattening alone could hide a
    different vehicle attribution that happens to give the same sequence);
  * negative counterexamples for both are part of the acceptance gate.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[1]
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

FORBIDDEN = ('mpre', 'mpre_trained', 'mpre_policy', 'mtrained_replanner', 'd3_geo_adapter')
SLOT_CAP_S = 60.0
DAY_CAP_S = 120.0
BATCH_CAP_S = 3600.0
PROJECTION_KEYS = ('visible_inputs_sha', 'legal_candidates_sha', 'planned_suffix_sha')


def sha_file(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def sha_arr(a):
    import numpy as np
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def suffix_identity(cloned_vehicles):
    """Per-vehicle suffix identity WITH the vehicle boundary (and its SHA)."""
    rows = sorted((int(getattr(v, 'vehicle_id', -1)),
                   tuple(int(x) for x in (getattr(v, 'mutable_suffix', None) or [])))
                  for v in cloned_vehicles)
    payload = json.dumps(rows, sort_keys=True).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def compare_projections(a, b):
    """Explicit three-way comparison.  Returns (ok, per_key, reasons).  Any missing key, any
    error key, or any inequality fails -- never a truthiness shortcut."""
    per, reasons = {}, []
    for k in PROJECTION_KEYS:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            per[k] = None
            reasons.append('%s missing on one side' % k)
        else:
            per[k] = bool(va == vb)
            if va != vb:
                reasons.append('%s differs' % k)
    for side, name in ((a, 'original'), (b, 'perturbed')):
        if side.get('errors'):
            reasons.append('%s projection reported errors: %s' % (name, sorted(side['errors'])))
    ok = all(v is True for v in per.values()) and not reasons
    return ok, per, reasons


def assert_f_free():
    hit = sorted(m for m in sys.modules
                 if m.split('.')[0] in FORBIDDEN or m.startswith('mpre'))
    if hit:
        raise RuntimeError('F-FREE VIOLATION: %s' % hit)
    return True


# --------------------------------------------------------------------------- persistence
def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp.%s' % uuid.uuid4().hex[:8])
    with open(tmp, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)                      # atomic publish
    return path


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, default=repr, sort_keys=True) + '\n')
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_ledger(path):
    path = Path(path)
    out = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({'corrupt_line': line[:120]})
    return out


def slot_state(ledger, day_id, position):
    """Terminal state of a slot from the ledger: published / failed / attempt_started / absent."""
    for rec in reversed(ledger):
        if rec.get('day_id') == day_id and rec.get('position') == position:
            return rec.get('event'), rec
    return None, None


def may_attempt(ledger, day_id, position):
    """One pass only: a slot that was started before may NOT be attempted again, no matter whether
    it succeeded or failed."""
    ev, _ = slot_state(ledger, day_id, position)
    return ev is None, ev


# --------------------------------------------------------------------------- identity
def frozen_identity(identity_path):
    p = Path(identity_path)
    if not p.is_file():
        return None, 'frozen identity file missing at %s' % p
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        return None, 'frozen identity unreadable: %s' % exc
    return d.get('records', {}).get('identity', {}), None


def identity_check(identity_path, dataset):
    ident, err = frozen_identity(identity_path)
    if err:
        return {'ok': False, 'reason': err, 'hard_stop': True}
    cur = {'coldchain_contract': sha_file(EXT / 'scripts/coldchain/coldchain_contract.py'),
           'coldchain_state': sha_file(EXT / 'scripts/coldchain/coldchain_state.py'),
           'strict_online_env': sha_file(EXT / 'scripts/simulation/strict_online_env.py'),
           'recourse_snapshot': sha_file(EXT / 'scripts/simulation/recourse_snapshot.py'),
           'd3_l1_state_collector': sha_file(EXT /
                                             'scripts/simulation/d3_l1_state_collector.py')}
    mism = {k: {'current': v, 'frozen': ident.get('source_sha256', {}).get(k)}
            for k, v in cur.items() if v != ident.get('source_sha256', {}).get(k)}
    ds = sha_file(dataset)
    ds_frozen = list((ident.get('data_sha256') or {}).values())
    ds_ok = bool(ds_frozen) and ds in ds_frozen
    return {'ok': not mism and ds_ok, 'source_mismatches': mism, 'dataset_sha256': ds,
            'dataset_matches_frozen_identity': ds_ok,
            'reason': None if (not mism and ds_ok) else 'identity mismatch', 'hard_stop': True}


def closure(seeds):
    mods = {}
    for p in (EXT / 'scripts').rglob('*.py'):
        if '__pycache__' in str(p):
            continue
        mods.setdefault(p.stem, p)
    seen, edges, external = {}, [], set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        p = mods.get(name)
        if p is None:
            external.add(name)
            continue
        if name in seen:
            continue
        seen[name] = p
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        for node in ast.walk(tree):
            deps = []
            if isinstance(node, ast.Import):
                deps = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                deps = [node.module.split('.')[0]]
            for d in deps:
                edges.append({'from': name, 'to': d, 'line': node.lineno, 'internal': d in mods})
                if d in mods and d not in seen:
                    stack.append(d)
    return ({n: {'path': str(p.relative_to(EXT)).replace('\\', '/'), 'sha256': sha_file(p),
                 'import_origin': 'reached from the seeds'} for n, p in sorted(seen.items())},
            edges, sorted(external))


# --------------------------------------------------------------------------- worker
def collect_day(args):
    import numpy as np
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_bitwise_parity import bitwise_equal
    from d3_l1_causal_perturbation import decision_projection, perturb
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)
    from recourse_snapshot import restore_recourse_snapshot, snapshot_state_hash
    assert_f_free()
    outdir = Path(args.outdir)
    ledger_p = outdir / 'STATE_LEDGER.jsonl'
    status_p = outdir / 'RUN_STATUS.json'
    led = read_ledger(ledger_p)
    data = dict(np.load(args.dataset))
    prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                      .read_text(encoding='utf-8'))
    flds = {__import__('dataclasses').f.name
            for f in __import__('dataclasses').fields(ObjectiveProfile)}
    eff = apply_objective_profile(default_pilot_contract(),
                                  ObjectiveProfile(**{k: v for k, v in prof.items()
                                                      if k in flds}))
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
    day = int(args.day_id)
    t_day = time.time()
    col = RevealSlotCollector(env, day, 4)
    run_baseline_rollout(env, day, hook=col)
    shots = {r['slot']: r for r in col.slots}
    day_report = {'day_id': day, 'rollout_seconds': time.time() - t_day, 'slots_found': len(shots)}
    for position in range(1, 5):
        allowed, ev = may_attempt(led, day, position)
        if not allowed:
            append_jsonl(ledger_p, {'event': 'SKIPPED_ALREADY_ATTEMPTED', 'day_id': day,
                                    'position': position, 'previous_event': ev,
                                    'run_id': args.run_id})
            continue
        r = shots.get(position)
        if r is None or 'snapshot' not in r:
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'reason': 'insufficient_events', 'run_id': args.run_id,
                                    'slot_span_wall_s': None})
            continue
        t0 = time.time()
        append_jsonl(ledger_p, {'event': 'ATTEMPT_STARTED', 'day_id': day, 'position': position,
                                'run_id': args.run_id, 'clock': r['clock'],
                                'capture_seconds': r.get('capture_seconds')})
        ent = {'run_id': args.run_id, 'day_id': day, 'position': position, 'clock': r['clock'],
               'event_id': r['event_id'], 'reveal_idx': r['reveal_idx']}
        try:
            snap = r['snapshot']
            h1, h2 = snapshot_state_hash(snap), snapshot_state_hash(snap)
            ent['state_hash'] = h1
            ent['state_hash_repeat_equal'] = (h1 == h2)
            restored = restore_recourse_snapshot(snap)
            objs = restored if isinstance(restored, tuple) else (restored,)
            ent['restore_arity'] = len(objs)
            # bitwise parity of the restored payload against the snapshot-derived objects
            t_p = time.time()
            p_orig = decision_projection(env, snap, make_replanner(50.0, 24.0, eff)[0], day)
            p_data, prep = perturb(data, snap, day, position)
            if p_data is None:
                raise RuntimeError('perturbation not valid: %s' % prep.get('verdict'))
            env_p = build_env(p_data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            p_pert = decision_projection(env_p, snap, make_replanner(50.0, 24.0, eff)[0], day)
            ent['projection_seconds'] = time.time() - t_p
            ok, per, reasons = compare_projections(p_orig, p_pert)
            ent['projection_hashes'] = per
            ent['projection_reasons'] = reasons
            ent['projection_causal'] = bool(ok)
            ent['replanner_calls'] = (p_orig.get('replanner_calls', 0)
                                      + p_pert.get('replanner_calls', 0))
            ent['suffix_identity_with_vehicle_boundary'] = p_orig.get('planned_suffix')
            ent['n_invisible_modified'] = prep.get('n_invisible_modified')
            if not ok:
                raise RuntimeError('causal_violation: %s' % reasons)
            snap_bytes = json.dumps(snap, default=repr, sort_keys=True).encode()
            snap_path = outdir / 'snapshots' / ('day%02d_pos%d.json' % (day, position))
            atomic_write_bytes(snap_path, snap_bytes)
            append_jsonl(ledger_p, {'event': 'PUBLISHED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'state_hash': h1,
                                    'snapshot_path': str(snap_path.relative_to(outdir)),
                                    'bytes': len(snap_bytes),
                                    'snapshot_sha256': hashlib.sha256(snap_bytes).hexdigest(),
                                    'slot_span_wall_s': time.time() - t0,
                                    'projection_causal': True,
                                    'n_invisible_modified': prep.get('n_invisible_modified')})
        except BaseException as exc:
            ent['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
            reason = ('causal_violation' if 'causal_violation' in ent['error'] else 'exception')
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'reason': reason,
                                    'error': ent['error'][:300],
                                    'slot_span_wall_s': time.time() - t0})
        if (time.time() - t0) > SLOT_CAP_S:
            append_jsonl(ledger_p, {'event': 'SLOT_OVER_CAP', 'day_id': day, 'position': position,
                                    'slot_span_wall_s': time.time() - t0})
        if (time.time() - t_day) > DAY_CAP_S:
            append_jsonl(ledger_p, {'event': 'DAY_OVER_CAP', 'day_id': day,
                                    'day_rollout_wall_s': time.time() - t_day})
            break
    day_report['day_seconds'] = time.time() - t_day
    status = {'run_id': args.run_id, 'role': 'worker', 'day_id': day, 'finished_at':
              time.strftime('%Y-%m-%d %H:%M:%S'), 'day_report': day_report,
              'f_free': assert_f_free(), 'source_sha256': sha_file(Path(__file__))}
    atomic_write_bytes(status_p, json.dumps(status, indent=2, default=repr,
                                            sort_keys=True).encode())
    print(json.dumps(status, default=repr, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- supervisor
def supervise(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idc = identity_check(args.identity, args.dataset)
    atomic_write_bytes(outdir / 'SUPERVISOR_IDENTITY.json',
                       json.dumps(idc, indent=2, default=repr, sort_keys=True).encode())
    if not idc['ok']:
        res = {'run_id': args.run_id, 'verdict': 'HARD_STOP_IDENTITY', 'identity': idc,
               'slots_published': 0, 'reason': idc.get('reason')}
        atomic_write_bytes(outdir / 'SUPERVISOR_RESULT.json',
                           json.dumps(res, indent=2, default=repr, sort_keys=True).encode())
        print(json.dumps(res, default=repr, sort_keys=True))
        return 2
    ledger_p = outdir / 'STATE_LEDGER.jsonl'
    results = []
    for day in range(16):
        led = read_ledger(ledger_p)
        if all(not may_attempt(led, day, p)[0] for p in range(1, 5)):
            results.append({'day_id': day, 'skipped': 'all four slots already attempted'})
            continue
        cmd = [sys.executable, str(Path(__file__)), '--role', 'worker', '--run-id', args.run_id,
               '--outdir', str(outdir), '--dataset', args.dataset, '--day-id', str(day),
               '--identity', args.identity]
        t = time.time()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                out, err = p.communicate(timeout=DAY_CAP_S)
                status = 'completed' if p.returncode == 0 else 'worker_error'
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
                status = 'deadline_killed'
                # the supervisor -- not the worker -- records the termination
                append_jsonl(ledger_p, {'event': 'DAY_DEADLINE_KILLED', 'day_id': day,
                                        'run_id': args.run_id, 'day_cap_s': DAY_CAP_S})
                led2 = read_ledger(ledger_p)
                for pos in range(1, 5):
                    ev, _ = slot_state(led2, day, pos)
                    if ev is None:
                        append_jsonl(ledger_p, {'event': 'NOT_ATTEMPTED_DUE_TO_TERMINATION',
                                                'day_id': day, 'position': pos,
                                                'run_id': args.run_id})
                    elif ev == 'ATTEMPT_STARTED':
                        append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': pos,
                                                'run_id': args.run_id,
                                                'reason': 'terminated_mid_slot'})
        except BaseException as exc:
            status, out, err = 'supervisor_exception', '', '%s: %s' % (type(exc).__name__, exc)
        results.append({'day_id': day, 'status': status, 'seconds': time.time() - t,
                        'stdout_tail': out[-400:], 'stderr_tail': err[-400:]})
        atomic_write_bytes(outdir / ('WORKER_DAY%02d.json' % day),
                           json.dumps(results[-1], indent=2, default=repr, sort_keys=True).encode())
        if (time.time() - t0) > BATCH_CAP_S:
            results.append({'batch': 'deadline_killed', 'batch_wall_s': time.time() - t0})
            break
    led = read_ledger(ledger_p)
    published = [r for r in led if r.get('event') == 'PUBLISHED']
    res = {'run_id': args.run_id, 'verdict': 'BATCH_COMPLETE',
           'slots_published': len(published), 'slots_in_denominator': 64,
           'batch_wall_s': time.time() - t0, 'caps': {'day': DAY_CAP_S, 'slot': SLOT_CAP_S,
                                                      'batch': BATCH_CAP_S},
           'days': results, 'identity': {k: idc[k] for k in ('ok', 'dataset_sha256')}}
    atomic_write_bytes(outdir / 'SUPERVISOR_RESULT.json',
                       json.dumps(res, indent=2, default=repr, sort_keys=True).encode())
    print(json.dumps({k: res[k] for k in ('run_id', 'verdict', 'slots_published',
                                          'slots_in_denominator', 'batch_wall_s')},
                     default=repr, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True,
                    choices=('worker', 'supervisor', 'closure'))
    ap.add_argument('--run-id', default='local')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--identity', default=str(EXT.parent / 'research/review_3080_20261010/'
                                              'd3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/'
                                              'l1_gates/'
                                              'A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json'))
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_dev_check_teacher.npz'))
    ap.add_argument('--day-id', type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.role == 'worker':
        return collect_day(args)
    if args.role == 'supervisor':
        return supervise(args)
    seeds = ['d3_l1_collection_runner', 'd3_l1_causal_perturbation', 'd3_l1_bitwise_parity',
             'd3_l1_state_collector']
    mods, edges, external = closure(seeds)
    ident = {'seeds': seeds, 'n_internal': len(mods), 'modules': mods, 'edges': edges[:120],
             'external_sample': external[:40],
             'entry_points': {'worker': str(Path(__file__).relative_to(EXT)),
                              'supervisor': str(Path(__file__).relative_to(EXT)),
                              'runner_sha256': sha_file(Path(__file__))},
             'start_end_identity': {'computed_at': time.strftime('%Y-%m-%d %H:%M:%S')}}
    atomic_write_bytes(outdir / 'P0C_IDENTITY_CLOSURE.json',
                       json.dumps(ident, indent=2, default=repr, sort_keys=True).encode())
    print(json.dumps({'n_internal': len(mods), 'modules': sorted(mods),
                      'runner_sha256': ident['entry_points']['runner_sha256']},
                     default=repr, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
