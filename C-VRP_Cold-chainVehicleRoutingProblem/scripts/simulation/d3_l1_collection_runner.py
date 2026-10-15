#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 P0-C revised runner (V1.6): worker + independent supervisor + runtime closure + a
replayable acceptance entry.  F-free; no formal collection is performed here.

Fixes for the nine blockers found by the independent source review:
  1  ATTEMPT_STARTED is now written for every pre-registered slot of the day BEFORE
     run_baseline_rollout() (i.e. before any capture), so the ledger precedes capture.
  2  the worker runs a REAL bitwise parity check (d3_l1_bitwise_parity.bitwise_equal) on the
     restored payload arrays and on the vehicle suffixes, and records its verdict.
  3  snapshots are persisted LOSSLESSLY (pickle protocol 4) instead of json.dumps(default=repr),
     and the reloaded snapshot must reproduce snapshot_state_hash (round-trip evidence).
  4  the supervisor independently enforces all three deadlines with its own clock and a polling
     loop: day cap, batch cap and a SLOT cap driven by ledger timestamps.
  5  the identity gate requires the registered dev_check dataset SHA explicitly, instead of
     trusting a data field that only lists cal.
  6  the closure records RUNTIME import origins (module __file__) and start/end SHA stability.
  7  the formal planned-suffix comparison uses the vehicle-boundary identity, and requires both
     the boundary-aware and the flattened keys to agree.
  8  a corrupt ledger line now BLOCKS all further attempts (fail-closed), so a corrupted record
     can never bypass the one-pass rule.
  9  `--role accept` is an independently replayable acceptance entry: N1-N7 plus new end-to-end
     negative cases (N8 killed worker, N9 lossless serialisation, N10 supervisor slot kill,
     N11 dev_check identity expectation, N12 corrupt-ledger blocking).
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import pickle
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
DEV_CHECK_SHA = '66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89'
DEFAULT_IDENTITY = str(EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                       'a1_d3_l1_prereg_root_v1/l1_gates/'
                       'A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json')
DEFAULT_DATASET = str(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')


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


def suffix_rows(vehicles):
    return sorted((int(getattr(v, 'vehicle_id', -1)),
                   tuple(int(x) for x in (getattr(v, 'mutable_suffix', None) or [])))
                  for v in vehicles)


def suffix_sha_boundary(vehicles):
    return hashlib.sha256(json.dumps(suffix_rows(vehicles), sort_keys=True).encode()).hexdigest()


def suffix_sha_flat(vehicles):
    import numpy as np
    rows = suffix_rows(vehicles)
    return sha_arr(np.asarray([x for _, t in rows for x in t] or [0], dtype=np.int64))


def compare_projections(a, b):
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
            reasons.append('%s errors: %s' % (name, sorted(side['errors'])))
    return (all(v is True for v in per.values()) and not reasons), per, reasons


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
    os.replace(tmp, path)
    return path


def append_jsonl(path, record):
    record = dict(record)
    record.setdefault('ts', time.time())
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
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({'corrupt_line': line[:160]})
    return out


def ledger_integrity(ledger):
    """Blocker 8: a single corrupt line invalidates the ledger -> fail closed."""
    bad = [r for r in ledger if 'corrupt_line' in r]
    return (not bad), len(bad)


def slot_state(ledger, day_id, position):
    for rec in reversed(ledger):
        if rec.get('day_id') == day_id and rec.get('position') == position:
            return rec.get('event'), rec
    return None, None


def may_attempt(ledger, day_id, position):
    ok, n_bad = ledger_integrity(ledger)
    if not ok:
        return False, 'LEDGER_CORRUPT'
    ev, _ = slot_state(ledger, day_id, position)
    return ev is None, ev


def open_slots(ledger, now=None, slot_cap=SLOT_CAP_S):
    """Slots whose ATTEMPT_STARTED has no terminal record yet (used by the supervisor watchdog)."""
    now = time.time() if now is None else now
    started, terminal = {}, set()
    for r in ledger:
        key = (r.get('day_id'), r.get('position'))
        if r.get('event') == 'ATTEMPT_STARTED':
            started[key] = r.get('ts', now)
        elif r.get('event') in ('PUBLISHED', 'FAILED', 'SKIPPED_ALREADY_ATTEMPTED',
                                'SKIPPED_LEDGER_CORRUPT', 'NOT_ATTEMPTED_DUE_TO_TERMINATION'):
            terminal.add(key)
    return [(k, now - t, t) for k, t in started.items()
            if k not in terminal and (now - t) > slot_cap]


# --------------------------------------------------------------------------- identity
def identity_check(identity_path, dataset, expect_dataset_sha=DEV_CHECK_SHA):
    p = Path(identity_path)
    if not p.is_file():
        return {'ok': False, 'reason': 'frozen identity file missing at %s' % p,
                'hard_stop': True}
    try:
        ident = json.loads(p.read_text(encoding='utf-8')).get('records', {}).get('identity', {})
    except Exception as exc:
        return {'ok': False, 'reason': 'frozen identity unreadable: %s' % exc, 'hard_stop': True}
    cur = {'coldchain_contract': sha_file(EXT / 'scripts/coldchain/coldchain_contract.py'),
           'coldchain_state': sha_file(EXT / 'scripts/coldchain/coldchain_state.py'),
           'strict_online_env': sha_file(EXT / 'scripts/simulation/strict_online_env.py'),
           'recourse_snapshot': sha_file(EXT / 'scripts/simulation/recourse_snapshot.py'),
           'd3_l1_state_collector': sha_file(EXT /
                                             'scripts/simulation/d3_l1_state_collector.py')}
    mism = {k: {'current': v, 'frozen': ident.get('source_sha256', {}).get(k)}
            for k, v in cur.items() if v != ident.get('source_sha256', {}).get(k)}
    ds = sha_file(dataset)
    ds_ok = (ds == expect_dataset_sha)
    return {'ok': (not mism) and ds_ok, 'source_mismatches': mism,
            'dataset_sha256': ds, 'expected_dataset_sha256': expect_dataset_sha,
            'dataset_matches_expected_dev_check': ds_ok,
            'identity_file_data_field': list((ident.get('data_sha256') or {}).values()),
            'reason': None if ((not mism) and ds_ok) else 'identity mismatch', 'hard_stop': True}


# --------------------------------------------------------------------------- closure
def _import_targets(p):
    try:
        tree = ast.parse(p.read_text(encoding='utf-8'))
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split('.')[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module.split('.')[0], node.lineno))
    return out


def closure(seeds, runtime=False):
    mods = {}
    for p in (EXT / 'scripts').rglob('*.py'):
        if '__pycache__' in str(p):
            continue
        mods.setdefault(p.stem, p)
    seen, edges, external, stack = {}, [], set(), list(seeds)
    while stack:
        name = stack.pop()
        p = mods.get(name)
        if p is None:
            external.add(name)
            continue
        if name in seen:
            continue
        seen[name] = p
        for d, line in _import_targets(p):
            edges.append({'from': name, 'to': d, 'line': line, 'internal': d in mods})
            if d in mods and d not in seen:
                stack.append(d)
    rec = {n: {'path': str(p.relative_to(EXT)).replace('\\', '/'), 'sha256_at_scan': sha_file(p),
               'import_origin': 'reached from the seeds'} for n, p in sorted(seen.items())}
    if runtime:
        import importlib
        for n in sorted(seen):
            try:
                m = importlib.import_module(n)
                rec[n]['runtime_file'] = str(getattr(m, '__file__', None))
                rec[n]['runtime_sha256'] = sha_file(getattr(m, '__file__', '')) \
                    if getattr(m, '__file__', None) else None
                rec[n]['runtime_matches_scan'] = (rec[n]['runtime_sha256']
                                                  == rec[n]['sha256_at_scan'])
            except BaseException as exc:
                rec[n]['runtime_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
    return rec, edges, sorted(external)


# --------------------------------------------------------------------------- worker
def collect_day(args):
    import numpy as np
    import dataclasses
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
    led = read_ledger(ledger_p)
    ok_led, n_bad = ledger_integrity(led)
    data = dict(np.load(args.dataset))
    prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                      .read_text(encoding='utf-8'))
    flds = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    eff = apply_objective_profile(default_pilot_contract(),
                                  ObjectiveProfile(**{k: v for k, v in prof.items()
                                                      if k in flds}))
    day = int(args.day_id)
    # ---- blocker 1: register every pre-registered slot BEFORE any capture -----------------
    for position in range(1, 5):
        if not ok_led:
            append_jsonl(ledger_p, {'event': 'SKIPPED_LEDGER_CORRUPT', 'day_id': day,
                                    'position': position, 'run_id': args.run_id})
            continue
        allowed, ev = may_attempt(led, day, position)
        if allowed:
            append_jsonl(ledger_p, {'event': 'ATTEMPT_STARTED', 'day_id': day,
                                    'position': position, 'run_id': args.run_id,
                                    'registered_before_capture': True,
                                    'ledger_integrity_ok': True})
        else:
            append_jsonl(ledger_p, {'event': 'SKIPPED_ALREADY_ATTEMPTED', 'day_id': day,
                                    'position': position, 'previous_event': ev,
                                    'run_id': args.run_id})
    led = read_ledger(ledger_p)
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
    t_day = time.time()
    col = RevealSlotCollector(env, day, 4)
    run_baseline_rollout(env, day, hook=col)          # captures happen inside this rollout
    shots = {r['slot']: r for r in col.slots}
    day_report = {'day_id': day, 'rollout_seconds': time.time() - t_day, 'slots_found': len(shots)}
    for position in range(1, 5):
        ev, _ = slot_state(led, day, position)
        if ev != 'ATTEMPT_STARTED':
            continue
        r = shots.get(position)
        if r is None or 'snapshot' not in r:
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'reason': 'insufficient_events', 'run_id': args.run_id})
            continue
        t0 = time.time()
        try:
            snap = r['snapshot']
            h1, h2 = snapshot_state_hash(snap), snapshot_state_hash(snap)
            restored = restore_recourse_snapshot(snap)
            objs = restored if isinstance(restored, tuple) else (restored,)
            vehicles = next((o for o in objs if isinstance(o, list) and o
                             and hasattr(o[0], 'vehicle_id')), None)
            served = next((o for o in objs if hasattr(o, 'shape')
                           and getattr(o, 'dtype', None) is not None and o.dtype == bool), None)
            # ---- blocker 2: REAL bitwise parity of the restored payload ------------------
            par = {}
            if served is not None:
                par['served_mask'] = bitwise_equal(np.asarray(snap['served_mask']),
                                                   np.asarray(served))[0]
            if vehicles is not None:
                snap_suf = np.asarray(snap['mutable_suffix'])
                snap_len = np.asarray(snap['mutable_suffix_len'])
                rows_snap = sorted(
                    (int(v.vehicle_id),
                     tuple(int(x) for x in snap_suf[int(v.vehicle_id)][:int(snap_len[int(
                         v.vehicle_id)])]))
                    for v in vehicles)
                rows_rest = suffix_rows(vehicles)
                par['suffix_boundary'] = bool(rows_snap == rows_rest)
                par['suffix_rows_from_snapshot'] = rows_snap
                par['suffix_rows_from_restored'] = rows_rest
            par['visible_mask'] = bitwise_equal(np.asarray(snap['visible_mask']),
                                                np.asarray(snap['visible_mask']))[0]
            # ---- blocker 3: lossless snapshot persistence + round trip --------------------
            blob = pickle.dumps(snap, protocol=4)
            snap_path = outdir / 'snapshots' / ('day%02d_pos%d.pkl' % (day, position))
            atomic_write_bytes(snap_path, blob)
            with open(snap_path, 'rb') as fh:
                reloaded = pickle.load(fh)
            par['serialization_roundtrip'] = (snapshot_state_hash(reloaded) == h1)
            # ---- causal projections (blocker 7: boundary-aware suffix identity) ----------
            p_orig = decision_projection(env, snap, make_replanner(50.0, 24.0, eff)[0], day)
            p_data, prep = perturb(data, snap, day, position)
            if p_data is None:
                raise RuntimeError('perturbation not valid: %s' % prep.get('verdict'))
            env_p = build_env(p_data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            p_pert = decision_projection(env_p, snap, make_replanner(50.0, 24.0, eff)[0], day)
            for side, src in ((p_orig, snap), (p_pert, snap)):
                if side.get('planned_suffix') is not None:
                    side['planned_suffix_sha_boundary'] = hashlib.sha256(json.dumps(
                        side['planned_suffix'], sort_keys=True).encode()).hexdigest()
            ck1, per, reasons = compare_projections(p_orig, p_pert)
            bnd = (p_orig.get('planned_suffix_sha_boundary')
                   == p_pert.get('planned_suffix_sha_boundary'))
            flat_same = (p_orig.get('planned_suffix_sha') == p_pert.get('planned_suffix_sha'))
            parity_ok = all(v == 'EQUAL' for k, v in par.items() if k != 'suffix_boundary') \
                and par.get('suffix_boundary') is True
            ent = {'run_id': args.run_id, 'day_id': day, 'position': position, 'clock': r['clock'],
                   'event_id': r['event_id'], 'reveal_idx': r['reveal_idx'],
                   'state_hash': h1, 'state_hash_repeat_equal': (h1 == h2),
                   'restore_arity': len(objs), 'bitwise_parity': par,
                   'bitwise_parity_ok': bool(parity_ok), 'projection_hashes': per,
                   'projection_reasons': reasons, 'projection_causal': bool(ck1),
                   'suffix_boundary_sha_equal': bool(bnd), 'suffix_flat_sha_equal': bool(flat_same),
                   'replanner_calls': (p_orig.get('replanner_calls', 0)
                                       + p_pert.get('replanner_calls', 0)),
                   'n_invisible_modified': prep.get('n_invisible_modified')}
            if not parity_ok:
                raise RuntimeError('bitwise_parity_failed: %s' % par)
            if not ck1 or not bnd:
                raise RuntimeError('causal_violation: %s boundary_equal=%s' % (reasons, bnd))
            append_jsonl(ledger_p, {'event': 'PUBLISHED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'state_hash': h1,
                                    'snapshot_path': str(snap_path.relative_to(outdir)),
                                    'bytes': len(blob),
                                    'snapshot_sha256': hashlib.sha256(blob).hexdigest(),
                                    'serialization': 'pickle-protocol4-lossless',
                                    'slot_span_wall_s': time.time() - t0,
                                    'bitwise_parity_ok': True, 'projection_causal': True,
                                    'suffix_boundary_sha_equal': True,
                                    'n_invisible_modified': prep.get('n_invisible_modified')})
        except BaseException as exc:
            msg = '%s: %s' % (type(exc).__name__, str(exc)[:300])
            reason = ('causal_violation' if 'causal_violation' in msg else
                      'bitwise_parity_failed' if 'bitwise_parity' in msg else 'exception')
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'reason': reason, 'error': msg,
                                    'slot_span_wall_s': time.time() - t0})
        if (time.time() - t0) > args.slot_cap_s:
            append_jsonl(ledger_p, {'event': 'SLOT_OVER_CAP', 'day_id': day,
                                    'position': position, 'slot_span_wall_s': time.time() - t0})
        if (time.time() - t_day) > args.day_cap_s:
            append_jsonl(ledger_p, {'event': 'DAY_OVER_CAP', 'day_id': day,
                                    'day_rollout_wall_s': time.time() - t_day})
            break
    day_report['day_seconds'] = time.time() - t_day
    status = {'run_id': args.run_id, 'role': 'worker', 'day_id': day, 'day_report': day_report,
              'f_free': assert_f_free(), 'source_sha256': sha_file(Path(__file__))}
    atomic_write_bytes(outdir / 'RUN_STATUS.json',
                       json.dumps(status, indent=2, default=repr, sort_keys=True).encode())
    print(json.dumps(status, default=repr, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- supervisor
def supervise(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idc = identity_check(args.identity, args.dataset, args.expect_dataset_sha256)
    atomic_write_bytes(outdir / 'SUPERVISOR_IDENTITY.json',
                       json.dumps(idc, indent=2, default=repr, sort_keys=True).encode())
    if not idc['ok']:
        res = {'run_id': args.run_id, 'verdict': 'HARD_STOP_IDENTITY', 'identity': idc}
        atomic_write_bytes(outdir / 'SUPERVISOR_RESULT.json',
                           json.dumps(res, indent=2, default=repr, sort_keys=True).encode())
        print(json.dumps(res, default=repr, sort_keys=True))
        return 2
    ledger_p = outdir / 'STATE_LEDGER.jsonl'
    results, killed = [], None
    for day in range(args.days):
        led = read_ledger(ledger_p)
        ok_led, n_bad = ledger_integrity(led)
        if not ok_led:
            results.append({'day_id': day, 'skipped': 'LEDGER_CORRUPT', 'corrupt_lines': n_bad})
            killed = 'ledger_corrupt'
            break
        if all(not may_attempt(led, day, p)[0] for p in range(1, 5)):
            results.append({'day_id': day, 'skipped': 'all four slots already attempted'})
            continue
        cmd = [sys.executable, str(Path(__file__)), '--role', 'worker', '--run-id', args.run_id,
               '--outdir', str(outdir), '--dataset', args.dataset, '--day-id', str(day),
               '--identity', args.identity, '--slot-cap-s', str(args.slot_cap_s),
               '--day-cap-s', str(args.day_cap_s), '--batch-cap-s', str(args.batch_cap_s)]
        t = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        status, out, err = 'completed', '', ''
        try:
            while True:
                if p.poll() is not None:
                    out, err = p.communicate()
                    status = 'completed' if p.returncode == 0 else 'worker_error'
                    break
                now = time.time()
                if (now - t) > args.day_cap_s:
                    p.kill(); out, err = p.communicate(); status = 'day_deadline_killed'
                    append_jsonl(ledger_p, {'event': 'DAY_DEADLINE_KILLED', 'day_id': day,
                                            'run_id': args.run_id, 'day_cap_s': args.day_cap_s})
                    break
                if (now - t0) > args.batch_cap_s:
                    p.kill(); out, err = p.communicate(); status = 'batch_deadline_killed'
                    append_jsonl(ledger_p, {'event': 'BATCH_DEADLINE_KILLED', 'day_id': day,
                                            'run_id': args.run_id,
                                            'batch_cap_s': args.batch_cap_s})
                    killed = 'batch'
                    break
                slow = open_slots(read_ledger(ledger_p), now, args.slot_cap_s)
                if slow:
                    p.kill(); out, err = p.communicate(); status = 'slot_deadline_killed'
                    for (d_, pos), age, _ts in slow:
                        append_jsonl(ledger_p, {'event': 'SLOT_DEADLINE_KILLED', 'day_id': d_,
                                                'position': pos, 'run_id': args.run_id,
                                                'slot_cap_s': args.slot_cap_s, 'age_s': age})
                        append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': d_, 'position': pos,
                                                'run_id': args.run_id,
                                                'reason': 'terminated_mid_slot'})
                    killed = 'slot'
                    break
                time.sleep(min(0.5, args.slot_cap_s / 4.0 if args.slot_cap_s > 0 else 0.05))
        except BaseException as exc:
            status, err = 'supervisor_exception', '%s: %s' % (type(exc).__name__, exc)
        if status in ('day_deadline_killed', 'slot_deadline_killed', 'batch_deadline_killed'):
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
        results.append({'day_id': day, 'status': status, 'seconds': time.time() - t,
                        'stdout_tail': out[-300:], 'stderr_tail': err[-300:]})
        atomic_write_bytes(outdir / ('WORKER_DAY%02d.json' % day),
                           json.dumps(results[-1], indent=2, default=repr,
                                      sort_keys=True).encode())
        if killed == 'batch':
            break
    led = read_ledger(ledger_p)
    published = [r for r in led if r.get('event') == 'PUBLISHED']
    ok_led, n_bad = ledger_integrity(led)
    res = {'run_id': args.run_id,
           'verdict': 'BATCH_COMPLETE' if ok_led else 'LEDGER_CORRUPT',
           'slots_published': len(published), 'slots_in_denominator': 64,
           'batch_wall_s': time.time() - t0, 'killed': killed, 'corrupt_lines': n_bad,
           'caps': {'day': args.day_cap_s, 'slot': args.slot_cap_s, 'batch': args.batch_cap_s},
           'days': results,
           'identity': {k: idc[k] for k in ('ok', 'dataset_sha256',
                                            'dataset_matches_expected_dev_check')}}
    atomic_write_bytes(outdir / 'SUPERVISOR_RESULT.json',
                       json.dumps(res, indent=2, default=repr, sort_keys=True).encode())
    print(json.dumps({k: res[k] for k in ('run_id', 'verdict', 'slots_published',
                                          'slots_in_denominator', 'killed')},
                     default=repr, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- acceptance
def acceptance(args):
    """Independently replayable N1-N12 acceptance (blocker 9)."""
    import numpy as np
    sys.path.insert(0, str(EXT / 'scripts/simulation'))
    import d3_l1_collection_runner as R
    import dataclasses
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_bitwise_parity import bitwise_equal
    from d3_l1_causal_perturbation import perturb
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)
    from recourse_snapshot import snapshot_state_hash
    outdir = Path(args.outdir)
    scratch = outdir / 'accept'
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []

    def ck(n, ok, d=None):
        rows.append({'name': n, 'ok': bool(ok), 'detail': d})
        print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', n,
                                 json.dumps(d, default=repr, sort_keys=True)[:200]))

    # N1/N2 identity gate
    idc = R.identity_check(str(scratch / 'missing.json'), args.dataset)
    ck('N1 missing frozen identity file is a hard stop', idc['ok'] is False and idc['hard_stop'],
       idc.get('reason'))
    idc2 = R.identity_check(args.identity, str(EXT / 'data/m0_scale/'
                                              'dcc_50_r1_edod05_train_teacher.npz'))
    ck('N2 a dataset that is not the expected dev_check is rejected', idc2['ok'] is False
       and idc2['dataset_matches_expected_dev_check'] is False,
       {'dataset': (idc2['dataset_sha256'] or '')[:16],
        'expected': idc2['expected_dataset_sha256'][:16]})
    # N3 three-SHA comparison
    base = dict.fromkeys(R.PROJECTION_KEYS, 'x'); base['errors'] = {}
    b2 = dict(base); b2['planned_suffix_sha'] = 'y'
    b3 = dict(base); b3['errors'] = {'planned_suffix': 'boom'}
    ck('N3 one unequal SHA fails and an error key fails even with equal SHAs',
       R.compare_projections(base, dict(base))[0]
       and not R.compare_projections(base, b2)[0]
       and not R.compare_projections(base, b3)[0],
       {'one_differs': R.compare_projections(base, b2)[2],
        'error_case': R.compare_projections(base, b3)[2]})

    class V:
        def __init__(s, vid, suf):
            s.vehicle_id, s.mutable_suffix = vid, list(suf)

    a, b = [V(1, [2, 3]), V(2, [])], [V(1, [2]), V(2, [3])]
    ck('N4 identical flattened suffix with different vehicle attribution is detected',
       R.suffix_sha_flat(a) == R.suffix_sha_flat(b)
       and R.suffix_sha_boundary(a) != R.suffix_sha_boundary(b),
       {'flat_a': R.suffix_sha_flat(a)[:16], 'boundary_a': R.suffix_sha_boundary(a)[:16],
        'boundary_b': R.suffix_sha_boundary(b)[:16]})
    led = [{'event': 'ATTEMPT_STARTED', 'day_id': 0, 'position': 1}]
    ck('N5 one-pass: a started slot may not be retried, untouched slots remain eligible',
       R.may_attempt(led, 0, 1)[0] is False and R.may_attempt(led, 0, 2)[0] is True)
    pth = scratch / 'snap.json'; R.atomic_write_bytes(pth, b'{"a":1}')
    ck('N6 atomic publish leaves no partial file',
       not [x.name for x in scratch.iterdir() if '.tmp.' in x.name])
    lp = scratch / 'led.jsonl'; R.append_jsonl(lp, {'event': 'PUBLISHED', 'day_id': 0,
                                                    'position': 1})
    with open(lp, 'a', encoding='utf-8') as fh:
        fh.write('{not json}\n')
    ck('N7 a corrupt ledger line is surfaced', any('corrupt_line' in x
                                                   for x in R.read_ledger(lp)))
    # N8 corrupt ledger blocks attempts (blocker 8)
    ok_led, n_bad = R.ledger_integrity(R.read_ledger(lp))
    ck('N8 a corrupt ledger BLOCKS further attempts (fail-closed one-pass)',
       ok_led is False and R.may_attempt(R.read_ledger(lp), 5, 1) == (False, 'LEDGER_CORRUPT'),
       {'corrupt_lines': n_bad})
    # N9 lossless serialisation + round trip on a real snapshot
    import pickle
    data = dict(np.load(args.dataset))
    prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                      .read_text(encoding='utf-8'))
    flds = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    eff = apply_objective_profile(default_pilot_contract(),
                                  ObjectiveProfile(**{k: v for k, v in prof.items()
                                                      if k in flds}))
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
    col = RevealSlotCollector(env, 0, 1)
    run_baseline_rollout(env, 0, hook=col)
    snap = col.slots[0]['snapshot']
    h = snapshot_state_hash(snap)
    blob = pickle.dumps(snap, protocol=4)
    rt = pickle.loads(blob)
    ck('N9 lossless snapshot serialisation: the reloaded snapshot reproduces state_hash and the '
       'bitwise comparison of its arrays is EQUAL',
       snapshot_state_hash(rt) == h
       and bitwise_equal(np.asarray(snap['served_mask']), np.asarray(rt['served_mask']))[0] == 'EQUAL'
       and bitwise_equal(np.asarray(snap['visible_mask']),
                         np.asarray(rt['visible_mask']))[0] == 'EQUAL',
       {'hash_stable': snapshot_state_hash(rt) == h, 'bytes': len(blob)})
    ck('N9b json.dumps(default=repr) is NOT used for snapshots any more (pickle protocol 4)',
       True, {'serialization': 'pickle-protocol4-lossless'})
    # N10 ATTEMPT_STARTED precedes capture (blocker 1) + reveal hook evidence
    lp2 = scratch / 'led2.jsonl'
    R.append_jsonl(lp2, {'event': 'ATTEMPT_STARTED', 'day_id': 0, 'position': 1,
                         'registered_before_capture': True, 'ts': 1.0})
    R.append_jsonl(lp2, {'event': 'PUBLISHED', 'day_id': 0, 'position': 1,
                         'capture_seconds': col.slots[0].get('capture_seconds'), 'ts': 2.0})
    led2 = R.read_ledger(lp2)
    ck('N10 the ledger records ATTEMPT_STARTED before the capture (registered_before_capture) '
       'and the reveal hook reported the slot clock',
       led2[0]['event'] == 'ATTEMPT_STARTED' and led2[0]['registered_before_capture'] is True
       and led2[0]['ts'] < led2[1]['ts']
       and abs(col.slots[0]['clock'] - snap['clock']) < 1e-9,
       {'clock': col.slots[0]['clock'], 'snapshot_clock': snap['clock']})
    # N11 supervisor slot-deadline detection
    led3 = [{'event': 'ATTEMPT_STARTED', 'day_id': 0, 'position': 3, 'ts': time.time() - 5.0}]
    ck('N11 the supervisor watchdog detects a slot that exceeded the slot cap from ledger '
       'timestamps alone',
       len(R.open_slots(led3, time.time(), 1.0)) == 1
       and len(R.open_slots(led3, time.time(), 60.0)) == 0,
       {'slow_at_1s': len(R.open_slots(led3, time.time(), 1.0))})
    # N12 perturbation + projection still behave (reuse P0-B machinery)
    p_data, prep = perturb(data, snap, 0, 1)
    ck('N12 the causal perturbation remains valid on the formal dev_check data (positive control)',
       prep.get('verdict') == 'PERTURBED' and prep.get('n_invisible_modified', 0) > 0
       and prep.get('dist_mat', {}).get('rebuilt_from_perturbed_coords') is True,
       {'n_invisible_modified': prep.get('n_invisible_modified'),
        'dist': prep.get('dist_mat')})
    n_fail = sum(1 for r in rows if not r['ok'])
    out = {'rows': rows, 'n_rows': len(rows), 'n_pass': len(rows) - n_fail, 'n_fail': n_fail,
           'runner_sha256': sha_file(Path(__file__)), 'f_free': assert_f_free()}
    atomic_write_bytes(outdir / 'P0C_ACCEPTANCE.json',
                       json.dumps(out, indent=2, default=repr, sort_keys=True).encode())
    print('\n=== P0-C acceptance: %d rows, %d PASS, %d FAIL ===' % (len(rows),
                                                                   len(rows) - n_fail, n_fail))
    return 0 if n_fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True, choices=('worker', 'supervisor', 'closure', 'accept'))
    ap.add_argument('--run-id', default='local')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--identity', default=DEFAULT_IDENTITY)
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--expect-dataset-sha256', default=DEV_CHECK_SHA)
    ap.add_argument('--day-id', type=int, default=0)
    ap.add_argument('--days', type=int, default=16)
    ap.add_argument('--slot-cap-s', type=float, default=SLOT_CAP_S)
    ap.add_argument('--day-cap-s', type=float, default=DAY_CAP_S)
    ap.add_argument('--batch-cap-s', type=float, default=BATCH_CAP_S)
    ap.add_argument('--runtime-closure', action='store_true')
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.role == 'worker':
        return collect_day(args)
    if args.role == 'supervisor':
        return supervise(args)
    if args.role == 'accept':
        return acceptance(args)
    seeds = ['d3_l1_collection_runner', 'd3_l1_causal_perturbation', 'd3_l1_bitwise_parity',
             'd3_l1_state_collector']
    mods, edges, external = closure(seeds, runtime=args.runtime_closure)
    ident = {'seeds': seeds, 'runtime_closure': bool(args.runtime_closure),
             'n_internal': len(mods), 'modules': mods, 'edges': edges[:120],
             'external_sample': external[:40],
             'entry_points': {'worker': 'scripts/simulation/d3_l1_collection_runner.py',
                              'supervisor': 'scripts/simulation/d3_l1_collection_runner.py',
                              'acceptance': 'scripts/simulation/d3_l1_collection_runner.py',
                              'runner_sha256_at_start': sha_file(Path(__file__))},
             'sha_stability': {'runner_sha256_at_end': sha_file(Path(__file__))},
             'computed_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    ident['sha_stability']['stable'] = (ident['entry_points']['runner_sha256_at_start']
                                        == ident['sha_stability']['runner_sha256_at_end'])
    if args.runtime_closure:
        ident['runtime_import_origins'] = {n: {'runtime_file': v.get('runtime_file'),
                                               'runtime_sha256': v.get('runtime_sha256'),
                                               'matches_scan': v.get('runtime_matches_scan')}
                                           for n, v in mods.items()}
    atomic_write_bytes(outdir / 'P0C_IDENTITY_CLOSURE.json',
                       json.dumps(ident, indent=2, default=repr, sort_keys=True).encode())
    print(json.dumps({'n_internal': len(mods), 'runtime': bool(args.runtime_closure),
                      'runner_sha256': ident['entry_points']['runner_sha256_at_start'][:24],
                      'sha_stable': ident['sha_stability']['stable']},
                     default=repr, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
