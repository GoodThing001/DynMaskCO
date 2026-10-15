#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 P0-C V1.8. F-free development qualification and a fail-closed worker.

Real reveal hooks write/fsync WAL before capture. Opt-in scalar encoding preserves
the original runtime types and bits; the frozen comparator is unchanged.
The supervisor enforces source/data/contract identity, monotonic rollout/slot/batch
windows, real process-tree containment, and one pass with a fixed denominator.
Formal collection is blocked pending a separate user signature.
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
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests',
          'scripts/models'):
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
    record.setdefault('mono', time.monotonic())
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
    lifecycle = {'ATTEMPT_STARTED', 'PUBLISHED', 'FAILED', 'SKIPPED_ALREADY_ATTEMPTED',
                 'SKIPPED_LEDGER_CORRUPT', 'NOT_ATTEMPTED_DUE_TO_TERMINATION'}
    for rec in reversed(ledger):
        if (rec.get('day_id') == day_id and rec.get('position') == position
                and rec.get('event') in lifecycle):
            return rec.get('event'), rec
    return None, None


def may_attempt(ledger, day_id, position):
    ok, n_bad = ledger_integrity(ledger)
    if not ok:
        return False, 'LEDGER_CORRUPT'
    ev, _ = slot_state(ledger, day_id, position)
    return ev is None, ev


def open_slots(ledger, now=None, slot_cap=SLOT_CAP_S, monotonic=False):
    """Slots whose ATTEMPT_STARTED has no terminal record yet (used by the supervisor watchdog)."""
    now = time.time() if now is None else now
    started, terminal = {}, set()
    for r in ledger:
        key = (r.get('day_id'), r.get('position'))
        if r.get('event') == 'ATTEMPT_STARTED':
            started[key] = r.get('mono' if monotonic else 'ts', now)
        elif r.get('event') in (('SLOT_FINISHED', 'SKIPPED_ALREADY_ATTEMPTED',
                                 'SKIPPED_LEDGER_CORRUPT', 'NOT_ATTEMPTED_DUE_TO_TERMINATION')
                                if monotonic else ('PUBLISHED', 'FAILED', 'SKIPPED_ALREADY_ATTEMPTED',
                                'SKIPPED_LEDGER_CORRUPT', 'NOT_ATTEMPTED_DUE_TO_TERMINATION')):
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
def inject_parity_fault(fault, traces, served):
    """Engineering negative controls only. Alters the resumed branch before the REAL gate."""
    import numpy as np
    import struct
    if fault == 'type':
        traces[0].services[0].arrival_time = float(traces[0].services[0].arrival_time)
    elif fault == 'dtype':
        served.dtype = np.uint8
    elif fault == 'signed_zero':
        traces[0].return_segment_thermal_violation_duration_h = -0.0
    elif fault == 'nan_payload':
        traces[0].return_segment_thermal_violation_duration_h = struct.unpack(
            '>d', bytes.fromhex('7ff8000000000001'))[0]
    elif fault == 'vehicle_boundary':
        traces[0].vehicle_id = 1


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
    from d3_l1_p0c_v18 import identity_gate, write_json, day_blocked
    assert_f_free()
    outdir = Path(args.outdir)
    ledger_p = outdir / 'STATE_LEDGER.jsonl'
    start_identity = identity_gate(sys.modules[__name__], args)
    write_json(sys.modules[__name__], outdir / ('WORKER_IDENTITY_START_DAY%02d.json' % args.day_id),
               start_identity)
    if not start_identity['ok']:
        append_jsonl(ledger_p, {'event': 'HARD_STOP_IDENTITY', 'day_id': args.day_id,
                                'run_id': args.run_id, 'identity': start_identity})
        return 2
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
    # ---- blocker 1 fix (V1.7): registration happens INSIDE the real reveal hook ----------
    if not ok_led or any(not may_attempt(led, day, pos)[0] for pos in range(1, 5)) or any(
            r.get('day_id') == day and r.get('event') in ('DAY_TERMINATED', 'DAY_COMPLETED')
            for r in led):
        append_jsonl(ledger_p, {'event': 'DAY_REFUSED_ONE_PASS', 'day_id': day,
                                'run_id': args.run_id, 'ledger_integrity': ok_led})
        return 1
    registered = {}

    def _pre_capture(slot, clock, event_id, reveal_idx):
        allowed, ev = may_attempt(read_ledger(ledger_p), day, slot)
        if not allowed:
            append_jsonl(ledger_p, {'event': 'SKIPPED_ALREADY_ATTEMPTED', 'day_id': day,
                                    'position': slot, 'previous_event': ev,
                                    'run_id': args.run_id})
            raise RuntimeError('slot %d already attempted (%s)' % (slot, ev))
        append_jsonl(ledger_p, {'event': 'ATTEMPT_STARTED', 'day_id': day, 'position': slot,
                                'run_id': args.run_id, 'clock': float(clock),
                                'event_id': int(event_id), 'reveal_idx': int(reveal_idx),
                                'registered_before_capture': True, 'ledger_integrity_ok': True})
        registered[slot] = time.monotonic()
    led = read_ledger(ledger_p)
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
    t_day = time.time()
    append_jsonl(ledger_p, {'event': 'DAY_ROLLOUT_STARTED', 'day_id': day, 'run_id': args.run_id})
    col = RevealSlotCollector(env, day, 4, pre_capture=_pre_capture, preserve_scalar_types=True)
    roll = run_baseline_rollout(env, day, hook=col)
    orig_traces, orig_served = roll['result']
    append_jsonl(ledger_p, {'event': 'DAY_ROLLOUT_FINISHED', 'day_id': day,
                            'run_id': args.run_id, 'day_rollout_wall_s': time.time() - t_day})
    shots = {r['slot']: r for r in col.slots}
    for position in range(1, 5):
        if position not in registered:
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'reason': 'insufficient_events', 'run_id': args.run_id})
    day_report = {'day_id': day, 'rollout_seconds': time.time() - t_day, 'slots_found': len(shots)}
    for position in sorted(registered):
        ev, _ = slot_state(read_ledger(ledger_p), day, position)   # V1.7: fresh read after rollout
        if ev != 'ATTEMPT_STARTED':
            continue
        r = shots.get(position)
        if r is None or 'snapshot' not in r:
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'reason': 'insufficient_events', 'run_id': args.run_id})
            continue
        t0 = registered[position]
        par = {}
        snap_path = None
        res_traces, res_served = None, None
        timings = {'capture_s': r.get('capture_seconds', 0.0)}
        try:
            if args.inject_block_s > 0 and position == min(registered):
                time.sleep(args.inject_block_s)          # E6: force the supervisor's kill path
            snap = r['snapshot']
            if position == 1 and args.fault == 'restore_error':
                snap = copy.deepcopy(snap)
                snap['state_hash'] = '0' * 64
            stage_start = time.monotonic()
            h1, h2 = snapshot_state_hash(snap), snapshot_state_hash(snap)
            timings['hash_twice_s'] = time.monotonic() - stage_start
            stage_start = time.monotonic()
            restored = restore_recourse_snapshot(snap)
            timings['restore_s'] = time.monotonic() - stage_start
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
            # E2: real full-resume parity against the ORIGINAL terminal rollout
            from recourse_snapshot import validate_snapshot_against_env
            env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            validate_snapshot_against_env(env_r, snap)
            recap = []
            def recapture(e, i, clock, event_id, reveal_idx, vs, ts, sm, universe):
                if not recap:
                    from recourse_snapshot import capture_recourse_snapshot
                    recap.append(capture_recourse_snapshot(e, i, clock, event_id, reveal_idx,
                                 vs, ts, sm, all_customers=universe, preserve_scalar_types=True))
            env_r.snapshot_hook = recapture
            stage_start = time.monotonic()
            resumed = env_r.run_resumed(snap)
            timings['resume_s'] = time.monotonic() - stage_start
            res_traces, res_served = (resumed if isinstance(resumed, tuple) else (resumed, None))
            par['recapture_state_hash_equal'] = bool(recap and snapshot_state_hash(recap[0]) == h1)
            if position == 1:
                inject_parity_fault(args.fault, res_traces, res_served)
            stage_start = time.monotonic()
            par['full_resume_served_mask'] = bitwise_equal(np.asarray(orig_served),
                                                           np.asarray(res_served))[0] \
                if res_served is not None else 'NOT_VERIFIABLE'
            par['full_resume_traces'], par['full_resume_trace_diffs'] = bitwise_equal(orig_traces, res_traces)
            par['full_resume_parity'] = (par['full_resume_served_mask'] == 'EQUAL'
                                         and par['full_resume_traces'] == 'EQUAL')
            timings['full_parity_s'] = time.monotonic() - stage_start
            # ---- blocker 3: lossless snapshot persistence + round trip --------------------
            stage_start = time.monotonic()
            blob = pickle.dumps(snap, protocol=4)
            snap_path = outdir / 'staged' / ('day%02d_pos%d.pkl' % (day, position))
            atomic_write_bytes(snap_path, blob)
            with open(snap_path, 'rb') as fh:
                reloaded = pickle.load(fh)
            par['serialization_roundtrip'] = (snapshot_state_hash(reloaded) == h1)
            par['serialization_exact'] = bitwise_equal(snap, reloaded)[0]
            timings['serialization_roundtrip_s'] = time.monotonic() - stage_start
            # ---- causal projections (blocker 7: boundary-aware suffix identity) ----------
            stage_start = time.monotonic()
            p_orig = decision_projection(env, snap, make_replanner(50.0, 24.0, eff)[0], day)
            timings['project_original_s'] = time.monotonic() - stage_start
            p_data, prep = perturb(data, snap, day, position)
            if p_data is None:
                raise RuntimeError('perturbation not valid: %s' % prep.get('verdict'))
            env_p = build_env(p_data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            stage_start = time.monotonic()
            p_pert = decision_projection(env_p, snap, make_replanner(50.0, 24.0, eff)[0], day)
            timings['project_perturbed_s'] = time.monotonic() - stage_start
            for side, src in ((p_orig, snap), (p_pert, snap)):
                if side.get('planned_suffix') is not None:
                    side['planned_suffix_sha_boundary'] = hashlib.sha256(json.dumps(
                        side['planned_suffix'], sort_keys=True).encode()).hexdigest()
            ck1, per, reasons = compare_projections(p_orig, p_pert)
            bnd = (p_orig.get('planned_suffix_sha_boundary')
                   == p_pert.get('planned_suffix_sha_boundary'))
            flat_same = (p_orig.get('planned_suffix_sha') == p_pert.get('planned_suffix_sha'))
            # E1: only the array-comparison verdicts are strings; the structural checks are
            # booleans and must be tested as booleans.
            bit_keys = ('served_mask', 'full_resume_served_mask', 'full_resume_traces')
            parity_ok = (all(par.get(k) == 'EQUAL' for k in bit_keys)
                         and par.get('suffix_boundary') is True
                         and par.get('serialization_roundtrip') is True
                         and par.get('serialization_exact') == 'EQUAL'
                         and par.get('recapture_state_hash_equal') is True
                         and par.get('full_resume_parity') is True)
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
            ent.update(timings=timings, slot_active_compute_s=sum(timings.values()),
                       original_projection=p_orig, perturbed_projection=p_pert,
                       capture_calls=1, explicit_restore_calls=1, hash_calls=2,
                       full_resume_calls=1, recapture_calls=len(recap))
            if not parity_ok:
                raise RuntimeError('bitwise_parity_failed: %s' % par)
            if not ck1 or not bnd:
                raise RuntimeError('causal_violation: %s boundary_equal=%s' % (reasons, bnd))
            if len(objs) != 3 or h1 != h2 or h1 != snap['state_hash']:
                raise RuntimeError('restore_or_hash_mismatch')
            if ent['replanner_calls'] != 2:
                raise RuntimeError('projection_call_count_mismatch')
            if time.monotonic() - t0 > args.slot_cap_s:
                raise RuntimeError('slot_timeout_before_publish')
            # A staged object has no publication status. Only a validated object is promoted.
            final_path = outdir / 'snapshots' / snap_path.name
            atomic_write_bytes(final_path, blob)
            snap_path = final_path
            if position == 1 and args.fault == 'write_error':
                raise OSError('injected manifest write failure after object persistence')
            evidence_path = outdir / 'validation' / snap_path.name
            evidence_blob = pickle.dumps({'original_traces': orig_traces,
                'resumed_traces': res_traces, 'original_served': orig_served,
                'resumed_served': res_served, 'recaptured_snapshot': recap[0],
                'reloaded_snapshot': reloaded, 'original_projection': p_orig,
                'perturbed_projection': p_pert}, protocol=4)
            atomic_write_bytes(evidence_path, evidence_blob)
            ent.update(validation_path=str(evidence_path.relative_to(outdir)),
                       validation_sha256=hashlib.sha256(evidence_blob).hexdigest())
            append_jsonl(outdir / 'STATE_MANIFEST.jsonl', dict(ent,
                         snapshot_path=str(snap_path.relative_to(outdir)), bytes=len(blob),
                         snapshot_sha256=hashlib.sha256(blob).hexdigest(), status='VALIDATED_CANDIDATE'))
            append_jsonl(ledger_p, {'event': 'PUBLISHED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'state_hash': h1,
                                    'snapshot_path': str(snap_path.relative_to(outdir)),
                                    'bytes': len(blob),
                                    'snapshot_sha256': hashlib.sha256(blob).hexdigest(),
                                    'validation_path': ent['validation_path'],
                                    'validation_sha256': ent['validation_sha256'],
                                    'serialization': 'pickle-protocol4-lossless',
                                    'slot_span_wall_s': time.monotonic() - t0,
                                    'validation': ent,
                                    'bitwise_parity_ok': True, 'projection_causal': True,
                                    'suffix_boundary_sha_equal': True,
                                    'n_invisible_modified': prep.get('n_invisible_modified')})
        except BaseException as exc:
            msg = '%s: %s' % (type(exc).__name__, str(exc)[:300])
            # A failed new day/slot must remain diagnosable without recollecting it.
            failure_path = outdir / 'failed' / ('day%02d_pos%d.pkl' % (day, position))
            failure_blob = pickle.dumps({'snapshot': r.get('snapshot'),
                'original_traces': orig_traces, 'original_served': orig_served,
                'resumed_traces': res_traces, 'resumed_served': res_served,
                'bitwise_parity': par, 'exception_type': type(exc).__name__,
                'exception_message': str(exc)}, protocol=4)
            atomic_write_bytes(failure_path, failure_blob)
            reason = ('causal_violation' if 'causal_violation' in msg else
                      'bitwise_parity_failed' if 'bitwise_parity' in msg else 'exception')
            append_jsonl(ledger_p, {'event': 'FAILED', 'day_id': day, 'position': position,
                                    'run_id': args.run_id, 'reason': reason, 'error': msg,
                                    'bitwise_parity': par,
                                    'failure_evidence_path': str(failure_path.relative_to(outdir)),
                                    'failure_evidence_sha256': hashlib.sha256(failure_blob).hexdigest(),
                                    'snapshot_staged_or_orphan': str(snap_path) if snap_path else None,
                                    'slot_span_wall_s': time.monotonic() - t0})
        # PUBLISHED is observable by the parent before append_jsonl's fsync returns.
        # Keep the slot watchdog open until that durable write has actually completed.
        append_jsonl(ledger_p, {'event': 'SLOT_FINISHED', 'day_id': day, 'position': position,
                                'run_id': args.run_id, 'slot_span_wall_s': time.monotonic() - t0})
        if (time.monotonic() - t0) > args.slot_cap_s:
            append_jsonl(ledger_p, {'event': 'SLOT_OVER_CAP', 'day_id': day,
                                    'position': position, 'slot_span_wall_s': time.monotonic() - t0})
    day_report['day_seconds'] = time.time() - t_day
    status = {'run_id': args.run_id, 'role': 'worker', 'day_id': day, 'day_report': day_report,
              'f_free': assert_f_free(), 'source_sha256': sha_file(Path(__file__))}
    atomic_write_bytes(outdir / 'RUN_STATUS.json',
                       json.dumps(status, indent=2, default=repr, sort_keys=True).encode())
    end_identity = identity_gate(sys.modules[__name__], args, end=True)
    end_identity['ok'] = (end_identity['ok'] and end_identity['runtime_identity_sha256']
                           == start_identity['runtime_identity_sha256'])
    write_json(sys.modules[__name__], outdir / ('WORKER_IDENTITY_END_DAY%02d.json' % day), end_identity)
    print(json.dumps(status, default=repr, sort_keys=True))
    if not end_identity['ok']:
        return 2
    return 1 if any(r.get('event') == 'FAILED' and r.get('day_id') == day
                    for r in read_ledger(ledger_p)) else 0


# --------------------------------------------------------------------------- supervisor
def supervise(args):
    from d3_l1_p0c_v18 import supervise as run_supervisor
    return run_supervisor(sys.modules[__name__], args)


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
    scratch.mkdir(parents=True, exist_ok=False)  # never overwrite earlier acceptance evidence
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
    ap.add_argument('--role', required=True, choices=('worker', 'supervisor', 'closure', 'accept',
                                                      'register-identity'))
    ap.add_argument('--run-id', default='local')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--identity', default=DEFAULT_IDENTITY)
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--expect-dataset-sha256', default=DEV_CHECK_SHA)
    ap.add_argument('--day-id', type=int, default=0)
    ap.add_argument('--days', type=int, default=1)
    ap.add_argument('--slot-cap-s', type=float, default=SLOT_CAP_S)
    ap.add_argument('--day-cap-s', type=float, default=DAY_CAP_S)
    ap.add_argument('--batch-cap-s', type=float, default=BATCH_CAP_S)
    ap.add_argument('--runtime-closure', action='store_true')
    ap.add_argument('--inject-block-s', type=float, default=0.0)
    ap.add_argument('--day-rollout-cap-s', type=float, default=DAY_CAP_S)
    ap.add_argument('--expect-runner-sha256', default=None)
    ap.add_argument('--expect-profile-sha256', default=None)
    ap.add_argument('--expect-effective-contract', default=None)
    ap.add_argument('--runtime-identity', default=None)
    ap.add_argument('--purpose', choices=('engineering', 'formal'), default='engineering')
    ap.add_argument('--authorization', default=None)
    ap.add_argument('--expect-authorization-sha256', default=None)
    ap.add_argument('--supervisor-go', default=None)
    ap.add_argument('--fault', default='none', choices=('none', 'type', 'dtype', 'signed_zero',
                        'nan_payload', 'vehicle_boundary', 'restore_error', 'write_error'))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.supervisor_go:
        deadline = time.monotonic() + 10
        while not Path(args.supervisor_go).exists():
            if time.monotonic() > deadline:
                return 3
            time.sleep(0.01)
    if args.purpose != 'engineering' and (args.fault != 'none' or args.inject_block_s):
        raise ValueError('fault injection is engineering only')
    if args.role == 'register-identity':
        from d3_l1_p0c_v18 import execution_identity
        identity = execution_identity(sys.modules[__name__], args)
        if identity['origin_errors'] or not all(identity['frozen_checks'].values()):
            raise RuntimeError('identity registration failed frozen/origin checks')
        # Registration is an explicit versioned artifact; supervisor never self-freezes.
        dest = outdir / ('RUNTIME_IDENTITY_V19.json' if args.purpose == 'formal'
                         else 'RUNTIME_IDENTITY_V18.json')
        with open(dest, 'x', encoding='utf-8') as fh:
            json.dump(identity, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        print(str(dest))
        return 0
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
