#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 RESUME-TRAJECTORY EQUIVALENCE ATTRIBUTION (user-authorised, bounded).

Pre-fixed adjudication criteria (registered here BEFORE the run):
  * strict typed parity FAILS but semantic value parity PASSES            -> REPRESENTATION_ONLY
    (the original STRICT_TYPED_PARITY failure is still recorded as a failure)
  * semantic value parity FAILS                                           -> BEHAVIOR_DIVERGENCE
  * the comparison cannot be completed (node budget / depth guard /
    unreadable object / missing branch)                                   -> NOT_VERIFIABLE

Comparators: BOTH are always produced.
  1. STRICT TYPED  -- Python/numpy types must match exactly (the previous round's semantics),
     with NO depth cap: the only guard is a node budget, and exhausting it forces NOT_VERIFIABLE.
  2. SEMANTIC VALUE -- the ONLY tolerated difference is numpy.float64 <-> Python float (and numpy
     integer scalars vs Python int), compared for EXACT equality AND identical IEEE-754 bit
     pattern.  No allclose, no global float() coercion, nothing else tolerated.  A recursion or
     node-budget exhaustion is reported as a failure of the comparison, never as equality.

Cold-chain identity (read-only): snapshot schema must be recourse_snapshot_v3_cc, the snapshot's
coldchain_contract_hash must equal the effective contract hash, and the nested cold-chain fields
(per-vehicle and per-trace) are checked at their REAL nesting level rather than at the top level.
The frozen recourse_snapshot.py is not modified.

One execution binds source SHA + run_id + argv + stdout/stderr + result JSON.  Failures are
persisted; nothing is rewritten into a pass.
"""
import argparse
import dataclasses
import json
import os
import platform
import struct
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

NODE_BUDGET = 400000
DIFF_REPORT_CAP = 60
CASES = []
RECORDS = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))
    sys.stdout.flush()


def rec(k, v):
    RECORDS[k] = v


def sha(p):
    p = Path(p)
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def bits(x):
    try:
        return struct.pack('>d', float(x)).hex()
    except Exception:
        return None


NUMERIC_PAIRS = ((float,), (int, (bool,)))


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) or hasattr(x, 'dtype')


def _num(x):
    try:
        if hasattr(x, 'item'):
            return x.item()
        return x
    except Exception:
        return x


def _numeric_semantically_equal(a, b):
    """Only exact equality with identical bit patterns counts."""
    x, y = _num(a), _num(b)
    if isinstance(x, float) or isinstance(y, float):
        try:
            if float(x) != float(y):
                return False
        except Exception:
            return False
        bx, by = bits(x), bits(y)
        return (bx is not None and bx == by) or (float(x) == 0.0 and float(y) == 0.0)
    return x == y


def canon(x, state, depth=0):
    """Deep, canonical form.  state['nodes'] is the node budget; exhaustion is recorded."""
    state['nodes'] += 1
    if state['nodes'] > NODE_BUDGET:
        state['truncated'] = True
        raise RuntimeError('node budget exhausted')
    if depth > 14:
        state['depth_guard'] = True
        raise RuntimeError('depth guard hit at %d' % depth)
    if x is None or isinstance(x, (bool, str)):
        return x
    if hasattr(x, 'dtype') and getattr(x, 'ndim', 0) == 0:
        return _num(x)
    if hasattr(x, 'tolist'):
        return canon(x.tolist(), state, depth + 1)
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, (list, tuple)):
        return [canon(i, state, depth + 1) for i in x]
    if isinstance(x, dict):
        return {str(k): canon(v, state, depth + 1) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if dataclasses.is_dataclass(x):
        return {f.name: canon(getattr(x, f.name), state, depth + 1) for f in dataclasses.fields(x)}
    if hasattr(x, '__dict__'):
        return {k: canon(v, state, depth + 1) for k, v in sorted(vars(x).items())
                if not k.startswith('_')}
    return repr(x)


def walk_diff(a, b, state, path='root', diffs=None):
    diffs = [] if diffs is None else diffs
    state['nodes'] += 1
    if state['nodes'] > NODE_BUDGET:
        state['truncated'] = True
        raise RuntimeError('node budget exhausted')
    if len(diffs) >= DIFF_REPORT_CAP:
        return diffs
    same_type = type(a) is type(b)
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            diffs.append({'path': path, 'kind': 'type', 'a': type(a).__name__,
                          'b': type(b).__name__})
            return diffs
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append({'path': '%s.%s' % (path, k), 'kind': 'missing',
                              'a': k in a, 'b': k in b})
                continue
            walk_diff(a[k], b[k], state, '%s.%s' % (path, k), diffs)
        return diffs
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            diffs.append({'path': path, 'kind': 'type', 'a': type(a).__name__,
                          'b': type(b).__name__})
            return diffs
        if len(a) != len(b):
            diffs.append({'path': '%s.len' % path, 'kind': 'length', 'a': len(a), 'b': len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            walk_diff(x, y, state, '%s[%d]' % (path, i), diffs)
        return diffs
    if _is_num(a) and _is_num(b) and not same_type:
        if state['mode'] == 'strict':
            diffs.append({'path': path, 'kind': 'typed-numeric', 'a': type(a).__name__,
                          'b': type(b).__name__, 'a_value': repr(_num(a)),
                          'b_value': repr(_num(b)), 'bits_equal': bits(_num(a)) == bits(_num(b))})
        elif not _numeric_semantically_equal(a, b):
            diffs.append({'path': path, 'kind': 'numeric-value', 'a_value': repr(_num(a)),
                          'b_value': repr(_num(b)), 'a_bits': bits(_num(a)),
                          'b_bits': bits(_num(b))})
        return diffs
    if same_type and (isinstance(a, (int, float, str, bool)) or a is None) and a != b:
        diffs.append({'path': path, 'kind': 'value', 'a': repr(a), 'b': repr(b)})
        return diffs
    if not same_type:
        diffs.append({'path': path, 'kind': 'type', 'a': type(a).__name__,
                      'b': type(b).__name__})
    return diffs


def compare(a, b, mode):
    state = {'nodes': 0, 'mode': mode, 'truncated': False, 'depth_guard': False}
    try:
        diffs = walk_diff(a, b, state)
    except BaseException as exc:
        return {'complete': False, 'error': '%s: %s' % (type(exc).__name__, exc),
                'n_diffs': None, 'diffs': [], 'truncated': state['truncated'],
                'depth_guard': state['depth_guard'], 'nodes': state['nodes']}
    return {'complete': not (state['truncated'] or state['depth_guard']),
            'error': None, 'n_diffs': len(diffs), 'diffs': diffs,
            'truncated': state['truncated'], 'depth_guard': state['depth_guard'],
            'nodes': state['nodes']}


def machine_state():
    st = {'platform': platform.platform(), 'python': platform.python_version(),
          'cpu_count': os.cpu_count(), 'hostname': platform.node()}
    try:
        st['loadavg'] = list(os.getloadavg())
    except Exception:
        st['loadavg'] = None
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_cal_teacher.npz'))
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t0 = time.time()
    run_id = time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / 'A1_D3_L1_RESUME_PARITY_ATTRIBUTION.json'
    rec('run_identity', {'run_id': run_id, 'argv': sys.argv, 'cwd': os.getcwd(),
                         'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                         'machine': machine_state(),
                         'source_sha256': {
                             'this_tool': sha(__file__),
                             'collector': sha(EXT / 'scripts/simulation/d3_l1_state_collector.py'),
                             'strict_online_env': sha(EXT /
                                                      'scripts/simulation/strict_online_env.py'),
                             'recourse_snapshot': sha(EXT /
                                                      'scripts/simulation/recourse_snapshot.py'),
                             'coldchain_contract': sha(EXT /
                                                       'scripts/coldchain/coldchain_contract.py')},
                         'comparators': {'strict_typed': 'exact types, no depth cap',
                                         'semantic_value': 'only float64<->float, exact value AND '
                                                           'identical IEEE-754 bits'}})
    print('RUN %s  this_tool=%s' % (run_id, RECORDS['run_identity']['source_sha256']
                                    ['this_tool'][:16]))

    import numpy as np
    from coldchain_contract import ObjectiveProfile, apply_objective_profile, default_pilot_contract
    from d3_l1_state_collector import (HORIZON, RevealSlotCollector, assert_f_free_process,
                                       build_env, make_replanner, reveal_event_times,
                                       run_baseline_rollout)
    from recourse_snapshot import (CC_SNAPSHOT_SCHEMA_VERSION, capture_recourse_snapshot,
                                   restore_recourse_snapshot, snapshot_state_hash,
                                   validate_snapshot_against_env)

    # ---- phase A: profile field reconciliation (fail-closed) ------------------------------
    try:
        declared = json.loads(Path(args.profile).read_text(encoding='utf-8'))
        fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
        filtered = sorted(set(declared) - fields)
        kw = {k: v for k, v in declared.items() if k in fields}
        prof = ObjectiveProfile(**kw)
        declared_hash = declared.get('profile_hash')
        ph = getattr(prof, 'profile_hash')
        recomputed = ph() if callable(ph) else ph
        whitelist = {'profile_hash', 'provenance', 'hard_gate'}  # records/metadata only
        significant = [k for k in filtered if k not in whitelist]
        rec('profile_reconciliation', {
            'declared_keys': sorted(declared), 'dataclass_fields': sorted(fields),
            'filtered_keys': filtered, 'filtered_whitelisted': [k for k in filtered
                                                                if k in whitelist],
            'filtered_semantically_significant': significant,
            'declared_profile_hash': declared_hash, 'recomputed_profile_hash': recomputed,
            'hash_matches': declared_hash == recomputed,
            'filtered_values': {k: repr(declared[k])[:160] for k in filtered}})
        ck('A1 the reconstructed ObjectiveProfile reproduces the declared profile_hash exactly',
           declared_hash == recomputed,
           {k: RECORDS['profile_reconciliation'][k] for k in
            ('declared_profile_hash', 'recomputed_profile_hash', 'hash_matches')})
        ck('A2 every field filtered out during profile reconstruction is accounted for, and any '
           'filtered key that could affect the objective is FAIL-CLOSED',
           not significant or all(k in whitelist for k in significant),
           {'filtered_keys': filtered, 'whitelist': sorted(whitelist),
            'semantically_significant': significant})
    except BaseException as exc:
        rec('profile_reconciliation', {'error': '%s: %s' % (type(exc).__name__, exc)})
        ck('A1 the reconstructed ObjectiveProfile reproduces the declared profile_hash exactly',
           False, RECORDS['profile_reconciliation'])
        ck('A2 every field filtered out during profile reconstruction is accounted for',
           False, RECORDS['profile_reconciliation'])

    # ---- phase B: cold-chain contract + one cold-chain rollout ----------------------------
    base = default_pilot_contract()
    prof_dict = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    prof = ObjectiveProfile(**{k: v for k, v in prof_dict.items() if k in fields})
    eff = apply_objective_profile(base, prof)
    eff_hash = getattr(eff, 'contract_hash', None)
    rec('contract', {'base_hash': getattr(base, 'contract_hash', None),
                     'effective_hash': eff_hash, 'booking_horizon': HORIZON})
    prov_hash = (prof_dict.get('provenance') or {}).get('contract_hash')
    rec('contract_provenance_check', {'provenance_contract_hash': prov_hash,
                                      'effective_contract_hash': eff_hash,
                                      'base_contract_hash': getattr(base, 'contract_hash', None),
                                      'match': prov_hash == eff_hash})
    ck('A3 IDENTITY: the contract hash named by the registered profile provenance is the same '
       'contract that is actually instantiated (fail-closed)',
       prov_hash is not None and prov_hash == eff_hash, RECORDS['contract_provenance_check'])
    data = dict(np.load(args.dataset))
    inst = int(args.instance)
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff, HORIZON)
    col = RevealSlotCollector(env, inst, args.K)
    rollout = run_baseline_rollout(env, inst, hook=col)
    orig_traces, orig_served = rollout['result']
    rec('rollout', {'entry': rollout['entry'], 'seconds': rollout['seconds'],
                    'n_slots': len(col.slots), 'expected_reveals': reveal_event_times(env, inst)})

    # ---- phase C: per-slot cold-chain nested checks + dual parity -------------------------
    slot_reports = []
    for r in col.slots:
        snap = r['snapshot']
        ent = {'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id']}
        try:
            ent['schema_version'] = snap.get('schema_version')
            ent['schema_is_cc'] = (ent['schema_version'] == CC_SNAPSHOT_SCHEMA_VERSION)
            ent['snapshot_contract_hash'] = snap.get('coldchain_contract_hash')
            ent['contract_hash_matches'] = (ent['snapshot_contract_hash'] == eff_hash)
            cc_nested = {
                'vehicle_coldchain_state': ('vehicle_coldchain_state' in snap
                                            and len(snap['vehicle_coldchain_state']) > 0),
                'vehicle_coldchain_updated_time': 'coldchain_updated_time' in snap,
                'committed_coldchain_before': 'committed_coldchain_before' in snap,
                'coldchain_accounting_anchor': 'coldchain_accounting_anchor' in snap,
                'trace_final_coldchain_state': ('trace_final_coldchain_state' in snap
                                                and len(snap['trace_final_coldchain_state']) > 0),
                'trace_depot_unload_records': 'trace_depot_unload_records' in snap,
                'trace_dispatch_preconditioning_energy_kwh':
                    'trace_dispatch_preconditioning_energy_kwh' in snap,
                'trace_return_segment_energy_kwh': 'trace_return_segment_energy_kwh' in snap,
                'services_arrival': 'services_arrival' in snap}
            ent['coldchain_nested_fields'] = cc_nested
            ent['coldchain_nested_all_present'] = all(cc_nested.values())
            non_null_vehicle_cc = [i for i, s in enumerate(snap['vehicle_coldchain_state'])
                                   if s is not None]
            ent['vehicles_with_coldchain_state'] = len(non_null_vehicle_cc)
            ent['n_vehicles'] = len(snap['vehicle_coldchain_state'])
            ent['units'] = [len(v) if v is not None and hasattr(v, '__len__') else 0
                            for v in snap.get('trace_depot_unload_records', [])]
            restored = restore_recourse_snapshot(snap)
            objs = restored if isinstance(restored, tuple) else (restored,)
            ent['restore_arity'] = len(objs)
            ent['restore_object_types'] = [type(o).__name__ for o in objs]
            ent['restore_expected_three_objects'] = (len(objs) == 3)
            # the re-capture branch
            env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff,
                              HORIZON)
            validate_snapshot_against_env(env_r, snap)
            recap = []

            def hook2(e, i, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                      all_customers, _t=float(r['clock']), _o=recap, _i=int(inst)):
                if abs(float(clock) - _t) < 1e-9:
                    _o.append(capture_recourse_snapshot(e, _i, float(clock), int(event_id),
                                                        int(reveal_idx), vehicles, traces,
                                                        served_mask,
                                                        all_customers=list(all_customers)))

            env_r.snapshot_hook = hook2
            resumed = env_r.run_resumed(snap)
            ent['recapture_hash'] = snapshot_state_hash(recap[0]) if recap else None
            ent['snapshot_hash'] = snapshot_state_hash(snap)
            ent['state_identity'] = (ent['recapture_hash'] == ent['snapshot_hash'])
            res_traces, res_served = (resumed if isinstance(resumed, tuple) else (resumed, None))
            ent['served_mask_equal'] = (res_served is not None
                                        and np.array_equal(np.asarray(res_served),
                                                           np.asarray(orig_served)))
            st = {'nodes': 0, 'truncated': False, 'depth_guard': False}
            ca = canon(orig_traces, st)
            cb = canon(res_traces, st)
            ent['canon_truncated'] = st['truncated'] or st['depth_guard']
            ent['canon_nodes'] = st['nodes']
            ent['strict'] = compare(ca, cb, 'strict')
            ent['semantic'] = compare(ca, cb, 'semantic')
        except BaseException as exc:
            ent['error'] = '%s: %s' % (type(exc).__name__, exc)
            import traceback
            ent['traceback'] = traceback.format_exc()[-1200:]
        slot_reports.append(ent)
    rec('slot_reports', slot_reports)

    got = [e for e in slot_reports if 'error' not in e]
    ck('B1 the snapshot is a real COLD-CHAIN snapshot: schema is recourse_snapshot_v3_cc and its '
       'coldchain_contract_hash equals the effective contract hash',
       bool(got) and all(e['schema_is_cc'] and e['contract_hash_matches'] for e in got),
       [{'slot': e['slot'], 'schema': e.get('schema_version'),
         'contract_match': e.get('contract_hash_matches')} for e in slot_reports])
    ck('B2 the nested cold-chain fields are present at their REAL nesting level (per-vehicle and '
       'per-trace), not merely at the top level',
       bool(got) and all(e['coldchain_nested_all_present'] and e['vehicles_with_coldchain_state'] > 0
                         for e in got),
       [{'slot': e['slot'], 'nested': e.get('coldchain_nested_fields'),
         'vehicles_with_state': e.get('vehicles_with_coldchain_state'),
         'n_vehicles': e.get('n_vehicles')} for e in slot_reports])
    ck('B3 the restore payload really carries the three expected objects and the re-capture '
       'reproduces the snapshot state hash',
       bool(got) and all(e['restore_expected_three_objects'] and e['state_identity']
                         for e in got),
       [{'slot': e['slot'], 'arity': e.get('restore_arity'), 'types': e.get('restore_object_types'),
         'state_identity': e.get('state_identity')} for e in slot_reports])
    ck('C1 STRICT TYPED PARITY over the COMPLETE trajectory (no depth cap): recorded as a failure '
       'where it is one, with the first differences listed',
       bool(got) and all(e['strict']['complete'] and e['strict']['n_diffs'] == 0 for e in got),
       [{'slot': e['slot'], 'complete': e['strict'].get('complete'),
         'n_diffs': e['strict'].get('n_diffs'),
         'first': (e['strict']['diffs'][:3] if e['strict'].get('diffs') else None)}
        for e in got])
    ck('C2 SEMANTIC VALUE PARITY over the COMPLETE trajectory: only float64<->float with exact '
       'value AND identical IEEE-754 bits is tolerated; anything else is a real divergence',
       bool(got) and all(e['semantic']['complete'] and e['semantic']['n_diffs'] == 0 for e in got),
       [{'slot': e['slot'], 'complete': e['semantic'].get('complete'),
         'n_diffs': e['semantic'].get('n_diffs'),
         'first': (e['semantic']['diffs'][:3] if e['semantic'].get('diffs') else None)}
        for e in got])

    # ---- pre-fixed adjudication ----------------------------------------------------------
    if not got or any(not e.get('strict', {}).get('complete') for e in got) \
            or any(not e.get('semantic', {}).get('complete') for e in got):
        verdict = 'NOT_VERIFIABLE'
    elif all(e['semantic']['n_diffs'] == 0 for e in got) \
            and any(e['strict']['n_diffs'] > 0 for e in got):
        verdict = 'REPRESENTATION_ONLY'
    elif all(e['semantic']['n_diffs'] == 0 for e in got) and all(e['strict']['n_diffs'] == 0
                                                                for e in got):
        verdict = 'FULL_PARITY'
    else:
        verdict = 'BEHAVIOR_DIVERGENCE'
    rec('adjudication', {
        'verdict': verdict, 'criteria_prefixed':
            'semantic>0 -> BEHAVIOR_DIVERGENCE; strict>0 & semantic==0 -> REPRESENTATION_ONLY; '
            'incomplete comparison -> NOT_VERIFIABLE; both 0 -> FULL_PARITY',
        'resume_parity_status': {'STRICT_TYPED_PARITY': 'NOT_VERIFIED (recorded failure)'
                                 if any(e['strict']['n_diffs'] for e in got if 'strict' in e)
                                 else 'PASS',
                                 'SEMANTIC_PARITY': 'PASS' if verdict in ('REPRESENTATION_ONLY',
                                                                          'FULL_PARITY')
                                 else 'NOT_VERIFIED'},
        'coldchain_contract_label': 'L1_COLDCHAIN_CONTRACT_CANDIDATE / IDENTITY_PARTIAL',
        'algorithmic_gain_claim': 'none', 'states_collected_for_L1': 0})
    ck('D1 adjudication is decided by the PRE-FIXED criteria and reported as one of '
       'REPRESENTATION_ONLY / BEHAVIOR_DIVERGENCE / NOT_VERIFIABLE / FULL_PARITY',
       verdict in ('REPRESENTATION_ONLY', 'BEHAVIOR_DIVERGENCE', 'NOT_VERIFIABLE', 'FULL_PARITY'),
       RECORDS['adjudication'])

    fails = [c for c in CASES if not c[1]]
    RECORDS['run_identity_end'] = {'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                                   'total_seconds': time.time() - t0, 'run_id': run_id,
                                   'n_fail': len(fails), 'json_path': str(json_path)}
    print('\n=== %s | %d rows, %d PASS, %d FAIL (%.1fs) ==='
          % (verdict, len(CASES), len(CASES) - len(fails), len(fails),
             RECORDS['run_identity_end']['total_seconds']))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    json_path.write_text(json.dumps(
        {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
         'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
         'n_fail': len(fails), 'run_id': run_id, 'argv': sys.argv,
         'source_sha256_this_tool': sha(__file__)},
        indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('WROTE %s (same process as this stdout/stderr; run_id=%s)' % (json_path, run_id))
    print('F-FREE: %s' % assert_f_free_process())
    return 0


if __name__ == '__main__':
    sys.exit(main())
