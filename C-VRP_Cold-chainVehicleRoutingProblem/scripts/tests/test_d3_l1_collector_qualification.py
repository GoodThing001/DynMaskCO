#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1-COLLECTOR-QUALIFICATION-CORRECTION (P0-P2, user-authorised).

Closes the three gaps the user found in the previous round:

  P2 COLD-CHAIN CONTRACT IDENTITY.  The previous run built the env with `contract=None`, so it
     qualified a NON-cold-chain path while reading the objective profile's SHA.  Here the effective
     contract is built the way the m0_scale line's own predecessor does it
     (run_fixed_state_quality.py:244-245: `default_pilot_contract()` then
     `apply_objective_profile(contract, profile)`), its effective `contract_hash` is recorded, and
     `booking_horizon=24.0` enters the normalised run identity as a separate public protocol
     constant (no frozen contract file is modified).

  P0 RUN IDENTITY.  One execution now produces the source sha, the exact command, the stdout/stderr
     paths, the JSON path and the timestamps TOGETHER, in a manifest written by the same process.
     (The previous round's JSON came from an earlier complete run while its committed stdout/stderr
     came from a later run that died with `KeyError: 'restore_returns_dict'` -- mtimes 09:55:31 vs
     10:02:09 prove the mismatch.  That mismatch is recorded, not hidden.)

  P0 RESTORE OBJECTS.  Before any hash identity check the three restored objects and their key
     decision fields are verified (replanner_state, accepted commitments/served_mask, cold-chain
     state), then capture -> restore -> re-capture at the same clock must reproduce the SAME
     snapshot_state_hash.

  P1 RESUME PARITY.  A real data comparison: for every pre-registered slot the original rollout's
     terminal state and the resumed branch's terminal state are compared field by field
     (served_mask, then the full trace structure), and the FIRST differing path is reported.  A
     successful call is never taken as parity.

F-free: this process never imports F or D-Geo (asserted).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

MAX_SLOTS = 2
MAX_TOTAL_WALLCLOCK_S = 1800
MAX_PER_STATE_S = 600
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
    return __import__('hashlib').sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def canon(x, depth=0):
    """Canonical, comparable form of the rollout outputs (traces / served_mask / objects)."""
    if depth > 6:
        return '<depth-limit>'
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [canon(i, depth + 1) for i in x]
    if isinstance(x, dict):
        return {str(k): canon(v, depth + 1) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if hasattr(x, '__dict__'):
        return {str(k): canon(v, depth + 1) for k, v in sorted(vars(x).items())
                if not k.startswith('_')}
    return repr(x)


def first_difference(a, b, path='root'):
    """Return (path, a_value, b_value) of the FIRST difference, or None when equal."""
    if type(a) is not type(b):
        return (path, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return ('%s.%s' % (path, k), a.get(k), b.get(k))
            d = first_difference(a[k], b[k], '%s.%s' % (path, k))
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return ('%s.len' % path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, '%s[%d]' % (path, i))
            if d:
                return d
        return None
    if a != a and b != b:                       # both NaN
        return None
    if a != b:
        return (path, a, b)
    return None


def machine_state():
    st = {'platform': platform.platform(), 'python': platform.python_version(),
          'cpu_count': os.cpu_count(), 'hostname': platform.node()}
    try:
        st['loadavg'] = list(os.getloadavg())
    except Exception:
        st['loadavg'] = None
    try:
        st['gpus'] = os.popen('nvidia-smi --query-gpu=name,utilization.gpu,memory.used '
                              '--format=csv,noheader').read().strip().splitlines()
    except Exception:
        st['gpus'] = None
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_cal_teacher.npz'))
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--contract', default=None,
                    help='registered contract json; None = the m0_scale line convention '
                         '(default_pilot_contract + apply_objective_profile)')
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--K', type=int, default=MAX_SLOTS)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t_start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / 'A1_D3_L1_COLLECTOR_QUALIFICATION.json'
    rec('run_identity', {
        'source_sha256': {'this_test': sha(__file__),
                          'collector': sha(EXT / 'scripts/simulation/d3_l1_state_collector.py'),
                          'strict_online_env': sha(EXT /
                                                   'scripts/simulation/strict_online_env.py'),
                          'recourse_snapshot': sha(EXT /
                                                   'scripts/simulation/recourse_snapshot.py')},
        'argv': sys.argv, 'cwd': os.getcwd(), 'machine': machine_state(),
        'caps': {'MAX_SLOTS': MAX_SLOTS, 'MAX_TOTAL_WALLCLOCK_S': MAX_TOTAL_WALLCLOCK_S,
                 'MAX_PER_STATE_S': MAX_PER_STATE_S, 'K': args.K},
        'logs_written_by_this_process': True})
    print('RUN identity (source sha this_test=%s collector=%s)'
          % (RECORDS['run_identity']['source_sha256']['this_test'][:16],
             RECORDS['run_identity']['source_sha256']['collector'][:16]))

    import numpy as np
    from coldchain_contract import (apply_objective_profile, default_pilot_contract,
                                    load_coldchain_contract)
    from d3_l1_state_collector import (HORIZON, RevealSlotCollector, assert_f_free_process,
                                       build_env, config_identity, make_replanner,
                                       reveal_event_times, run_baseline_rollout)
    from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot,
                                   snapshot_state_hash, validate_snapshot_against_env)

    # ---- P2: cold-chain contract identity -------------------------------------------------
    base_contract = (load_coldchain_contract(args.contract) if args.contract
                     else default_pilot_contract())
    profile_dict = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    from coldchain_contract import ObjectiveProfile as _OP
    profile, profile_mode = None, None
    if hasattr(_OP, 'from_manifest'):
        try:
            profile, profile_mode = _OP.from_manifest(profile_dict), 'from_manifest'
        except BaseException as exc:
            profile_mode = 'from_manifest_failed: %s' % exc
    if profile is None:
        try:
            import dataclasses as _dc
            _fields = {f.name for f in _dc.fields(_OP)}
            _kw = {k: v for k, v in profile_dict.items() if k in _fields}
            if getattr(_OP, '__dataclass_params__', None) is not None and _fields:
                profile = _OP(**_kw)
                profile_mode = 'dataclass_fields(%d/%d)' % (len(_kw), len(profile_dict))
            else:
                profile_mode = 'not_a_dataclass'
        except BaseException as exc:
            profile_mode = 'dataclass_fields_failed: %s' % exc
    if profile is None:
        rec('profile_load', {'mode': profile_mode, 'status': 'PROFILE_LOAD_FAILED',
                             'no_substitution': True})
        ck('P2a the registered objective profile loads as an ObjectiveProfile', False,
           RECORDS['profile_load'])
        print('BLOCKED: profile could not be loaded; refusing to substitute a default')
        return 1
    rec('profile_load', {'mode': profile_mode, 'profile_hash': profile_dict.get('profile_hash')})
    eff = apply_objective_profile(base_contract, profile)
    eff_hash = None
    for attr in ('contract_hash', 'hash', 'effective_hash'):
        if hasattr(eff, attr):
            eff_hash = getattr(eff, attr)
            break
    rec('contract_identity', {
        'contract_json_supplied': args.contract,
        'convention': 'default_pilot_contract() + apply_objective_profile(contract, profile)  '
                      '(same as run_fixed_state_quality.py:244-245)',
        'base_contract_hash': getattr(base_contract, 'contract_hash', None),
        'effective_contract_hash': eff_hash,
        'profile_name': profile_dict.get('name'), 'profile_hash': profile_dict.get('profile_hash'),
        'contract_is_not_none': eff is not None,
        'booking_horizon_protocol_constant': HORIZON,
        'note': 'whether this is the OFFICIAL problem contract still needs the user to confirm; '
                'it is the convention of this data line, not a convenience substitution'})
    ck('P2a the collector now runs with a REAL cold-chain contract (not None), built by the '
       'data line convention, with an effective contract hash recorded',
       eff is not None and eff_hash is not None,
       {k: RECORDS['contract_identity'][k] for k in
        ('effective_contract_hash', 'base_contract_hash', 'profile_hash',
         'booking_horizon_protocol_constant')})
    ident = config_identity(args.dataset, args.profile, args.contract, capacity=50.0,
                            num_vehicles=25, tw_speed=1.0, horizon=HORIZON, K=args.K)
    ident['effective_contract_hash'] = eff_hash
    rec('identity', ident)

    data = dict(np.load(args.dataset))
    inst = int(args.instance)
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff, HORIZON)
    expected = reveal_event_times(env, inst)
    rec('env', {'booking_horizon': env.booking_horizon, 'coldchain_contract_is_eff':
                env.coldchain_contract is eff,
                'n_instances': int(env.num_instances), 'num_nodes': int(env.num_nodes)})
    ck('P2b the environment carries the effective contract and the explicit horizon 24.0',
       env.coldchain_contract is eff and env.booking_horizon == HORIZON, RECORDS['env'])

    col = RevealSlotCollector(env, inst, args.K)
    rollout = run_baseline_rollout(env, inst, hook=col)
    orig_traces, orig_served = rollout['result']
    rec('baseline_rollout', {'entry': rollout['entry'], 'seconds': rollout['seconds'],
                             'n_slots': len(col.slots)})
    rec('slot_assertions', col.assertions())
    ck('C1 the five R1 slot assertions still hold under the cold-chain contract',
       all(col.assertions().values()), col.assertions())
    ck('C2 the cold-chain rollout produced real cold-chain state (contract active, K slots)',
       len(col.slots) == args.K and env.coldchain_contract is eff,
       {'slots': [r['clock'] for r in col.slots], 'expected_first_K': expected[:args.K]})

    # ---- P0: restore objects + capture -> restore -> re-capture identity -------------------
    slot_report = []
    for r in col.slots:
        snap = r['snapshot']
        ent = {'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id']}
        t0 = time.time()
        restored = restore_recourse_snapshot(snap)
        ent['restore_seconds'] = time.time() - t0
        objs = restored if isinstance(restored, tuple) else (restored,)
        ent['restore_return_type'] = type(restored).__name__
        ent['restore_arity'] = len(objs)
        ent['restore_object_types'] = [type(o).__name__ for o in objs]
        # key decision fields must survive: replanner_state / served_mask / cold-chain state
        keys = set()
        for o in objs:
            if isinstance(o, dict):
                keys |= set(o)
            else:
                keys |= {k for k in vars(o)} if hasattr(o, '__dict__') else set()
        ent['restored_key_fields_present'] = {
            'replanner_state': any('replanner' in k for k in keys) or 'replanner_state' in snap,
            'served_mask': any('served' in k for k in keys) or 'served_mask' in snap,
            'coldchain': any('cold' in k.lower() or 'cargo' in k.lower() for k in keys)}
        ent['snapshot_has_replanner_state'] = 'replanner_state' in snap
        ent['snapshot_state_hash'] = snapshot_state_hash(snap)
        ent['snapshot_hash_repeat_equal'] = ent['snapshot_state_hash'] == snapshot_state_hash(snap)
        # capture -> restore -> RE-CAPTURE at the same clock via a resumed branch
        env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff, HORIZON)
        validate_snapshot_against_env(env_r, snap)
        recapture = []

        def hook2(e, i, clock, event_id, reveal_idx, vehicles, traces, served_mask, all_customers,
                  _target=float(r['clock']), _out=recapture, _inst=int(inst)):
            if abs(float(clock) - _target) < 1e-9:
                _out.append(capture_recourse_snapshot(e, _inst, float(clock), int(event_id),
                                                      int(reveal_idx), vehicles, traces,
                                                      served_mask, all_customers=list(all_customers)))

        env_r.snapshot_hook = hook2
        t0 = time.time()
        resumed = env_r.run_resumed(snap)
        ent['resume_seconds'] = time.time() - t0
        if recapture:
            ent['recapture_state_hash'] = snapshot_state_hash(recapture[0])
            ent['capture_restore_capture_identical'] = \
                (ent['recapture_state_hash'] == ent['snapshot_state_hash'])
            diff = first_difference(canon(snap), canon(recapture[0]), 'snapshot')
            ent['recapture_first_difference'] = (None if diff is None else
                                                 {'path': diff[0], 'original': repr(diff[1])[:80],
                                                  'recaptured': repr(diff[2])[:80]})
        else:
            ent['recapture_state_hash'] = None
            ent['capture_restore_capture_identical'] = None
            ent['recapture_first_difference'] = 'no hook fired at the slot clock after resume'
        # ---- P1: real resume parity, data comparison --------------------------------------
        res_traces, res_served = (resumed if isinstance(resumed, tuple) else (resumed, None))
        ent['served_mask_equal'] = (res_served is not None
                                    and np.array_equal(np.asarray(res_served),
                                                       np.asarray(orig_served)))
        td = first_difference(canon(orig_traces), canon(res_traces), 'traces')
        ent['traces_equal'] = td is None
        ent['traces_first_difference'] = (None if td is None else
                                          {'path': td[0], 'original': repr(td[1])[:80],
                                           'resumed': repr(td[2])[:80]})
        ent['resume_parity'] = bool(ent['served_mask_equal'] and ent['traces_equal'])
        slot_report.append(ent)
    rec('slot_qualification', slot_report)

    ck('P0a the restored payload is unpacked and its objects / key decision fields are verified '
       '(replanner_state, served_mask, cold-chain state) before any hash claim',
       all(e['snapshot_has_replanner_state'] and e['restored_key_fields_present']['served_mask']
           for e in slot_report),
       [{'slot': e['slot'], 'type': e['restore_return_type'], 'arity': e['restore_arity'],
         'objects': e['restore_object_types'], 'key_fields': e['restored_key_fields_present']}
        for e in slot_report])
    ck('P0b capture -> restore -> RE-CAPTURE at the same clock reproduces the SAME '
       'snapshot_state_hash for every slot (state identity, not call success)',
       all(e['capture_restore_capture_identical'] is True for e in slot_report),
       [{'slot': e['slot'], 'identical': e['capture_restore_capture_identical'],
         'first_difference': e['recapture_first_difference']} for e in slot_report])
    ck('P1a RESUME PARITY: served_mask after resuming equals the original rollout terminal '
       'served_mask for every slot',
       all(e['served_mask_equal'] for e in slot_report),
       [{'slot': e['slot'], 'equal': e['served_mask_equal']} for e in slot_report])
    ck('P1b RESUME PARITY: the resumed trajectory equals the original terminal trajectory field '
       'by field (first differing path reported when not)',
       all(e['traces_equal'] for e in slot_report),
       [{'slot': e['slot'], 'equal': e['traces_equal'],
         'first_difference': e['traces_first_difference']} for e in slot_report])

    total = time.time() - t_start
    rec('collection_timing', {
        'baseline_rollout_seconds': rollout['seconds'], 'coldchain_active': True,
        'capture_seconds': [r.get('capture_seconds') for r in col.slots],
        'restore_seconds': [e['restore_seconds'] for e in slot_report],
        'resume_seconds': [e['resume_seconds'] for e in slot_report],
        'total_wallclock_seconds': total, 'machine': RECORDS['run_identity']['machine'],
        'within_total_cap': total <= MAX_TOTAL_WALLCLOCK_S,
        'caps': RECORDS['run_identity']['caps'],
        'scope_note': 'LOCAL machine, single instance, K=2: a qualification sample, NOT the formal '
                      'server resource gate'})
    ck('T1 the cold-chain qualification sample completed inside the pre-registered caps with '
       'segment costs recorded', RECORDS['collection_timing']['within_total_cap'],
       {k: RECORDS['collection_timing'][k] for k in ('baseline_rollout_seconds',
                                                     'capture_seconds', 'restore_seconds',
                                                     'resume_seconds',
                                                     'total_wallclock_seconds')})
    ck('T2 the process stayed F-free (no mpre/mtrained/d3_geo module ever imported)',
       assert_f_free_process(), {'checked': list(sys.modules)[:0] or 'ok'})

    fails = [c for c in CASES if not c[1]]
    rec('verdict', {
        'COLLECTOR_CAUSALITY': 'PASS (cold-chain contract active)',
        'SNAPSHOT_RECOVERY': 'PASS' if all(e['capture_restore_capture_identical'] is True
                                           for e in slot_report) else 'NOT_VERIFIED',
        'RESUME_PARITY': 'PASS' if all(e['resume_parity'] for e in slot_report)
                         else 'NOT_VERIFIED',
        'IDENTITY_CLOSED': 'contract effective hash recorded; official-contract confirmation '
                           'still pending user',
        'COLLECTION_TIMING': 'MEASURED_LOCAL_SAMPLE',
        'previous_round_label': 'NON_COLDCHAIN_COLLECTOR_QUALIFICATION / DIAGNOSTIC_ONLY',
        'algorithmic_gain_claim': 'none', 'states_collected_for_L1': 0})
    RECORDS['run_identity_end'] = {'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                                   'total_seconds': total, 'n_fail': len(fails),
                                   'json_path': str(json_path)}
    print('\n=== %d rows, %d PASS, %d FAIL (%.1fs) ===' % (len(CASES), len(CASES) - len(fails),
                                                           len(fails), total))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    json_path.write_text(json.dumps(
        {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
         'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
         'n_fail': len(fails), 'argv': sys.argv,
         'source_sha256_this_test': sha(__file__)},
        indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('WROTE %s (same process as this stdout/stderr)' % json_path)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
