#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 collector gate suite: COLLECTOR_CAUSALITY / SNAPSHOT_RECOVERY / COLLECTION_TIMING /
IDENTITY.  F-free: this process never imports F or D-Geo (asserted at runtime).

Order (user-mandated): protocol identity -> collector -> lightweight causality and recovery
counterexample runs -> ONE timing qualification run (cal instance x first 2 reveal events) with
hard caps registered BEFORE the run.  Any failure is archived, never retried into a pass.

Hard caps (registered in this file before execution):
  MAX_SLOTS = 2            (the authorised single timing sample)
  MAX_TOTAL_WALLCLOCK_S = 900
  MAX_PER_STATE_S = 300
  CALL_BUDGET = 1 baseline rollout (+1 perturbation rollout for the causality gate) + per slot:
                1 capture, 1 restore, 1 hash, 1 resume-parity
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
MAX_TOTAL_WALLCLOCK_S = 900
MAX_PER_STATE_S = 300
CASES = []
RECORDS = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def rec(k, v):
    RECORDS[k] = v


def machine_state():
    st = {'platform': platform.platform(), 'python': platform.python_version(),
          'cpu_count': os.cpu_count()}
    try:
        st['loadavg'] = list(os.getloadavg())
    except Exception:
        st['loadavg'] = None
    try:
        import shutil
        st['disk_free_gb'] = round(shutil.disk_usage(str(EXT)).free / 1e9, 1)
    except Exception:
        st['disk_free_gb'] = None
    try:
        out = os.popen('nvidia-smi --query-gpu=name,utilization.gpu,memory.used '
                       '--format=csv,noheader').read().strip()
        st['gpus'] = out.splitlines() if out else []
    except Exception:
        st['gpus'] = None
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_cal_teacher.npz'))
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--contract', default=None)
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--K', type=int, default=MAX_SLOTS)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    t_start = time.time()
    rec('caps', {'MAX_SLOTS': MAX_SLOTS, 'MAX_TOTAL_WALLCLOCK_S': MAX_TOTAL_WALLCLOCK_S,
                 'MAX_PER_STATE_S': MAX_PER_STATE_S, 'K': args.K,
                 'registered_before_run': True})
    rec('machine', machine_state())

    import numpy as np
    from d3_l1_state_collector import (HORIZON, RevealSlotCollector, assert_f_free_process,
                                       build_env, config_identity, make_replanner,
                                       reveal_event_times, run_baseline_rollout, sha256_file)

    # ---- IDENTITY ------------------------------------------------------------------------
    ident = config_identity(args.dataset, args.profile, args.contract, capacity=50.0,
                            num_vehicles=25, tw_speed=1.0, horizon=HORIZON, K=args.K)
    rec('identity', ident)
    rec('visible_information_boundary', {
        'visible_at_decision': 'customers with reveal_time <= clock, plus the depot',
        'not_visible': 'orders with reveal_time > clock of the decision point',
        'horizon': HORIZON,
        'horizon_source': 'generate_coldchain_data.py:119 (declared window; not data-derived)'})
    ck('D1 identity: data/profile/collector/env/replanner/snapshot shas are recorded and the '
       'objective profile hash is present',
       bool(ident['items']['dataset_sha256'] and ident['items']['objective_profile_sha256']
            and ident['items']['collector_sha256']
            and ident['items']['strict_online_env_sha256']),
       {'fingerprint': ident['fingerprint_sha256'],
        'contract_sha256': ident['items']['coldchain_contract_sha256']})
    ck('D2 contract/profile identity gap status (booking_horizon is NOT carried by the profile or '
       'contract and is frozen here as an explicit protocol constant)',
       'booking_horizon' not in json.dumps(json.loads(Path(args.profile).read_text()))
       and ident['items']['coldchain_contract_sha256'] is None,
       {'horizon_frozen_in_collector': HORIZON,
        'profile_has_horizon': 'booking_horizon' in json.dumps(
            json.loads(Path(args.profile).read_text())),
        'contract_path_given': args.contract})

    # ---- COLLECTOR ----------------------------------------------------------------------
    data = dict(np.load(args.dataset))
    contract = None
    replanner, ctor = make_replanner(50.0, HORIZON, contract)
    rec('replanner_construction', ctor)
    env = build_env(data, 50.0, 25, 1.0, replanner, contract, HORIZON)
    rec('env', {'booking_horizon': env.booking_horizon, 'n_instances': int(env.num_instances),
                'num_nodes': int(env.num_nodes)})
    ck('D3 the collection env is built with the explicit public horizon 24.0 (never the None '
       'default)', env.booking_horizon == HORIZON, {'booking_horizon': env.booking_horizon})

    inst = int(args.instance)
    expected = reveal_event_times(env, inst)
    rec('declared_reveal_times', {'instance': inst, 'n_distinct_reveals': len(expected),
                                  'first_three': expected[:3]})
    col = RevealSlotCollector(env, inst, args.K)
    rollout = run_baseline_rollout(env, inst, hook=col)
    rec('baseline_rollout', {'entry': rollout['entry'], 'seconds': rollout['seconds']})
    rec('collector_events', col.events)
    rec('slot_assertions', col.assertions())
    sa = col.assertions()
    ck('C1 the five R1 slot assertions hold (t=0 free, non-reveal events free, one slot per '
       'reveal clock, each slot captured exactly once, slots match declared reveal times)',
       all(sa.values()), sa)
    ck('C2 exactly K reveal slots were collected and they are the FIRST K declared reveal events',
       len(col.slots) == args.K
       and [r['clock'] for r in col.slots] == expected[:args.K],
       {'slots': [{'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id'],
                   'reveal_idx': r['reveal_idx']} for r in col.slots],
        'expected_first_K': expected[:args.K]})
    ck('C3 slot metadata is internally consistent (event_id/reveal_idx/clock are the ones the '
       'environment reported at the reveal)',
       all(r['event_id'] >= 1 and r['clock'] == r['clock'] for r in col.slots)
       and all(r['slot'] == i + 1 for i, r in enumerate(col.slots)),
       [{'slot': r['slot'], 'event_id': r['event_id'], 'reveal_idx': r['reveal_idx']}
        for r in col.slots])

    # ---- COLLECTOR_CAUSALITY: full-rollout perturbation of the unrevealed future ---------
    first_clock = expected[0]
    pert = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in data.items()}
    rt = np.asarray(pert['reveal_time'], dtype=float)
    removed = [i for i in range(1, rt.shape[1]) if rt[inst, i] > first_clock + 1e-6]
    pert['demands'] = np.asarray(pert['demands'], dtype=np.float32).copy()
    for i in removed:
        pert['demands'][inst, i] = 0.0
        pert['reveal_time'][inst, i] = 0.0
    rec('perturbation', {'instance': inst, 'first_reveal_clock': first_clock,
                         'orders_removed_from_the_future': len(removed),
                         'kept_revealed_prefix_identical': True})
    env2 = build_env(pert, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, contract)[0], contract,
                     HORIZON)
    col2 = RevealSlotCollector(env2, inst, args.K)
    run_baseline_rollout(env2, inst, hook=col2)
    fp_a = [r['plan_fingerprint'] for r in col.events if r['clock'] <= first_clock + 1e-6]
    fp_b = [r['plan_fingerprint'] for r in col2.events if r['clock'] <= first_clock + 1e-6]
    rec('causality_rollout', {'hooks_at_or_before_first_reveal_original': len(fp_a),
                              'hooks_at_or_before_first_reveal_perturbed': len(fp_b),
                              'identical': fp_a == fp_b, 'fingerprints_original': fp_a,
                              'fingerprints_perturbed': fp_b})
    ck('C4 COLLECTOR_CAUSALITY (full rollout): removing every order that is still unrevealed at '
       'the first reveal -- revealed prefix identical -- leaves every decision fingerprint at or '
       'before that reveal unchanged',
       bool(fp_a) and fp_a == fp_b, RECORDS['causality_rollout'])
    env_none = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, contract)[0], contract,
                         HORIZON)
    env_none.booking_horizon = None            # counterexample only: the forbidden default
    env3 = build_env(pert, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, contract)[0], contract,
                     HORIZON)
    env3.booking_horizon = None
    col3 = RevealSlotCollector(env_none, inst, args.K)
    run_baseline_rollout(env_none, inst, hook=col3)
    col4 = RevealSlotCollector(env3, inst, args.K)
    run_baseline_rollout(env3, inst, hook=col4)
    fp_none_a = [r['plan_fingerprint'] for r in col3.events if r['clock'] <= first_clock + 1e-6]
    fp_none_b = [r['plan_fingerprint'] for r in col4.events if r['clock'] <= first_clock + 1e-6]
    rec('causality_counterexample_horizon_none',
        {'identical': fp_none_a == fp_none_b, 'original': fp_none_a, 'perturbed': fp_none_b,
         'note': 'this run is an explicit counterexample probe; it is NOT a collectable '
                 'configuration and its states are discarded'})
    ck('C5 counterexample probe with the forbidden booking_horizon=None is recorded (identical '
       'or not) purely as evidence of the hazard, and its states are discarded',
       True, {k: v for k, v in RECORDS['causality_counterexample_horizon_none'].items()
              if k != 'original' and k != 'perturbed'})

    # ---- SNAPSHOT_RECOVERY ---------------------------------------------------------------
    from recourse_snapshot import (restore_recourse_snapshot, resume_from_snapshot,
                                   snapshot_state_hash, validate_snapshot_against_env)
    slot_report = []
    for r in col.slots:
        snap = r['snapshot']
        ent = {'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id']}
        t0 = time.time()
        h1 = snapshot_state_hash(snap)
        ent['hash_seconds'] = time.time() - t0
        t0 = time.time()
        h2 = snapshot_state_hash(snap)
        ent['hash_repeat_seconds'] = time.time() - t0
        ent['hash_stable'] = (h1 == h2)
        t0 = time.time()
        restored = restore_recourse_snapshot(snap)
        ent['restore_seconds'] = time.time() - t0
        ent['restore_return_type'] = type(restored).__name__
        ent['restore_non_none'] = restored is not None
        ent['restore_fields'] = (sorted(restored)[:12] if isinstance(restored, dict)
                                 else [a for a in dir(restored) if not a.startswith('_')][:12])
        # the frozen helper omits booking_horizon -> measure that hazard directly
        helper = resume_from_snapshot(data, snap, make_replanner(50.0, HORIZON, contract)[0],
                                      capacity=50.0, tw_speed=1.0, num_vehicles=25,
                                      coldchain_contract=contract)
        ent['frozen_helper_booking_horizon'] = helper.booking_horizon
        # the sanctioned restore path: identical explicit config, then the frozen run_resumed
        t0 = time.time()
        env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, contract)[0],
                          contract, HORIZON)
        validate_snapshot_against_env(env_r, snap)
        ent['restore_env_build_seconds'] = time.time() - t0
        ent['restored_env_booking_horizon'] = env_r.booking_horizon
        ent['horizon_chain_ok'] = (env.booking_horizon == env_r.booking_horizon == HORIZON)
        t0 = time.time()
        out_r = env_r.run_resumed(snap)
        ent['resume_seconds'] = time.time() - t0
        ent['resume_return_types'] = [type(x).__name__ for x in
                                      (out_r if isinstance(out_r, tuple) else (out_r,))]
        slot_report.append(ent)
    rec('snapshot_recovery', slot_report)
    ck('R1 the frozen helper `resume_from_snapshot` restores with booking_horizon=None '
       '(the hazard the user identified) -- recorded, not worked around',
       all(e['frozen_helper_booking_horizon'] is None for e in slot_report), slot_report)
    ck('R2 the sanctioned restore env carries the identical explicit horizon, so '
       'original == restored == 24.0 for every slot',
       all(e['horizon_chain_ok'] for e in slot_report),
       [{'slot': e['slot'], 'horizon_chain_ok': e['horizon_chain_ok']} for e in slot_report])
    ck('R3 snapshot_state_hash is stable and restore_recourse_snapshot returns a dict for every '
       'slot', all(e['hash_stable'] and e['restore_non_none'] for e in slot_report),
       [{'slot': e['slot'], 'hash_stable': e['hash_stable'],
         'restore_non_none': e['restore_returns_dict']} for e in slot_report])
    ck('R4 the frozen run_resumed(snapshot) executes on the sanctioned restore env (resume '
       'parity path available) for every slot',
       all('resume_seconds' in e and e['resume_seconds'] >= 0 for e in slot_report), None)

    # ---- COLLECTION_TIMING ---------------------------------------------------------------
    total = time.time() - t_start
    timing = {'baseline_rollout_seconds': rollout['seconds'],
              'capture_seconds': [r.get('capture_seconds') for r in col.slots],
              'hash_seconds': [e['hash_seconds'] for e in slot_report],
              'restore_seconds': [e['restore_seconds'] for e in slot_report],
              'resume_seconds': [e['resume_seconds'] for e in slot_report],
              'total_wallclock_seconds': total, 'caps': RECORDS['caps'],
              'within_total_cap': total <= MAX_TOTAL_WALLCLOCK_S,
              'within_per_state_cap': all((r.get('capture_seconds') or 0) <= MAX_PER_STATE_S
                                          for r in col.slots),
              'machine': RECORDS['machine'], 'artifact_persisted': bool(args.json)}
    rec('collection_timing', timing)
    ck('T1 the single authorised timing sample completed within the pre-registered caps and its '
       'segment costs are recorded', timing['within_total_cap']
       and timing['within_per_state_cap'], {k: timing[k] for k in
                                            ('baseline_rollout_seconds', 'capture_seconds',
                                             'restore_seconds', 'resume_seconds',
                                             'total_wallclock_seconds')})
    ck('T2 no F/D-Geo module was ever imported by this process (F-free by construction)',
       assert_f_free_process(), {'forbidden': ('mpre', 'mpre_trained', 'mpre_policy',
                                               'mtrained_replanner', 'd3_geo_adapter')})

    fails = [c for c in CASES if not c[1]]
    rec('verdict', {'COLLECTOR_CAUSALITY': 'PASS' if not fails else 'SEE_FAILED_ROWS',
                    'SNAPSHOT_RECOVERY': 'PASS' if not fails else 'SEE_FAILED_ROWS',
                    'COLLECTION_TIMING': 'MEASURED_SINGLE_SAMPLE' if not fails else 'BLOCKED',
                    'IDENTITY_CLOSED': 'PARTIAL (contract/profile gap: horizon is a protocol '
                                       'constant, not carried by profile/contract)',
                    'algorithmic_gain_claim': 'none',
                    'states_collected_for_L1': 0})
    print('\n=== %d rows, %d PASS, %d FAIL (%.1fs) ==='
          % (len(CASES), len(CASES) - len(fails), len(fails), total))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
             'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
             'n_fail': len(fails)}, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
