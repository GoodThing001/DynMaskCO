#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1-D3-L1 SERVER F-FREE TIMING QUALIFICATION (user-authorised, one run).

Pre-registered BEFORE execution (written to disk first, then never relaxed):
  K = 2 reveal slots on cal instance 0 (R1 = first K reveal events)
  total wall clock <= 1800 s, per-slot <= 600 s
  calls: 1 F-free baseline rollout, then per slot: 1 capture, 1 restore, 1 hash recompute,
         1 resume-parity run.  No retry loop, no re-run to obtain a nicer sample.

Qualification is NOT just runtime:
  Q_server = Q_identity AND Q_contract AND Q_recovery AND Q_causal AND Q_budget
Every gate is recorded separately; any failure blocks formal collection.

Per-slot incremental persistence: each slot is appended to a JSONL file as soon as it finishes, so
an interruption leaves the completed slots, the stop reason and the unfinished records on disk.
tmux is not treated as persistence.  All raw failures and exceptions are kept.

The F-free chain uses no accelerator; GPU visibility is recorded for the record only.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

K_MAX = 2
TOTAL_CAP_S = 1800.0
PER_SLOT_CAP_S = 600.0
EFF_HASH = 'edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833'
BASE_HASH = '32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0'
PROFILE_HASH = 'df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6'
HORIZON = 24.0
GATES = {}
REC = {}


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def env_facts():
    e = {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'XLA_FLAGS', 'OMP_NUM_THREADS',
                                        'MKL_NUM_THREADS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                                        'CONDA_DEFAULT_ENV')}
    try:
        e['loadavg'] = list(os.getloadavg())
    except Exception:
        e['loadavg'] = None
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'cpu_count': os.cpu_count(), 'hostname': platform.node(), 'env': e,
            'gpus': (os.popen('nvidia-smi --query-gpu=name,utilization.gpu,memory.used '
                              '--format=csv,noheader').read().strip().splitlines()
                     or None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_cal_teacher.npz'))
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--identity', default=None, help='frozen L1 contract protocol identity JSON')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--K', type=int, default=K_MAX)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / 'A1_D3_L1_SERVER_TIMING_SLOTS.jsonl'
    started = time.time()

    # ---- budget registration FIRST -------------------------------------------------------
    budget = {'registered_before_run': True, 'K': args.K, 'TOTAL_CAP_S': TOTAL_CAP_S,
              'PER_SLOT_CAP_S': PER_SLOT_CAP_S,
              'calls': {'rollout': 1, 'per_slot': ['capture', 'restore', 'hash', 'parity']},
              'no_retry_loop': True, 'no_budget_relaxation_after_results': True,
              'machine': env_facts(), 'argv': sys.argv, 'cwd': os.getcwd(),
              'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    (outdir / 'A1_D3_L1_SERVER_TIMING_BUDGET.json').write_text(
        json.dumps(budget, indent=2, default=repr), encoding='utf-8')
    print('BUDGET registered: K=%d total<=%.0fs per-slot<=%.0fs' % (args.K, TOTAL_CAP_S,
                                                                    PER_SLOT_CAP_S))

    import numpy as np
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       reveal_event_times, run_baseline_rollout)
    from recourse_snapshot import (CC_SNAPSHOT_SCHEMA_VERSION, capture_recourse_snapshot,
                                   restore_recourse_snapshot, snapshot_state_hash,
                                   validate_snapshot_against_env)
    import test_d3_l1_resume_parity_attribution as ATT

    prof_dict = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    fields = {f.name for f in __import__('dataclasses').fields(ObjectiveProfile)}
    prof = ObjectiveProfile(**{k: v for k, v in prof_dict.items() if k in fields})
    base = default_pilot_contract()
    eff = apply_objective_profile(base, prof)
    eff_h = getattr(eff, 'contract_hash', None)

    # ---- Q_identity ----------------------------------------------------------------------
    cur = {'coldchain_contract': sha(EXT / 'scripts/coldchain/coldchain_contract.py'),
           'coldchain_state': sha(EXT / 'scripts/coldchain/coldchain_state.py'),
           'strict_online_env': sha(EXT / 'scripts/simulation/strict_online_env.py'),
           'recourse_snapshot': sha(EXT / 'scripts/simulation/recourse_snapshot.py'),
           'd3_l1_state_collector': sha(EXT / 'scripts/simulation/d3_l1_state_collector.py'),
           'dataset': sha(args.dataset), 'profile': sha(args.profile)}
    frozen = None
    if args.identity and Path(args.identity).is_file():
        frozen = json.loads(Path(args.identity).read_text(encoding='utf-8'))
        frozen = frozen.get('records', {}).get('identity', {}).get('source_sha256')
    diffs = {}
    if frozen:
        for k, v in frozen.items():
            if k in cur and cur[k] != v:
                diffs[k] = {'server': cur[k], 'frozen': v}
    REC['identity'] = {'current': cur, 'frozen_source_sha256': frozen, 'mismatches': diffs,
                       'identity_file_supplied': args.identity}
    GATES['Q_identity'] = bool(frozen) and not diffs
    print('Q_identity=%s mismatches=%s' % (GATES['Q_identity'], list(diffs)))

    # ---- Q_contract ---------------------------------------------------------------------
    from d3_l1_state_collector import HORIZON as COL_HORIZON
    REC['contract'] = {'base_hash': getattr(base, 'contract_hash', None), 'effective_hash': eff_h,
                       'profile_hash': prof_dict.get('profile_hash'), 'horizon': COL_HORIZON}
    GATES['Q_contract_pending'] = (eff_h == EFF_HASH
                                   and prof_dict.get('profile_hash') == PROFILE_HASH
                                   and getattr(base, 'contract_hash', None) == BASE_HASH
                                   and COL_HORIZON == HORIZON)
    print('Q_contract(part1)=%s eff=%s' % (GATES['Q_contract_pending'], str(eff_h)[:16]))

    # ---- rollout + per-slot work with incremental persistence ----------------------------
    data = dict(np.load(args.dataset))
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff, HORIZON)
    expected = reveal_event_times(env, args.instance)
    t0 = time.time()
    col = RevealSlotCollector(env, args.instance, args.K)
    rollout = run_baseline_rollout(env, args.instance, hook=col)
    rollout_s = time.time() - t0
    orig_traces, orig_served = rollout['result']
    REC['rollout'] = {'entry': rollout['entry'], 'seconds': rollout_s,
                      'n_slots': len(col.slots), 'expected_first_K': expected[:args.K]}
    GATES['Q_causal'] = bool(all(col.assertions().values())) and len(col.slots) == args.K
    REC['causal'] = {'slot_assertions': col.assertions(), 'horizon_original': env.booking_horizon,
                     'slots': [{'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id'],
                                'reveal_idx': r['reveal_idx']} for r in col.slots]}
    print('Q_causal=%s slots=%d' % (GATES['Q_causal'], len(col.slots)))

    recovery_ok, contract_ok = True, True
    with jsonl.open('a', encoding='utf-8') as fh:
        for r in col.slots:
            ent = {'slot': r['slot'], 'clock': r['clock'], 'event_id': r['event_id'],
                   'capture_seconds': r.get('capture_seconds')}
            slot_start = time.time()
            try:
                snap = r['snapshot']
                ent['schema_version'] = snap.get('schema_version')
                ent['snapshot_contract_hash'] = snap.get('coldchain_contract_hash')
                contract_ok = contract_ok and (ent['schema_version'] == CC_SNAPSHOT_SCHEMA_VERSION
                                               and ent['snapshot_contract_hash'] == EFF_HASH)
                t = time.time()
                h1 = snapshot_state_hash(snap)
                ent['hash_seconds'] = time.time() - t
                t = time.time()
                restored = restore_recourse_snapshot(snap)
                ent['restore_seconds'] = time.time() - t
                objs = restored if isinstance(restored, tuple) else (restored,)
                ent['restore_arity'] = len(objs)
                env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff,
                                  HORIZON)
                validate_snapshot_against_env(env_r, snap)
                ent['restored_horizon'] = env_r.booking_horizon
                recap = []

                def hook2(e, i, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                          all_customers, _t=float(r['clock']), _o=recap, _i=int(args.instance)):
                    if abs(float(clock) - _t) < 1e-9:
                        _o.append(capture_recourse_snapshot(e, _i, float(clock), int(event_id),
                                                            int(reveal_idx), vehicles, traces,
                                                            served_mask,
                                                            all_customers=list(all_customers)))

                env_r.snapshot_hook = hook2
                t = time.time()
                resumed = env_r.run_resumed(snap)
                ent['parity_seconds'] = time.time() - t
                ent['state_identity'] = bool(recap) and snapshot_state_hash(recap[0]) == h1
                res_traces, res_served = (resumed if isinstance(resumed, tuple)
                                          else (resumed, None))
                ent['served_mask_equal'] = bool(res_served is not None
                                                and np.array_equal(np.asarray(res_served),
                                                                   np.asarray(orig_served)))
                st = {'nodes': 0, 'truncated': False, 'depth_guard': False}
                ca, cb = ATT.canon(orig_traces, st), ATT.canon(res_traces, st)
                cmp_res = ATT.compare(ca, cb, 'semantic')
                ent['semantic_complete'] = cmp_res['complete']
                ent['semantic_diffs'] = cmp_res['n_diffs']
                ent['canon_truncated'] = st['truncated'] or st['depth_guard']
                recovery_ok = recovery_ok and ent['state_identity'] and ent['served_mask_equal'] \
                    and cmp_res['complete'] and cmp_res['n_diffs'] == 0
            except BaseException as exc:
                ent['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
                ent['traceback_tail'] = traceback.format_exc()[-800:]
                recovery_ok = False
            ent['slot_seconds'] = time.time() - slot_start
            ent['within_per_slot_cap'] = ent['slot_seconds'] <= PER_SLOT_CAP_S
            recovery_ok = recovery_ok and ent['within_per_slot_cap']
            fh.write(json.dumps(ent, default=repr, sort_keys=True) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
            print('slot %s persisted (%.2fs) state_identity=%s'
                  % (ent['slot'], ent['slot_seconds'], ent.get('state_identity')))
    GATES['Q_recovery'] = bool(recovery_ok)
    GATES['Q_contract'] = bool(contract_ok and GATES.pop('Q_contract_pending'))
    total = time.time() - started
    GATES['Q_budget'] = (total <= TOTAL_CAP_S and len(col.slots) == args.K)
    REC['timing'] = {'rollout_seconds': rollout_s, 'total_seconds': total,
                     'TOTAL_CAP_S': TOTAL_CAP_S, 'PER_SLOT_CAP_S': PER_SLOT_CAP_S,
                     'slots_jsonl': str(jsonl), 'machine': budget['machine'],
                     'slots_completed': len(col.slots)}
    REC['gates'] = GATES
    q_server = all(GATES.get(k) for k in ('Q_identity', 'Q_contract', 'Q_recovery', 'Q_causal',
                                          'Q_budget'))
    REC['verdict'] = {
        'Q_server': 'PASS' if q_server else 'NOT_PASSED',
        'formula': 'Q_identity AND Q_contract AND Q_recovery AND Q_causal AND Q_budget',
        'gate_details': GATES,
        'formal_collection_released': False,
        'note': 'this is a timing/identity qualification on cal instance 0 only; it is NOT the '
                'formal dev_check x K=4 collection and NOT an F-vs-D-Geo comparison',
        'states_collected_for_L1': 0, 'algorithmic_gain_claim': 'none'}
    (outdir / 'A1_D3_L1_SERVER_TIMING_QUALIFICATION.json').write_text(
        json.dumps(REC, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== Q_server=%s | gates=%s | total=%.2fs ===' % (REC['verdict']['Q_server'],
                                                             GATES, total))
    return 0 if q_server else 1


if __name__ == '__main__':
    sys.exit(main())
