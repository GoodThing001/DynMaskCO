"""Small real-generator training-seed teacher probe; no training or online gate.

The fixed small configuration below is ONLY an engineering diagnostic. It is
not the formal I3 configuration and must not be selected by development gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_counterfactual_teacher import (CaptureTeacherReplanner, ReplaySafeSaaReplanner,
    TRAIN_SEED, replay_action, scenario_margin, teacher_preference)
from scenario_saa import CondHistoricalSampler, build_history, make_c0_contract_v2
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from run_identity import dataset_identity, dataset_meta_sha256
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from strict_online_env import StrictOnlineEnv

CONFIG = {'version': 'engineering_small_train_probe_v1', 'train_seed': TRAIN_SEED,
          'history_days': 12, 'probe_days': 3, 'orders': 20,
          'capture_ordinals': [0, 4, 9, 14, 19], 'budget': 700, 'vehicles': 3,
          'K': 2, 'continuation_time_limit': 'infinity', 'minimum_gap': 5,
          'future_policy': 'density', 'energy_pricing': 'marginal',
          'shadow_mode': 'greedy', 'not_formal_training_config': True}
SCALE200_CONFIG = {**CONFIG, 'version': 'engineering_200_order_train_probe_v1',
    'history_days': 200, 'probe_days': 1, 'orders': 200,
    'capture_ordinals': [24, 99, 174], 'budget': 710.6548152249378,
    'cooling_share': 3.7208946347769882, 'vehicles': 15,
    'budget_provenance': 'Existing 200 training-day rho=0.60 budget; fixed before probe',
    'K': 2, 'continuation_time_limit': 'infinity'}


def probe():
    start = time.perf_counter()
    contract = make_c0_contract_v2()
    full = generate_dataset(CONFIG['history_days'], CONFIG['orders'], TRAIN_SEED)
    history = build_history(full)
    ds = add_v2_initial_quality({k: v[:CONFIG['probe_days']].copy()
                                for k, v in full.items()}, contract)
    data_hash, data_meta = dataset_identity(ds), dataset_meta_sha256(ds)
    rows, days = [], []
    for day in range(CONFIG['probe_days']):
        # Exclude the replayed day from its teacher's historical scenario pool.
        # Full real future remains available only through private replay data.
        pool = [h for idx, h in enumerate(history) if idx != day]

        def factory(capture=False):
            cls = CaptureTeacherReplanner if capture else ReplaySafeSaaReplanner
            opts = {'capture_ordinals': CONFIG['capture_ordinals']} if capture else {}
            return cls(budget=CONFIG['budget'], capacity=50,
                booking_horizon=BOOKING_HORIZON, contract=contract,
                cooling_share=CONFIG.get('cooling_share', 1),
                sampler=CondHistoricalSampler(pool), reject_penalty={0: 5, 1: 10, 2: 15},
                K=CONFIG['K'], time_limit=float('inf'), arm_seed=7001,
                shadow_mode='greedy', energy_pricing='marginal',
                future_policy='density', **opts)

        policy, records = factory(capture=True), []
        env = StrictOnlineEnv(ds, capacity=50, tw_speed=SPEED_KMH / KM_PER_UNIT,
            num_vehicles=CONFIG['vehicles'], replanner=policy,
            coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        env.snapshot_hook = policy.capture_hook(records, train_seed=TRAIN_SEED,
            dataset_sha256=data_hash, dataset_meta_sha256=data_meta)
        day_start = time.perf_counter()
        traces, _ = env.run(day)
        day_elapsed = time.perf_counter() - day_start
        original = evaluate_trace_a1(traces, {k: v[day] for k, v in ds.items()}, contract,
            policy._accepted, policy._rejected, policy.budget, policy.reject_penalty)
        days.append({'day_id': day, 'hard_feasible': original['hard_feasible'],
                     'utility': original['utility'], 'captured': len(records),
                     'elapsed_s': day_elapsed})
        for ordinal, record in zip(CONFIG['capture_ordinals'], records):
            branches, branch_elapsed = [], []
            for accept in (True, False):
                branch_start = time.perf_counter()
                branches.append(replay_action(record, ds, factory, contract, BOOKING_HORIZON,
                                SPEED_KMH / KM_PER_UNIT, accept))
                branch_elapsed.append(time.perf_counter() - branch_start)
            a, r = branches
            preference = teacher_preference(a, r, minimum_gap=CONFIG['minimum_gap'])
            chosen = a if record.candidate in policy._accepted else r
            parity = (chosen['action_feasible'] and
                      chosen['accepted'] == sorted(policy._accepted) and
                      chosen['rejected'] == sorted(policy._rejected) and
                      all(abs(chosen[m] - original[m]) < 1e-8 for m in
                          ('utility', 'energy_kwh', 'distance_km', 'revenue',
                           'fuel_cost', 'reject_loss')))
            draws = factory().sampler.sample(record.visible,
                np.random.default_rng([9031, day, ordinal]), CONFIG['K'])
            margin = scenario_margin(record, ds, factory, contract, BOOKING_HORIZON,
                                     SPEED_KMH / KM_PER_UNIT, draws)
            row = {'day_id': day, 'ordinal': ordinal, 'candidate': record.candidate,
                   'clock': record.visible.clock, 'record_hash': record.record_hash,
                   'continuation_signature': record.continuation_signature,
                   'branch_elapsed_s': branch_elapsed,
                   'same_action_full_trajectory_parity': bool(parity), **preference,
                   'hist_shadow_margin': margin['margin'],
                   'branches': [{k: b.get(k) for k in ('action_feasible', 'hard_feasible',
                       'utility', 'revenue', 'fuel_cost', 'reject_loss', 'failures', 'timeouts')}
                       for b in branches]}
            rows.append(row)
        if len(records) != len(CONFIG['capture_ordinals']):
            raise RuntimeError('Expected capture ordinals incomplete; no silent trimming')
    gaps = [row['teacher_delta'] for row in rows if row['teacher_valid']]
    summary = {'records': len(rows), 'valid_teachers': len(gaps),
        'eligible': sum(row['decision_eligible'] for row in rows),
        'positive_delta': sum(g > 0 for g in gaps), 'negative_delta': sum(g < 0 for g in gaps),
        'delta_min': min(gaps) if gaps else None, 'delta_max': max(gaps) if gaps else None,
        'all_same_action_parity': all(row['same_action_full_trajectory_parity'] for row in rows),
        'all_original_hard_feasible': all(day['hard_feasible'] for day in days)}
    files = [Path(__file__), ROOT / 'training/i3_counterfactual_teacher.py',
             ROOT / 'evaluation/scenario_saa.py', ROOT / 'evaluation/run_exp_energy_c0.py',
             ROOT / 'simulation/strict_online_env.py', ROOT / 'simulation/recourse_snapshot.py',
             ROOT / 'evaluation/coldchain_evaluator_a1.py']
    return {'purpose': 'engineering diagnostic, no MaskCO training or online gain claim',
            'config': CONFIG, 'dataset_sha256': data_hash, 'dataset_meta_sha256': data_meta,
            'history_dataset_sha256': dataset_identity(full),
            'history_dataset_meta_sha256': dataset_meta_sha256(full),
            'elapsed_s': time.perf_counter() - start,
            'contract_hash': contract.contract_hash,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files}, 'days': days, 'summary': summary, 'records': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scale200', action='store_true',
                        help='Fixed 1 real 200-order training day/3 predeclared ordinals, K=2/infinite budget diagnostic')
    args = parser.parse_args()
    if args.scale200:
        CONFIG = SCALE200_CONFIG
    result = probe()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False))
    if not result['summary']['all_same_action_parity']:
        raise SystemExit('FAIL: real training-seed replay parity')
