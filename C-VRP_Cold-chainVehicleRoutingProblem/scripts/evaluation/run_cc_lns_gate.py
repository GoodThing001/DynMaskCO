"""N0 闭环 gate：JF1-H-F vs CC-LNS-RH（1s / 4s 两档预算）。

主指标 ΔJ = J_method − J_baseline（负=改善），实例等权；服务/硬约束优先。
完整报告逐实例 D/Q/E、hard vector、逐事件耗时 p50/p95、候选/改进/改变计数、fallback。

用法：
    python scripts/evaluation/run_cc_lns_gate.py \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 1.0 --seed 0 --max-instances 16 --out results/m0_scale/cc_lns_1s_dev
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from feature_local_replanner import PrePrepareContextCollector
from cc_lns_replanner import CCLNSReplanner
from counterfactual_teacher import _eval
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import hard_vector_from_outcome, service_ok, outcome_protocol_error
from service_first import paired_bootstrap_ci


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(
        name=data['name'], distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']), energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']), lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'), dev_statistics=data.get('dev_statistics'))


def _cost(o, objective):
    return float(o['distance_cost'] if objective == 'distance' else o['coldchain_cost'])


def _timing(event_log):
    t = [e['elapsed_s'] for e in event_log]
    if not t:
        return {'n': 0, 'p50': 0.0, 'p95': 0.0, 'total': 0.0, 'max': 0.0}
    a = np.asarray(t)
    return {'n': len(t), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)), 'total': float(a.sum()),
            'max': float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=1.0)
    ap.add_argument('--guard', action='store_true', help='N0-R 预留资源保护')
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None

    rows = []
    all_event_logs = []
    for i in range(n):
        # baseline
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        # method
        collector = PrePrepareContextCollector(deindex=True)
        replanner = CCLNSReplanner(budget_s=args.budget, contract=eff, seed=args.seed,
                                   capacity=args.capacity, num_vehicles=args.num_vehicles,
                                   guard_reserved=args.guard)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        tm, sm = menv.run(i)
        m = _eval(menv, i, tm, args.objective, served_mask=sm)
        err = outcome_protocol_error(m, args.objective, strict_repair=True)
        if err is not None:
            raise RuntimeError(f"inst {i} PROTOCOL_ERROR {err}")
        timing = _timing(replanner.event_log)
        rep_time = sum(e.get('replanner_elapsed_s', 0.0) for e in replanner.event_log)
        rows.append({'inst_idx': i,
                     'base_J': _cost(base, args.objective),
                     'method_J': _cost(m, args.objective),
                     'delta': _cost(m, args.objective) - _cost(base, args.objective),
                     'base_D': float(base['distance_cost']), 'method_D': float(m['distance_cost']),
                     'base_Q': float(base['quality_loss']), 'method_Q': float(m['quality_loss']),
                     'base_E': float(base['energy_kwh']), 'method_E': float(m['energy_kwh']),
                     'base_service_ok': bool(service_ok(base, args.objective)),
                     'method_service_ok': bool(service_ok(m, args.objective)),
                     'method_hard': hard_vector_from_outcome(m),
                     'n_attempts': int(replanner.online_stats.get('n_attempts', 0)),
                     'n_feasible': int(replanner.online_stats.get('n_feasible', 0)),
                     'n_unique': int(replanner.online_stats.get('n_unique', 0)),
                     'n_duplicate': int(replanner.online_stats.get('n_duplicate', 0)),
                     'n_eval': int(replanner.online_stats.get('n_eval', 0)),
                     'n_partition_fail': int(replanner.online_stats.get('n_partition_fail', 0)),
                     'n_reconstruct_fail': int(replanner.online_stats.get('n_reconstruct_fail', 0)),
                     'n_improve': int(replanner.online_stats.get('n_improve', 0)),
                     'n_vehicles_changed': int(replanner.online_stats.get('n_vehicles_changed', 0)),
                     'P0_invalid': int(replanner.online_stats.get('n_P0_invalid', 0)),
                     'replanner_elapsed_s': float(rep_time),
                     'timing': timing})
        print(f"  [inst {i}] base={rows[-1]['base_J']:.4f} method={rows[-1]['method_J']:.4f} "
              f"Δ={rows[-1]['delta']:+.4f} attempt={rows[-1]['n_attempts']} "
              f"unique={rows[-1]['n_unique']} imp={rows[-1]['n_improve']} "
              f"svc={rows[-1]['method_service_ok']}", flush=True)
        for e in replanner.event_log:
            e['inst_idx'] = i
        all_event_logs.extend(replanner.event_log)

    deltas = [r['delta'] for r in rows]
    comp = {}
    for k in ('D', 'Q', 'E'):
        base_v = [r[f'base_{k}'] for r in rows]
        meth_v = [r[f'method_{k}'] for r in rows]
        comp[k] = {'mean_base': float(np.mean(base_v)), 'mean_method': float(np.mean(meth_v)),
                   'mean_delta': float(np.mean(np.asarray(meth_v) - np.asarray(base_v)))}
    summary = {
        'budget': args.budget, 'guard': bool(args.guard), 'n': n, 'mean_delta': float(np.mean(deltas)),
        'ci': list(paired_bootstrap_ci(deltas)),
        'base_service_ok': all(r['base_service_ok'] for r in rows),
        'method_service_ok': all(r['method_service_ok'] for r in rows),
        'component_deltas': comp,
        'n_attempts_total': int(sum(r['n_attempts'] for r in rows)),
        'n_feasible_total': int(sum(r['n_feasible'] for r in rows)),
        'n_unique_total': int(sum(r['n_unique'] for r in rows)),
        'n_duplicate_total': int(sum(r['n_duplicate'] for r in rows)),
        'n_eval_total': int(sum(r['n_eval'] for r in rows)),
        'n_partition_fail_total': int(sum(r['n_partition_fail'] for r in rows)),
        'n_reconstruct_fail_total': int(sum(r['n_reconstruct_fail'] for r in rows)),
        'n_improve_total': int(sum(r['n_improve'] for r in rows)),
        'n_vehicles_changed_total': int(sum(r['n_vehicles_changed'] for r in rows)),
        'P0_invalid_total': int(sum(r['P0_invalid'] for r in rows)),
        'P0_feasible_all': all(r['P0_invalid'] == 0 for r in rows),
        'replanner_elapsed_total': float(sum(r['replanner_elapsed_s'] for r in rows)),
        'rows': rows,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'event_log.json'), 'w') as f:
        json.dump(all_event_logs, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'baseline_J', 'method_J', 'delta', 'base_D', 'method_D',
                    'base_Q', 'method_Q', 'base_E', 'method_E', 'service_ok', 'n_attempts',
                    'n_unique', 'n_improve', 'n_changed', 'n_partition_fail', 'time_p50',
                    'time_p95', 'replanner_total'])
        for r in rows:
            w.writerow([r['inst_idx'], f"{r['base_J']:.4f}", f"{r['method_J']:.4f}",
                        f"{r['delta']:+.4f}", f"{r['base_D']:.4f}", f"{r['method_D']:.4f}",
                        f"{r['base_Q']:.4f}", f"{r['method_Q']:.4f}", f"{r['base_E']:.4f}",
                        f"{r['method_E']:.4f}", int(r['method_service_ok']),
                        r['n_attempts'], r['n_unique'], r['n_improve'], r['n_vehicles_changed'],
                        r['n_partition_fail'],
                        f"{r['timing']['p50']:.4f}", f"{r['timing']['p95']:.4f}",
                        f"{r['replanner_elapsed_s']:.4f}"])
    print(f"\n=== cc_lns gate (budget={args.budget}s, n={n}) ===")
    print(f"  Δ={summary['mean_delta']:+.4f} CI={[round(x,4) for x in summary['ci']]} "
          f"svc={summary['method_service_ok']}")
    print(f"  D/Q/E: D={comp['D']['mean_delta']:+.4f} Q={comp['Q']['mean_delta']:+.4f} "
          f"E={comp['E']['mean_delta']:+.4f}")
    print(f"  attempts={summary['n_attempts_total']} unique={summary['n_unique_total']} "
          f"improve={summary['n_improve_total']} changed={summary['n_vehicles_changed_total']} "
          f"pfail={summary['n_partition_fail_total']} P0_feasible={summary['P0_feasible_all']}")
    print(f"saved: {args.out}/summary.json + per_instance.csv + event_log.json")


if __name__ == '__main__':
    main()
