"""N2 闭环 gate：JF1-H-F vs MaskCO 条件联合修复器（新训练）。

用法：
    python scripts/evaluation/run_repair_gate.py \
        --ckpt results/m0_scale/repair_s42/model.ckpt \
        --encoder-ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 1.0 --max-instances 16 --out results/m0_scale/repair_s42_dev
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
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from feature_local_replanner import PrePrepareContextCollector
from dynmaskco_cc_repair_replanner import RepairReplanner, RepairScorer, load_repair_model
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--encoder-ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=1.0)
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

    model = load_repair_model(args.ckpt, args.encoder_ckpt)
    scorer = RepairScorer(model)

    rows = []
    for i in range(n):
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        collector = PrePrepareContextCollector(deindex=True)
        replanner = RepairReplanner(model, scorer, contract=eff, budget_s=args.budget,
                                    seed=args.seed, capacity=args.capacity,
                                    num_vehicles=args.num_vehicles, guard_reserved=True)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        tm, sm = menv.run(i)
        m = _eval(menv, i, tm, args.objective, served_mask=sm)
        err = outcome_protocol_error(m, args.objective, strict_repair=True)
        if err is not None:
            raise RuntimeError(f"inst {i} PROTOCOL_ERROR {err}")
        rows.append({'inst_idx': i, 'base_J': _cost(base, args.objective),
                     'method_J': _cost(m, args.objective),
                     'delta': _cost(m, args.objective) - _cost(base, args.objective),
                     'base_D': float(base['distance_cost']), 'method_D': float(m['distance_cost']),
                     'base_Q': float(base['quality_loss']), 'method_Q': float(m['quality_loss']),
                     'base_E': float(base['energy_kwh']), 'method_E': float(m['energy_kwh']),
                     'base_service_ok': bool(service_ok(base, args.objective)),
                     'method_service_ok': bool(service_ok(m, args.objective)),
                     'n_attempts': int(replanner.online_stats.get('n_attempts', 0)),
                     'n_changed': int(replanner.online_stats.get('n_changed', 0))})
        print(f"  [inst {i}] base={rows[-1]['base_J']:.4f} method={rows[-1]['method_J']:.4f} "
              f"Δ={rows[-1]['delta']:+.4f} attempt={rows[-1]['n_attempts']} "
              f"svc={rows[-1]['method_service_ok']}", flush=True)

    deltas = [r['delta'] for r in rows]
    comp = {}
    for k in ('D', 'Q', 'E'):
        bv = [r[f'base_{k}'] for r in rows]; mv = [r[f'method_{k}'] for r in rows]
        comp[k] = {'mean_base': float(np.mean(bv)), 'mean_method': float(np.mean(mv)),
                   'mean_delta': float(np.mean(np.asarray(mv) - np.asarray(bv)))}
    summary = {'budget': args.budget, 'n': n, 'mean_delta': float(np.mean(deltas)),
               'ci': list(paired_bootstrap_ci(deltas)),
               'base_service_ok': all(r['base_service_ok'] for r in rows),
               'method_service_ok': all(r['method_service_ok'] for r in rows),
               'component_deltas': comp,
               'n_attempts_total': int(sum(r['n_attempts'] for r in rows)),
               'n_changed_total': int(sum(r['n_changed'] for r in rows)),
               'rows': rows}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'baseline_J', 'method_J', 'delta', 'base_D', 'method_D',
                    'base_Q', 'method_Q', 'base_E', 'method_E', 'service_ok', 'n_attempts',
                    'n_changed'])
        for r in rows:
            w.writerow([r['inst_idx'], f"{r['base_J']:.4f}", f"{r['method_J']:.4f}",
                        f"{r['delta']:+.4f}", f"{r['base_D']:.4f}", f"{r['method_D']:.4f}",
                        f"{r['base_Q']:.4f}", f"{r['method_Q']:.4f}", f"{r['base_E']:.4f}",
                        f"{r['method_E']:.4f}", int(r['method_service_ok']),
                        r['n_attempts'], r['n_changed']])
    print(f"\n=== repair gate (budget={args.budget}s, n={n}) ===")
    print(f"  Δ={summary['mean_delta']:+.4f} CI={[round(x,4) for x in summary['ci']]} "
          f"svc={summary['method_service_ok']}")
    print(f"  D/Q/E: D={comp['D']['mean_delta']:+.4f} Q={comp['Q']['mean_delta']:+.4f} "
          f"E={comp['E']['mean_delta']:+.4f}")
    print(f"saved: {args.out}/summary.json + per_instance.csv")


if __name__ == '__main__':
    main()
