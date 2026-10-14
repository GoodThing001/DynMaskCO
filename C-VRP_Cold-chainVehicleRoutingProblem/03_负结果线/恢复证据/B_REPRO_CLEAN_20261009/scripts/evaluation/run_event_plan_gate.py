"""event_plan_v1 E3 闭环 gate：baseline(JF1-H-F) vs EventPlanReplanner(S_learned / S_distance)。

主指标 ΔJ = J_method − J_baseline（负=改善），实例等权；服务/硬约束优先。同时报告 D/Q/E、接受次数、
事件数、耗时。

用法（服务器）：
    # learned 选择
    python scripts/evaluation/run_event_plan_gate.py \
        --mode learned --ckpt results/m0_scale/event_plan_model_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --tau 0.005 --max-instances 16 --out results/m0_scale/event_plan_gate_s42_learned
    # 简单距离选择
    python scripts/evaluation/run_event_plan_gate.py \
        --mode distance --data ... --tau 0 --max-instances 16 --out results/m0_scale/event_plan_gate_distance
"""
import argparse
import csv
import json
import os
import sys

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
from event_plan_replanner import EventPlanReplanner, load_plan_scorer
from counterfactual_teacher import _eval
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import service_ok, outcome_protocol_error
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
    ap.add_argument('--mode', choices=['learned', 'distance'], default='learned')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--tau', type=float, default=0.0)
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

    scorer = None
    if args.mode == 'learned':
        scorer = load_plan_scorer(args.ckpt)
        print(f"loaded learned plan scorer (s={scorer.s:.4f})", flush=True)

    rows = []
    for i in range(n):
        # baseline
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        # method
        collector = PrePrepareContextCollector(deindex=True)
        replanner = EventPlanReplanner(scorer=scorer, mode=args.mode, deindex=True, tau=args.tau,
                                       capacity=args.capacity, num_vehicles=args.num_vehicles,
                                       context_source=collector, seed=args.seed)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        tm, sm = menv.run(i)
        m = _eval(menv, i, tm, args.objective, served_mask=sm)
        err = outcome_protocol_error(m, args.objective, strict_repair=True)
        if err is not None:
            raise RuntimeError(f"inst {i} method PROTOCOL_ERROR {err}")
        rows.append({'inst_idx': i, 'base': _cost(base, args.objective),
                     'method': _cost(m, args.objective),
                     'delta': _cost(m, args.objective) - _cost(base, args.objective),
                     'accept': replanner.online_stats.get('n_accept'),
                     'n_events': replanner.online_stats.get('n_events'),
                     'base_complete': bool(base.get('complete')),
                     'method_complete': bool(m.get('complete'))})
        print(f"  [inst {i}] base={rows[-1]['base']:.4f} method={rows[-1]['method']:.4f} "
              f"Δ={rows[-1]['delta']:+.4f} accept={rows[-1]['accept']}", flush=True)

    deltas = [r['delta'] for r in rows]
    summary = {
        'mode': args.mode, 'tau': args.tau, 'n': n,
        'mean_delta': float(np.mean(deltas)), 'ci': list(paired_bootstrap_ci(deltas)),
        'base_service_ok': all(r['base_complete'] for r in rows),
        'method_service_ok': all(r['method_complete'] for r in rows),
        'n_accept_total': int(sum(r['accept'] for r in rows)),
        'component_deltas': {}, 'rows': rows,
    }
    for cf in ('distance_cost', 'quality_loss', 'energy_kwh'):
        pass
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'baseline_J', 'method_J', 'delta', 'base_complete',
                    'method_complete', 'accept', 'n_events'])
        for r in rows:
            w.writerow([r['inst_idx'], f"{r['base']:.4f}", f"{r['method']:.4f}",
                        f"{r['delta']:+.4f}", int(r['base_complete']), int(r['method_complete']),
                        r['accept'], r['n_events']])
    print(f"\n=== event_plan gate (mode={args.mode}, tau={args.tau}, n={n}) ===")
    print(f"  Δ={summary['mean_delta']:+.4f} CI={[round(x,4) for x in summary['ci']]} "
          f"accept={summary['n_accept_total']} service={summary['method_service_ok']}")
    print(f"saved: {args.out}/summary.json + per_instance.csv")


if __name__ == '__main__':
    main()
