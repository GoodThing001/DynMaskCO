"""event_plan_v2 E3 闭环 gate：baseline(JF1-H-F) vs EventPlanReplanner。

mode ∈ {distance, proxy, learned}：
  - distance：修正的 S_distance（含返仓段；无阈值，只选更短）；
  - proxy：无学习物理代理 S_proxy（J 单位阈值 tau 网格）；
  - learned：A/B 表示评分器（s·(f(P)-f(P0)) > tau）。

主指标 ΔJ = J_method − J_baseline（负=改善），实例等权；服务/硬约束优先。
完整报告 D/Q/E 分量、接受次数、事件数、耗时、hard vector / service_ok。

用法（本地/服务器）：
    python scripts/evaluation/run_event_plan_gate.py \
        --mode learned --representation B \
        --ckpt results/m0_scale/event_plan_v2_B_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --tau 0.005 --max-instances 16 --out results/m0_scale/event_plan_v2_B_s42_dev
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
from event_plan_replanner import EventPlanReplanner, load_plan_scorer
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
    ap.add_argument('--mode', choices=['learned', 'distance', 'proxy'], default='learned')
    ap.add_argument('--representation', choices=['A', 'B'], default='A')
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
        args.representation = scorer.representation
        print(f"loaded learned plan scorer (s={scorer.s:.4f} repr={scorer.representation})",
              flush=True)

    rows = []
    t0 = time.time()
    for i in range(n):
        # baseline
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        # method
        collector = PrePrepareContextCollector(deindex=True)
        replanner = EventPlanReplanner(
            scorer=scorer, mode=args.mode, representation=args.representation,
            deindex=True, tau=args.tau, capacity=args.capacity, num_vehicles=args.num_vehicles,
            context_source=collector, seed=args.seed, contract=eff)
        menv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        tm, sm = menv.run(i)
        m = _eval(menv, i, tm, args.objective, served_mask=sm)
        err = outcome_protocol_error(m, args.objective, strict_repair=True)
        if err is not None:
            raise RuntimeError(f"inst {i} method PROTOCOL_ERROR {err}")
        rows.append({'inst_idx': i,
                     'base_J': _cost(base, args.objective),
                     'method_J': _cost(m, args.objective),
                     'delta': _cost(m, args.objective) - _cost(base, args.objective),
                     'base_D': float(base['distance_cost']),
                     'method_D': float(m['distance_cost']),
                     'base_Q': float(base['quality_loss']),
                     'method_Q': float(m['quality_loss']),
                     'base_E': float(base['energy_kwh']),
                     'method_E': float(m['energy_kwh']),
                     'accept': replanner.online_stats.get('n_accept'),
                     'n_events': replanner.online_stats.get('n_events'),
                     'base_service_ok': bool(service_ok(base, args.objective)),
                     'method_service_ok': bool(service_ok(m, args.objective)),
                     'base_hard': hard_vector_from_outcome(base),
                     'method_hard': hard_vector_from_outcome(m)})
        print(f"  [inst {i}] base={rows[-1]['base_J']:.4f} method={rows[-1]['method_J']:.4f} "
              f"Δ={rows[-1]['delta']:+.4f} accept={rows[-1]['accept']}", flush=True)

    deltas = [r['delta'] for r in rows]
    comp = {}
    for k, label in (('D', 'distance_cost'), ('Q', 'quality_loss'), ('E', 'energy_kwh')):
        base_v = [r[f'base_{k}'] for r in rows]
        meth_v = [r[f'method_{k}'] for r in rows]
        comp[k] = {'mean_base': float(np.mean(base_v)), 'mean_method': float(np.mean(meth_v)),
                   'mean_delta': float(np.mean(np.asarray(meth_v) - np.asarray(base_v)))}
    summary = {
        'mode': args.mode, 'representation': args.representation, 'tau': args.tau, 'n': n,
        'mean_delta': float(np.mean(deltas)), 'ci': list(paired_bootstrap_ci(deltas)),
        'base_service_ok': all(r['base_service_ok'] for r in rows),
        'method_service_ok': all(r['method_service_ok'] for r in rows),
        'n_accept_total': int(sum(r['accept'] for r in rows)),
        'component_deltas': comp,
        'total_runtime': time.time() - t0,
        'rows': rows,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'baseline_J', 'method_J', 'delta', 'base_D', 'method_D',
                    'base_Q', 'method_Q', 'base_E', 'method_E', 'base_service_ok',
                    'method_service_ok', 'accept', 'n_events'])
        for r in rows:
            w.writerow([r['inst_idx'], f"{r['base_J']:.4f}", f"{r['method_J']:.4f}",
                        f"{r['delta']:+.4f}", f"{r['base_D']:.4f}", f"{r['method_D']:.4f}",
                        f"{r['base_Q']:.4f}", f"{r['method_Q']:.4f}",
                        f"{r['base_E']:.4f}", f"{r['method_E']:.4f}",
                        int(r['base_service_ok']), int(r['method_service_ok']),
                        r['accept'], r['n_events']])
    print(f"\n=== event_plan gate (mode={args.mode} repr={args.representation} tau={args.tau} "
          f"n={n}) ===")
    print(f"  Δ={summary['mean_delta']:+.4f} CI={[round(x,4) for x in summary['ci']]} "
          f"accept={summary['n_accept_total']} service={summary['method_service_ok']}")
    print(f"  D/Q/E deltas: D={comp['D']['mean_delta']:+.4f} Q={comp['Q']['mean_delta']:+.4f} "
          f"E={comp['E']['mean_delta']:+.4f}")
    print(f"saved: {args.out}/summary.json + per_instance.csv")


if __name__ == '__main__':
    main()
