"""feature-local 70 维在线 gate：JF1-H-F baseline vs FeatureLocalReplanner（K=1，⑤ 闭环诊断）。

与 run_dynmaskco_cc_gate 同一 strict-online 协议，差异：
  - 方法 = FeatureLocalReplanner（ScoringMLP(70)）+ PrePrepareContextCollector（pre-prepare context）；
  - collector 注册到方法 env 的 snapshot_hook，读 pre-prepare 状态（方案 a，不改 strict_online_env）；
  - τ 由 --tau 传入（CAL 选出的 τ*；τ=0 行为诊断；τ=inf 无修改对照，应 == baseline）。

主指标：ΔJ = J_method − J_baseline（负值=轨迹改善），实例为评价单位。同时报告 service/硬约束、
D/Q/E、接受次数、首个计划分叉、认证拒绝原因、耗时。

用法（服务器）：
    python scripts/evaluation/run_feature_local_gate.py \
        --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --deindex --K 1 --tau 0.005 --max-instances 2 --out results/m0_scale/gate_fl_F_H_s42_tau0005
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
from feature_local_replanner import (FeatureLocalReplanner, PrePrepareContextCollector,
                                     load_feature_local_scorer)
from counterfactual_teacher import _eval
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import service_ok, hard_vector_from_outcome, outcome_protocol_error
from service_first import paired_bootstrap_ci


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    profile = ObjectiveProfile(
        name=data['name'], distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']), energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']), lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'), dev_statistics=data.get('dev_statistics'))
    if profile.profile_hash != data.get('profile_hash'):
        raise ValueError(f"profile hash mismatch: {profile.profile_hash[:12]}")
    return profile


def _make_env(dataset, capacity, num_vehicles, replanner, objective, profile, contract=None):
    if objective == 'coldchain':
        contract = apply_objective_profile(contract or default_pilot_contract(), profile)
    else:
        contract = None
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=replanner, coldchain_contract=contract)


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def _run_instance(dataset, capacity, num_vehicles, objective, profile, contract, inst_idx,
                  scorer, deindex, K, tau):
    # baseline（JF1-H-F）
    benv = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile,
                     contract)
    traces_b, served_b = benv.run(inst_idx)
    base = _eval(benv, inst_idx, traces_b, objective, served_mask=served_b)
    err = outcome_protocol_error(base, objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: baseline PROTOCOL_ERROR {err}")

    # 方法（FeatureLocalReplanner + collector）
    collector = PrePrepareContextCollector(deindex=deindex)
    replanner = FeatureLocalReplanner(scorer=scorer, K=K, tau=tau, capacity=capacity,
                                      deindex=deindex, context_source=collector)
    menv = _make_env(dataset, capacity, num_vehicles, replanner, objective, profile, contract)
    menv.snapshot_hook = collector.hook
    traces_m, served_m = menv.run(inst_idx)
    m1 = _eval(menv, inst_idx, traces_m, objective, served_mask=served_m)
    err = outcome_protocol_error(m1, objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx}: method PROTOCOL_ERROR {err}")

    stats = getattr(menv.replanner, 'online_stats', {})
    timing = getattr(menv.replanner, 'timing', {})
    return {'inst_idx': inst_idx, 'baseline': base, 'method': m1,
            'served_equal': bool(np.array_equal(served_b, served_m)),
            'online_stats': stats, 'timing': timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--tau', type=float, default=0.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())

    scorer = load_feature_local_scorer(args.ckpt)
    print(f"loaded feature-local scorer (s={scorer.s:.4f})", flush=True)

    results = []
    os.makedirs(os.path.join(args.out, 'instances'), exist_ok=True)
    t0 = time.time()
    for i in range(n):
        r = _run_instance(dataset, args.capacity, args.num_vehicles, args.objective, profile,
                          contract, i, scorer, args.deindex, args.K, args.tau)
        results.append(r)
        with open(os.path.join(args.out, 'instances', f'inst_{i}.json'), 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  [inst {i}] base={_cost(r['baseline'], args.objective):.4f} "
              f"method={_cost(r['method'], args.objective):.4f} "
              f"accept={r['online_stats'].get('n_accept', 0)} "
              f"keep={r['online_stats'].get('n_keep', 0)} "
              f"cert_rej={r['online_stats'].get('n_certificate_reject', 0)} "
              f"t={time.time()-t0:.0f}s", flush=True)

    base = [r['baseline'] for r in results]
    m = [r['method'] for r in results]

    def _full_ok(o):
        return (service_ok(o, args.objective) and int(o.get('terminal_unresolved', 0)) == 0
                and int(o.get('ownership_violations', 0)) == 0)

    base_ok = all(_full_ok(o) for o in base)
    m_ok = all(_full_ok(o) for o in m)
    paired = [i for i in range(n) if _full_ok(base[i]) and _full_ok(m[i])]
    deltas = [_cost(m[i], args.objective) - _cost(base[i], args.objective) for i in paired]
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    ci = list(paired_bootstrap_ci(deltas)) if deltas else None

    field = 'distance_cost' if args.objective == 'distance' else 'coldchain_cost'
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', f'baseline_{field}', f'method_{field}', 'delta',
                    'baseline_complete', 'method_complete', 'served_equal'])
        for i in range(n):
            w.writerow([i, f"{_cost(base[i], args.objective):.4f}",
                        f"{_cost(m[i], args.objective):.4f}",
                        f"{_cost(m[i], args.objective) - _cost(base[i], args.objective):+.4f}",
                        int(base[i]['complete']), int(m[i]['complete']),
                        int(results[i]['served_equal'])])

    summary = {
        'ckpt': args.ckpt, 'tau': args.tau, 'K': args.K, 'deindex': bool(args.deindex),
        'n': n, 'baseline_service_ok': bool(base_ok), 'method_service_ok': bool(m_ok),
        'paired_n': len(paired),
        'baseline_mean': float(np.mean([_cost(base[i], args.objective) for i in paired])) if paired else None,
        'method_mean': float(np.mean([_cost(m[i], args.objective) for i in paired])) if paired else None,
        'mean_delta': mean_delta, 'ci': ci,
        'online_stats': {k: int(sum(r['online_stats'].get(k, 0) for r in results))
                         for k in ['n_eligible', 'n_candidates', 'n_score', 'n_keep', 'n_defer',
                                   'n_accept', 'n_certificate_reject', 'n_unexpected_error']},
        'component_deltas': {},
        'timing': {k: float(sum(r['timing'].get(k, 0.0) for r in results))
                   for k in ['input_prep', 'model_score', 'cert']},
    }
    for cf in ('distance_cost', 'quality_loss', 'energy_kwh'):
        cd = [m[i][cf] - base[i][cf] for i in paired]
        summary['component_deltas'][cf] = (float(np.mean(cd)) if cd else None)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== feature-local gate (tau={args.tau}, K={args.K}, n={n}) ===")
    print(f"  [service] baseline={base_ok} method={m_ok} (paired n={len(paired)}/{n})")
    print(f"  [cost] baseline={summary['baseline_mean']:.4f} method={summary['method_mean']:.4f} "
          f"Δ={mean_delta:+.4f} CI={ci}")
    print(f"  [stats] accept={summary['online_stats']['n_accept']} "
          f"keep={summary['online_stats']['n_keep']} "
          f"cert_rej={summary['online_stats']['n_certificate_reject']}")
    print(f"  [components] D={summary['component_deltas']['distance_cost']:+.4f} "
          f"Q={summary['component_deltas']['quality_loss']:+.4f} "
          f"E={summary['component_deltas']['energy_kwh']:+.4f}")
    print(f"  saved: {args.out}/per_instance.csv + summary.json")


if __name__ == '__main__':
    main()
