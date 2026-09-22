"""feature-local B/ONE/FULL 三策略闭环对比（⑤ 归因诊断第一步）。

对每个实例，跑三种策略并保存决策日志：
  - B     = JF1-H-F baseline；
  - ONE   = FeatureLocalReplanner(max_total_accepts=1)：只执行首次通过认证的 learned 动作，此后回 baseline；
  - FULL  = FeatureLocalReplanner(max_total_accepts=None)：当前 K=1 完整策略。

主指标：ΔJ = J_strategy − J_B（负=改善）。同时保存 decision_log + event_plans，供
「反复迁移/计划撤销/commit 变化」与 ĝ vs g_B 归因。

用法（服务器）：
    python scripts/evaluation/run_feature_local_onefull.py \
        --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --deindex --K 1 --tau 0.005 --max-instances 4 --out results/m0_scale/onefull_F_H_s42
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
from feature_local_replanner import (FeatureLocalReplanner, PrePrepareContextCollector,
                                     load_feature_local_scorer)
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


def _run_strategy(dataset, capacity, num_vehicles, objective, profile, contract, inst_idx,
                  scorer, deindex, K, tau, max_accepts=None, max_accepts_per_event=None):
    collector = PrePrepareContextCollector(deindex=deindex)
    replanner = FeatureLocalReplanner(scorer=scorer, K=K, tau=tau, capacity=capacity,
                                      deindex=deindex, context_source=collector,
                                      max_total_accepts=max_accepts,
                                      max_accepts_per_event=max_accepts_per_event)
    env = _make_env(dataset, capacity, num_vehicles, replanner, objective, profile, contract)
    env.snapshot_hook = collector.hook
    traces, served = env.run(inst_idx)
    out = _eval(env, inst_idx, traces, objective, served_mask=served)
    err = outcome_protocol_error(out, objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {inst_idx} strategy PROTOCOL_ERROR {err}")
    return out, replanner.decision_log, replanner.event_plans, replanner.online_stats


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
    print(f"loaded scorer (s={scorer.s:.4f}) tau={args.tau} max_instances={n}", flush=True)

    rows = []
    os.makedirs(os.path.join(args.out, 'instances'), exist_ok=True)
    for i in range(n):
        # B（baseline）
        benv = _make_env(dataset, args.capacity, args.num_vehicles, make_continuation(),
                         args.objective, profile, contract)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        # ONE + EVENT-ONE + FULL
        one, one_log, one_ev, one_stats = _run_strategy(
            dataset, args.capacity, args.num_vehicles, args.objective, profile, contract, i,
            scorer, args.deindex, args.K, args.tau, max_accepts=1)
        evone, evone_log, evone_ev, evone_stats = _run_strategy(
            dataset, args.capacity, args.num_vehicles, args.objective, profile, contract, i,
            scorer, args.deindex, args.K, args.tau, max_accepts_per_event=1)
        full, full_log, full_ev, full_stats = _run_strategy(
            dataset, args.capacity, args.num_vehicles, args.objective, profile, contract, i,
            scorer, args.deindex, args.K, args.tau, max_accepts=None)
        rec = {'inst_idx': i, 'base': _cost(base, args.objective),
               'one': _cost(one, args.objective), 'evone': _cost(evone, args.objective),
               'full': _cost(full, args.objective),
               'one_accept': one_stats.get('n_accept'), 'evone_accept': evone_stats.get('n_accept'),
               'full_accept': full_stats.get('n_accept')}
        rows.append(rec)
        with open(os.path.join(args.out, 'instances', f'inst_{i}.json'), 'w') as f:
            json.dump({'inst_idx': i, 'base': base, 'one': one, 'evone': evone, 'full': full,
                       'one_decision_log': one_log, 'one_event_plans': one_ev,
                       'evone_decision_log': evone_log, 'evone_event_plans': evone_ev,
                       'full_decision_log': full_log, 'full_event_plans': full_ev,
                       'one_stats': one_stats, 'evone_stats': evone_stats,
                       'full_stats': full_stats}, f, indent=2, default=str)
        print(f"  [inst {i}] B={rec['base']:.4f} ONE={rec['one']:.4f} "
              f"(Δ={rec['one']-rec['base']:+.4f}) EVENT-ONE={rec['evone']:.4f} "
              f"(Δ={rec['evone']-rec['base']:+.4f}) FULL={rec['full']:.4f} "
              f"(Δ={rec['full']-rec['base']:+.4f}) evone_accept={rec['evone_accept']} "
              f"full_accept={rec['full_accept']}", flush=True)

    d_one = [r['one'] - r['base'] for r in rows]
    d_evone = [r['evone'] - r['base'] for r in rows]
    d_full = [r['full'] - r['base'] for r in rows]
    d_full_evone = [r['full'] - r['evone'] for r in rows]
    d_evone_one = [r['evone'] - r['one'] for r in rows]
    summary = {
        'ckpt': args.ckpt, 'tau': args.tau, 'n': n,
        'one_mean_delta': float(np.mean(d_one)), 'one_ci': list(paired_bootstrap_ci(d_one)),
        'evone_mean_delta': float(np.mean(d_evone)), 'evone_ci': list(paired_bootstrap_ci(d_evone)),
        'full_mean_delta': float(np.mean(d_full)), 'full_ci': list(paired_bootstrap_ci(d_full)),
        'full_minus_evone': float(np.mean(d_full_evone)),
        'full_minus_evone_ci': list(paired_bootstrap_ci(d_full_evone)),
        'evone_minus_one': float(np.mean(d_evone_one)),
        'evone_minus_one_ci': list(paired_bootstrap_ci(d_evone_one)),
        'rows': rows,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== B/ONE/EVENT-ONE/FULL (tau={args.tau}, n={n}) ===")
    print(f"  ONE       Δ={summary['one_mean_delta']:+.4f} CI={[round(x,4) for x in summary['one_ci']]}")
    print(f"  EVENT-ONE Δ={summary['evone_mean_delta']:+.4f} CI={[round(x,4) for x in summary['evone_ci']]}")
    print(f"  FULL      Δ={summary['full_mean_delta']:+.4f} CI={[round(x,4) for x in summary['full_ci']]}")
    print(f"  FULL−EVENT-ONE ={summary['full_minus_evone']:+.4f} CI={[round(x,4) for x in summary['full_minus_evone_ci']]}")
    print(f"  EVENT-ONE−ONE  ={summary['evone_minus_one']:+.4f} CI={[round(x,4) for x in summary['evone_minus_one_ci']]}")
    print(f"saved: {args.out}/summary.json + instances/")


if __name__ == '__main__':
    main()
