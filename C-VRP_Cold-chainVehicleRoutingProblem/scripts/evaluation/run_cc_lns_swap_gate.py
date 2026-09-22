"""闭环验证（Q2）：JF1-H-F vs R vs R+两客户交换（相同预算计时边界）。

主指标 ΔJ = J_method − J_baseline（负=改善），实例等权；服务/硬约束优先。R 与 R+交换 使用
**同一个搜索核心**（cc_lns_swap_search，仅 swap 开关不同）与相同预算/轮次/候选上限，因此二者
差异唯一来源 = 两客户交换算子。报告实际耗时、逐实例 D/Q/E、ΔJ 与配对 CI。

回答：固定状态里 J_vis 更低的修复，是否在真实动态闭环中仍存在？（可能因提前返仓/资源占用
损害后续动态服务。）

用法（服务器）：
    python scripts/evaluation/run_cc_lns_swap_gate.py \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 4.0 --n-rounds 2 --max-attempts 64 --seed 0 \
        --max-instances 16 --out results/m0_scale/swap_gate_4s_dev
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
from cc_swap import CCLNSwapReplanner
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


def _run_method(dataset, i, eff, args, swap):
    replanner = CCLNSwapReplanner(budget_s=args.budget, contract=eff, seed=args.seed,
                                  capacity=args.capacity, num_vehicles=args.num_vehicles,
                                  guard_reserved=True, swap=swap, n_rounds=args.n_rounds,
                                  max_attempts=args.max_attempts)
    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                          replanner=replanner, coldchain_contract=eff)
    t, s = env.run(i)
    m = _eval(env, i, t, args.objective, served_mask=s)
    err = outcome_protocol_error(m, args.objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {i} PROTOCOL_ERROR {err}")
    return m, replanner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=4.0)
    ap.add_argument('--n-rounds', type=int, default=2)
    ap.add_argument('--max-attempts', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-instances', type=int, default=16)
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
        benv = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, args.objective, served_mask=sb)
        mR, repR = _run_method(dataset, i, eff, args, swap=False)
        mS, repS = _run_method(dataset, i, eff, args, swap=True)
        rows.append({
            'inst_idx': i, 'base_J': _cost(base, args.objective),
            'R_J': _cost(mR, args.objective), 'S_J': _cost(mS, args.objective),
            'R_delta': _cost(mR, args.objective) - _cost(base, args.objective),
            'S_delta': _cost(mS, args.objective) - _cost(base, args.objective),
            'swap_effect': _cost(mS, args.objective) - _cost(mR, args.objective),
            'base_D': float(base['distance_cost']), 'R_D': float(mR['distance_cost']),
            'S_D': float(mS['distance_cost']),
            'base_Q': float(base['quality_loss']), 'R_Q': float(mR['quality_loss']),
            'S_Q': float(mS['quality_loss']),
            'base_E': float(base['energy_kwh']), 'R_E': float(mR['energy_kwh']),
            'S_E': float(mS['energy_kwh']),
            'base_service_ok': bool(service_ok(base, args.objective)),
            'R_service_ok': bool(service_ok(mR, args.objective)),
            'S_service_ok': bool(service_ok(mS, args.objective)),
            'S_hard': hard_vector_from_outcome(mS),
            'R_n_attempts': int(repR.online_stats.get('n_attempts', 0)),
            'S_n_attempts': int(repS.online_stats.get('n_attempts', 0)),
            'S_n_swap': int(repS.online_stats.get('n_swap', 0)),
            'R_n_improve': int(repR.online_stats.get('n_improve', 0)),
            'S_n_improve': int(repS.online_stats.get('n_improve', 0)),
            'R_elapsed': float(sum(e.get('replanner_elapsed_s', 0.0) for e in repR.event_log)),
            'S_elapsed': float(sum(e.get('replanner_elapsed_s', 0.0) for e in repS.event_log)),
            'R_timing': _timing(repR.event_log), 'S_timing': _timing(repS.event_log),
        })
        print(f"  [inst {i}] base={rows[-1]['base_J']:.4f} R={rows[-1]['R_J']:.4f} "
              f"S={rows[-1]['S_J']:.4f} Rdelta={rows[-1]['R_delta']:+.4f} "
              f"Sdelta={rows[-1]['S_delta']:+.4f} swapdelta={rows[-1]['swap_effect']:+.4f} "
              f"n_swap={rows[-1]['S_n_swap']} svc={rows[-1]['S_service_ok']}", flush=True)
        for e in repS.event_log:
            e['inst_idx'] = i
        all_event_logs.extend(repS.event_log)

    r_deltas = [r['R_delta'] for r in rows]
    s_deltas = [r['S_delta'] for r in rows]
    swap_effects = [r['swap_effect'] for r in rows]
    comp = {}
    for k in ('D', 'Q', 'E'):
        b = [r[f'base_{k}'] for r in rows]
        mr = [r[f'R_{k}'] for r in rows]
        ms = [r[f'S_{k}'] for r in rows]
        comp[k] = {'mean_base': float(np.mean(b)), 'mean_R': float(np.mean(mr)),
                   'mean_S': float(np.mean(ms)),
                   'R_delta': float(np.mean(np.asarray(mr) - np.asarray(b))),
                   'S_delta': float(np.mean(np.asarray(ms) - np.asarray(b)))}
    summary = {
        'budget': args.budget, 'n_rounds': args.n_rounds, 'max_attempts': args.max_attempts,
        'seed': args.seed, 'n': n,
        'mean_R_delta': float(np.mean(r_deltas)), 'R_ci': list(paired_bootstrap_ci(r_deltas)),
        'mean_S_delta': float(np.mean(s_deltas)), 'S_ci': list(paired_bootstrap_ci(s_deltas)),
        'mean_swap_effect': float(np.mean(swap_effects)),
        'swap_effect_ci': list(paired_bootstrap_ci(swap_effects)),
        'base_service_ok': all(r['base_service_ok'] for r in rows),
        'R_service_ok': all(r['R_service_ok'] for r in rows),
        'S_service_ok': all(r['S_service_ok'] for r in rows),
        'component_deltas': comp,
        'R_n_attempts_total': int(sum(r['R_n_attempts'] for r in rows)),
        'S_n_attempts_total': int(sum(r['S_n_attempts'] for r in rows)),
        'S_n_swap_total': int(sum(r['S_n_swap'] for r in rows)),
        'R_elapsed_total': float(sum(r['R_elapsed'] for r in rows)),
        'S_elapsed_total': float(sum(r['S_elapsed'] for r in rows)),
        'rows': rows,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'event_log.json'), 'w') as f:
        json.dump(all_event_logs, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'baseline_J', 'R_J', 'S_J', 'R_delta', 'S_delta',
                    'swap_effect', 'base_D', 'R_D', 'S_D', 'base_Q', 'R_Q', 'S_Q',
                    'base_E', 'R_E', 'S_E', 'R_service', 'S_service', 'S_n_swap',
                    'R_time_p50', 'S_time_p50', 'R_elapsed', 'S_elapsed'])
        for r in rows:
            w.writerow([r['inst_idx'], f"{r['base_J']:.4f}", f"{r['R_J']:.4f}",
                        f"{r['S_J']:.4f}", f"{r['R_delta']:+.4f}", f"{r['S_delta']:+.4f}",
                        f"{r['swap_effect']:+.4f}", f"{r['base_D']:.4f}", f"{r['R_D']:.4f}",
                        f"{r['S_D']:.4f}", f"{r['base_Q']:.4f}", f"{r['R_Q']:.4f}",
                        f"{r['S_Q']:.4f}", f"{r['base_E']:.4f}", f"{r['R_E']:.4f}",
                        f"{r['S_E']:.4f}", int(r['R_service_ok']), int(r['S_service_ok']),
                        r['S_n_swap'], f"{r['R_timing']['p50']:.4f}",
                        f"{r['S_timing']['p50']:.4f}", f"{r['R_elapsed']:.4f}",
                        f"{r['S_elapsed']:.4f}"])
    print(f"\n=== swap gate (budget={args.budget}s, n_rounds={args.n_rounds}, "
          f"max_attempts={args.max_attempts}, n={n}) ===")
    print(f"  R:  delta={summary['mean_R_delta']:+.4f} CI={[round(x,4) for x in summary['R_ci']]} "
          f"svc={summary['R_service_ok']}")
    print(f"  S:  delta={summary['mean_S_delta']:+.4f} CI={[round(x,4) for x in summary['S_ci']]} "
          f"svc={summary['S_service_ok']}")
    print(f"  swap_effect (S-R)={summary['mean_swap_effect']:+.4f} "
          f"CI={[round(x,4) for x in summary['swap_effect_ci']]}")
    print(f"  D/Q/E: D R={comp['D']['R_delta']:+.4f} S={comp['D']['S_delta']:+.4f} | "
          f"Q R={comp['Q']['R_delta']:+.4f} S={comp['Q']['S_delta']:+.4f} | "
          f"E R={comp['E']['R_delta']:+.4f} S={comp['E']['S_delta']:+.4f}")
    print(f"  n_attempts R={summary['R_n_attempts_total']} S={summary['S_n_attempts_total']} "
          f"n_swap={summary['S_n_swap_total']} elapsed R={summary['R_elapsed_total']:.2f} "
          f"S={summary['S_elapsed_total']:.2f}")
    print(f"saved: {args.out}/summary.json + per_instance.csv + event_log.json")


if __name__ == '__main__':
    main()
