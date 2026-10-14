"""Step 5 闭环 gate：B / R / M-pre / M-trained 四方法比较。

主指标 ΔJ = J_method − J_baseline（负=改善），实例等权；服务/硬约束优先。
M-pre 与 M-trained 共用统一逐步策略（argmax）与相同预算/认证/接受；M-pre=预训练 CVRP backbone，
M-trained=训练后全模型。报告逐实例 D/Q/E、ΔJ 配对 CI、候选/改进/耗时。

用法（服务器）：
    python scripts/evaluation/run_step5_gate.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
        --model-ckpt results/m0_scale/mpre_reinforce_s42/model.ckpt \
        --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --budget 4.0 --max-instances 16 --out results/m0_scale/step5_cal
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

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from cc_lns_replanner import CCLNSReplanner
from mtrained_replanner import (load_trained_model, _make_extract, mpre_score_fn,
                                mtrained_score_fn, PolicyReplanner)
from mpre import load_cvrp_model
from counterfactual_teacher import _eval
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)
from hard_gate import hard_vector_from_outcome, service_ok, outcome_protocol_error
from service_first import paired_bootstrap_ci


def _load_profile(path):
    with open(path) as f:
        d = json.load(f)
    if d.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                            quality_scale=float(d['quality_scale']),
                            energy_scale=float(d['energy_scale']),
                            lambda_quality=float(d['lambda_quality']),
                            lambda_energy=float(d['lambda_energy']),
                            scale_source=d.get('scale_source', 'pilot'),
                            dev_statistics=d.get('dev_statistics'))


def _cost(o, objective):
    return float(o['distance_cost'] if objective == 'distance' else o['coldchain_cost'])


def _dummy_state():
    """合成最小状态，用于预编译 score_fn（JIT 编译不计入在线预算）。"""
    import numpy as _np
    return {
        'raw_3d': _np.zeros((1, 1, 3), _np.float32),
        'node_valid': _np.ones((1, 1), bool),
        'node_feats': _np.zeros((1, 1, 10), _np.float32),
        'adjmat': _np.zeros((1, 1, 1), _np.float32),
        'cust': _np.zeros((1, 1), _np.int32),
        'pred': _np.zeros((1, 1), _np.int32),
        'succ': _np.zeros((1, 1), _np.int32),
        'action_feats': _np.zeros((1, 1, 20), _np.float32),
        'timestep': 0.0,
    }


def _warmup(args):
    """预编译 M-pre / M-trained 的 padded score_fn，返回编译耗时（单列，不计在线预算）。"""
    import time
    t0 = time.perf_counter()
    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=args.seed)
    dummy = _dummy_state()
    mpre_score_fn(backbone, 51, 2048)(dummy)
    mtrained_score_fn(model, 51, 2048)(dummy)
    return float(time.perf_counter() - t0)


def _timing(event_log):
    t = [e['elapsed_s'] for e in event_log]
    if not t:
        return {'p50': 0.0, 'p95': 0.0, 'total': 0.0}
    a = np.asarray(t)
    return {'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'total': float(a.sum())}


def _run(dataset, i, eff, args, method):
    if method == 'B':
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=make_continuation(), coldchain_contract=eff)
        t, s = env.run(i)
        return _eval(env, i, t, args.objective, served_mask=s), None
    if method == 'R':
        rep = CCLNSReplanner(budget_s=args.budget, contract=eff, seed=args.seed,
                             capacity=args.capacity, num_vehicles=args.num_vehicles,
                             guard_reserved=True)
    elif method == 'Mpre':
        backbone = load_cvrp_model(args.cvrp_ckpt)[0]
        rep = PolicyReplanner(mpre_score_fn(backbone, 51, 2048), _make_extract, contract=eff,
                              budget_s=args.budget, seed=args.seed, capacity=args.capacity,
                              num_vehicles=args.num_vehicles, guard_reserved=True,
                              tw_max=args.tw_max)
    elif method == 'Mtrained':
        model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=args.seed)
        rep = PolicyReplanner(mtrained_score_fn(model, 51, 2048), _make_extract, contract=eff,
                              budget_s=args.budget, seed=args.seed, capacity=args.capacity,
                              num_vehicles=args.num_vehicles, guard_reserved=True,
                              tw_max=args.tw_max)
    else:
        raise ValueError(method)
    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                          replanner=rep, coldchain_contract=eff)
    t, s = env.run(i)
    m = _eval(env, i, t, args.objective, served_mask=s)
    err = outcome_protocol_error(m, args.objective, strict_repair=True)
    if err is not None:
        raise RuntimeError(f"inst {i} {method} PROTOCOL_ERROR {err}")
    return m, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--budget', type=float, default=4.0)
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
    args.tw_max = float(dataset['tw_end'][:, 0].max())
    methods = ['B', 'R', 'Mpre', 'Mtrained']

    compile_s = _warmup(args)
    print(f"warm-up compile: {compile_s:.1f}s (excluded from online budget)", flush=True)

    rows = []
    for i in range(n):
        res = {}
        for method in methods:
            m, rep = _run(dataset, i, eff, args, method)
            res[method] = {'J': _cost(m, args.objective),
                           'D': float(m['distance_cost']), 'Q': float(m['quality_loss']),
                           'E': float(m['energy_kwh']),
                           'svc': bool(service_ok(m, args.objective)),
                           'hard': hard_vector_from_outcome(m),
                           'attempts': int(rep.online_stats.get('n_attempts', 0)) if rep else 0,
                           'improve': int(rep.online_stats.get('n_improve', 0)) if rep else 0,
                           'elapsed': (sum(e.get('replanner_elapsed_s', 0.0) for e in rep.event_log)
                                       if rep else 0.0),
                           'timing': _timing(rep.event_log) if rep else {'p50': 0.0, 'p95': 0.0, 'total': 0.0}}
        row = {'inst': i}
        for method in methods:
            row[f'{method}_J'] = res[method]['J']
            row[f'{method}_delta'] = res[method]['J'] - res['B']['J']
            row[f'{method}_svc'] = res[method]['svc']
            for k in 'DQE':
                row[f'{method}_{k}'] = res[method][k]
        row['R_vs_Mpre'] = res['Mpre']['J'] - res['R']['J']
        row['R_vs_Mtrained'] = res['Mtrained']['J'] - res['R']['J']
        row['Mpre_vs_Mtrained'] = res['Mtrained']['J'] - res['Mpre']['J']
        row['Mtrained_attempts'] = res['Mtrained']['attempts']
        row['Mtrained_improve'] = res['Mtrained']['improve']
        row['Mtrained_elapsed'] = res['Mtrained']['elapsed']
        rows.append(row)
        print(f"  [inst {i}] B={res['B']['J']:.4f} R={res['R']['J']:.4f} "
              f"Mpre={res['Mpre']['J']:.4f} Mtrained={res['Mtrained']['J']:.4f} "
              f"R-Mtr={row['R_vs_Mtrained']:+.4f} Mpre-Mtr={row['Mpre_vs_Mtrained']:+.4f} "
              f"svc={res['Mtrained']['svc']}", flush=True)

    summary = {'budget': args.budget, 'n': n, 'rows': rows, 'compile_s': compile_s}
    for method in methods:
        deltas = [r[f'{method}_delta'] for r in rows]
        summary[f'{method}_mean_delta'] = float(np.mean(deltas))
        summary[f'{method}_ci'] = list(paired_bootstrap_ci(deltas))
        summary[f'{method}_svc'] = all(r[f'{method}_svc'] for r in rows)
    for pair in ['R_vs_Mpre', 'R_vs_Mtrained', 'Mpre_vs_Mtrained']:
        ds = [r[pair] for r in rows]
        summary[f'{pair}_mean'] = float(np.mean(ds))
        summary[f'{pair}_ci'] = list(paired_bootstrap_ci(ds))
    summary['Mtrained_attempts_total'] = int(sum(r['Mtrained_attempts'] for r in rows))
    summary['Mtrained_improve_total'] = int(sum(r['Mtrained_improve'] for r in rows))
    summary['Mtrained_elapsed_total'] = float(sum(r['Mtrained_elapsed'] for r in rows))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['inst', 'B_J', 'R_J', 'Mpre_J', 'Mtrained_J', 'R_delta', 'Mpre_delta',
                    'Mtrained_delta', 'R_vs_Mtrained', 'Mpre_vs_Mtrained', 'Mtr_attempts',
                    'Mtr_improve', 'Mtr_elapsed'])
        for r in rows:
            w.writerow([r['inst'], f"{r['B_J']:.4f}", f"{r['R_J']:.4f}", f"{r['Mpre_J']:.4f}",
                        f"{r['Mtrained_J']:.4f}", f"{r['R_delta']:+.4f}", f"{r['Mpre_delta']:+.4f}",
                        f"{r['Mtrained_delta']:+.4f}", f"{r['R_vs_Mtrained']:+.4f}",
                        f"{r['Mpre_vs_Mtrained']:+.4f}", r['Mtrained_attempts'],
                        r['Mtrained_improve'], f"{r['Mtrained_elapsed']:.2f}"])
    print(f"\n=== Step5 gate (budget={args.budget}s, n={n}) ===")
    for method in methods:
        print(f"  {method}: Δ={summary[f'{method}_mean_delta']:+.4f} "
              f"CI={[round(x,4) for x in summary[f'{method}_ci']]} svc={summary[f'{method}_svc']}")
    for pair in ['R_vs_Mpre', 'R_vs_Mtrained', 'Mpre_vs_Mtrained']:
        print(f"  {pair}: mean={summary[f'{pair}_mean']:+.4f} "
              f"CI={[round(x,4) for x in summary[f'{pair}_ci']]}")
    print(f"saved: {args.out}/summary.json + per_instance.csv")


if __name__ == '__main__':
    main()
