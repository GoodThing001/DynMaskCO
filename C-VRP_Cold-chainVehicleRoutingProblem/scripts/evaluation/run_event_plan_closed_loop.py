"""event_plan_v2：CAL→DEV 闭环驱动（阈值网格 + 锁定 + DEV 主结果）。

对每个方法（distance / proxy / learned A/B），在 CAL-16 上跑阈值网格 {0, 0.005, 0.02, inf}，
按 service-first 选最小 mean ΔJ（平局取较大 τ），再在 DEV-16 上跑锁定 τ。distance 无阈值，
直接在 DEV 跑一次。baseline 各数据集只跑一次并复用；全部 seed 保留。

用法：
    python scripts/evaluation/run_event_plan_closed_loop.py \
        --data-cal data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --data-dev data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --capacity 50 --num-vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --A42 results/m0_scale/event_plan_v2_A_s42/model.ckpt \
        --A43 results/m0_scale/event_plan_v2_A_s43/model.ckpt \
        --A44 results/m0_scale/event_plan_v2_A_s44/model.ckpt \
        --B42 ... --B43 ... --B44 ... \
        --out results/m0_scale/closed_loop
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

TAU_GRID = [0.0, 0.005, 0.02, float('inf')]


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


def _run_baseline(dataset, n, capacity, num_vehicles, objective, eff):
    rows = []
    for i in range(n):
        benv = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                               replanner=make_continuation(), coldchain_contract=eff)
        tb, sb = benv.run(i)
        base = _eval(benv, i, tb, objective, served_mask=sb)
        rows.append({'inst_idx': i, 'J': _cost(base, objective),
                     'D': float(base['distance_cost']), 'Q': float(base['quality_loss']),
                     'E': float(base['energy_kwh']),
                     'service_ok': bool(service_ok(base, objective)),
                     'hard': hard_vector_from_outcome(base)})
    return rows


def _run_method(dataset, n, capacity, num_vehicles, objective, eff, mode, representation,
                scorer, tau, seed):
    rows = []
    t0 = time.time()
    for i in range(n):
        collector = PrePrepareContextCollector(deindex=True)
        replanner = EventPlanReplanner(
            scorer=scorer, mode=mode, representation=representation, deindex=True, tau=tau,
            capacity=capacity, num_vehicles=num_vehicles, context_source=collector,
            seed=seed, contract=eff)
        menv = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                               replanner=replanner, coldchain_contract=eff)
        menv.snapshot_hook = collector.hook
        tm, sm = menv.run(i)
        m = _eval(menv, i, tm, objective, served_mask=sm)
        err = outcome_protocol_error(m, objective, strict_repair=True)
        if err is not None:
            raise RuntimeError(f"inst {i} PROTOCOL_ERROR {err}")
        rows.append({'inst_idx': i, 'J': _cost(m, objective),
                     'D': float(m['distance_cost']), 'Q': float(m['quality_loss']),
                     'E': float(m['energy_kwh']),
                     'service_ok': bool(service_ok(m, objective)),
                     'accept': replanner.online_stats.get('n_accept'),
                     'n_events': replanner.online_stats.get('n_events')})
    return rows, time.time() - t0


def _paired_delta(base_rows, method_rows):
    return [m['J'] - b['J'] for b, m in zip(base_rows, method_rows)]


def _select_tau(base_rows, cal_runs):
    """cal_runs: {tau: method_rows}。service-first 选最小 mean ΔJ，平局取较大 τ。"""
    best_tau = None
    best_mean = None
    for tau in sorted(TAU_GRID):
        rows = cal_runs[tau]
        if not all(r['service_ok'] for r in rows):
            continue
        mean = float(np.mean(_paired_delta(base_rows, rows)))
        if best_mean is None or mean < best_mean - 1e-12:
            best_mean = mean
            best_tau = tau
        elif abs(mean - best_mean) <= 1e-12 and tau > (best_tau if best_tau is not None else -1):
            # 平局取较大 τ（inf 最大）
            best_tau = tau
    return best_tau, best_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-cal', required=True)
    ap.add_argument('--data-dev', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--A42', default=None); ap.add_argument('--A43', default=None)
    ap.add_argument('--A44', default=None)
    ap.add_argument('--B42', default=None); ap.add_argument('--B43', default=None)
    ap.add_argument('--B44', default=None)
    ap.add_argument('--max-instances-cal', type=int, default=16)
    ap.add_argument('--max-instances-dev', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cal_ds = dict(np.load(args.data_cal))
    dev_ds = dict(np.load(args.data_dev))
    n_cal = min(args.max_instances_cal, cal_ds['coords'].shape[0])
    n_dev = min(args.max_instances_dev, dev_ds['coords'].shape[0])
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None

    # baseline（各数据集一次）
    cal_base = _run_baseline(cal_ds, n_cal, args.capacity, args.num_vehicles, args.objective, eff)
    dev_base = _run_baseline(dev_ds, n_dev, args.capacity, args.num_vehicles, args.objective, eff)

    report = {'baseline_cal_mean_J': float(np.mean([r['J'] for r in cal_base])),
              'baseline_dev_mean_J': float(np.mean([r['J'] for r in dev_base])),
              'methods': {}}

    def _summarize(base_rows, method_rows, runtime):
        d = _paired_delta(base_rows, method_rows)
        return {'mean_delta': float(np.mean(d)), 'ci': list(paired_bootstrap_ci(d)),
                'service_ok': all(r['service_ok'] for r in method_rows),
                'accept_total': int(sum(r['accept'] for r in method_rows)),
                'D_delta': float(np.mean([m['D'] - b['D'] for b, m in zip(base_rows, method_rows)])),
                'Q_delta': float(np.mean([m['Q'] - b['Q'] for b, m in zip(base_rows, method_rows)])),
                'E_delta': float(np.mean([m['E'] - b['E'] for b, m in zip(base_rows, method_rows)])),
                'runtime': runtime}

    # distance（DEV 一次）
    dev_rows, rt = _run_method(dev_ds, n_dev, args.capacity, args.num_vehicles, args.objective,
                               eff, 'distance', 'A', None, 0.0, 0)
    report['methods']['distance'] = {'mode': 'distance', 'dev': _summarize(dev_base, dev_rows, rt)}

    # proxy（CAL 网格 → DEV）
    cal_runs = {}
    for tau in TAU_GRID:
        rows, _ = _run_method(cal_ds, n_cal, args.capacity, args.num_vehicles, args.objective,
                              eff, 'proxy', 'A', None, tau, 0)
        cal_runs[tau] = rows
    tau_s, _ = _select_tau(cal_base, cal_runs)
    dev_rows, rt = _run_method(dev_ds, n_dev, args.capacity, args.num_vehicles, args.objective,
                               eff, 'proxy', 'A', None, tau_s, 0)
    report['methods']['proxy'] = {'mode': 'proxy', 'tau_selected': tau_s,
                                  'cal': {str(t): _summarize(cal_base, r, 0.0)
                                          for t, r in cal_runs.items()},
                                  'dev': _summarize(dev_base, dev_rows, rt)}

    # learned A/B（每 seed：CAL 网格 → DEV）
    for rep in ('A', 'B'):
        for sd in (42, 43, 44):
            ckpt = getattr(args, f'{rep}{sd}')
            if ckpt is None:
                continue
            scorer = load_plan_scorer(ckpt)
            cal_runs = {}
            for tau in TAU_GRID:
                rows, _ = _run_method(cal_ds, n_cal, args.capacity, args.num_vehicles,
                                      args.objective, eff, 'learned', rep, scorer, tau, sd)
                cal_runs[tau] = rows
            tau_s, _ = _select_tau(cal_base, cal_runs)
            dev_rows, rt = _run_method(dev_ds, n_dev, args.capacity, args.num_vehicles,
                                       args.objective, eff, 'learned', rep, scorer, tau_s, sd)
            key = f'{rep}_s{sd}'
            report['methods'][key] = {'mode': 'learned', 'representation': rep, 'seed': sd,
                                      'tau_selected': tau_s,
                                      'cal': {str(t): _summarize(cal_base, r, 0.0)
                                              for t, r in cal_runs.items()},
                                      'dev': _summarize(dev_base, dev_rows, rt)}
            print(f"[{key}] tau_selected={tau_s} dev Δ={report['methods'][key]['dev']['mean_delta']:+.4f}",
                  flush=True)

    with open(os.path.join(args.out, 'closed_loop.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n=== closed loop summary ===")
    print(f"  baseline CAL={report['baseline_cal_mean_J']:.4f} DEV={report['baseline_dev_mean_J']:.4f}")
    for k, v in report['methods'].items():
        if 'dev' in v:
            print(f"  {k:8s} tau={v.get('tau_selected')} DEV Δ={v['dev']['mean_delta']:+.4f} "
                  f"CI={[round(x,4) for x in v['dev']['ci']]} accept={v['dev']['accept_total']}")
    print(f"saved: {args.out}/closed_loop.json")


if __name__ == '__main__':
    main()
