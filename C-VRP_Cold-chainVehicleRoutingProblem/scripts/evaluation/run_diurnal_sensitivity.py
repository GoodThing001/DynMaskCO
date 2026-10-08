"""昼夜环境温度敏感性诊断：测时变制冷能耗是否材料（时钟摆放头腔的上界）。

做法：固定 baseline（JF1-H）轨迹不变，只把昼夜环境温度峰值 phase 移到一天不同时刻
（amp=0 退化为常数），重跑 baseline 得终局 energy_kwh。能量跨 phase 的 spread = "若能把
载货窗口移到低温时段能省多少"的上界。spread ≈0 → 该方向判死；spread 材料 → 继续测因果分量。

用法：
    python scripts/evaluation/run_diurnal_sensitivity.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/diurnal_sensitivity
"""
import argparse
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from counterfactual_teacher import _eval
from run_headroom_census import _load_profile


def _run_baseline(dataset, contract, capacity, num_vehicles, max_instances):
    costs, energies, distances, qualities = [], [], [], []
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        o = _eval(env, inst, traces, 'coldchain', served_mask=served)
        costs.append(float(o['coldchain_cost']))
        energies.append(float(o['energy_kwh']))
        distances.append(float(o['distance_km']))
        qualities.append(float(o['quality_loss']))
    return {'cost_mean': float(np.mean(costs)), 'energy_mean': float(np.mean(energies)),
            'distance_mean': float(np.mean(distances)), 'quality_mean': float(np.mean(qualities))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--amplitudes', default='0,6')
    ap.add_argument('--phases', default='2,8,14,20')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)

    amps = [float(x) for x in args.amplitudes.split(',') if x.strip()]
    phases = [float(x) for x in args.phases.split(',') if x.strip()]

    results = {}
    for amp in amps:
        for phase in phases:
            # 构造带昼夜参数的 contract（用 replace 改 thermal 字段）
            from dataclasses import replace as dc_replace
            th = dc_replace(base.thermal, diurnal_amplitude_c=amp, diurnal_phase_hour=phase)
            contract = dc_replace(base, thermal=th)
            r = _run_baseline(dataset, contract, args.capacity, args.num_vehicles,
                              args.max_instances)
            key = f'amp{amp:g}_phase{phase:g}'
            results[key] = r
            print(f"  {key}: cost={r['cost_mean']:.4f} energy={r['energy_mean']:.2f} "
                  f"distance={r['distance_mean']:.3f} quality={r['quality_mean']:.4f}", flush=True)

    # 汇总：能量跨 phase 的 spread（同 amp）
    with open(os.path.join(args.out, 'sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== diurnal energy sensitivity ===')
    for amp in amps:
        e = [results[f'amp{amp:g}_phase{p:g}']['energy_mean'] for p in phases]
        spread = max(e) - min(e)
        rel = spread / np.mean(e) if np.mean(e) > 0 else 0.0
        print(f"  amp={amp:g}: energy across phases {[round(x,2) for x in e]}  "
              f"spread={spread:.2f} kWh ({rel*100:.1f}% of mean energy)")
    print(f"saved: {args.out}/sweep.json")


if __name__ == '__main__':
    main()
