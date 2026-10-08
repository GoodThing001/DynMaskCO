"""预冷证伪门 · 第一步：全预冷 vs 全不预冷的 myopic + 终局头腔。

物理：订单到田间为环境温度(25°C)。预冷=取货时冷却到存储温度(能耗 qty*cp*ΔT，慢衰减)；
不预冷=田间温度装车(零能耗，但 Arrhenius 衰减快 1.3–22.4×，冷冻最严重)。
跑 baseline(JF1-H)两次（precooled True/False），比较终局 coldchain_cost + D/Q/E 分量。

用法：
    python scripts/evaluation/run_precool_gate.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/precool_gate
"""
import argparse
import json
import os
import sys
from dataclasses import replace as dc_replace

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
from coldchain_contract import default_pilot_contract, apply_objective_profile
from counterfactual_teacher import _eval
from run_headroom_census import _load_profile


def _run_baseline(dataset, contract, capacity, num_vehicles, max_instances):
    costs, energies, qualities, distances = [], [], [], []
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        o = _eval(env, inst, traces, 'coldchain', served_mask=served)
        costs.append(float(o['coldchain_cost']))
        energies.append(float(o['energy_kwh']))
        qualities.append(float(o['quality_loss']))
        distances.append(float(o['distance_km']))
    return {'cost': float(np.mean(costs)), 'energy': float(np.mean(energies)),
            'quality': float(np.mean(qualities)), 'distance': float(np.mean(distances))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)

    results = {}
    for precooled in (True, False):
        contract = dc_replace(base, thermal=dc_replace(base.thermal, precooled=precooled))
        r = _run_baseline(dataset, contract, args.capacity, args.num_vehicles,
                          args.max_instances)
        results[f'precooled_{precooled}'] = r
        print(f"  precooled={precooled}: cost={r['cost']:.4f} quality={r['quality']:.3f} "
              f"energy={r['energy']:.1f} distance={r['distance']:.3f}", flush=True)

    a = results['precooled_True']
    b = results['precooled_False']
    summary = {
        'precooled_True': a, 'precooled_False': b,
        'headroom_cost': b['cost'] - a['cost'],   # >0 = 不预冷更差(预冷有效)
        'quality_delta': b['quality'] - a['quality'],
        'energy_delta': a['energy'] - b['energy'],
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== precool gate (myopic+terminal via baseline) ===")
    print(f"  headroom_cost(precool saves) = {summary['headroom_cost']:.4f}")
    print(f"  quality_delta = {summary['quality_delta']:.3f}  energy_delta = {summary['energy_delta']:.1f} kWh")
    print(f"saved: {args.out}/summary.json")


if __name__ == '__main__':
    main()
