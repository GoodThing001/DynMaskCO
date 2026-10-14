"""派车延迟头腔诊断：固定昼夜相位，扫"延迟派车 Δ 小时"的终局能耗曲线。

这是"时变环境温度 → 时钟摆放决策"的因果分量直接测量：dispatch_delay_hour 把每条载货腿的
时钟往后挪 Δ（等价于策略"等到更凉时段再派车"），只改能耗、不改温度动力学。headroom =
energy(Δ=0) − min_Δ energy(Δ)。曲线在 Δ 增大的初期就显著下降 → 因果可捕获的时钟摆放头腔材料。

用法：
    python scripts/evaluation/run_dispatch_delay_sweep.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/dispatch_delay_sweep
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
    costs, energies = [], []
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        o = _eval(env, inst, traces, 'coldchain', served_mask=served)
        costs.append(float(o['coldchain_cost']))
        energies.append(float(o['energy_kwh']))
    return float(np.mean(costs)), float(np.mean(energies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--phase', type=float, default=14.0)
    ap.add_argument('--delays', default='0,1,2,3,4,5,6,8')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)
    base = dc_replace(base, thermal=dc_replace(base.thermal, diurnal_amplitude_c=6.0,
                                               diurnal_phase_hour=args.phase))

    delays = [float(x) for x in args.delays.split(',') if x.strip()]
    results = {}
    for d in delays:
        contract = dc_replace(base, thermal=dc_replace(base.thermal, dispatch_delay_hour=d))
        cost, energy = _run_baseline(dataset, contract, args.capacity, args.num_vehicles,
                                     args.max_instances)
        results[f'delay{d:g}'] = {'cost': cost, 'energy': energy}
        print(f"  delay={d:g}h: cost={cost:.4f} energy={energy:.2f}", flush=True)

    costs = [results[f'delay{d:g}']['cost'] for d in delays]
    energies = [results[f'delay{d:g}']['energy'] for d in delays]
    best_i = int(np.argmin(costs))
    summary = {
        'phase': args.phase, 'delays': delays,
        'cost_curve': {f'delay{d:g}': results[f'delay{d:g}']['cost'] for d in delays},
        'energy_curve': {f'delay{d:g}': results[f'delay{d:g}']['energy'] for d in delays},
        'headroom_cost': costs[0] - min(costs),
        'headroom_energy': energies[0] - min(energies),
        'best_delay': delays[best_i],
    }
    with open(os.path.join(args.out, 'sweep.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== dispatch-delay headroom (phase={args.phase:g}) ===")
    print(f"  cost curve: {[round(c,4) for c in costs]}")
    print(f"  energy curve: {[round(e,1) for e in energies]}")
    print(f"  headroom_cost={summary['headroom_cost']:.4f}  headroom_energy={summary['headroom_energy']:.1f} kWh  best_delay={delays[best_i]:g}h")
    print(f"saved: {args.out}/sweep.json")


if __name__ == '__main__':
    main()
