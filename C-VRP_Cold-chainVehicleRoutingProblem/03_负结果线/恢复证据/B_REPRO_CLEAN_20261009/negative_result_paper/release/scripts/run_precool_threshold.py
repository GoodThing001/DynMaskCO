"""预冷证伪门 · 第二步：选择性预冷（按 temp_class 阈值）的头腔曲线。

阈值 precool_min_temp_class = k 表示"只预冷 temp_class >= k 的订单"：
  k=0 全预冷、k=1 预冷冷藏+冷冻、k=2 只预冷冷冻、k=3 全不预冷。
测阈值决策是否规则可捕获（若最优 k 有明显头腔且 k 是简单规则 → 规则可捕获）。

用法：
    python scripts/evaluation/run_precool_threshold.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/precool_threshold
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
    costs, energies, qualities = [], [], []
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        o = _eval(env, inst, traces, 'coldchain', served_mask=served)
        costs.append(float(o['coldchain_cost']))
        energies.append(float(o['energy_kwh']))
        qualities.append(float(o['quality_loss']))
    return {'cost': float(np.mean(costs)), 'energy': float(np.mean(energies)),
            'quality': float(np.mean(qualities))}


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
    for k in (0, 1, 2, 3):
        contract = dc_replace(base, thermal=dc_replace(base.thermal, precool_min_temp_class=k))
        r = _run_baseline(dataset, contract, args.capacity, args.num_vehicles,
                          args.max_instances)
        results[f'k{k}'] = r
        print(f"  k={k}: cost={r['cost']:.4f} quality={r['quality']:.3f} energy={r['energy']:.1f}",
              flush=True)

    with open(os.path.join(args.out, 'sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)

    costs = [results[f'k{k}']['cost'] for k in (0, 1, 2, 3)]
    print(f"\n=== precool threshold sweep ===")
    print(f"  cost curve k=0..3: {[round(c,4) for c in costs]}")
    print(f"  best k = {int(np.argmin(costs))}, headroom over k=3 = {costs[3]-min(costs):.4f}")
    print(f"  headroom over k=0 (always precool) = {costs[0]-min(costs):.4f}")
    print(f"saved: {args.out}/sweep.json")


if __name__ == '__main__':
    main()
