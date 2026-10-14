"""中心站批量预冷头腔诊断（最小版：visit vs not）。

机制：中心预冷站有规模经济（折扣 precool_station_discount），但车辆绕行站点有距离代价。
田间预冷(use_station=False) vs 中心站批量预冷(use_station=True，能耗×折扣) + 绕行距离 detour。
headroom = cost(田间) − [cost(中心站能耗) + detour/distance_scale]。

用法：
    python scripts/evaluation/run_precool_station.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/precool_station
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
    ap.add_argument('--discounts', default='0.3,0.5,0.7')
    ap.add_argument('--detours', default='1,2,3,4')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)
    dist_scale = profile.distance_scale

    cost_field, energy_field = _run_baseline(dataset, base, args.capacity, args.num_vehicles,
                                             args.max_instances)
    print(f"  field precool: cost={cost_field:.4f} energy={energy_field:.1f}", flush=True)

    discounts = [float(x) for x in args.discounts.split(',') if x.strip()]
    detours = [float(x) for x in args.detours.split(',') if x.strip()]
    results = {'cost_field': cost_field, 'energy_field': energy_field, 'headroom': {}}
    for d in discounts:
        contract = dc_replace(base, thermal=dc_replace(base.thermal, use_station=True,
                                                       precool_station_discount=d))
        cost_st, energy_st = _run_baseline(dataset, contract, args.capacity, args.num_vehicles,
                                           args.max_instances)
        for detour in detours:
            cost_total = cost_st + detour / dist_scale
            headroom = cost_field - cost_total
            key = f'discount{d:g}_detour{detour:g}'
            results['headroom'][key] = {'cost_station_total': cost_total, 'headroom': headroom,
                                        'energy_station': energy_st}
            print(f"  discount={d:g} detour={detour:g}: station_cost={cost_total:.4f} "
                  f"headroom={headroom:.4f}", flush=True)

    with open(os.path.join(args.out, 'sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n=== central-station precool headroom ===")
    print(f"  field precool cost = {cost_field:.4f}")
    print(f"saved: {args.out}/sweep.json")


if __name__ == '__main__':
    main()
