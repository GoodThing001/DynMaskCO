"""中心站组合路由头腔诊断（站点作为序列中可选节点）。

机制（重新设计，制造真正的组合决策）：
- 站点放在区域角落（非质心，绕行有意义），订单田间温度装车（未预冷，快衰减）。
- 车辆路由 = [depot → 取货... → (站点) → 取货... → depot]。站点批量预冷有规模经济（折扣）。
- 决策：站点插入序列哪个位置（绕站前取的订单批量预冷、绕站后取的田间预冷）。
- headroom(k) = 折扣节省(前k单) − 绕行距离(k) − 暖衰减(前k单在绕行期间的快衰减)。

用法：
    python scripts/evaluation/run_station_routing.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --out results/m0_scale/station_routing
"""
import argparse
import json
import math
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
from run_headroom_census import _load_profile


REF_RATE = {0: 0.0010, 1: 0.0020, 2: 0.0035}
ACT = {0: 3500.0, 1: 4500.0, 2: 5500.0}
T_REF = 277.15
T_FIELD = 298.15
T_STOR = {0: 291.15, 1: 277.15, 2: 255.15}
PV = {0: 1.0, 1: 1.5, 2: 2.0}


def rate(cls, temp_k):
    return REF_RATE[cls] * math.exp(ACT[cls] * (1.0 / T_REF - 1.0 / temp_k))


def _routes(dataset, contract, capacity, num_vehicles, max_instances):
    """跑 baseline，返回每实例每辆车的取货序列 [(customer, x, y, pickup_time), ...]"""
    routes = []
    for inst in range(max_instances):
        env = StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        traces, served = env.run(inst)
        inst_routes = []
        coords = env.coords[inst]
        for tr in traces:
            seq = []
            for sr in tr.services:
                c = int(sr.node)
                if c <= 0:
                    continue
                seq.append((c, float(coords[c, 0]), float(coords[c, 1]), float(sr.service_finish)))
            if seq:
                inst_routes.append(seq)
        routes.append(inst_routes)
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--station', default='0.1,0.1')
    ap.add_argument('--discount', type=float, default=0.5)
    ap.add_argument('--cp-field', type=float, default=0.15)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    base = apply_objective_profile(default_pilot_contract(), profile)
    dist_scale = profile.distance_scale
    energy_scale = profile.energy_scale
    quality_scale = profile.quality_scale
    speed = base.units.speed_kmph
    temp_cls = dataset['temp_class']
    dem = dataset['demands']
    initial_q = dataset['initial_quality']

    sx, sy = [float(x) for x in args.station.split(',')]
    station = (sx, sy)

    routes = _routes(dataset, base, args.capacity, args.num_vehicles, args.max_instances)

    def d(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # 每辆车的最佳站点位置 headroom
    all_headrooms = []  # 每辆车的最优 headroom
    n_positions = 0
    for inst, inst_routes in enumerate(routes):
        for seq in inst_routes:
            n = len(seq)
            if n < 2:
                continue
            # 预冷能耗 per order (field)
            precool_kwh = []
            for (c, x, y, t) in seq:
                cls = int(temp_cls[inst, c])
                dT = max(0.0, 25.0 - (T_STOR[cls] - 273.15))
                precool_kwh.append(float(dem[inst, c]) * args.cp_field * dT)
            # 每订单 field 与 storage 衰减率差（暖衰减用）
            rate_diff = []
            for (c, x, y, t) in seq:
                cls = int(temp_cls[inst, c])
                rate_diff.append(rate(cls, T_FIELD) - rate(cls, T_STOR[cls]))
            best_h = -1e9
            best_k = -1
            for k in range(0, n + 1):
                # 绕行距离：depot -> [c1..ck] -> 站点 -> [c_{k+1}..cn] -> depot 的额外距离
                if k == 0:
                    detour = d((0.5, 0.5), station) + d(station, (seq[0][1], seq[0][2])) - d((0.5, 0.5), (seq[0][1], seq[0][2]))
                elif k == n:
                    detour = d((seq[-1][1], seq[-1][2]), station) + d(station, (0.5, 0.5)) - d((seq[-1][1], seq[-1][2]), (0.5, 0.5))
                else:
                    detour = d((seq[k-1][1], seq[k-1][2]), station) + d(station, (seq[k][1], seq[k][2])) - d((seq[k-1][1], seq[k-1][2]), (seq[k][1], seq[k][2]))
                detour = max(0.0, detour)
                # 折扣节省（前 k 单批量预冷）
                save_energy = (1 - args.discount) * sum(precool_kwh[:k])
                # 暖衰减（前 k 单在绕行时间内的快衰减）
                detour_time = detour / speed
                warm_q = 0.0
                for i in range(k):
                    c = seq[i][0]
                    cls = int(temp_cls[inst, c])
                    warm_q += PV[cls] * dem[inst, c] * rate_diff[i] * detour_time
                # headroom（归一化）
                h = (save_energy / energy_scale) - (detour / dist_scale) - (warm_q / quality_scale)
                if h > best_h:
                    best_h = h
                    best_k = k
            all_headrooms.append(best_h)
            n_positions += 1

    all_headrooms = np.array(all_headrooms)
    summary = {
        'station': args.station, 'discount': args.discount, 'cp_field': args.cp_field,
        'n_vehicles_routes': int(n_positions),
        'headroom_mean': float(all_headrooms.mean()),
        'headroom_median': float(np.median(all_headrooms)),
        'headroom_p90': float(np.percentile(all_headrooms, 90)),
        'n_positive': int(np.sum(all_headrooms > 1e-6)),
        'n_negative': int(np.sum(all_headrooms < -1e-6)),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== station routing combinatorial headroom (station={args.station}) ===")
    print(f"  n_routes={n_positions}  headroom mean={summary['headroom_mean']:.4f} "
          f"median={summary['headroom_median']:.4f} p90={summary['headroom_p90']:.4f}")
    print(f"  positive={summary['n_positive']} negative={summary['n_negative']}")
    print(f"saved: {args.out}/summary.json")


if __name__ == '__main__':
    main()
