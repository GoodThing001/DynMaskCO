"""O0-CC 前置诊断：coldchain 目标分量的 scale 合理性。

统计 baseline（JF1-H）在一个数据文件上 distance/quality/energy 的 raw 量级，判断 pilot
参数是否让 energy 过度主导 coldchain_cost。本脚本只作「诊断」；正式冻结 scale/λ 请用
`freeze_coldchain_scale.py`（在独立 DEV-CAL、多场景等权，输出具名 objective profile）。

注意（阻塞项 4）：诊断用的数据只能是 DEV-CAL 或明确的 split；不能用 O0-CC 判定用的 VAL
来冻结 scale。均值归一化后 baseline 的三分量贡献为 (1, λ_Q, λ_E)。

用法（纯 NumPy，baseline 很快）：
    python scripts/evaluation/diagnose_coldchain_scale.py \
        --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
        --capacity 50 --num_vehicles 25 --max_instances 64 \
        --out results/o0cc/scale_diag
"""
import argparse, os, csv, json, sys
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
from joint_fleet import JointAssignmentReplanner
from counterfactual_teacher import _eval
from coldchain_contract import default_pilot_contract


def main():
    parser = argparse.ArgumentParser(description='coldchain scale 合理性诊断')
    parser.add_argument('--data', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--max_instances', type=int, default=64)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    n = min(args.max_instances, dataset['coords'].shape[0])
    contract = default_pilot_contract()
    continuation = JointAssignmentReplanner('heuristic')

    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                          replanner=continuation, coldchain_contract=contract)

    dists, quals, energies, costs = [], [], [], []
    for i in range(n):
        traces, _ = env.run(i)
        out = _eval(env, i, traces, 'coldchain')
        dists.append(float(out['distance_cost']))
        quals.append(float(out['quality_loss']))
        energies.append(float(out['energy_kwh']))
        costs.append(float(out['coldchain_cost']))

    dists = np.array(dists); quals = np.array(quals); energies = np.array(energies)
    costs = np.array(costs)

    def stat(x):
        return float(np.mean(x)), float(np.std(x))

    d_mean, d_std = stat(dists)
    q_mean, q_std = stat(quals)
    e_mean, e_std = stat(energies)
    c_mean, c_std = stat(costs)

    # 当前 pilot 归一化贡献（÷scale ×λ）
    sc = contract.objective
    contrib_d = d_mean / sc.distance_scale
    contrib_q = sc.lambda_quality * q_mean / sc.quality_scale
    contrib_e = sc.lambda_energy * e_mean / sc.energy_scale
    total = contrib_d + contrib_q + contrib_e

    # 均值归一化后，baseline 三分量贡献 = (1.0, λ_Q, λ_E)（均值/scale = 1.0 定义）。
    # 纠正旧实现的语义错误：不能用 pilot 的绝对贡献比当作「建议贡献比」。
    sugg = {'distance': 1.0, 'quality': sc.lambda_quality, 'energy': sc.lambda_energy}

    result = {
        'n': n,
        'raw_mean': {'distance_cost': d_mean, 'quality_loss': q_mean, 'energy_kwh': e_mean,
                     'coldchain_cost': c_mean},
        'raw_std': {'distance_cost': d_std, 'quality_loss': q_std, 'energy_kwh': e_std},
        'pilot_scales': {
            'distance_scale': sc.distance_scale, 'quality_scale': sc.quality_scale,
            'energy_scale': sc.energy_scale, 'lambda_quality': sc.lambda_quality,
            'lambda_energy': sc.lambda_energy},
        'pilot_normalized_contribution': {
            'distance': contrib_d, 'quality': contrib_q, 'energy': contrib_e},
        'pilot_contribution_fraction': {
            'distance': contrib_d / total, 'quality': contrib_q / total,
            'energy': contrib_e / total},
        'energy_dominates': bool(contrib_e / total > 0.5),
        'suggested_scale_mean_normalized': {
            'distance_scale': d_mean, 'quality_scale': q_mean, 'energy_scale': e_mean,
            'lambda_quality': sc.lambda_quality, 'lambda_energy': sc.lambda_energy},
        'suggested_contribution_ratio': sugg,
    }

    print(f"=== coldchain scale 诊断 (n={n}) ===")
    print(f"  raw 均值: distance={d_mean:.3f}  quality={q_mean:.4f}  energy={e_mean:.2f}  "
          f"coldchain_cost={c_mean:.3f}")
    print(f"  raw std:  distance={d_std:.3f}  quality={q_std:.4f}  energy={e_std:.2f}")
    print(f"  pilot 归一化贡献: distance={contrib_d:.3f}  quality={contrib_q:.3f}  "
          f"energy={contrib_e:.3f}")
    print(f"  pilot 贡献占比: distance={contrib_d/total:.1%}  quality={contrib_q/total:.1%}  "
          f"energy={contrib_e/total:.1%}")
    print(f"  energy 主导(>50%)? {result['energy_dominates']}")
    print(f"  建议(均值归一化) scale: distance={d_mean:.3f}  quality={q_mean:.4f}  "
          f"energy={e_mean:.2f}（此时三分量 ~O(1)，λ 才有相对权重意义）")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'scale_diag.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, 'per_instance_raw.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'distance_cost', 'quality_loss', 'energy_kwh', 'coldchain_cost'])
        for i in range(n):
            w.writerow([i, f"{dists[i]:.4f}", f"{quals[i]:.4f}", f"{energies[i]:.4f}",
                        f"{costs[i]:.4f}"])
    print(f"  saved: {args.out}/scale_diag.json + per_instance_raw.csv")


if __name__ == '__main__':
    main()
