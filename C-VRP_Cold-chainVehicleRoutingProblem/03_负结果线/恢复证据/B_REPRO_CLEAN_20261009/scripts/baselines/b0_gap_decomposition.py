"""
B0 Teacher-Value Diagnostic —— 分解 OR-Tools 的 17.6% gap 为 sequencing vs fleet。

三层 comparator（导师 §9）：
  NN       = GreedyReplanner('nn')
  OR-fixed = ORFixedReplanner（冻结 assignment，只优化单车 suffix sequencing）
  OR-joint = ORToolsReplanner（跨车重分配，全 fleet VRPTW）

输出：
  G_seq   = (J_NN − J_OR-fixed) / J_NN        —— sequencing 潜力
  G_fleet = (J_OR-fixed − J_OR-joint) / J_OR-fixed —— fleet allocation 潜力

用法（先小规模 smoke，再 256 train episodes）：
    python scripts/baselines/b0_gap_decomposition.py \
        --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
        --capacity 50 --num_vehicles 25 --time_limit_ms 500 \
        --num_instances 256 --out results/b0_gap_decomposition
"""

import sys, os, argparse, csv, time
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'baselines'))

from strict_online_env import StrictOnlineEnv, GreedyReplanner
from authoritative_evaluator import evaluate_execution_trace
from b0_or_expert import ORFixedReplanner
from ortools_rolling_horizon import ORToolsReplanner
from service_first import (service_equivalent_idx, paired_bootstrap_ci, win_tie_loss,
                           write_service_matrix, write_solver_status_csv)


def run_policy(name, dataset, capacity, num_vehicles, replanner, n):
    env = StrictOnlineEnv(dataset, capacity, tw_speed=1.0, num_vehicles=num_vehicles,
                          replanner=replanner)
    per_inst = []
    t0 = time.time()
    for i in range(n):
        print(f"  [{name}] instance {i}/{n} ...", flush=True)
        t_start = time.time()
        traces, served_mask = env.run(i)
        m = evaluate_execution_trace(
            traces, env.coords[i], env.tw_start[i], env.tw_end[i],
            env.service_time[i], env.demands[i], capacity, speed=1.0,
            dist_mat=env.dist_mat[i])
        m['runtime'] = time.time() - t_start
        per_inst.append(m)
        if (i + 1) % 16 == 0:
            print(f"  [{name}] {i+1}/{n} | avg_cost={np.mean([x['distance_cost'] for x in per_inst]):.2f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    return per_inst


def main():
    parser = argparse.ArgumentParser(description='B0 gap decomposition (NN / OR-fixed / OR-joint)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--time_limit_ms', type=int, default=500)
    parser.add_argument('--num_instances', type=int, default=256)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    n = min(args.num_instances, dataset['coords'].shape[0])
    os.makedirs(args.out, exist_ok=True)

    policies = [
        ('NN', GreedyReplanner('nn')),
        ('OR-fixed', ORFixedReplanner(args.capacity, args.time_limit_ms)),
        ('OR-joint', ORToolsReplanner(args.capacity, args.num_vehicles, args.time_limit_ms)),
    ]
    methods = [name for name, _ in policies]

    results = {}
    for name, replanner in policies:
        print(f"=== {name} ===", flush=True)
        results[name] = run_policy(name, dataset, args.capacity, args.num_vehicles, replanner, n)

    # ---- P0-R3: service matrix + solver coverage 先行 ----
    write_service_matrix(os.path.join(args.out, 'service_matrix.csv'), results, methods, n)
    write_solver_status_csv(os.path.join(args.out, 'solver_status_or_fixed.csv'),
                            policies[1][1].solve_status_log, n, method='OR-fixed')

    def cost_on(idx, name):
        return float(np.mean([results[name][i]['distance_cost'] for i in idx])) if idx else float('nan')

    def complete_rate(name):
        return float(np.mean([results[name][i]['complete'] for i in range(n)]))

    def cost_unpaired(name):
        rs = [results[name][i]['distance_cost'] for i in range(n) if results[name][i]['complete']]
        return float(np.mean(rs)) if rs else float('nan')

    # ---- service-equivalent 配对（P0-R3：同一 hard outcome 才比较 cost）----
    seq_idx = service_equivalent_idx(results, ['NN', 'OR-fixed'], n)
    fleet_idx = service_equivalent_idx(results, ['OR-fixed', 'OR-joint'], n)

    J_nn = cost_on(seq_idx, 'NN')
    J_fixed_seq = cost_on(seq_idx, 'OR-fixed')
    J_fixed_fleet = cost_on(fleet_idx, 'OR-fixed')
    J_joint = cost_on(fleet_idx, 'OR-joint')
    G_seq = (J_nn - J_fixed_seq) / J_nn if J_nn else float('nan')
    G_fleet = (J_fixed_fleet - J_joint) / J_fixed_fleet if J_fixed_fleet else float('nan')

    seq_deltas = [results['NN'][i]['distance_cost'] - results['OR-fixed'][i]['distance_cost']
                  for i in seq_idx]
    fleet_deltas = [results['OR-fixed'][i]['distance_cost'] - results['OR-joint'][i]['distance_cost']
                    for i in fleet_idx]
    seq_ci = paired_bootstrap_ci(seq_deltas)
    fleet_ci = paired_bootstrap_ci(fleet_deltas)
    seq_wtl = win_tie_loss(seq_deltas)
    fleet_wtl = win_tie_loss(fleet_deltas)

    print("\n=== B0 Gap Decomposition (service-equivalent paired) ===")
    print(f"  [unpaired, reference] NN={cost_unpaired('NN'):.2f} ({complete_rate('NN'):.1%})  "
          f"OR-fixed={cost_unpaired('OR-fixed'):.2f} ({complete_rate('OR-fixed'):.1%})  "
          f"OR-joint={cost_unpaired('OR-joint'):.2f} ({complete_rate('OR-joint'):.1%})")
    print(f"  [service-eq n={len(seq_idx)}] NN={J_nn:.2f}  OR-fixed={J_fixed_seq:.2f}")
    print(f"  [service-eq n={len(fleet_idx)}] OR-fixed={J_fixed_fleet:.2f}  OR-joint={J_joint:.2f}")
    print(f"  G_seq   (sequencing)  = {G_seq:.3%}  (delta={np.mean(seq_deltas) if seq_deltas else float('nan'):+.3f}, "
          f"CI {seq_ci}, W/T/L {seq_wtl})")
    print(f"  G_fleet (fleet alloc) = {G_fleet:.3%}  (delta={np.mean(fleet_deltas) if fleet_deltas else float('nan'):+.3f}, "
          f"CI {fleet_ci}, W/T/L {fleet_wtl})")

    # ---- 逐实例 CSV ----
    with open(os.path.join(args.out, 'gap_decomposition.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id', 'nn_cost', 'fixed_cost', 'joint_cost',
                    'nn_complete', 'fixed_complete', 'joint_complete',
                    'nn_unserved', 'fixed_unserved', 'joint_unserved'])
        for i in range(n):
            a, b, c = results['NN'][i], results['OR-fixed'][i], results['OR-joint'][i]
            w.writerow([i, f"{a['distance_cost']:.4f}", f"{b['distance_cost']:.4f}",
                        f"{c['distance_cost']:.4f}", int(a['complete']), int(b['complete']),
                        int(c['complete']), a['n_unserved'], b['n_unserved'], c['n_unserved']])

    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'n', 'ci_lo', 'ci_hi', 'win', 'tie', 'loss'])
        w.writerow(['G_seq', f"{G_seq:.4f}", len(seq_idx), f"{seq_ci[0]:.4f}", f"{seq_ci[1]:.4f}",
                    seq_wtl[0], seq_wtl[1], seq_wtl[2]])
        w.writerow(['G_fleet', f"{G_fleet:.4f}", len(fleet_idx), f"{fleet_ci[0]:.4f}",
                    f"{fleet_ci[1]:.4f}", fleet_wtl[0], fleet_wtl[1], fleet_wtl[2]])
        w.writerow(['NN_service_eq_cost', f"{J_nn:.4f}", len(seq_idx), '', '', '', '', ''])
        w.writerow(['OR_fixed_service_eq_cost_seq', f"{J_fixed_seq:.4f}", len(seq_idx), '', '', '', '', ''])
        w.writerow(['OR_fixed_service_eq_cost_fleet', f"{J_fixed_fleet:.4f}", len(fleet_idx), '', '', '', '', ''])
        w.writerow(['OR_joint_service_eq_cost', f"{J_joint:.4f}", len(fleet_idx), '', '', '', '', ''])

    print(f"\n  output: {args.out}/service_matrix.csv + solver_status_or_fixed.csv "
          f"+ gap_decomposition.csv + summary.csv")


if __name__ == '__main__':
    main()
