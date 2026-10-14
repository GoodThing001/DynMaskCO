"""
JF2 Exact-Vehicle Oracle Utility Ceiling（导师 2026-09-01 决策树的第 5 步）。

回答：假如有「完美 exact-vehicle correction oracle」，这个动作空间最多值多少钱？

三路比较（VAL16/32，同 strict-online skeleton）：
  JF1-H        = greedy min-travel assignment + greedy sequencing          （24.50 baseline）
  OR-assign+gr = OR-joint 完整 assignment（joint partition）+ greedy 重排   （assignment ceiling 上界）
  OR-joint     = OR-joint assignment + OR sequencing                       （full fleet，数值随 --time_limit_ms 预算）

判读：
  - 若 OR-assign+gr 只比 JF1-H 好一点点（24.50→24.3x）→ assignment 动作空间天花板很低，
    即使完美 classifier 也拿不回 gap → 正式结束 exact-vehicle 路线。
  - 若 OR-assign+gr 接近 OR-joint（同预算下）→ assignment 本身有巨大价值，是现有 per-customer
    learner 抓不住（因为 per-customer regret 小而分散，需 joint partition），→ 转 partition。

用法（服务器，需 OR-Tools）：
    python scripts/evaluation/run_exact_vehicle_oracle.py \
      --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
      --num_instances 32 --time_limit_ms 50 --out results/jf2/oracle_ceiling
"""
import sys, os, argparse, csv, time
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'baselines'))

from strict_online_env import StrictOnlineEnv, Replanner
from authoritative_evaluator import evaluate_execution_trace
from joint_fleet import JointAssignmentReplanner
from ortools_rolling_horizon import ORToolsReplanner
from b0_or_expert import solve_or_fixed
from service_first import (service_equivalent_idx, paired_bootstrap_ci, win_tie_loss,
                           write_service_matrix, write_solver_status_csv)


class ORAssignmentGreedySeqReplanner(Replanner):
    """OR-joint 的完整 assignment（joint partition）+ 每辆车 greedy 重排。

    隔离出「assignment 本身值多少」：把 OR 的分区决策保留，但 sequencing 用 JF1-H 的贪心
    （而非 OR 的精确 TSPTW 排序），从而与 B0 的「sequencing 已饱和」对齐。

    P0-R：greedy sequencing 用 _greedy_sequence_report（显式报告 drop），并把 assigned（OR 分区）
    vs returned（greedy 幸存）写入 solve_status_log，供四路 coverage 追溯（no silent drop）。
    """

    def __init__(self, capacity, num_vehicles, time_limit_ms):
        self.or_rp = ORToolsReplanner(capacity, num_vehicles, time_limit_ms)
        self.inner = JointAssignmentReplanner('heuristic')
        self.solve_status_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        # 1. OR-joint 求解 → v.mutable_suffix = OR route（含 assignment + sequencing）
        self.or_rp.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
        # 2. 每辆 active 车：提取 OR 分配给它的客户集，用 greedy 重排（显式报告 drop）
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            assigned = [int(n) for n in v.mutable_suffix if int(n) != 0]
            if not assigned:
                continue
            suffix, dropped = self.inner._greedy_sequence_report(env, inst_idx, v, assigned)
            returned = [int(n) for n in suffix if int(n) != 0]
            self.solve_status_log.append({
                'inst_idx': int(inst_idx), 'vehicle_id': int(v.vehicle_id),
                'k': len(assigned), 'status': 'greedy_seq',
                'assigned_set': tuple(sorted(assigned)),
                'returned_set': tuple(sorted(returned)),
            })
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix


class JF1HAssignORTWSeqReplanner(Replanner):
    """JF1-H 的 assignment（joint min-travel）+ OR 的单车 TSPTW 重排。

    第 4 个 comparator：隔离「JF1-H 的 partition 下，sequencing 还能压多少」。若它 ≈ OR-joint
    （~23.0），说明 gap 在 sequencing（无论谁的 partition）；若 ≈ JF1-H（~24.4），说明 sequencing
    收益只在 OR 的（更密）partition 上，partition 与 sequencing 强耦合。
    """

    def __init__(self, capacity, time_limit_ms):
        self.inner = JointAssignmentReplanner('heuristic')
        self.capacity = capacity
        self.time_limit_ms = time_limit_ms
        self.solve_status_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        # 1. JF1-H 的 assignment + greedy seq
        self.inner.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)
        jf1h_suffix = {v.vehicle_id: list(v.mutable_suffix)
                       for v in vehicles if v.status in ('idle', 'ready')}
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        # 2. 每辆车单车 TSPTW 重排（frozen JF1-H assignment）
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            # assigned = JF1-H 实际服务的 per-vehicle 集（inner.plan 的 greedy survivors）。
            # JF1-H 自身若 greedy 放不下某客户，则该客户本就不在任何车的 partition 里；
            # solve_or_fixed 只重排这批幸存客户，不跨车重分配（frozen JF1-H partition）。
            assigned = [o for o in jf1h_suffix.get(v.vehicle_id, []) if o != 0]
            if not assigned:
                continue
            feasible, route, status = solve_or_fixed(
                env.coords[inst_idx], env.demands[inst_idx], env.tw_start[inst_idx],
                env.tw_end[inst_idx], env.service_time[inst_idx], env.capacity,
                self.time_limit_ms, int(v.current_node), v.ready_time, v.current_load,
                assigned, dist_mat=env.dist_mat[inst_idx], tw_speed=env.tw_speed)
            if not feasible:
                self.solve_status_log.append({
                    'inst_idx': int(inst_idx), 'vehicle_id': int(v.vehicle_id),
                    'k': len(assigned), 'status': status,
                    'assigned_set': tuple(sorted(assigned)),
                    'returned_set': tuple(sorted(assigned)),
                })
                continue
            returned = tuple(sorted(int(n) for n in route
                                    if n not in (int(v.current_node), 0)))
            self.solve_status_log.append({
                'inst_idx': int(inst_idx), 'vehicle_id': int(v.vehicle_id),
                'k': len(assigned), 'status': status,
                'assigned_set': tuple(sorted(assigned)), 'returned_set': returned,
            })
            suffix = [n for n in route if n != int(v.current_node)]
            if suffix == [0] and v.current_node != 0 and has_future:
                v.mutable_suffix = []
            else:
                v.mutable_suffix = suffix


def run_method(replanner, dataset, n):
    per = []
    for i in range(n):
        env = StrictOnlineEnv(dataset, 50, 1.0, 25, replanner=replanner)
        t_start = time.time()
        traces, served = env.run(i)
        m = evaluate_execution_trace(
            traces, env.coords[i], env.tw_start[i], env.tw_end[i],
            env.service_time[i], env.demands[i], 50, speed=1.0, dist_mat=env.dist_mat[i])
        m['runtime'] = time.time() - t_start
        per.append(m)
    return per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_instances', type=int, default=32)
    parser.add_argument('--time_limit_ms', type=int, default=50)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    n = min(args.num_instances, dataset['coords'].shape[0])

    policies = [
        ('JF1-H', JointAssignmentReplanner('heuristic')),
        ('OR-assign+gr', ORAssignmentGreedySeqReplanner(50, 25, args.time_limit_ms)),
        ('JF1H-assign+TSPTW', JF1HAssignORTWSeqReplanner(50, args.time_limit_ms)),
        ('OR-joint', ORToolsReplanner(50, 25, args.time_limit_ms)),
    ]
    names = [name for name, _ in policies]

    results = {}
    for name, rp in policies:
        print(f"=== {name} ===", flush=True)
        results[name] = run_method(rp, dataset, n)

    # ---- P0-R3: service matrix + solver coverage 先行 ----
    write_service_matrix(os.path.join(args.out, 'service_matrix.csv'), results, names, n)
    write_solver_status_csv(os.path.join(args.out, 'solver_status_jf1h_tsptw.csv'),
                            policies[2][1].solve_status_log, n, method='JF1H-assign+TSPTW')
    write_solver_status_csv(os.path.join(args.out, 'solver_status_or_assign_gr.csv'),
                            policies[1][1].solve_status_log, n, method='OR-assign+gr')

    def complete(name):
        return float(np.mean([m['complete'] for m in results[name]]))

    def paired_cost(a, b):
        idx = service_equivalent_idx(results, [a, b], n)
        ca = float(np.mean([results[a][i]['distance_cost'] for i in idx])) if idx else float('nan')
        cb = float(np.mean([results[b][i]['distance_cost'] for i in idx])) if idx else float('nan')
        return ca, cb, idx

    print(f"\n=== Exact-Vehicle Oracle Utility Ceiling（{n} instances）===")
    for name in names:
        print(f"  {name:18s} : complete={complete(name):.1%}")

    # ---- paired service-equivalent gap 分解（P0-R3：每个 delta 用各自 pair 的 service-eq 子集）----
    c_hg_a, c_og_a, idx_a = paired_cost('JF1-H', 'OR-assign+gr')
    c_hg_s, c_ht_s, idx_s = paired_cost('JF1-H', 'JF1H-assign+TSPTW')
    c_og_s, c_ot_s, idx_sor = paired_cost('OR-assign+gr', 'OR-joint')
    print(f"\n  [gap 分解，paired service-equivalent]")
    print(f"    assignment（OR partition vs JF1-H，都 greedy seq）: {c_og_a - c_hg_a:+.4f}  (n={len(idx_a)})")
    print(f"    sequencing（TSPTW vs greedy，都 JF1-H partition）: {c_ht_s - c_hg_s:+.4f}  (n={len(idx_s)})")
    print(f"    sequencing（TSPTW vs greedy，都 OR partition）   : {c_ot_s - c_og_s:+.4f}  (n={len(idx_sor)})")

    # ---- instance-level interaction，只在四路 service-equivalent 子集上算（P0-R2）----
    idx4 = service_equivalent_idx(results, names, n)
    c_ot = np.array([results['OR-joint'][i]['distance_cost'] for i in idx4])
    c_og = np.array([results['OR-assign+gr'][i]['distance_cost'] for i in idx4])
    c_ht = np.array([results['JF1H-assign+TSPTW'][i]['distance_cost'] for i in idx4])
    c_hg = np.array([results['JF1-H'][i]['distance_cost'] for i in idx4])
    inter = c_ot - c_og - c_ht + c_hg
    inter_map = {int(i): float(inter[k]) for k, i in enumerate(idx4)}
    ci_lo, ci_hi = paired_bootstrap_ci(inter)
    print(f"\n  [instance-level interaction I_n = C_OT-C_OG-C_HT+C_HG, "
          f"service-equivalent n={len(idx4)}]")
    print(f"    mean={inter.mean():+.4f}  median={np.median(inter):+.4f}")
    print(f"    95% bootstrap CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    P(I_n < 0) = {(inter < 0).mean():.1%}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'oracle_ceiling.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instance_id'] + [f'{name}_cost' for name in names]
                   + ['complete_all', 'service_equiv', 'assign_gap', 'interaction'])
        for i in range(n):
            comp_all = all(results[name][i]['complete'] for name in names)
            w.writerow([i] + [f"{results[name][i]['distance_cost']:.4f}" for name in names]
                       + [int(comp_all), int(i in idx4),
                          f"{results['OR-assign+gr'][i]['distance_cost'] - results['JF1-H'][i]['distance_cost']:+.4f}",
                          f"{inter_map[i]:+.4f}" if i in inter_map else ""])
    with open(os.path.join(args.out, 'interaction_stats.json'), 'w') as f:
        import json
        json.dump({'n_service_equiv': int(len(idx4)), 'n_total': int(n),
                   'mean': float(inter.mean()), 'median': float(np.median(inter)),
                   'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                   'p_negative': float((inter < 0).mean())}, f, indent=2)
    print(f"  saved: {args.out}/service_matrix.csv + solver_status_jf1h_tsptw.csv "
          f"+ oracle_ceiling.csv + interaction_stats.json")


if __name__ == '__main__':
    main()
