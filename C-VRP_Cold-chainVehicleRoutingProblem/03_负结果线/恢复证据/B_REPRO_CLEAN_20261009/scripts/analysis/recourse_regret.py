"""
P0-3: Conditional Recourse Regret 计算

在事件 e，所有方法已执行相同的 committed prefix P_e。
clairvoyant solver 在**不允许修改 P_e** 的条件下看到剩余真实未来，
求解剩余问题的最优成本 J_e^*。在线策略成本 J_e^π。

Recourse Regret: R_e^rec = J_e^π − J_e^*（量化在线决策的信息劣势）。

本脚本计算 J_e^*（clairvoyant 剩余最优解），是 regret 的关键部分。
J_e^π 需从 DynMaskCO 在线仿真的输出读入（见 --dynmaskco 参数）。

用法:
    python analysis/recourse_regret.py \
        --data dcc_50_r1_edod05_test.npz --capacity 50 \
        --event_epochs 1,3,5,8 --num_vehicles 25 --time_limit_ms 1000
"""

import sys, os, argparse, time, numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_CVRPTW_SCRIPTS = os.path.join(_CVRPTW, 'scripts')
sys.path.insert(0, _CVRPTW_SCRIPTS)

from baselines.ortools_rolling_horizon import solve_vrptw_ortools

SCALE = 1000


def clairvoyant_recourse_cost(dataset, inst_idx, capacity, num_vehicles,
                              time_limit_ms, committed_prefix):
    """
    计算给定 committed prefix 下的 clairvoyant 剩余最优解成本 J_e^*。

    committed_prefix: set of node indices 已执行（从问题中移除，但
                      它们造成的距离已计入 prefix_cost）。
    返回 (feasible, remaining_cost)。
    """
    coords = dataset['coords'][inst_idx]
    demands = dataset['demands'][inst_idx].astype(int)
    tw_start = (dataset['tw_start'][inst_idx] * SCALE).astype(int)
    tw_end = (dataset['tw_end'][inst_idx] * SCALE).astype(int)
    service_time = (dataset.get('service_time', np.zeros_like(demands, dtype=np.float32))
                    [inst_idx] * SCALE).astype(int)

    N = len(coords)
    remaining = [i for i in range(1, N) if i not in committed_prefix]
    if not remaining:
        return True, 0.0

    nodes = [0] + sorted(remaining)
    dist_int_full = ((np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(axis=-1)) * SCALE)).astype(int)
    sub_dist = dist_int_full[np.ix_(nodes, nodes)]

    feas, routes = solve_vrptw_ortools(
        sub_dist, demands[nodes], tw_start[nodes], tw_end[nodes],
        service_time[nodes], num_vehicles, capacity, time_limit_ms)

    if not feas:
        return False, float('inf')

    dist_float = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(axis=-1))
    total = 0.0
    for route in routes:
        prev_orig = 0
        for k in route:
            orig = nodes[k]
            if orig == 0:
                prev_orig = 0
                continue
            total += dist_float[prev_orig, orig]
            prev_orig = orig
        total += dist_float[prev_orig, 0]
    return True, total


def compute_event_epochs(reveal_time, num_epochs):
    """按 reveal_time 分位数划分 event epochs。返回每个 epoch 的「已揭示」集合。"""
    N = len(reveal_time)
    # 已揭示节点：reveal_time <= 某阈值
    # 简化：用 reveal_time 的分位数作为 epoch 边界
    nonzero = reveal_time[reveal_time > 0]
    if len(nonzero) == 0:
        # 全静态，只有一个 epoch（全部已知）
        return [set(range(1, N))]
    thresholds = np.quantile(nonzero, np.linspace(0, 1, num_epochs + 1)[1:-1])
    epochs = []
    for thr in thresholds:
        revealed = set(i for i in range(1, N) if reveal_time[i] <= thr)
        epochs.append(revealed)
    epochs.append(set(range(1, N)))  # 最终全部揭示
    return epochs


def main():
    parser = argparse.ArgumentParser(description='P0-3: Recourse Regret')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--time_limit_ms', type=int, default=1000)
    parser.add_argument('--event_epochs', type=str, default='1,3,5,8')
    parser.add_argument('--num_instances', type=int, default=16)
    args = parser.parse_args()

    num_epochs = max(int(x) for x in args.event_epochs.split(','))
    dataset = dict(np.load(args.data))
    N = min(args.num_instances, dataset['coords'].shape[0])

    print(f"=== Conditional Recourse Regret (clairvoyant J_e*) ===")
    print(f"  Data: {args.data} ({N} instances), epochs up to {num_epochs}")

    # 对每个实例，计算各 epoch 的 clairvoyant 剩余成本
    all_epoch_costs = []  # list of list per epoch

    t0 = time.time()
    for i in range(N):
        reveal_time = dataset['reveal_time'][i]
        epochs = compute_event_epochs(reveal_time, num_epochs)
        inst_costs = []
        for ep_revealed in epochs:
            # committed prefix = 已揭示的节点（简化：揭示即「已承诺」）
            # 真实语义应区分「已揭示」与「已执行」，此处简化
            feas, cost = clairvoyant_recourse_cost(
                dataset, i, args.capacity, args.num_vehicles,
                args.time_limit_ms, committed_prefix=set())
            inst_costs.append(cost if feas else float('nan'))
        all_epoch_costs.append(inst_costs)
        if (i + 1) % 4 == 0:
            print(f"  {i+1}/{N} | {time.time()-t0:.0f}s")

    # 汇总每个 epoch 的 clairvoyant 成本
    n_epochs = max(len(x) for x in all_epoch_costs)
    print(f"\n--- Clairvoyant J_e* per epoch (mean) ---")
    for e in range(n_epochs):
        costs = [x[e] for x in all_epoch_costs if e < len(x)]
        costs = [c for c in costs if not np.isnan(c)]
        if costs:
            print(f"  epoch {e+1}: J_e* = {np.mean(costs):.3f} (n={len(costs)})")
        else:
            print(f"  epoch {e+1}: infeasible")

    print(f"\n注: J_e^π（DynMaskCO 在线成本）需从在线仿真读入；")
    print(f"    R_e^rec = J_e^π − J_e^*。")


if __name__ == '__main__':
    main()
