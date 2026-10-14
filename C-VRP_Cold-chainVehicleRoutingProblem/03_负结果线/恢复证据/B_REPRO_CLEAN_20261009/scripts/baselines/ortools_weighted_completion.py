"""
④ matched-objective OR-Tools 对比 — weighted completion time 基线

用 OR-Tools 求解「距离 + λ_q × weighted completion time（∑ K_i·t_i）」，
作为 DynMaskCO 品质感知的 matched-objective 基线。

实现：OR-Tools time dimension + SetCumulVarSoftUpperBound(node, 0, K_i×SCALE)，
使得 penalty = ∑ K_i × CumulVar(node_i) = weighted completion time。

诚实定位（导师指示）：
  不说「OR-Tools 做不到」，而说「DynMaskCO 引入了 standard distance-TW 目标之外的
  state/time-dependent costs」。若 OR-Tools 在 matched objective 上仍赢，
  则诚实承认「冷链是问题建模扩展，非竞争优势」。

用法：
  python baselines/ortools_weighted_completion.py \
      --data <test.npz> --capacity 50 --time_limit_ms 5000 --num_instances 8
"""

import sys, os, argparse, time
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts'))

K_TEMP = np.array([0.01, 0.002, 0.0002], dtype=np.float32)
SCALE = 1000  # 距离/时间整数化


def solve_weighted_completion(coords, demands, tw_start, tw_end, service_time,
                              temp_class, capacity, time_limit_ms, lambda_q):
    """OR-Tools 求解：距离 + λ_q × weighted completion time。"""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    N = len(coords)
    dist_float = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(axis=-1))
    dist_int = (dist_float * SCALE).astype(int)

    num_vehicles = max(1, int(np.ceil(demands[1:].sum() / capacity)) + 2)
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return int(dist_int[manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)])

    def time_cb(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return int(dist_int[i, j] + service_time[i] * SCALE)

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def demand_cb(from_idx):
        return int(demands[manager.IndexToNode(from_idx)])

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_cb),
        0, [capacity] * num_vehicles, True, "Capacity")

    horizon = int(tw_end.max() * SCALE)
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, N):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(
            int(tw_start[node] * SCALE), int(tw_end[node] * SCALE))

    # weighted completion time：penalty = K_i × CumulVar(node_i)
    for node in range(1, N):
        k = K_TEMP[int(temp_class[node])]
        # SetCumulVarSoftUpperBound(node, 0, penalty)：cumul > 0 时 penalty = K_i × cumul
        time_dim.SetCumulVarSoftUpperBound(
            manager.NodeToIndex(node), 0, int(k * lambda_q * SCALE))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromMilliseconds(time_limit_ms)
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = routing.SolveWithParameters(params)
    if not sol:
        return None

    # 提取路线 + 计算指标
    total_dist = 0.0
    weighted_t = 0.0
    cum = {}
    for v in range(num_vehicles):
        idx = routing.Start(v)
        prev = 0
        cum_time = 0.0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node == 0:
                prev = 0
                cum_time = 0.0
            else:
                total_dist += dist_float[prev, node]
                cum_time += dist_float[prev, node] + service_time[prev]
                k = K_TEMP[int(temp_class[node])]
                weighted_t += k * cum_time
                prev = node
            idx = sol.Value(routing.NextVar(idx))
        total_dist += dist_float[prev, 0]
    return {'distance': total_dist, 'weighted_t': weighted_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--time_limit_ms', type=int, default=5000)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--lambda_q', type=float, default=1.0)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    N = min(args.num_instances, dataset['coords'].shape[0])

    print(f"=== OR-Tools weighted completion time（matched-objective）===")
    print(f"  data: {args.data}（{N} 实例）")
    print(f"  λ_q={args.lambda_q}, budget={args.time_limit_ms}ms/instance")

    results = []
    for i in range(N):
        r = solve_weighted_completion(
            dataset['coords'][i], dataset['demands'][i].astype(int),
            dataset['tw_start'][i], dataset['tw_end'][i],
            dataset['service_time'][i], dataset['temp_class'][i].astype(int),
            args.capacity, args.time_limit_ms, args.lambda_q)
        results.append(r)
        if r is not None:
            print(f"  inst {i}: dist={r['distance']:.3f} weighted_t={r['weighted_t']:.4f}")

    feas = [r for r in results if r is not None]
    if feas:
        print(f"\n  feasible {len(feas)}/{N}")
        print(f"  mean distance: {np.mean([r['distance'] for r in feas]):.3f}")
        print(f"  mean weighted_t: {np.mean([r['weighted_t'] for r in feas]):.4f}")


if __name__ == '__main__':
    main()
