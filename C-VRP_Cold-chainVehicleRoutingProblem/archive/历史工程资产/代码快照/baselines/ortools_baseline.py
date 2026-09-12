"""
OR-Tools baseline for DCCVRP.
Solves static CVRPTW (all orders known) as upper bound for MaskCO comparison.

用法:
    python baselines/ortools_baseline.py \
        --data dcc_50_r1_edod05_test.npz --capacity 50 \
        --time_limit 60 --num_vehicles 15
"""

import sys, os, argparse, time, numpy as np
_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _CVRPTW)

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_cvrptw_ortools(dataset, inst_idx, capacity, num_vehicles, time_limit):
    """用 OR-Tools 求解单个 CVRPTW 实例。"""
    coords = dataset['coords'][inst_idx]
    demands = dataset['demands'][inst_idx]
    tw_start = dataset['tw_start'][inst_idx]
    tw_end = dataset['tw_end'][inst_idx]
    service_time = dataset.get('service_time',
        np.zeros_like(demands, dtype=np.float32))[inst_idx]
    num_nodes = len(coords)

    # 距离矩阵 (OR-Tools 需要整数)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    scale = 1000
    dist_int = (dist * scale).astype(int)

    # 创建路由模型
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)  # depot=0
    routing = pywrapcp.RoutingModel(manager)

    # 距离回调
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return dist_int[from_node, to_node]

    transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    # 容量约束
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        return int(demands[from_node])

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx, 0, [capacity] * num_vehicles, True, 'Capacity')

    # 时间窗约束
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int((dist[from_node, to_node] + service_time[from_node]) * scale)

    time_callback_idx = routing.RegisterTransitCallback(time_callback)
    horizon = int(tw_end[0] * scale * 1.5)  # depot TW as horizon
    routing.AddDimension(
        time_callback_idx, horizon, horizon, False, 'Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    for node in range(1, num_nodes):
        idx = manager.NodeToIndex(node)
        time_dimension.CumulVar(idx).SetRange(
            int(tw_start[node] * scale),
            int(tw_end[node] * scale))

    # 求解参数
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.seconds = time_limit
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None, None, 0

    # 提取路径
    total_cost = 0.0
    total_load = 0
    feasible = True
    routes_list = []

    for vehicle_id in range(num_vehicles):
        idx = routing.Start(vehicle_id)
        route = []
        route_load = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            if node > 0:
                route_load += demands[node]
            idx = solution.Value(routing.NextVar(idx))
        if route:
            route.append(0)
            routes_list.append(route)
            total_load += route_load

        # 计算距离成本
        for k in range(len(route) - 1):
            total_cost += dist[route[k], route[k + 1]]

    # 合并为单一路径
    full_route = []
    for seg in routes_list:
        full_route.extend(seg[:-1])
    full_route.append(0)

    return full_route, total_cost, len(routes_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--num_vehicles', type=int, default=15)
    parser.add_argument('--num_instances', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    num_instances = min(args.num_instances, dataset['coords'].shape[0])

    print(f"OR-Tools CVRPTW Baseline")
    print(f"  Data: {args.data} ({num_instances} instances)")
    print(f"  Capacity={args.capacity}, Vehicles={args.num_vehicles}, TimeLimit={args.time_limit}s")

    costs = []
    feas_count = 0
    solved = 0
    t0 = time.time()

    for i in range(num_instances):
        route, cost, n_veh = solve_cvrptw_ortools(
            dataset, i, args.capacity, args.num_vehicles, args.time_limit)
        if cost is not None:
            costs.append(cost)
            solved += 1
        if (i + 1) % 4 == 0:
            print(f"  {i+1}/{num_instances} | avg_cost={np.mean(costs):.2f} | "
                  f"solved={solved} | {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n--- OR-Tools Results ---")
    print(f"  Solved: {solved}/{num_instances}")
    print(f"  Avg cost: {np.mean(costs):.2f}" if costs else "  All failed")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/num_instances:.1f}s/inst)")
    print(f"  Reference cost (MaskCO opt): {dataset['opt_costs'][:num_instances].mean():.2f}")

    if args.output:
        np.savez(args.output, costs=costs)


if __name__ == '__main__':
    main()
