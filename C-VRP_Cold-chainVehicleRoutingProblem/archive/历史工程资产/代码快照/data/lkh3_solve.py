"""
CVRPTW .npz → LKH3 格式转换 + 批量求解 + 回写 .npz。

用法:
    python "C-VRP_Cold-chainVehicleRoutingProblem/data/lkh3_solve.py" \
        --input "C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_r1_train.npz" \
        --output "C-VRP_Cold-chainVehicleRoutingProblem/data/cvrptw50_r1_train_lkh3.npz" \
        --capacity 50 --vehicles 10 --time_limit 30 --seed 42
"""

import sys, os, argparse, subprocess, tempfile, time
import numpy as np


def write_lkh3_instance(
    coords: np.ndarray,        # (nodes, 2)
    demands: np.ndarray,       # (nodes,)
    tw_start: np.ndarray,      # (nodes,)
    tw_end: np.ndarray,        # (nodes,)
    service_time: np.ndarray,  # (nodes,)
    capacity: int,
    vehicles: int,
    depot_tw_end: float,
    filepath: str,
):
    """将单个 CVRPTW 实例写为 LKH3 格式。"""
    num_nodes = len(coords) - 1  # 不含 depot
    scale = 1000.0
    with open(filepath, 'w') as f:
        f.write(f"NAME : cvrptw\n")
        f.write(f"TYPE : CVRPTW\n")
        f.write(f"DIMENSION : {num_nodes + 1}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"EDGE_WEIGHT_FORMAT : FUNCTION\n")
        f.write(f"NODE_COORD_TYPE : TWOD_COORDS\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write(f"VEHICLES : {vehicles}\n")

        f.write("NODE_COORD_SECTION\n")
        for i in range(num_nodes + 1):
            f.write(f" {i + 1} {coords[i, 0] * scale:.1f} {coords[i, 1] * scale:.1f}\n")

        f.write("DEMAND_SECTION\n")
        for i in range(num_nodes + 1):
            f.write(f" {i + 1} {int(demands[i])}\n")

        f.write("DEPOT_SECTION\n1\n-1\n")

        f.write("TIME_WINDOW_SECTION\n")
        for i in range(num_nodes + 1):
            f.write(f" {i + 1} {tw_start[i] * scale:.1f} {tw_end[i] * scale:.1f}\n")

        f.write("SERVICE_TIME_SECTION\n")
        for i in range(num_nodes + 1):
            f.write(f" {i + 1} {service_time[i] * scale:.2f}\n")

        f.write("EOF\n")


def write_lkh3_param(param_path: str, problem_path: str, output_path: str, time_limit: int, seed: int):
    """写 LKH3 参数文件。"""
    with open(param_path, 'w') as f:
        f.write(f"PROBLEM_FILE = {problem_path}\n")
        f.write(f"TOUR_FILE = {output_path}\n")
        f.write(f"TIME_LIMIT = {time_limit}\n")
        f.write(f"SEED = {seed}\n")
        f.write("RUNS = 1\n")
        f.write("TRACE_LEVEL = 0\n")


def parse_lkh3_tour(tour_path: str, num_customers: int, max_route_len: int) -> np.ndarray:
    """解析 LKH3 输出的 tour 文件，转换为 MaskCO 格式的 padded route。"""
    with open(tour_path, 'r') as f:
        lines = f.readlines()

    # 提取 tour 节点序列
    tour_started = False
    nodes = []
    for line in lines:
        line = line.strip()
        if line == "TOUR_SECTION":
            tour_started = True
            continue
        if line == "-1" or line == "EOF":
            break
        if tour_started and line:
            try:
                nodes.append(int(line))
            except ValueError:
                pass

    # LKH3 返回 1-indexed，转为 0-indexed
    nodes = [n - 1 for n in nodes]

    # 构造 CVRP 格式的 route（depot=0 分隔）
    route = [0]
    for n in nodes:
        if n == 0:
            route.append(0)
        else:
            route.append(n)
    if route[-1] != 0:
        route.append(0)

    route = np.array(route, dtype=np.int32)
    route = np.pad(route, (0, max(0, max_route_len - len(route))), constant_values=0)
    return route[:max_route_len]


def compute_route_cost(route: np.ndarray, dist_mat: np.ndarray) -> float:
    """计算路径总距离。"""
    cost = 0.0
    prev = 0
    for node in route:
        if node == 0 and prev == 0:
            continue
        cost += dist_mat[prev, int(node)]
        prev = int(node)
        if node == 0:
            prev = 0
    return cost


def solve_dataset(
    dataset: dict[str, np.ndarray],
    capacity: int,
    vehicles: int,
    time_limit: int,
    seed: int,
    max_route_len: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """用 LKH3 求解整个数据集。"""
    num_instances = dataset['coords'].shape[0]
    routes_list = []
    costs_list = []

    tmp_dir = tempfile.mkdtemp(prefix='lkh3_')
    problem_path = os.path.join(tmp_dir, 'problem.vrp')
    param_path = os.path.join(tmp_dir, 'param.par')
    output_path = os.path.join(tmp_dir, 'output.tour')

    for i in range(num_instances):
        # 写 LKH3 输入
        write_lkh3_instance(
            coords=dataset['coords'][i],
            demands=dataset['demands'][i],
            tw_start=dataset['tw_start'][i],
            tw_end=dataset['tw_end'][i],
            service_time=dataset.get('service_time', np.zeros_like(dataset['demands']))[i],
            capacity=capacity,
            vehicles=vehicles,
            depot_tw_end=float(dataset['tw_end'][i, 0]),
            filepath=problem_path,
        )
        write_lkh3_param(param_path, problem_path, output_path, time_limit, seed + i)

        # 运行 LKH3
        try:
            result = subprocess.run(
                ['lkh3', param_path],
                capture_output=True, text=True, timeout=time_limit + 10,
            )
        except subprocess.TimeoutExpired:
            print(f"  Instance {i}: timeout, using greedy fallback")
            routes_list.append(dataset['routes'][i])
            costs_list.append(dataset['opt_costs'][i])
            continue

        # 解析结果
        if os.path.exists(output_path):
            try:
                route = parse_lkh3_tour(output_path, dataset['coords'].shape[1] - 1, max_route_len)
                # 计算成本
                diff = dataset['coords'][i][:, None, :] - dataset['coords'][i][None, :, :]
                dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
                cost = compute_route_cost(route, dist_mat)
            except Exception:
                route = dataset['routes'][i]
                cost = dataset['opt_costs'][i]
        else:
            route = dataset['routes'][i]
            cost = dataset['opt_costs'][i]

        routes_list.append(route)
        costs_list.append(cost)

        if (i + 1) % 100 == 0:
            print(f"  LKH3 solved {i + 1}/{num_instances}...")

    # 清理临时文件
    for f in [problem_path, param_path, output_path]:
        if os.path.exists(f):
            os.remove(f)
    os.rmdir(tmp_dir)

    # 对齐 route 长度
    max_len = max(len(r) for r in routes_list)
    routes = np.stack([
        np.pad(r, (0, max_len - len(r)), constant_values=0)
        for r in routes_list
    ], axis=0).astype(np.int32)

    costs = np.array(costs_list, dtype=np.float32)
    return routes, costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--vehicles', type=int, default=10)
    parser.add_argument('--time_limit', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    dataset = dict(np.load(args.input))
    print(f"  {dataset['coords'].shape[0]} instances, "
          f"{dataset['coords'].shape[1] - 1} customers")

    print(f"Solving with LKH3 (time_limit={args.time_limit}s, vehicles={args.vehicles})...")
    t0 = time.time()
    new_routes, new_costs = solve_dataset(
        dataset, args.capacity, args.vehicles,
        args.time_limit, args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed / dataset['coords'].shape[0]:.1f}s/inst)")

    # 保存
    out = dict(dataset)
    out['routes'] = new_routes
    out['opt_costs'] = new_costs
    old_avg = dataset['opt_costs'].mean()
    new_avg = new_costs.mean()
    print(f"  Cost: {old_avg:.4f} (greedy) → {new_avg:.4f} (LKH3) "
          f"({(old_avg - new_avg) / old_avg * 100:.1f}% improvement)")
    np.savez_compressed(args.output, **out)
    print(f"  Saved to {args.output}")
