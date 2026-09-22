"""
生成 CVRPTW 训练/测试数据。

支持格式：
1. Solomon (1987) 6 类实例：C1, C2, R1, R2, RC1, RC2
2. 随机生成实例（均匀/聚类坐标 + 时间窗）
3. 从 TSPTW 数据转换（利用已有的 tsptw50 数据）

数据保存为 .npz 格式，包含：
- coords: (N, nodes+1, 2) — 含 depot
- demands: (N, nodes+1) — depot demand=0
- tw_start: (N, nodes+1)
- tw_end: (N, nodes+1)
- service_time: (N, nodes+1)
- routes: (N, padded_len) — 参考解（从 LKH3 或启发式生成）
- opt_costs: (N,) (可选)

用法：
    python -m data.generate_cvrptw_data \
        --problem_size 50 \
        --num_instances 1280 \
        --type solomon_r1 \
        --output data/cvrptw50_r1_train.npz
"""

import sys
import os
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT
sys.path.insert(0, str(MASKCO_ROOT))

import numpy as np
import argparse
from typing import Literal


# ============================================================
# Solomon 实例生成器
# ============================================================

def generate_solomon_instance(
    num_customers: int = 100,
    instance_type: Literal['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'] = 'R1',
    capacity: int = 200,
    seed: int = 0,
) -> dict:
    """
    生成单个 Solomon 风格的 CVRPTW 实例。

    参考 Solomon (1987) 的生成规则：
    - C (Clustered): 客户聚集成簇，时间窗宽
    - R (Random): 客户随机分布，时间窗窄
    - RC (Random-Clustered): 混合

    返回 dict:
        coords, demands, tw_start, tw_end, service_time, routes, opt_cost
    """
    rng = np.random.default_rng(seed)
    total_nodes = num_customers + 1  # +1 for depot

    # --- Depot ---
    depot_x, depot_y = 0.5, 0.5

    # --- 生成坐标 ---
    if instance_type.startswith('C'):
        # 聚类：2-4 个簇中心，每个簇内用高斯分布
        n_clusters = rng.integers(2, 5)
        cluster_centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
        coords_list = []
        for i in range(num_customers):
            center = cluster_centers[i % n_clusters]
            x = np.clip(center[0] + rng.normal(0, 0.05), 0.02, 0.98)
            y = np.clip(center[1] + rng.normal(0, 0.05), 0.02, 0.98)
            coords_list.append([x, y])

    elif instance_type.startswith('RC'):
        # 一半聚类 + 一半随机
        n_clustered = num_customers // 2
        n_random = num_customers - n_clustered
        n_clusters = rng.integers(2, 4)
        centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
        coords_list = []
        for i in range(n_clustered):
            center = centers[i % n_clusters]
            x = np.clip(center[0] + rng.normal(0, 0.05), 0.02, 0.98)
            y = np.clip(center[1] + rng.normal(0, 0.05), 0.02, 0.98)
            coords_list.append([x, y])
        for i in range(n_random):
            coords_list.append([rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)])

    else:  # R
        coords_list = rng.uniform(0.02, 0.98, size=(num_customers, 2)).tolist()

    coords_list = [[depot_x, depot_y]] + coords_list
    coords = np.array(coords_list, dtype=np.float32)

    # --- 距离矩阵 ---
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))

    # --- 需求量 ---
    demands = np.zeros(total_nodes, dtype=np.int32)
    demands[1:] = rng.integers(1, min(capacity // 5, 50) + 1, size=num_customers)

    # --- 时间窗 ---
    if instance_type.endswith('1'):
        # 窄时间窗 → 车容量小，时间约束紧
        tw_width_factor = 0.2
        horizon = 24.0
    else:
        # 宽时间窗 → 车容量大，时间约束松
        tw_width_factor = 0.6
        horizon = 48.0

    depot_tw_start, depot_tw_end = 0.0, horizon

    tw_start = np.zeros(total_nodes, dtype=np.float32)
    tw_end = np.zeros(total_nodes, dtype=np.float32)
    tw_start[0], tw_end[0] = depot_tw_start, depot_tw_end

    for i in range(1, total_nodes):
        # 基于到 depot 的距离生成时间窗中心
        dist_to_depot = dist_mat[i, 0]
        center = dist_to_depot * 10 + rng.uniform(0, horizon * 0.4)
        width = horizon * tw_width_factor * rng.uniform(0.5, 1.5)
        tw_start[i] = max(0, center - width / 2)
        tw_end[i] = min(horizon, center + width / 2)

    # --- 服务时间 ---
    service_time = np.zeros(total_nodes, dtype=np.float32)
    service_time[1:] = 0.1 + demands[1:].astype(np.float32) * 0.05

    # --- 贪心构造初始解（用作训练参考） ---
    route, route_cost = _greedy_cvrptw_solution(
        coords, demands, tw_start, tw_end, service_time,
        capacity, dist_mat
    )

    return {
        'coords': coords,
        'demands': demands,
        'tw_start': tw_start,
        'tw_end': tw_end,
        'service_time': service_time,
        'routes': route,
        'opt_cost': route_cost,
    }


def _greedy_cvrptw_solution(
    coords, demands, tw_start, tw_end, service_time,
    capacity, dist_mat,
    max_route_len: int = 200,
) -> tuple[np.ndarray, float]:
    """
    贪心构造 CVRPTW 的初始解。

    算法：最近邻 + 时间窗可行性检查。
    返回 (padded_route, total_cost)。
    """
    total_nodes = len(coords)
    unvisited = set(range(1, total_nodes))
    routes_segments = []

    max_routes = len(unvisited) + 1  # 安全上限：每辆车至少服务 1 个客户
    route_count = 0

    while unvisited:
        route_count += 1
        if route_count > max_routes:
            # 安全防护：如果生成路线数超过客户数，说明存在不可服务的孤立客户
            # 将剩余客户强制插入（忽略 TW，使用 depot→客户→depot 的退化路径）
            for j in list(unvisited):
                segment = [0, j, 0]
                routes_segments.append(segment)
                unvisited.discard(j)
            break

        current = 0  # depot
        current_time = 0.0
        current_load = 0
        segment = [0]

        while unvisited:
            # 找最近的可行客户
            best_next = None
            best_cost = float('inf')

            for j in unvisited:
                if current_load + demands[j] > capacity:
                    continue
                travel_time = dist_mat[current, j]
                arrive_time = max(current_time + service_time[current] + travel_time, tw_start[j])
                if arrive_time <= tw_end[j]:
                    # 还要检查返回 depot 是否可行
                    return_time = arrive_time + service_time[j] + dist_mat[j, 0]
                    if return_time <= tw_end[0]:
                        c = dist_mat[current, j]
                        if c < best_cost:
                            best_cost = c
                            best_next = j
                            best_arrive = arrive_time

            if best_next is None:
                break  # 当前路线已满，回 depot 开始新路线

            segment.append(best_next)
            unvisited.remove(best_next)
            current_time = best_arrive
            current_load += demands[best_next]
            current = best_next

        segment.append(0)  # 回 depot
        routes_segments.append(segment)

    # 拼接所有路线段
    full_route = []
    for seg in routes_segments:
        full_route.extend(seg[:-1])  # 去掉中间的 depot
    full_route.append(0)  # 最后一个 depot

    route = np.array(full_route, dtype=np.int32)
    route = np.pad(route, (0, max(0, max_route_len - len(route))), constant_values=0)

    # 计算总成本
    total_cost = 0.0
    for i in range(len(full_route) - 1):
        total_cost += dist_mat[full_route[i], full_route[i + 1]]

    return route, total_cost


# ============================================================
# 批量数据生成
# ============================================================

def generate_dataset(
    num_instances: int = 1280,
    num_customers: int = 50,
    instance_type: str = 'R1',
    capacity: int = 50,
    max_route_len: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    批量生成 CVRPTW 数据集。
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31, size=num_instances, dtype=np.int32)

    coords_list, demands_list = [], []
    tw_start_list, tw_end_list = [], []
    service_time_list, routes_list = [], []
    opt_costs_list = []

    for i, s in enumerate(seeds):
        inst = generate_solomon_instance(
            num_customers=num_customers,
            instance_type=instance_type,
            capacity=capacity,
            seed=int(s),
        )
        coords_list.append(inst['coords'])
        demands_list.append(inst['demands'])
        tw_start_list.append(inst['tw_start'])
        tw_end_list.append(inst['tw_end'])
        service_time_list.append(inst['service_time'])

        # 对齐 route 长度
        route = inst['routes']
        route = np.pad(route, (0, max(0, max_route_len - len(route))), constant_values=0)
        routes_list.append(route[:max_route_len])
        opt_costs_list.append(inst['opt_cost'])

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_instances} instances...")

    return {
        'coords': np.stack(coords_list, axis=0).astype(np.float32),
        'demands': np.stack(demands_list, axis=0).astype(np.int32),
        'tw_start': np.stack(tw_start_list, axis=0).astype(np.float32),
        'tw_end': np.stack(tw_end_list, axis=0).astype(np.float32),
        'service_time': np.stack(service_time_list, axis=0).astype(np.float32),
        'routes': np.stack(routes_list, axis=0).astype(np.int32),
        'opt_costs': np.array(opt_costs_list, dtype=np.float32),
    }


def convert_tsptw_to_cvrptw(
    tsptw_data: dict[str, np.ndarray],
    capacity: int = 50,
) -> dict[str, np.ndarray]:
    """
    从 TSPTW 数据转换为 CVRPTW 数据。

    TSPTW 数据通常包含：
    - coords: (N, nodes, 2)
    - tw_start, tw_end: (N, nodes)
    - tour: (N, nodes) — TSP 路径

    需要额外生成：
    - demands: 随机生成
    - service_time: 根据 demand 估算
    - routes: 可能违反容量约束，需要重新求解
    """
    num_instances = tsptw_data['coords'].shape[0]
    num_nodes = tsptw_data['coords'].shape[1]
    rng = np.random.default_rng(42)

    demands = rng.integers(1, min(capacity // 5, 10) + 1, size=(num_instances, num_nodes))
    demands[:, 0] = 0  # depot demand = 0

    service_time = np.zeros_like(demands, dtype=np.float32)
    service_time[:, 1:] = 0.1 + demands[:, 1:].astype(np.float32) * 0.05

    # TSPTW 的 tour 是 TSP 解，需要转换为 CVRP 路径（插入 depot 分隔符）
    # 简化：直接用贪心法重新求解
    routes_list, costs_list = [], []
    for i in range(num_instances):
        diff = tsptw_data['coords'][i][:, None, :] - tsptw_data['coords'][i][None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
        route, cost = _greedy_cvrptw_solution(
            tsptw_data['coords'][i], demands[i],
            tsptw_data['tw_start'][i], tsptw_data['tw_end'][i],
            service_time[i], capacity, dist_mat, max_route_len=200,
        )
        routes_list.append(route)
        costs_list.append(cost)
        if (i + 1) % 50 == 0:
            print(f"  Converted {i + 1}/{num_instances} TSPTW → CVRPTW...")

    # 对齐 route 长度
    max_len = max(len(r) for r in routes_list)
    routes_padded = np.stack([
        np.pad(r, (0, max_len - len(r)), constant_values=0)
        for r in routes_list
    ], axis=0)

    return {
        'coords': tsptw_data['coords'].astype(np.float32),
        'demands': demands.astype(np.int32),
        'tw_start': tsptw_data['tw_start'].astype(np.float32),
        'tw_end': tsptw_data['tw_end'].astype(np.float32),
        'service_time': service_time.astype(np.float32),
        'routes': routes_padded.astype(np.int32),
        'opt_costs': np.array(costs_list, dtype=np.float32),
    }


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CVRPTW dataset')
    parser.add_argument('--problem_size', type=int, default=50,
                        help='Number of customers (excluding depot)')
    parser.add_argument('--num_instances', type=int, default=1280,
                        help='Number of instances to generate')
    parser.add_argument('--type', type=str, default='R1',
                        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2', 'mixed'],
                        help='Solomon instance type')
    parser.add_argument('--capacity', type=int, default=50,
                        help='Vehicle capacity')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--convert_from_tsptw', type=str, default=None,
                        help='Path to TSPTW .npz to convert')
    args = parser.parse_args()

    if args.convert_from_tsptw:
        print(f"Converting TSPTW data from {args.convert_from_tsptw}...")
        tsptw = dict(np.load(args.convert_from_tsptw))
        dataset = convert_tsptw_to_cvrptw(tsptw, capacity=args.capacity)
    else:
        print(f"Generating {args.num_instances} {args.type}-type CVRPTW instances "
              f"({args.problem_size} customers, capacity={args.capacity})...")
        dataset = generate_dataset(
            num_instances=args.num_instances,
            num_customers=args.problem_size,
            instance_type=args.type,
            capacity=args.capacity,
            seed=args.seed,
        )

    np.savez_compressed(args.output, **dataset)
    print(f"Saved to {args.output}")
    print(f"Keys: {list(dataset.keys())}")
    for k, v in dataset.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
