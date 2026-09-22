"""
冷链 CVRPTW 数据生成（Exp-10 Step 1）。

扩展 CVRPTW 5D → 6D: [x, y, demand, tw_start, tw_end, temp_class]

温度等级:
  0 = 常温 (ambient, 无特殊要求)
  1 = 冷藏 (refrigerated, ~4°C)
  2 = 冷冻 (frozen, ~-18°C)

车辆类型:
  0 = 常温车 (仅常温)
  1 = 冷藏车 (冷藏+常温)
  2 = 冷冻车 (冷冻+冷藏+常温, 多温区)

用法:
    python "C-VRP_Cold-chainVehicleRoutingProblem/data/generate_coldchain_data.py" \
        --problem_size 50 --num_instances 128 --type R1 --capacity 50 \
        --output "C-VRP_Cold-chainVehicleRoutingProblem/data/coldchain50_r1_test.npz"
"""

import sys, os
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT
sys.path.insert(0, str(MASKCO_ROOT))

import numpy as np
import argparse

# 温度等级分布概率
TEMP_DIST = [0.4, 0.35, 0.25]  # 常温, 冷藏, 冷冻


def generate_coldchain_instance(
    num_customers=50, instance_type='R1', capacity=50, seed=0,
    edod=0.0,  # Phase B: 0.0=静态, 0.2=弱动态, 0.5=中动态, 0.8=强动态
    asymmetric=False,      # 工业落地1: 非对称交通矩阵
    perturb_prob=0.0,      # 工业落地2: 长尾扰动概率
    perturb_scale=3.0,     # 扰动放大系数
    temp_dist=None,        # 温区分布 [常温, 冷藏, 冷冻]，None=默认 [0.4, 0.35, 0.25]
):
    """生成单实例冷链 CVRPTW 数据。"""
    rng = np.random.default_rng(seed)
    total_nodes = num_customers + 1  # +depot
    depot_x, depot_y = 0.5, 0.5

    # --- 坐标 ---
    if instance_type.startswith('C'):
        n_clusters = rng.integers(2, 5)
        centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
        coords_list = [[depot_x, depot_y]]
        for i in range(num_customers):
            c = centers[i % n_clusters]
            coords_list.append([np.clip(c[0] + rng.normal(0, 0.05), 0.02, 0.98),
                                np.clip(c[1] + rng.normal(0, 0.05), 0.02, 0.98)])
    elif instance_type.startswith('RC'):
        n_half = num_customers // 2
        coords_list = [[depot_x, depot_y]]
        for _ in range(n_half):
            coords_list.append([rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)])
        n_clusters = rng.integers(2, 4)
        centers = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
        for i in range(num_customers - n_half):
            c = centers[i % n_clusters]
            coords_list.append([np.clip(c[0] + rng.normal(0, 0.05), 0.02, 0.98),
                                np.clip(c[1] + rng.normal(0, 0.05), 0.02, 0.98)])
    else:
        coords_list = [[depot_x, depot_y]]
        for _ in range(num_customers):
            coords_list.append([rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)])

    coords = np.array(coords_list, dtype=np.float32)

    # --- 距离矩阵 ---
    diff = coords[:, None, :] - coords[None, :, :]
    euclidean_dist = np.sqrt((diff ** 2).sum(axis=-1))

    # --- 非对称交通矩阵 (工业落地1) ---
    if asymmetric:
        # 基础: 欧式距离 → 加非对称噪声模拟真实路网
        # A→B ≠ B→A: 单行道、红绿灯、立交桥导致
        asymmetry_noise = rng.uniform(0.85, 1.35, size=(total_nodes, total_nodes))
        asymmetry_noise = (asymmetry_noise + 1.0 / (asymmetry_noise.T + 1e-8)) / 2  # make asymmetric
        cost_matrix = euclidean_dist * asymmetry_noise
        # 早晚高峰效应: 特定时间段的边成本增加
        rush_hour_mask = rng.random((total_nodes, total_nodes)) < 0.15
        cost_matrix[rush_hour_mask] *= rng.uniform(1.2, 2.0, size=rush_hour_mask.sum())
        dist_mat = cost_matrix
    else:
        dist_mat = euclidean_dist

    # --- 长尾扰动 (工业落地2) ---
    perturbed_service = np.zeros(total_nodes, dtype=bool)
    perturbed_edges = []
    if perturb_prob > 0:
        # 随机选择节点延长服务时间 (模拟客户扯皮)
        n_perturb = max(1, int(num_customers * perturb_prob))
        perturb_indices = rng.choice(num_customers, size=n_perturb, replace=False) + 1
        perturbed_service[perturb_indices] = True
        # 随机增加部分边的成本 (模拟突发交通管制)
        n_edge_perturb = max(1, int(total_nodes * total_nodes * perturb_prob * 0.3))
        ei = rng.integers(0, total_nodes, size=n_edge_perturb)
        ej = rng.integers(0, total_nodes, size=n_edge_perturb)
        perturbed_edges = list(zip(ei, ej))

    # --- 需求量 ---
    demands = np.zeros(total_nodes, dtype=np.int32)
    demands[1:] = rng.integers(1, min(capacity // 5, 50) + 1, size=num_customers)

    # --- 温度等级 ---
    temp_class = np.zeros(total_nodes, dtype=np.int32)  # depot=0
    _td = TEMP_DIST if temp_dist is None else temp_dist
    temp_class[1:] = rng.choice(3, size=num_customers, p=_td)

    # --- 时间窗 ---
    if instance_type.endswith('1'):
        tw_width, horizon = 0.2, 24.0
    else:
        tw_width, horizon = 0.6, 48.0

    tw_start = np.zeros(total_nodes, dtype=np.float32)
    tw_end = np.zeros(total_nodes, dtype=np.float32)
    tw_start[0], tw_end[0] = 0.0, horizon

    for i in range(1, total_nodes):
        center = dist_mat[i, 0] * 10 + rng.uniform(0, horizon * 0.4)
        width = horizon * tw_width * rng.uniform(0.5, 1.5)
        tw_start[i] = max(0, center - width / 2)
        tw_end[i] = min(horizon, center + width / 2)

    # --- 服务时间 ---
    service_time = np.zeros(total_nodes, dtype=np.float32)
    service_time[1:] = 0.1 + demands[1:].astype(np.float32) * 0.05
    # 长尾扰动: 被选中的节点服务时间放大
    if perturb_prob > 0:
        service_time[perturbed_service] *= rng.uniform(perturb_scale * 0.5, perturb_scale,
                                                        size=perturbed_service.sum())
        # 被选中的边成本放大 (模拟交通管制)
        for (i, j) in perturbed_edges:
            if i != j:
                dist_mat[i, j] *= rng.uniform(1.5, 5.0)

    # --- 揭示时间（Phase B：动态场景） ---
    reveal_time = np.zeros(total_nodes, dtype=np.float32)  # depot=0
    visible_mask = np.ones(total_nodes, dtype=np.float32)   # 1=可见, 0=未来订单(屏蔽)
    if edod > 0:
        n_dynamic = int(num_customers * edod)
        dynamic_indices = rng.choice(num_customers, size=n_dynamic, replace=False) + 1
        # 修复 2026-08-26（导师 D1）：约束 reveal_time 满足 release feasibility。
        # 原 rng.uniform(0, horizon*0.8) 导致 42% 未来客户 reveal_time > tw_end，
        # 即订单揭示时时间窗已关闭，non-anticipatory 下结构上无法服务。
        # 必要可服务条件：r_i + τ_i + s_i <= b_i（τ_i 为从 depot 到 i 的乐观旅行时间下界）。
        for idx in dynamic_indices:
            travel_min = dist_mat[idx, 0]  # travel = dist（speed=1.0，与解码器一致）
            max_reveal = max(0.0, tw_end[idx] - travel_min - service_time[idx])
            reveal_time[idx] = rng.uniform(0, max_reveal) if max_reveal > 0 else 0.0
        # 未来订单在初始时刻不可见 (reveal_time > 0)
        visible_mask[dynamic_indices] = 0.0

    # --- 贪心构造初始解（Oracle: 使用全部信息） ---
    route, route_cost = _greedy_coldchain_solution(
        coords, demands, tw_start, tw_end, service_time,
        temp_class, capacity, dist_mat
    )

    # --- 历史 P0-4 代理特征（仅保留 checkpoint / 数据格式兼容） ---
    # 下面两个数组是基于参考路线的 delivery-style 累计量，不是 v4
    # pickup-to-depot 的执行轨迹真值。C0 权威 quality/energy 只能由
    # strict_online_env + coldchain_state + coldchain_evaluator 产生。
    K_TEMP = np.array([0.01, 0.002, 0.0002], dtype=np.float32)  # 常温/冷藏/冷冻
    quality_loss = np.zeros(total_nodes, dtype=np.float32)
    cum_time = 0.0
    prev = 0
    for node in route:
        if node == 0:
            cum_time = 0.0; prev = 0; continue
        cum_time += dist_mat[prev, node] + service_time[prev]
        k = K_TEMP[min(int(temp_class[node]), 2)]
        quality_loss[node] = 1.0 - np.exp(-k * cum_time)
        prev = node

    # --- 制冷能耗矩阵 (P0-4 冷链物理建模) ---
    T_TARGET = np.array([25.0, 4.0, -18.0], dtype=np.float32)  # 0=常温, 1=冷藏, 2=冷冻
    T_OUTSIDE = 25.0
    alpha, beta = 1.0, 0.05
    delta_T = np.abs(T_OUTSIDE - T_TARGET[temp_class.astype(np.int32)])  # (nodes,)
    energy_mat = alpha * dist_mat * (1.0 + beta * delta_T[None, :])  # (nodes, nodes)

    return {
        'coords': coords,
        'demands': demands,
        'tw_start': tw_start,
        'tw_end': tw_end,
        'service_time': service_time,
        'temp_class': temp_class,
        'reveal_time': reveal_time,
        'visible_mask': visible_mask,  # P0-2: 因果屏蔽
        # 通用旧键继续供历史模型读取；显式别名防止新代码误当 C0 标签。
        'quality_loss': quality_loss,
        'energy_mat': energy_mat,
        'legacy_delivery_reference_quality': quality_loss.copy(),
        'legacy_edge_energy_proxy': energy_mat.copy(),
        # v4 取货语义：货物在 service_finish 入舱时从该品质开始计时。
        'initial_quality': np.ones(total_nodes, dtype=np.float32),
        'routes': route,
        'opt_costs': route_cost,
        'dist_mat': dist_mat,
        'asymmetric': asymmetric,
        'perturb_prob': perturb_prob,
    }


def _vehicle_can_serve(vehicle_temp_class, customer_temp_class):
    """检查车辆是否能服务该温度的客户。"""
    if vehicle_temp_class >= customer_temp_class:
        return True
    return False


def _greedy_coldchain_solution(
    coords, demands, tw_start, tw_end, service_time,
    temp_class, capacity, dist_mat, max_route_len=200,
):
    """贪心构造含温度约束的初始解。"""
    total_nodes = len(coords)
    unvisited = set(range(1, total_nodes))
    routes_segments = []

    VEHICLE_TYPES = [2, 1, 0]  # 优先用多温区车

    while unvisited:
        best_segment = None
        best_cost = float('inf')

        for vtype in VEHICLE_TYPES:
            current = 0
            current_time = 0.0
            current_load = 0
            segment = [0]
            visited_this_route = set()

            while unvisited - visited_this_route:
                best_next = None
                best_edge_cost = float('inf')

                for j in unvisited - visited_this_route:
                    if not _vehicle_can_serve(vtype, int(temp_class[j])):
                        continue
                    if current_load + demands[j] > capacity:
                        continue
                    travel_time = dist_mat[current, j]
                    arrive_time = max(current_time + service_time[current] + travel_time, tw_start[j])
                    if arrive_time <= tw_end[j]:
                        return_time = arrive_time + service_time[j] + dist_mat[j, 0]
                        if return_time <= tw_end[0]:
                            if dist_mat[current, j] < best_edge_cost:
                                best_edge_cost = dist_mat[current, j]
                                best_next = j
                                best_arrive = arrive_time

                if best_next is None:
                    break

                segment.append(best_next)
                visited_this_route.add(best_next)
                current_time = best_arrive
                current_load += demands[best_next]
                current = best_next

            segment.append(0)
            if len(segment) > 2:  # 至少访问了 1 个客户
                route_cost = sum(dist_mat[segment[i], segment[i+1]] for i in range(len(segment)-1))
                if route_cost < best_cost:
                    best_cost = route_cost
                    best_segment = (segment, visited_this_route)

        if best_segment is None:
            # 退化：逐个插入剩余客户
            for j in list(unvisited):
                routes_segments.append([0, j, 0])
                unvisited.discard(j)
            break

        segment, visited = best_segment
        routes_segments.append(segment)
        unvisited -= visited

    full_route = []
    for seg in routes_segments:
        full_route.extend(seg[:-1])
    full_route.append(0)

    route = np.array(full_route, dtype=np.int32)
    route = np.pad(route, (0, max(0, max_route_len - len(route))), constant_values=0)
    total_cost = sum(dist_mat[full_route[i], full_route[i+1]] for i in range(len(full_route)-1))
    return route, total_cost


def generate_dataset(num_instances, num_customers, instance_type, capacity, max_route_len=200, seed=0, edod=0.0, asymmetric=False, perturb_prob=0.0, perturb_scale=3.0, temp_dist=None):
    """批量生成冷链数据集。"""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=num_instances, dtype=np.int32)

    out = {'coords': [], 'demands': [], 'tw_start': [], 'tw_end': [],
           'service_time': [], 'temp_class': [], 'reveal_time': [], 'visible_mask': [],
           'quality_loss': [], 'energy_mat': [],
           'legacy_delivery_reference_quality': [], 'legacy_edge_energy_proxy': [],
           'initial_quality': [],
           'routes': [], 'opt_costs': [], 'dist_mat': []}

    for i, s in enumerate(seeds):
        inst = generate_coldchain_instance(num_customers, instance_type, capacity, int(s), edod=edod, asymmetric=asymmetric, perturb_prob=perturb_prob, perturb_scale=perturb_scale, temp_dist=temp_dist)
        for k in out:
            arr = inst[k]
            if k == 'routes':
                arr = np.pad(arr, (0, max(0, max_route_len - len(arr))), constant_values=0)[:max_route_len]
            out[k].append(arr)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_instances} instances...")

    return {k: np.stack(v, axis=0).astype(
        np.float32 if ('cost' in k or 'coord' in k or 'tw_' in k or 'service' in k or 'reveal' in k or 'dist_mat' in k or 'quality' in k or 'energy' in k)
        else np.int32) for k, v in out.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--num_instances', type=int, default=128)
    parser.add_argument('--type', type=str, default='R1')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--edod', type=float, default=0.0,
                        help='动态程度: 0.0=静态, 0.2=弱动态, 0.5=中动态, 0.8=强动态')
    parser.add_argument('--asymmetric', action='store_true', default=False,
                        help='工业落地1: 生成非对称交通矩阵 (模拟真实路网)')
    parser.add_argument('--perturb_prob', type=float, default=0.0,
                        help='工业落地2: 长尾扰动概率 (0.05=5%%节点/边被污染)')
    parser.add_argument('--perturb_scale', type=float, default=3.0,
                        help='扰动放大系数 (默认3x)')
    parser.add_argument('--temp_dist', type=str, default=None,
                        help='温区分布 "常温,冷藏,冷冻" 概率，如 "0,0,1"=全冷冻（温度驱动验证）')
    args = parser.parse_args()

    # 解析温区分布
    temp_dist = None
    if args.temp_dist is not None:
        temp_dist = [float(x) for x in args.temp_dist.split(',')]
        assert abs(sum(temp_dist) - 1.0) < 1e-6, f"temp_dist 必须和为 1，得到 {temp_dist}"

    print(f"Generating {args.num_instances} {args.type} cold-chain instances "
          f"({args.problem_size} customers, capacity={args.capacity}, EDoD={args.edod})")
    if temp_dist is not None:
        print(f"  Temp distribution: {temp_dist}")
    if args.asymmetric:
        print(f"  Asymmetric cost matrix: ON")
    if args.perturb_prob > 0:
        print(f"  Long-tail perturbations: prob={args.perturb_prob}, scale={args.perturb_scale}")
    dataset = generate_dataset(args.num_instances, args.problem_size,
                               args.type, args.capacity, seed=args.seed, edod=args.edod,
                               asymmetric=args.asymmetric,
                               perturb_prob=args.perturb_prob, perturb_scale=args.perturb_scale,
                               temp_dist=temp_dist)
    np.savez_compressed(args.output, **dataset)
    print(f"Saved to {args.output}")
    for k, v in dataset.items():
        print(f"  {k}: {v.shape} {v.dtype}")
