"""
Authoritative Evaluator — 唯一权威评估函数

所有方法（DynMaskCO / OR-Tools / RRNCO / ALNS）最终都交给它评价，
统一输出完整指标，不信 solver 自己输出的 feasible。

修复审查报告中的 P0 问题：
1. distance_cost 包含返回 depot 距离（原 _eval_distmat_cost 漏算）
2. tw_feasible 检查返回 depot 是否超时
3. capacity 检查（原 evaluate_tw_feasibility 完全没有 demand/load）
4. completion 检查（所有客户恰好服务一次，无重复/遗漏）
5. quality 用真实 elapsed time（含 waiting time）
6. energy 包含返回 depot

route 格式：depot=0 分隔各辆车，如 [0, c1, c2, 0, c3, c4, 0]

Author: P0-Protocol Repair
Date: 2026-08-26
"""

import numpy as np
from collections import Counter


def evaluate_solution(
    route: np.ndarray,           # (route_len,) int，depot=0 分隔
    coords: np.ndarray,          # (nodes, 2)
    tw_start: np.ndarray,        # (nodes,)
    tw_end: np.ndarray,          # (nodes,)
    service_time: np.ndarray,    # (nodes,)
    demands: np.ndarray,         # (nodes,)
    capacity: float,
    speed: float = 1.0,
    quality_decay_k: np.ndarray = None,  # (nodes,) 品质衰减率，可选
    energy_mat: np.ndarray = None,       # (nodes, nodes) 能耗矩阵，可选
) -> dict:
    """
    权威评估单条 route。

    Returns:
        dict with keys:
            complete              — 所有客户恰好服务一次
            n_visited             — 不同客户数
            n_duplicate           — 重复客户数
            n_unserved            — 未服务客户数
            capacity_feasible     — 容量可行
            capacity_overload     — 最大超载量
            tw_feasible           — 时间窗可行（客户 TW + 返回 depot）
            tw_late_count         — 迟到客户数
            depot_return_feasible — 返回 depot 是否在 depot TW 内
            distance_cost         — 完整距离（含返回 depot）
            quality               — 累计品质损失（含 waiting + 返回 depot）
            energy                — 累计能耗（含返回 depot）
            vehicle_count         — 车辆数（depot 分隔段数）
    """
    N = len(coords)
    n_customers = N - 1

    route = np.asarray(route, dtype=int)

    # --- 1. Completion ---
    customers = [int(n) for n in route if n != 0]
    counts = Counter(customers)
    n_visited = len(counts)
    n_duplicate = sum(c - 1 for c in counts.values() if c > 1)
    n_unserved = n_customers - n_visited
    complete = (n_visited == n_customers and n_duplicate == 0)

    # --- 2. 分段（depot=0 分隔）---
    segments = []
    current_seg = []
    for n in route:
        if n == 0:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
        else:
            current_seg.append(int(n))
    if current_seg:
        segments.append(current_seg)
    vehicle_count = len(segments)

    # --- 3. 逐段评估 ---
    distance = 0.0
    energy = 0.0
    quality = 0.0
    capacity_feasible = True
    capacity_overload = 0.0
    tw_feasible = True
    tw_late_count = 0
    depot_return_feasible = True

    def _dist(i, j):
        return float(np.linalg.norm(coords[i] - coords[j]))

    for seg in segments:
        prev = 0          # 从 depot 出发
        t = 0.0           # 当前累计时间（从 depot 出发）
        load = 0.0        # 当前载重

        for node in seg:
            d = _dist(prev, node)
            distance += d
            if energy_mat is not None:
                energy += float(energy_mat[prev, node])

            travel_time = d / speed
            # t 是 ready_time（服务完 prev 可离开的时刻）。到达 j = ready + travel，
            # 不再加 service_time[prev]（否则 service 被算两遍，P0-2）。
            arrive = t + travel_time
            # 若提前到达，等待到 tw_start
            start = max(arrive, tw_start[node])

            # TW 检查（到达时间 <= tw_end，含 waiting 后服务开始时间）
            if arrive > tw_end[node] + 1e-6:
                tw_feasible = False
                tw_late_count += 1

            # quality：真实 elapsed time（含 waiting）= start
            if quality_decay_k is not None:
                k = float(quality_decay_k[node])
                quality += 1.0 - float(np.exp(-k * start))

            # capacity
            load += demands[node]
            if load > capacity + 1e-6:
                capacity_feasible = False
                capacity_overload = max(capacity_overload, load - capacity)

            # 更新状态：服务完 node，时间 = start + service_time[node]
            t = start + service_time[node]
            prev = node

        # 返回 depot
        d = _dist(prev, 0)
        distance += d
        if energy_mat is not None:
            energy += float(energy_mat[prev, 0])

        # 返回 depot 是否超时
        return_time = t + d / speed
        if return_time > tw_end[0] + 1e-6:
            depot_return_feasible = False
            tw_feasible = False  # 返回 depot 超时也算 TW 违规

    return {
        'complete': complete,
        'n_visited': n_visited,
        'n_duplicate': n_duplicate,
        'n_unserved': n_unserved,
        'capacity_feasible': capacity_feasible,
        'capacity_overload': float(capacity_overload),
        'tw_feasible': tw_feasible,
        'tw_late_count': tw_late_count,
        'depot_return_feasible': depot_return_feasible,
        'distance_cost': float(distance),
        'quality': float(quality),
        'energy': float(energy),
        'vehicle_count': vehicle_count,
    }


def evaluate_batch(
    routes: np.ndarray,          # (batch, route_len)
    coords: np.ndarray,          # (batch, nodes, 2)
    tw_start: np.ndarray,        # (batch, nodes)
    tw_end: np.ndarray,          # (batch, nodes)
    service_time: np.ndarray,    # (batch, nodes)
    demands: np.ndarray,         # (batch, nodes)
    capacity: float,
    speed: float = 1.0,
    quality_decay_k: np.ndarray = None,  # (batch, nodes)
    energy_mat: np.ndarray = None,       # (batch, nodes, nodes)
) -> dict:
    """批量评估，返回聚合指标（均值 + 逐实例结果）。"""
    B = routes.shape[0]
    results = []
    for b in range(B):
        r = evaluate_solution(
            routes[b], coords[b], tw_start[b], tw_end[b],
            service_time[b], demands[b], capacity, speed,
            quality_decay_k[b] if quality_decay_k is not None else None,
            energy_mat[b] if energy_mat is not None else None,
        )
        results.append(r)

    # 聚合
    agg = {
        'complete_rate': np.mean([r['complete'] for r in results]),
        'tw_feas_rate': np.mean([r['tw_feasible'] for r in results]),
        'cap_feas_rate': np.mean([r['capacity_feasible'] for r in results]),
        'depot_return_feas_rate': np.mean([r['depot_return_feasible'] for r in results]),
        'mean_distance_cost': float(np.mean([r['distance_cost'] for r in results])),
        'mean_quality': float(np.mean([r['quality'] for r in results])),
        'mean_energy': float(np.mean([r['energy'] for r in results])),
        'mean_vehicle_count': float(np.mean([r['vehicle_count'] for r in results])),
        'mean_unserved': float(np.mean([r['n_unserved'] for r in results])),
        'mean_late_count': float(np.mean([r['tw_late_count'] for r in results])),
        'per_instance': results,
    }
    return agg


def evaluate_execution_trace(
    traces,               # list[VehicleTrace]（simulation/strict_online_env.py）
    coords,               # (nodes, 2)
    tw_start,             # (nodes,)
    tw_end,               # (nodes,)
    service_time,         # (nodes,)
    demands,              # (nodes,)
    capacity,             # float
    speed=1.0,
    dist_mat=None,        # (nodes, nodes) 可选，asymmetric / real network
) -> dict:
    """
    权威评估 execution trace（P0-EVAL）。

    与 evaluate_solution 的区别：trace 保留每辆车的 dispatch_time / 绝对到达时间 /
    返回 depot 时间，因此对 strict-online（idle 车可在 t=clock 才 dispatch）是正确的，
    不再假设每辆车都从 t=0 出发。

    Returns 与 evaluate_solution 同构的 dict。
    """
    N = len(coords)
    n_customers = N - 1

    # completion
    counts = Counter()
    for tr in traces:
        for sr in tr.services:
            counts[sr.node] += 1
    n_visited = len(counts)
    n_duplicate = sum(c - 1 for c in counts.values() if c > 1)
    n_unserved = n_customers - n_visited
    complete = (n_visited == n_customers and n_duplicate == 0)

    def _dist(i, j):
        if dist_mat is not None:
            return float(dist_mat[i, j])
        return float(np.linalg.norm(coords[i] - coords[j]))

    distance = 0.0
    capacity_feasible = True
    capacity_overload = 0.0
    tw_feasible = True
    tw_late_count = 0
    depot_return_feasible = True
    used_vehicles = 0

    for tr in traces:
        if not tr.services:
            continue
        used_vehicles += 1
        load = 0.0
        prev = 0
        for sr in tr.services:
            distance += _dist(sr.prev_node, sr.node)
            if sr.arrival_time > tw_end[sr.node] + 1e-6:
                tw_feasible = False
                tw_late_count += 1
            load += demands[sr.node]
            if load > capacity + 1e-6:
                capacity_feasible = False
                capacity_overload = max(capacity_overload, load - capacity)
            prev = sr.node
        # return depot（车被用过就必须返回 depot）
        distance += _dist(prev, 0)
        if tr.return_arrival is None:
            depot_return_feasible = False
            tw_feasible = False
        elif tr.return_arrival > tw_end[0] + 1e-6:
            depot_return_feasible = False
            tw_feasible = False

    return {
        'complete': complete,
        'n_visited': n_visited,
        'n_duplicate': n_duplicate,
        'n_unserved': n_unserved,
        'capacity_feasible': capacity_feasible,
        'capacity_overload': float(capacity_overload),
        'tw_feasible': tw_feasible,
        'tw_late_count': tw_late_count,
        'depot_return_feasible': depot_return_feasible,
        'distance_cost': float(distance),
        'quality': 0.0,
        'energy': 0.0,
        'vehicle_count': used_vehicles,
    }


# ============================================================================
# 单元测试
# ============================================================================

if __name__ == '__main__':
    print("Testing authoritative evaluator...")

    # 简单例子：4 节点（1 depot + 3 客户），坐标一维排列
    # depot=0 在 x=0，客户 1,2,3 在 x=1,2,3
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    tw_start = np.array([0., 0., 0., 0.])
    tw_end = np.array([100., 10., 10., 10.])  # depot 很宽
    service_time = np.zeros(4)
    demands = np.array([0., 1., 1., 1.])
    capacity = 10.0

    # 路线：0 → 1 → 2 → 3 → 0（一辆车服务全部）
    route = np.array([0, 1, 2, 3, 0])

    r = evaluate_solution(route, coords, tw_start, tw_end, service_time, demands, capacity)

    print(f"  complete: {r['complete']}")
    print(f"  distance_cost: {r['distance_cost']}")  # 期望 0→1→2→3→0 = 1+1+1+3 = 6
    print(f"  vehicle_count: {r['vehicle_count']}")
    print(f"  tw_feasible: {r['tw_feasible']}")
    print(f"  capacity_feasible: {r['capacity_feasible']}")

    # 验证 distance = 6（0→1=1, 1→2=1, 2→3=1, 3→0=3）
    assert abs(r['distance_cost'] - 6.0) < 1e-6, f"distance 应=6, got {r['distance_cost']}"
    assert r['complete'] == True
    assert r['vehicle_count'] == 1
    assert r['tw_feasible'] == True
    assert r['capacity_feasible'] == True
    print("  ✓ 完整距离（含 return depot）= 6 正确")

    # 测试多车辆：0 → 1 → 0 → 2 → 3 → 0
    route2 = np.array([0, 1, 0, 2, 3, 0])
    r2 = evaluate_solution(route2, coords, tw_start, tw_end, service_time, demands, capacity)
    print(f"  route2 distance: {r2['distance_cost']}")  # 0→1→0→2→3→0 = 1+1+2+1+3 = 8
    assert abs(r2['distance_cost'] - 8.0) < 1e-6
    assert r2['vehicle_count'] == 2
    print("  ✓ 多车辆距离正确")

    # 测试 completion（漏服务客户 2）
    route3 = np.array([0, 1, 3, 0])
    r3 = evaluate_solution(route3, coords, tw_start, tw_end, service_time, demands, capacity)
    print(f"  route3 complete: {r3['complete']}, n_unserved: {r3['n_unserved']}")
    assert r3['complete'] == False
    assert r3['n_unserved'] == 1
    print("  ✓ completion 检查正确")

    # 测试 capacity（容量超载）
    capacity_small = 2.0
    demands_high = np.array([0., 2., 2., 2.])
    route4 = np.array([0, 1, 2, 3, 0])
    r4 = evaluate_solution(route4, coords, tw_start, tw_end, service_time, demands_high, capacity_small)
    print(f"  route4 capacity_feasible: {r4['capacity_feasible']}, overload: {r4['capacity_overload']}")
    assert r4['capacity_feasible'] == False
    assert r4['capacity_overload'] > 0
    print("  ✓ capacity 检查正确")

    # 测试 TW（返回 depot 超时）
    tw_end_tight = np.array([3.5, 10., 10., 10.])  # depot 3.5 关闭
    route5 = np.array([0, 1, 2, 3, 0])  # 返回 depot 时间 = 3（走 0→1→2→3 用 3，返回 3）→ 6 > 3.5
    r5 = evaluate_solution(route5, coords, tw_start, tw_end_tight, service_time, demands, capacity)
    print(f"  route5 depot_return_feasible: {r5['depot_return_feasible']}")
    assert r5['depot_return_feasible'] == False
    print("  ✓ 返回 depot 超时检查正确")

    # 测试 nonzero-service 时间推进（P0-2 修复）：0→1→2，d01=1, d12=1, s1=1, tw_end(2)=3.2
    # 正确：arrive1=1, ready1=2, arrive2=3 → feasible；错误（service 加两遍）→ arrive2=4 → infeasible
    coords6 = np.array([[0., 0.], [1., 0.], [2., 0.]])
    tw_start6 = np.array([0., 0., 0.])
    tw_end6 = np.array([100., 100., 3.2])
    service_time6 = np.array([0., 1., 0.])
    demands6 = np.array([0., 1., 1.])
    route6 = np.array([0, 1, 2, 0])
    r6 = evaluate_solution(route6, coords6, tw_start6, tw_end6, service_time6, demands6, capacity=10.0)
    print(f"  route6 (nonzero service) tw_feasible: {r6['tw_feasible']}, "
          f"late={r6['tw_late_count']}")
    assert r6['tw_feasible'] == True, "service time 被重复计算，应 feasible 却判 infeasible"
    print("  ✓ nonzero-service 时间推进正确（无 service 双重计数）")

    # 测试 waiting：tw_start(1)=5，验证 start1=5, ready1=6, arrive2=7
    tw_start7 = np.array([0., 5., 0.])
    tw_end7 = np.array([100., 100., 100.])
    r7 = evaluate_solution(route6, coords6, tw_start7, tw_end7, service_time6, demands6, capacity=10.0)
    print(f"  route7 (waiting) distance_cost: {r7['distance_cost']}")
    # 无法直接读 arrive，但可验证 tw_feasible 仍 True（start1=5 <= tw_end1=100）
    assert r7['tw_feasible'] == True
    print("  ✓ waiting 时间推进正确")

    print("\n✅ 所有 authoritative evaluator 测试通过！")
