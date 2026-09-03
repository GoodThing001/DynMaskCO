"""
Phase 3a: Resource-State Feasible Decoder v3 — 完整约束掩码

v1: arrival+load extraction (正确, viol 12.7→4.8, −62%)
v2: 重写 extraction 但 generation 阶段空路线下逻辑错误 (viol 4.8→43.9)
v3: 回退到 v1 extraction (已验证正确) + 往返可行性 + 安全裕度

核心约束 (per-edge, 按此顺序检查):
  1. i 可达 (arrival_i >= 0) 或 i=0
  2. j 未访问 (j not in current route) 或 j=0
  3. TW: arrive[i] + service[i] + dist(i,j)/speed <= tw_end[j] - MARGIN
  4. CAP: load_at_i + demand[j] <= capacity (j≠0)
  5. ROUND-TRIP: j 服务后能否在 depot tw_end 前返回 (关键增设!)

设计: 纯 NumPy, 每步 decode_step 前调用。
"""

import numpy as np
from typing import Optional

TW_SAFETY_MARGIN = 0.05  # hours


def extract_arrival_times(
    neighbors: np.ndarray,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service_time: np.ndarray,
    speed: float = 1.0,
):
    """v1 版本 — 已验证正确。沿 successor chain 计算到达时间。"""
    B = neighbors.shape[0]
    nodes = neighbors.shape[1]

    arrivals = np.full((B, nodes), -1.0, dtype=np.float32)
    in_route = np.zeros((B, nodes), dtype=bool)
    in_route[:, 0] = True
    arrivals[:, 0] = 0.0

    for b in range(B):
        nbrs = neighbors[b]
        # Mark nodes in route
        for n in range(1, nodes):
            pred = nbrs[n, 0]
            succ = nbrs[n, 1]
            if pred >= 0 and pred < nodes:
                in_route[b, n] = True
            elif succ >= 0 and succ < nodes:
                in_route[b, n] = True

        # Forward pass from depot
        visited = set()
        for _ in range(nodes):
            # Find a route start: node whose pred is 0, not yet visited
            found_start = False
            for n in range(1, nodes):
                if nbrs[n, 0] == 0 and n not in visited:
                    current = n
                    t = 0.0
                    # travel from depot to first node of segment
                    d = np.sqrt(((coords[b, 0] - coords[b, current]) ** 2).sum() + 1e-10)
                    t = service_time[b, 0] + d / speed
                    t = max(t, tw_start[b, current])
                    arrivals[b, current] = t
                    visited.add(current)
                    found_start = True
                    break
            if not found_start:
                break

            # Follow successor chain
            for _ in range(nodes):
                succ = nbrs[current, 1]
                if succ <= 0 or succ >= nodes or succ in visited:
                    break
                d = np.sqrt(((coords[b, current] - coords[b, succ]) ** 2).sum() + 1e-10)
                t += service_time[b, current] + d / speed
                t = max(t, tw_start[b, succ])
                arrivals[b, succ] = t
                visited.add(succ)
                current = succ

    return arrivals, in_route


def extract_route_loads(
    neighbors: np.ndarray,
    demands: np.ndarray,
    in_route: np.ndarray,
    capacity: float,
):
    """v1 版本 — 已验证正确。沿 successor chain 计算累计载重。"""
    B = neighbors.shape[0]
    nodes = neighbors.shape[1]
    loads = np.full((B, nodes), -1.0, dtype=np.float32)
    loads[:, 0] = 0.0

    for b in range(B):
        nbrs = neighbors[b]
        for start_node in range(1, nodes):
            if nbrs[start_node, 0] == 0 and in_route[b, start_node]:
                current = start_node
                load = demands[b, start_node]
                loads[b, current] = load
                for _ in range(nodes):
                    succ = nbrs[current, 1]
                    if succ <= 0 or succ >= nodes:
                        break
                    load += demands[b, succ]
                    loads[b, succ] = load
                    current = succ

    return loads


def compute_resource_feasibility_mask(
    neighbors: np.ndarray,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service_time: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    speed: float = 1.0,
    tw_feasible_static: Optional[np.ndarray] = None,
):
    """
    计算资源可行性掩码 v3。

    边 (i,j) 全部的检查:
      1. i 可达 或 i=0
      2. j 未访问 或 j=0
      3. TW: arrive_i + svc_i + dist_ij/speed <= tw_end_j - MARGIN
      4. CAP: load_i + demand_j <= capacity (j≠0)
      5. ROUND-TRIP: j 服务后能否在 tw_end[0] 前返回 depot
    """
    B = neighbors.shape[0]
    N = neighbors.shape[1]

    # Step 1-2: v1 extraction (verified correct)
    arrivals, in_route = extract_arrival_times(
        neighbors, coords, tw_start, tw_end, service_time, speed)
    loads = extract_route_loads(neighbors, demands, in_route, capacity)

    # Step 3: Precompute distance/travel matrices
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
    travel_mat = dist_mat / speed                        # (B, N, N)
    travel_ij = travel_mat + service_time[:, :, None]    # (B, N, N): travel_time(i,j) + service[i]

    # Step 4: Build mask
    mask = np.ones((B, N, N), dtype=bool)

    for b in range(B):
        tw_e = tw_end[b]
        tw_s = tw_start[b]
        dmd = demands[b]
        depot_close = tw_e[0] - TW_SAFETY_MARGIN

        for i in range(N):
            # --- Row: i must be reachable ---
            if arrivals[b, i] < 0 and i != 0:
                mask[b, i, :] = False
                continue

            arr_i = arrivals[b, i] if arrivals[b, i] >= 0 else 0.0
            load_i = loads[b, i] if loads[b, i] >= 0 else 0.0

            for j in range(N):
                if i == j:
                    mask[b, i, j] = False
                    continue

                # --- j = 0: close route, check depot return ---
                if j == 0:
                    ret = arr_i + travel_ij[b, i, 0]  # arrive_i + svc_i + dist(i,0)
                    if ret > depot_close:
                        mask[b, i, 0] = False
                    continue

                # --- j already in route ---
                if in_route[b, j]:
                    mask[b, i, j] = False
                    continue

                # --- TW: arrive at j ---
                arrive_j = arr_i + travel_ij[b, i, j]  # > arrive_i + svc_i + dist(i,j)
                arrive_j = max(arrive_j, tw_s[j])
                if arrive_j > tw_e[j] - TW_SAFETY_MARGIN:
                    mask[b, i, j] = False
                    continue

                # --- Capacity ---
                if load_i + dmd[j] > capacity:
                    mask[b, i, j] = False
                    continue

                # --- Round-trip: j can return to depot ---
                depart_j = arrive_j + service_time[b, j]
                ret_j = depart_j + dist_mat[b, j, 0] / speed
                if ret_j > depot_close:
                    mask[b, i, j] = False
                    continue

    # --- Depot row: start a new route from depot ---
    mask[:, 0, 0] = False
    for b in range(B):
        tw_e = tw_end[b]
        tw_s = tw_start[b]
        dmd = demands[b]
        depot_close = tw_e[0] - TW_SAFETY_MARGIN
        for j in range(1, N):
            if in_route[b, j]:
                mask[b, 0, j] = False
                continue
            arr = service_time[b, 0] + dist_mat[b, 0, j] / speed
            arr = max(arr, tw_s[j])
            if arr > tw_e[j] - TW_SAFETY_MARGIN:
                mask[b, 0, j] = False
                continue
            if dmd[j] > capacity:
                mask[b, 0, j] = False
                continue
            # round-trip from depot via j
            ret = arr + service_time[b, j] + dist_mat[b, j, 0] / speed
            if ret > depot_close:
                mask[b, 0, j] = False

    # 修复 2026-08-26：删除「Always allow depot reachable」（原 mask[:, :, 0] = True
    # 覆盖了上面 j==0 分支的 depot return 检查，导致超时的返回 depot 边被重新允许）。
    # depot return 可行性已在主循环 j==0 分支正确设置（ret > depot_close → mask=False）。

    # Ensure at least SOME edges from depot for generation phase
    for b in range(B):
        if not mask[b, 0, 1:].any():
            # Fallback: allow depot to any unvisited node
            mask[b, 0, 1:] = ~in_route[b, 1:]

    return mask
