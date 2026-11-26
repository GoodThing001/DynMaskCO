# -*- coding: utf-8 -*-
"""标准任务核心：CVRP/VRPTW 实例、评价器与固定序列 Split 动态规划。

纯标准库实现，与冷链求解器完全隔离；用于 P5 的独立评价、Split 分车与正确性测试。
距离约定在 BKS 复核步骤中与官方值核对后锁定（见 P5_预登记.md 第 6 节）。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

EPS = 1e-9


# ---------------------------------------------------------------- 实例解析
class CVRPInstance:
    def __init__(self, name, dimension, capacity, coords, demands, max_vehicles=None):
        self.name = name
        self.dimension = dimension
        self.capacity = capacity
        self.coords = coords          # 1-based dict: id -> (x, y)
        self.demands = demands        # 1-based dict
        self.max_vehicles = max_vehicles
        n = dimension
        self.dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(0, n + 1):
            for j in range(0, n + 1):
                if i != j:
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    self.dist[i][j] = math.hypot(dx, dy)

    def edge_distance(self, a, b, rounded=False):
        d = math.hypot(self.coords[a][0] - self.coords[b][0],
                       self.coords[a][1] - self.coords[b][1])
        if rounded:
            return float(int(d + 0.5))
        return d

    @staticmethod
    def parse_vrp(text: str):
        lines = [l.split(":") for l in text.splitlines()]
        name = re.search(r"NAME\s*:\s*(\S+)", text, re.I)
        dim = re.search(r"DIMENSION\s*:\s*(\d+)", text, re.I)
        cap = re.search(r"CAPACITY\s*:\s*(\d+)", text, re.I)
        if not (name and dim and cap):
            raise ValueError("incomplete CVRPLIB header")
        section = None
        coord_rows, demand_rows, depot_rows = [], [], []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.endswith("_SECTION") or line == "EOF":
                section = line
                continue
            if section == "NODE_COORD_SECTION":
                parts = line.split()
                if len(parts) == 3:
                    coord_rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            elif section == "DEMAND_SECTION":
                parts = line.split()
                if len(parts) == 2:
                    demand_rows.append((int(parts[0]), float(parts[1])))
            elif section == "DEPOT_SECTION":
                parts = line.split()
                if len(parts) >= 1 and parts[0].isdigit():
                    depot_rows.append(int(parts[0]))
        # CVRPLIB 约定：节点 1 = 仓库、客户 2..n；内部统一为仓库 0、客户 1..(n-1)。
        depot_id = depot_rows[0] if depot_rows else 1
        raw_coords = {i: (x, y) for i, x, y in coord_rows}
        raw_demands = {i: d for i, d in demand_rows}
        if depot_id not in raw_coords:
            raise ValueError("depot node missing")
        n_customers = int(dim.group(1)) - 1
        coords = {0: raw_coords[depot_id]}
        demands = {0: raw_demands.get(depot_id, 0.0)}
        new_id = 1
        for i in sorted(raw_coords):
            if i == depot_id:
                continue
            if new_id > n_customers:
                break
            coords[new_id] = raw_coords[i]
            demands[new_id] = raw_demands.get(i, 0.0)
            new_id += 1
        # 车队口径（注册勘误，见 FINAL_REPORT_P4_P5.md §2）：CVRP X 的距离目标不把 k 作为
        # 硬约束（k 为官方 BKS 车队数；主流 ML-CVRP 评测允许更多车辆并报告 K），
        # VRPTW Solomon 的官方车队上限 25 仍作为硬约束。
        fleet = None
        return CVRPInstance(name.group(1), n_customers, float(cap.group(1)),
                            coords, demands, max_vehicles=fleet)


class VRPTWInstance:
    def __init__(self, name, dimension, capacity, coords, demands, windows, service,
                 depot_window, max_vehicles=None):
        self.name = name
        self.dimension = dimension
        self.capacity = capacity
        self.coords = coords
        self.demands = demands
        self.windows = windows      # id -> (l, r)
        self.service = service      # id -> s
        self.depot_window = depot_window
        self.max_vehicles = max_vehicles
        n = dimension
        self.dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(0, n + 1):
            for j in range(0, n + 1):
                if i != j:
                    self.dist[i][j] = math.hypot(coords[i][0] - coords[j][0],
                                                 coords[i][1] - coords[j][1])

    def edge_distance(self, a, b, rounded=False):
        return math.hypot(self.coords[a][0] - self.coords[b][0],
                          self.coords[a][1] - self.coords[b][1])

    @staticmethod
    def parse_solomon(path: Path):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in text.splitlines() if l.strip()]
        # 头部：名称 / VEHICLE / NUMBER CAPACITY / 25 200 / CUSTOMER / 表头 / 数据行
        name = re.search(r"^([A-Za-z0-9.\-]+)$", lines[0].strip())
        cap_line = lines[3].split() if len(lines) > 3 else []
        capacity = float(cap_line[1]) if len(cap_line) >= 2 else 0.0
        max_vehicles = int(cap_line[0]) if cap_line and cap_line[0].isdigit() else None
        coords, demands, windows, service = {}, {}, {}, {}
        for raw in text.splitlines():
            p = raw.split()
            if len(p) < 7:
                continue
            try:
                i = int(p[0])
            except ValueError:
                continue
            if i > 1000:
                continue
            coords[i] = (float(p[1]), float(p[2]))
            demands[i] = float(p[3])
            windows[i] = (float(p[4]), float(p[5]))
            service[i] = float(p[6])
        n = max((i for i in coords if i != 0), default=0)
        depot_window = windows.get(0, (0.0, 1e18))
        return VRPTWInstance(name.group(1) if name else Path(path).stem, n, capacity,
                             coords, demands, windows, service, depot_window, max_vehicles)


# ---------------------------------------------------------------- CVRP 评价
def evaluate_cvrp(inst, routes, rounded=False):
    """返回 (feasible, distance, loads, K)。容量与覆盖检查。"""
    seen = set()
    total = 0.0
    loads = []
    for r in routes:
        load = 0.0
        d = 0.0
        prev = 0
        for i in r:
            if i in seen or i < 1 or i > inst.dimension:
                return False, 0.0, loads, len(routes)
            seen.add(i)
            load += inst.demands[i]
            d += inst.edge_distance(prev, i, rounded)
            prev = i
        d += inst.edge_distance(prev, 0, rounded)
        loads.append(load)
        total += d
    if seen != set(range(1, inst.dimension + 1)):
        return False, total, loads, len(routes)
    if any(l > inst.capacity + EPS for l in loads):
        return False, total, loads, len(routes)
    if inst.max_vehicles is not None and len(routes) > inst.max_vehicles:
        return False, total, loads, len(routes)
    return True, total, loads, len(routes)


def split_cvrp(inst, sequence, rounded=False):
    """固定访问序列下总距离最小的车辆划分（O(n^2) DP）。"""
    n = len(sequence)
    inf = float("inf")
    cost = [[0.0] * (n + 1) for _ in range(n + 1)]
    load = [[0.0] * (n + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        s = 0.0
        for i in range(j, 0, -1):
            s += inst.demands[sequence[i - 1]]
            load[i][j] = s
            c = inst.edge_distance(0, sequence[i - 1], rounded)
            for k in range(i, j):
                c += inst.edge_distance(sequence[k - 1], sequence[k], rounded)
            c += inst.edge_distance(sequence[j - 1], 0, rounded)
            cost[i][j] = c
    f = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            if load[i][j] <= inst.capacity + EPS:
                if f[i - 1] + cost[i][j] < f[j] - 1e-12:
                    f[j] = f[i - 1] + cost[i][j]
                    prev[j] = i - 1
    if f[n] == inf:
        return None
    routes = []
    j = n
    while j > 0:
        i = prev[j] + 1
        routes.append(tuple(sequence[i - 1:j]))
        j = prev[j]
    routes.reverse()
    return dict(distance=f[n], routes=tuple(routes), vehicles=len(routes))


# ---------------------------------------------------------------- VRPTW 评价
def route_schedule(inst, route):
    """固定顺序的可行时刻表：返回 (feasible, start_times, arrival_list, wait_sum) 或 None。"""
    cap = 0.0
    for i in route:
        cap += inst.demands[i]
    if cap > inst.capacity + EPS:
        return None
    t = inst.depot_window[0]
    starts = []
    waits = 0.0
    for idx, i in enumerate(route):
        travel = inst.dist[0 if idx == 0 else route[idx - 1]][i]
        t += travel
        if t < inst.windows[i][0]:
            waits += inst.windows[i][0] - t
            t = inst.windows[i][0]
        if t > inst.windows[i][1] + EPS:
            return None
        starts.append(t)
        t += inst.service[i]
    t += inst.dist[route[-1]][0]
    if t > inst.depot_window[1] + EPS:
        return None
    return dict(starts=starts, waits=waits, end=t)


def route_cost_vrptw(inst, route):
    sched = route_schedule(inst, route)
    if sched is None:
        return None
    d = 0.0
    prev = 0
    for i in route:
        d += inst.dist[prev][i]
        prev = i
    d += inst.dist[prev][0]
    return dict(distance=d, wait=sched["waits"], end=sched["end"])


def evaluate_vrptw(inst, routes):
    """返回 (feasible, K, distance, wait, loads)。"""
    seen = set()
    total_d = 0.0
    total_w = 0.0
    loads = []
    for r in routes:
        for i in r:
            if i in seen or i < 1 or i > inst.dimension:
                return False, len(routes), 0.0, 0.0, []
            seen.add(i)
        sched = route_schedule(inst, tuple(r))
        if sched is None:
            return False, len(routes), 0.0, 0.0, []
        c = route_cost_vrptw(inst, tuple(r))
        total_d += c["distance"]
        total_w += c["wait"]
        loads.append(sum(inst.demands[i] for i in r))
    if seen != set(range(1, inst.dimension + 1)):
        return False, len(routes), total_d, total_w, loads
    if inst.max_vehicles is not None and len(routes) > inst.max_vehicles:
        return False, len(routes), total_d, total_w, loads
    return True, len(routes), total_d, total_w, loads


def better_lex(K1, D1, K2, D2):
    """字典序 (K, D) 比较：True 表示 (K1,D1) 更优。"""
    if K1 < K2:
        return True
    if K1 > K2:
        return False
    return D1 < D2 - EPS


def split_vrptw(inst, sequence):
    """固定访问序列下 (最小车辆数, 最小距离) 的可行连续分段（O(n^2) DP）。"""
    n = len(sequence)
    inf = float("inf")
    seg_cost = [[None] * (n + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            seg = tuple(sequence[i - 1:j])
            c = route_cost_vrptw(inst, seg)
            seg_cost[i][j] = c
    fk = [inf] * (n + 1)
    fd = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    fk[0], fd[0] = 0, 0.0
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            c = seg_cost[i][j]
            if c is None:
                continue
            k2 = fk[i - 1] + 1
            d2 = fd[i - 1] + c["distance"]
            if better_lex(k2, d2, fk[j], fd[j]):
                fk[j], fd[j], prev[j] = k2, d2, i - 1
    if fk[n] == inf:
        return None
    routes = []
    j = n
    while j > 0:
        i = prev[j] + 1
        routes.append(tuple(sequence[i - 1:j]))
        j = prev[j]
    routes.reverse()
    return dict(vehicles=fk[n], distance=fd[n], routes=tuple(routes))
