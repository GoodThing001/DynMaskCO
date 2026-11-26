# -*- coding: utf-8 -*-
"""V2 目标适配器：把锁版 ORDER_SPLIT 的经济目标替换为标准任务目标。

CVRP：最小化总距离（EUC_2D 逐边取整口径）。
VRPTW：字典序 (K, D) —— 先最小化车辆数，再最小化总距离。

统一 API（供 masked_repair / 池搜索使用）：
  key(new_route, delta) -> 可比较的候选键
  is_better(a, b)       -> a 严格优于 b
  regret(best, second)  -> 后悔值（越大越应优先插入）
  insert_delta / detour / feasible_insert / fits / route_ok / route_proxy
  evaluate(routes) / score(routes)

字典序标量化：VRPTW 后悔值 = K 差×BIG + 距离差，BIG=1e6 使任何 K 差压倒距离差，
与 (K,D) 字典序严格一致（不是车辆惩罚系数，不引入可调超参数）。
"""
from __future__ import annotations

import math

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw,
                                                  route_schedule)

BIG = 1e6  # 字典序标量化权重：K 差 > 任何可观测距离差


class CvrpAdapter:
    """标准 CVRP：总距离最小化（逐边整数舍入）。"""

    objective = "cvrp"
    lexicographic = False

    def __init__(self, inst: CVRPInstance):
        self.inst = inst

    # ---- 完整解 -----------------------------------------------------------
    def evaluate(self, routes):
        ok, d, loads, k = evaluate_cvrp(self.inst, routes, rounded=True)
        return ok, d, k

    def score(self, routes):
        ok, d, _, _ = evaluate_cvrp(self.inst, routes, rounded=True)
        return (-d) if ok else -math.inf

    # ---- 路线局部 ---------------------------------------------------------
    def route_ok(self, route):
        if not route:
            return False
        return (sum(self.inst.demands[i] for i in route)
                <= self.inst.capacity + 1e-9)

    def route_proxy(self, route):
        if not route:
            return 0.0
        if not self.route_ok(route):
            return -math.inf
        d = self.inst.edge_distance(0, route[0], rounded=True)
        for a, b in zip(route, route[1:]):
            d += self.inst.edge_distance(a, b, rounded=True)
        d += self.inst.edge_distance(route[-1], 0, rounded=True)
        return d

    def insert_delta(self, route, pos, cust):
        seq = [0] + list(route) + [0]
        a, b = seq[pos], seq[pos + 1]
        return (self.inst.edge_distance(a, cust, rounded=True)
                + self.inst.edge_distance(cust, b, rounded=True)
                - self.inst.edge_distance(a, b, rounded=True))

    def detour(self, route, pos, cust):
        return self.insert_delta(route, pos, cust)

    def fits(self, route, cust):
        return (sum(self.inst.demands[i] for i in route) + self.inst.demands[cust]
                <= self.inst.capacity + 1e-9)

    def feasible_insert(self, route, pos, cust):
        return self.fits(route, cust)

    def singleton_delta(self, cust):
        return 2.0 * self.inst.edge_distance(0, cust, rounded=True)

    # ---- 键与比较 ---------------------------------------------------------
    def key(self, new_route, delta):
        return -delta

    def is_better(self, a, b):
        return a > b + 1e-9

    def regret(self, best, second):
        if second is None or not math.isfinite(second):
            return BIG
        return best - second  # -(delta_best) - (-delta_second) = delta_second - delta_best


class VrptwAdapter:
    """标准 VRPTW：字典序 (K, D)。"""

    objective = "vrptw"
    lexicographic = True

    def __init__(self, inst: VRPTWInstance):
        self.inst = inst

    def evaluate(self, routes):
        ok, k, d, w, loads = evaluate_vrptw(self.inst, routes)
        return ok, d, k

    def score(self, routes):
        ok, k, d, _, _ = evaluate_vrptw(self.inst, routes)
        return -(k * BIG + d) if ok else -math.inf

    def route_ok(self, route):
        return bool(route) and route_schedule(self.inst, tuple(route)) is not None

    def route_proxy(self, route):
        if not route:
            return 0.0
        if route_schedule(self.inst, tuple(route)) is None:
            return -math.inf
        d = 0.0
        prev = 0
        for i in route:
            d += self.inst.dist[prev][i]
            prev = i
        return d + self.inst.dist[prev][0]

    def insert_delta(self, route, pos, cust):
        seq = [0] + list(route) + [0]
        a, b = seq[pos], seq[pos + 1]
        return (self.inst.edge_distance(a, cust) + self.inst.edge_distance(cust, b)
                - self.inst.edge_distance(a, b))

    def detour(self, route, pos, cust):
        return self.insert_delta(route, pos, cust)

    def fits(self, route, cust):
        return (sum(self.inst.demands[i] for i in route) + self.inst.demands[cust]
                <= self.inst.capacity + 1e-9)

    def feasible_insert(self, route, pos, cust):
        if not self.fits(route, cust):
            return False
        new_route = list(route[:pos]) + [cust] + list(route[pos:])
        return route_schedule(self.inst, tuple(new_route)) is not None

    def singleton_delta(self, cust):
        return 2.0 * self.inst.edge_distance(0, cust)

    # ---- 键与比较（字典序：K 优先，再距离） -------------------------------
    def key(self, new_route, delta):
        return (1 if new_route else 0, delta)

    def is_better(self, a, b):
        if a[0] != b[0]:
            return a[0] < b[0]
        return a[1] < b[1] - 1e-9

    def regret(self, best, second):
        if second is None or not math.isfinite(second[1]):
            return BIG
        return (second[0] - best[0]) * BIG + (second[1] - best[1])


def adapter_for(inst):
    return CvrpAdapter(inst) if isinstance(inst, CVRPInstance) else VrptwAdapter(inst)
