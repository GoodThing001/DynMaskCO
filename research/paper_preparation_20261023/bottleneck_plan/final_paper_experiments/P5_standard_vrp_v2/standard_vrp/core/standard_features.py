# -*- coding: utf-8 -*-
"""V2 标准任务神经输入构造（严格对齐冻结 partial_batch 语义）。

权威来源：`C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/profit_core.py::partial_batch`
（feature_schema = static-profit-maskco-features-v0-16channels-partial-only）。

冻结语义（maskco_base 变体，quality_context=False）：
  ch0-1   (x-x0)/1440, (y-y0)/1440
  ch2     需求/20000（冻结分母；冷链 20 吨容量）
  ch3-5   l/1440, r/1440, service/1440（标准 CVRP 无约束 → 0、1、0）
  ch6     rate*1440 = 0.266112
  ch7     最低品质 0.8
  ch8     log1p(price*20000/1440) = 3.1842076959…
  ch9     cold = 0.0972
  ch10    float(i == 0 or i not in removed)  ← **仓库恒 1；可见客户 1；被遮 0**
  ch11-15 在 maskco_base 下**全零**（quality_context 路径仅在 quality_only/directed_only 变体启用）
  邻接    仅保留两端都可见的路线边（对称加倍，directed=False）；被遮节点行/列清零
  timestep= adjacency[1:].sum() / (n * 2)   ← 分母随 n 变化（V1 固定 200 为缺陷）

V1 偏离（P5-V2 审计 F1–F5）：
  F1 仓库 ch10=0（应为 1）
  F2 ch11–15 被赋动态值、ch12=1.0（maskco_base 应全零）
  F3 窗口构建器未给任何可见节点置 ch10=1（全部 0）
  F4 窗口构建器动态通道缺失/不一致
  F5 timestep 固定 /200（应 /(n*2)）
"""
from __future__ import annotations

import math

import numpy as np

HORIZON = 1440.0
NORM_DEMAND = 20000.0
FROZEN = dict(rate_times_horizon=0.266112, minimum=0.8, log_value=3.184207695922324,
              cold=0.0972)
# 训练分布代理（reserved_validation 27 件冷链 100 客输入）：坐标通道中位 = 0.0475，
# 需求通道中位 = 0.0500。STD_NORM 用它们对齐标准任务输入分布（登记于 16channel_table.md §2）。
TRAIN_COORD_MEDIAN = 0.0475
TRAIN_DEMAND_MEDIAN = 0.0500
SCHEMA = "static-profit-maskco-features-v0-16channels-partial-only"


def _schedule(inst, route):
    from standard_vrp.core.standard_evaluator import route_schedule
    return route_schedule(inst, tuple(route))


class StandardFeatureBuilder:
    """完整实例特征构造（n 任意），严格对齐冻结 partial_batch。

    mode:
      "v2_fix"   —— 冻结语义（ch10 仓库/可见=1、ch11-15 全零、timestep 随 n）
      "v1"       —— P5-V1 编码（动态通道计算 + ch12=1.0 + timestep/200），仅用于复现 V1
      "v2_norm"  —— 训练语义对齐（STD_NORM，Phase 4 定稿：需求按容量比例等）
    """

    def __init__(self, inst, objective, mode="v2_fix"):
        self.inst = inst
        self.objective = objective
        self.mode = mode
        self.n = inst.dimension
        self.depot = inst.coords[0]
        self.base = self._static_features()

    # ---- 通道 0–9（静态） -------------------------------------------------
    def _demand_norm(self, d):
        if self.mode == "v2_norm":
            return d / max(self.inst.capacity, 1e-9)
        return d / NORM_DEMAND

    def _coord_scale(self):
        if self.mode != "v2_norm":
            return HORIZON
        # 训练语义对齐：把实例坐标通道的分布映射到训练分布（中位 = TRAIN_COORD_MEDIAN）。
        offs = []
        x0, y0 = self.depot
        for i in range(1, self.n + 1):
            x, y = self.inst.coords[i]
            offs.append(abs(x - x0))
            offs.append(abs(y - y0))
        if not offs:
            return HORIZON
        offs.sort()
        med = offs[len(offs) // 2]
        if med <= 0:
            return HORIZON
        return med / TRAIN_COORD_MEDIAN

    def _static_features(self):
        n = self.n
        base = np.zeros((n + 1, 16), np.float32)
        scale = self._coord_scale()
        for i in range(0, n + 1):
            x, y = self.inst.coords[i]
            base[i, 0] = (x - self.depot[0]) / scale
            base[i, 1] = (y - self.depot[1]) / scale
            base[i, 2] = self._demand_norm(self.inst.demands.get(i, 0.0))
            if self.objective == "vrptw":
                l, r = self.inst.windows.get(i, (0.0, HORIZON))
                base[i, 3] = l / HORIZON
                base[i, 4] = r / HORIZON
                base[i, 5] = self.inst.service.get(i, 0.0) / HORIZON
            else:
                base[i, 3] = 0.0
                base[i, 4] = 1.0
                base[i, 5] = 0.0
            base[i, 6] = FROZEN["rate_times_horizon"]
            base[i, 7] = FROZEN["minimum"]
            base[i, 8] = FROZEN["log_value"]
            base[i, 9] = FROZEN["cold"]
        if self.mode == "v1":
            for i in range(0, n + 1):
                base[i, 12] = 1.0
        return base

    # ---- 邻接（仅两端可见） ----------------------------------------------
    def _adjacency(self, routes, removed, n_nodes):
        adj = np.zeros((n_nodes, n_nodes), np.float32)
        for r in routes:
            seq = (0,) + tuple(r) + (0,)
            for a, b in zip(seq[:-1], seq[1:]):
                if a in removed or b in removed:
                    continue  # 遮蔽邻接不得泄漏
                adj[a, b] += 1
                adj[b, a] += 1
        return adj

    def _timestep(self, adjacency):
        # 冻结：adjacency[1:].sum()/(n*(1 if directed else 2))
        return float(adjacency[1:].sum()) / (self.n * 2)

    def build(self, routes, removed):
        """routes：部分解（不含被遮客）；removed：被遮集合。返回 batch dict。"""
        removed = set(removed)
        n = self.n
        features = self.base.copy()
        for i in range(0, n + 1):
            features[i, 10] = float(i == 0 or i not in removed)  # 仓库恒 1
        if self.mode == "v1":
            self._dynamic_v1(features, routes, removed)
        adjacency = self._adjacency(routes, removed, n + 1)
        return dict(features=features[None], adjacency=adjacency[None],
                    timestep=np.array([self._timestep(adjacency)], np.float32))

    def _dynamic_v1(self, features, routes, removed):
        """P5-V1 的动态通道（仅用于复现 V1；训练语义为全零）。"""
        if self.objective != "vrptw":
            for r in routes:
                u = sum(self.inst.demands.get(i, 0.0) for i in r) / NORM_DEMAND
                for i in r:
                    features[i, 15] = u
            return
        for r in routes:
            sched = _schedule(self.inst, r)
            if sched is None:
                continue
            starts = sched["starts"]
            for idx, i in enumerate(r):
                completion = starts[idx] + self.inst.service.get(i, 0.0)
                features[i, 11] = (completion - self.inst.depot_window[0]) / HORIZON
                features[i, 14] = (self.inst.windows[i][1] - starts[idx]) / HORIZON
            features[0, 11] = sched["end"] / HORIZON
            u = sum(self.inst.demands.get(i, 0.0) for i in r) / NORM_DEMAND
            for i in r:
                features[i, 15] = u


def window_indices(inst, removed, window_cap=101):
    """窗口规则（P5_预登记 §4）：仓库 + 被遮客 + 最近未遮客补足至 window_cap。"""
    n = inst.dimension
    removed = set(removed)
    others = [i for i in range(1, n + 1) if i not in removed]
    if len(others) + len(removed) + 1 <= window_cap:
        return list(range(0, n + 1))
    others.sort(key=lambda i: min(inst.dist[i][j] for j in removed))
    keep = sorted(removed) + others[:window_cap - 1 - len(removed)]
    return [0] + keep


class WindowedFeatureBuilder(StandardFeatureBuilder):
    """n>100 窗口特征：与完整构建器通道语义完全一致，仅按窗口索引提取。

    修复 V1 F3/F4/F5：可见节点/仓库置 ch10=1；动态通道按 mode 一致处理；
    timestep 使用窗口邻接的同一公式（分母为窗口节点数×2？——不，冻结公式的分母是
    实例客户数 n，窗口只是取子集；此处两种口径都记录，最终以一致性测试锁定）。
    """

    def __init__(self, inst, objective, mode="v2_fix", window_cap=101):
        super().__init__(inst, objective, mode)
        self.window_cap = window_cap

    def build_windowed(self, routes, removed):
        """返回 (window_ids, batch, pos_of)。batch 的特征/邻接与完整构建器对应子阵一致。"""
        removed = set(removed)
        window = window_indices(self.inst, removed, self.window_cap)
        pos_of = {i: p for p, i in enumerate(window)}
        w = len(window)
        features = np.zeros((w, 16), np.float32)
        for p, i in enumerate(window):
            features[p] = self.base[i]
            features[p, 10] = float(i == 0 or i not in removed)  # 修复 F3
        if self.mode == "v1":
            # 动态通道：基于完整实例调度计算，再按窗口提取（修复 F4 的一致性）
            full = self.base.copy()
            self._dynamic_v1(full, routes, removed)
            for p, i in enumerate(window):
                features[p, 11:16] = full[i, 11:16]
        adjacency = np.zeros((w, w), np.float32)
        for r in routes:
            seq = (0,) + tuple(r) + (0,)
            for a, b in zip(seq[:-1], seq[1:]):
                if a in removed or b in removed:
                    continue
                pa, pb = pos_of.get(a), pos_of.get(b)
                if pa is not None and pb is not None:
                    adjacency[pa, pb] += 1
                    adjacency[pb, pa] += 1
        ts = float(adjacency[1:].sum()) / (self.n * 2)  # 冻结分母为实例客户数
        return window, dict(features=features[None], adjacency=adjacency[None],
                            timestep=np.array([ts], np.float32)), pos_of
