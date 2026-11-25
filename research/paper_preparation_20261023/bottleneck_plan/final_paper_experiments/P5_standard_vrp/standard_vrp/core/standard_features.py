# -*- coding: utf-8 -*-
"""标准任务神经输入构造（冻结 16 通道 schema 的中性编码，见 P5_预登记 §4）。

通道约定（冻结 partial_batch 同构）：
 0–1  (x−x0)/1440, (y−y0)/1440
 2    需求/20000
 3–5  l/1440, r/1440, service/1440（CVRP 无约束中性值 0, 1, 0）
 6    rate*1440 = 0.266112（冻结冷链常数，所有冷链实例恒同）
 7    最低品质 0.8
 8    log1p(price*20000/1440) = 3.1842077（冻结常数）
 9    cold = 0.0972（冻结常数）
 10   掩码可见标志（1=未遮）
 11   (完成龄)/1440（标准调度，rate=0 → 无品质衰减）
 12   (品质−0.8)/0.2 = 1.0（无衰减时品质恒为 1）
 13   log1p(price*需求*0/(1+cold)) = 0
 14   (r−服务开始)/1440 松弛
 15   路线装载率 = 路线需求合计/20000
"""
from __future__ import annotations

import math

import numpy as np

HORIZON = 1440.0
NORM_DEMAND = 20000.0
FROZEN = dict(rate_times_horizon=0.266112, minimum=0.8, log_value=3.184207695922324,
              cold=0.0972)


class StandardFeatureBuilder:
    """为 (instance, routes, removed) 构造 [1, n+1, 16] 特征与 [1, n+1, n+1] 邻接。"""

    def __init__(self, inst, objective):
        self.inst = inst
        self.objective = objective
        self.n = inst.dimension
        self.depot = inst.coords[0]
        # 实例级不变特征（0–9 通道）
        base = np.zeros((self.n + 1, 16), np.float32)
        for i in range(0, self.n + 1):
            x, y = inst.coords[i]
            base[i, 0] = (x - self.depot[0]) / HORIZON
            base[i, 1] = (y - self.depot[1]) / HORIZON
            base[i, 2] = inst.demands.get(i, 0.0) / NORM_DEMAND
            if objective == "vrptw":
                l, r = inst.windows.get(i, (0.0, HORIZON))
                base[i, 3] = l / HORIZON
                base[i, 4] = r / HORIZON
                base[i, 5] = inst.service.get(i, 0.0) / HORIZON
            else:
                base[i, 3] = 0.0
                base[i, 4] = 1.0
                base[i, 5] = 0.0
            base[i, 6] = FROZEN["rate_times_horizon"]
            base[i, 7] = FROZEN["minimum"]
            base[i, 8] = FROZEN["log_value"]
            base[i, 9] = FROZEN["cold"]
            base[i, 12] = 1.0   # 无衰减：品质余量 (1−0.8)/0.2 = 1
            base[i, 13] = 0.0   # 无货值项：rate=0
        self.base = base

    def build(self, routes, removed):
        """routes: 当前部分解（不含被遮客）；removed: 被遮客户集合。"""
        removed = set(removed)
        n = self.n
        features = self.base.copy()
        adjacency = np.zeros((n + 1, n + 1), np.float32)
        # 部分路线调度：标准时刻（rate=0 无衰减）
        if self.objective == "vrptw":
            from standard_vrp.core.standard_evaluator import route_schedule
            for r in routes:
                sched = route_schedule(self.inst, tuple(r))
                if sched is None:
                    continue
                starts = sched["starts"]
                for idx, i in enumerate(r):
                    completion = starts[idx] + self.inst.service.get(i, 0.0)
                    features[i, 11] = (completion - self.inst.depot_window[0]) / HORIZON
                    features[i, 14] = (self.inst.windows[i][1] - starts[idx]) / HORIZON
                features[0, 11] = sched["end"] / HORIZON
        # 邻接：路线边（未遮边）
        for r in routes:
            seq = (0,) + tuple(r) + (0,)
            for a, b in zip(seq[:-1], seq[1:]):
                adjacency[a, b] += 1
                adjacency[b, a] += 1
        # 装载率（通道 15）
        for r in routes:
            u = sum(self.inst.demands.get(i, 0.0) for i in r) / NORM_DEMAND
            for i in r:
                features[i, 15] = u
        # 遮客：通道 10=0、邻接清零
        for i in removed:
            features[i, 10] = 0.0
        for i in range(1, n + 1):
            if i not in removed:
                features[i, 10] = 1.0
        adjacency[list(removed), :] = 0.0
        adjacency[:, list(removed)] = 0.0
        timestep = np.array([adjacency[1:].sum() / 200], np.float32)
        return dict(features=features[None], adjacency=adjacency[None], timestep=timestep)


def window_indices(inst, removed, window_cap=101):
    """n>100 窗口规则（P5_预登记 §4）：仓库 + 被遮客 + 60 个最近未遮客，返回 0-based 节点表。"""
    n = inst.dimension
    removed = set(removed)
    others = [i for i in range(1, n + 1) if i not in removed]
    if len(others) + len(removed) + 1 <= window_cap:
        return list(range(0, n + 1))
    others.sort(key=lambda i: min(inst.dist[i][j] for j in removed))
    keep = sorted(removed) + others[:window_cap - 1 - len(removed)]
    return [0] + keep


class WindowedFeatureBuilder(StandardFeatureBuilder):
    """n>100 窗口特征：在窗口内构造冻结输入，热图按窗口索引映射回原实例。"""

    def __init__(self, inst, objective, window_cap=101):
        super().__init__(inst, objective)
        self.window_cap = window_cap

    def build_windowed(self, routes, removed):
        """返回 (window_ids, batch, pos_of) —— window_ids 为原实例节点 id 表（含仓库0）。"""
        removed = set(removed)
        window = window_indices(self.inst, removed, self.window_cap)
        pos_of = {i: p for p, i in enumerate(window)}
        w = len(window)
        features = np.zeros((w, 16), np.float32)
        adjacency = np.zeros((w, w), np.float32)
        for p, i in enumerate(window):
            features[p] = self.base[i]
        # 部分路线（仅窗口内邻接）
        for r in routes:
            seq = [0] + list(r) + [0]
            for a, b in zip(seq[:-1], seq[1:]):
                pa, pb = pos_of.get(a), pos_of.get(b)
                if pa is not None and pb is not None:
                    adjacency[pa, pb] += 1
                    adjacency[pb, pa] += 1
        for i in removed:
            p = pos_of[i]
            features[p, 10] = 0.0
        adjacency[list(pos_of[i] for i in removed), :] = 0.0
        adjacency[:, list(pos_of[i] for i in removed)] = 0.0
        timestep = np.array([adjacency[1:].sum() / 200], np.float32)
        return window, dict(features=features[None], adjacency=adjacency[None], timestep=timestep), pos_of
