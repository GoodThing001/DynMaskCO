# -*- coding: utf-8 -*-
"""V2 距离条件残差校准：逐项对齐锁版 calibration.py::residual_edges。

锁版公式（核实自 reserved_validation/solver/calibration.py）：
    spatial  = -d / scale          （scale = 各行第 8 小距离的中位数）
    rank_D   = 1 + argsort(argsort(-d))        （1 基排名，stable）
    rank_H   = 1 + argsort(argsort(-heat))     （1 基排名，stable）
    residual = clip(log(rank_D / rank_H), -3, 3)
    heat_cal = spatial + alpha * residual     （alpha 冻结于 calibrated.json）
对角线置 -inf（排除自身节点）；`-inf`/NaN 热图分数按"缺失"处理，不参与排名，
由调用方回退到距离规则（不得把缺失解释为模型拒绝该边）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
# 解锁版 alpha 的三条来源（按优先级）：包内副本 → 环境变量 → 仓库只读路径。
# 包内副本是锁版 calibrated.json 的逐字节拷贝（SHA 见 manifests/frozen_identity.json），
# 保证远端部署自足；任何来源都读不到时**直接报错**，不得静默回退（D8 教训）。
PACKAGED = Path(__file__).resolve().parent / "calibrated.json"
REPO_LOCKED = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023"
                   r"\bottleneck_plan\maskco_perishable_external_20261119"
                   r"\reserved_validation\solver\calibrated.json")
ENV_KEY = "MASKCO_P5V2_CALIBRATED"
LOCKED_CALIBRATED = PACKAGED


def calibrated_path():
    import os
    for cand in (PACKAGED, Path(os.environ[ENV_KEY]) if os.environ.get(ENV_KEY) else None,
                 REPO_LOCKED):
        if cand is not None and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        "locked calibrated.json not found (packaged/env/repo); 校准热图不得静默缺失")


def locked_alpha():
    """读取锁版冻结 alpha（缺失即报错）。"""
    data = json.loads(calibrated_path().read_text(encoding="utf-8"))
    return float(data["alpha"])


def residual_edges(distance, heat):
    """返回 (spatial, residual)；形状与 distance 一致（含仓库行/列）。"""
    d = np.asarray(distance, dtype=float)
    learned = np.array(heat, dtype=float, copy=True)
    spatial = -d.copy()
    np.fill_diagonal(learned, -np.inf)
    np.fill_diagonal(spatial, -np.inf)

    def ranks(weights):
        # 缺失（-inf/NaN）排到最后：先置为极小有限值参与稳定排序，再单独屏蔽
        w = np.array(weights, dtype=float, copy=True)
        missing = ~np.isfinite(w)
        w[missing] = -np.inf
        order = np.argsort(-w, axis=1, kind="stable")
        r = 1 + np.argsort(order, axis=1, kind="stable")
        r = r.astype(float)
        r[missing] = np.inf  # 缺失边的排名视为无穷大（永不被选中）
        return r

    rank_d = ranks(spatial)
    rank_h = ranks(learned)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.isfinite(rank_h), np.log(rank_d / rank_h), 0.0)
    delta = np.clip(delta, -3.0, 3.0)
    scale = float(np.median(np.partition(d, 8, axis=1)[:, 8])) if d.shape[0] > 9 else float(np.median(d[d > 0]))
    if not scale > 0:
        scale = float(np.median(d[d > 0]))
    return np.ascontiguousarray(-d / scale), np.ascontiguousarray(delta)


def calibrated_heat(distance, heat, alpha=None):
    """校准热图：spatial + alpha*residual（alpha 缺省取锁版冻结值）。"""
    a = locked_alpha() if alpha is None else float(alpha)
    spatial, residual = residual_edges(distance, heat)
    return np.ascontiguousarray(spatial + a * residual)
