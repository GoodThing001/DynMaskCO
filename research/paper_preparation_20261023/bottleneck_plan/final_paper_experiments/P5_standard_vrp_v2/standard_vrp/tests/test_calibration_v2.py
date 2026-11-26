# -*- coding: utf-8 -*-
"""V2 校准模块单元测试（对齐锁版 calibration.py 语义）。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from standard_vrp.core.calibration import residual_edges, calibrated_heat, locked_alpha


def test_identity_zero_residual():
    """神经排名与距离排名一致时残差为零。"""
    n = 6
    d = np.array([[abs(i - j) for j in range(n)] for i in range(n)], dtype=float)
    s, r = residual_edges(d, -d.copy())
    off = ~np.eye(n, dtype=bool)
    assert np.abs(r[off]).max() < 1e-12, np.abs(r[off]).max()
    print("校准 PASS：排名一致 → 残差 0")


def test_promote_and_demote():
    """热图排名优于距离 → 正补偿；劣于距离 → 负补偿；截断 ±3。"""
    n = 6
    d = np.array([[abs(i - j) for j in range(n)] for i in range(n)], dtype=float)
    h = -d.copy()
    # 距离排名第 1 的边 (0,1) 在热图中被压到最差 → rank_H 很大 → 负残差
    h[0, 1] = -1e6
    s, r = residual_edges(d, h)
    assert r[0, 1] < 0, r[0, 1]
    # 距离排名第 5 的边 (0,5) 在热图中被提到第一 → 正残差
    h2 = -d.copy()
    h2[0, 5] = 1e6
    s2, r2 = residual_edges(d, h2)
    assert r2[0, 5] > 0, r2[0, 5]
    # 截断
    h3 = -d.copy()
    h3[0, 1] = -1e12
    _, r3 = residual_edges(d, h3)
    assert -3.0 - 1e-12 <= r3[0, 1] <= 0, r3[0, 1]
    print("校准 PASS：正向/负向补偿与 ±3 截断正确")


def test_missing_heat_and_no_nan():
    """缺失热图（-inf）不得产生 NaN；校准结果有限。"""
    n = 5
    d = np.array([[abs(i - j) for j in range(n)] for i in range(n)], dtype=float)
    h = np.full((n, n), -np.inf)
    s, r = residual_edges(d, h)
    assert np.isfinite(r[np.isfinite(r)]).all()
    assert not np.isnan(r).any(), "缺失热图不得产生 NaN"
    ch = calibrated_heat(d, h)
    assert np.isfinite(ch).all(), "校准热图必须全有限"
    print("校准 PASS：缺失热图无 NaN，校准热图有限")


def test_locked_alpha():
    a = locked_alpha()
    assert abs(a - 0.25) < 1e-12, a
    print("校准 PASS：锁版冻结 alpha =", a)


def test_calibrated_differs_from_raw_heat():
    """校准热图与原始热图必须存在可验证差异（D4 修复证据）。"""
    n = 8
    d = np.array([[abs(i - j) for j in range(n)] for i in range(n)], dtype=float)
    h = -d.copy()
    h[0, 7] = 1e6  # 明显偏离距离排序
    ch = calibrated_heat(d, h)
    diff = np.abs(ch - h)
    assert diff.max() > 0.5, diff.max()
    print("校准 PASS：校准热图与原始热图最大差异 %.3f" % diff.max())


def main():
    test_identity_zero_residual()
    test_promote_and_demote()
    test_missing_heat_and_no_nan()
    test_locked_alpha()
    test_calibrated_differs_from_raw_heat()
    print("ALL P5-V2 CALIBRATION TESTS PASSED")


if __name__ == "__main__":
    main()
