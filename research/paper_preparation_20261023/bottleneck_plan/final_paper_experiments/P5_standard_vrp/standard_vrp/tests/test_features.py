# -*- coding: utf-8 -*-
"""标准特征构造单测：形状、掩码与中性通道不变式。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance
from standard_vrp.core.standard_features import StandardFeatureBuilder, window_indices


def test_features():
    coords = {0: (0, 0), 1: (10, 0), 2: (10, 10), 3: (0, 10), 4: (20, 0), 5: (20, 20), 6: (5, 30), 7: (-8, 15)}
    demands = {0: 0, 1: 10, 2: 12, 3: 9, 4: 15, 5: 14, 6: 8, 7: 11}
    inst = CVRPInstance("f_cvrp", 7, 30, coords, demands)
    fb = StandardFeatureBuilder(inst, "cvrp")
    routes = ((1, 2, 3), (4, 5),)
    removed = {6, 7}
    batch = fb.build(routes, removed)
    assert batch["features"].shape == (1, 8, 16), batch["features"].shape
    assert batch["adjacency"].shape == (1, 8, 8)
    f = batch["features"][0]
    assert np.allclose(f[6, 10], 0.0) and np.allclose(f[7, 10], 0.0), "removed flag must be 0"
    assert np.allclose(f[1, 10], 1.0), "visible flag must be 1"
    assert np.allclose(f[:, 3], 0.0) and np.allclose(f[:, 4], 1.0), "CVRP neutral windows"
    assert np.allclose(f[:, 6], 0.266112) and np.allclose(f[:, 9], 0.0972), "frozen constants"
    assert np.allclose(f[6, 12], 1.0) and np.allclose(f[6, 13], 0.0), "no-decay neutral channels"
    # VRPTW 特征
    inst2 = VRPTWInstance("f_tw", 7, 30, coords,
                          {i: 10 for i in range(1, 8)},
                          {1: (0, 30), 2: (20, 60), 3: (0, 30), 4: (0, 60), 5: (30, 90), 6: (0, 100), 7: (0, 100)},
                          {i: 5 for i in range(1, 8)}, (0, 200), None)
    fb2 = StandardFeatureBuilder(inst2, "vrptw")
    b2 = fb2.build(((1, 2), (3, 4)), {5, 6, 7})
    f2 = b2["features"][0]
    assert np.allclose(f2[1, 3], 0.0) and np.allclose(f2[2, 3], 20 / 1440.0), "TW l channel"
    assert f2[1, 11] >= 0 and np.allclose(f2[:, 12], 1.0), "quality margin neutral"
    # 窗口规则
    idx = window_indices(inst, {6, 7}, window_cap=5)
    assert len(idx) <= 5 and 0 in idx and 6 in idx and 7 in idx, idx
    print("standard features tests passed")


if __name__ == "__main__":
    test_features()
    print("ALL FEATURE TESTS PASSED")
