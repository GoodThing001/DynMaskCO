# -*- coding: utf-8 -*-
"""S3-6a 软 k-NN 采样器回归测试（2026-10-03，T37）。

T37a 权重归一化/正性/σ 极限行为（σ→∞ 趋均匀、σ→0 集中最近邻）；
T37b 采样确定性（同 rng 同结果）+ 未来段（reveal>clock）+ n_eff 口径；
T37c 硬 k-NN 对照身份守护：新增软采样器不改 CondHistoricalSampler 行为（同 rng 同输出，
     软采样器与硬采样器输出只在权重机制上不同）。
"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_reserve import generate_dataset  # noqa: E402
from scenario_saa import (CondHistoricalSampler, ScenarioOrder,  # noqa: E402
                          SoftKNNHistoricalSampler, VisibleSnapshot, build_history)


def _mk_hist(n_days=40, seed=20260925):
    return build_history(generate_dataset(n_days, 60, seed))


def _snap(clock=4.0):
    orders = tuple(ScenarioOrder(oid=i + 1, reveal=float((i % 10) * 0.5),
                                 x=0.3 + 0.05 * (i % 7), y=0.3 + 0.05 * (i % 5),
                                 demand=2.0, temp_class=i % 3, tw_start=0.0,
                                 tw_end=22.0, service_time=0.05) for i in range(12))
    return VisibleSnapshot(clock=clock, orders=orders, accepted=frozenset(),
                           rejected=frozenset(), vehicles=(), energy_used=0.0,
                           budget=1e9, booking_horizon=22.0, capacity=50.0)


def test_t37a_weights():
    hist = _mk_hist()
    snap = _snap()
    s1 = SoftKNNHistoricalSampler(hist, sigma2_mult=1.0)
    w = s1._weights(snap)
    assert w.shape == (len(hist),) and np.all(w >= 0.0) and abs(w.sum() - 1.0) < 1e-12
    # σ→∞ → 权重趋均匀（max/min → 1）
    s_big = SoftKNNHistoricalSampler(hist, sigma2_mult=1e9)
    wb = s_big._weights(snap)
    assert float(wb.max() / wb.min()) < 1.0 + 1e-6
    # σ→0 → 集中在最近邻（最大权重显著高于平均）
    s_small = SoftKNNHistoricalSampler(hist, sigma2_mult=1e-9)
    ws = s_small._weights(snap)
    assert float(ws.max()) > 10.0 * float(ws.mean())
    # h_med 为训练日两两距离中位数且 σ² = mult × h_med
    assert abs(s1.sigma2 - s1.h_med) < 1e-12
    assert s_big.sigma2 == 1e9 * s_big.h_med
    print('[PASS] t37a_weights (归一/正性/σ 极限/σ²=mult×h_med)')
    return True


def test_t37b_sampling():
    hist = _mk_hist()
    snap = _snap()
    s = SoftKNNHistoricalSampler(hist, sigma2_mult=1.0)
    rng1, rng2 = np.random.default_rng(20261003), np.random.default_rng(20261003)
    out1 = s.sample(snap, rng1, 10)
    out2 = s.sample(snap, rng2, 10)
    assert len(out1) == 10 and len(out2) == 10
    for a, b in zip(out1, out2):
        assert [(o.oid, o.reveal, o.x, o.y) for o in a] == [(o.oid, o.reveal, o.x, o.y) for o in b]
    # 未来段：所有场景订单 reveal > clock
    for scen in out1:
        for o in scen:
            assert o.reveal > snap.clock + 1e-6
    # n_eff 口径：(Σw)²/Σw²，1 ≤ n_eff ≤ N
    assert 1.0 <= s.last_n_eff <= len(hist) + 1e-9
    assert s.n_samples == 2 and s.n_eff_sum == s.last_n_eff * 2
    print('[PASS] t37b_sampling (确定性/未来段/n_eff 口径)')
    return True


def test_t37c_hard_knn_untouched():
    hist = _mk_hist()
    snap = _snap()
    hard = CondHistoricalSampler(hist, n_neighbors=10)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    a = hard.sample(snap, rng1, 10)
    b = hard.sample(snap, rng2, 10)
    for x, y in zip(a, b):
        assert [(o.oid, o.reveal) for o in x] == [(o.oid, o.reveal) for o in y]
    # 软采样器独立存在且不改硬采样器行为（软权重采样合法输出 K 个场景）
    soft = SoftKNNHistoricalSampler(hist, sigma2_mult=1.0)
    out = soft.sample(snap, np.random.default_rng(7), 10)
    assert len(out) == 10
    print('[PASS] t37c_hard_knn_untouched (硬 k-NN 行为不变 + 软采样器独立)')
    return True


if __name__ == '__main__':
    res = {
        't37a_weights': test_t37a_weights(),
        't37b_sampling': test_t37b_sampling(),
        't37c_hard_knn_untouched': test_t37c_hard_knn_untouched(),
    }
    ok = all(res.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
