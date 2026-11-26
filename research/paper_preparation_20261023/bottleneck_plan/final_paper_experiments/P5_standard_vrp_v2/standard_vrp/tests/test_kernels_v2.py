# -*- coding: utf-8 -*-
"""P5-V2 内核等价性与时间语义测试（枚举/暴力对照）。

覆盖审计遗留项：
1. 固定序列 Split 与暴力划分对照（CVRP 距离目标、VRPTW 字典序 (K,D)）；
2. 跨车 bundle 选取范围（2-4 条相邻路线、≤24 客户）；
3. 在线预算与轮内截止（含 bugfix 变体）；
4. working/best 解管理：返回解可复算且不劣于初始见证，reset 语义可观测。
"""
from __future__ import annotations

import itertools
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from standard_vrp.adapters.objective import adapter_for
from standard_vrp.core.standard_evaluator import (
    CVRPInstance, VRPTWInstance, evaluate_cvrp, evaluate_vrptw, split_cvrp, split_vrptw)
from standard_vrp.solver.pool_search import bundle, standard_search


def _cvrp(n, cap, seed):
    rng = random.Random(seed)
    coords = {0: (50.0, 50.0)}
    dem = {0: 0.0}
    for i in range(1, n + 1):
        coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100))
        dem[i] = float(rng.randint(1, 9))
    return CVRPInstance("t%d" % n, n, cap, coords, dem)


def _vrptw(n, cap, seed):
    rng = random.Random(seed)
    coords = {0: (50.0, 50.0)}
    dem = {0: 0.0}
    win = {0: (0.0, 1000.0)}
    svc = {0: 0.0}
    for i in range(1, n + 1):
        coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100))
        dem[i] = float(rng.randint(1, 5))
        lo = rng.uniform(0, 200)
        win[i] = (lo, lo + 120.0)
        svc[i] = 5.0
    return VRPTWInstance("tv%d" % n, n, cap, coords, dem, win, svc, (0.0, 1000.0))


def _brute_cvrp(inst, seq):
    """枚举所有连续划分，取最小总距离（不可行划分记 inf）。"""
    n = len(seq)
    dist, dem, cap = inst.dist, inst.demands, inst.capacity
    best = float("inf")
    for cuts in itertools.product([0, 1], repeat=n - 1):
        total, load, prev, ok = 0.0, 0.0, 0, True
        for k, c in enumerate(seq):
            load += dem[c]
            if load > cap + 1e-9:
                ok = False
                break
            total += dist[prev][c]
            prev = c
            if k < n - 1 and cuts[k]:
                total += dist[prev][0]
                prev, load = 0, 0.0
        if ok:
            total += dist[prev][0]
            best = min(best, total)
    return best


def test_split_cvrp_matches_bruteforce():
    for seed in (1, 2, 3):
        inst = _cvrp(8, 20.0, seed)
        seq = list(range(1, 9))
        random.Random(seed).shuffle(seq)
        out = split_cvrp(inst, seq)
        want = _brute_cvrp(inst, seq)
        assert out is not None
        assert abs(out["distance"] - want) <= 1e-6 * max(1.0, want), (out, want)
        # 划分可复算：routes 拼接后与序列一致，且评价距离等于 DP 值
        assert [c for r in out["routes"] for c in r] == seq
        ok, val, loads, k = evaluate_cvrp(inst, [tuple(r) for r in out["routes"]])
        assert ok and abs(val - out["distance"]) <= 1e-6 * max(1.0, val)
        assert k == out["vehicles"]
        assert all(l <= inst.capacity + 1e-9 for l in loads)


def _brute_vrptw(inst, seq):
    n = len(seq)
    best = None
    for cuts in itertools.product([0, 1], repeat=n - 1):
        routes, cur = [], []
        for k, c in enumerate(seq):
            cur.append(c)
            if k < n - 1 and cuts[k]:
                routes.append(cur)
                cur = []
        routes.append(cur)
        K, D = len(routes), 0.0
        ok = True
        for r in routes:
            load = sum(inst.demands[c] for c in r)
            if load > inst.capacity + 1e-9:
                ok = False
                break
            cost = _route_cost(inst, r)
            if cost is None:
                ok = False
                break
            D += cost
        if not ok:
            continue
        if best is None or (K, D) < best:
            best = (K, D)
    return best


def _route_cost(inst, route):
    t, prev, total = inst.depot_window[0], 0, 0.0
    for c in route:
        d = inst.dist[prev][c]
        total += d
        t = max(t + d, inst.windows[c][0])
        if t > inst.windows[c][1] + 1e-9:
            return None
        t += inst.service[c]
        prev = c
    d = inst.dist[prev][0]
    total += d
    if t + d > inst.depot_window[1] + 1e-6:
        return None
    return total


def test_split_vrptw_matches_bruteforce_lex():
    for seed in (4, 5):
        inst = _vrptw(7, 14.0, seed)
        seq = list(range(1, 8))
        random.Random(seed).shuffle(seq)
        out = split_vrptw(inst, seq)
        want = _brute_vrptw(inst, seq)
        assert want is not None
        assert out is not None
        assert (out["vehicles"], round(out["distance"], 6)) == (want[0], round(want[1], 6)), \
            (out, want)
        assert [c for r in out["routes"] for c in r] == seq
        ok, k, d, _, _ = evaluate_vrptw(inst, [tuple(r) for r in out["routes"]])
        assert ok, "split_vrptw produced infeasible routes"
        assert k == out["vehicles"]
        assert abs(d - out["distance"]) <= 1e-6 * max(1.0, d)


def test_bundle_bounds():
    inst = _cvrp(40, 40.0, 7)
    rng = random.Random(7)
    routes = [[i] for i in range(1, 41)]
    rng.shuffle(routes)
    for idx in (0, 5, 20):
        picked = bundle(routes, inst.dist, idx)
        assert 2 <= len(picked) <= 4, picked
        assert idx in picked
        assert picked == tuple(sorted(picked))
        assert sum(len(routes[k]) for k in picked) <= 24
        # 单条路线超过 bundle 上限时不可成 bundle
        big = [[i for i in range(1, 26)]]
        assert bundle(big, inst.dist, 0) == ()


def test_deadline_respected_both_variants():
    inst = _cvrp(30, 40.0, 11)
    adapter = adapter_for(inst)
    for variant in ("full", "bugfix"):
        t0 = time.perf_counter()
        out = standard_search(adapter, seconds=3, seed=3, heatmap_fn=None,
                              split_enabled=True, slot=False, variant=variant)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 3.0 + 0.3, (variant, elapsed)
        ok, val, k = adapter.evaluate(out["routes"])
        assert ok and k >= 1


def test_working_best_management():
    inst = _cvrp(25, 40.0, 13)
    adapter = adapter_for(inst)
    out = standard_search(adapter, seconds=3, seed=5, heatmap_fn=None,
                          split_enabled=True, slot=False, variant="full")
    ok, val, k = adapter.evaluate([tuple(r) for r in out["routes"]])
    assert ok
    from standard_vrp.solver.pool_search import nn_witness
    ok0, val0, k0 = adapter.evaluate([tuple(r) for r in nn_witness(inst)])
    assert ok0
    assert val <= val0 + 1e-6, (val, val0)
    assert out["epochs"] >= 1
    # reset 语义：epochs % 4 == 0 时状态回置 best；迭代数应不小于 epoch 数
    assert out["iterations"] >= out["epochs"]


def test_deadline_large_instance_both_variants():
    """大实例（n=200）下两种变体都不得超出在线预算（含硬截止守卫）。"""
    inst = _cvrp(200, 60.0, 21)
    adapter = adapter_for(inst)
    for variant in ("full", "bugfix"):
        t0 = time.perf_counter()
        out = standard_search(adapter, seconds=4, seed=9, heatmap_fn=None,
                              split_enabled=True, slot=False, variant=variant)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 4.0 + 0.3, (variant, elapsed)
        ok, val, k = adapter.evaluate([tuple(r) for r in out["routes"]])
        assert ok and k >= 1
        assert set(out["history"]) == {"5", "15", "30", "60"}


def test_repair_batch_related_and_acceptance():
    """related 模式（偶数次迭代亲和度掩码）与 SA 接受须可运行且不改坏记录解。"""
    from standard_vrp.solver.masked_repair import repair_batch
    inst = _cvrp(60, 40.0, 31)
    adapter = adapter_for(inst)
    from standard_vrp.solver.pool_search import nn_witness
    state = [list(r) for r in nn_witness(inst)]
    ok, val, k = adapter.evaluate([tuple(r) for r in state])
    assert ok
    res = repair_batch(adapter, state, state, list(range(1, 41)), heat=None, seed=5,
                       seconds=0.4, tolerance=1.5, related=True, regret=False)
    ok2, val2, k2 = adapter.evaluate([tuple(r) for r in res["best"]])
    assert ok2
    assert val2 <= val + 1e-6, (val2, val)
    assert res["iterations"] >= 1
    res2 = repair_batch(adapter, state, state, list(range(1, 41)), heat=None, seed=7,
                        seconds=0.4, tolerance=1.5, related=True, regret=True)
    ok3, val3, k3 = adapter.evaluate([tuple(r) for r in res2["best"]])
    assert ok3 and val3 <= val + 1e-6
