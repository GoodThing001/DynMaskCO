# -*- coding: utf-8 -*-
"""标准 ALNS 与 VNS（标准任务独立实现，距离 / 字典序 (K,D) 目标）。

ALNS：Ropke–Pisinger 框架（随机移除 / 最坏移除 / 相关移除；贪心 / regret-2 修复；模拟退火）。
VNS：relocate / or-opt / 2-opt* / swap 邻域，first-improvement + 震荡。
共同初解 = 最近邻见证（与本文方法一致）。
"""
from __future__ import annotations

import math
import random
import time

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw,
                                                  route_cost_vrptw, better_lex,
                                                  route_schedule)


def _score(inst, routes, objective):
    if objective == "cvrp":
        ok, d, _, _ = evaluate_cvrp(inst, routes, rounded=True)
        return (ok, d, len(routes))
    ok, k, d, _, _ = evaluate_vrptw(inst, routes)
    return (ok, d, k)


def nn_witness_std(inst):
    from standard_vrp.solver.standard_pool_search import nn_witness
    return nn_witness(inst)


def _greedy_insert(inst, partial, cust, objective):
    """把客户插入当前解的最优位置；返回 (routes, cost) 或 None。

    可行性只按被修改路线检查（partial 尚缺其他被移除客户，全解覆盖检查必然失败）：
    CVRP 检查容量、VRPTW 检查该路线可行时刻表；插入保持 K 不变，按局部距离增量择优，
    无可行位置时开新车。
    """
    best = None  # (delta, ri, pos)
    for ri, _r in enumerate(partial):
        r = list(_r)
        for pos in range(len(r) + 1):
            nr = tuple(r[:pos] + [cust] + r[pos:])
            if objective == "cvrp":
                if sum(inst.demands[i] for i in nr) > inst.capacity + 1e-9:
                    continue
            else:
                if route_schedule(inst, nr) is None:
                    continue
            delta = _insert_delta(inst, r, pos, cust, rounded=(objective == "cvrp"))
            if best is None or delta < best[0] - 1e-9:
                best = (delta, ri, pos)
    if best is not None:
        trial = list(partial)
        r = list(trial[best[1]])
        trial[best[1]] = tuple(r[:best[2]] + [cust] + r[best[2]:])
        return tuple(tuple(r) for r in trial if r)
    # 开新车
    trial = list(partial) + [[cust]]
    return tuple(tuple(r) for r in trial if r)


def _insert_delta(inst, r, pos, cust, rounded=False):
    """在路线 r 的位置 pos 插入客户 cust 的局部距离增量（含仓库边）。"""
    seq = [0] + list(r) + [0]
    a, b = seq[pos], seq[pos + 1]
    return (inst.edge_distance(a, cust, rounded) + inst.edge_distance(cust, b, rounded)
            - inst.edge_distance(a, b, rounded))


def _remove_customers(inst, routes, q, kind, rng):
    flat = [i for r in routes for i in r]
    n_remove = min(q, len(flat))
    if kind == "random":
        return rng.sample(flat, n_remove)
    if kind == "worst":
        costs = []
        for r in routes:
            for idx, i in enumerate(r):
                prev = r[idx - 1] if idx > 0 else 0
                nxt = r[idx + 1] if idx + 1 < len(r) else 0
                saving = inst.dist[prev][i] + inst.dist[i][nxt] - inst.dist[prev][nxt]
                costs.append((saving, i))
        costs.sort(key=lambda x: -x[0])
        return [i for _, i in costs[:n_remove]]
    # related（Shaw）：以随机种子客户为基准，按距离取最近
    seed_cust = rng.choice(flat)
    others = [i for i in flat if i != seed_cust]
    others.sort(key=lambda i: inst.dist[seed_cust][i])
    return [seed_cust] + others[:n_remove - 1]


def alns_search(inst, seconds=60, seed=1, objective="cvrp", rho=None):
    rng = random.Random(seed)
    started = time.perf_counter()
    routes = nn_witness_std(inst)
    ok0, d0, k0 = _score(inst, routes, objective)
    best = routes
    best_d, best_k = d0, k0
    current = routes
    cur_d, cur_k = d0, k0
    n = inst.dimension
    q_base = max(4, int(0.1 * n) + 1)
    T0 = 0.02 * (cur_d / max(1, len([i for r in current for i in r])))
    T = T0
    iterations = 0
    cutoff = seconds - 0.8
    deadline = started + cutoff
    while time.perf_counter() - started < cutoff:
        q = rng.randint(1, min(40, q_base + 4))
        kind = rng.choice(("random", "worst", "related"))
        removed = _remove_customers(inst, current, q, kind, rng)
        partial = tuple(tuple(i for i in r if i not in removed) for r in current)
        partial = tuple(r for r in partial if r)
        if _expired(deadline):
            break
        # 修复：贪心 / regret-2（轮内截止守卫：大实例单轮可能超余量）
        if rng.random() < 0.5:
            cand = partial
            for cust in removed:
                if _expired(deadline):
                    break
                out = _greedy_insert(inst, list(cand), cust, objective)
                if out is None:
                    break
                cand = out
        else:
            cand = partial
            remaining = list(removed)
            while remaining and not _expired(deadline):
                best_ins = None
                for cust in remaining:
                    out = _greedy_insert(inst, list(cand), cust, objective)
                    if out is not None:
                        ok, d, k = _score(inst, out, objective)
                        val = (k, d) if objective == "vrptw" else (d, k)
                        if best_ins is None or val < best_ins[0]:
                            best_ins = (val, out, cust)
                if best_ins is None:
                    break
                cand = best_ins[1]
                remaining.remove(best_ins[2])
        if _expired(deadline):
            break
        if cand is None:
            continue
        ok, d, k = _score(inst, cand, objective)
        if not ok:
            continue
        accept = False
        if objective == "cvrp":
            if d < cur_d - 1e-9:
                accept = True
            elif T > 1e-9:
                accept = rng.random() < math.exp(-(d - cur_d) / T)
        else:
            if better_lex(k, d, cur_k, cur_d):
                accept = True
            elif T > 1e-9:
                accept = rng.random() < math.exp(-(d - cur_d) / T)
        if accept:
            current, cur_d, cur_k = cand, d, k
            if objective == "cvrp":
                if d < best_d - 1e-9:
                    best, best_d = cand, d
            else:
                if better_lex(k, d, best_k, best_d):
                    best, best_d, best_k = cand, d, k
        T = max(1e-6, T * 0.9995)
        iterations += 1
    ok, d, k = _score(inst, best, objective)
    return dict(routes=best, distance=d, vehicles=k, iterations=iterations)


def vns_search(inst, seconds=60, seed=1, objective="cvrp"):
    rng = random.Random(seed)
    started = time.perf_counter()
    routes = nn_witness_std(inst)
    ok, d, k = _score(inst, routes, objective)
    best, best_d, best_k = routes, d, k
    iterations = 0
    cutoff = seconds - 0.8
    deadline = started + cutoff
    while time.perf_counter() - started < cutoff:
        improved = False
        # 邻域 1：relocate（把客户移到另一位置/路线）
        routes, improved, d, k = _relocate(inst, routes, objective, rng, deadline)
        if improved:
            if objective == "cvrp":
                if d < best_d - 1e-9:
                    best, best_d = routes, d
            else:
                if better_lex(k, d, best_k, best_d):
                    best, best_d, best_k = routes, d, k
            continue
        # 邻域 2：swap
        routes, improved, d, k = _swap(inst, routes, objective, rng, deadline)
        if improved:
            if objective == "cvrp":
                if d < best_d - 1e-9:
                    best, best_d = routes, d
            else:
                if better_lex(k, d, best_k, best_d):
                    best, best_d, best_k = routes, d, k
            continue
        # 邻域 3：2-opt（路线内反转段）
        routes, improved, d, k = _two_opt(inst, routes, objective, rng, deadline)
        if improved:
            if objective == "cvrp":
                if d < best_d - 1e-9:
                    best, best_d = routes, d
            else:
                if better_lex(k, d, best_k, best_d):
                    best, best_d, best_k = routes, d, k
            continue
        # 震荡：随机打散一个路线段
        routes = _shake(inst, routes, rng)
        ok, d, k = _score(inst, routes, objective)
        iterations += 1
        if time.perf_counter() - started >= cutoff:
            break
    ok, d, k = _score(inst, best, objective)
    return dict(routes=best, distance=d, vehicles=k, iterations=iterations)


def _expired(deadline):
    return deadline is not None and time.perf_counter() >= deadline


def _relocate(inst, routes, objective, rng, deadline=None):
    flat = [(i, r) for r in routes for i in r]
    order = list(range(len(flat)))
    rng.shuffle(order)
    for oi in order:
        if _expired(deadline):
            return routes, False, *_score(inst, routes, objective)[1:]
        i, src = flat[oi]
        src_idx = routes.index(src)
        # 同路线重定位 + 异路线
        cands = []
        for ti, t in enumerate(routes):
            if _expired(deadline):
                return routes, False, *_score(inst, routes, objective)[1:]
            for pos in range(len(t) + 1):
                if ti == src_idx and (pos == src.index(i) or pos == src.index(i) + 1):
                    continue
                trial = [list(r) for r in routes]
                trial[src_idx].remove(i)
                trial[ti].insert(pos if ti != src_idx or pos <= len(trial[ti]) else len(trial[ti]), i)
                trial = tuple(tuple(r) for r in trial if r)
                ok, d, k = _score(inst, trial, objective)
                if ok:
                    cands.append((d, k, trial))
        if cands:
            if objective == "cvrp":
                cands.sort(key=lambda x: x[0])
            else:
                cands.sort(key=lambda x: (x[1], x[0]))
            d, k, trial = cands[0]
            ok0, d0, k0 = _score(inst, routes, objective)
            if (d < d0 - 1e-9) if objective == "cvrp" else better_lex(k, d, k0, d0):
                return trial, True, d, k
    return routes, False, *_score(inst, routes, objective)[1:]


def _swap(inst, routes, objective, rng, deadline=None):
    flat = [i for r in routes for i in r]
    order = list(range(len(flat)))
    rng.shuffle(order)
    for a in order[:60]:
        if _expired(deadline):
            return routes, False, *_score(inst, routes, objective)[1:]
        for b in order:
            if a >= b:
                continue
            i, j = flat[a], flat[b]
            trial = [list(r) for r in routes]
            for r in trial:
                if i in r and j in r:
                    ri = r.index(i)
                    rj = r.index(j)
                    r[ri], r[rj] = r[rj], r[ri]
                    break
            else:
                ri = [r.index(i) for r in trial if i in r][0]
                rj = [r.index(j) for r in trial if j in r][0]
                ti = [idx for idx, r in enumerate(trial) if i in r][0]
                tj = [idx for idx, r in enumerate(trial) if j in r][0]
                trial[ti][ri] = j
                trial[tj][rj] = i
            trial = tuple(tuple(r) for r in trial if r)
            ok, d, k = _score(inst, trial, objective)
            if not ok:
                continue
            ok0, d0, k0 = _score(inst, routes, objective)
            if (d < d0 - 1e-9) if objective == "cvrp" else better_lex(k, d, k0, d0):
                return trial, True, d, k
    return routes, False, *_score(inst, routes, objective)[1:]


def _two_opt(inst, routes, objective, rng, deadline=None):
    order = list(range(len(routes)))
    rng.shuffle(order)
    for ri in order:
        if _expired(deadline):
            return routes, False, *_score(inst, routes, objective)[1:]
        r = list(routes[ri])
        if len(r) < 4:
            continue
        for a in range(len(r) - 1):
            for b in range(a + 2, len(r)):
                if _expired(deadline):
                    return routes, False, *_score(inst, routes, objective)[1:]
                trial = list(routes)
                seg = r[a:b + 1][::-1]
                nr = r[:a] + seg + r[b + 1:]
                trial[ri] = tuple(nr)
                trial = tuple(tuple(x) for x in trial if x)
                ok, d, k = _score(inst, trial, objective)
                if not ok:
                    continue
                ok0, d0, k0 = _score(inst, routes, objective)
                if (d < d0 - 1e-9) if objective == "cvrp" else better_lex(k, d, k0, d0):
                    return trial, True, d, k
    return routes, False, *_score(inst, routes, objective)[1:]


def _shake(inst, routes, rng):
    routes = [list(r) for r in routes]
    for r in routes:
        if len(r) >= 3 and rng.random() < 0.3:
            a = rng.randrange(len(r))
            b = rng.randrange(len(r))
            r[a], r[b] = r[b], r[a]
    return tuple(tuple(r) for r in routes if r)
