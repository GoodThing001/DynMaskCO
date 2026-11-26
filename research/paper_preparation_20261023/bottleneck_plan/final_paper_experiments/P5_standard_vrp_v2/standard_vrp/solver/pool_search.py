# -*- coding: utf-8 -*-
"""V2 池搜索：以冷链锁版 ORDER_SPLIT（HYBRID）为蓝本的机制完整恢复。

恢复的机制（SOURCE_AUDIT M1–M7 与锁版 12 项清单）：
  M1 掩码池（40 客）+ 自适应双模式（related / random，奖励自适应）
  M2 冲突池（现有邻接在热图中排名靠后者）
  M3 每 epoch 一次神经前向，普通重构与跨车重构共享
  M4 跨车 bundle：随机起始 + 空间最近 2–4 条路线（≤24 客），其客户并入本 epoch 掩码
  M5 六配置短时试选（3 排序 × 2 算子，每配置 ≤0.04s），采用最优配置持续搜索
  M6 跨车顺序扰动（随机排列 + 25% 反转 + 1–4 次热图引导扰动）+ 固定序列 Split DP
  M7 工作解 SA 接受 + 历史最好解单调 + 每 4 epoch 重置
  M8 组破坏规模（贪心 6–17 / regret 10）与组内多次重构
  M9 预测异常留痕（类型/堆栈/次数/覆盖率/回退），不静默吞掉

SA 温度（登记的无量纲归一化，不改目标、不在正式面板选参）：
  temperature = 2.5 * max(0.10, 1 - elapsed/seconds) * |best_score| / n
"""
from __future__ import annotations

import math
import random
import time
import traceback

import numpy as np

from standard_vrp.core.calibration import calibrated_heat
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance, route_schedule,
                                                 split_cvrp, split_vrptw)
from standard_vrp.solver.masked_repair import (_TimeUp, masked_repair,
                                              repair_batch)

MASK_POOL = 40
BUNDLE_CAP = 24
SPLIT_PERIOD = 3
RESET_PERIOD = 4
CONFIG_TRIAL_S = 0.04
EPOCH_CAP_S = 1.6
REPAIR_SLICE_S = 0.10
TEMPERATURE_BASE = 2.5
TRIAL_ALLOWANCE_S = 0.10
TRIAL_TOLERANCE = 1.5


class PredictorStats:
    """神经预测异常与覆盖率留痕（M9）。"""

    def __init__(self):
        self.attempts = 0
        self.success = 0
        self.failed = 0
        self.fallbacks = 0
        self.forward_s = 0.0
        self.errors = []

    def note_error(self, exc, where):
        self.failed += 1
        self.errors.append(dict(where=where, type=type(exc).__name__, message=str(exc)[:300],
                                traceback=traceback.format_exc()[-1200:]))

    def note_fallback(self):
        self.fallbacks += 1

    def summary(self):
        return dict(attempts=self.attempts, success=self.success, failed=self.failed,
                    fallbacks=self.fallbacks, forward_s=round(self.forward_s, 4),
                    errors=self.errors[:20])


def nn_witness(inst):
    """共同初解（确定性构造，必须覆盖全部客户）。

    CVRP：最近邻链 + 容量断链（原语义）。
    VRPTW：按 ready time 升序做"最优可行插位"构造（缺此步骤时紧时窗实例会被漏客，
    造成共同初解不可行——2026-11-26 开发面板复验发现，见 BUGFIX_REPORT D7）。
    两条路径都不使用随机数，保证所有版本共用同一初解。
    """
    n = inst.dimension
    if not isinstance(inst, VRPTWInstance):
        limit = getattr(inst, "max_vehicles", None)
        remaining = set(range(1, n + 1))
        routes = []
        while remaining:
            route, cur = [], 0
            while remaining:
                nxt = min(remaining, key=lambda i: inst.dist[cur][i])
                trial = route + [nxt]
                if sum(inst.demands[i] for i in trial) > inst.capacity + 1e-9:
                    break
                route.append(nxt)
                remaining.remove(nxt)
                cur = nxt
            if route:
                routes.append(tuple(route))
            else:  # 容量不足以单独成线（异常实例）：单独成车并继续
                only = min(remaining)
                routes.append((only,))
                remaining.discard(only)
        return tuple(routes)

    order = sorted(range(1, n + 1),
                   key=lambda i: (inst.windows[i][0], inst.windows[i][1], i))
    routes = []          # list[list[int]]
    for cust in order:
        best = None
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                trial = route[:pos] + [cust] + route[pos:]
                if route_schedule(inst, tuple(trial)) is None:
                    continue
                seq = [0] + route + [0]
                delta = (inst.edge_distance(seq[pos], cust)
                         + inst.edge_distance(cust, seq[pos + 1])
                         - inst.edge_distance(seq[pos], seq[pos + 1]))
                if best is None or delta < best[0] - 1e-12:
                    best = (delta, ri, pos)
        if best is not None:
            routes[best[1]].insert(best[2], cust)
        else:
            routes.append([cust])
    limit = getattr(inst, "max_vehicles", None)
    if limit is not None:
        routes = _shrink_fleet(inst, routes, limit)
    return tuple(tuple(r) for r in routes)


def _shrink_fleet(inst, routes, limit):
    """车队压缩：把短线客户并入其他路线，直到车辆数不超过限额或无法再并入。"""
    routes = [list(r) for r in routes if r]
    while len(routes) > limit:
        sizes = sorted(range(len(routes)), key=lambda k: len(routes[k]))
        moved_any = False
        for slot in sizes:
            source = routes[slot]
            plan, ok = [], True
            for cust in list(source):
                best = None
                for ri, route in enumerate(routes):
                    if ri == slot:
                        continue
                    for pos in range(len(route) + 1):
                        trial = route[:pos] + [cust] + route[pos:]
                        if route_schedule(inst, tuple(trial)) is None:
                            continue
                        seq = [0] + route + [0]
                        delta = (inst.edge_distance(seq[pos], cust)
                                 + inst.edge_distance(cust, seq[pos + 1])
                                 - inst.edge_distance(seq[pos], seq[pos + 1]))
                        if best is None or delta < best[0] - 1e-12:
                            best = (delta, ri, pos)
                if best is None:
                    ok = False
                    break
                plan.append((cust, best[1], best[2]))
            if ok:
                for cust, ri, pos in plan:
                    routes[ri].insert(pos, cust)
                routes.pop(slot)
                moved_any = True
                break
        if not moved_any:
            break
    return routes


def _ranks(heat):
    mat = np.asarray(heat, dtype=float)
    mat = np.where(np.isfinite(mat), mat, -np.inf)
    order = np.argsort(-mat, axis=1, kind="stable")
    return (1 + np.argsort(order, axis=1, kind="stable")).tolist()


def conflict_pool(routes, removed, heat):
    """锁版 conflict_pool：现有邻接热图排名靠后的 12 客 + 每人最佳替代 + 组内前 8。"""
    if heat is None:
        return list(removed)
    n = len(heat) - 1
    rank = _ranks(heat)
    neighbours = {}
    for r in routes:
        seq = (0,) + tuple(r) + (0,)
        for a, i, b in zip(seq[:-1], seq[1:], seq[2:]):
            neighbours[i] = (a, b)
    ids = sorted(removed)
    def conflict_key(i):
        if i not in neighbours:
            return n + 1
        a, b = neighbours[i]
        return max(rank[i][a], rank[i][b])
    conflicts = sorted(ids, key=conflict_key, reverse=True)
    chosen = set(conflicts[:12])
    for i in conflicts[:12]:
        if i not in neighbours:
            continue
        a, b = neighbours[i]
        for j in sorted(ids, key=lambda j: rank[i][j] + rank[j][i]):
            if j not in (i, a, b):
                chosen.add(j)
                break
    chosen.update(ids[:8])
    return [i for i in ids if i in chosen]


def bundle(routes, dist, index):
    """锁版 bundle：随机起始路线 + 空间最近 2–4 条、总客户 ≤24 的路线组。"""
    if index >= len(routes) or not routes[index] or len(routes[index]) > BUNDLE_CAP:
        return ()
    origin = routes[index]
    close = []
    for k, r in enumerate(routes):
        if k == index or not r:
            continue
        close.append((min(dist[i][j] for i in origin for j in r), k))
    close.sort()
    size = len(origin)
    selected = [index]
    for _, k in close:
        if size + len(routes[k]) <= BUNDLE_CAP:
            selected.append(k)
            size += len(routes[k])
        if len(selected) == 4:
            break
    return tuple(sorted(selected)) if len(selected) >= 2 else ()


def propose_orders(parts, weights, seed, limit):
    """锁版 proposals：随机路线排列（25% 反转）+ 1–4 次热图引导扰动。"""
    rng = random.Random(seed)
    ids = sorted(i for route in parts for i in route)
    priority = {}
    for i in ids:
        scored = sorted(((weights[i][j] + weights[j][i] if weights is not None else 0.0, j)
                         for j in ids if j != i), reverse=True)
        priority[i] = [j for _, j in scored]
    for it in range(limit):
        seq = []
        for k in rng.sample(range(len(parts)), len(parts)):
            r = parts[k]
            seq.extend(reversed(r) if rng.random() < 0.25 else r)
        for _ in range(1 + it % 4):
            i = ids[rng.randrange(len(ids))]
            u = seq.index(i)
            avail = [j for j in priority[i] if abs(seq.index(j) - u) > 1]
            if not avail:
                continue
            j = avail[int((rng.random() ** 2) * min(8, len(avail)))]
            v = seq.index(j)
            if rng.random() < 0.5:
                l, r = sorted((u, v))
                seq[l + 1:r + 1] = list(reversed(seq[l + 1:r + 1]))
            else:
                seq.pop(u)
                v = seq.index(j)
                seq.insert(v + (1 if rng.random() < 0.5 else 0), i)
        yield list(seq)


# ------------------------------------------------------------------ 排序
def ranking_for(adapter, state, mask, kind, heat):
    """客户重插顺序（锁版语义：distance/heat/calib 三种排序）。

    kind 0: 距离增量最小优先（纯距离）；
    kind 1: 热图边分（H(i,a)+H(i,b)-H(a,b)）最大优先；
    kind 2: 校准热图同理。
    热图缺失（-inf）的客户排在后面（距离序），不产生 NaN。
    """
    mask = list(mask)
    if kind == 0 or heat is None:
        return sorted(mask, key=lambda i: min((adapter.inst.dist[i][j] for r in state for j in [0] + list(r)),
                                              default=0.0))
    def score(i):
        best = -math.inf
        for r in state:
            seq = [0] + list(r) + [0]
            for pos in range(len(seq) - 1):
                a, b = seq[pos], seq[pos + 1]
                h = heat[a][i] + heat[i][b] - heat[a][b]
                if math.isfinite(h) and h > best:
                    best = h
        return best
    return sorted(mask, key=lambda i: (-score(i), i))


# ------------------------------------------------------------------ 主搜索
CHECKPOINTS = (5.0, 15.0, 30.0, 60.0)


def standard_search(adapter, seconds=60, seed=1, heatmap_fn=None, split_enabled=True,
                    slot=False, mode_name="hybrid", variant="full", diagnostics=None):
    """池搜索主循环。

    variant:
      "full"   —— 锁版 HYBRID 全机制（STD_FIX / STD_NORM）
      "bugfix" —— 仅缺陷修复，保持 P5-V1 的简化搜索（STD_BUGFIX）：
                   每个 epoch 六配置全量试选（无时间片）、跨车仅 2 条随机路线 + 单条顺序链、
                   严格改善才更新、无 SA/重置/冲突池/自适应
      mode_name: "hybrid"/"choice"/"pool"（仅 variant="full" 时生效）
    """
    inst = adapter.inst
    n = inst.dimension
    bugfix = variant == "bugfix"
    started = time.perf_counter()
    # 内部截止：保守留出"单步不可分割操作"的余量（numpy O(n^2) 步在负载竞争下可达数秒）
    margin = 1.0 + 0.0015 * n
    deadline = started + seconds - max(0.8, margin)
    hard = started + seconds - (margin + 0.3)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    pred = PredictorStats()
    dist = inst.dist

    state = [list(r) for r in nn_witness(inst)]
    best = [list(r) for r in state]
    best_score = adapter.score([tuple(r) for r in best])
    init_score = best_score

    rewards = np.zeros(2)
    uses = np.zeros(2, dtype=int)
    ranking_uses = np.zeros(3, dtype=int)
    regret_epochs = 0
    epochs = iterations = improvements = 0
    split_blocks = split_orders = split_offers = 0
    fallbacks = 0
    history = {}

    def score_of(routes):
        return adapter.score([tuple(r) for r in routes])

    while time.perf_counter() < deadline:
        epoch_start = time.perf_counter()
        before_epoch = score_of(state)
        if (not bugfix) and epochs % RESET_PERIOD == 0:
            state = [list(r) for r in best]
        # ---- 掩码池 ----
        pool = list(nrng.choice(np.arange(1, n + 1), size=min(MASK_POOL, n), replace=False))
        selected = ()
        if bugfix:
            # STD_BUGFIX：V1 语义——每 3 epoch 随机 2 条路线（总客户 ≤24），不并入掩码
            if split_enabled and epochs % SPLIT_PERIOD == 0 and len(state) >= 2:
                idx = int(nrng.integers(len(state)))
                cand = [k for k in range(len(state)) if k != idx and state[k]
                        and len(state[idx]) + len(state[k]) <= BUNDLE_CAP]
                if cand:
                    selected = tuple(sorted((idx, int(nrng.choice(cand)))))
        elif split_enabled and epochs % SPLIT_PERIOD == 0 and len(state) >= 2:
            selected = bundle(state, dist, int(nrng.integers(len(state))))
            if selected:
                hidden = [i for k in selected for i in state[k]]
                hidden_set = set(hidden)
                leftover = [i for i in range(1, n + 1) if i not in hidden_set]
                need = max(0, MASK_POOL - len(hidden))
                extra = list(nrng.choice(leftover, size=min(need, len(leftover)), replace=False)) if leftover else []
                pool = list(hidden + extra)[:MASK_POOL]
        # ---- 每 epoch 一次前向（留痕） ----
        heat = None
        if heatmap_fn is not None:
            pred.attempts += 1
            t0 = time.perf_counter()
            try:
                heat = heatmap_fn(inst, [tuple(r) for r in state], set(pool), None)
                if heat is None or not np.isfinite(np.asarray(heat, dtype=float)).any():
                    heat = None
                else:
                    pred.success += 1
            except Exception as exc:  # noqa: BLE001
                pred.note_error(exc, "epoch_forward")
                heat = None
                fallbacks += 1
                pred.note_fallback()
            pred.forward_s += time.perf_counter() - t0
        # ---- 跨车顺序搜索 + Split DP ----
        if selected and time.perf_counter() < deadline:
            split_blocks += 1
            parts = [tuple(state[k]) for k in selected]
            before = score_of(state)
            best_parts = parts
            best_sc = before
            best_cand = None
            stop = min(deadline, epoch_start + EPOCH_CAP_S, time.perf_counter() + 0.30)
            weights = heat if heat is not None else None
            if bugfix:
                # V1 语义：热图/距离最近邻单链
                orders = [_nearest_chain(parts, heat, dist)]
            else:
                orders = propose_orders(parts, weights, seed * 100 + epochs, 100000)
            for seq in orders:
                if time.perf_counter() >= stop:
                    break
                split_orders += 1
                out = _split_sequence(adapter, seq)
                if out is None:
                    continue
                cand = [list(r) for k, r in enumerate(state) if k not in selected] + [list(r) for r in out]
                sc = score_of(cand)
                if sc > best_sc + 1e-9:
                    best_parts, best_sc, best_cand = out, sc, cand
            if best_cand is not None:
                state = best_cand
                split_offers += 1
                if best_sc > best_score + 1e-9:
                    best = [list(r) for r in state]
                    best_score = best_sc
        # ---- 校准热图 ----
        heat_cal = None
        if heat is not None and mode_name in ("hybrid", "choice"):
            try:
                heat_cal = calibrated_heat(dist, heat)
            except Exception as exc:  # noqa: BLE001
                # 校准失败必须留痕（D8：曾因硬编码路径 + 静默回退导致校准从未生效）
                pred.note_error(exc, "calibration")
                fallbacks += 1
        # ---- 自适应双模式（仅 full） ----
        if bugfix:
            active_mode = 0
            uses[0] += 1
            working_pool = pool
        else:
            if epochs < 2 or epochs % RESET_PERIOD == 0:
                active_mode = int(np.argmin(uses))
            else:
                active_mode = int(np.argmax(rewards + 0.1 / np.sqrt(1 + uses)))
            uses[active_mode] += 1
            working_pool = conflict_pool(state, pool, heat) if (active_mode and heat is not None) else pool
            if not working_pool:
                working_pool = pool
        # ---- 六配置短时试选（锁版 pool_search.cpp L192–218：三选项 × 两算子） ----
        end = min(deadline, epoch_start + EPOCH_CAP_S)
        op_choice, ranking_choice = "greedy", 0
        heat_choice = None
        options = [None]
        if heat is not None:
            options = [None, heat] if heat_cal is None else [None, heat, heat_cal]
        if (bugfix or mode_name in ("hybrid", "choice")) and end - time.perf_counter() > 0.30:
            if bugfix:
                trials = []
                for rk, opt in enumerate([o for o in options if o is not None] or [None]):
                    for op in ("greedy", "regret2"):
                        if time.perf_counter() >= end - 0.005:
                            break
                        allowance = max(0.0, end - time.perf_counter() - 0.01)
                        if allowance <= 0:
                            break
                        try:
                            child, _ = masked_repair(adapter, [tuple(r) for r in state],
                                                     set(working_pool),
                                                     ranking_for(adapter, state, working_pool, rk, opt),
                                                     op, heat=opt, slot=slot,
                                                     deadline=time.perf_counter() + allowance)
                        except _TimeUp:
                            continue
                        trials.append((score_of(child), op, rk, opt, child))
                if trials:
                    trials.sort(key=lambda x: -x[0])
                    sc, op_choice, ranking_choice, heat_choice, child = trials[0]
                    ranking_uses[min(ranking_choice, 2)] += 1
                    regret_epochs += int(op_choice == "regret2")
                    if sc > best_score + 1e-9:
                        best = [list(r) for r in child]
                        best_score = sc
                    if sc >= score_of(state) - 1e-9:
                        state = [list(r) for r in child]
            else:
                proposals = [(rk, g) for g in (0, 1) for rk in range(len(options))]
                seed_trial = int(nrng.integers(2 ** 32))
                outcomes = []
                for proposal, (rk, g) in enumerate(proposals):
                    allowance = min(TRIAL_ALLOWANCE_S, end - time.perf_counter() - 0.005)
                    if allowance <= 0:
                        break
                    duration = min(allowance, CONFIG_TRIAL_S)
                    if duration < CONFIG_TRIAL_S:
                        continue
                    res = repair_batch(adapter, state, state, working_pool,
                                       heat=options[rk], seed=seed_trial, seconds=duration,
                                       tolerance=TRIAL_TOLERANCE,
                                       related=bool(active_mode), regret=bool(g),
                                       deadline=deadline, hard_deadline=hard)
                    iterations += res["iterations"]
                    improvements += res["improvements"]
                    outcomes.append((score_of(res["best"]),
                                     -((proposal - epochs) % len(proposals)), rk, g,
                                     res["state"], res["best"]))
                if outcomes:
                    sc, _tie, ranking_choice, g_choice, w_state, w_best = max(
                        outcomes, key=lambda x: x[:2])
                    heat = options[ranking_choice]
                    ranking_uses[ranking_choice] += 1
                    regret_epochs += int(g_choice)
                    op_choice = "regret2" if g_choice else "greedy"
                    if sc > best_score + 1e-9:
                        best = [list(r) for r in w_best]
                        best_score = sc
                    state = [list(r) for r in w_state]
        # ---- 组内多次重构（锁版 L219–229：工作解 + SA 接受） ----
        while time.perf_counter() < end:
            allowance = min(REPAIR_SLICE_S, end - time.perf_counter() - 0.002)
            if allowance <= 0:
                break
            temperature = TEMPERATURE_BASE * max(0.10, 1.0 - (time.perf_counter() - started) / seconds)
            if bugfix:
                k = min(len(working_pool), 10 if op_choice == "regret2" else 6 + rng.randrange(12))
                mask = set(rng.sample(list(working_pool), k))
                opt_b = heat_choice if heat_choice is not None else None
                try:
                    child, _ = masked_repair(adapter, [tuple(r) for r in state], mask,
                                             ranking_for(adapter, state, mask, 0, opt_b),
                                             op_choice, heat=opt_b, slot=slot,
                                             deadline=min(time.perf_counter() + allowance, hard))
                except _TimeUp:
                    break
                iterations += 1
                child_score = score_of(child)
                if child_score > best_score + 1e-9:
                    best = [list(r) for r in child]
                    best_score = child_score
                    improvements += 1
                if child_score > score_of(state) + 1e-9:
                    state = [list(r) for r in child]
                continue
            res = repair_batch(adapter, state, best, working_pool, heat=heat,
                               seed=int(nrng.integers(2 ** 32)), seconds=allowance,
                               tolerance=temperature, related=bool(active_mode),
                               regret=(op_choice == "regret2"), deadline=deadline,
                               hard_deadline=hard)
            iterations += res["iterations"]
            state = [list(r) for r in res["state"]]
            if res["best_value"] > best_score + 1e-9:
                best = [list(r) for r in res["best"]]
                best_score = res["best_value"]
                improvements += res["improvements"]
        # ---- 奖励更新 ----
        gain = max(0.0, score_of(best) - before_epoch)
        rewards[active_mode] = 0.65 * rewards[active_mode] + 0.35 * min(3.0, math.log1p(gain))
        epochs += 1
        # ---- 5/15/30/60 秒历史最好解快照（协议 §5，纯记录，不影响搜索） ----
        elapsed = time.perf_counter() - started
        for cp in CHECKPOINTS:
            if cp not in history and elapsed >= cp:
                ok_c, val_c, k_c = adapter.evaluate([tuple(r) for r in best])
                history[str(int(cp))] = dict(distance=val_c, vehicles=k_c,
                                             at=round(elapsed, 3))

    final = [tuple(r) for r in best]
    ok, val, k = adapter.evaluate(final)
    for cp in CHECKPOINTS:
        if cp not in history:
            history[str(int(cp))] = dict(distance=val, vehicles=k,
                                        at=round(time.perf_counter() - started, 3))
    stats = dict(ok=ok, score=adapter.score(final), distance=val, vehicles=k,
                 initial_score=init_score, epochs=epochs, iterations=iterations,
                 improvements=improvements, split_blocks=split_blocks, split_orders=split_orders,
                 split_offers=split_offers, ranking_counts=ranking_uses.tolist(),
                 regret_epochs=regret_epochs, mask_mode_counts=uses.tolist(),
                 fallbacks=fallbacks, predictor=pred.summary(), history=history,
                 elapsed=round(time.perf_counter() - started, 3))
    if diagnostics is not None:
        diagnostics.update(stats)
    return dict(routes=final, **stats)


def _split_sequence(adapter, seq):
    inst = adapter.inst
    if isinstance(inst, CVRPInstance):
        out = split_cvrp(inst, list(seq), rounded=True)
        return out["routes"] if out else None
    out = split_vrptw(inst, list(seq))
    return out["routes"] if out else None


def _nearest_chain(parts, heat, dist):
    """V1 语义的跨车顺序：把路线块拼接后按热图（或距离）最近邻链排序。"""
    ids = [i for r in parts for i in r]
    left = set(ids)
    seq, cur = [], 0
    while left:
        if heat is not None:
            nxt = max(left, key=lambda i: (heat[cur][i] if math.isfinite(heat[cur][i]) else -math.inf))
            if not math.isfinite(heat[cur][nxt]):
                nxt = min(left, key=lambda i: dist[cur][i])
        else:
            nxt = min(left, key=lambda i: dist[cur][i])
        seq.append(nxt)
        left.remove(nxt)
        cur = nxt
    return seq
