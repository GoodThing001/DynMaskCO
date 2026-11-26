# -*- coding: utf-8 -*-
"""V2 掩码重构内核：锁版 masked_repair.cpp 的逐行移植（目标经 ObjectiveAdapter 适配）。

锁版权威语义（`masked_repair.cpp`，repo 内只读）：
  * `choose_slot`：枚举全部容量可行插位（含新开单车候选），
      detour = d(a,i)+d(i,b)-d(a,b)，learned = H(a,i)+H(i,b)-H(a,b)；
      贪心 = 距离序前 6 + 其余热图序前 6（不足则按距离序补扫到出现可行）；
      Regret = 热图序 top-8 + 最短可行插位保护（替换末位）；
      候选的真实增益 = 目标函数值差（不可行 → 跳过）；
  * `reconstruct`：剥掉掩码客户；任一部分路线不可行 → 原样返回父解；
      贪心按掩码**给定顺序**插入，Regret 按客户编号升序扫描、取后悔值最大（严格 >）；
  * `repair_batch`：每次迭代重洗顺序 → k = min(|group|, regret?10 : 6+rng()%12) → 取前 k；
      related 模式（mode=1）在偶数次迭代按 affinity=H[anchor,i]+H[i,anchor] 降序做
      "平方随机"抽样组掩码；重构后若优于 record 则更新 record；
      接受条件 new >= u-eps 或 (tolerance>0 且 new > best_u-6*tolerance 且
      rand < exp((new-u)/tolerance))；record 的更新先于接受判定。

与锁版唯一的实现差别（已登记）：本文件为纯 Python 移植（无 C++ 内核），
吞吐低约两个数量级；因此上游时间片（0.04s 试选 / 0.10s 内层）下
单次重构无法完成，属**已知效率差异**（见 MECHANISM_PARITY.md §5）。
"""
from __future__ import annotations

import math
import random
import time

NEG_INF = -math.inf
EPS = 1e-7
BAD = -math.inf


class _TimeUp(Exception):
    """预算耗尽（上游用于丢弃未完成候选）。"""


def _guard(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise _TimeUp


# ------------------------------------------------------------------ 目标值
def route_value(adapter, route):
    """路线目标值（越大越好）；不可行返回 -inf（对应锁版 Economy::value 的 bad）。"""
    if not route:
        return 0.0
    v = adapter.route_proxy(route)
    return -v if math.isfinite(v) else BAD


def solution_value(adapter, routes):
    total = 0.0
    for r in routes:
        v = route_value(adapter, r)
        if not math.isfinite(v):
            return BAD
        total += v
    return total


def _edge_score(heat, a, i, b):
    """锁版 learned = H(a,i)+H(i,b)-H(a,b)；缺失（-inf/NaN）返回 None。"""
    if heat is None:
        return None
    h_ai, h_ib, h_ab = heat[a][i], heat[i][b], heat[a][b]
    if not (math.isfinite(h_ai) and math.isfinite(h_ib) and math.isfinite(h_ab)):
        return None
    return h_ai + h_ib - h_ab


# ------------------------------------------------------------------ 选位
def _choose_slot(adapter, routes, values, cust, heat=None, regret=False,
                 topk=8, greedy_distance_slots=6, greedy_heat_slots=6, deadline=None):
    """锁版 choose_slot。返回 (Move, regret)；Move = (route_index, pos)。

    route_index == len(routes) 表示新开单车。
    values: 每条路线的目标值缓存（与锁版一致，插入后由调用方更新）。
    """
    slots = []  # (detour, learned, ri, pos)
    for ri, route in enumerate(routes):
        if not adapter.fits(route, cust):
            continue
        n = len(route)
        for pos in range(n + 1):
            if deadline is not None and (pos & 7) == 0 and time.perf_counter() >= deadline:
                raise _TimeUp     # 大实例单次枚举可达数百毫秒：按槽位细粒度守卫
            a = route[pos - 1] if pos else 0
            b = route[pos] if pos < n else 0
            detour = adapter.detour(route, pos, cust)
            slots.append((detour, _edge_score(heat, a, cust, b), ri, pos))

    # 新开单车候选（锁版恒在列）
    best = (len(routes), 0)
    best_gain = route_value(adapter, [cust])
    if not slots:
        return best, float(abs(best_gain)) if math.isfinite(best_gain) else 1e6

    def gain_at(k):
        detour, _l, ri, pos = slots[k]
        route = list(routes[ri])
        route.insert(pos, cust)
        v = route_value(adapter, route)
        return NEG_INF if not math.isfinite(v) else v - values[ri]

    key_of = lambda k: (slots[k][2], slots[k][3])
    by_distance = sorted(range(len(slots)), key=lambda k: (slots[k][0], key_of(k)))

    def by_heat(idx):
        return sorted(idx, key=lambda k: (-(slots[k][1] if slots[k][1] is not None
                                           else NEG_INF), slots[k][0], key_of(k)))

    second = BAD
    feasible = False

    def inspect(k):
        nonlocal best, best_gain, second, feasible
        g = gain_at(k)
        if not math.isfinite(g):
            return
        feasible = True
        if g > best_gain + 1e-9 or (abs(g - best_gain) <= 1e-9 and key_of(k) < best):
            second = best_gain
            best, best_gain = key_of(k), g
        else:
            second = max(second, g)

    if regret:
        selected = by_heat(range(len(slots)))[:min(topk, len(slots))]
        # 最短可行插位保护（锁版：距离序扫描，遇首个可行即止）
        first = None
        for k in by_distance:
            if first is not None and slots[k][0] > slots[first][0]:
                break
            if not math.isfinite(gain_at(k)):
                continue
            if first is None:
                first = k
            if k in selected:
                break
        else:
            pass
        if first is not None and first not in selected and selected:
            selected[-1] = first
        for k in selected:
            inspect(k)
    else:
        keep = by_distance[:min(greedy_distance_slots, len(by_distance))]
        rest = [k for k in range(len(slots)) if k not in keep]
        selected = keep + by_heat(rest)[:greedy_heat_slots]
        for k in selected:
            inspect(k)
        if not feasible:
            for k in by_distance:
                if k in selected:
                    continue
                inspect(k)
                if feasible:
                    break
    regret_value = best_gain - second if math.isfinite(second) else best_gain
    return best, float(regret_value)


def choose_slot(adapter, routes, values=None, cust=None, heat=None, regret=False,
                topk=8, slot=False, greedy_distance_slots=6, greedy_heat_slots=6):
    """兼容包装：旧签名 choose_slot(adapter, routes, cust, ...) 仍可用。

    返回 (key, move, regret)；key 为 adapter 比较键（CVRP 为 -detour，VRPTW 为 (K标记,detour)）。
    """
    if cust is None:
        cust, values = values, None
    if values is None:
        values = [route_value(adapter, r) for r in routes]
    move, regret_value = _choose_slot(adapter, routes, values, cust, heat=heat, regret=regret,
                                      topk=topk, greedy_distance_slots=greedy_distance_slots,
                                      greedy_heat_slots=greedy_heat_slots)
    if move[0] >= len(routes):
        key = adapter.key(True, adapter.singleton_delta(cust))
    else:
        key = adapter.key(False, adapter.detour(routes[move[0]], move[1], cust))
    return key, move, regret_value


def insert_customer(adapter, routes, values, cust, move):
    ri, pos = move
    if ri >= len(routes):
        routes.append([cust])
        values.append(route_value(adapter, [cust]))
        return
    route = routes[ri]
    route.insert(pos, cust)
    values[ri] = route_value(adapter, route)


# ------------------------------------------------------------------ 重构
def reconstruct(adapter, parent, mask, heat=None, regret=False, log=None,
                deadline=None):
    """锁版 reconstruct；返回重建解，父解剥离后不可行时返回 None。"""
    hidden = set(mask)
    rebuilt, values = [], []
    for route in parent:
        part = [v for v in route if v not in hidden]
        if not part:
            continue
        u = route_value(adapter, part)
        if not math.isfinite(u):
            return None
        rebuilt.append(part)
        values.append(u)
    if regret:
        pending = sorted(mask)
        while pending:
            _guard(deadline)
            highest, winner, chosen = BAD, None, None
            for j, cust in enumerate(pending):
                move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat,
                                         regret=True, deadline=deadline)
                if winner is None or reg > highest:
                    highest, winner, chosen = reg, j, move
            insert_customer(adapter, rebuilt, values, pending[winner], chosen)
            if log is not None:
                log.append(dict(cust=int(pending[winner]), regret=float(highest),
                                move=tuple(int(x) for x in chosen)))
            pending.pop(winner)
    else:
        for cust in mask:
            _guard(deadline)
            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat,
                                     regret=False, deadline=deadline)
            insert_customer(adapter, rebuilt, values, cust, move)
            if log is not None:
                log.append(dict(cust=int(cust), regret=float(reg),
                                move=tuple(int(x) for x in move)))
    return tuple(tuple(r) for r in rebuilt if r)


# ------------------------------------------------------------------ 批搜索
def repair_batch(adapter, state, best, group, heat=None, seed=0, seconds=0.1,
                 tolerance=1.5, related=False, regret=False, iteration_cap=0,
                 deadline=None, hard_deadline=None):
    """锁版 repair_batch：在当前/记录解上做时间片内的掩码重构。

    返回 dict(state=..., best=..., iterations=..., improvements=..., calls=...)
    """
    stop = time.perf_counter() + seconds
    if deadline is not None:
        stop = min(stop, deadline)
    rng = random.Random(seed)
    state = [list(r) for r in state]
    record = [list(r) for r in best]
    u = solution_value(adapter, state)
    best_u = solution_value(adapter, record)
    order = list(group)
    iterations = improvements = calls = 0
    while time.perf_counter() < stop and (iteration_cap <= 0 or iterations < iteration_cap):
        rng.shuffle(order)
        k = min(len(order), 10 if regret else 6 + rng.randrange(12))
        mask = order[:k]
        if related and iterations % 2 == 0 and k >= 2:
            anchor = order[0]
            priority = []
            for i in order:
                if i == anchor:
                    continue
                if heat is not None:
                    affinity = heat[anchor][i] + heat[i][anchor]
                    if not math.isfinite(affinity):
                        affinity = NEG_INF
                else:
                    affinity = -2.0 * adapter.inst.dist[anchor][i]
                priority.append((affinity, i))
            priority.sort(reverse=True)
            mask = [anchor]
            for _ in range(k - 1):
                pos = int(rng.random() * rng.random() * len(priority))
                mask.append(priority.pop(pos)[1])
            rng.shuffle(mask)
        try:
            nxt = reconstruct(adapter, state, mask, heat=heat, regret=regret,
                              deadline=hard_deadline)
        except _TimeUp:
            break
        calls += 1
        new_u = u if nxt is None else solution_value(adapter, nxt)
        if new_u > best_u + EPS:
            record = [list(r) for r in nxt]
            best_u = new_u
            improvements += 1
        if new_u >= u - EPS or (tolerance > 0 and new_u > best_u - 6 * tolerance
                                and rng.random() < math.exp((new_u - u) / max(tolerance, 1e-9))):
            if nxt is not None:
                state = [list(r) for r in nxt]
                u = new_u
        iterations += 1
    return dict(state=[tuple(r) for r in state], best=[tuple(r) for r in record],
                iterations=iterations, improvements=improvements, calls=calls,
                state_value=u, best_value=best_u)


# ------------------------------------------------------------------ 兼容包装
def masked_repair(adapter, parent, removed, ranking, operator, rng=None, heat=None,
                  slot=False, deadline=None, log=None):
    """单次重构包装（供单元测试与诊断）：mask 顺序由 ranking 决定（贪心）。"""
    mask = [c for c in ranking if c in set(removed)]
    mask += [c for c in removed if c not in set(mask)]
    regret = operator == "regret2"
    _guard(deadline)
    out = reconstruct(adapter, [list(r) for r in parent], mask, heat=heat, regret=regret,
                      log=log, deadline=deadline)
    if out is None:
        return tuple(tuple(r) for r in parent), False
    ok, val, k = adapter.evaluate(out)
    if not ok:
        return tuple(tuple(r) for r in parent), False
    ok0, val0, k0 = adapter.evaluate(parent)
    if adapter.lexicographic:
        improved = (k, val) < (k0, val0)
    else:
        improved = val < val0 - 1e-9
    return out, improved
