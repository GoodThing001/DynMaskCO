# -*- coding: utf-8 -*-
"""标准任务掩码重构求解器（standard_vrp 独立模块）。

与冷链 pool_search 同构的搜索骨架，目标与评价完全替换为标准任务：
  - 40 客掩码（n<=100）或窗口掩码（n>100：掩码40客 + 预测窗口=仓库+被遮客+60近邻，见 P5_预登记）；
  - 三种候选排序（纯距离 / 原始热图 / 距离校准热图）× 两种重插算子（贪心 / regret-2）六组合试选；
  - 每 3 轮跨车顺序搜索 + 标准 Split 分车（CVRP 最小距离 / VRPTW 字典序 (K,D)）；
  - 接受：严格改善标准目标；60 秒预算。

此模块不导入任何冷链模块；神经网络输入按 P5_预登记第 4 节的中性编码构造。
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw,
                                                  split_cvrp, split_vrptw, better_lex,
                                                  route_schedule)


class _TimeUp(Exception):
    """预算耗尽：丢弃本轮未完成候选，立即退出搜索（不接纳部分解）。"""


def _guard(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise _TimeUp


def masked_repair(inst, routes, removed, ranking, operator, rng, objective="cvrp",
                  heat=None, slot=False, deadline=None):
    """一次掩码重构：按候选顺序逐个重插被遮客户。

    ranking: 客户重插顺序（list of ids）。
    operator: "greedy"（每客户选当前最优插位）或 "regret2"（先按后悔值排序再贪心）。
    heat: (n+1)×(n+1) 热图；非 None 时每个客户的候选插位按热图取前 8（冻结 topk=8 语义）。
    slot: True 时追加最短可行绕行插位保护（SLOT 语义）。
    返回 (new_routes, improved) —— 仅当目标严格改善时更新。
    """
    def candidate_positions(ri, r, cust):
        if heat is None:
            return list(range(len(r) + 1))
        # 插位评分：插在位置 pos（r[:pos]+[cust]+r[pos:]）后热图相邻边分之和
        scored = []
        for pos in range(len(r) + 1):
            seq = [0] + r + [0]
            a, b = seq[pos], seq[pos + 1]
            scored.append((heat[cust][a] + heat[cust][b], pos))
        if max(s for s, _ in scored) - min(s for s, _ in scored) < 1e-12:
            return list(range(len(r) + 1))  # 无热图信号（窗口外）→ 距离全位置评价
        scored.sort(key=lambda x: -x[0])
        chosen = [pos for _, pos in scored[:8]]
        if slot:
            # 最短可行绕行保护：几何绕行增量最小的位置（不足 8 时已在列表中）
            detour = []
            for pos in range(len(r) + 1):
                seq = [0] + r + [0]
                a, b = seq[pos], seq[pos + 1]
                detour.append((inst.dist[a][cust] + inst.dist[cust][b] - inst.dist[a][b], pos))
            detour.sort(key=lambda x: x[0])
            if detour[0][1] not in chosen:
                chosen.append(detour[0][1])
        return chosen
    if objective == "cvrp":
        # 路线局部可行性：被遮客户尚未重插，全解覆盖检查必然失败，
        # 因此只检查被修改路线的容量 + 局部距离增量（冷链接口语义的标准任务适配）。
        current = [list(r) for r in routes]
        if operator == "regret2":
            # 后悔值 = 次优插位增量 - 最优插位增量
            def order_key(cust):
                _guard(deadline)
                best, second = math.inf, math.inf
                for ri, r in enumerate(current):
                    if sum(inst.demands[i] for i in r) + inst.demands[cust] > inst.capacity + 1e-9:
                        continue
                    for pos in candidate_positions(ri, r, cust):
                        delta = _insert_delta(inst, r, pos, cust, rounded=True)
                        if delta < best - 1e-9:
                            second, best = best, delta
                        elif delta < second - 1e-9:
                            second = delta
                return second - best if best < math.inf else -math.inf
            remaining = sorted(removed, key=order_key, reverse=True)
        else:
            remaining = list(removed)
        for cust in remaining:
            _guard(deadline)
            best_delta, best_ri, best_pos = math.inf, -1, -1
            for ri, r in enumerate(current):
                if sum(inst.demands[i] for i in r) + inst.demands[cust] > inst.capacity + 1e-9:
                    continue
                for pos in candidate_positions(ri, r, cust):
                    delta = _insert_delta(inst, r, pos, cust, rounded=True)
                    if delta < best_delta - 1e-9:
                        best_delta, best_ri, best_pos = delta, ri, pos
            if best_ri >= 0:
                current[best_ri].insert(best_pos, cust)
            else:
                current.append([cust])  # 无可行位置 → 开新车
        out = tuple(tuple(r) for r in current if r)
        ok, d, _, _ = evaluate_cvrp(inst, out, rounded=True)
        _, d0, _, _ = evaluate_cvrp(inst, routes, rounded=True)
        return out, ok and d < d0 - 1e-9
    else:
        # VRPTW：路线局部可行性 = 被修改路线的可行时刻表；插入保持 K 不变，按距离增量择优。
        current = [list(r) for r in routes]
        for cust in (list(removed)):
            _guard(deadline)
            best = None  # (delta, ri, pos)
            for ri, r in enumerate(current):
                for pos in candidate_positions(ri, r, cust):
                    nr = tuple(r[:pos] + [cust] + r[pos:])
                    if route_schedule(inst, nr) is None:
                        continue
                    delta = _insert_delta(inst, r, pos, cust, rounded=False)
                    if best is None or delta < best[0] - 1e-9:
                        best = (delta, ri, pos)
            if best is not None:
                current[best[1]].insert(best[2], cust)
            else:
                current.append([cust])
        out = tuple(tuple(r) for r in current if r)
        ok, k, d, _, _ = evaluate_vrptw(inst, out)
        ok0, k0, d0, _, _ = evaluate_vrptw(inst, routes)
        return out, ok and better_lex(k, d, k0, d0)


def _insert_delta(inst, r, pos, cust, rounded=False):
    """在路线 r 的位置 pos 插入客户 cust 的局部距离增量（含仓库边）。"""
    seq = [0] + r + [0]
    a, b = seq[pos], seq[pos + 1]
    return (inst.edge_distance(a, cust, rounded) + inst.edge_distance(cust, b, rounded)
            - inst.edge_distance(a, b, rounded))


def standard_search(inst, seconds=60, seed=1, objective="cvrp", heatmap_fn=None,
                    neutral=None, split_enabled=True, slot=False, verbose=False):
    """60 秒标准掩码重构搜索骨架。

    split_enabled=False → 关闭跨车 Split 分车（MASK/SLOT/DIST/DIST_SLOT 语义）；
    slot=True → 最短可行绕行插位保护（即使热图 top-8 漏掉也保留该位置）。"""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    n = inst.dimension
    started = time.perf_counter()
    # 共同初解：最近邻见证（与冷链冻结初始器同构）
    routes = nn_witness(inst)
    if objective == "cvrp":
        _, d0, _, _ = evaluate_cvrp(inst, routes, rounded=True)
        best_d = d0
        better = lambda d: d < best_d - 1e-9
    else:
        _, k0, d0, _, _ = evaluate_vrptw(inst, routes)
        best_k, best_d = k0, d0

    def better_than(k, d):
        if objective == "cvrp":
            return better(d)
        return better_lex(k, d, best_k, best_d)

    def accept(routes_new):
        nonlocal best_d, best_k, best_routes
        if time.perf_counter() - started >= seconds - 0.05:
            return False  # 截止后不接纳（冻结语义：逾期候选拒绝）
        if objective == "cvrp":
            ok, d, _, _ = evaluate_cvrp(inst, routes_new, rounded=True)
            if ok and d < best_d - 1e-9:
                best_d, best_routes = d, routes_new
                return True
            return False
        ok, k, d, _, _ = evaluate_vrptw(inst, routes_new)
        if ok and better_lex(k, d, best_k, best_d):
            best_k, best_d, best_routes = k, d, routes_new
            return True
        return False

    best_routes = routes
    mask_size = min(40, n)
    epochs = 0
    cutoff = seconds - 0.8  # 末轮评估与收尾余量（在线预算保持 ≤60s）
    deadline = started + cutoff  # 轮内截止：任何未完成候选到达此点即丢弃
    while time.perf_counter() - started < cutoff:
        try:
            mask = nrng.choice(np.arange(1, n + 1), size=mask_size, replace=False).tolist()
            removed = set(mask)
            partial = tuple(tuple(i for i in r if i not in removed) for r in best_routes)
            partial = tuple(r for r in partial if r)
            # 每轮一次热图预测（复用冻结语义：一次前向供时段内多次重构）
            heat = None
            if heatmap_fn is not None:
                try:
                    heat = heatmap_fn(inst, partial, removed, neutral)
                except Exception:
                    heat = None
            # 三种排序 × 两种算子试选（标准目标）；heat=None 时三种排序退化为同距离序，去重
            rankings = ("distance", "heatmap", "calib") if heat is not None else ("distance",)
            candidates = []
            for ranking_name in rankings:
                order = rank_customers(inst, partial, removed, ranking_name, heat, rng)
                use_heat = heat if ranking_name in ("heatmap", "calib") else None
                for op in ("greedy", "regret2"):
                    new_routes, improved = masked_repair(inst, partial, removed, order, op, rng,
                                                         objective, heat=use_heat, slot=slot,
                                                         deadline=deadline)
                    if new_routes is not None:
                        if objective == "cvrp":
                            _, d, _, _ = evaluate_cvrp(inst, new_routes, rounded=True)
                            score = -d
                        else:
                            _, k, d, _, _ = evaluate_vrptw(inst, new_routes)
                            score = (-k, -d)
                        candidates.append((score, new_routes))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, chosen = candidates[0]
                accept(chosen)
            # 每 3 轮跨车顺序搜索 + Split（split_enabled=False 时关闭）
            if split_enabled and epochs % 3 == 0:
                split_search(inst, best_routes, heatmap_fn, neutral, rng, objective, accept)
            epochs += 1
        except _TimeUp:
            break
    if objective == "cvrp":
        _, d, _, _ = evaluate_cvrp(inst, best_routes, rounded=True)
        return dict(routes=best_routes, distance=d, vehicles=len(best_routes), epochs=epochs)
    _, k, d, _, _ = evaluate_vrptw(inst, best_routes)
    return dict(routes=best_routes, vehicles=k, distance=d, epochs=epochs)


def _sweep_witness(inst, k):
    """CVRP 车队约束见证：角度扫描 / FFD / NN 三种构造，取第一个 ≤k 组的容量可行解。"""
    n = inst.dimension
    cap = inst.capacity + 1e-9
    x0, y0 = inst.coords[0]

    def order_routes(groups):
        routes = []
        for g in groups:
            r, cur, left = [], 0, set(g)
            while left:
                nxt = min(left, key=lambda i: inst.dist[cur][i])
                r.append(nxt)
                left.remove(nxt)
                cur = nxt
            routes.append(tuple(r))
        return routes

    def feasible(routes):
        if len(routes) > k:
            return False
        for r in routes:
            if sum(inst.demands[i] for i in r) > cap:
                return False
        return len({i for r in routes for i in r}) == n

    # 构造 1：角度扫描，容量满即开新组
    ang = {i: math.atan2(inst.coords[i][1] - y0, inst.coords[i][0] - x0) for i in range(1, n + 1)}
    groups, cur, load = [], [], 0.0
    for c in sorted(range(1, n + 1), key=lambda i: ang[i]):
        d = inst.demands[c]
        if cur and load + d > cap:
            groups.append(cur)
            cur, load = [], 0.0
        cur.append(c)
        load += d
    if cur:
        groups.append(cur)
    routes = order_routes(groups)
    if feasible(routes):
        return tuple(routes)

    # 构造 2：BFD（需求降序、最小剩余容量适配）——bin-packing 更强
    for key in (lambda i: -inst.demands[i], lambda i: inst.demands[i]):
        groups, loads = [], []
        ok_bfd = True
        for c in sorted(range(1, n + 1), key=key):
            d = inst.demands[c]
            best_gi, best_slack = None, None
            for gi, gl in enumerate(loads):
                if gl + d <= cap and (best_slack is None or (cap - gl - d) < best_slack):
                    best_gi, best_slack = gi, cap - gl - d
            if best_gi is None:
                groups.append([c])
                loads.append(d)
            else:
                groups[best_gi].append(c)
                loads[best_gi] += d
        routes = order_routes(groups)
        if feasible(routes):
            return tuple(routes)

    # 兜底：组数 > k 时合并最小的两组（容量允许）
    while len(routes) > k:
        routes.sort(key=lambda r: sum(inst.demands[i] for i in r))
        merged = routes[0] + routes[1]
        if sum(inst.demands[i] for i in merged) > cap:
            break
        routes = [merged] + routes[2:]
    return tuple(routes)


def nn_witness(inst):
    """初始车队（尊重车队限额）。

    - CVRP 带 max_vehicles（Uchoa X 官方固定车队 k）：角度扫描分组（每组建容量可行段），
      组内最近邻排序，必要时合并最小两组压到 k 组——O(n log n)，避免大规模实例的
      逐客户插位分布开销；
    - VRPTW 带 max_vehicles（Solomon 25）：最近邻链达限额后剩余客户插入现有路线可行位置，
      直接插入不可行时深度-1 弹出修复，仍不可行强制插入（覆盖完整优先）。
    """
    n = inst.dimension
    remaining = set(range(1, n + 1))
    routes = []
    limit = getattr(inst, "max_vehicles", None)
    is_tw = isinstance(inst, VRPTWInstance)
    if limit is not None and not is_tw:
        return _sweep_witness(inst, limit)
    while remaining:
        if limit is not None and len(routes) == limit:
            lists = [list(r) for r in routes]

            def route_feasible(nr):
                if is_tw:
                    return route_schedule(inst, tuple(nr)) is not None
                return sum(inst.demands[i] for i in nr) <= inst.capacity + 1e-9

            def ins_delta(r, pos, c):
                return _insert_delta(inst, r, pos, c, rounded=(not is_tw))

            def feas_pos(c, skip=None):
                best = None
                for ri, r in enumerate(lists):
                    if ri == skip:
                        continue
                    for pos in range(len(r) + 1):
                        if not route_feasible(r[:pos] + [c] + r[pos:]):
                            continue
                        delta = ins_delta(r, pos, c)
                        if best is None or delta < best[0] - 1e-9:
                            best = (delta, ri, pos)
                return best

            progress = True
            while remaining and progress:
                progress = False
                for cust in sorted(remaining):
                    hit = feas_pos(cust)
                    if hit is None:
                        # 深度-1 弹出：移出 x 插入 cust，再把 x 放回可行位置
                        for ri, r in enumerate(lists):
                            for q, x in enumerate(r):
                                without = r[:q] + r[q + 1:]
                                for p in range(len(without) + 1):
                                    nr = without[:p] + [cust] + without[p:]
                                    if not route_feasible(nr):
                                        continue
                                    trial = lists[:ri] + [list(nr)] + lists[ri + 1:]
                                    home = None
                                    for rj, r2 in enumerate(trial):
                                        for pos2 in range(len(r2) + 1):
                                            if not route_feasible(r2[:pos2] + [x] + r2[pos2:]):
                                                continue
                                            delta = ins_delta(r2, pos2, x)
                                            if home is None or delta < home[0] - 1e-9:
                                                home = (delta, rj, pos2)
                                    if home is not None:
                                        lists[:] = trial
                                        lists[home[1]].insert(home[2], x)
                                        remaining.remove(cust)
                                        progress = True
                                        hit = "eject"
                                        break
                                if hit is not None:
                                    break
                            if hit is not None:
                                break
                    if hit is None:
                        # 强制插入绕行最小位置（覆盖完整优先；极少见）
                        best = None
                        for ri, r in enumerate(lists):
                            for pos in range(len(r) + 1):
                                delta = ins_delta(r, pos, cust)
                                if best is None or delta < best[0] - 1e-9:
                                    best = (delta, ri, pos)
                        if best is not None:
                            lists[best[1]].insert(best[2], cust)
                            remaining.remove(cust)
                            progress = True
                        continue
                    if hit != "eject":
                        lists[hit[1]].insert(hit[2], cust)
                        remaining.remove(cust)
                        progress = True
            routes = [tuple(r) for r in lists if r]
            break  # 已到限额；剩余（若有）不再追加路线，保持 K≤limit
        route = []
        cur = 0
        while remaining:
            nxt = min(remaining, key=lambda i: inst.dist[cur][i])
            trial = route + [nxt]
            if is_tw:
                if route_schedule(inst, tuple(trial)) is None:
                    break
            else:
                if sum(inst.demands[i] for i in trial) > inst.capacity + 1e-9:
                    break
            route.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        routes.append(tuple(route))
    return tuple(routes)


def rank_customers(inst, partial, removed, ranking_name, heat, rng):
    """客户重插顺序：distance=按距离增量最小者优先；heatmap/calib=按热图边分。"""
    if ranking_name == "distance" or heat is None:
        return sorted(removed, key=lambda i: min((inst.dist[i][j] for j in range(inst.dimension + 1) if j != i), default=0.0))
    return sorted(removed, key=lambda i: -max((heat[i][j] for j in range(inst.dimension + 1) if j != i), default=0.0))


def split_search(inst, routes, heatmap_fn, neutral, rng, objective, accept):
    """跨车顺序搜索：选相近路线组，热图/距离提出访问顺序，标准 Split 分车。"""
    n = inst.dimension
    if len(routes) < 2:
        return
    # 选 2 条总客户 <=24 的相近路线
    idx = list(range(len(routes)))
    rng.shuffle(idx)
    for a in idx:
        for b in idx:
            if a >= b:
                continue
            group = list(routes[a]) + list(routes[b])
            if len(group) > 24:
                continue
            seq = propose_order(inst, group, heatmap_fn, neutral, rng)
            if objective == "cvrp":
                out = split_cvrp(inst, seq, rounded=True)
                if out:
                    new = tuple(r for i, r in enumerate(routes) if i not in (a, b)) + out["routes"]
                    accept(new)
            else:
                out = split_vrptw(inst, seq)
                if out:
                    new = tuple(r for i, r in enumerate(routes) if i not in (a, b)) + out["routes"]
                    accept(new)
            return


def propose_order(inst, group, heatmap_fn, neutral, rng):
    """提出访问顺序：热图最近邻链或距离最近邻链（固定规则，不事后调参）。"""
    if heatmap_fn is not None:
        try:
            heat = heatmap_fn(inst, tuple(), set(group), neutral)
            seq, cur = [], 0
            left = set(group)
            while left:
                nxt = max(left, key=lambda i: heat[cur][i])
                seq.append(nxt)
                left.remove(nxt)
                cur = nxt
            return tuple(seq)
        except Exception:
            pass
    seq, cur = [], 0
    left = set(group)
    while left:
        nxt = min(left, key=lambda i: inst.dist[cur][i])
        seq.append(nxt)
        left.remove(nxt)
        cur = nxt
    return tuple(seq)
