"""增强修复验证（不训练）：R 完整修复后的有限两客户交换（允许跨车）。

只增加这一种算子，不引入 beam / 随机重启 / 新目标：给定 R（regret-2）完整重建出的候选计划
与 mask 客户集合，枚举 mask 客户之间的两客户交换（跨车 / 同车），对每个交换做完整分区认证
+ `J_vis` 评价，返回更优的完整计划。保留未掩码客户的车辆归属与相对顺序；不触碰 committed、
已入舱货物或保护车辆（这些客户本就不在 mask 内，交换只移动 mask 客户）。所有评价计入预算。

两个验证问题：
  1. 固定状态：同一公开状态下，是否存在优于 R、且能用合法插入步骤表达的局部完整修复？
  2. 闭环：这种改善是否在真实动态闭环中仍然存在？

`cc_lns_swap_search(swap=False)` 与 `cc_lns_replanner.cc_lns_search` 行为一致（同一 mask 序列、
同一认证/评价/接受规则），仅多了 `swap` / `n_rounds` / `max_attempts` 参数。`swap=True` 时在
每次 R 重建候选之后，枚举该 mask 的两客户交换作为额外候选，共用去重/认证/评价/接受与预算。
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans, plan_hash
from dynmaskco_cc_context import decision_pool_from_vehicles
from visible_state import build_visible_state, evaluate_visible_plan
from cc_lns_replanner import (REQUEST_SEQUENCE, N_ROUNDS, _select_one_mask, _plan_suffixes,
                              _partition_ok, _reconstruct)


def swap_two_customers(plans, c1, c2):
    """交换 c1/c2 在各自 suffix 中的位置（跨车或同车）。返回新 plans；任一缺失返回 None。

    只移动这两个客户，其余客户（含未掩码客户）的车辆归属与相对顺序不变。
    """
    c1, c2 = int(c1), int(c2)
    loc = {}
    for vid, p in plans.items():
        suf = list(p.suffix)
        for i, x in enumerate(suf):
            if int(x) == c1:
                loc[c1] = (vid, i)
            elif int(x) == c2:
                loc[c2] = (vid, i)
    if c1 not in loc or c2 not in loc:
        return None
    out = {}
    for vid, p in plans.items():
        suf = list(p.suffix)
        if loc[c1][0] == vid:
            suf[loc[c1][1]] = c2
        if loc[c2][0] == vid:
            suf[loc[c2][1]] = c1
        out[vid] = type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                           tuple(suf))
    return out


def iter_swap_pairs(mask_customers):
    """mask 客户之间的所有两客户交换对（确定性顺序：按 (小 id, 大 id) 升序）。"""
    cs = sorted(int(c) for c in mask_customers)
    return [(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]


def _suffix_tw_ok(vis, plan_suffixes):
    """检查候选 suffix 的 TW + 返仓 TW（不含容量；容量由 evaluate_visible_plan 抛错捕获）。

    交换只移动 mask 客户，可能破坏时间窗；evaluate_visible_plan 的 suffix 段不查 TW（只通过
    transition_segment 查容量），故这里补一个时间推进的 TW 检查，拒绝晚到/晚返仓的交换。
    """
    for k in range(len(vis.vehicles)):
        v = vis.vehicles[k]
        if v.status in ('closed', 'returning'):
            continue
        cur = vis.idx(v.current_node)
        t = float(v.ready_time)
        if v.committed_next > 0:
            cur = vis.idx(v.committed_next)
            t = float(v.committed_finish)
        for c in plan_suffixes.get(v.vid, ()):
            if int(c) <= 0:
                continue
            ci = vis.idx(int(c))
            d = float(vis.dist_mat[cur][ci])
            arrive = t + d / vis.tw_speed
            if arrive > float(vis.tw_end[ci]) + 1e-6:
                return False
            t = max(arrive, float(vis.tw_start[ci])) + float(vis.service_time[ci])
            cur = ci
        if t + float(vis.dist_mat[cur][0]) / vis.tw_speed > float(vis.depot_tw_end) + 1e-6:
            return False
    return True


def _event(clock, env, J0, best_J, stats):
    return {'event': int(getattr(env, 'event_id', -1)), 'clock': float(clock),
            'J0': float(J0), 'J_best': float(best_J), 'n_attempts': stats['n_attempts'],
            'n_swap': stats['n_swap'], 'n_improve': stats['n_improve'],
            'stop_reason': stats['stop_reason'], 'elapsed_s': stats['elapsed_s'],
            'protected_ids': stats.get('protected_ids', []),
            'P0_feasible': bool(stats.get('P0_feasible', True))}


def cc_lns_swap_search(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
                       deferred, contract, budget_s, seed=0, protected=frozenset(),
                       swap=False, n_rounds=N_ROUNDS, max_attempts=None):
    """R 搜索（swap=False）或 R + 两客户交换（swap=True）。返回 (best, best_J, stats, event)。

    stats 额外含 best_D / best_Q / best_E（最终最佳计划的 D/Q/E）。
    swap=True 时，每个 R 重建候选之后枚举该 mask 的两客户交换作为额外候选，受 budget_s 与
    max_attempts（总候选上限，重建 + 交换）约束；交换候选走同一去重/认证/评价/接受路径。
    """
    objective = contract.objective
    mutable_ids = {int(v) for v in replan_ids} - set(protected)
    pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
    t0 = time.perf_counter()
    vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                              served_mask, visible_ids, replan_ids, deferred)
    P0 = build_vehicle_plans(env, inst_idx, vehicles)
    r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), contract, objective)
    J0 = r0.J_vis
    P0_feasible = bool(r0.feasible) and bool(r0.finite)
    best = P0
    best_J = J0
    best_D, best_Q, best_E = r0.D, r0.Q, r0.E
    cap = (n_rounds * len(REQUEST_SEQUENCE)) if max_attempts is None else int(max_attempts)

    stats = {'n_attempts': 0, 'n_swap': 0, 'n_reconstruct_fail': 0, 'n_feasible': 0,
             'n_duplicate': 0, 'n_unique': 0, 'n_eval': 0, 'n_eval_infeasible': 0,
             'n_partition_fail': 0, 'n_improve': 0, 'J0': J0, 'J_best': J0,
             'best_D': best_D, 'best_Q': best_Q, 'best_E': best_E,
             'P0_feasible': P0_feasible, 'P0_finite': bool(r0.finite),
             'protected_ids': sorted(int(x) for x in protected),
             'stop_reason': 'normal', 'elapsed_s': 0.0}
    if not P0_feasible:
        stats['stop_reason'] = 'P0_invalid'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _event(clock, env, J0, best_J, stats)
    if not pool:
        stats['stop_reason'] = 'empty_pool'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _event(clock, env, J0, best_J, stats)

    seen = {plan_hash(P0)}
    eval_cache = {}
    holder = {'best': best, 'best_J': best_J, 'best_D': best_D, 'best_Q': best_Q,
              'best_E': best_E}

    def _consider(cand):
        ch = plan_hash(cand)
        if ch in seen:
            stats['n_duplicate'] += 1
            return
        seen.add(ch)
        stats['n_unique'] += 1
        ok, _ = _partition_ok(env, inst_idx, clock, vehicles, cand, deferred, served_mask)
        if not ok:
            stats['n_partition_fail'] += 1
            return
        # 交换可能破坏 TW/返仓；先做时间窗预检（容量由 evaluate_visible_plan 抛错捕获）
        if not _suffix_tw_ok(vis, _plan_suffixes(cand)):
            stats['n_eval_infeasible'] += 1
            return
        if ch in eval_cache:
            r = eval_cache[ch]
        else:
            try:
                r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract, objective)
            except ValueError:
                # 容量等硬违反（transition_segment 抛错）→ 视为不可行
                stats['n_eval_infeasible'] += 1
                return
            stats['n_eval'] += 1
            if not r.finite or not r.feasible:
                stats['n_eval_infeasible'] += 1
                return
            eval_cache[ch] = r
        if r.J_vis < holder['best_J'] - 1e-9:
            holder['best'] = cand
            holder['best_J'] = r.J_vis
            holder['best_D'] = r.D
            holder['best_Q'] = r.Q
            holder['best_E'] = r.E
            stats['n_improve'] += 1

    for _round in range(n_rounds):
        for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
            if time.perf_counter() - t0 > budget_s:
                stats['stop_reason'] = 'budget'
                break
            if stats['n_attempts'] >= cap:
                stats['stop_reason'] = 'candidate_cap'
                break
            rng_seed = seed * 10000 + _round * 100 + req_idx
            mask = _select_one_mask(env, inst_idx, holder['best'], pool, mask_size, rng_seed)
            if not mask:
                continue
            partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time,
                                    p.anchor_load, tuple(x for x in p.suffix if x not in mask))
                       for vid, p in holder['best'].items()}
            cand = _reconstruct(env, inst_idx, partial, mask, mutable_ids)
            stats['n_attempts'] += 1
            if cand is None:
                stats['n_reconstruct_fail'] += 1
                continue
            stats['n_feasible'] += 1
            _consider(cand)
            if swap:
                for c1, c2 in iter_swap_pairs(mask):
                    if time.perf_counter() - t0 > budget_s:
                        stats['stop_reason'] = 'budget'
                        break
                    if stats['n_attempts'] >= cap:
                        stats['stop_reason'] = 'candidate_cap'
                        break
                    cand2 = swap_two_customers(cand, c1, c2)
                    if cand2 is None:
                        continue
                    stats['n_attempts'] += 1
                    stats['n_swap'] += 1
                    _consider(cand2)
        if stats['stop_reason'] in ('budget', 'candidate_cap'):
            break
    if stats['stop_reason'] == 'normal' and stats['n_attempts'] >= cap:
        stats['stop_reason'] = 'candidate_cap'
    stats['J_best'] = holder['best_J']
    stats['best_D'] = holder['best_D']
    stats['best_Q'] = holder['best_Q']
    stats['best_E'] = holder['best_E']
    stats['elapsed_s'] = time.perf_counter() - t0
    return holder['best'], holder['best_J'], stats, _event(clock, env, J0, holder['best_J'],
                                                           stats)


class CCLNSwapReplanner(JF1HRepairReplanner):
    """R + 两客户交换：JF1-H-F baseline 后做预算内的非学习联合搜索（含 swap），写回最佳计划。

    与 CCLNSReplanner 相同（guard_reserved=True 为 N0-R 预留保护），仅搜索核心换成
    cc_lns_swap_search。swap=True 表示在 R 重建后加入两客户交换。
    """

    def __init__(self, budget_s=1.0, contract=None, seed=0, capacity=50.0, num_vehicles=25,
                 guard_reserved=False, swap=True, n_rounds=N_ROUNDS, max_attempts=None):
        super().__init__(slack_vehicles=1)
        self.budget_s = float(budget_s)
        self.contract = contract
        self.seed = int(seed)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.guard_reserved = bool(guard_reserved)
        self.swap = bool(swap)
        self.n_rounds = int(n_rounds)
        self.max_attempts = max_attempts
        self.online_stats = defaultdict(int)
        self.event_log = []

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = defaultdict(int)
            self.event_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        t_replanner = time.perf_counter()
        reserved_ids = set()
        if self.guard_reserved:
            idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'),
                              reverse=True)
            reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        self._reset(inst_idx)
        contract = self.contract if self.contract is not None else env.coldchain_contract
        if contract is None:
            return
        protected = set()
        if self.guard_reserved and reserved_ids:
            P0_view = build_vehicle_plans(env, inst_idx, vehicles)
            for vid in reserved_ids:
                v = vehicles[vid]
                p = P0_view.get(vid)
                if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                    protected.add(vid)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids))
        mutable_ids -= protected
        best, best_J, stats, ev = cc_lns_swap_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, contract, self.budget_s, seed=self.seed,
            protected=protected, swap=self.swap, n_rounds=self.n_rounds,
            max_attempts=self.max_attempts)
        has_future = env.has_future_reveal(inst_idx, clock, served_mask)
        ok, _detail = _partition_ok(env, inst_idx, clock, vehicles, best,
                                    self.deferred_customers, served_mask)
        changed = 0
        if ok:
            for v in vehicles:
                if v.vehicle_id not in mutable_ids:
                    continue
                p = best.get(v.vehicle_id)
                if p is None:
                    continue
                new_suffix = (list(p.suffix) + [0] if p.suffix
                              else ([] if (p.anchor_node != 0 and has_future) else [0]))
                if new_suffix != v.mutable_suffix:
                    changed += 1
                v.mutable_suffix = new_suffix
            self.sync_deferred_from_vehicles(vehicles)
        for k, v in stats.items():
            if k == 'P0_feasible':
                self.online_stats['n_P0_invalid'] += (0 if v else 1)
            elif k in ('J0', 'J_best', 'stop_reason', 'elapsed_s', 'P0_finite', 'protected_ids',
                       'best_D', 'best_Q', 'best_E'):
                self.online_stats[k] = v
            else:
                self.online_stats[k] += v
        self.online_stats['n_vehicles_changed'] += changed
        self.online_stats['n_partition_fail_writeback'] += (0 if ok else 1)
        ev['replanner_elapsed_s'] = time.perf_counter() - t_replanner
        ev['writeback_changed'] = changed
        ev['writeback_partition_ok'] = ok
        ev['reserved_ids'] = sorted(int(x) for x in reserved_ids)
        ev['protected_ids'] = sorted(int(x) for x in protected)
        ev['reserved_enabled'] = sorted(int(x) for x in reserved_ids - protected)
        self.event_log.append(ev)
