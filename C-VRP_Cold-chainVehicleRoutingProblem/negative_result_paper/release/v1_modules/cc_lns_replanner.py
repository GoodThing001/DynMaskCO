"""N0-r1：CC-LNS-RH —— 非学习联合修复搜索参照（deployable cold-chain LNS）。

以 JF1-H-F 计划为初始解，在可变车队范围内做「移除整组 mask 客户 → regret-2 重建」的
完整计划迭代，用可见场景评价 `J_vis` 比较候选，最终只写回当前最佳完整计划。

r1 修正（相对 N0 初版）：
  - 固定请求序列 8+8（每轮 (2,2,2,4,4,4,8,8)），不再用 round_cands 漏递增；
  - 每个候选过完整分区认证（committed ⊎ suffix ⊎ deferred），携带自己的 deferred；
  - 未派车且无任务的车保持未派车（增量 0，预冷只在真正出车时收一次，见 visible_state）；
  - 按 plan_hash 去重并缓存评价分数，seen 初始含 KEEP；计数分尝试/重建成功/独特非KEEP/评价；
  - 计时从视图构造前起算（整 replanner 与内部搜索分别记录）；记录停止原因；
  - P0 的 feasible/finite 显式记录，不静默伪装 KEEP 成功。

规模（固定）：mask 2/4/8（序列 2,2,2,4,4,4,8,8）、两轮、总 16 次重建尝试、预算 1s/4s 两档。
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action,
                             plan_hash)
from dynmaskco_cc_context import (validate_full_partition, mask_candidate_pool,
                                  validate_mask_scope)
from visible_state import build_visible_state, evaluate_visible_plan

REQUEST_SEQUENCE = (2, 2, 2, 4, 4, 4, 8, 8)   # 每轮 8 次请求
N_ROUNDS = 2
MAX_ATTEMPTS = N_ROUNDS * len(REQUEST_SEQUENCE)   # 16


def _detour(env, inst_idx, plans, customer):
    for p in plans.values():
        suf = list(p.suffix)
        if int(customer) in suf:
            i = suf.index(int(customer))
            pred = int(p.anchor_node) if i == 0 else int(suf[i - 1])
            succ = 0 if i == len(suf) - 1 else int(suf[i + 1])
            return (float(env.dist_mat[inst_idx, pred, customer])
                    + float(env.dist_mat[inst_idx, customer, succ])
                    - float(env.dist_mat[inst_idx, pred, succ]))
    return 0.0


def _select_one_mask(env, inst_idx, plans, pool, mask_size, seed):
    """按绕行量降序 + 种子扰动选一个 mask（最多 mask_size 客户）。"""
    k = min(mask_size, len(pool))
    if k == 0:
        return []
    ranked = sorted(pool, key=lambda c: (-_detour(env, inst_idx, plans, c), int(c)))
    rng = np.random.default_rng(seed)
    cand_pool = ranked[: min(2 * k, len(ranked))]
    idx = rng.choice(len(cand_pool), size=k, replace=False)
    return [int(cand_pool[i]) for i in sorted(idx)]


def _reconstruct(env, inst_idx, partial, mask_customers, mutable_ids):
    """regret-2 重建：把 mask 客户整体重新插入 partial。返回完整 plan 或 None（不可行）。"""
    P = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
         for vid, p in partial.items()}
    remaining = list(mask_customers)
    while remaining:
        best_regret = -float('inf')
        best_c = None
        best_action = None
        for c in remaining:
            cands, _ = enumerate_actions_from_plans(env, inst_idx, P, c,
                                                    allowed_vehicle_ids=mutable_ids)
            feas = [x for x in cands if x.feasible]
            if not feas:
                continue
            feas.sort(key=lambda x: (x.incremental_distance, x.action.action_id()))
            best_cost = feas[0].incremental_distance
            second_cost = (feas[1].incremental_distance if len(feas) > 1 else best_cost + 1e6)
            regret = second_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_c = c
                best_action = feas[0].action
        if best_c is None:
            return None
        P = apply_action(P, best_action, allowed_vehicle_ids=mutable_ids)
        remaining.remove(best_c)
    return P


def _plan_suffixes(plans):
    return {vid: tuple(int(x) for x in p.suffix if int(x) != 0) for vid, p in plans.items()}


def _partition_ok(env, inst_idx, clock, vehicles, cand, deferred, served_mask):
    committed = {int(v.committed_next) for v in vehicles
                 if v.status == 'committed' and v.committed_next not in (None, 0)}
    universe = [int(c) for c in range(1, env.num_nodes)
                if env.demands[inst_idx, c] > 0 and not served_mask[int(c)]
                and env.reveal_time[inst_idx, c] <= clock + 1e-6
                and int(c) not in committed]
    suffix_set = {int(x) for p in cand.values() for x in p.suffix}
    trial_deferred = {int(c) for c in deferred if c not in suffix_set and c not in committed}
    ok, detail = validate_full_partition(
        committed, cand, trial_deferred, universe, served_mask=served_mask,
        future_fn=lambda c: env.reveal_time[inst_idx, c] > clock + 1e-6)
    return ok, detail


def cc_lns_search(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
                  deferred, contract, budget_s, seed=0, protected=frozenset()):
    """返回 (best_plan, best_J, stats, event_record)。protected 为禁止 LNS 启用的预留车。"""
    objective = contract.objective
    mutable_ids = {int(v) for v in replan_ids} - set(protected)
    pool = mask_candidate_pool(vehicles, served_mask, visible_ids, protected=protected)
    t0 = time.perf_counter()
    vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                              served_mask, visible_ids, replan_ids, deferred)
    P0 = build_vehicle_plans(env, inst_idx, vehicles)
    r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), contract, objective)
    J0 = r0.J_vis
    P0_feasible = bool(r0.feasible) and bool(r0.finite)
    best = P0
    best_J = J0

    stats = {'n_attempts': 0, 'n_reconstruct_fail': 0, 'n_feasible': 0, 'n_duplicate': 0,
             'n_unique': 0, 'n_eval': 0, 'n_eval_infeasible': 0, 'n_partition_fail': 0,
             'n_improve': 0, 'J0': J0, 'J_best': J0, 'P0_feasible': P0_feasible,
             'P0_finite': bool(r0.finite), 'protected_ids': sorted(int(x) for x in protected),
             'stop_reason': 'normal', 'elapsed_s': 0.0}
    if not P0_feasible:
        # P0 评分无效：停止本事件可选搜索，保留 baseline 输出及其 deferred（不拿部分成本比较）
        stats['stop_reason'] = 'P0_invalid'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _event(clock, env, J0, best_J, stats)
    if not pool:
        stats['stop_reason'] = 'empty_pool'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _event(clock, env, J0, best_J, stats)

    seen = {plan_hash(P0)}   # KEEP 入 seen，KEEP-identical 候选不评价
    eval_cache = {}
    for _round in range(N_ROUNDS):
        for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
            if time.perf_counter() - t0 > budget_s:
                stats['stop_reason'] = 'budget'
                break
            rng_seed = seed * 10000 + _round * 100 + req_idx
            mask = _select_one_mask(env, inst_idx, best, pool, mask_size, rng_seed)
            if not mask:
                continue
            validate_mask_scope(vehicles, served_mask, visible_ids, best, mask, mutable_ids,
                                protected=protected)
            partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time,
                                    p.anchor_load, tuple(x for x in p.suffix if x not in mask))
                       for vid, p in best.items()}
            cand = _reconstruct(env, inst_idx, partial, mask, mutable_ids)
            stats['n_attempts'] += 1
            if cand is None:
                stats['n_reconstruct_fail'] += 1
                continue
            stats['n_feasible'] += 1
            ch = plan_hash(cand)
            if ch in seen:
                stats['n_duplicate'] += 1
                continue
            seen.add(ch)
            stats['n_unique'] += 1
            ok, _detail = _partition_ok(env, inst_idx, clock, vehicles, cand, deferred,
                                        served_mask)
            if not ok:
                stats['n_partition_fail'] += 1
                continue
            if ch in eval_cache:
                J = eval_cache[ch]
            else:
                r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract, objective)
                stats['n_eval'] += 1
                if not r.finite or not r.feasible:
                    stats['n_eval_infeasible'] += 1
                    continue
                J = r.J_vis
                eval_cache[ch] = J
            if J < best_J - 1e-9:
                best = cand
                best_J = J
                stats['n_improve'] += 1
        if stats['stop_reason'] == 'budget':
            break
    if stats['stop_reason'] == 'normal' and stats['n_attempts'] >= MAX_ATTEMPTS:
        stats['stop_reason'] = 'candidate_cap'
    stats['J_best'] = best_J
    stats['elapsed_s'] = time.perf_counter() - t0
    return best, best_J, stats, _event(clock, env, J0, best_J, stats)


def _event(clock, env, J0, best_J, stats):
    return {'event': int(getattr(env, 'event_id', -1)), 'clock': float(clock),
            'J0': float(J0), 'J_best': float(best_J), 'n_attempts': stats['n_attempts'],
            'n_improve': stats['n_improve'], 'stop_reason': stats['stop_reason'],
            'elapsed_s': stats['elapsed_s'], 'protected_ids': stats.get('protected_ids', []),
            'P0_feasible': bool(stats.get('P0_feasible', True))}


class CCLNSReplanner(JF1HRepairReplanner):
    """CC-LNS-RH：JF1-H-F baseline 后做预算内的非学习联合搜索，写回最佳完整计划。

    guard_reserved=True（N0-R）：禁止 LNS 启用 JF1-H-F 本事件仍保留的 idle 预留车。
    """

    def __init__(self, budget_s=1.0, contract=None, seed=0, capacity=50.0, num_vehicles=25,
                 guard_reserved=False):
        super().__init__(slack_vehicles=1)
        self.budget_s = float(budget_s)
        self.contract = contract
        self.seed = int(seed)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.guard_reserved = bool(guard_reserved)
        self.online_stats = defaultdict(int)
        self.event_log = []

    def _reset(self, inst_idx):
        if not hasattr(self, '_inst') or self._inst != int(inst_idx):
            self._inst = int(inst_idx)
            self.online_stats = defaultdict(int)
            self.event_log = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        t_replanner = time.perf_counter()
        # 1. 按 JF1-H-F 规则识别预留车（最高 vehicle_id 的 idle 车，slack_vehicles=1）
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
        # 2. 保护集合 = 预留车中仍未派车、仍在 depot、P0 无客户任务者（baseline 已启用则不再保留）
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
        best, best_J, stats, ev = cc_lns_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, contract, self.budget_s, seed=self.seed,
            protected=protected)
        # 写回（WAIT/RETURN 语义；保护车不改写，保持 P0 指令）
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
            elif k in ('J0', 'J_best', 'stop_reason', 'elapsed_s', 'P0_finite', 'protected_ids'):
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
