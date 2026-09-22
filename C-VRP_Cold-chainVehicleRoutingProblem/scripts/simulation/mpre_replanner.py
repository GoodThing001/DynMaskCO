"""M-pre 在线重构：预训练 MaskCO CVRP 边 logits → 贪心重插（未微调对照）。

复用完整预训练 CVRP model（`models/mpre.py::load_cvrp_model`）。每个决策点：
  1. 构建可见节点索引 + 编码可见节点（coord+demand）→ H（一次）；
  2. 对每个 mask：构建部分计划邻接 A（0/1）→ timestep=|mask|/Nv → decode 出对称边 logits L；
  3. 整体移除 mask，贪心按「增删边评分」重插：
       score(c at pred→succ) = L[pred][c] + L[c][succ] − L[pred][succ]
     regret = 最佳−次佳（同 regret-2 的「最该先插」语义）；
  4. 完整认证 + J_vis 接受（沿用 cc_lns_replanner 的 _partition_ok / evaluate_visible_plan）。

这是「真实预训练重构」对照，不训练、不接随机头。见 MaskCO直接目标训练工作包.md 第二步。
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from jf1h_repair import JF1HRepairReplanner
from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action,
                             plan_hash)
from dynmaskco_cc_context import decision_pool_from_vehicles
from dynmaskco_cc_graph import (build_visible_index, build_plan_adjacency,
                                resolve_action_endpoints)
from visible_state import build_visible_state, evaluate_visible_plan
from cc_lns_replanner import (REQUEST_SEQUENCE, N_ROUNDS, _select_one_mask, _plan_suffixes,
                              _partition_ok)
from mpre import load_cvrp_model, encode_cvrp, decode_edge_logits


def _plan_copy(plans):
    return {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
            for vid, p in plans.items()}


def retained_edge_ratio(A0, A_partial):
    """保留边比例 = |A0 与 A_partial 共有边| / |A0 边|（上三角，去对角）。原邻接无边时 0。

    对齐原始 CVRP 训练 keep_prob=timestep 语义（保留比例，不是删除比例）。
    """
    u0 = np.triu(np.asarray(A0), k=1) > 0.5
    up = np.triu(np.asarray(A_partial), k=1) > 0.5
    n_orig = int(u0.sum())
    if n_orig == 0:
        return 0.0
    n_retained = int((u0 & up).sum())
    return n_retained / n_orig


def mpre_reconstruct(env, inst_idx, partial, mask, mutable_ids, L, node_to_local):
    """贪心按「增删边评分」重插 mask 客户。返回 (plan, steps) 或 (None, [])。

    评分按真实增删边集合：增两条边 (pred→c, c→succ)；仅当 pred≠succ 时删 pred→succ 边。
    空车开路线 pred=succ=depot 无自环可删，对角掩码哨兵不进评分算术。
    """
    P = _plan_copy(partial)
    remaining = list(mask)
    steps = []
    while remaining:
        best_regret = -float('inf')
        best_c = None
        best_action = None
        for c in remaining:
            cands, _ = enumerate_actions_from_plans(env, inst_idx, P, int(c),
                                                    allowed_vehicle_ids=mutable_ids)
            feas = [x for x in cands if x.feasible]
            if not feas:
                continue
            scored = []
            for x in feas:
                a = x.action
                e, _v = resolve_action_endpoints(a.customer, a.slot, a.predecessor,
                                                 a.successor, P, node_to_local)
                ci, pi, si = int(e[0]), int(e[2]), int(e[3])
                s = float(L[pi, ci] + L[ci, si])
                if pi != si:
                    s -= float(L[pi, si])
                scored.append((s, a))
            scored.sort(key=lambda t: -t[0])
            best_s = scored[0][0]
            second_s = scored[1][0] if len(scored) > 1 else best_s - 1e6
            regret = best_s - second_s
            if regret > best_regret:
                best_regret = regret
                best_c = c
                best_action = scored[0][1]
        if best_c is None:
            return None, []
        a = best_action
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        steps.append({'customer': int(a.customer), 'slot_kind': a.slot.kind,
                      'slot_anchor': int(a.slot.anchor), 'position': int(a.position)})
        remaining.remove(best_c)
    return P, steps


def cc_lns_mpre_search(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
                       deferred, contract, budget_s, model, capacity, seed=0,
                       protected=frozenset(), n_rounds=N_ROUNDS):
    """M-pre 搜索（复用预训练 CVRP 边 logits 做贪心重插）。返回 (best, best_J, stats, event)。"""
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
    best_res = r0

    stats = {'n_attempts': 0, 'n_reconstruct_fail': 0, 'n_feasible': 0, 'n_duplicate': 0,
             'n_unique': 0, 'n_eval': 0, 'n_eval_infeasible': 0, 'n_partition_fail': 0,
             'n_improve': 0, 'J0': J0, 'J_best': J0, 'P0_feasible': P0_feasible,
             'best_D': r0.D, 'best_Q': r0.Q, 'best_E': r0.E,
             'protected_ids': sorted(int(x) for x in protected),
             'stop_reason': 'normal', 'elapsed_s': 0.0}
    if not P0_feasible or not pool:
        stats['stop_reason'] = 'P0_invalid' if not P0_feasible else 'empty_pool'
        stats['elapsed_s'] = time.perf_counter() - t0
        return best, best_J, stats, _event(clock, env, J0, best_J, stats)

    # 编码可见节点（一次）
    num_nodes = env.coords.shape[1]
    visible = np.zeros(num_nodes, bool)
    visible[0] = True
    visible |= (env.reveal_time[inst_idx] <= clock + 1e-6)
    vis_ids_enc = [int(i) for i in range(1, num_nodes) if visible[i]]
    node_to_local, local_to_node, Nv = build_visible_index(vis_ids_enc)
    coords = env.coords[inst_idx][np.asarray(local_to_node, dtype=np.int32)]
    demands = env.demands[inst_idx][np.asarray(local_to_node, dtype=np.int32)]
    H = encode_cvrp(model, coords[None], demands[None], capacity)   # [1, Nv, 512]

    seen = {plan_hash(P0)}
    eval_cache = {}
    holder = {'best': best, 'best_J': best_J, 'best_res': best_res}

    for _round in range(n_rounds):
        for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
            if time.perf_counter() - t0 > budget_s:
                stats['stop_reason'] = 'budget'
                break
            rng_seed = seed * 10000 + _round * 100 + req_idx
            mask = _select_one_mask(env, inst_idx, holder['best'], pool, mask_size, rng_seed)
            if not mask:
                continue
            partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                    tuple(x for x in p.suffix if x not in mask))
                       for vid, p in holder['best'].items()}
            A0 = build_plan_adjacency(holder['best'], node_to_local, Nv)
            A = build_plan_adjacency(partial, node_to_local, Nv)
            timestep = retained_edge_ratio(A0, A)   # 保留边比例（对齐 keep_prob 语义）
            L = decode_edge_logits(model, H, np.array([timestep], np.float32), A[None])[0]
            cand, _steps = mpre_reconstruct(env, inst_idx, partial, mask, mutable_ids, L,
                                            node_to_local)
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
            ok, _ = _partition_ok(env, inst_idx, clock, vehicles, cand, deferred, served_mask)
            if not ok:
                stats['n_partition_fail'] += 1
                continue
            if ch in eval_cache:
                r = eval_cache[ch]
            else:
                r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract, objective)
                stats['n_eval'] += 1
                if not r.finite or not r.feasible:
                    stats['n_eval_infeasible'] += 1
                    continue
                eval_cache[ch] = r
            if r.J_vis < holder['best_J'] - 1e-9:
                holder['best'] = cand
                holder['best_J'] = r.J_vis
                holder['best_res'] = r
                stats['n_improve'] += 1
        if stats['stop_reason'] == 'budget':
            break
    stats['J_best'] = holder['best_J']
    stats['best_D'] = holder['best_res'].D
    stats['best_Q'] = holder['best_res'].Q
    stats['best_E'] = holder['best_res'].E
    stats['elapsed_s'] = time.perf_counter() - t0
    return holder['best'], holder['best_J'], stats, _event(clock, env, J0, holder['best_J'],
                                                           stats)


def _event(clock, env, J0, best_J, stats):
    return {'event': int(getattr(env, 'event_id', -1)), 'clock': float(clock),
            'J0': float(J0), 'J_best': float(best_J), 'n_attempts': stats['n_attempts'],
            'n_improve': stats['n_improve'], 'stop_reason': stats['stop_reason'],
            'elapsed_s': stats['elapsed_s'], 'protected_ids': stats.get('protected_ids', []),
            'P0_feasible': bool(stats.get('P0_feasible', True))}


class MPreReplanner(JF1HRepairReplanner):
    """M-pre 在线 replanner：JF1-H-F baseline 后用预训练 CVRP 边 logits 做预算内重构。"""

    def __init__(self, model, contract=None, budget_s=1.0, seed=0, capacity=50.0,
                 num_vehicles=25, guard_reserved=True, n_rounds=N_ROUNDS):
        super().__init__(slack_vehicles=1)
        self.model = model
        self.contract = contract
        self.budget_s = float(budget_s)
        self.seed = int(seed)
        self.capacity = float(capacity)
        self.num_vehicles = int(num_vehicles)
        self.guard_reserved = bool(guard_reserved)
        self.n_rounds = int(n_rounds)
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
        best, best_J, stats, ev = cc_lns_mpre_search(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
            self.deferred_customers, contract, self.budget_s, self.model, self.capacity,
            seed=self.seed, protected=protected, n_rounds=self.n_rounds)
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
            elif k in ('J0', 'J_best', 'stop_reason', 'elapsed_s', 'protected_ids',
                       'best_D', 'best_Q', 'best_E'):
                self.online_stats[k] = v
            else:
                self.online_stats[k] += v
        self.online_stats['n_vehicles_changed'] += changed
        self.online_stats['n_partition_fail_writeback'] += (0 if ok else 1)
        ev['replanner_elapsed_s'] = time.perf_counter() - t_replanner
        ev['writeback_changed'] = changed
        ev['writeback_partition_ok'] = ok
        self.event_log.append(ev)
