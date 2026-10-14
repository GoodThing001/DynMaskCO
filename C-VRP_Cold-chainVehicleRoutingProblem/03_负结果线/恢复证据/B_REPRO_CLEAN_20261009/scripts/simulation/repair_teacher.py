"""N1 teacher：局部修复步骤标签采集（带预留保护的 R 轨迹 + 较充分同目标搜索）。

对每个决策点（prepare + JF1-H-F 形成 P0 + 确定保护集合之后、LNS 搜索之前），用较充分
regret-2 搜索（多轮、更长预算）生成候选，并记录每次重构的**逐步插入序列**：

    P_before → mask → P_partial → (customer, slot, position) × k → P_target

训练对象 = 条件联合重构（预测下一步合法插入），不是旧 g_B 回归。保护集合与 J_vis 沿用
N0-R（guard_reserved）。regret-2 对固定 (partial, mask) 是确定性的：一份就一份，不复制充数。
"""
from __future__ import annotations

import time

import numpy as np

from action_contract import (build_vehicle_plans, enumerate_actions_from_plans, apply_action)
from dynmaskco_cc_context import decision_pool_from_vehicles
from visible_state import build_visible_state, evaluate_visible_plan
from cc_lns_replanner import (REQUEST_SEQUENCE, _select_one_mask, _plan_suffixes,
                              _partition_ok, plan_hash)


def _plan_copy(plans):
    return {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
            for vid, p in plans.items()}


def reconstruct_with_steps(env, inst_idx, partial, mask_customers, mutable_ids):
    """regret-2 重建 + 记录逐步插入。返回 (plan, steps)；不可行返回 (None, [])。

    steps 每项 = dict(customer, slot_kind, slot_anchor, position, predecessor, successor)。
    """
    P = _plan_copy(partial)
    remaining = list(mask_customers)
    steps = []
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
            return None, []
        a = best_action
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        steps.append({'customer': int(a.customer), 'slot_kind': a.slot.kind,
                      'slot_anchor': int(a.slot.anchor), 'position': int(a.position),
                      'predecessor': int(a.predecessor), 'successor': int(a.successor)})
        remaining.remove(best_c)
    return P, steps


def collect_event_records(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids,
                          deferred, contract, protected, budget_s, seed, n_rounds=8):
    """一个决策点的 teacher 记录：较充分搜索 + 逐步插入序列。返回 (records, stats)。

    每条 record 含 P_partial / mask / steps / P_target / J_before / J_target / protected。
    """
    objective = contract.objective
    mutable_ids = {int(v) for v in replan_ids} - set(protected)
    pool = decision_pool_from_vehicles(vehicles, served_mask, visible_ids)
    vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                              served_mask, visible_ids, replan_ids, deferred)
    P0 = build_vehicle_plans(env, inst_idx, vehicles)
    r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), contract, objective)
    J0 = r0.J_vis

    stats = {'n_attempts': 0, 'n_reconstruct_ok': 0, 'n_partition_fail': 0, 'elapsed_s': 0.0,
             'stop_reason': 'normal'}
    if not (r0.feasible and r0.finite) or not pool:
        stats['stop_reason'] = 'P0_invalid' if not r0.feasible else 'empty_pool'
        return [], stats, P0, J0

    best = P0
    best_J = J0
    seen = {plan_hash(P0)}
    records = []
    t0 = time.perf_counter()
    for _round in range(n_rounds):
        for req_idx, mask_size in enumerate(REQUEST_SEQUENCE):
            if time.perf_counter() - t0 > budget_s:
                stats['stop_reason'] = 'budget'
                break
            rng_seed = seed * 10000 + _round * 100 + req_idx
            mask = _select_one_mask(env, inst_idx, best, pool, mask_size, rng_seed)
            if not mask:
                continue
            partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                    tuple(x for x in p.suffix if x not in mask))
                       for vid, p in best.items()}
            cand, steps = reconstruct_with_steps(env, inst_idx, partial, mask, mutable_ids)
            stats['n_attempts'] += 1
            if cand is None:
                continue
            stats['n_reconstruct_ok'] += 1
            ch = plan_hash(cand)
            if ch in seen:
                continue
            seen.add(ch)
            ok, _detail = _partition_ok(env, inst_idx, clock, vehicles, cand, deferred,
                                        served_mask)
            if not ok:
                stats['n_partition_fail'] += 1
                continue
            r = evaluate_visible_plan(vis, _plan_suffixes(cand), contract, objective)
            if not r.finite or not r.feasible:
                continue
            Jc = r.J_vis
            records.append({
                'inst': int(inst_idx), 'event': int(getattr(env, 'event_id', -1)),
                'clock': float(clock), 'protected': sorted(int(x) for x in protected),
                'mutable_ids': sorted(int(x) for x in mutable_ids),
                'mask': [int(x) for x in mask],
                'P_partial': {str(vid): {'anchor_node': int(p.anchor_node),
                                         'anchor_time': float(p.anchor_time),
                                         'anchor_load': float(p.anchor_load),
                                         'suffix': [int(x) for x in p.suffix]}
                              for vid, p in partial.items()},
                'steps': steps,
                'P_target': {str(vid): {'anchor_node': int(p.anchor_node),
                                        'anchor_time': float(p.anchor_time),
                                        'anchor_load': float(p.anchor_load),
                                        'suffix': [int(x) for x in p.suffix]}
                             for vid, p in cand.items()},
                'deferred': sorted(int(x) for x in deferred),
                'J_before': float(J0), 'J_target': float(Jc),
                'improved': bool(Jc < J0 - 1e-9), 'seed': int(seed), 'round': int(_round),
                'request': int(req_idx)})
            if Jc < best_J - 1e-9:
                best = cand
                best_J = Jc
        if stats['stop_reason'] == 'budget':
            break
    stats['elapsed_s'] = time.perf_counter() - t0
    # 去重后保留 J_vis 最优的前 3 条（不同 mask/incumbent 的参考）
    records.sort(key=lambda r: r['J_target'])
    return records[:3], stats, best, best_J
