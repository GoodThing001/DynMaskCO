"""O0-D / O0-CC：greedy sequential action-space oracle。

按预先冻结的 customer order 逐个执行 greedy oracle：

  - 每个决策点，待分配客户按 (tw_end, reveal_time, customer_id) 字典序排序；
  - 对当前 customer 枚举 P0-A 定义的全部可行 action；
  - 每个候选 action 强制后，用同一 JF1-H continuation rollout 到终局；
  - 按 (hard_outcome, cost, action_id) 字典序选最优；
  - 只把选中的 action 写回 oracle 当前 plan，处理下一个 customer；
  - 全部客户处理完，环境推进到下一个真实事件（on-trajectory，不复用 baseline snapshot）。

不做同一事件内所有客户 action 的笛卡尔积联合搜索，以控制 rollout 成本。
O0-D 用 objective='distance'，O0-CC 用 objective='coldchain'（候选集/顺序/预算完全相同）。

显式 KEEP/DEFER（阻塞项 7）：当 JF1-H incumbent plan 遗漏某个 visible 客户时，该客户
没有 incumbent/no-op 候选；oracle 必须显式评估「保持遗漏（DEFER）」与「插入」，否则
会因多服务客户而在 hard outcome 上虚假改善，不再是严格上界。
"""

from __future__ import annotations

from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot,
                               snapshot_state_hash)
from action_contract import (apply_action, enumerate_actions_from_plans, find_customer_slot,
                             plan_hash)
from counterfactual_teacher import (lex_key, rollout_action, _incumbent_plans,
                                    _snapshot_with_force, _eval)
from hard_gate import service_ok, select_improving, outcome_protocol_error


def _outcome_cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def customer_order_key(env, inst_idx):
    """固定 customer order：tw_end → reveal_time → customer_id（O0 运行前冻结）。"""
    tw_end = env.tw_end[inst_idx]
    reveal = env.reveal_time[inst_idx]
    return lambda c: (float(tw_end[int(c)]), float(reveal[int(c)]), int(c))


def mutable_vehicle_ids(snapshot):
    """本次 replan 的 idle/ready 车辆 id（与 baseline 修复版一致的 mutable 范围）。"""
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    return {v.vehicle_id for v in vehicles
            if v.status in ('idle', 'ready') and v.needs_replan}


def decision_pool(snapshot):
    """决策点待分配客户：visible & unserved & 未 committed & 不在非 replan 车冻结 tail。

    与修复版 baseline 的 get_global_pool 边界一致：非 replan 车的 tail 冻结，其客户
    不得进入 oracle 的重分配池（否则同一客户会被两辆车服务）。
    """
    served = snapshot['served_mask']
    visible = [int(c) for c in snapshot['customer_universe']
               if bool(snapshot['visible_mask'][int(c)])]
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    locked = set()
    for v in vehicles:
        if v.status == 'committed' and v.committed_next not in (None, 0):
            locked.add(int(v.committed_next))
        if v.vehicle_id not in replan_ids:
            for n in v.mutable_suffix:
                if int(n) != 0:
                    locked.add(int(n))
    return [c for c in visible if not served[int(c)] and int(c) not in locked]


def _rollout_plan(env, snapshot, plans, continuation, objective, objective_contract,
                  mutable_ids=None):
    """force 当前 full-fleet plan 后 continuation rollout 到终局（DEFER 评估用）。"""
    snap2 = _snapshot_with_force(env, snapshot, plans, mutable_ids=mutable_ids)
    traces, served = env.run_resumed(snap2)
    outcome = _eval(env, int(snapshot['instance_id']), traces, objective, objective_contract,
                    served_mask=served)
    return outcome, plan_hash(plans)


def sequential_oracle_plan(env, snapshot, continuation, objective='distance',
                           objective_contract=None, log=None):
    """在决策点按 customer order 逐个 greedy oracle 决策，返回 oracle full-fleet plan。

    ``env`` 是独立的 rollout 环境（replanner=continuation），供每个候选 action 的
    common-continuation rollout 使用，与真实 oracle trajectory 环境隔离。

    ``log``（可选）为 list，追加每个 customer 的逐决策记录：
    {event_id, customer, num_candidates, num_feasible, incumbent_exists,
     selected_action, reject_reasons, n_rollouts}。
    """
    inst_idx = int(snapshot['instance_id'])
    pool = sorted(decision_pool(snapshot), key=customer_order_key(env, inst_idx))
    mutable_ids = mutable_vehicle_ids(snapshot)
    plans = _incumbent_plans(env, snapshot, continuation)
    # 严格 repair 审计只对 JF1-H-F（有 repair_stats）强制；旧 JF1-H 无 repair 审计则跳过
    strict_repair = hasattr(continuation, 'repair_stats')

    for customer in pool:
        src_vid, _ = find_customer_slot(plans, customer)
        cands, _ = enumerate_actions_from_plans(env, inst_idx, plans, customer,
                                                allowed_vehicle_ids=mutable_ids)

        # KEEP（已在 suffix）/ DEFER（未分配）baseline：先显式评估「保持当前完整计划」。
        keep_outcome, _ = _rollout_plan(env, snapshot, plans, continuation,
                                        objective, objective_contract,
                                        mutable_ids=mutable_ids)
        _err = outcome_protocol_error(keep_outcome, objective, strict_repair=strict_repair)
        if _err is not None:
            raise RuntimeError(f"PROTOCOL_ERROR: KEEP outcome invalid ({_err})")
        keep_cost = _outcome_cost(keep_outcome, objective)
        n_rollouts = 1
        reject_reasons = {}
        cost_id_pairs = []
        n_hard_pass = 0

        for cand in cands:
            if not cand.feasible:
                reject_reasons[cand.reason] = reject_reasons.get(cand.reason, 0) + 1
                continue
            outcome, _ = rollout_action(env, snapshot, cand.action, continuation, plans,
                                        objective=objective, objective_contract=objective_contract,
                                        allowed_vehicle_ids=mutable_ids, mutable_ids=mutable_ids)
            n_rollouts += 1
            _err = outcome_protocol_error(outcome, objective, strict_repair=strict_repair)
            if _err is not None:
                raise RuntimeError(f"PROTOCOL_ERROR: candidate {cand.action.action_id()} "
                                   f"outcome invalid ({_err})")
            if not service_ok(outcome, objective):
                reject_reasons['not_service_ok'] = reject_reasons.get('not_service_ok', 0) + 1
                continue
            n_hard_pass += 1
            cost_id_pairs.append((_outcome_cost(outcome, objective), cand.action.action_id()))

        # 顺序无关选择：τ 只判断是否离开 KEEP，改善候选之间按 (cost, action_id) 稳定取最小。
        best = select_improving(cost_id_pairs, keep_cost)
        best_action = None
        if best is not None:
            for cand in cands:
                if cand.action.action_id() == best[1]:
                    best_action = cand.action
                    break

        selected_cost = best[0] if best is not None else keep_cost
        if log is not None:
            log.append({
                'seq': len(log),  # 真实执行顺序（不被后续 customer_id 排序破坏）
                'event_id': int(snapshot['event_id']),
                'state_hash': snapshot_state_hash(snapshot),
                'customer': int(customer),
                'num_candidates': len(cands),
                'num_feasible': sum(1 for c in cands if c.feasible),
                'num_rolled_out': n_rollouts - 1,
                'num_hard_pass': n_hard_pass,
                'incumbent_exists': src_vid is not None,
                'keep_cost': keep_cost,
                'selected_cost': selected_cost,
                'selected_delta': selected_cost - keep_cost,
                'selected_action': (best_action.action_id() if best_action is not None
                                    else ('__KEEP__' if src_vid is not None else '__DEFER__')),
                'reject_reasons': reject_reasons,
                'n_rollouts': n_rollouts,
            })

        if best_action is not None:
            plans = apply_action(plans, best_action, allowed_vehicle_ids=mutable_ids)
    return plans


def make_oracle_hook(rollout_env, continuation, objective='distance',
                     objective_contract=None, log=None):
    """构造 sequential-oracle 决策钩子（在每个决策点写 env.force_suffix）。

    ``log``（可选）为 list，收集所有决策点的逐事件决策记录（供 coverage/审计）。
    """

    def hook(env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers):
        snapshot = capture_recourse_snapshot(
            env, inst_idx, clock, event_id, reveal_idx,
            vehicles, traces, served_mask, all_customers)
        plan = sequential_oracle_plan(rollout_env, snapshot, continuation, objective,
                                      objective_contract, log=log)
        snap2 = _snapshot_with_force(rollout_env, snapshot, plan,
                                     mutable_ids=mutable_vehicle_ids(snapshot))
        env.force_suffix = {tuple(k): list(v) for k, v in snap2['force_suffix']}

    return hook
