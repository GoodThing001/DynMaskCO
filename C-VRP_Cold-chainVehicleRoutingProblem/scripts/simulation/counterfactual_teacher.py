"""
P0-U：common-continuation counterfactual utility teacher（02 手册 §P0-U / 01 主控 Layer IV）。

对 snapshot 决策点的当前 customer j，枚举 action，force action 后用**同一 continuation**
（JF1-H）rollout 到终局，得到 counterfactual outcome Y(z,a)。

语义（01 Layer IV / 02 §P0-U1）：
  - z ⊕ a：在决策点，continuation 先产生 incumbent full-fleet plan，apply_action 只移动 j，
    得到新 full-fleet plan；用 env.force_suffix 覆盖决策点这一步的 plan（其它客户保持
    incumbent 位置），随后 commit 各车第一个客户。
  - 后续事件由 continuation（env.replanner）继续处理 → 终局 outcome。
  - incumbent（no-op）action 的 outcome == baseline rollout outcome（delta=0，Gate U）。

outcome = lexicographic (unserved, duplicate, tw_viol, cap_viol, return_viol) + terminal distance。
（quality/energy 未闭环，不进 outcome。）

author: P0-U
date: 2026-09-02
"""
from collections import Counter

from strict_online_env import StrictOnlineEnv
from recourse_snapshot import (restore_recourse_snapshot, clone_snapshot, snapshot_state_hash)
from action_contract import (build_vehicle_plans, apply_action, enumerate_actions,
                             plan_hash)
from authoritative_evaluator import evaluate_execution_trace


def lex_key(outcome):
    """lexicographic outcome key（evaluation_contract §7：先 hard 后 cost，小 = 好）。"""
    return (int(outcome['n_unserved']), int(outcome['n_duplicate']),
            int(not outcome['tw_feasible']), int(not outcome['capacity_feasible']),
            int(not outcome['depot_return_feasible']), float(outcome['distance_cost']))


def _eval(env, inst_idx, traces):
    return evaluate_execution_trace(
        traces, env.coords[inst_idx], env.tw_start[inst_idx], env.tw_end[inst_idx],
        env.service_time[inst_idx], env.demands[inst_idx], env.capacity, speed=env.tw_speed,
        dist_mat=env.dist_mat[inst_idx])


def _decision_context(snapshot):
    """从 snapshot 提取 continuation.plan 所需的决策点上下文。"""
    inst_idx = int(snapshot['instance_id'])
    clock = float(snapshot['clock'])
    served_mask = snapshot['served_mask']
    visible_ids = [int(c) for c in snapshot['customer_universe']
                   if bool(snapshot['visible_mask'][int(c)])]
    return inst_idx, clock, served_mask, visible_ids


def _incumbent_plans(env, snapshot, continuation):
    """在决策点跑 continuation 得到 incumbent full-fleet plan（用于 apply_action 输入）。"""
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
    return build_vehicle_plans(env, inst_idx, vehicles)


def _snapshot_with_force(env, snapshot, force_plan):
    """把 force_plan（vid -> 纯客户 suffix）写成 snapshot 的 force_suffix。

    重建 continuation 风格的完整 mutable_suffix：
      - 有客户 → 客户 + [0]（尾部 return 指令）；
      - 无客户且 ready@customer 且有 future reveal → []（WAIT）；
      - 否则 → [0]（return depot）。
    """
    inst_idx = int(snapshot['instance_id'])
    clock = float(snapshot['clock'])
    custs = [int(c) for c in snapshot['customer_universe']]
    has_future = any(not bool(snapshot['served_mask'][c])
                     and float(env.reveal_time[inst_idx, c]) > clock + 1e-6
                     for c in custs)
    force = {}
    for vid, p in force_plan.items():
        if not p.suffix:
            suffix = [] if (p.anchor_node != 0 and has_future) else [0]
        else:
            suffix = list(p.suffix) + [0]
        force[(int(snapshot['event_id']), int(vid))] = suffix
    snap2 = clone_snapshot(snapshot)
    snap2['force_suffix'] = [[list(k), list(v)] for k, v in sorted(force.items())]
    snap2['state_hash'] = snapshot_state_hash(snap2)
    return snap2


def rollout_baseline(env, snapshot):
    """continuation 从 snapshot 正常 rollout 到终局（无 force）。返回 outcome dict。"""
    traces, _ = env.run_resumed(snapshot)
    return _eval(env, int(snapshot['instance_id']), traces)


def rollout_action(env, snapshot, action, continuation, incumbent_plans=None):
    """force action 后 continuation rollout 到终局。返回 (outcome, force_plan_hash)。

    incumbent_plans 若为 None 则在此计算（continuation 在决策点的输出）。
    """
    inst_idx = int(snapshot['instance_id'])
    if incumbent_plans is None:
        incumbent_plans = _incumbent_plans(env, snapshot, continuation)
    new_plans = apply_action(incumbent_plans, action)
    snap2 = _snapshot_with_force(env, snapshot, new_plans)
    traces, _ = env.run_resumed(snap2)
    outcome = _eval(env, inst_idx, traces)
    return outcome, plan_hash(new_plans)


def enumerate_counterfactuals(env, snapshot, continuation, customer):
    """对当前 customer 枚举所有 action 的 counterfactual outcome（含 incumbent）。

    先在决策点跑 continuation 得到 incumbent full-fleet plan（customer 被分配、incumbent 存在），
    再基于它枚举 action + apply + force rollout。

    返回 (rows, baseline)。rows 为 list of dict：{action_id, incumbent, feasible, reason,
    outcome, lex_key, delta_cost, delta_lex, force_plan_hash}。
    """
    inst_idx = int(snapshot['instance_id'])
    baseline = rollout_baseline(env, snapshot)
    bl = lex_key(baseline)

    # 决策点 incumbent plan：continuation 分配所有 pool 客户 → customer 已分配
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
    incumbent_plans = build_vehicle_plans(env, inst_idx, vehicles)

    cands, _ = enumerate_actions(env, inst_idx, vehicles, customer)
    out = []
    for c in cands:
        if c.feasible:
            outcome, ph = rollout_action(env, snapshot, c.action, continuation, incumbent_plans)
            lk = lex_key(outcome)
            row = {
                'action_id': c.action.action_id(),
                'incumbent': c.action.incumbent,
                'feasible': True,
                'reason': None,
                'outcome': outcome,
                'lex_key': lk,
                'delta_cost': float(outcome['distance_cost']) - bl[5],
                'delta_lex': (lk < bl) - (lk > bl),  # -1 更优, 0 同, +1 更差
                'force_plan_hash': ph,
            }
        else:
            row = {
                'action_id': c.action.action_id(),
                'incumbent': c.action.incumbent,
                'feasible': False,
                'reason': c.reason,
                'outcome': None,
                'lex_key': None,
                'delta_cost': None,
                'delta_lex': None,
                'force_plan_hash': None,
            }
        out.append(row)
    return out, baseline
