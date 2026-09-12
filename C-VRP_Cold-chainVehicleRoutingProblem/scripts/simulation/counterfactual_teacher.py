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

默认 outcome 保持 P0-U 的 distance lexicographic 语义。C0 可显式选择
``objective='coldchain'``，在相同 action/continuation 下改用闭环 D/Q/E 终局结果。

author: P0-U
date: 2026-09-02
"""
from collections import Counter

from strict_online_env import StrictOnlineEnv
from recourse_snapshot import (restore_recourse_snapshot, clone_snapshot, snapshot_state_hash)
from action_contract import (build_vehicle_plans, apply_action, enumerate_actions,
                             enumerate_actions_from_plans, plan_hash)
from authoritative_evaluator import evaluate_execution_trace


def lex_key(outcome, objective='distance'):
    """lexicographic outcome key（evaluation_contract §7：先 hard 后 cost，小 = 好）。"""
    hard = (int(outcome['n_unserved']), int(outcome['n_duplicate']),
            int(not outcome['tw_feasible']), int(not outcome['capacity_feasible']),
            int(not outcome['depot_return_feasible']))
    if objective == 'distance':
        return hard + (float(outcome['distance_cost']),)
    if objective != 'coldchain':
        raise ValueError("objective must be 'distance' or 'coldchain'")
    coldchain_hard = (
        int(not outcome['temperature_hard_feasible']),
        int(not outcome['all_orders_picked']),
        int(not outcome['all_cargo_delivered_to_depot']),
        int(not outcome['terminal_manifests_empty']),
    )
    return hard + coldchain_hard + (float(outcome['coldchain_cost']),)


def _eval(env, inst_idx, traces, objective='distance', objective_contract=None, served_mask=None):
    kwargs = {}
    if objective == 'coldchain':
        if env.coldchain_contract is None:
            raise ValueError("coldchain objective requires env.coldchain_contract")
        instance = {
            'coords': env.coords[inst_idx],
            'tw_start': env.tw_start[inst_idx],
            'tw_end': env.tw_end[inst_idx],
            'service_time': env.service_time[inst_idx],
            'demands': env.demands[inst_idx],
            'dist_mat': env.dist_mat[inst_idx],
            'temp_class': env.temp_class[inst_idx],
            'initial_quality': env.initial_quality[inst_idx],
            'speed': env.tw_speed,
        }
        kwargs = {
            'coldchain_contract': env.coldchain_contract,
            'dataset_instance': instance,
            'objective_contract': objective_contract,
        }
    elif objective != 'distance':
        raise ValueError("objective must be 'distance' or 'coldchain'")
    outcome = evaluate_execution_trace(
        traces, env.coords[inst_idx], env.tw_start[inst_idx], env.tw_end[inst_idx],
        env.service_time[inst_idx], env.demands[inst_idx], env.capacity, speed=env.tw_speed,
        dist_mat=env.dist_mat[inst_idx], **kwargs)
    # 附加 repair 审计（本次 continuation 的 ownership_violations / terminal_unresolved）。
    # run_resumed 已 restore_state 重置审计，故只统计本分支，不混入上一个候选。
    rp = getattr(env, 'replanner', None)
    if rp is not None and hasattr(rp, 'repair_stats') and served_mask is not None:
        outcome['ownership_violations'] = int(rp.repair_stats.get('ownership_violations', 0))
        outcome['terminal_unresolved'] = sum(
            1 for c in rp.deferred_customers if not served_mask[int(c)])
    return outcome


def _decision_context(snapshot):
    """从 snapshot 提取 continuation.plan 所需的决策点上下文。"""
    inst_idx = int(snapshot['instance_id'])
    clock = float(snapshot['clock'])
    served_mask = snapshot['served_mask']
    visible_ids = [int(c) for c in snapshot['customer_universe']
                   if bool(snapshot['visible_mask'][int(c)])]
    return inst_idx, clock, served_mask, visible_ids


def _incumbent_plans(env, snapshot, continuation):
    """在决策点跑 continuation 得到 incumbent full-fleet plan（用于 apply_action 输入）。

    与真实 _plan_and_commit 对齐：先 prepare_decision_point（锚点时间一致），再恢复
    continuation 决策状态（deferred），再跑 plan。这样 no-op 候选 == 原 baseline。
    """
    inst_idx, clock, served_mask, visible_ids = _decision_context(snapshot)
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    env.prepare_decision_point(clock, vehicles)
    if hasattr(continuation, 'restore_state'):
        continuation.restore_state(snapshot.get('replanner_state'))
    replan_ids = {v.vehicle_id for v in vehicles
                  if v.status in ('idle', 'ready') and v.needs_replan}
    continuation.plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                      replan_ids=replan_ids)
    return build_vehicle_plans(env, inst_idx, vehicles)


def _snapshot_with_force(env, snapshot, force_plan, mutable_ids=None):
    """把 force_plan（vid -> 纯客户 suffix）写成 snapshot 的 force_suffix。

    只对 mutable_ids（本次 replan 的 idle/ready 车）写 force；非 mutable 车保留 snapshot
    原始指令（逐字不变，no-op 不得把 WAIT 悄悄改成返仓）。

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
    if mutable_ids is None:
        mutable_ids = set(force_plan.keys())
    else:
        mutable_ids = set(int(v) for v in mutable_ids)
    force = {}
    for vid, p in force_plan.items():
        if int(vid) not in mutable_ids:
            continue
        if not p.suffix:
            suffix = [] if (p.anchor_node != 0 and has_future) else [0]
        else:
            suffix = list(p.suffix) + [0]
        force[(int(snapshot['event_id']), int(vid))] = suffix
    snap2 = clone_snapshot(snapshot)
    snap2['force_suffix'] = [[list(k), list(v)] for k, v in sorted(force.items())]
    snap2['state_hash'] = snapshot_state_hash(snap2)
    return snap2


def rollout_baseline(env, snapshot, objective='distance', objective_contract=None):
    """continuation 从 snapshot 正常 rollout 到终局（无 force）。返回 outcome dict。"""
    traces, served = env.run_resumed(snapshot)
    return _eval(env, int(snapshot['instance_id']), traces, objective, objective_contract,
                 served_mask=served)


def rollout_action(env, snapshot, action, continuation, incumbent_plans=None,
                   objective='distance', objective_contract=None,
                   allowed_vehicle_ids=None, mutable_ids=None):
    """force action 后 continuation rollout 到终局。返回 (outcome, force_plan_hash)。

    allowed_vehicle_ids / mutable_ids：与 baseline 一致的范围，贯穿 apply 与 force 写回
    （NEW_ROUTE physical matching 与冻结车保留都限定在该范围）。incumbent_plans 若为 None
    则在此计算（continuation 在决策点的输出）。
    """
    inst_idx = int(snapshot['instance_id'])
    if incumbent_plans is None:
        incumbent_plans = _incumbent_plans(env, snapshot, continuation)
    new_plans = apply_action(incumbent_plans, action, allowed_vehicle_ids=allowed_vehicle_ids)
    snap2 = _snapshot_with_force(env, snapshot, new_plans, mutable_ids=mutable_ids)
    traces, served = env.run_resumed(snap2)
    outcome = _eval(env, inst_idx, traces, objective, objective_contract, served_mask=served)
    return outcome, plan_hash(new_plans)


def _mutable_ids_from_snapshot(snapshot):
    """本次 replan 的 idle/ready 车辆 id（与 sequential_oracle 的 mutable scope 一致）。"""
    vehicles, _, _ = restore_recourse_snapshot(snapshot)
    return {v.vehicle_id for v in vehicles
            if v.status in ('idle', 'ready') and v.needs_replan}


def enumerate_counterfactuals(env, snapshot, continuation, customer,
                              objective='distance', objective_contract=None):
    """对当前 customer 枚举所有 action 的 counterfactual outcome（含 incumbent）。

    复用与 runner 相同的协议：_incumbent_plans（prepare_decision_point + restore_state）
    + mutable scope 贯穿枚举/apply/rollout。返回 (rows, baseline)。
    """
    inst_idx = int(snapshot['instance_id'])
    baseline = rollout_baseline(env, snapshot, objective, objective_contract)
    bl = lex_key(baseline, objective)

    incumbent_plans = _incumbent_plans(env, snapshot, continuation)
    mutable_ids = _mutable_ids_from_snapshot(snapshot)

    cands, _ = enumerate_actions_from_plans(env, inst_idx, incumbent_plans, customer,
                                            allowed_vehicle_ids=mutable_ids)
    out = []
    for c in cands:
        if c.feasible:
            outcome, ph = rollout_action(
                env, snapshot, c.action, continuation, incumbent_plans,
                objective=objective, objective_contract=objective_contract,
                allowed_vehicle_ids=mutable_ids, mutable_ids=mutable_ids)
            lk = lex_key(outcome, objective)
            row = {
                'action_id': c.action.action_id(),
                'incumbent': c.action.incumbent,
                'feasible': True,
                'reason': None,
                'outcome': outcome,
                'lex_key': lk,
                'delta_cost': (float(outcome['distance_cost'])
                               - float(baseline['distance_cost'])),
                'delta_distance': (float(outcome['distance_cost'])
                                   - float(baseline['distance_cost'])),
                'delta_lex': (lk < bl) - (lk > bl),  # -1 更优, 0 同, +1 更差
                'force_plan_hash': ph,
            }
            if objective == 'coldchain':
                row.update({
                    'hard_outcome': _hard_outcome(outcome),
                    'distance_cost': float(outcome['distance_cost']),
                    'quality_loss': float(outcome['quality_loss']),
                    'energy_kwh': float(outcome['energy_kwh']),
                    'num_unsalable': int(outcome['num_unsalable']),
                    'thermal_violation_count': int(
                        outcome['thermal_violation_count']),
                    'thermal_violation_duration_h': float(
                        outcome['thermal_violation_duration_h']),
                    'coldchain_cost': float(outcome['coldchain_cost']),
                    'delta_quality': (float(outcome['quality_loss'])
                                      - float(baseline['quality_loss'])),
                    'delta_energy': (float(outcome['energy_kwh'])
                                     - float(baseline['energy_kwh'])),
                    'delta_coldchain_cost': (float(outcome['coldchain_cost'])
                                             - float(baseline['coldchain_cost'])),
                    'contract_hash': outcome['contract_hash'],
                })
        else:
            row = {
                'action_id': c.action.action_id(),
                'incumbent': c.action.incumbent,
                'feasible': False,
                'reason': c.reason,
                'outcome': None,
                'lex_key': None,
                'delta_cost': None,
                'delta_distance': None,
                'delta_lex': None,
                'force_plan_hash': None,
            }
            if objective == 'coldchain':
                row.update({
                    'hard_outcome': None,
                    'distance_cost': None,
                    'quality_loss': None,
                    'energy_kwh': None,
                    'num_unsalable': None,
                    'thermal_violation_count': None,
                    'thermal_violation_duration_h': None,
                    'coldchain_cost': None,
                    'delta_quality': None,
                    'delta_energy': None,
                    'delta_coldchain_cost': None,
                    'contract_hash': env.coldchain_contract.contract_hash,
                })
        out.append(row)
    return out, baseline


def _hard_outcome(outcome):
    return {
        'complete': bool(outcome['complete']),
        'tw_feasible': bool(outcome['tw_feasible']),
        'capacity_feasible': bool(outcome['capacity_feasible']),
        'depot_return_feasible': bool(outcome['depot_return_feasible']),
        'temperature_hard_feasible': bool(outcome['temperature_hard_feasible']),
        'all_orders_picked': bool(outcome['all_orders_picked']),
        'all_cargo_delivered_to_depot': bool(
            outcome['all_cargo_delivered_to_depot']),
        'terminal_manifests_empty': bool(outcome['terminal_manifests_empty']),
    }
