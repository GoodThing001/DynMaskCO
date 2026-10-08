"""A-v1 终局评价器 + 数据接通（独立于 v1 的 coldchain_evaluator.py）。

A-v1 语义（2026-09-24 冻结）：
  - 鱼（class 1/2）零阶感官评分，返仓绝对评分 >= 5 才可售；番茄（class 0）一阶营养比例，
    无硬可售阈值，salable=None（不评价）。
  - 合法拒单：不是「全部订单服务」，只要求「已接受订单全部服务」（承诺兑现）。
  - 主效用 = 服务收入 − 燃油成本 − 拒绝损失；品质不货币化，鱼可售率/番茄品质独立报告。

复用 C0 状态机（transition_segment 的 trace），不重演第二条热轨迹。
"""
from __future__ import annotations

import numpy as np

REV = {0: 10.0, 1: 20.0, 2: 30.0}
FUEL_COST_PER_KM = 0.1
KM_PER_UNIT = 20.0


def add_v2_initial_quality(dataset, contract):
    """给 dataset 加 initial_quality（v2 语义：鱼初始评分 9，番茄初始比例 1）。

    旧数据不含 initial_quality，env 默认填 1 会把鱼判到可售线（5）以下。
    返回加了字段的新 dataset（浅拷贝 + 新数组）。
    """
    q2 = contract.quality
    tc = dataset['temp_class']
    init = np.ones_like(tc, dtype=np.float32)
    for c in range(len(q2.initial_value)):
        init[tc == c] = float(q2.initial_value[c])
    out = dict(dataset)
    out['initial_quality'] = init
    return out


def evaluate_trace_a1(traces, dataset_instance, contract, accepted_set, rejected_set,
                      B, reject_penalty_by_class):
    """A-v1 终局评价。返回 dict。

    硬约束：承诺兑现(accepted ⊆ served)、鱼可售(score>=5)、时间窗、容量、能耗<=B。
    效用 = 服务收入 − 燃油 − 拒绝损失（无 K_EFF 品质扣减）。
    """
    demands = np.asarray(dataset_instance['demands'])
    temp_class = np.asarray(dataset_instance['temp_class'], dtype=np.int64)
    tw_end = np.asarray(dataset_instance['tw_end'])
    capacity = contract.operational.shared_vehicle_capacity

    served_list = [s.node for t in traces for s in t.services]
    served_set = set(served_list)
    served_counter = {}
    for o in served_list:
        served_counter[o] = served_counter.get(o, 0) + 1

    failures = []
    # 承诺兑现 + 无重复服务 + 未接受不服务
    if accepted_set - served_set:
        failures.append('commitment_violation')
    if any(c > 1 for c in served_counter.values()):
        failures.append('duplicate_service')
    if served_set - accepted_set:
        failures.append('served_not_accepted')
    # 所有已揭示订单必须恰好接受或拒绝（互斥 + 覆盖）
    universe_set = set(int(i) for i in range(1, len(demands)) if demands[i] > 0)
    if accepted_set & rejected_set:
        failures.append('order_both_accepted_and_rejected')
    unaccounted = universe_set - accepted_set - rejected_set
    if unaccounted:
        failures.append('unaccounted_order')

    # 时间窗 / 容量 / 返仓 / 服务前揭示
    depot_deadline = float(tw_end[0])
    for t in traces:
        load = 0.0
        for s in t.services:
            load += float(demands[s.node])
            if load > capacity + 1e-6:
                failures.append('capacity_violation')
            if s.arrival_time > float(tw_end[s.node]) + 1e-6:
                failures.append('customer_tw_violation')
            if s.depart_time < float(dataset_instance['reveal_time'][s.node]) - 1e-6:
                failures.append('served_before_reveal')
        if not t.services:
            continue
        if t.return_arrival is None or t.return_arrival > depot_deadline + 1e-6:
            failures.append('depot_return_violation')

    # 能耗 + 距离 + 交付记录
    energy_kwh = 0.0
    distance_km = 0.0
    delivered = []
    missing_final_state = False
    for t in traces:
        st = t.final_coldchain_state
        if t.services and st is None:
            missing_final_state = True
            continue
        if st is None:
            continue
        if st.cargo_manifest or st.total_load > 1e-9:
            failures.append('cargo_not_cleared')
        energy_kwh += float(st.cumulative_energy_kwh)
        distance_km += float(st.cumulative_distance_km)
        delivered.extend(st.delivered_to_depot)
    if missing_final_state:
        failures.append('missing_final_state')
    delivered_orders = {r.order_id for r in delivered}
    if accepted_set - delivered_orders:
        failures.append('accepted_not_delivered')

    budget_violated = energy_kwh > B + 1e-9
    if budget_violated:
        failures.append('budget_violation')

    # 鱼可售（salable=False 是硬失败）；番茄 salable=None（不评价，不算失败）
    fish_unsalable = [r for r in delivered if r.salable is False]
    if fish_unsalable:
        failures.append('fish_unsalable')

    # 报告：各温区服务/拒绝/可售
    universe = [int(i) for i in range(1, len(demands)) if demands[i] > 0]
    service_rate = {}
    reject_rate = {}
    salable_rate = {}
    for c in range(3):
        cls_orders = [o for o in universe if temp_class[o] == c]
        cls_served = [o for o in cls_orders if o in served_set]
        cls_rejected = [o for o in cls_orders if o in rejected_set]
        service_rate[str(c)] = len(cls_served) / max(len(cls_orders), 1)
        reject_rate[str(c)] = len(cls_rejected) / max(len(cls_orders), 1)
        cls_records = [r for r in delivered if r.temp_class == c]
        cls_evaluated = [r for r in cls_records if r.salable is not None]
        salable_rate[str(c)] = (
            sum(r.salable for r in cls_evaluated) / max(len(cls_evaluated), 1)
            if cls_evaluated else None)

    # 番茄平均品质（连续，不评价）
    tomato_quality = np.mean(
        [r.delivered_quality for r in delivered if r.temp_class == 0]) if any(
        r.temp_class == 0 for r in delivered) else None

    # 效用
    revenue = sum(REV[int(temp_class[o])] for o in served_set)
    fuel = distance_km * FUEL_COST_PER_KM
    reject_loss = sum(float(reject_penalty_by_class[int(temp_class[o])]) for o in rejected_set)
    utility = revenue - fuel - reject_loss

    hard_feasible = len(failures) == 0
    return {
        'utility': float(utility),
        'revenue': float(revenue),
        'fuel_cost': float(fuel),
        'reject_loss': float(reject_loss),
        'energy_kwh': float(energy_kwh),
        'distance_km': float(distance_km),
        'budget_violated': bool(budget_violated),
        'hard_feasible': bool(hard_feasible),
        'failures': sorted(set(failures)),
        'n_served': int(len(served_set)),
        'n_rejected': int(len(rejected_set)),
        'n_fish_unsalable': int(len(fish_unsalable)),
        'service_rate': service_rate,
        'reject_rate': reject_rate,
        'salable_rate': salable_rate,
        'tomato_mean_quality': (float(tomato_quality) if tomato_quality is not None else None),
    }
