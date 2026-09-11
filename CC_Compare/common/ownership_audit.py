"""公共 ownership audit（B1.1 P0-3）：所有 baseline 共用的逐事件 + 终局审计。

独立于任何具体方法（PyVRP/RRNCO/CaDA 都走同一份审计）。对每个决策点的
fleet plan 状态（plans_after）与快照（committed / served / visible）做交叉检查：

  违规（计入 ownership_violations，protocol error）：
    - duplicate_suffix        同一客户出现在多辆车 suffix（含同车重复）
    - duplicate_committed     同一客户被多辆车 committed
    - committed_and_suffix    客户同时在 committed leg 与 suffix 中
    - served_in_plan          已服务客户重新进入计划
    - future_in_plan          未 reveal 客户出现在计划（未来泄漏）
    - invalid_customer        客户编号非法（0 / 越界 / 非客户节点）

  合法记录（不计违规）：
    - deferred                可见未服务客户不在任何计划（允许，记入 deferred 名单）

终局：
    - terminal_unresolved     reveal 过但从未被服务的客户数
    - ownership_violations    全运行违规计数（按 客户×事件 计数）
"""
from __future__ import annotations

from collections import Counter


def audit_decision_point(plans, committed_by, served_mask, visible_ids,
                         customer_universe):
    """审计一个决策点的 fleet plan。plans = build_vehicle_plans 输出；
    committed_by = {customer_id: [vid, ...]}（来自快照 committed_next）。

    返回 (violations: list[dict], deferred: list[int])。
    violations 每条含 {type, customer, vehicle_ids}。
    """
    violations = []
    suffix_loc = {}        # customer -> list of vid（可含同车重复）
    for vid, p in plans.items():
        for c in p.suffix:
            suffix_loc.setdefault(int(c), []).append(vid)

    visible = set(int(x) for x in visible_ids)
    served = {int(c) for c in np_where(served_mask)}
    universe = set(int(c) for c in customer_universe)

    all_plan = set(suffix_loc) | set(committed_by)
    for c in sorted(all_plan):
        if c not in universe or c <= 0:
            violations.append({'type': 'invalid_customer', 'customer': c,
                               'vehicle_ids': sorted(set(suffix_loc.get(c, [])
                                                         + committed_by.get(c, [])))})
            continue
        s_locs = suffix_loc.get(c, [])
        c_locs = committed_by.get(c, [])
        if len(s_locs) > 1:
            violations.append({'type': 'duplicate_suffix', 'customer': c,
                               'vehicle_ids': sorted(set(s_locs))})
        if len(c_locs) > 1:
            violations.append({'type': 'duplicate_committed', 'customer': c,
                               'vehicle_ids': sorted(set(c_locs))})
        if s_locs and c_locs:
            violations.append({'type': 'committed_and_suffix', 'customer': c,
                               'vehicle_ids': sorted(set(s_locs + c_locs))})
        if c in served and (s_locs or c_locs):
            violations.append({'type': 'served_in_plan', 'customer': c,
                               'vehicle_ids': sorted(set(s_locs + c_locs))})
        if c not in visible and (s_locs or c_locs):
            violations.append({'type': 'future_in_plan', 'customer': c,
                               'vehicle_ids': sorted(set(s_locs + c_locs))})

    deferred = sorted(c for c in visible - served if c not in suffix_loc
                      and c not in committed_by)
    return violations, deferred


def audit_terminal(served_mask, customer_universe):
    """终局：reveal 过（全部客户 reveal_time ≤ 终局时钟）但从未服务的客户。

    env 事件循环只在 all_served 或 horizon 终止；horizon 之后所有客户必已 reveal，
    因此 terminal_unresolved = 终局未服务客户集合。
    """
    return [int(c) for c in customer_universe
            if not bool(served_mask[int(c)])]


def np_where(mask):
    import numpy as np
    return np.where(np.asarray(mask))[0]


def summarize_audit(audit_records):
    """汇总整轮审计记录（每决策点一条）→ 终局统计。"""
    total_violations = 0
    by_type = Counter()
    deferred_events = 0
    for rec in audit_records:
        total_violations += len(rec['violations'])
        for v in rec['violations']:
            by_type[v['type']] += 1
        if rec['deferred']:
            deferred_events += 1
    return {'ownership_violations': int(total_violations),
            'violations_by_type': dict(by_type),
            'n_events_with_deferred': int(deferred_events)}
