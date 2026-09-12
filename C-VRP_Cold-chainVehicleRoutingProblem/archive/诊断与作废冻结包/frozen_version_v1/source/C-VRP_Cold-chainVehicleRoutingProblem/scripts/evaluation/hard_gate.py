"""共享 cold-chain / distance hard gate（阶段 C 步骤 4）。

单一纯函数生成 outcome-level hard vector，供 audit、freeze、候选评估、service Gate 与
最终报告共用，保证「hard-pass」语义跨模块一致（尤其 accounting 项贯穿候选选择）。

注意：terminal_unresolved / ownership_violations 是 repair-level 字段，不在 outcome 里，
由调用方（audit/freeze）在 outcome hard vector 之上补充。
"""
from __future__ import annotations

import math

import numpy as np


def _is_bool(v):
    return isinstance(v, (bool, np.bool_))


def outcome_protocol_error(outcome, objective='coldchain', strict_repair=False) -> str | None:
    """返回 PROTOCOL_ERROR 原因字符串；None 表示 outcome 协议完整（可安全用于比较）。

    与「合法模拟得到的服务失败」区分：
      - 缺必需字段 / 类型错误 / NaN/Inf / accounting 不一致 → PROTOCOL_ERROR（停止正式判定）；
      - TW / capacity / temperature / 服务失败 → 合法不可接受候选（候选拒绝，非协议错误）。
    numeric / boolean / accounting 字段随 objective 变化。

    strict_repair=True 时，repair 字段（ownership_violations / terminal_unresolved）必须存在、
    为有限非负整数（不 int() 截断），缺失报协议错误——用于正式 JF1-H-F 路径（其 _eval 已附加
    repair 审计）。默认 False 保持对无 repair 审计的旧 outcome 兼容。
    """
    required = ['complete', 'n_unserved', 'n_duplicate', 'tw_feasible',
                'capacity_feasible', 'depot_return_feasible', 'distance_cost']
    numeric = ['distance_cost']
    boolean = ['complete', 'tw_feasible', 'capacity_feasible', 'depot_return_feasible']
    count = ['n_unserved', 'n_duplicate']
    accounting = []
    if objective == 'coldchain':
        required += ['temperature_hard_feasible', 'all_orders_picked',
                     'all_cargo_delivered_to_depot', 'terminal_manifests_empty',
                     'distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost']
        numeric += ['distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost']
        boolean += ['temperature_hard_feasible', 'all_orders_picked',
                    'all_cargo_delivered_to_depot', 'terminal_manifests_empty']
        accounting = ['trace_accounting_consistent', 'distance_accounting_consistent']

    # accounting 不一致 = 协议错误（账本对不上是 bug，不是合法路由失败）
    for k in accounting:
        if k not in outcome or outcome[k] is None:
            return f'missing required field: {k}'
        if not _is_bool(outcome[k]) or not bool(outcome[k]):
            return f'accounting inconsistent: {k}={outcome[k]!r}'

    # 缺字段
    for k in required:
        if k not in outcome or outcome[k] is None:
            return f'missing required field: {k}'

    # 数值有限
    for k in numeric:
        v = outcome[k]
        if _is_bool(v) or not isinstance(v, (int, float, np.integer, np.floating)):
            return f'non-numeric field: {k}={v!r}'
        if not math.isfinite(float(v)):
            return f'non-finite value in {k}: {v}'

    # 计数字段非负整数（不得用 0 代替缺失；小数计数 = 协议错误，不截断）
    for k in count:
        v = outcome[k]
        if _is_bool(v) or not isinstance(v, (int, float, np.integer, np.floating)):
            return f'non-numeric count field: {k}={v!r}'
        fv = float(v)
        if not math.isfinite(fv) or fv < 0:
            return f'invalid count field: {k}={v}'
        if fv != int(fv):
            return f'non-integer count field: {k}={v}'

    # repair 审计字段：有限非负整数；strict_repair 下必须存在
    for k in ('ownership_violations', 'terminal_unresolved'):
        v = outcome.get(k)
        if v is None:
            if strict_repair:
                return f'missing required repair field: {k}'
            continue
        if _is_bool(v) or not isinstance(v, (int, float, np.integer, np.floating)):
            return f'non-numeric repair field: {k}={v!r}'
        fv = float(v)
        if not math.isfinite(fv) or fv < 0 or fv != int(fv):
            return f'invalid repair field: {k}={v}'

    # ownership 破坏 = 协议错误
    own = outcome.get('ownership_violations')
    if own is not None and int(own) > 0:
        return f'ownership_violations: {own}'

    # 布尔字段类型
    for k in boolean:
        if not _is_bool(outcome[k]):
            return f'non-boolean field: {k}={outcome[k]!r}'

    return None


def hard_vector_from_outcome(outcome) -> dict:
    """从 evaluate_coldchain_trace / _eval 的 outcome dict 生成 outcome-level hard vector。

    计数字段用精确相等（不 int() 截断），缺失字段判 False（不得默认 0 掩盖缺字段）。
    注意：本函数只做「服务可行性」布尔化；NaN/Inf/缺字段/accounting 一致性应由
    outcome_protocol_error 先行拦截。
    """
    return {
        'complete': bool(outcome.get('complete')),
        'n_unserved_zero': outcome.get('n_unserved') == 0,
        'n_duplicate_zero': outcome.get('n_duplicate') == 0,
        'tw_feasible': bool(outcome.get('tw_feasible')),
        'capacity_feasible': bool(outcome.get('capacity_feasible')),
        'depot_return_feasible': bool(outcome.get('depot_return_feasible')),
        'temperature_hard_feasible': bool(outcome.get('temperature_hard_feasible')),
        'all_orders_picked': bool(outcome.get('all_orders_picked')),
        'all_cargo_delivered_to_depot': bool(outcome.get('all_cargo_delivered_to_depot')),
        'terminal_manifests_empty': bool(outcome.get('terminal_manifests_empty')),
        'trace_accounting_consistent': bool(outcome.get('trace_accounting_consistent')),
        'distance_accounting_consistent': bool(outcome.get('distance_accounting_consistent')),
    }


def hard_pass(outcome) -> bool:
    """outcome 是否完整 hard-pass（所有 outcome-level 硬约束均满足）。"""
    return all(hard_vector_from_outcome(outcome).values())


def service_ok(outcome, objective='distance') -> bool:
    """服务优先 blocking Gate（与 hard_pass 一致；distance 忽略冷链专属项）。

    coldchain 额外要求温度硬约束、全部入舱/返仓、manifest 清空、trace/distance accounting
    一致。distance 只需 complete + TW/cap/return（冷链字段缺省为 True，因无冷链合同）。
    repair-level 的 terminal_unresolved（若已采集）> 0 判 service fail。
    """
    if int(outcome.get('terminal_unresolved', 0)) > 0:
        return False
    hv = hard_vector_from_outcome(outcome)
    if objective != 'coldchain':
        return (hv['complete'] and hv['n_unserved_zero'] and hv['n_duplicate_zero']
                and hv['tw_feasible'] and hv['capacity_feasible']
                and hv['depot_return_feasible'])
    return all(hv.values())


def improvement_tau(keep_cost):
    """严格改善容差（阶段 C 步骤 4）：低于噪声水平的「改善」不改变当前计划。"""
    return 1e-9 + 1e-9 * max(1.0, abs(float(keep_cost)))


def select_improving(cost_id_pairs, keep_cost, tau=None):
    """在严格优于 keep 的候选中取 (cost, action_id) 最小者；无则 None（KEEP/DEFER）。

    τ 只用于判断「是否值得离开 KEEP」；改善候选之间按实际 cost + 稳定 action_id 选择，
    不依赖候选遍历顺序。cost_id_pairs 为 [(cost, action_id), ...]。返回 (cost, action_id)
    或 None。防御性检查：keep 与候选 cost 必须有限，禁止 NaN/Inf 进入排序。
    """
    if tau is None:
        tau = improvement_tau(keep_cost)
    kc = float(keep_cost)
    if not math.isfinite(kc):
        raise ValueError(f"keep_cost non-finite: {keep_cost}")
    improving = []
    for cost, aid in cost_id_pairs:
        c = float(cost)
        if not math.isfinite(c):
            raise ValueError(f"candidate cost non-finite for {aid}: {cost}")
        if c < kc - tau:
            improving.append((c, str(aid)))
    if not improving:
        return None
    return min(improving, key=lambda x: (x[0], x[1]))
