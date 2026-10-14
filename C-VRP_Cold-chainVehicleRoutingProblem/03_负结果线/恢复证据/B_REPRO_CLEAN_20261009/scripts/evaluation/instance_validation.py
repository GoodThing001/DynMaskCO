"""共享严格实例验收（控制/分析层，不在「计算版本」冻结清单内）。

驱动在复用已有实例前、聚合器在逐实例 Gate 复算时，共用同一验收口径，避免「完整才复用」
与「失败即停」在两处各自实现导致口径漂移。本模块只依赖 hard_gate（计算层）的
outcome_protocol_error / service_ok，不改变任何计算行为。

状态分类：
  missing        记录缺失（可补算）
  error          记录含 error 字段（worker 异常，停止）
  protocol_fail  outcome 协议错误 / 字段缺失 / 计数不一致 / inst_idx 不符 / 非退化违反（停止）
  service_fail   service Gate 未通过（合法不可接受候选，停止）
  missing_local  local 应采集但字段缺失（仅 require_local 时；停止）
  valid          完整、可安全用于汇总
"""
import math

from hard_gate import outcome_protocol_error, service_ok


def _is_int(v):
    # 严格整数：bool 不是 int；float 即便整值也拒绝（防 1.0 静默截断）
    return isinstance(v, int) and not isinstance(v, bool)


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def validate_local(loc):
    """local 子结构完整性：event_deltas 与 records 的 eligible 事件逐项对账。

    event_deltas 只在「有客户的决策点」追加，records 每个决策点都追加；因此对每个
    n_customer_deltas>0 的记录，其 mean_customer_delta 必须与对应的 event_deltas 逐项一致。
    计数字段必须是合法非负严格整数（不接受负数/小数/布尔），数值字段必须是有限非布尔数，
    records 元素必须是 dict。
    """
    if not isinstance(loc, dict):
        return 'local 非 dict'
    ed = loc.get('event_deltas')
    recs = loc.get('records')
    if not isinstance(ed, list) or not isinstance(recs, list):
        return 'event_deltas / records 缺失或非 list'

    def _is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    ed_idx = 0
    for j, r in enumerate(recs):
        if not isinstance(r, dict):
            return f'record[{j}] 非 dict：{r!r}'
        n = r.get('n_customer_deltas')
        if not _is_int(n):
            return f'record[{j}] n_customer_deltas 非严格整数：{n!r}'
        if n < 0:
            return f'record[{j}] n_customer_deltas 为负：{n}'
        nc = r.get('n_customers')
        if not _is_int(nc) or nc < 0:
            return f'record[{j}] n_customers 非非负严格整数：{nc!r}'
        if n > nc:
            return f'record[{j}] n_customer_deltas({n}) > n_customers({nc})'
        if n == 0:
            continue
        mean = r.get('mean_customer_delta')
        if not _is_num(mean) or not math.isfinite(float(mean)):
            return f'record[{j}] mean_customer_delta 非有限非布尔数：{mean!r}'
        if ed_idx >= len(ed):
            return 'event_deltas 数量少于 eligible 记录'
        if not _is_num(ed[ed_idx]) or not math.isfinite(float(ed[ed_idx])):
            return f'event_deltas[{ed_idx}] 非有限非布尔数：{ed[ed_idx]!r}'
        if abs(float(mean) - float(ed[ed_idx])) > 1e-9:
            return f'event_deltas[{ed_idx}] 与 record[{j}].mean_customer_delta 不一致：' \
                   f'{ed[ed_idx]!r} vs {mean!r}'
        ed_idx += 1
    if ed_idx != len(ed):
        return f'event_deltas({len(ed)}) 与 eligible 记录({ed_idx}) 数量不一致'
    return None


def validate_instance_record(rec, idx, objective='coldchain', require_local=False,
                             expected_code_sha256=None, expected_contract_hash=None):
    """返回 (state, err)。rec 为已解析 inst JSON 或 None（缺失）。"""
    if rec is None:
        return 'missing', None
    if not isinstance(rec, dict):
        return 'protocol_fail', '记录非 dict'
    if rec.get('error'):
        return 'error', str(rec['error'])[:200]

    raw = rec.get('inst_idx')
    if not _is_int(raw):
        return 'protocol_fail', f'inst_idx 非严格整数：{raw!r}'
    if raw != idx:
        return 'protocol_fail', f'inst_idx 不符：{raw} != {idx}'

    if expected_code_sha256 is not None:
        actual = rec.get('code_sha256')
        if actual != expected_code_sha256:
            return 'protocol_fail', f'code_sha256 与冻结计算版本不符：{actual} != ' \
                                    f'{expected_code_sha256[:12]}...'

    for side in ('baseline', 'oracle'):
        out = rec.get(side)
        if not isinstance(out, dict):
            return 'protocol_fail', f'{side} 缺失'
        if expected_contract_hash is not None and out.get('contract_hash') != expected_contract_hash:
            return 'protocol_fail', f'{side} contract_hash 与 frozen effective_contract_hash 不符'
        err = outcome_protocol_error(out, objective, strict_repair=True)
        if err is not None:
            return 'protocol_fail', f'{side} protocol: {err}'
        if not service_ok(out, objective):
            return 'service_fail', f'{side} service fail'
        for k in ('coldchain_cost', 'distance_cost', 'quality_loss', 'energy_kwh'):
            if k in out and not math.isfinite(float(out[k])):
                return 'protocol_fail', f'{side} non-finite {k}'

    # 逐实例终局非退化：oracle 不得比 baseline 更差（固定浮点容差）
    b_cost = _cost(rec['baseline'], objective)
    o_cost = _cost(rec['oracle'], objective)
    if o_cost - b_cost > 1e-6 * max(1.0, abs(b_cost)):
        return 'protocol_fail', f'非退化违反：Δ={o_cost - b_cost}'

    if require_local or rec.get('local') is not None:
        loc = rec.get('local')
        if not isinstance(loc, dict):
            return 'missing_local', 'local 字段缺失'
        err = validate_local(loc)
        if err is not None:
            return 'protocol_fail', f'local: {err}'
    return 'valid', None
