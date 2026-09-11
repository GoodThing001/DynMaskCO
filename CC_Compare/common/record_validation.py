"""记录校验（B1）：损坏 / 缺字段 / NaN / Inf 记录必须被拒绝。

分层校验：
  1) identity 层：必需字段、类型、hash 非空；
  2) event 层：每事件必需字段、数值有限、visible/served/state hash 非空；
  3) action 层：action_type 合法、suffix 元素为 int、certificate 字段合法；
  4) outcome 层：复用项目 hard_gate.outcome_protocol_error（正式口径，禁止自造）；
  5) hard_vector 层：与 outcome 派生的 hard vector 必须一致（不得自报）。

返回 (error: str | None, checks: list[str])。error 非 None = 记录被拒绝。
"""
import math

import numpy as np

from hard_gate import outcome_protocol_error, hard_vector_from_outcome
import baseline_contract as bc

VALID_ACTION_TYPES = {'KEEP', 'DEFER', 'INSERT', 'NEW_ROUTE',
                      'COMMIT', 'CLOSE', 'WAIT'}


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _is_int(v):
    if not _is_number(v):
        return False
    f = float(v)
    return math.isfinite(f) and f == int(f)


def _finite(v):
    return _is_number(v) and math.isfinite(float(v))


def validate_instance_record(record, objective='coldchain', strict_repair=True):
    """校验完整实例记录。返回 (error, checks)。"""
    checks = []

    def fail(msg):
        return msg, checks

    def ok(msg):
        checks.append(msg)

    if not isinstance(record, dict):
        return fail('record 不是 dict')

    # ---- 1) identity 层 ----
    if record.get('schema_version') != bc.RECORD_SCHEMA_VERSION:
        return fail(f"schema_version 不匹配: {record.get('schema_version')!r} "
                    f"(期望 {bc.RECORD_SCHEMA_VERSION})")
    ok('schema_version')
    for k in bc.IDENTITY_FIELDS:
        if k not in record:
            return fail(f'identity 缺字段: {k}')
    ok('identity 字段齐全')
    if not _is_int(record['instance_id']) or record['instance_id'] < 0:
        return fail(f"instance_id 非法: {record['instance_id']!r}")
    if not _is_int(record['instance_seed']) or not _is_int(record['random_seed']):
        return fail('instance_seed / random_seed 必须是有限整数')
    if record['objective'] not in ('distance', 'coldchain'):
        return fail(f"objective 非法: {record['objective']!r}")
    if record['runtime_budget'] is not None and not _finite(record['runtime_budget']):
        return fail(f"runtime_budget 非有限: {record['runtime_budget']!r}")
    for k in ('method_name', 'method_revision', 'adapter_revision'):
        if not isinstance(record[k], str) or not record[k]:
            return fail(f'{k} 必须为非空字符串')
    for k in ('checkpoint_hash', 'code_hash', 'data_hash', 'profile_hash'):
        if not isinstance(record[k], str) or not record[k]:
            return fail(f'{k} 必须为非空字符串')
    ok('identity 值合法')

    # ---- 2) event 层 ----
    events = record.get('events')
    if not isinstance(events, list) or not events:
        return fail('events 缺失或为空（必须至少含 t=0 决策点）')
    seen_eids = []
    prev_clock = -1.0
    for i, ev in enumerate(events):
        tag = f'events[{i}]'
        if not isinstance(ev, dict):
            return fail(f'{tag} 不是 dict')
        for k in ('event_id', 'clock', 'visible_ids_hash', 'served_mask_hash',
                  'state_hash', 'replan_vehicle_ids', 'vehicles',
                  'model_input_customers', 'model_runtime_s', 'fallback_triggered',
                  'visible_customer_ids', 'served_customer_ids'):
            if k not in ev:
                return fail(f'{tag} 缺字段: {k}')
        if not _is_int(ev['event_id']) or ev['event_id'] < 0:
            return fail(f'{tag} event_id 非法: {ev["event_id"]!r}')
        seen_eids.append(ev['event_id'])
        if not _finite(ev['clock']) or ev['clock'] < 0:
            return fail(f'{tag} clock 非法: {ev["clock"]!r}')
        if ev['clock'] < prev_clock - 1e-9:
            return fail(f'{tag} clock 非单调递减: {ev["clock"]} < {prev_clock}')
        prev_clock = ev['clock']
        if not _finite(ev['model_runtime_s']) or ev['model_runtime_s'] < 0:
            return fail(f'{tag} model_runtime_s 非法: {ev["model_runtime_s"]!r}')
        if not isinstance(ev['fallback_triggered'], bool):
            return fail(f'{tag} fallback_triggered 非 bool')
        if 'budget_exceeded' in ev and not isinstance(ev['budget_exceeded'], bool):
            return fail(f'{tag} budget_exceeded 非 bool')
        for hk in ('visible_ids_hash', 'served_mask_hash', 'state_hash'):
            if not isinstance(ev[hk], str) or not ev[hk]:
                return fail(f'{tag} {hk} 为空')
        if not isinstance(ev['replan_vehicle_ids'], list):
            return fail(f'{tag} replan_vehicle_ids 非 list')
        if len(ev['replan_vehicle_ids']) != len(set(ev['replan_vehicle_ids'])):
            return fail(f'{tag} replan_vehicle_ids 含重复')
        for vid in ev['replan_vehicle_ids']:
            if not _is_int(vid) or vid < 0:
                return fail(f'{tag} replan_vehicle_ids 含非法 id: {vid!r}')
        if not isinstance(ev['vehicles'], list):
            return fail(f'{tag} vehicles 非 list')
        vids_in_event = []
        for vi, v in enumerate(ev['vehicles']):
            for k in ('vehicle_id', 'status', 'anchor_node', 'ready_time', 'load',
                      'committed_next', 'mutable_suffix_before'):
                if k not in v:
                    return fail(f'{tag}.vehicles[{vi}] 缺字段: {k}')
            if not _is_int(v['vehicle_id']):
                return fail(f'{tag}.vehicles[{vi}] vehicle_id 非法')
            vids_in_event.append(v['vehicle_id'])
            if not _finite(v['ready_time']) or not _finite(v['load']):
                return fail(f'{tag}.vehicles[{vi}] ready_time/load 非有限')
            for x in v['mutable_suffix_before']:
                if not _is_int(x):
                    return fail(f'{tag}.vehicles[{vi}] suffix 含非整数: {x!r}')
        if len(vids_in_event) != len(set(vids_in_event)):
            return fail(f'{tag} vehicles 含重复 vehicle_id')
        # replan ids 必须存在于本事件车辆中
        ev_vids = set(vids_in_event)
        if not set(ev['replan_vehicle_ids']) <= ev_vids:
            return fail(f'{tag} replan_vehicle_ids 含不存在车辆: '
                        f'{set(ev["replan_vehicle_ids"]) - ev_vids}')
        # model_input_customers：唯一合法整数 ⊆ visible 且不 ∩ served
        mic = ev['model_input_customers']
        if not isinstance(mic, list):
            return fail(f'{tag} model_input_customers 非 list')
        for c in mic:
            if not _is_int(c) or c <= 0:
                return fail(f'{tag} model_input_customers 含非法客户: {c!r}')
        if len(mic) != len(set(mic)):
            return fail(f'{tag} model_input_customers 含重复')
        vis_set = set(int(c) for c in ev['visible_customer_ids'])
        if not set(int(c) for c in mic) <= vis_set:
            return fail(f'{tag} model_input_customers 超出 visible: '
                        f'{set(mic) - vis_set}')
        served_set = set(int(c) for c in ev['served_customer_ids'])
        if set(int(c) for c in mic) & served_set:
            return fail(f'{tag} model_input_customers 含已服务客户: '
                        f'{set(mic) & served_set}')
        # pool/protected 分区：visible_unserved == protected ∪ pool（无交叉）。
        # 无 replan 的决策点（plan persistence）无决策视图 → 两字段为 None，跳过。
        if ev.get('pool_customer_ids') is not None or ev.get('protected_customer_ids') is not None:
            if ev.get('pool_customer_ids') is None or ev.get('protected_customer_ids') is None:
                return fail(f'{tag} pool/protected 必须同时存在或同时为 None')
            pool = set(int(c) for c in ev['pool_customer_ids'])
            prot = set(int(c) for c in ev['protected_customer_ids'])
            vis_unserved = vis_set - served_set
            if pool & prot:
                return fail(f'{tag} pool 与 protected 交叉: {pool & prot}')
            if (pool | prot) != vis_unserved:
                return fail(f'{tag} pool ∪ protected != visible_unserved: '
                            f'missing={(vis_unserved - (pool | prot))} '
                            f'extra={((pool | prot) - vis_unserved)}')
    if len(seen_eids) != len(set(seen_eids)):
        return fail(f'event_id 重复: {seen_eids}')
    ok(f'event 层合法（{len(events)} 个决策点，id 单调唯一、clock 非递减、'
       'input 子集校验通过）')

    # ---- 3) action 层（execution 与 plan_diff 分层校验）----
    actions = record.get('actions')
    if not isinstance(actions, list):
        return fail('actions 非 list')
    event_ids = {ev['event_id']: ev for ev in events}
    EXEC_TYPES = {'COMMIT', 'CLOSE', 'WAIT'}
    DIFF_TYPES = {'KEEP', 'DEFER', 'INSERT', 'NEW_ROUTE'}
    for i, a in enumerate(actions):
        tag = f'actions[{i}]'
        if not isinstance(a, dict):
            return fail(f'{tag} 不是 dict')
        for k in ('event_id', 'vehicle_id', 'action_type', 'action_layer',
                  'suffix_before', 'suffix_after'):
            if k not in a:
                return fail(f'{tag} 缺字段: {k}')
        if a['action_type'] not in VALID_ACTION_TYPES:
            return fail(f'{tag} action_type 非法: {a["action_type"]!r}')
        layer = a['action_layer']
        if layer == 'execution':
            if a['action_type'] not in EXEC_TYPES:
                return fail(f'{tag} execution 层 action_type 非法: {a["action_type"]!r}')
        elif layer == 'plan_diff':
            if a['action_type'] not in DIFF_TYPES:
                return fail(f'{tag} plan_diff 层 action_type 非法: {a["action_type"]!r}')
            if not a.get('writeback_ok'):
                return fail(f'{tag} plan_diff 动作 writeback_ok 必须为 True')
            if a['action_type'] in ('INSERT', 'NEW_ROUTE'):
                cert = a.get('certificate')
                if not cert or cert.get('certificate_scope') != 'final_vehicle_suffix':
                    return fail(f'{tag} INSERT/NEW_ROUTE 缺 certificate '
                                f'(scope=final_vehicle_suffix)')
                if not cert.get('feasible'):
                    return fail(f'{tag} INSERT/NEW_ROUTE certificate 不可行'
                                f'（客户在计划中但路线不可行）: reason={cert.get("reason")}')
        else:
            return fail(f'{tag} action_layer 非法: {layer!r}')
        if not _is_int(a['event_id']) or not _is_int(a['vehicle_id']):
            return fail(f'{tag} event_id/vehicle_id 非法')
        # 可解析性：event 存在，vehicle 存在于该事件
        ev = event_ids.get(a['event_id'])
        if ev is None:
            return fail(f'{tag} 引用不存在的事件 {a["event_id"]}')
        if a['vehicle_id'] not in {v['vehicle_id'] for v in ev['vehicles']}:
            return fail(f'{tag} vehicle {a["vehicle_id"]} 不在事件 {a["event_id"]} 中')
        if a['customer_id'] is not None:
            if not _is_int(a['customer_id']):
                return fail(f'{tag} customer_id 非法: {a["customer_id"]!r}')
            if a['action_type'] in DIFF_TYPES and a['customer_id'] > 0:
                if a['customer_id'] not in set(int(c) for c in ev['visible_customer_ids']):
                    return fail(f'{tag} 客户 {a["customer_id"]} 在事件 {a["event_id"]} '
                                f'不可见（plan_diff 动作引用未来客户）')
        for k in ('suffix_before', 'suffix_after'):
            if not isinstance(a[k], list) or any(not _is_int(x) for x in a[k]):
                return fail(f'{tag} {k} 非法: {a[k]!r}')
        if a['action_type'] == 'INSERT' and not _is_int(a.get('position')):
            return fail(f'{tag} INSERT 缺合法 position')
        if 'certificate' in a:
            c = a['certificate']
            if not isinstance(c, dict) or 'feasible' not in c:
                return fail(f'{tag} certificate 非法')
            if not isinstance(c['feasible'], bool):
                return fail(f'{tag} certificate.feasible 非 bool')
            for sk in ('tw_slack', 'cap_slack', 'return_slack'):
                if sk in c and c[sk] is not None and not _finite(c[sk]):
                    return fail(f'{tag} certificate.{sk} 非有限: {c[sk]!r}')
        if 'writeback_ok' in a and not isinstance(a['writeback_ok'], bool):
            return fail(f'{tag} writeback_ok 非 bool')
    n_exec = sum(1 for a in actions if a['action_layer'] == 'execution')
    n_diff = sum(1 for a in actions if a['action_layer'] == 'plan_diff')
    ok(f'action 层合法（execution={n_exec} 条 / plan_diff={n_diff} 条，分层校验通过）')

    # ---- 4) outcome 层（项目正式口径）----
    outcome = record.get('outcome')
    if not isinstance(outcome, dict):
        return fail('outcome 缺失或非 dict')
    err = outcome_protocol_error(outcome, objective, strict_repair=strict_repair)
    if err is not None:
        return fail(f'outcome PROTOCOL_ERROR: {err}')
    ok('outcome 通过 outcome_protocol_error')
    # repair/audit 归属：公共 runner 审计所得真实整数，不允默认值掩盖
    if outcome.get('repair_applicable') not in (True, False):
        return fail(f"outcome.repair_applicable 非法: {outcome.get('repair_applicable')!r}")
    if not isinstance(outcome.get('audit_source'), str) or not outcome['audit_source']:
        return fail('outcome.audit_source 必须非空（所有权审计归属）')
    ok('outcome 审计归属明确（repair_applicable + audit_source）')

    # ---- 4b) audit 层 ----
    audit = record.get('audit')
    if not isinstance(audit, dict):
        return fail('audit 缺失（公共 ownership audit 结果必须落盘）')
    if not _is_int(audit.get('ownership_violations')) or audit['ownership_violations'] < 0:
        return fail(f"audit.ownership_violations 非法: {audit.get('ownership_violations')!r}")
    if audit['ownership_violations'] != int(outcome['ownership_violations']):
        return fail(f'audit 与 outcome ownership_violations 不一致: '
                    f'{audit["ownership_violations"]} vs {outcome["ownership_violations"]}')
    if not _is_int(audit.get('terminal_unresolved')) or audit['terminal_unresolved'] < 0:
        return fail(f"audit.terminal_unresolved 非法: {audit.get('terminal_unresolved')!r}")
    if audit['terminal_unresolved'] != int(outcome['terminal_unresolved']):
        return fail(f'audit 与 outcome terminal_unresolved 不一致: '
                    f'{audit["terminal_unresolved"]} vs {outcome["terminal_unresolved"]}')
    if not isinstance(audit.get('per_event'), list):
        return fail('audit.per_event 非 list')
    ok('audit 层与 outcome 一致')

    # ---- 5) hard vector 层：必须与 outcome 派生一致（不得自报）----
    hv = record.get('hard_vector')
    if not isinstance(hv, dict):
        return fail('hard_vector 缺失')
    derived = hard_vector_from_outcome(outcome)
    for k, v in derived.items():
        if hv.get(k) != v:
            return fail(f'hard_vector 与 outcome 不一致: {k} 记录={hv.get(k)!r} '
                        f'派生={v!r}')
    ok('hard_vector 与 outcome 一致')

    # ---- 6) 执行轨迹层 ----
    tr = record.get('execution_trace')
    if not isinstance(tr, dict):
        return fail('execution_trace 缺失或非 dict')
    if 'vehicles' not in tr or not isinstance(tr['vehicles'], list):
        return fail('execution_trace.vehicles 缺失')
    for vi, vt in enumerate(tr['vehicles']):
        tag = f'execution_trace.vehicles[{vi}]'
        if not isinstance(vt, dict) or not _is_int(vt.get('vehicle_id')):
            return fail(f'{tag} 非法')
        for sv in vt.get('services', []):
            for k in ('arrival_time', 'service_start', 'service_finish',
                      'segment_distance_km', 'segment_quality_loss',
                      'segment_energy_kwh'):
                if k not in sv or not _finite(sv[k]):
                    return fail(f'{tag}.services 字段 {k} 非法: {sv.get(k)!r}')
    ok('execution_trace 层合法')

    return None, checks
