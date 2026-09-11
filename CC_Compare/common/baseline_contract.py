"""统一 baseline 输出合同（B1）：实例记录 schema + 确定性 hash 工具。

三层结构（对应规划 §六）：
  - identity  顶层身份：method / checkpoint / code / data / profile hash、instance、seed、
              objective、runtime budget；
  - events    每个决策点：event_id / clock / revealed ids / visible/served/state hash /
              replan 车辆 / 每车状态与 suffix 前后、模型输入客户集合、模型耗时、fallback；
  - actions   每个提交动作：KEEP / DEFER / INSERT / NEW_ROUTE / COMMIT / CLOSE / WAIT，
              suffix 前后、certificate（TW/capacity/return）、写回校验；
  - execution_trace  每车 dispatch/services/return/unload 明细（trace_export 生成）；
  - outcome   终局：完整 hard vector + D/Q/E/J（共同 evaluator 重算，不信任自报）。

schema_version 变化 = 正式协议变更，必须显式升级并重跑 B1 Gate。
"""
import hashlib
import json

import numpy as np

RECORD_SCHEMA_VERSION = 'cc-compare-baseline-v1'

IDENTITY_FIELDS = (
    'schema_version', 'method_name', 'method_revision', 'adapter_revision',
    'checkpoint_hash', 'code_hash', 'data_hash', 'profile_hash',
    'instance_id', 'instance_seed', 'objective', 'runtime_budget', 'random_seed',
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(arr) -> str:
    """数组内容的确定性 hash（与 dtype/shape 绑定）。"""
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:32]


def list_hash(items) -> str:
    """可 JSON 序列化列表的确定性 hash。"""
    return sha256_hex(json.dumps(items, sort_keys=True, default=str).encode('utf-8'))[:32]


_HASH_EXCLUDE_TOP = ('code_hash', 'data_hash', 'profile_hash', 'checkpoint_hash',
                    'runtime_s', 'decision_hash', 'artifact_hash')
_HASH_EXCLUDE_DEEP = ('model_runtime_s',)   # 计时字段对决策内容不敏感（determinism 除外）


def _strip_timing(obj):
    """深拷贝并剔除计时字段（模型/运行耗时不影响决策身份）。"""
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items()
                if k not in _HASH_EXCLUDE_DEEP}
    if isinstance(obj, list):
        return [_strip_timing(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_timing(x) for x in obj)
    return obj


def decision_hash(record) -> str:
    """决策确定性 hash（deterministic parity 用）：只对决策与执行内容敏感。

    排除：身份字段（code/data/profile/checkpoint hash）、计时字段
    （runtime_s / model_runtime_s）、决策身份本身。
    """
    body = {k: v for k, v in record.items() if k not in _HASH_EXCLUDE_TOP}
    body = _strip_timing(body)
    payload = json.dumps(body, sort_keys=True, default=_json_default)
    return sha256_hex(payload.encode('utf-8'))


def artifact_hash(record) -> str:
    """复现身份 hash：绑定完整身份（方法/代码/数据/权重/profile）+ 决策内容。

    用于复现审计（同一 artifact_hash = 同一方法版本在同一数据/权重/代码下
    产生同一决策内容）。计时字段除外（不可复现）：顶层 runtime_s 与事件层
    model_runtime_s 均不参与。
    """
    body = _strip_timing(record)
    body = {k: v for k, v in body.items()
            if k not in ('artifact_hash', 'runtime_s')}
    payload = json.dumps(body, sort_keys=True, default=_json_default)
    return sha256_hex(payload.encode('utf-8'))


# 兼容旧名（B1 初版曾用 record_state_hash / record_hash）
record_state_hash = decision_hash


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def new_record(identity: dict) -> dict:
    """建空记录并强制 identity 字段完整（schema_version 由合同强制，不接受覆盖）。"""
    identity = dict(identity)
    identity['schema_version'] = RECORD_SCHEMA_VERSION
    missing = [k for k in IDENTITY_FIELDS if k not in identity]
    if missing:
        raise ValueError(f'identity 缺字段: {missing}')
    rec = dict(identity)
    rec.update({
        'events': [],
        'actions': [],
        'execution_trace': {},
        'outcome': None,
        'hard_vector': None,
        'audit': None,
        'protocol': {'error': None, 'checks': []},
        'runtime_s': 0.0,
        'decision_hash': None,
        'artifact_hash': None,
    })
    return rec


def commit_action_record(event_id, vehicle_id, customer_id, action_type, position,
                         suffix_before, suffix_after, action_layer, certificate=None,
                         writeback_ok=None, commit_info=None):
    """标准 action 记录。action_type ∈
    KEEP / DEFER / INSERT / NEW_ROUTE / COMMIT / CLOSE / WAIT。
    action_layer：'execution'（env 实际执行：COMMIT/CLOSE/WAIT，每决策点每车一条）
                'plan_diff'（fleet plan 前后 diff 的审计性解释：KEEP/DEFER/INSERT/
                NEW_ROUTE，每个受影响客户一条）。
    两层不得混合统计动作数量或运行时间。
    """
    rec = {
        'event_id': int(event_id),
        'vehicle_id': int(vehicle_id),
        'customer_id': None if customer_id is None else int(customer_id),
        'action_type': action_type,
        'action_layer': action_layer,
        'position': None if position is None else int(position),
        'suffix_before': [int(x) for x in suffix_before],
        'suffix_after': [int(x) for x in suffix_after],
        'action_hash': list_hash([event_id, vehicle_id, customer_id, action_type,
                                  position, suffix_before, suffix_after]),
    }
    if certificate is not None:
        rec['certificate'] = {
            'feasible': bool(certificate.get('feasible')),
            'reason': certificate.get('reason'),
            'tw_slack': _finite_or_none(certificate.get('tw_slack')),
            'cap_slack': _finite_or_none(certificate.get('cap_slack')),
            'return_slack': _finite_or_none(certificate.get('return_slack')),
            # 证书验证的是「变更后整条车辆 suffix」，不是单客户插入的逐步可行性
            'certificate_scope': 'final_vehicle_suffix',
        }
    if writeback_ok is not None:
        # 写回语义：最终 fleet plan 与 proposal 一致（Bridge 写回后重建 plan 核对）
        rec['writeback_ok'] = bool(writeback_ok)
    if commit_info is not None:
        rec['commit'] = commit_info
    return rec


def _finite_or_none(x):
    if x is None:
        return None
    f = float(x)
    return f if np.isfinite(f) else None


def event_record(snapshot, replan_ids, vehicles):
    """从 recourse snapshot 生成事件层记录。

    model_input_customers / model_runtime_s / budget_exceeded / fallback_triggered
    由 runner 在 run 结束后按 event_id 从 bridge plan 记录匹配填充
    （snapshot 时本事件 plan 尚未执行，不能读 adapter 上一事件的陈旧值）。
    """
    from recourse_snapshot import snapshot_state_hash

    visible_ids = [int(c) for c in snapshot['customer_universe']
                   if bool(snapshot['visible_mask'][int(c)])]
    vlist = []
    for v in vehicles:
        vlist.append({
            'vehicle_id': int(v.vehicle_id),
            'status': str(v.status),
            'anchor_node': int(v.current_node),
            'ready_time': float(v.ready_time),
            'load': float(v.current_load),
            'committed_next': (None if v.committed_next is None else int(v.committed_next)),
            'mutable_suffix_before': [int(x) for x in v.mutable_suffix],
        })
    rec = {
        'event_id': int(snapshot['event_id']),
        'clock': float(snapshot['clock']),
        'revealed_customer_ids': [],      # 由 runner 与上一事件 diff 后填充
        'visible_customer_ids': [int(x) for x in visible_ids],
        'served_customer_ids': sorted(int(c) for c in snapshot['customer_universe']
                                      if bool(snapshot['served_mask'][int(c)])),
        'visible_ids_hash': array_hash(np.asarray(visible_ids, dtype=np.int32)),
        'served_mask_hash': array_hash(np.asarray(snapshot['served_mask'], dtype=bool)),
        'state_hash': snapshot_state_hash(snapshot),
        'replan_vehicle_ids': sorted(int(x) for x in replan_ids),
        'vehicles': vlist,
        'model_input_customers': [],      # post-hoc 填充（见 docstring）
        'model_runtime_s': 0.0,
        'budget_exceeded': False,
        'fallback_triggered': False,
    }
    return rec
