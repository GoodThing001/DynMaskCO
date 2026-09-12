"""contract_l3_spec.py — L3 合同迁移的严格判等规范（纯逻辑，无副作用）。

L3 是**精确等价 Gate**，不是统计优效性实验。判等原则：
  - 只允许「身份 + 计时 + 运行标识」字段不同；
  - 其余字段（决策、轨迹、结果、安全、非计时统计）必须逐字段 bit 级一致；
  - 浮点不设容差（旧/新 runtime manifest 相同、执行代码与环境相同，应精确一致）；
  - 缺字段不得默认为相等，必须报告 JSON path；
  - `decision_hash` 单独严格校验（必须相同）。

行为必比字段（任何差异 → BEHAVIOR_CHANGED）：
  method_name / method_revision / adapter_revision / instance_id / instance_seed /
  scene_instance_id / objective / random_seed / schema_version /
  events / actions / execution_trace / outcome / hard_vector / audit / protocol / stats

仅允许排除（身份/计时/运行标识）：
  code_hash / data_hash / profile_hash / checkpoint_hash / artifact_hash
  runtime_s / model_runtime_s / adapter_runtime_s / inference_time_s
  run_id / run_uuid / generated_at / completed_at / recorded_at / timestamp / host
"""
import json

# 允许在任意深度跳过的字段（身份 + 计时 + 运行标识）。
# 注意：decision_hash 不在此列——它必须相同，单独严格校验。
EXCLUDE_KEYS = frozenset({
    'code_hash', 'data_hash', 'profile_hash', 'checkpoint_hash', 'artifact_hash',
    'runtime_s', 'model_runtime_s', 'adapter_runtime_s', 'inference_time_s',
    'runtime_total_s', 'wall_time_s', 'duration_ms',
    'run_id', 'run_uuid', 'generated_at', 'completed_at', 'recorded_at',
    'timestamp', 'host', 'started_at', 'finished_at',
})

# 9 cell 的精确集合（type, edod）
EXPECTED_CELLS = frozenset({
    ('R1', 0.2), ('R1', 0.5), ('R1', 0.8),
    ('C1', 0.2), ('C1', 0.5), ('C1', 0.8),
    ('RC1', 0.2), ('RC1', 0.5), ('RC1', 0.8),
})


class BehaviorDiff:
    """单处行为差异的规范化描述。"""

    def __init__(self, path, old, new):
        self.path = path
        self.old = old
        self.new = new

    def to_dict(self):
        return {'path': self.path, 'old': self.old, 'new': self.new}

    def __repr__(self):
        return f'BehaviorDiff({self.path}: {self.old!r} -> {self.new!r})'


def _norm(v):
    """把 numpy 标量 / tuple / set 规整为可比较/可序列化的纯 Python 值。"""
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _norm(v.tolist())
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, set):
        return sorted(_norm(x) for x in v)
    return v


def _diff(o, n):
    return _norm(o), _norm(n)


def compare_behavior(old, new):
    """严格判等两份 instance record。

    返回 (verdict, diffs)：
      verdict ∈ {'BEHAVIOR_UNCHANGED', 'BEHAVIOR_CHANGED', 'MISSING_FIELD'}
      diffs  = [BehaviorDiff, ...]
    任一行为字段不同 → BEHAVIOR_CHANGED；任一单边缺字段 → MISSING_FIELD。
    decision_hash 不在 EXCLUDE_KEYS，故自然参与逐字段比较（必须相同）。
    """
    diffs = []
    _recurse(old, new, '$', diffs)
    missing = any(('<missing>' in d.path) for d in diffs)
    if missing:
        verdict = 'MISSING_FIELD'
    elif diffs:
        verdict = 'BEHAVIOR_CHANGED'
    else:
        verdict = 'BEHAVIOR_UNCHANGED'
    return verdict, diffs


def _recurse(old, new, path, diffs):
    if isinstance(old, dict) and isinstance(new, dict):
        keys = set(old.keys()) | set(new.keys())
        for k in sorted(keys):
            if k in EXCLUDE_KEYS:
                continue
            kp = f'{path}.{k}'
            if k not in old:
                diffs.append(BehaviorDiff(f'{kp}<missing>', None, _norm(new[k])))
            elif k not in new:
                diffs.append(BehaviorDiff(f'{kp}<missing>', _norm(old[k]), None))
            else:
                _recurse(old[k], new[k], kp, diffs)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append(BehaviorDiff(f'{path}.length', len(old), len(new)))
        for i in range(min(len(old), len(new))):
            _recurse(old[i], new[i], f'{path}[{i}]', diffs)
    else:
        oo, nn = _diff(old, new)
        if oo != nn:
            diffs.append(BehaviorDiff(path, oo, nn))


def validate_cell_set(cells):
    """校验 9 cell 集合：精确 R1/C1/RC1 × 0.2/0.5/0.8，无缺失、无重复。

    返回 (ok, problems)。
    """
    problems = []
    seen = set()
    for c in cells:
        key = (c.get('type'), float(c.get('edod')))
        if key in seen:
            problems.append(f'重复 cell: {key}')
        seen.add(key)
    missing = sorted(EXPECTED_CELLS - seen, key=lambda t: (t[0], t[1]))
    extra = sorted(seen - EXPECTED_CELLS, key=lambda t: (t[0], t[1]))
    if missing:
        problems.append(f'缺失 cell: {missing}')
    if extra:
        problems.append(f'多余 cell: {extra}')
    return (not problems), problems


def dumps(obj):
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
