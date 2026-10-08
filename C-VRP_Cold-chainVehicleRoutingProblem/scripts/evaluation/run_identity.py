# -*- coding: utf-8 -*-
"""Q-04（2026-09-29，严格版）：A-v1 运行身份封存共享模块。

步骤 2 门驱动与外部强求解器驱动共用同一套身份计算。严格规则：
- 合同身份 = 合同对象已有的完整 `contract_hash`（asdict 全字段序列化，
  覆盖开门热量/COP/时间单位等——不得用部分字段挑选）；
- 必需身份字段**双侧必须存在**，任一侧缺失即不通过（None==None 不算相等）；
- 计时/容量等行为配置（time_limit/capacity/num_vehicles/n_orders）纳入配对核对；
- 源码 hash 启动时封存 + 运行结束后复查，中途变更 → source_stable=False → 不通过。
"""
from __future__ import annotations

import hashlib
import io
import os

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 外部臂全集（含本驱动与求解器适配；run_identity.py 自身也封存——检查逻辑不可漂移）
SOURCE_FILES_EXTERNAL = [
    'scripts/evaluation/run_a1_external_accept.py',
    'scripts/evaluation/solver_accept_replanner.py',
    'scripts/evaluation/run_identity.py',
    'scripts/evaluation/scenario_saa.py',
    'scripts/evaluation/run_exp_energy_c0.py',
    'scripts/evaluation/coldchain_evaluator_a1.py',
    'scripts/evaluation/run_exp_reserve.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
]
# 步骤 2 门全集（含预算标定 run_exp_encoder_v3；run_identity.py 自身同样封存）
SOURCE_FILES_STEP2 = [
    'scripts/evaluation/run_a1_step2_gate.py',
    'scripts/evaluation/run_identity.py',
    'scripts/evaluation/scenario_saa.py',
    'scripts/evaluation/run_exp_energy_c0.py',
    'scripts/evaluation/coldchain_evaluator_a1.py',
    'scripts/evaluation/run_exp_reserve.py',
    'scripts/evaluation/run_exp_encoder_v3.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
]
# 两类运行共享的计算文件（env/认证/合同/数据生成）——跨运行配对时逐文件核对
SHARED_SOURCE_FILES = [
    'scripts/evaluation/scenario_saa.py',
    'scripts/evaluation/run_exp_energy_c0.py',
    'scripts/evaluation/coldchain_evaluator_a1.py',
    'scripts/evaluation/run_exp_reserve.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
]

# 配对必需字段（双侧缺失任一 → 不通过）
_REQUIRED_PAIRING_FIELDS = ('shared_source_sha256', 'contract_sha256', 'data_sha256',
                            'data_meta_sha256', 'budget_B', 'cooling_share',
                            'shared_config')


def _sha256_file(rel):
    h = hashlib.sha256()
    with io.open(os.path.join(_SCRIPTS, '..', rel), 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def source_identity(files):
    return {rel: _sha256_file(rel) for rel in sorted(files)}


def source_seal(files):
    """启动封存：运行前读一次源码 hash；`seal_end` 于运行结束后复查。"""
    return {'files': sorted(files), 'startup_sha256': source_identity(files),
            'end_sha256': None, 'stable': None}


def seal_end(seal):
    end = source_identity(seal['files'])
    seal['end_sha256'] = end
    seal['stable'] = (end == seal['startup_sha256'])
    return seal


def contract_identity(contract):
    """严格版：直接使用合同对象自带的完整 contract_hash（覆盖全部物理/单位/品质字段）。"""
    return contract.contract_hash


def dataset_identity(ds):
    h = hashlib.sha256()
    for k in sorted(ds):
        h.update(k.encode('utf-8'))
        h.update(np.asarray(ds[k]).tobytes())
    return h.hexdigest()


def dataset_meta_identity(ds):
    """数据 shape/dtype 显式身份（与字节 hash 并列记录，防形状/类型漂移被忽略）。"""
    return {k: {'shape': list(np.asarray(ds[k]).shape),
                'dtype': str(np.asarray(ds[k]).dtype)} for k in sorted(ds)}


def dataset_meta_sha256(ds):
    import json
    s = json.dumps(dataset_meta_identity(ds), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pairing_identity_check(ident, baseline):
    """基线缺失身份 → 不升级正式裁决；有身份则逐项核对必需字段（存在性 + 相等性）、
    共享行为配置、训练种子与源码稳定性。"""
    b_id = baseline.get('identity')
    if not b_id:
        return {'identity_verified': False,
                'reason': 'baseline_identity_missing（旧批次，不升级正式裁决）'}
    problems = []
    for k in _REQUIRED_PAIRING_FIELDS:
        if not ident.get(k) or not b_id.get(k):
            problems.append('%s_missing' % k)
        elif b_id.get(k) != ident.get(k):
            problems.append('%s_mismatch' % k)
    if not b_id.get('seeds', {}).get('train') or not ident.get('seeds', {}).get('train'):
        problems.append('train_seed_missing')
    elif int(b_id['seeds']['train']) != int(ident['seeds']['train']):
        problems.append('train_seed_mismatch')
    if b_id.get('source_stable') is not True:
        problems.append('baseline_source_not_stable')
    if ident.get('source_stable') is not True:
        problems.append('current_source_not_stable')
    return {'identity_verified': not problems, 'problems': problems}
