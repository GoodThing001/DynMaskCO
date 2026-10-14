#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的 **`nonlearning_solver` 身份闭包**（T1 的落地）。

Role C **没有模型权重**，所以它既不能伪造 `checkpoint_sha256`，也不能标 `rule`
（`rule` 会连求解器体检一起豁免）。它在 `qualification_gate.METHOD_CLASSES` 里的正确类型是
**`nonlearning_solver`**：**不要求** checkpoint，但**强制** `provider_dependency_sha256`
（源码 + 依赖闭包）与 `solver_config_sha256`（求解器/机制配置），且共享身份
（`source_stable` + 合同 + 数据 + 数据元）与配对身份**照查不豁免**。

本模块把两件事钉死：

1. **依赖清单**直接取自 `role_c_harness.ROLE_C_DEPENDENCY_ROLES`（每条都写了"为什么进来"），
   而不是另抄一份 —— 两份清单迟早会漂移，而漂移恰好会让闭包**看起来**通过。
   缺文件即报错（fail-closed）：空清单会返回 `None`，那**不是**"已闭合"。
2. **求解器配置**里同时包含**共享旋钮**（逐字段取自 `P1A_CONTROL_FROZEN.json`，不重打）
   与 **Role C 自己的机制旋钮**（`inner_gate` / λ 估计器 / 近视基准 / LS 规格哈希 /
   邻域预算 / 场景数与 `arm_seed`）。规格 §2.2.2 ④ 要求 `inner_gate` **计入**
   `solver_config_sha256` ⇒ 按结果偷换档位必然改变身份，不可能静默。

⚠️ **`arm_seed` 的来历（2026-10-14 用户裁定后**已补录**进冻结文件）**：
`P1A_CONTROL_FROZEN.json` 的 `frozen_config` 原先**漏记** `arm_seed`，而它**完整决定**场景流。
取值从产出该冻结证据的 runner 读出：`run_a1_step2_gate.py:51`
`ARM_SEEDS = {"uncond_hist": 7001, "cond_hist": 7002, "explicit_feat": 7003, ...}`
（`diag_local_timing.py:23` 一致）⇒ 主对照 `cond_hist` = **7002**，**不是**
`SaaReplanner` 的默认值 **7001**。**用户已裁定补录入冻结文件**（属补全可复核字段，
不是更改既有冻结值），故本模块现在**优先读冻结文件**的 `arm_seed`，并把"该值本轮之前
未被记录 + 出处"作为**补录留痕**写进配置（信息本身也进哈希，事后可审计）。
"""
from __future__ import annotations

import json
import os

import nonlearning_identity as ni
import qualification_gate as qg

import role_c_forward as fwd
import role_c_harness as hn
import role_c_local_search as ls
import role_c_opportunity_cost as oc

__all__ = [
    'IdentityError', 'ROLE_C_METHOD_CLASS', 'FROZEN_CONTROL_REL',
    'CONTROL_ARM_SEEDS_SOURCE', 'CONTROL_ARM_SEEDS', 'repo_root',
    'dependency_files', 'load_frozen_control', 'shared_config', 'frozen_anchors',
    'role_c_solver_config', 'build_role_c_identity', 'verify_role_c_identity',
    'gate_evidence', 'evaluate_identity',
]

ROLE_C_METHOD_CLASS = 'nonlearning_solver'
FROZEN_CONTROL_REL = 'CC_Compare/P1A_CONTROL_FROZEN.json'

# 出处：run_a1_step2_gate.py:51（diag_local_timing.py:23 一致）。冻结配置**未记录**该字段。
CONTROL_ARM_SEEDS_SOURCE = 'run_a1_step2_gate.py:51 ARM_SEEDS (diag_local_timing.py:23 一致)'
CONTROL_ARM_SEEDS = {'uncond_hist': 7001, 'cond_hist': 7002, 'explicit_feat': 7003,
                     'a1_consensus': 7004, 'a1_rollout': 7005, 'soft_knn': 7006}


class IdentityError(ValueError):
    """身份/配置输入不合法。fail-closed。"""


def repo_root():
    """仓库根（本文件在 `<root>/CC_Compare/tools/` 下）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dependency_files(base=None):
    """`provider_dependency_sha256` 的依赖清单（绝对路径），**缺文件即报错**。

    清单来源 = `role_c_harness.ROLE_C_DEPENDENCY_ROLES` 的键（**单一来源**，不另抄一份）。
    """
    root = base or repo_root()
    out = []
    for rel in sorted(hn.ROLE_C_DEPENDENCY_ROLES):
        p = os.path.join(root, rel.replace('/', os.sep))
        if not os.path.isfile(p):
            raise IdentityError(
                '依赖清单里的文件不存在：%s（role=%r）。空/缺项的清单**不是**"已闭合" —— '
                '闭包必须覆盖真实存在的文件，否则 `provider_dependency_sha256` 会给出一个'
                '看起来有效、实际没盖住任何东西的哈希。'
                % (rel, hn.ROLE_C_DEPENDENCY_ROLES[rel]))
        out.append(p)
    if not out:
        raise IdentityError('依赖清单为空 ⇒ 拒绝构造身份（空清单会返回 None 冒充"已闭合"）')
    return tuple(out)


def load_frozen_control(path=None):
    """读冻结对照配置（**只读**；本模块与测试都**不写**它）。"""
    p = path or os.path.join(repo_root(), FROZEN_CONTROL_REL.replace('/', os.sep))
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def frozen_anchors(frozen=None):
    """共享身份锚点（`budget_B` / `cooling_share` / 合同 / 数据 / 数据元 / 种子）。"""
    f = frozen if frozen is not None else load_frozen_control()
    a = f['shared_identity_anchor']
    return {'budget_B': a['budget_B'], 'cooling_share': a['cooling_share'],
            'contract_sha256': a['contract_sha256'], 'data_sha256': a['data_sha256'],
            'data_meta_sha256': a['data_meta_sha256'],
            'main_penalty': a['main_penalty'], 'seeds': a['seeds']}


def shared_config(frozen=None):
    """**共享旋钮**：逐字段取自冻结文件（不重打），Role C 必须与之相同才能配对。

    另附 `booking_horizon`（`must_record_causal_flag`）—— 冻结配置要求 P1-A **显式记录**
    它，而既有 gate 的 `config` 里没有该字段。
    """
    f = frozen if frozen is not None else load_frozen_control()
    cfg = dict(f['frozen_config'])
    cfg['booking_horizon'] = f['must_record_causal_flag']['booking_horizon']
    return cfg


def role_c_solver_config(*, inner_gate=fwd.INNER_GATE_PRIMARY,
                         arm_seed=None, ls_kwargs=None, energy_pricing=None,
                         standby_orders=None, future_policy=None, time_limit=None,
                         K=None, k_neighbors=None, frozen=None):
    """Role C 的 `solver_config_sha256` 输入：共享旋钮 + 机制旋钮 + 冻结规格哈希。

    除机制旋钮外**一律从冻结文件取默认值** ⇒ 想改共享旋钮必须显式传入，而一旦传入
    就会改变 `solver_config_sha256`（进而与对照的身份不再相同）。

    `arm_seed` **优先取冻结文件**（2026-10-14 补录后已记录）；仅当冻结文件缺该字段时
    才回落到 `CONTROL_ARM_SEEDS['cond_hist']`，并把回落这件事写进配置（fail-loud 留痕）。
    """
    f = frozen if frozen is not None else load_frozen_control()
    sh = shared_config(f)
    if inner_gate not in fwd.INNER_GATES:
        raise IdentityError('inner_gate 只能是 %s' % (' / '.join(fwd.INNER_GATES),))
    frozen_seed = sh.get('arm_seed')
    if arm_seed is None:
        if frozen_seed is None:
            raise IdentityError(
                '冻结文件缺 arm_seed 且调用方也未给出 ⇒ 拒绝猜（该值完整决定场景流，'
                '猜错会静默换掉两臂共用的场景集合）。')
        arm_seed = int(frozen_seed)
        seed_src = 'frozen_config'
    else:
        seed_src = 'caller'
    if not isinstance(arm_seed, int) or isinstance(arm_seed, bool):
        raise IdentityError('arm_seed 必须是显式 int（见模块头：它决定场景流）')
    kw = dict(ls_kwargs or {})
    return {
        'method_class': ROLE_C_METHOD_CLASS,
        'shared': {
            **sh,
            'arm_seed': int(arm_seed),
            'K': int(K if K is not None else sh['K']),
            'k_neighbors': int(k_neighbors if k_neighbors is not None
                               else sh['k_neighbors']),
            'time_limit': float(time_limit if time_limit is not None
                                else sh['time_limit']),
            'energy_pricing': energy_pricing or sh['energy_pricing'],
            'standby_orders': (standby_orders if standby_orders is not None
                               else sh['standby_orders']),
            'future_policy': future_policy or sh['future_policy'],
        },
        'role_c_mechanism': {
            'formulation': '(i) myopic_net - lambda_B * e_o > 0 AND certify_ok',
            'lambda_estimator': oc.LAMBDA_ESTIMATOR_SHADOW,
            'myopic_base': oc.MYOPIC_BASE_COMPONENTS,
            'inner_gate': inner_gate,
            'certify_mode': 'final',
            'infeasible_penalty': -1e5,
            'future_order': 'reveal_ascending_then_node',
            'insertion_fail_policy': 'hard_short_circuit_reject (insertion_infeasible)',
        },
        'role_c_scenario': {
            'sampler': 'CondHistoricalSampler',
            'arm_seed': {'value': int(arm_seed),
                         'source': seed_src,
                         'recorded_in_frozen_control': frozen_seed is not None,
                         'added_to_frozen_by': ('user ruling 2026-10-14 '
                                                '(previously missing field)'),
                         'provenance': CONTROL_ARM_SEEDS_SOURCE,
                         'note': '冻结配置原先漏记该字段；值从产出该冻结证据的 runner 读出，'
                                 '已由用户裁定补录'},
            'rng_convention': hn.SAMPLER_RNG_CONVENTION,
        },
        'role_c_route_search': {
            'route': ls.SOURCE_SPEC['route'],
            'lineage': ls.SOURCE_SPEC['lineage'],
            'neighborhood_id': ls.NEIGHBORHOOD_ID,
            'source_spec_sha256': ls.SOURCE_SPEC_SHA256,
            'max_passes': int(kw.get('max_passes', 2)),
            'max_moves': int(kw.get('max_moves', 4000)),
            'rng': None,
        },
        'role_c_forward': {'spec_sha256': fwd.FORWARD_SPEC_SHA256,
                           'rng': None},
        'shared_identity_anchor': frozen_anchors(f),
    }


def build_role_c_identity(*, config, base=None, source_stable=True, frozen=None):
    """构造 Role C 的身份块（`nonlearning_solver`：**不含也不得含** checkpoint）。"""
    a = frozen_anchors(frozen)
    root = base or repo_root()
    ident = ni.build_identity(dependency_files(root), config,
                              contract_sha256=a['contract_sha256'],
                              data_sha256=a['data_sha256'],
                              data_meta_sha256=a['data_meta_sha256'],
                              source_stable=source_stable, base=root,
                              extra={'method_class': ROLE_C_METHOD_CLASS})
    return ident


def verify_role_c_identity(identity, config, base=None):
    """**独立重算**并比对（跑批前自检；真正裁决仍由 `qualification_gate.evaluate`）。

    返回 `(ok, reasons)`。
    """
    root = base or repo_root()
    ident = dict(identity or {})
    if ident.get('checkpoint_sha256'):
        return False, ['nonlearning_solver **不得**声明 checkpoint_sha256（疑似伪造或误标）']
    if ident.get('method_class') != ROLE_C_METHOD_CLASS:
        return False, ['method_class 必须是 %r，得到 %r'
                       % (ROLE_C_METHOD_CLASS, ident.get('method_class'))]
    return ni.verify_closure(ident, dependency_files(root), config, base=root)


def gate_evidence(*, identity, config, per_day, solver_meta, pairing,
                  participation=None, base=None, method_class=ROLE_C_METHOD_CLASS):
    """组装 `qualification_gate.evaluate` 的 `evidence`（`actual_hashes` **从磁盘重算**）。

    `participation` 对 `nonlearning_solver` 是**结构性不适用**（没有模型参与度产物），
    故缺省 `None`；`solver_meta` **必须**提供（求解器体检**不豁免**）。
    """
    root = base or repo_root()
    actual = ni.recompute_actual_hashes(dependency_files(root), config, base=root)
    return {'identity': identity, 'pairing': pairing, 'per_day': per_day,
            'solver_meta': solver_meta, 'participation': participation,
            'method_class': method_class, 'actual_hashes': actual}


def evaluate_identity(*, identity, config, per_day, solver_meta, pairing,
                      participation=None, base=None):
    """T1 端到端：身份块 + 独立重算 + `qualification_gate` 裁决。"""
    ev = gate_evidence(identity=identity, config=config, per_day=per_day,
                       solver_meta=solver_meta, pairing=pairing,
                       participation=participation, base=base)
    return qg.evaluate(ev, method_class=ROLE_C_METHOD_CLASS)
