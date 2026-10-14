#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**未来版本**的正式资格门（只读；绝不改写既有 gate 的裁决字段）。

背景（2026-10-14 独立复核，源码逐字确认）：
现有运行器的正式裁决只有一个条件 ——

    run_a1_external_accept.py:404
        blk[key]['formal_adjudication'] = bool(pairing['identity_verified'])

即「身份合法 ⇒ 实验正式合格」。但协议要求的是合取：

    Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_nocrash ∧ Q_nondegen ∧ Q_method

且 `Q_identity` 本身也不完整：`_run_identity()` 的 15 个字段里**没有 checkpoint 哈希**，
封存清单也不含 provider 的依赖闭包（例：RRNCO 适配器 import 的
`coordinator` / `pickup_certificate` / `preference` / `subproblem` 四个模块都不在清单内）。

本模块的定位：
  * **只用于新批次**；对既有 gate **只读**，不重写任何字段（用户明确要求）；
  * 把上面 7 项写成**可单测的纯函数**，并提供反例测试（`test_qualification_gate.py`）。

设计要点（避免重演缺陷 J 那一类"判据结构性不可能成立"）：
  * 每一项都返回 `ok ∈ {True, False, None}`；`None` = **结构性不适用**并必须附 `reason`；
  * 合取只在**适用**项上取 `False` 否决 —— 规则臂没有学习权重，就不该被 participation 判失败。

用法：
    from qualification_gate import evaluate
    res = evaluate(evidence, required_identity_fields=['checkpoint_sha256'])
    if not res['qualified']:
        print(res['vetoes'], res['criteria'])
"""
from __future__ import annotations

DEFAULT_REQUIRED_IDENTITY = ('checkpoint_sha256', 'provider_dependency_sha256')
RULE_METHODS = {'myopic', 'valuee20', 'valuee30'}


def _crit(ok, reason=''):
    return {'ok': ok, 'reason': reason}


# --------------------------------------------------------------------------- #
# 各项判据（纯函数，输入都是 dict/list，便于单测）
# --------------------------------------------------------------------------- #

def q_identity(identity, pairing, required_fields=DEFAULT_REQUIRED_IDENTITY):
    """身份合格 = 配对身份通过 **且** 必需身份字段齐备。

    ⚠️ 与旧口径的差别：旧 `formal_adjudication` 只看 `pairing.identity_verified`。
    新门额外要求 checkpoint / 依赖闭包哈希**确实存在于** identity 里 —— 否则
    `source_stable=True` 只能证明"源码没变"，**不能**证明"模型权重与依赖没变"。
    """
    if not pairing or pairing.get('identity_verified') is not True:
        return _crit(False, '配对身份未通过：%s' % (pairing or {}).get('problems'))
    missing = [f for f in required_fields if not (identity or {}).get(f)]
    if missing:
        return _crit(False, '身份缺必需字段（源码级身份不能替代权重/依赖身份）: %s'
                     % ', '.join(missing))
    if (identity or {}).get('source_stable') is not True:
        return _crit(False, 'source_stable != True（启动/结束源码哈希不一致）')
    return _crit(True)


def q_days(per_day, expected_days=40):
    ids = [d.get('i') for d in (per_day or []) if d.get('i') is not None]
    if len(ids) != expected_days:
        return _crit(False, '记录数 %d != 期望 %d' % (len(ids), expected_days))
    if len(set(ids)) != expected_days:
        dup = sorted({i for i in ids if ids.count(i) > 1})
        return _crit(False, 'day_id 重复: %s' % dup[:8])
    return _crit(True)


def q_finite(per_day):
    bad = [d.get('i') for d in (per_day or [])
           if not isinstance(d.get('utility'), (int, float))
           or d.get('utility') != d.get('utility')            # NaN
           or d.get('utility') in (float('inf'), float('-inf'))]
    if bad:
        return _crit(False, '效用非有限: %s' % bad[:8])
    if not per_day:
        return _crit(False, '无逐日记录')
    return _crit(True)


def q_hard(per_day):
    bad = [d.get('i') for d in (per_day or []) if d.get('hard_feasible') is not True]
    if bad:
        return _crit(False, '硬约束未全部满足: %s' % bad[:8])
    return _crit(True)


def q_nocrash(per_day):
    bad = [d.get('i') for d in (per_day or [])
           if 'execution_crash' in (d.get('failures') or [])]
    if bad:
        return _crit(False, '执行期异常（按缺失日处理）: %s' % bad[:8])
    return _crit(True)


def q_nondegen(per_day, method_key=None):
    """非"全拒单"退化 —— PyVRP 历史故障的指纹正是它。"""
    served = [d.get('n_served') for d in (per_day or []) if isinstance(d.get('n_served'), int)]
    if not served:
        return _crit(None, '该 runner 不产出 n_served（无法判定退化）')
    if sum(served) == 0:
        return _crit(False, '全拒单退化（sum(n_served)=0）：策略未产出任何服务，'
                            '效用不携带排序信息')
    return _crit(True)


def q_method(solver_meta, participation, method_key=None, is_rule=False):
    """方法健康：真实求解器调用 + 无 fallback + 模型参与证据在位。

    规则臂/经典求解器**结构性**不适用 → 返回 None + reason（不否决）。
    """
    if is_rule:
        return _crit(None, '纯规则臂：无学习权重、无外部求解器可体检（by design）')
    sm = solver_meta or {}
    n_solves, n_fail = sm.get('n_solves'), sm.get('n_fail')
    if not n_solves:
        return _crit(False, '未记录任何求解/推理调用（n_solves=%r）⇒ 无法证明方法真的被执行'
                     % n_solves)
    if n_fail:
        return _crit(False, 'fallback/求解失败 %d 次（n_solves=%d）' % (n_fail, n_solves))
    if participation is not None and participation.get('required'):
        if not participation.get('present'):
            return _crit(False, '学习型方法的参与度证据缺失: %s'
                         % participation.get('path'))
    return _crit(True)


# --------------------------------------------------------------------------- #

def evaluate(evidence, expected_days=40, required_identity_fields=DEFAULT_REQUIRED_IDENTITY):
    """返回 {'qualified': bool, 'criteria': {...}, 'vetoes': [...], 'not_applicable': [...]}。

    `evidence` 键（与真实 gate 对应）：
        identity      ← results[year]['_identity']
        pairing       ← results[year]['_pairing_identity']
        per_day       ← block['per_day']
        solver_meta   ← block['solver_meta']
        participation ← {'required': bool, 'present': bool, 'path': str} 或 None
        is_rule       ← bool
    """
    ev = evidence or {}
    is_rule = bool(ev.get('is_rule'))
    if is_rule:
        # 纯规则臂**没有学习权重**，其 gate 结构性没有 `_pairing_identity`
        # （README §13.3 缺陷 J 已核：产出脚本从不调用 pairing_identity_check）。
        # ⇒ 对它们要求 checkpoint/依赖身份 = 结构性不可能成立 ⇒ 记 None（不适用），
        # **不否决**。它们的可信度由 Q_days/Q_finite/Q_hard/Q_nocrash 覆盖。
        ident_crit = _crit(None, '纯规则臂：无学习权重、无配对身份块（by design，缺陷 J）')
    else:
        ident_crit = q_identity(ev.get('identity'), ev.get('pairing'),
                                required_identity_fields)
    criteria = {
        'Q_identity': ident_crit,
        'Q_days': q_days(ev.get('per_day'), expected_days),
        'Q_finite': q_finite(ev.get('per_day')),
        'Q_hard': q_hard(ev.get('per_day')),
        'Q_nocrash': q_nocrash(ev.get('per_day')),
        'Q_nondegen': q_nondegen(ev.get('per_day'), ev.get('method')),
        'Q_method': q_method(ev.get('solver_meta'), ev.get('participation'),
                             ev.get('method'), is_rule=is_rule),
    }
    vetoes = [k for k, v in criteria.items() if v['ok'] is False]
    na = [k for k, v in criteria.items() if v['ok'] is None]
    return {
        'qualified': not vetoes,
        'criteria': criteria,
        'vetoes': vetoes,
        'not_applicable': na,
        'note': '只在**适用**项上取合取；None = 结构性不适用并附理由（不否决）。'
                '本门只用于新批次，不改写既有 gate 字段。',
    }
