#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**未来版本**的正式资格门（只读；绝不改写既有 gate 的裁决字段）。

背景（2026-10-14 独立复核，源码逐字确认）：
现有运行器的正式裁决只有一个条件 ——

    run_a1_external_accept.py:404
        blk[key]['formal_adjudication'] = bool(pairing['identity_verified'])

即「身份合法 ⇒ 实验正式合格」。但协议要求的是合取：

    Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_nocrash ∧ Q_nondegen ∧ Q_method

    Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_nocrash ∧ Q_nondegen ∧ Q_method

且 `Q_identity` 本身也不完整：`_run_identity()` 的 15 个字段里**没有 checkpoint 哈希**，
封存清单也不含 provider 的依赖闭包（例：RRNCO 适配器 import 的
`coordinator` / `pickup_certificate` / `preference` / `subproblem` 四个模块都不在清单内）。

⇒ 因此**另设独立字段** `Q_dependency_closure`（2026-10-14 用户要求）：
`Q_core=True` **只能**读作「P0 定义的核心检查通过」，**不能**读成「整个方法复现身份已闭合」。
该字段**刻意不计入** `qualified` —— 对历史臂事后加入新否决条件等于事后改写资格；
正确做法是不追认、不删除、只如实标注边界。

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

import hashlib
import re

DEFAULT_REQUIRED_IDENTITY = ('checkpoint_sha256', 'provider_dependency_sha256')
# 共享身份字段（源码之外的合同/数据）——**所有**臂（含纯规则臂）都必须具备
SHARED_IDENTITY_FIELDS = ('contract_sha256', 'data_sha256', 'data_meta_sha256')
RULE_METHODS = {'myopic', 'valuee20', 'valuee30'}
_SHA_RE = re.compile(r'^[0-9a-f]{64}$')


def sha256_file(path):
    """从磁盘实际重算 SHA-256（供调用方做**独立**身份核对，而不是相信身份块的自述）。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _crit(ok, reason=''):
    return {'ok': ok, 'reason': reason}


# --------------------------------------------------------------------------- #
# 各项判据（纯函数，输入都是 dict/list，便于单测）
# --------------------------------------------------------------------------- #

def q_identity(identity, pairing, is_rule=False):
    """身份合格，**分两层**：

    ① **共享身份**（源码 `source_stable` + 合同 + 数据 + 数据元）——
       对**所有**臂（**含纯规则臂**）强制；
    ② **学习权重配对身份**（`pairing.identity_verified`）—— 仅非规则臂强制。

    2026-10-14 用户在远端源码核实：旧实现把 `is_rule=True` 的 `Q_identity` **整体**设为
    "不适用"，于是规则臂连源码/合同/数据身份都不再被要求 —— 这是把"**权重**不适用"
    错误放大成"**身份**不适用"。现在只豁免权重那一层，共享身份仍然强制。
    """
    if (identity or {}).get('source_stable') is not True:
        return _crit(False, 'source_stable != True（启动/结束源码哈希不一致）')
    missing = [f for f in SHARED_IDENTITY_FIELDS if not (identity or {}).get(f)]
    if missing:
        return _crit(False, '共享身份（源码/合同/数据）缺字段: %s' % ', '.join(missing))
    if is_rule:
        return _crit(None, '纯规则臂：无学习权重 ⇒ 仅"权重配对身份"不适用；'
                           '共享身份（源码/合同/数据/数据元）**已要求并通过**')
    if not pairing or pairing.get('identity_verified') is not True:
        return _crit(False, '配对身份未通过：%s' % (pairing or {}).get('problems'))
    return _crit(True)


def q_days(per_day, expected_days=40):
    """日完整性：**必须恰好是 `0..expected_days-1`**。

    2026-10-14 用户在远端源码核实：旧实现只查「40 条 + 40 个不同 ID」，
    因此 `1..40` 这种**整体平移**的集合也会通过 —— 那不是 40 天开发集。
    现在要求集合与 `range(expected_days)` **逐元素相等**；越界（如 40）同样被拒。
    """
    ids = [d.get('i') for d in (per_day or []) if d.get('i') is not None]
    if len(ids) != expected_days:
        return _crit(False, '记录数 %d != 期望 %d' % (len(ids), expected_days))
    try:
        got = sorted(int(i) for i in ids)
    except (TypeError, ValueError):
        return _crit(False, 'day_id 非整数: %s' % ids[:8])
    want = list(range(expected_days))
    if got != want:
        extra = sorted(set(got) - set(want))
        miss = sorted(set(want) - set(got))
        return _crit(False, 'day_id 集合 != 0..%d（越界 %s；缺失 %s）'
                     % (expected_days - 1, extra[:8], miss[:8]))
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


def q_validity(per_day, solver_meta=None, is_rule=False):
    """**协议有效性**：全拒单是否由**故障**造成。

    2026-10-14 用户要求把两件事分开：
      * **合法**的全拒策略（如 λ=3 规则：选择不服务任何订单）⇒ **不是**有效性失败，
        只是**无信息量**；
      * **求解器异常**导致的全拒（如 PyVRP 的 API 不兼容）⇒ **有效性失败**。

    **判据（2026-10-14 第二版修正）**：只在存在**正面故障证据**时否决 ——
    `execution_crash`、`n_fail > 0`、或 `n_solves` **显式为 0**。

    ⚠️ 旧实现把「`n_solves` **缺失**」也当作故障证据（`not n_solves`），这是一个**确定性
    逻辑缺陷**：纯规则臂 `is_rule=True` **结构性**没有求解器，`solver_meta` 天然为 `{}`
    ⇒ 一个**完全合法**的规则全拒臂会被误判为有效性失败。缺失字段不是失败证据。
    「方法到底有没有被执行」由 `q_method` 负责（对**非规则**臂，`n_solves` 缺失依然否决），
    两条判据职责分开后，这里不再承担"证明方法被调用"的职责，因此不会留下漏洞。
    """
    serves = [d.get('n_served') for d in (per_day or []) if isinstance(d.get('n_served'), int)]
    if not serves or sum(serves) > 0:
        return _crit(True)
    # —— 以下为「全拒单」分支：只认正面故障证据 ——
    sm = solver_meta or {}
    n_solves, n_fail = sm.get('n_solves'), sm.get('n_fail')
    crash = any('execution_crash' in (d.get('failures') or []) for d in (per_day or []))
    ev = []
    if crash:
        ev.append('crash=True')
    if (n_fail or 0) > 0:
        ev.append('n_fail=%s' % n_fail)
    if n_solves == 0:                       # 显式 0 才是异常；**缺失**不是
        ev.append('n_solves=0')
    if ev:
        return _crit(False, '全拒单**且**有正面故障证据（%s）⇒ 判为求解器/执行失败，'
                            '而非合法全拒策略' % ', '.join(ev))
    if is_rule:
        return _crit(True, '纯规则臂**结构性**无求解器（solver_meta=%r）；全拒单是合法策略选择，'
                           '不由 `Q_validity` 否决（信息量问题归 `Q_informativeness`）'
                     % (solver_meta,))
    if not sm or n_solves is None:
        return _crit(True, '无求解器健康元数据（solver_meta=%r）⇒ 缺失**不是**故障证据；'
                           '方法是否真被执行由 `Q_method` 判定' % (solver_meta,))
    return _crit(True, '全拒单但无故障证据（n_solves=%s, n_fail=%s）⇒ 视为合法全拒策略'
                 % (n_solves, n_fail))


def q_informativeness(per_day, method_key=None):
    """**信息量**（非否决项）：全拒单 ⇒ 效用不携带排序信息。

    与 `q_validity` 分开报告：合法全拒**不**使实验协议失效，但该臂**不可用于排名**。
    历史 λ=3 的判定（在 P0 矩阵里由 Q_nondegen 否决）**不改写** —— 这里只是新增一个
    语义更准的字段。
    """
    served = [d.get('n_served') for d in (per_day or []) if isinstance(d.get('n_served'), int)]
    if not served:
        return _crit(None, '该 runner 不产出 n_served（无法判定信息量）')
    if sum(served) == 0:
        return _crit(False, '全拒单 ⇒ 无信息量（不可用于排名；但若为合法策略则协议仍有效）')
    return _crit(True)


def q_nondegen(per_day, method_key=None):
    """【保留】零服务数识别（= 信息量）。仅为向后兼容保留；新代码用 `q_informativeness`。"""
    return q_informativeness(per_day, method_key)


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


def q_dependency_closure(identity, required_fields=DEFAULT_REQUIRED_IDENTITY, is_rule=False,
                         actual_hashes=None):
    """**独立**字段：完整复现身份是否闭合（checkpoint SHA + provider 依赖闭包）。

    2026-10-14 用户在远端源码核实：旧实现**只检查字段非空** —— 填入任意字符串
    （例如 `'a'*64`）也能通过 `qualified_full`。那是"门自己通过测试却仍放行应拒记录"。
    现在改为 **fail-closed 三重校验**：

      ① **字段齐备**：`required_fields` 一个不能少；
      ② **格式合法**：每个字段必须是 **64 位小写十六进制**（`^[0-9a-f]{64}$`）；
      ③ **独立重算相符**：调用方必须提供 `actual_hashes`（**由调用方从磁盘实际重算**
         checkpoint 与依赖源码清单得到），且与身份块**逐字段相等**。
         未提供 `actual_hashes` ⇒ **判不合格**（fail-closed），理由写明
         "未提供独立重算结果" —— 身份块的自述**不能**自证。

    ⇒ `Q_core=True` 只能读作「P0 定义的核心检查通过」，**不能**读成「复现身份已闭合」。
    """
    if is_rule:
        return _crit(None, '纯规则臂：无学习权重 ⇒ 无 checkpoint/依赖身份可闭合')
    ident = identity or {}
    missing = [f for f in required_fields if not ident.get(f)]
    if missing:
        return _crit(False, '缺 %s ⇒ 权重与依赖身份未闭合（source_stable 只覆盖源码）'
                     % ', '.join(missing))
    bad_fmt = [f for f in required_fields if not _SHA_RE.match(str(ident.get(f)))]
    if bad_fmt:
        return _crit(False, '字段格式非法（须为 64 位小写十六进制 SHA-256）: %s'
                     % ', '.join(bad_fmt))
    if not actual_hashes:
        return _crit(False, '未提供**独立重算**结果（actual_hashes）⇒ 无法证明身份与磁盘实际'
                            '文件一致；身份块自述不能自证（fail-closed）')
    mismatch = [f for f in required_fields
                if f in actual_hashes and str(actual_hashes[f]).lower() != str(ident.get(f)).lower()]
    absent = [f for f in required_fields if f not in actual_hashes]
    if absent:
        return _crit(False, '独立重算结果缺字段: %s' % ', '.join(absent))
    if mismatch:
        return _crit(False, '身份块与**磁盘实际重算**不符（文件被改动？）: %s' % ', '.join(mismatch))
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
    criteria = {
        # ⚠️ 规则臂**不再整体豁免身份**：`q_identity` 内部对共享身份（源码/合同/数据）
        # 仍然强制，只把"权重配对身份"那一层记为不适用。
        'Q_identity': q_identity(ev.get('identity'), ev.get('pairing'), is_rule=is_rule),
        'Q_days': q_days(ev.get('per_day'), expected_days),
        'Q_finite': q_finite(ev.get('per_day')),
        'Q_hard': q_hard(ev.get('per_day')),
        'Q_nocrash': q_nocrash(ev.get('per_day')),
        # **有效性**（否决项）：全拒单若伴随**正面故障证据** ⇒ 无效。
        # ⚠️ 规则臂的 `solver_meta={}` 是结构性事实，**不是**故障证据（第二版修正）。
        'Q_validity': q_validity(ev.get('per_day'), ev.get('solver_meta'), is_rule=is_rule),
        # **信息量**（仅报告，不否决）：合法全拒不使协议失效，但不可用于排名
        'Q_informativeness': q_informativeness(ev.get('per_day'), ev.get('method')),
        'Q_method': q_method(ev.get('solver_meta'), ev.get('participation'),
                             ev.get('method'), is_rule=is_rule),
    }
    # ⚠️ `Q_informativeness` **不在否决列表内**（2026-10-14 用户要求）：
    # 「合法全拒策略」协议仍然有效，只是该臂不可用于排名；把它当否决项等于又犯了
    # 「无信息量 = 实验无效」的混淆。第一版实现漏了这一步，被自己的反例测试抓出。
    NON_VETO = ('Q_informativeness',)
    vetoes = [k for k, v in criteria.items() if v['ok'] is False and k not in NON_VETO]
    na = [k for k, v in criteria.items() if v['ok'] is None]
    closure = q_dependency_closure(ev.get('identity'), required_identity_fields,
                                   is_rule=is_rule,
                                   actual_hashes=ev.get('actual_hashes'))
    return {
        # `qualified` = **P0 定义**的核心合格（与历史 38 个合格臂同口径，可跨批比较）
        'qualified': not vetoes,
        # `qualified_full` = 新版本的**完整**合格：再加权重/依赖身份闭包。
        # ⚠️ **新批次一律用 `qualified_full`**；历史批次用 `qualified` + 如实上报 closure。
        'qualified_full': (not vetoes) and (closure['ok'] is not False),
        'criteria': criteria,
        'vetoes': vetoes,
        'not_applicable': na,
        'Q_dependency_closure': closure,
        'informative': criteria['Q_informativeness']['ok'] is not False,
        'note': '只在**适用**项上取合取；None = 结构性不适用并附理由（不否决）。'
                '`qualified` = 身份（含规则臂的共享身份）∧ 完整 0..N-1 日 ∧ 有限 ∧ 硬可行 '
                '∧ 无崩溃 ∧ **Q_validity** ∧ 方法健康；**Q_informativeness 只报告不否决**'
                '（合法全拒策略协议仍有效，但不可用于排名）。'
                '`qualified_full` 额外要求 `Q_dependency_closure`（需独立重算，fail-closed）。',
    }
