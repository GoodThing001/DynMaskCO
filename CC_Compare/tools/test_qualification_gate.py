#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`qualification_gate` 的反例测试。

2026-10-14 用户核查后在远端源码里指出三个**门自身**的缺口，本文件把它们的反例补齐：
  ① `q_days()` 未要求 day_id 恰为 `0..N-1` ⇒ `1..40` 也通过 → 新增「ID 平移」「ID 越界」反例；
  ② `q_dependency_closure()` 只查字段非空 ⇒ 任意字符串即可通过 → 新增「格式非法」
     与「**哈希格式正确但对应文件被篡改**（独立重算不符）」反例；
  ③ 规则臂被**整体**豁免身份 ⇒ 连源码/合同/数据身份都不再要求 → 新增
     「规则臂 source_stable=False」「规则臂缺合同哈希」反例。
另按用户要求把「合法全拒」与「故障导致全拒」分开（`Q_validity` vs `Q_informativeness`）。
**第二版（2026-10-14 复核后）**：`q_validity` 旧实现用 `not n_solves` 把「字段**缺失**」
当成「求解器异常」—— 纯规则臂结构性 `solver_meta={}`，合法的规则全拒臂因此被误判失败。
现改为**只认正面故障证据**（crash / n_fail>0 / 显式 n_solves==0）；「方法是否真被执行」
归 `Q_method`。反例见 ㉒–㉕ 与 `validity_checks()`。

运行：python CC_Compare/tools/test_qualification_gate.py   （退出码 0 = 全过）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 注：本模块不自绑 sys.stdout（`qualification_gate` 也不绑），避免双重 TextIOWrapper
# 关闭底层 buffer 的问题。
from qualification_gate import evaluate                     # noqa: E402

N = 40
_SHARED = {'contract_sha256': 'c' * 64, 'data_sha256': 'd' * 64,
           'data_meta_sha256': 'e' * 64}
_HASHES = {'checkpoint_sha256': 'a' * 64, 'provider_dependency_sha256': 'b' * 64}


def good_per_day(n=N, served=60):
    return [dict(i=k, utility=100.0 + k, hard_feasible=True, failures=[],
                 n_served=served, n_rejected=140, energy_kwh=700.0,
                 budget_violated=False, timeouts=0, elapsed_s=1.0) for k in range(n)]


def good_evidence(**over):
    ev = {
        'identity': dict(source_stable=True, **_SHARED, **_HASHES),
        'pairing': {'identity_verified': True, 'problems': []},
        'per_day': good_per_day(),
        'solver_meta': {'n_solves': 7761, 'n_fail': 0},
        'participation': {'required': True, 'present': True, 'path': 'participation/x.json'},
        'is_rule': False,
        'actual_hashes': dict(_HASHES),
    }
    ev.update(over)
    return ev


CASES = []


def case(name, expect_qualified, expect_veto=None, **over):
    CASES.append((name, expect_qualified, expect_veto, over))


# ---- 正例 ----
case('① 合格臂（全绿，含独立重算相符）', True)

# ---- 用户四类反例（其中的 checkpoint 缺失见 closure 段）----
case('② 源码匹配但只有 39/40 天', False, 'Q_days', per_day=good_per_day(39))
case('③ 某日 hard_feasible=False', False, 'Q_hard',
     per_day=[dict(d, hard_feasible=False) if d['i'] == 17 else d for d in good_per_day()])
case('④ 求解器全部抛异常（执行期崩溃）', False, 'Q_nocrash',
     per_day=[dict(d, utility=None, hard_feasible=False, failures=['execution_crash'],
                   n_served=None) for d in good_per_day()])

# ---- ① 新增：day_id 集合必须恰为 0..39 ----
case('⑤ day_id 整体平移 1..40（旧实现会误通过）', False, 'Q_days',
     per_day=[dict(d, i=d['i'] + 1) for d in good_per_day()])
case('⑥ day_id 越界（39 缺失、40 出现）', False, 'Q_days',
     per_day=[dict(d, i=(40 if d['i'] == 39 else d['i'])) for d in good_per_day()])
case('⑦ day_id 重复（40 条但只有 39 个唯一）', False, 'Q_days',
     per_day=[dict(d, i=(38 if d['i'] == 39 else d['i'])) for d in good_per_day()])

# ---- 其余劣化 ----
case('⑧ 效用含 NaN', False, 'Q_finite',
     per_day=[dict(d, utility=float('nan')) if d['i'] == 3 else d for d in good_per_day()])
case('⑨ 效用含 +inf', False, 'Q_finite',
     per_day=[dict(d, utility=float('inf')) if d['i'] == 3 else d for d in good_per_day()])
case('⑩ 配对身份未通过', False, 'Q_identity',
     pairing={'identity_verified': False, 'problems': ['baseline_identity_missing']})
case('⑪ source_stable=False（源码中途被改）', False, 'Q_identity',
     identity=dict(source_stable=False, **_SHARED, **_HASHES))
case('⑫ 共享身份缺合同哈希', False, 'Q_identity',
     identity=dict(source_stable=True, data_sha256='d' * 64, data_meta_sha256='e' * 64,
                   **_HASHES))
case('⑬ 求解器有 fallback（n_fail>0）', False, 'Q_method',
     solver_meta={'n_solves': 7761, 'n_fail': 12})
case('⑭ 学习型方法缺参与度证据', False, 'Q_method',
     participation={'required': True, 'present': False, 'path': 'participation/x.json'})
case('⑮ 未记录任何求解调用（无法证明被执行）', False, 'Q_method',
     solver_meta={'n_solves': 0, 'n_fail': 0})
case('⑯ 完全没有逐日记录', False, 'Q_days', per_day=[])

# ---- ⑧ 合法性 vs 信息量：用户要求分开 ----
case('⑰ 合法全拒（健康求解器）⇒ 协议**有效**但**无信息量**', True,
     per_day=[dict(d, n_served=0) for d in good_per_day()])
case('⑱ 全拒 + n_fail>0（故障导致）⇒ 有效性失败', False, 'Q_validity',
     per_day=[dict(d, n_served=0) for d in good_per_day()],
     solver_meta={'n_solves': 7761, 'n_fail': 5})

# ---- ⑨ 结构不适用：不得被否决，但共享身份仍强制 ----
case('⑲ 纯规则臂（含共享身份）⇒ 权重身份不适用，仍合格', True,
     is_rule=True, pairing={}, solver_meta={},
     participation={'required': False, 'present': False},
     identity=dict(source_stable=True, **_SHARED))
case('⑳ 纯规则臂 source_stable=False ⇒ **必须被拒**', False, 'Q_identity',
     is_rule=True, pairing={}, solver_meta={},
     participation={'required': False, 'present': False},
     identity=dict(source_stable=False, **_SHARED))
case('㉑ 纯规则臂缺合同哈希 ⇒ **必须被拒**', False, 'Q_identity',
     is_rule=True, pairing={}, solver_meta={},
     participation={'required': False, 'present': False},
     identity=dict(source_stable=True, data_sha256='d' * 64, data_meta_sha256='e' * 64))

# ---- ⑩ 第二版修正：**缺失** `n_solves` 不是故障证据 ----
# 用户 2026-10-14 复核指出的**确定性逻辑缺陷**：旧 `q_validity` 用 `not n_solves` 把
# 「字段缺失」当成「求解器异常」。纯规则臂结构性没有求解器（`solver_meta={}`），
# 于是一个**完全合法**的规则全拒臂会被误判为有效性失败。以下是那个**精确反例**。
case('㉒ 纯规则臂 + 合法全拒 + solver_meta={} ⇒ 有效但无信息量', True,
     is_rule=True, pairing={}, solver_meta={},
     participation={'required': False, 'present': False},
     identity=dict(source_stable=True, **_SHARED),
     per_day=[dict(d, n_served=0) for d in good_per_day()])
# 反面：**非**规则臂全拒且完全没有求解器元数据 ⇒ 不构成有效性失败（缺失≠故障），
# 但 `Q_method` 会否决「无法证明方法被执行」⇒ 整体仍不合格。**没有留下漏洞。**
case('㉓ 非规则臂 + 全拒 + solver_meta={} ⇒ 由 Q_method 否决（不由 Q_validity）', False,
     'Q_method', solver_meta={},
     per_day=[dict(d, n_served=0) for d in good_per_day()])
# 正面故障证据仍然必须否决：
case('㉔ 全拒 + 显式 n_solves=0 ⇒ 有效性失败', False, 'Q_validity',
     solver_meta={'n_solves': 0, 'n_fail': 0},
     per_day=[dict(d, n_served=0) for d in good_per_day()])
case('㉕ 全拒 + execution_crash ⇒ 有效性失败', False, 'Q_validity',
     per_day=[dict(d, n_served=0, failures=['execution_crash']) for d in good_per_day()])


def validity_checks():
    """把「全拒单」的三种成因逐一对齐（正面证据 / 结构性缺失 / 合法策略）。"""
    out = []
    allrej = [dict(d, n_served=0) for d in good_per_day()]

    # ① 用户指出的精确反例：规则臂全拒 + solver_meta={} ⇒ Q_validity 不得否决
    r = evaluate(good_evidence(is_rule=True, pairing={}, solver_meta={},
                               participation={'required': False, 'present': False},
                               identity=dict(source_stable=True, **_SHARED),
                               per_day=allrej))
    out.append(('规则臂全拒 + solver_meta={} ⇒ Q_validity=%s（不得判失败）、'
                'qualified=True、informative=False'
                % r['criteria']['Q_validity']['ok'],
                r['criteria']['Q_validity']['ok'] is True
                and r['qualified'] is True and r['informative'] is False
                and 'Q_validity' not in r['vetoes']))

    # ② 规则臂全拒 + **真的**崩溃 ⇒ 仍须否决（修正没有放宽故障判定）
    r = evaluate(good_evidence(is_rule=True, pairing={}, solver_meta={},
                               participation={'required': False, 'present': False},
                               identity=dict(source_stable=True, **_SHARED),
                               per_day=[dict(d, n_served=0, failures=['execution_crash'])
                                        for d in good_per_day()]))
    out.append(('规则臂全拒 + 崩溃 ⇒ 仍判有效性失败（修正未放宽故障）',
                r['criteria']['Q_validity']['ok'] is False
                and 'Q_validity' in r['vetoes']))

    # ③ 非规则臂全拒 + solver_meta={} ⇒ 缺失不是故障证据，但 Q_method 兜住
    r = evaluate(good_evidence(solver_meta={}, per_day=allrej))
    out.append(('非规则臂全拒 + solver_meta={} ⇒ Q_validity=True 但 Q_method 否决'
                '（缺失≠故障，且不留漏洞）',
                r['criteria']['Q_validity']['ok'] is True
                and 'Q_method' in r['vetoes'] and r['qualified'] is False))
    return out


def closure_checks():
    """`Q_dependency_closure`：fail-closed 三重校验（字段 / 格式 / **独立重算相符**）。"""
    out = []
    base = dict(source_stable=True, **_SHARED)

    r0 = evaluate(good_evidence(identity=dict(base, provider_dependency_sha256='b' * 64),
                               actual_hashes={'provider_dependency_sha256': 'b' * 64}))
    out.append(('缺 checkpoint ⇒ core 仍 True 但 qualified_full=False（新批次拒绝）',
                r0['qualified'] is True and r0['qualified_full'] is False))

    r1 = evaluate(good_evidence(identity=dict(base, checkpoint_sha256='a' * 64,
                                             provider_dependency_sha256='b' * 64),
                               actual_hashes=None))
    out.append(('② 无独立重算结果 ⇒ fail-closed 判不合格',
                r1['Q_dependency_closure']['ok'] is False
                and '独立重算' in r1['Q_dependency_closure']['reason']))

    r2 = evaluate(good_evidence(identity=dict(base, checkpoint_sha256='ZZ' * 32,
                                             provider_dependency_sha256='b' * 64),
                               actual_hashes=_HASHES))
    out.append(('② 哈希格式非法（非小写十六进制）⇒ 不合格',
                r2['Q_dependency_closure']['ok'] is False))

    r3 = evaluate(good_evidence(identity=dict(base, **_HASHES),
                               actual_hashes={'checkpoint_sha256': 'f' * 64,
                                              'provider_dependency_sha256': 'b' * 64}))
    out.append(('② **哈希格式正确但文件被篡改**（独立重算不符）⇒ 不合格',
                r3['Q_dependency_closure']['ok'] is False
                and r3['qualified_full'] is False))

    r4 = evaluate(good_evidence())
    out.append(('字段齐备 + 独立重算相符 ⇒ closure=True 且 qualified_full=True',
                r4['Q_dependency_closure']['ok'] is True and r4['qualified_full'] is True))

    r5 = evaluate(good_evidence(is_rule=True, pairing={}, solver_meta={},
                                identity=dict(source_stable=True, **_SHARED),
                                participation={'required': False, 'present': False}))
    out.append(('纯规则臂 ⇒ closure=None（不适用），qualified_full 仍 True',
                r5['Q_dependency_closure']['ok'] is None and r5['qualified_full'] is True))
    return out


def informativeness_checks():
    out = []
    r = evaluate(good_evidence(per_day=[dict(d, n_served=0) for d in good_per_day()]))
    out.append(('合法全拒：qualified=True 且 informative=False（历史 λ=3 判定不改写）',
                r['qualified'] is True and r['informative'] is False))
    r2 = evaluate(good_evidence())
    out.append(('有服务：informative=True', r2['informative'] is True))
    return out


def method_class_checks():
    """**方法身份类型**（2026-10-14 用户裁决后新增）。

    用户要求：「**不应**为它制造一个虚假的 checkpoint，也**不应**简单标记 `is_rule=True`
    来绕过求解器健康检查。」⇒ `nonlearning_solver` 必须同时满足：
      · 不要求（也不得声明）`checkpoint_sha256`；
      · 必须闭合 `provider_dependency_sha256` + `solver_config_sha256`（且需独立重算相符）；
      · `Q_method` **照查**（真实调用 + 无 fallback）—— 非学习 ≠ 免检；
      · 共享身份与配对身份仍然强制。
    """
    out = []
    solver_ident = dict(source_stable=True, **_SHARED,
                        provider_dependency_sha256='b' * 64,
                        solver_config_sha256='c' * 64)
    actual_solver = {'provider_dependency_sha256': 'b' * 64,
                     'solver_config_sha256': 'c' * 64}
    part_na = {'required': False, 'present': False}

    # ① 正例：无 checkpoint 的非学习求解器 ⇒ 合格（且不因此被要求伪造 checkpoint）
    r = evaluate(good_evidence(method_class='nonlearning_solver', identity=solver_ident,
                               actual_hashes=actual_solver, participation=part_na))
    out.append(('非学习求解器（无 checkpoint）⇒ qualified=True 且 qualified_full=True',
                r['qualified'] is True and r['qualified_full'] is True
                and r['method_class'] == 'nonlearning_solver'
                and 'checkpoint_sha256' not in r['required_identity_fields']))

    # ② 求解器健康**不豁免**：n_solves 缺失 / 有 fallback ⇒ 否决
    r = evaluate(good_evidence(method_class='nonlearning_solver', identity=solver_ident,
                               actual_hashes=actual_solver, participation=part_na,
                               solver_meta={}))
    out.append(('非学习求解器缺 n_solves ⇒ Q_method 否决（非学习≠免检）',
                r['criteria']['Q_method']['ok'] is False and 'Q_method' in r['vetoes']))
    r = evaluate(good_evidence(method_class='nonlearning_solver', identity=solver_ident,
                               actual_hashes=actual_solver, participation=part_na,
                               solver_meta={'n_solves': 900, 'n_fail': 7}))
    out.append(('非学习求解器有 fallback ⇒ Q_method 否决',
                r['criteria']['Q_method']['ok'] is False and 'Q_method' in r['vetoes']))

    # ③ 闭包字段：缺 solver_config_sha256 / 未独立重算 ⇒ qualified_full=False
    r = evaluate(good_evidence(method_class='nonlearning_solver',
                               identity=dict(source_stable=True, **_SHARED,
                                             provider_dependency_sha256='b' * 64),
                               actual_hashes={'provider_dependency_sha256': 'b' * 64},
                               participation=part_na))
    out.append(('非学习求解器缺 solver_config_sha256 ⇒ closure=False（core 仍 True）',
                r['qualified'] is True and r['qualified_full'] is False
                and r['Q_dependency_closure']['ok'] is False))
    r = evaluate(good_evidence(method_class='nonlearning_solver', identity=solver_ident,
                               actual_hashes=None, participation=part_na))
    out.append(('非学习求解器无独立重算 ⇒ fail-closed 判不合格',
                r['Q_dependency_closure']['ok'] is False))

    # ④ 防绕过：rule 却带求解器调用记录 ⇒ 否决
    r = evaluate(good_evidence(method_class='rule', pairing={}, participation=part_na,
                               identity=dict(source_stable=True, **_SHARED),
                               solver_meta={'n_solves': 500, 'n_fail': 0}))
    out.append(('标 `rule` 却带 n_solves ⇒ Q_class_consistency 否决（防绕过体检）',
                r['criteria']['Q_class_consistency']['ok'] is False
                and 'Q_class_consistency' in r['vetoes']))

    # ⑤ 防伪造：nonlearning_solver 却声明 checkpoint_sha256 ⇒ 否决
    r = evaluate(good_evidence(method_class='nonlearning_solver',
                               identity=dict(solver_ident, checkpoint_sha256='a' * 64),
                               actual_hashes=dict(actual_solver, checkpoint_sha256='a' * 64),
                               participation=part_na))
    out.append(('标 `nonlearning_solver` 却声明 checkpoint ⇒ Q_class_consistency 否决',
                r['criteria']['Q_class_consistency']['ok'] is False))

    # ⑥ 向后兼容：不给 method_class 时按 is_rule 推断，学习型仍强制 checkpoint
    r = evaluate(good_evidence())
    out.append(('未给 method_class ⇒ 推断为 learned，仍要求 checkpoint',
                r['method_class'] == 'learned'
                and 'checkpoint_sha256' in r['required_identity_fields']))
    r = evaluate(good_evidence(is_rule=True, pairing={}, solver_meta={},
                               participation=part_na,
                               identity=dict(source_stable=True, **_SHARED)))
    out.append(('未给 method_class 且 is_rule=True ⇒ 推断为 rule（行为不变）',
                r['method_class'] == 'rule' and r['qualified'] is True))

    # ⑦ 非学习求解器的共享/配对身份**不豁免**
    r = evaluate(good_evidence(method_class='nonlearning_solver',
                               identity=dict(solver_ident, contract_sha256=None),
                               actual_hashes=actual_solver, participation=part_na))
    out.append(('非学习求解器缺共享身份（合同哈希）⇒ 仍被拒',
                r['criteria']['Q_identity']['ok'] is False and 'Q_identity' in r['vetoes']))
    r = evaluate(good_evidence(method_class='nonlearning_solver', identity=solver_ident,
                               actual_hashes=actual_solver, participation=part_na,
                               pairing={'identity_verified': False, 'problems': ['x']}))
    out.append(('非学习求解器配对身份未通过 ⇒ 仍被拒（对照不是豁免对象）',
                r['criteria']['Q_identity']['ok'] is False))
    return out


def q_days_strict_checks():
    """`q_days` 的**非整数** day_id 反例（与矩阵同一缺陷，第三版修正）。"""
    out = []
    ok, why = evaluate(good_evidence(per_day=[dict(d, i=d['i'] + 0.5)
                                              for d in good_per_day()]))['criteria']['Q_days']['ok'], ''
    out.append(('gate·q_days：0.5..39.5 非整数日号 ⇒ 拒绝（int() 截断会误通过）', ok is False))
    r = evaluate(good_evidence(per_day=[dict(d, i=str(d['i'])) for d in good_per_day()]))
    out.append(('gate·q_days：数字字符串日号 ⇒ 拒绝', r['criteria']['Q_days']['ok'] is False))
    r = evaluate(good_evidence(per_day=[dict(d, i=bool(d['i'] % 2)) for d in good_per_day()]))
    out.append(('gate·q_days：bool 日号 ⇒ 拒绝', r['criteria']['Q_days']['ok'] is False))
    r = evaluate(good_evidence(per_day=[dict(d, i=(38 if d['i'] == 39 else d['i']))
                                        for d in good_per_day()]))
    out.append(('gate·q_days：重复日号 ⇒ 拒绝', r['criteria']['Q_days']['ok'] is False))
    return out


def run():
    passed = failed = 0
    for name, expect_q, expect_veto, over in CASES:
        res = evaluate(good_evidence(**over))
        ok = (res['qualified'] == expect_q) and (
            (expect_veto is None) or (expect_veto in res['vetoes']))
        if ok:
            passed += 1
            print('  PASS %-52s qualified=%-5s vetoes=%s'
                  % (name, res['qualified'], res['vetoes'] or '—'))
        else:
            failed += 1
            print('  FAIL %-52s 期望 qualified=%s veto=%s，实得 qualified=%s vetoes=%s'
                  % (name, expect_q, expect_veto, res['qualified'], res['vetoes']))
    for title, checks in (('\nQ_dependency_closure 三重校验：', closure_checks()),
                          ('\n全拒单成因对齐（Q_validity 第二版）：', validity_checks()),
                          ('\n方法身份类型（nonlearning_solver，2026-10-14 用户裁决）：',
                           method_class_checks()),
                          ('\ngate·q_days 非整数日号（第三版）：', q_days_strict_checks()),
                          ('\n有效性 / 信息量分离：', informativeness_checks())):
        print(title)
        for name, ok in checks:
            if ok:
                passed += 1
                print('  PASS %s' % name)
            else:
                failed += 1
                print('  FAIL %s' % name)
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run())
