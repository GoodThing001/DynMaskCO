#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`qualification_gate` 的反例测试（用户 P0 明确要求四类反例必须全部被拒绝）。

这是**纯本地、不依赖任何 gate 文件**的合成数据测试 —— 因为我们要验证的是"判据本身
能不能挡住坏数据"，而不是"当前批次好不好"。用真实 gate 测不出前者。

运行：python CC_Compare/tools/test_qualification_gate.py   （退出码 0 = 全过）
"""
import os
import sys
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualification_gate import evaluate                     # noqa: E402

N = 40


def good_per_day(n=N, served=60):
    return [dict(i=k, utility=100.0 + k, hard_feasible=True, failures=[],
                 n_served=served, n_rejected=140, energy_kwh=700.0,
                 budget_violated=False, timeouts=0, elapsed_s=1.0) for k in range(n)]


def good_evidence(**over):
    ev = {
        'identity': {'source_stable': True,
                     'checkpoint_sha256': 'a' * 64,
                     'provider_dependency_sha256': 'b' * 64},
        'pairing': {'identity_verified': True, 'problems': []},
        'per_day': good_per_day(),
        'solver_meta': {'n_solves': 7761, 'n_fail': 0},
        'participation': {'required': True, 'present': True, 'path': 'participation/x.json'},
        'is_rule': False,
    }
    ev.update(over)
    return ev


CASES = []


def case(name, expect_qualified, expect_veto=None, **over):
    CASES.append((name, expect_qualified, expect_veto, over))


# ---- 正例 ----
case('① 合格臂（全绿）', True)

# ---- 用户要求的四类反例 ----
case('② 源码匹配但只有 39/40 天', False, 'Q_days', per_day=good_per_day(39))
case('③ 某日 hard_feasible=False', False, 'Q_hard',
     per_day=[dict(d, hard_feasible=False) if d['i'] == 17 else d
              for d in good_per_day()])
case('④ 求解器全部抛异常（执行期崩溃）', False, 'Q_nocrash',
     per_day=[dict(d, utility=None, hard_feasible=False, failures=['execution_crash'],
                   n_served=None) for d in good_per_day()])
# 注：用户四类反例里的「checkpoint 身份缺失」**不在**此处 —— 它由 `qualified_full`
# 拒绝（见 closure_checks 的第一条），而不是由 P0 口径的核心门拒绝。
# 这样历史 38 个合格臂的 `qualified=True` 不被事后改写，同时新批次仍会拒绝它。
# ---- 其余必须被挡住的劣化 ----
case('⑥ 效用含 NaN', False, 'Q_finite',
     per_day=[dict(d, utility=float('nan')) if d['i'] == 3 else d for d in good_per_day()])
case('⑦ 效用含 +inf', False, 'Q_finite',
     per_day=[dict(d, utility=float('inf')) if d['i'] == 3 else d for d in good_per_day()])
case('⑧ 全拒单退化（PyVRP 故障指纹）', False, 'Q_nondegen',
     per_day=[dict(d, n_served=0) for d in good_per_day()])
case('⑨ day_id 重复（40 条但只有 39 个唯一）', False, 'Q_days',
     per_day=[dict(d, i=(39 if d['i'] == 38 else d['i'])) for d in good_per_day()])
case('⑩ 配对身份未通过', False, 'Q_identity',
     pairing={'identity_verified': False, 'problems': ['baseline_identity_missing']})
case('⑪ source_stable=False（源码中途被改）', False, 'Q_identity',
     identity={'source_stable': False, 'checkpoint_sha256': 'a' * 64,
               'provider_dependency_sha256': 'b' * 64})
case('⑫ 求解器有 fallback（n_fail>0）', False, 'Q_method',
     solver_meta={'n_solves': 7761, 'n_fail': 12})
case('⑬ 学习型方法缺参与度证据', False, 'Q_method',
     participation={'required': True, 'present': False, 'path': 'participation/x.json'})
case('⑭ 未记录任何求解调用（无法证明被执行）', False, 'Q_method',
     solver_meta={'n_solves': 0, 'n_fail': 0})
case('⑮ 完全没有逐日记录', False, 'Q_days', per_day=[])

# ---- 结构性不适用：必须**不**被否决 ----
case('⑯ 纯规则臂（无权重、无 solver_meta、无配对身份）', True,
     is_rule=True, solver_meta={},
     participation={'required': False, 'present': False},
     identity={'source_stable': True}, pairing={})
# 注：规则臂的 Q_identity / Q_method 应为 **None（不适用）** 而非 False ——
# 否则就是缺陷 J 那一类「判据结构性不可能成立 ⇒ 报告里永远有假失败」。


def closure_checks():
    """`Q_dependency_closure` 是**独立**字段：缺权重/依赖身份要如实标 False，
    但**不得**因此否决 `Q_formal_core`（否则就是用今天的新判据事后推翻历史合格臂）。"""
    out = []
    # 用户四类反例之一：checkpoint 身份缺失 ⇒ **新批次必须拒绝**（由 qualified_full）
    r0 = evaluate(good_evidence(identity={'source_stable': True,
                                         'provider_dependency_sha256': 'b' * 64}))
    out.append(('缺 checkpoint ⇒ core 保持 True 但**新批次门 qualified_full=False**（拒绝）',
                r0['qualified'] is True and r0['qualified_full'] is False
                and r0['Q_dependency_closure']['ok'] is False))
    r = evaluate(good_evidence(identity={'source_stable': True}))
    out.append(('缺 checkpoint/依赖哈希 ⇒ closure=False 且**不**否决 P0 core',
                r['Q_dependency_closure']['ok'] is False and r['qualified'] is True))
    r2 = evaluate(good_evidence())
    out.append(('字段齐备 ⇒ closure=True 且 qualified_full=True',
                r2['Q_dependency_closure']['ok'] is True and r2['qualified_full'] is True))
    r3 = evaluate(good_evidence(is_rule=True, pairing={}, solver_meta={},
                                identity={'source_stable': True},
                                participation={'required': False, 'present': False}))
    out.append(('纯规则臂 ⇒ closure=None（不适用），不影响 qualified_full',
                r3['Q_dependency_closure']['ok'] is None and r3['qualified_full'] is True))
    return out


def run():
    passed = failed = 0
    for name, expect_q, expect_veto, over in CASES:
        ev = good_evidence(**over)
        res = evaluate(ev)
        ok = (res['qualified'] == expect_q)
        veto_ok = (expect_veto is None) or (expect_veto in res['vetoes'])
        if ok and veto_ok:
            passed += 1
            print('  PASS %-44s qualified=%-5s vetoes=%s'
                  % (name, res['qualified'], res['vetoes'] or '—'))
        else:
            failed += 1
            print('  FAIL %-44s 期望 qualified=%s veto=%s，实得 qualified=%s vetoes=%s n/a=%s'
                  % (name, expect_q, expect_veto, res['qualified'],
                     res['vetoes'], res['not_applicable']))
    print('\nQ_dependency_closure 独立字段检查：')
    for name, ok in closure_checks():
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
