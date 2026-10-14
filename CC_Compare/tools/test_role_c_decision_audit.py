#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_decision_audit` 的回归：**每条不变式都必须能被违反并被抓到**。

方法：先造一份**干净日志**（全部通过），再对每种违规**只改一个字段**，断言被抓到。
另含 fail-closed 反例（缺字段 / 缺数值）与 T7 的"不适用 vs 失败"区分。

纯合成数据；**不读效用、不跑实验**。

运行：python CC_Compare/tools/test_role_c_decision_audit.py   （退出码 0 = 全过）
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import role_c_decision_audit as aud                           # noqa: E402

B = 710.6548152249378


def clean_log():
    """3 个事件：wait → accept → reject。承诺单调增长、已执行前缀冻结、无因果越界。"""
    return [
        dict(inst_idx=0, day_id=0, clock=1.0, info_max_reveal=1.0, order_id=None,
             action='wait', certified=False, certify_energy=None, budget_B=B,
             accepted_before=[], accepted_after=[],
             executed_before=[], executed_after=[], decision_hash='h0'),
        dict(inst_idx=0, day_id=0, clock=2.0, info_max_reveal=2.0, order_id=7,
             action='accept', certified=True, certify_energy=100.0, budget_B=B,
             accepted_before=[], accepted_after=[7],
             executed_before=[], executed_after=[], decision_hash='h1'),
        dict(inst_idx=0, day_id=0, clock=3.0, info_max_reveal=2.5, order_id=9,
             action='reject', certified=False, certify_energy=None, budget_B=B,
             accepted_before=[7], accepted_after=[7],
             executed_before=[7], executed_after=[7], decision_hash='h2'),
    ]


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    base = clean_log()

    # ---------- 干净日志必须全过 ----------
    r = aud.audit(base, log_repeat=copy.deepcopy(base))
    ck('干净日志 ⇒ T5/T6/T7 全过', r['passed'] is True and r['problems'] == [],
       'problems=%s' % r['problems'])

    # ---------- T5：因果越界 ----------
    bad = copy.deepcopy(base)
    bad[1]['info_max_reveal'] = 99.0                     # 用了 clock=2 之后的未来信息
    ok, pr = aud.check_order_and_causality(bad)
    ck('T5：用了真实未来信息（info_max_reveal > clock）⇒ FAIL',
       ok is False and any('因果越界' in p for p in pr), pr)

    # ---------- T5：先提交后认证 ----------
    bad = copy.deepcopy(base)
    bad[1]['certified'] = False
    ok, pr = aud.check_order_and_causality(bad)
    ck('T5：accept 时 certified=False（先提交后认证）⇒ FAIL',
       ok is False and any('顺序违规' in p for p in pr), pr)
    ck('T5：reject 不需要 certified（合法）',
       aud.check_order_and_causality([dict(base[2], certified=False)])[0] is True)

    # ---------- T6：承诺被撤销 ----------
    bad = copy.deepcopy(base)
    bad[2]['accepted_after'] = []                        # 把已接受的 7 撤掉了
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('T6：承诺被撤销（accepted 集合缩小）⇒ FAIL',
       ok is False and any('承诺被撤销' in p for p in pr), pr)

    # ---------- T6：已执行前缀回退 ----------
    bad = copy.deepcopy(base)
    bad[2]['executed_after'] = []
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('T6：已执行前缀回退 ⇒ FAIL',
       ok is False and any('前缀被改写' in p for p in pr), pr)

    # ---------- T6：预算越界 ----------
    bad = copy.deepcopy(base)
    bad[1]['certify_energy'] = B + 1.0
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('T6：certify_energy > B ⇒ FAIL',
       ok is False and any('预算越界' in p for p in pr), pr)

    # ---------- T6：重复接单 ----------
    bad = copy.deepcopy(base)
    bad[1]['accepted_before'] = [7]
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('T6：重复接同一单 ⇒ FAIL',
       ok is False and any('重复接单' in p for p in pr), pr)

    # ---------- T6：非接单事件却改了承诺 ----------
    bad = copy.deepcopy(base)
    bad[2]['accepted_after'] = [7, 9]                    # reject 却新增承诺
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('T6：reject 却改变承诺集合 ⇒ FAIL',
       ok is False and any('非接单事件却改变' in p for p in pr), pr)

    # ---------- fail-closed：缺字段 / 缺数值 ----------
    bad = copy.deepcopy(base)
    del bad[1]['certified']
    ok, pr = aud.check_order_and_causality(bad)
    ck('fail-closed：缺字段 ⇒ FAIL（没记录 ≠ 没违反）',
       ok is False and any('缺字段' in p for p in pr), pr)
    bad = copy.deepcopy(base)
    bad[1]['certify_energy'] = None
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('fail-closed：accept 未记 certify_energy ⇒ FAIL（无法证明 ≤ B）',
       ok is False and any('缺数值' in p for p in pr), pr)
    bad = copy.deepcopy(base)
    bad[0]['action'] = 'ACCEPT'
    ok, pr = aud.check_order_and_causality(bad)
    ck('fail-closed：action 非法值 ⇒ FAIL', ok is False and any('非法' in p for p in pr), pr)

    # ---------- T7 ----------
    ok, pr = aud.check_determinism(base, copy.deepcopy(base))
    ck('T7：两次运行完全相同 ⇒ PASS', ok is True and pr == [], pr)
    b2 = copy.deepcopy(base)
    b2[1]['decision_hash'] = 'DIFFERENT'
    ok, pr = aud.check_determinism(base, b2)
    ck('T7：第 2 条决策哈希不同 ⇒ FAIL', ok is False and any('不一致' in p for p in pr), pr)
    ok, pr = aud.check_determinism(base, base[:2])
    ck('T7：决策条数不同 ⇒ FAIL', ok is False and any('条数不同' in p for p in pr), pr)

    # ---------- T7 不适用 ≠ 失败 ----------
    r = aud.audit(base)
    ck('未提供重复日志 ⇒ T7 记 None（不适用），整体仍 passed',
       r['checks']['T7_determinism'] is None and r['passed'] is True)

    # ---------- 结构性：schema 常量化（必填 ∪ 条件必填 == 完整字段集）----------
    all_cond = set()
    for _a, fs in aud.CONDITIONAL_FIELDS.items():
        all_cond |= set(fs)
    ck('schema 自洽：REQUIRED_FIELDS ∪ 条件必填 == 干净日志字段集',
       (set(aud.REQUIRED_FIELDS) | all_cond) == set(base[0].keys()),
       'diff=%s' % ((set(aud.REQUIRED_FIELDS) | all_cond) ^ set(base[0].keys())))
    ck('条件必填只对 accept 生效（reject/wait 无需 certify_energy）',
       set(aud.CONDITIONAL_FIELDS.keys()) == {'accept'}
       and aud._missing(base[2], 0) is None)
    bad = copy.deepcopy(base)
    del bad[1]['order_id']
    ok, pr = aud.check_commitments_and_budget(bad)
    ck('fail-closed：accept 缺 order_id ⇒ FAIL（条件必填）',
       ok is False and any('条件必填' in p for p in pr), pr)

    passed = sum(1 for _n, ok, _d in cases if ok)
    for n, ok, detail in cases:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
