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
# 控制台编码：中文/Ĵ 在 GBK 控制台会 UnicodeEncodeError（看起来像失败、其实是打印崩）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass


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

    # ---------- T2/T3/T4：消融开关是否**真的接进决策路径** ----------
    # 判据只有一条：两个决策序列**不得完全相同**；差异大小**不作判据**。
    same = copy.deepcopy(base)
    r = aud.compare_ablations(base, same, label='opportunity_cost')
    ck('T2：消融前后**完全相同** ⇒ engaged=False（开关没接进去）',
       r['engaged'] is False, r['note'])

    tiny = copy.deepcopy(base)
    tiny[1]['decision_hash'] = 'h1-ablated'          # 3 条里只改 1 条
    r = aud.compare_ablations(base, tiny, label='opportunity_cost')
    ck('T2：**只改 1 条**（3 条中）也判 engaged=True —— 证明**不判显著性**',
       r['engaged'] is True and r['n_differing'] == 1
       and r['action_agreement'] is not None,
       'n_diff=%r agree=%r' % (r['n_differing'], r['action_agreement']))

    # accept 率变化要被如实**报告**（但报告不等于判据）
    flipped = copy.deepcopy(base)
    flipped[2] = dict(flipped[2], action='accept', certified=True,
                      certify_energy=50.0, order_id=9, decision_hash='h2x')
    r = aud.compare_ablations(base, flipped, label='opportunity_cost')
    ck('T2：接单率变化被**报告**（base=1/3 → ablated=2/3，Δ=+1/3）',
       abs(r['accept_rate_base'] - 1 / 3) < 1e-9
       and abs(r['accept_rate_ablated'] - 2 / 3) < 1e-9
       and abs(r['delta_accept_rate'] - 1 / 3) < 1e-9, r)

    # fail-closed：日志不完整 ⇒ engaged=None（不算通过）
    broken = copy.deepcopy(base)
    del broken[1]['decision_hash']
    r = aud.compare_ablations(base, broken, label='lookahead')
    ck('T3：消融日志缺字段 ⇒ engaged=None（fail-closed，不算接入）',
       r['engaged'] is None and r['problems'], r['problems'])

    # 三开关汇总：全接入 ⇒ PASS；任一未接入 ⇒ FAIL
    allpairs = {'opportunity_cost': (base, tiny),
                'lookahead': (base, flipped),
                'scenarios': (base, copy.deepcopy(base[:2]))}
    ok, rows, pr = aud.check_ablation_engagement(allpairs)
    ck('T2/T3/T4：三开关全部接入 ⇒ PASS', ok is True and len(rows) == 3, pr)
    onepairs = dict(allpairs, scenarios=(base, copy.deepcopy(base)))
    ok, rows, pr = aud.check_ablation_engagement(onepairs)
    ck('T2/T3/T4：其中 scenarios **未接入** ⇒ FAIL 且点名该开关',
       ok is False and any('scenarios' in p for p in pr), pr)
    ok, rows, pr = aud.check_ablation_engagement({})
    ck('fail-closed：未提供任何消融对照 ⇒ FAIL（不能默认通过）',
       ok is False and any('cases 为空' in p for p in pr), pr)
    ok, rows, pr = aud.check_ablation_engagement({'scenarios': base})
    ck('fail-closed：对照不是一对日志 ⇒ FAIL', ok is False, pr)

    # ---------- ★ 2026-10-14（审计 D3.3）：「完全相同」只在**机制确实被触发过**时判 FAIL ----------
    pair_same = (base, copy.deepcopy(base))
    # (a) 完全相同 + 未提供 activation ⇒ **not_applicable**（不是 FAIL）
    ok, rows, pr = aud.check_ablation_engagement({'opportunity_cost': pair_same})
    ck('★ (a) 完全相同 + activation 未提供 ⇒ not_applicable（不判 FAIL）',
       rows[0]['engaged'] is None and 'not_applicable' in rows[0]['note'], rows[0]['note'])
    # (b) 完全相同 + activation = 0（本集合内机制从未触发）⇒ **not_applicable**
    ok, rows, pr = aud.check_ablation_engagement(
        {'opportunity_cost': pair_same}, activation={'opportunity_cost': 0})
    ck('★ (b) 完全相同 + activation=0 ⇒ not_applicable（资源不稀缺时正确实现也会相同）',
       rows[0]['engaged'] is None and rows[0]['activation_count'] == 0, rows[0])
    # (c) 完全相同 + activation > 0（机制被触发过却毫无差别）⇒ **FAIL**
    ok, rows, pr = aud.check_ablation_engagement(
        {'opportunity_cost': pair_same}, activation={'opportunity_cost': 7})
    ck('★ (c) 完全相同 + activation>0 ⇒ **FAIL**（机制触发过却无差别 = 真没接线）',
       ok is False and rows[0]['engaged'] is False
       and any('opportunity_cost' in p for p in pr), pr)
    # (d) 有差异 + activation=0 ⇒ 仍判 engaged（有差异即说明机制起了作用）
    ok, rows, pr = aud.check_ablation_engagement(
        {'opportunity_cost': (base, tiny)}, activation={'opportunity_cost': 0})
    ck('★ (d) 有差异 + activation=0 ⇒ 仍判已接入（差异本身就是证据）',
       rows[0]['engaged'] is True, rows[0])
    # (e) activation 经 audit() 透传
    r = aud.audit(base, ablations={'opportunity_cost': pair_same},
                  ablation_activation={'opportunity_cost': 9})
    ck('★ (e) audit() 透传 ablation_activation ⇒ 记为失败',
       r['checks']['T2_T3_T4_ablation_engagement'] is False
       and r['ablation_rows'][0]['activation_count'] == 9, r['problems'])

    # audit() 的整合与"不适用 ≠ 失败"
    r = aud.audit(base, log_repeat=copy.deepcopy(base), ablations=allpairs)
    ck('audit() 整合：T2–T7 全过', r['passed'] is True and r['problems'] == [],
       r['problems'])
    r = aud.audit(base)
    # ⚠️ 2026-10-14 用户裁定③：T2 拆成 T2a（预设压力夹具上的实现正确性）与 T2b（真实日效果可识别性）。
    #    两者未提供时也进 `not_applicable` ⇒ 条目数由 2 变 4；判据改为"点名包含"而非"恰好等于 2"。
    ck('audit()：未提供重复日志/消融/T2a/T2b ⇒ 记 None（不适用）且**不进 problems**，整体 passed',
       r['checks']['T7_determinism'] is None
       and r['checks']['T2_T3_T4_ablation_engagement'] is None
       and r['checks']['T2a_shadow_price_correctness'] is None
       and r['problems'] == [] and len(r['not_applicable']) == 4
       and any('T7' in s for s in r['not_applicable'])
       and any('T2/T3/T4' in s for s in r['not_applicable'])
       and any('T2a' in s for s in r['not_applicable'])
       and any('T2b' in s for s in r['not_applicable'])
       and r['passed'] is True,
       'problems=%s na=%s' % (r['problems'], r['not_applicable']))
    r = aud.audit(base, ablations=onepairs)
    ck('audit()：有开关未接入 ⇒ passed=False', r['passed'] is False)

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
