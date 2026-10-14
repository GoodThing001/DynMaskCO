#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2026-10-14 用户四项裁定中 ①③ 的回归（纯函数、**不读效用**、不跑实验）。

覆盖：
  ① 近视基准：`m_o = R − ΔF + p_c`；**已经是完整 J_acc−J_rej 的量不得再加 p_c**；
     两臂语义身份（`myopic_base`）必须一致才算单因素；`factor_status` 的 two_factor 判定。
  ③ T2 拆分：T2a 在**预设压力夹具**上判数值行为（不要求动作不同）；
     T2b 四级可识别性，且必须用**截断后**的有效 λ_B，而不是 `raw>0`。

运行：python CC_Compare/tools/test_role_c_adjudication_20261014.py   （退出码 0 = 全过）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import role_c_opportunity_cost as rc        # noqa: E402
import role_c_controls as ctl               # noqa: E402
import role_c_decision_audit as aud         # noqa: E402

CASES = []


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))


def main():
    # ---------------- ① 近视基准与符号 ----------------
    ck('① m_o = R − ΔF + p_c（拒单罚金是正向增量）',
       rc.myopic_net(100.0, 30.0, 10.0) == 80.0)
    ck('① 反例 p_c=15/R=10/ΔF=0 ⇒ +25 ⇒ 接单',
       rc.myopic_net(10.0, 0.0, 15.0) == 25.0
       and rc.decide(myopic_net=25.0, e_o=0.0, lambda_B=0.0, certify_ok=True)[0] is True)

    ck('① 已是完整 J_acc−J_rej 的量原样返回（不加 p_c）',
       rc.myopic_net_from_full_margin(25.0) == 25.0)
    raised = False
    try:
        rc.myopic_net_from_full_margin(25.0, reject_loss=15.0)
    except ValueError as e:
        raised = '重复计入' in str(e)
    ck('① 完整差值 + reject_loss ⇒ 抛错（防"修好 D1 后再重复计一次"）', raised)

    v, base = rc.myopic_net_from_components_or_margin(
        myopic_base=rc.MYOPIC_BASE_COMPONENTS, revenue=10.0, fuel=0.0, reject_loss=15.0)
    ck('① 分量路径 ⇒ 25.0 且标注 components', v == 25.0 and base == rc.MYOPIC_BASE_COMPONENTS)
    v2, base2 = rc.myopic_net_from_components_or_margin(
        myopic_base=rc.MYOPIC_BASE_FULL_MARGIN, margin=25.0)
    ck('① 完整差值路径 ⇒ 25.0 且标注 full_margin', v2 == 25.0 and base2 == rc.MYOPIC_BASE_FULL_MARGIN)
    raised2 = False
    try:
        rc.myopic_net_from_components_or_margin(myopic_base='nonsense', margin=1.0)
    except ValueError:
        raised2 = True
    ck('① 未知 myopic_base ⇒ 拒绝（fail-closed）', raised2)

    # ---------------- ① 两臂单因素可归因 ----------------
    class _O:
        order_id = 3

    o, plan = _O(), {'p': 1}
    kw = dict(order=o, candidate_plan=plan, budget_B=1.5,
              marginal_energy_fn=lambda _x: 1.0, certify_fn=lambda _p: (True, 0.5),
              future_vals=[10.0, 4.0], future_energys=[1.0, 1.0])
    # 单因素：两臂同基准（Role C 用 myopic_net_fn，B-density 也传同一个）
    same_fn = lambda _x: 3.0
    _a, d_shadow_same = rc.decide_candidate(
        order=o, candidate_plan=plan, budget_B=1.5, myopic_net_fn=same_fn,
        marginal_energy_fn=lambda _x: 1.0, future_value_fn=lambda b: b,
        certify_fn=lambda _p: (True, 0.5))
    _b, d_density_same = ctl.decide_b_density_control(
        revenue_fn=lambda _x: 3.0, myopic_net_fn=same_fn, **kw)
    st_same, _d1 = ctl.factor_status(d_shadow_same, d_density_same)
    ck('① 两臂同基准 ⇒ factor_status = single_factor（主对照）', st_same == 'single_factor')

    # 双因素：B-density 沿用 REV 上游口径 ⇒ 只能算探索性第二因素
    _c, d_density_rev = ctl.decide_b_density_control(
        revenue_fn=lambda _x: 5.0, myopic_net_fn=None, **kw)
    st_two, diffs_two = ctl.factor_status(d_shadow_same, d_density_rev)
    ck('① B-density 用 REV ⇒ factor_status = two_factor_exploratory',
       st_two == 'two_factor_exploratory'
       and any(k == 'myopic_net' for k, _x, _y in diffs_two))
    ck('① REV 臂被标为 legacy 基准（不冒充 Role C）',
       d_density_rev.get('myopic_base') == rc.MYOPIC_BASE_REVENUE_ONLY
       and d_shadow_same.get('myopic_base') == rc.MYOPIC_BASE_COMPONENTS
       and d_density_same.get('myopic_base') == rc.MYOPIC_BASE_COMPONENTS)

    # 语义身份：数值偶然相同但组装方式不同 ⇒ 仍不可归因
    d_fake = dict(d_shadow_same)
    d_fake['myopic_base'] = rc.MYOPIC_BASE_FULL_MARGIN
    ok_fake, diffs_fake = ctl.arms_are_matched(d_fake, d_density_same)
    ck('① 数值相同但 myopic_base 语义不同 ⇒ 仍判不可归因',
       ok_fake is False and any(k == 'myopic_base' for k, _x, _y in diffs_fake))

    # ---------------- ③ T2a：预设压力夹具上的数值行为 ----------------
    fixture = [
        {'budget': 100.0, 'e_o': 2.0, 'v_hi': 200.0, 'v_lo': 190.0},   # raw=5 ⇒ λ=5
        {'budget': 10.0, 'e_o': 4.0, 'v_hi': 100.0, 'v_lo': 108.0},    # raw<0 ⇒ 截断 0
        {'budget': 100.0, 'e_o': 0.0, 'v_hi': 200.0, 'v_lo': 100.0},   # e=0 ⇒ λ=0
        {'budget': 1.0, 'e_o': 5.0, 'v_hi': 100.0, 'v_lo': 90.0},      # e>B ⇒ infeasible
    ]
    ok_a, rows_a, prob_a = aud.check_t2a_shadow_price_correctness(fixture)
    ck('③ T2a 夹具数值行为全符公式 ⇒ ok', ok_a is True and not prob_a,
       [r['checks'] for r in rows_a])
    ck('③ T2a 断言 raw 公式 / 截断 / 非负 / e=0 / 越界 五项都被逐条检查',
       len(rows_a) == 4 and all(len(r['checks']) == 6 for r in rows_a))
    ck('③ T2a 越界夹具确实置 infeasible 且评估点截到 0',
       rows_a[3]['infeasible'] is True)
    ck('③ T2a 负原始值被截断为 0（lambda_raw<0 但 lambda_B=0）',
       rows_a[1]['lambda_raw'] < 0 and rows_a[1]['lambda_B'] == 0.0)
    ok_e, rows_e, prob_e = aud.check_t2a_shadow_price_correctness([])
    ck('③ T2a 无夹具 ⇒ fail-closed（不算通过）', ok_e is False and bool(prob_e))

    # ---------------- ③ T2b：四级可识别性 ----------------
    v_na_raw, r1, p1 = aud.check_t2b_effect_identifiability(
        {'n_decisions': 100, 'n_raw_positive': 0, 'n_lambda_effective_positive': 0,
         'n_score_changed': 0, 'n_ordering_or_action_changed': 0})
    ck('③ T2b raw>0 为 0 ⇒ not_applicable 且不计 problem',
       v_na_raw == aud.T2B_NOT_APPLICABLE and not p1)

    v_na_clip, r2, p2 = aud.check_t2b_effect_identifiability(
        {'n_decisions': 100, 'n_raw_positive': 40, 'n_lambda_effective_positive': 0,
         'n_score_changed': 0, 'n_ordering_or_action_changed': 0})
    ck('③ T2b raw>0 但截断后有效 λ_B=0 ⇒ not_applicable（不是接线失败）',
       v_na_clip == aud.T2B_NOT_APPLICABLE and not p2)

    v_wire, r3, p3 = aud.check_t2b_effect_identifiability(
        {'n_decisions': 100, 'n_raw_positive': 40, 'n_lambda_effective_positive': 12,
         'n_score_changed': 0, 'n_ordering_or_action_changed': 0})
    ck('③ T2b 有效价格>0 但评分未变 ⇒ wiring_defect（唯一 FAIL 档）',
       v_wire == aud.T2B_WIRING_DEFECT and bool(p3))

    v_noeff, r4, p4 = aud.check_t2b_effect_identifiability(
        {'n_decisions': 100, 'n_raw_positive': 40, 'n_lambda_effective_positive': 12,
         'n_score_changed': 12, 'n_ordering_or_action_changed': 0})
    ck('③ T2b 评分变但动作未变 ⇒ activated_no_behavioural_effect，且不计 problem',
       v_noeff == aud.T2B_ACTIVATED_NO_EFFECT and not p4)

    v_eff, r5, p5 = aud.check_t2b_effect_identifiability(
        {'n_decisions': 100, 'n_raw_positive': 40, 'n_lambda_effective_positive': 12,
         'n_score_changed': 12, 'n_ordering_or_action_changed': 5})
    ck('③ T2b 动作也变 ⇒ activated_with_behavioural_effect（效果好坏仍归效果门）',
       v_eff == aud.T2B_ACTIVATED_WITH_EFFECT and not p5)

    # ---------------- ③ 接入总报告：T2a/T2b 的接线缺陷进入 problems ----------------
    rep_ok = aud.audit([], t2a_fixtures=fixture,
                       t2b_stats={'n_decisions': 100, 'n_raw_positive': 40,
                                  'n_lambda_effective_positive': 12, 'n_score_changed': 12,
                                  'n_ordering_or_action_changed': 0})
    ck('③ 总报告：T2a 通过 + T2b 无行为效应 ⇒ problems 为空（不把"无效应"当失败）',
       rep_ok['problems'] == [] and rep_ok['checks']['T2a_shadow_price_correctness'] is True
       and rep_ok['t2b_verdict'] == aud.T2B_ACTIVATED_NO_EFFECT)
    rep_bad = aud.audit([], t2b_stats={'n_decisions': 100, 'n_raw_positive': 40,
                                       'n_lambda_effective_positive': 12, 'n_score_changed': 0,
                                       'n_ordering_or_action_changed': 0})
    ck('③ 总报告：T2b 接线缺陷 ⇒ problems 非空',
       bool(rep_bad['problems']) and rep_bad['t2b_verdict'] == aud.T2B_WIRING_DEFECT)
    rep_na = aud.audit([])
    ck('③ 总报告：未提供 T2a/T2b ⇒ 进 not_applicable，不进 problems',
       rep_na['problems'] == []
       and any('T2a' in s for s in rep_na['not_applicable'])
       and any('T2b' in s for s in rep_na['not_applicable']))

    passed = sum(1 for _n, ok, _d in CASES if ok)
    for n, ok, detail in CASES:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   <- ' + str(detail))))
    failed = len(CASES) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
