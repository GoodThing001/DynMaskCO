#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_opportunity_cost` 的 **T9 值函数自洽**回归（纯函数、**不读效用**、不跑实验）。

对应用户 2026-10-14 纠错后的判据（`P1A_RoleC_设计规格` §2.1.1）：

  ① `λ_B ≥ 0`（含采样噪声导致的负差值情形）；
  ② `e_o = 0 ⇒ λ_B·e_o = 0`（不得凭空产生机会成本）；
  ③ 把 `Ĵ` 换成**常数** ⇒ `λ_B ≡ 0`（证明 λ_B 确实只来自 Ĵ 的**预算响应**）；
  ④ **不存在**把 `Ĵ` 再加进接单判据的第二项 —— 即写法 (iii) 被**API 层拒绝**；
  ⑤ 硬约束优先：`certify_ok=False` ⇒ 即使净收益巨大也必须拒单；
  ⑥ 单调性：`λ_B` 或 `e_o` 增大 ⇒ 接单**不会**变得更容易；
  ⑦ 消融 `λ_B≡0` ⇒ 判据退化为 `myopic_net > 0 ∧ certify_ok`（"可行即接"）。

运行：python CC_Compare/tools/test_role_c_opportunity_cost.py   （退出码 0 = 全过）
"""
import os
import sys
# 控制台编码：中文/Ĵ 在 GBK 控制台会 UnicodeEncodeError（看起来像失败、其实是打印崩）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import role_c_opportunity_cost as rc                          # noqa: E402


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- ① λ_B ≥ 0（即便 Ĵ 的差值为负）----------
    B, e = 710.6548152249378, 5.0
    lam, d = rc.shadow_price_budget(B, e, lambda b: 1000.0 + 0.0 * b)   # 常数 ⇒ 差 0
    ck('① Ĵ 为常数 ⇒ λ_B = 0', lam == 0.0, 'λ_B=%r' % lam)
    lam_neg, d_neg = rc.shadow_price_budget(B, e, lambda b: (0.0 if b >= B else 50.0))
    # Ĵ(B)=0 < Ĵ(B−e)=50 ⇒ 差值 −50/5 = −10 ⇒ 必须截到 0
    ck('① 差值为负（噪声）⇒ λ_B 截到 0（不得为负）', lam_neg == 0.0 and d_neg['lambda_raw'] < 0,
       'raw=%r λ=%r' % (d_neg['lambda_raw'], lam_neg))
    lam_pos, d_pos = rc.shadow_price_budget(B, e, lambda b: 0.0 if b >= B else -100.0)
    ck('① 正常情况下 λ_B = [Ĵ(B)−Ĵ(B−e)]/e > 0', lam_pos > 0 and abs(lam_pos - 20.0) < 1e-9,
       'λ_B=%r' % lam_pos)

    # ---------- ② e_o = 0 ⇒ 机会成本 0 ----------
    lam0, d0 = rc.shadow_price_budget(B, 0.0, lambda b: float(b) * 3.0)
    ck('② e_o=0 ⇒ λ_B=0', lam0 == 0.0, 'λ_B=%r' % lam0)
    acc0, g0 = rc.decide(myopic_net=1.0, e_o=0.0, lambda_B=999.0, certify_ok=True)
    ck('② e_o=0 ⇒ 机会成本=0 且 λ_B 不影响判决',
       g0['opportunity_cost'] == 0.0 and acc0 is True,
       'cost=%r' % g0['opportunity_cost'])

    # ---------- ③ Ĵ 换常数 ⇒ λ_B ≡ 0（对任意 B / e 都成立）----------
    tot = 0
    for bb in (0.0, 1.0, 100.0, 710.6548152249378, 5000.0):
        for ee in (0.01, 1.0, 37.5, 1e6):
            l, _dd = rc.shadow_price_budget(bb, ee, lambda b: 42.0)
            tot += (l == 0.0)
    ck('③ Ĵ≡常数 ⇒ 全部 %d 组 (B,e) 的 λ_B 均为 0' % (5 * 4), tot == 20, '命中 %d/20' % tot)

    # ---------- ④ 写法 (iii) 被 API 层拒绝（防重复计机会成本）----------
    raised = False
    try:
        rc.decide(myopic_net=100.0, e_o=5.0, lambda_B=2.0, certify_ok=True,
                  lookahead_delta=30.0)
    except ValueError as exc:
        raised = '重复计机会成本' in str(exc)
    ck('④ 同时传 λ_B 与 lookahead_delta ⇒ 抛错（写法 (iii) 禁用）', raised)

    # ---------- ⑤ 硬约束优先 ----------
    acc_h, g_h = rc.decide(myopic_net=1e9, e_o=1.0, lambda_B=0.0, certify_ok=False)
    ck('⑤ certify_ok=False ⇒ 净收益再大也拒单',
       acc_h is False and 'certify_plan 未通过' in g_h['reject_reason'])

    # ---------- ⑥ 单调性 ----------
    def acc(mn, eo, lB):
        return rc.decide(myopic_net=mn, e_o=eo, lambda_B=lB, certify_ok=True)[0]
    mono = (acc(50, 5, 1) and not acc(50, 50, 1)        # e_o↑ ⇒ 更难接
            and acc(50, 5, 1) and not acc(50, 5, 20)    # λ_B↑ ⇒ 更难接
            and not acc(50, 5, 20) and acc(500, 5, 20))  # myopic_net↑ ⇒ 更易接
    ck('⑥ 单调：e_o↑ / λ_B↑ ⇒ 更难接；myopic_net↑ ⇒ 更易接', mono)

    # ---------- ⑦ 消融 λ_B≡0 ⇒ "可行即接" ----------
    # 未消融：myopic_net=8, e_o=5, λ_B=2 ⇒ 8−10 = −2 ⇒ 拒
    a_on, g_on = rc.decide(myopic_net=8.0, e_o=5.0, lambda_B=2.0, certify_ok=True)
    a_off, g_off = rc.decide(myopic_net=8.0, e_o=5.0, lambda_B=2.0, certify_ok=True,
                             ablate_opportunity_cost=True)
    ck('⑦ 消融前拒、消融后接（λ_B≡0 ⇒ 退化为"可行即接"）',
       a_on is False and a_off is True and g_off['opportunity_cost'] == 0.0,
       'on=%s off=%s' % (a_on, a_off))
    ck('⑦ 消融后判据确实等于 myopic_net>0 ∧ certify_ok（恒等核验）',
       all(rc.decide(myopic_net=mn, e_o=3.0, lambda_B=7.0, certify_ok=True,
                     ablate_opportunity_cost=True)[0] == (mn > 0)
           for mn in (-1.0, 0.0, 0.5, 100.0)))

    # ---------- 边界：B − e_o < 0 ----------
    lb, db = rc.shadow_price_budget(3.0, 5.0, lambda b: -b)
    ck('边界：e_o > B ⇒ 评估点截到 0 且 infeasible 留痕',
       db['infeasible'] is True and db['b_lo_eval'] == 0.0, 'diag=%s' % {k: db[k] for k in ('b_lo', 'b_lo_eval', 'infeasible')})

    # ---------- myopic_net 不得再扣机会成本；符号为 accept−reject 净增益 ----------
    # ⚠️ 2026-10-14 D1 修正：此处原断言 `== 60.0` 把**错符号**写死成期望值
    #    （`revenue − fuel − reject_loss`）；正确值为 `100 − 30 + 10 = 80.0`。
    #    审计见 CC_Compare/P1A_RoleC_效用公式与消融门审计_2026-10-14.md §1.4。
    ck('myopic_net = revenue − fuel **+** reject_loss（接单避免了拒单罚金）',
       rc.myopic_net(100.0, 30.0, 10.0) == 80.0)
    ck('myopic_net 反例：p_c=15/R=10/F=0 ⇒ 必须判接单（旧符号会误拒）',
       rc.myopic_net(10.0, 0.0, 15.0) == 25.0
       and rc.decide(myopic_net=rc.myopic_net(10.0, 0.0, 15.0), e_o=0.0,
                     lambda_B=0.0, certify_ok=True)[0] is True)
    ck('myopic_net 反例（p_c=15 占用能耗后仍应接单，偏差 2·p_c=30 > δ=20）',
       rc.decide(myopic_net=rc.myopic_net(10.0, 0.0, 15.0), e_o=2.0,
                 lambda_B=5.0, certify_ok=True)[1]['net_after_opportunity_cost'] == 15.0)

    # ---------- 规格 ↔ 项目实现的**契约接地**（AST 静态检查，不导入、不跑实验）----------
    # 规格 §2.1 假设 `e_o = c0_marginal_energy(c, q, cooling_share, contract)`、
    # 硬约束来自 `certify_plan(..., contract, B)`。若上游改了签名，规格必须同步 ——
    # 这里用 AST 把该绑定**机器化**，避免"文档写着、代码早变了"。
    ck('契约接地：c0_marginal_energy / certify_plan 签名与规格一致', *[_contract_grounding()])
    ck('契约接地的**反面**：签名漂移必须被抓到', *[_contract_grounding_negative()])

    # ---------- 候选评估核心 decide_candidate（全注入 ⇒ 离线可测）----------
    # 用一个最小 order 替身 + 注入的四个函数，避免依赖 env / 项目模块。
    class _O:
        order_id = 7

    o = _O()
    plan = {'plan': 'stub'}
    B = 100.0
    # Ĵ: 预算越少价值越低 ⇒ λ_B > 0：J(b)=b 时 λ_B = (B-(B-e))/e = 1
    jhat = lambda b: float(b)
    cert_ok = lambda p: (True, 5.0)
    mn_fn = lambda _o: 3.0            # myopic_net = 3
    e_fn = lambda _o: 5.0             # e_o = 5 ⇒ 机会成本 = 1*5 = 5 ⇒ 3−5 = −2 ⇒ 拒
    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                                future_value_fn=jhat, certify_fn=cert_ok)
    ck('decide_candidate：λ_B=1、机会成本=5 ⇒ 净 −2 ⇒ 拒单',
       acc is False and abs(d['lambda_B'] - 1.0) < 1e-9
       and abs(d['opportunity_cost'] - 5.0) < 1e-9, d)

    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                                future_value_fn=jhat, certify_fn=cert_ok,
                                ablate_opportunity_cost=True)
    ck('decide_candidate：A 版（λ_B 强制 0）⇒ 3 > 0 ⇒ 接单，且 lambda_source 标为消融',
       acc is True and d['lambda_B'] == 0.0
       and d['lambda_diag']['lambda_source'] == 'ablate_opportunity_cost', d)

    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                                future_value_fn=jhat, certify_fn=cert_ok,
                                ablate_lookahead=True)
    ck('decide_candidate：去前瞻 ⇒ λ_B 也只能是 0，且 lambda_source 标为 ablate_lookahead',
       acc is True and d['lambda_B'] == 0.0
       and d['lambda_diag']['lambda_source'] == 'ablate_lookahead', d)
    ck('★ 两个消融**机制不同但 λ_B 都为 0**（诊断可区分，故 B−A 不被混淆）',
       d['lambda_diag']['lambda_source'] != 'ablate_opportunity_cost')

    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                myopic_net_fn=lambda _o: 1e9, marginal_energy_fn=e_fn,
                                future_value_fn=jhat, certify_fn=lambda p: (False, 999.0))
    ck('decide_candidate：certify 不过 ⇒ 净收益再大也拒单',
       acc is False and 'certify_plan 未通过' in d['reject_reason'], d)

    # ---------- 与写入器的集成：接受/拒绝都能落盘并通过审计 ----------
    import role_c_decision_log as rlog
    lg = rlog.DecisionLogger(B)
    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                                future_value_fn=jhat, certify_fn=cert_ok,
                                ablate_opportunity_cost=True)
    rc.log_decision(lg, d, inst_idx=0, day_id=0, clock=2.0, info_max_reveal=2.0,
                    decision_hash='d0')
    acc2, d2 = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                                  myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                                  future_value_fn=jhat, certify_fn=cert_ok)
    rc.log_decision(lg, d2, inst_idx=0, day_id=0, clock=3.0, info_max_reveal=3.0,
                    decision_hash='d1')
    r = lg.audit(log_repeat=[dict(x) for x in lg.records])
    ck('集成：decide_candidate → log_decision → 审计全过（接单+拒单各一）',
       r['passed'] is True and r['problems'] == []
       and [x['action'] for x in lg.records] == ['accept', 'reject'], r['problems'])

    # ---------- 机制独立性护栏（规格 §2.6）：不得沿用既有 knn/valuee 的密度阈值 λ ----------
    ck('默认 lambda_estimator = 内生影子价格（shadow_price_budget_response）',
       d['lambda_estimator'] == rc.LAMBDA_ESTIMATOR_SHADOW)
    raised = False
    try:
        rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                            myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                            future_value_fn=jhat, certify_fn=cert_ok,
                            lambda_estimator=rc.LAMBDA_ESTIMATOR_DENSITY)
    except ValueError as exc:
        raised = '换名重用' in str(exc)
    ck('★ 若声明用既有密度阈值 λ ⇒ **拒绝执行**（否则 Role C 退化为 knn 同胞）', raised)
    raised = False
    try:
        rc.decide_candidate(order=o, candidate_plan=plan, budget_B=B,
                            myopic_net_fn=mn_fn, marginal_energy_fn=e_fn,
                            future_value_fn=jhat, certify_fn=cert_ok,
                            lambda_estimator='whatever')
    except ValueError as exc:
        raised = '未知 lambda_estimator' in str(exc)
    ck('未知 lambda_estimator ⇒ 拒绝（fail-closed，不默认放行）', raised)

    # ---------- 事前边界规则 B2 / B3（2026-10-14 用户裁定 §三；不得按结果调整）----------
    # B2：e_o > B（预算扰动越界）⇒ 强制拒单，即使 myopic_net 巨大
    acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=1.0,
                                myopic_net_fn=lambda _o: 1e9, marginal_energy_fn=lambda _o: 5.0,
                                future_value_fn=jhat, certify_fn=cert_ok)
    ck('B2：e_o=5 > B=1（预算扰动越界）⇒ **强制拒单**（净收益再大也不接）',
       acc is False and d['boundary'] == 'B2_budget_out_of_range'
       and '越界' in d['boundary_reason'], d.get('boundary_reason'))

    # B3：V̂ 非有限 ⇒ λ_B 未定义 ⇒ 强制拒单（**不得**当作 0 —— 那会偏向接单）
    def _mk_v(kind):
        def _v(b):
            if kind == 'NaN':
                return float('nan')
            if kind == '+Inf':
                return float('inf')
            raise RuntimeError('V̂ 求值失败')
        return _v

    for label in ('NaN', '+Inf', '抛错'):
        acc, d = rc.decide_candidate(order=o, candidate_plan=plan, budget_B=100.0,
                                    myopic_net_fn=lambda _o: 1e9,   # 净收益极大
                                    marginal_energy_fn=lambda _o: 5.0,
                                    future_value_fn=_mk_v(label), certify_fn=cert_ok)
        ck('B3：V̂=%s ⇒ λ_B 未定义 ⇒ **强制拒单**' % label,
           acc is False and d['boundary'] == 'B3_lambda_undefined'
           and d['lambda_B'] is None, d.get('boundary_reason'))
    lam_u, diag_u = rc.shadow_price_budget(100.0, 5.0, lambda b: float('nan'))
    ck('B3：shadow_price_budget 对非有限 V̂ 返回 None 且 diag 标 undefined',
       lam_u is None and diag_u.get('undefined') is True, diag_u)
    lam1, d1 = rc.shadow_price_budget(100.0, 0.0, lambda b: float(b))
    ck('B1：e_o=0 ⇒ λ_B=0 且 undefined=False', lam1 == 0.0 and d1['undefined'] is False)
    lam4, d4 = rc.shadow_price_budget(100.0, 5.0, lambda b: (0.0 if b >= 100.0 else 50.0))
    ck('B4：负差值 ⇒ λ_B 截到 0 且 diag.clipped=True',
       lam4 == 0.0 and d4['clipped'] is True, d4)

    # ---------- D2.4：把"λ_B ≥ 0"这个**同义反复**换成有信息量的**报告量** ----------
    # λ_B 由 clip 构造 ⇒ "≥0"永不可能失败，不构成证据；证据在**未截断**的 lambda_raw。
    diags = [rc.shadow_price_budget(100.0, 5.0, lambda b: (0.0 if b >= 100.0 else 50.0))[1],
             rc.shadow_price_budget(100.0, 5.0, lambda b: float(b))[1],
             rc.shadow_price_budget(100.0, 5.0, lambda b: float('nan'))[1],
             rc.shadow_price_budget(100.0, 0.0, lambda b: float(b))[1]]
    st = rc.clipped_stats(diags)
    ck('D2.4：clipped_stats 报告计数/比例（1 个 raw<0、1 个 undefined、共 4）',
       st['n'] == 4 and st['n_clipped'] == 1 and st['n_undefined'] == 1
       and abs(st['clipped_ratio'] - 0.25) < 1e-12, st)
    ck('D2.4：明确标注**不设阈值**（诊断量而非门）', '不设阈值' in st['note'])
    ck('D2.4：空输入不炸（比例返回 None 而不是除零）',
       rc.clipped_stats([])['n'] == 0 and rc.clipped_stats([])['clipped_ratio'] is None)
    ck('边界规则清单被导出（供审计引用）',
       set(rc.BOUNDARY_RULES) == {'B1_e_o_zero', 'B2_budget_out_of_range',
                                  'B3_lambda_undefined', 'B4_negative_diff_clipped'})

    passed = sum(1 for _n, ok, _d in cases if ok)
    for n, ok, detail in cases:
        print(('  PASS ' if ok else '  FAIL ') + n + ('' if ok else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


def _contract_grounding_negative():
    """用**合成源码**证明该检查确实会失败（否则"通过"可能只是因为没检查到东西）。"""
    import tempfile
    bad = ('def c0_marginal_energy(c, q, cooling_share, contract):\n    return 0.0\n'
           'def certify_plan(env, inst_idx, plan, B):\n    return True\n')  # 少了几个参数
    p = os.path.join(tempfile.mkdtemp(prefix='rc_ground_'), 'fake.py')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(bad)
    ok, why = _contract_grounding(p)
    return (ok is False), why


def _contract_grounding(src=None):
    """用 `ast` 解析上游源码，核对两个函数签名（只读；不 import、不需要 numpy）。"""
    import ast
    if src is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src = os.path.join(root, 'C-VRP_Cold-chainVehicleRoutingProblem', 'scripts',
                           'evaluation', 'run_exp_energy_c0.py')
        if not os.path.isfile(src):
            return True, '上游源码不在本机（跳过；在持有树的机器上运行才会真正核对）'
    with open(src, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    want = {'c0_marginal_energy': ['c', 'q', 'cooling_share', 'contract'],
            'certify_plan': ['env', 'inst_idx', 'clock', 'vehicles', 'served_mask',
                             'plan', 'contract', 'B']}
    got = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in want:
            got[node.name] = [a.arg for a in node.args.args]
    for fn, exp in want.items():
        if got.get(fn) != exp:
            return False, '%s 签名 %r ≠ 规格假设 %r' % (fn, got.get(fn), exp)
    return True, '两个签名均与规格一致'


if __name__ == '__main__':
    sys.exit(main())
