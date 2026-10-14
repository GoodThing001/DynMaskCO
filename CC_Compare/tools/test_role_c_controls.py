#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_controls`（B-density 对照臂）的测试。

最重要的一条：**用 AST 抽取上游 `run_exp_reserve._marginal_lambda` 的源码，
`exec` 后与本模块的移植版逐点数值对拍** —— 这样"忠实移植"是被**证明**的，
而不是我口头声明的（上游函数只依赖 numpy，故可安全抽取执行；**不修改任何上游文件**）。

其余：λ 的边界语义、判决行为、与 Role C 的**标签区分**、两臂可归因性核对。

纯合成数据；**不读效用、不跑实验**。
运行：python CC_Compare/tools/test_role_c_controls.py   （退出码 0 = 全过）
"""
import ast
import os
import random
import sys
# 控制台编码：中文/Ĵ 在 GBK 控制台会 UnicodeEncodeError（看起来像失败、其实是打印崩）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass


import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import role_c_controls as ctl                                  # noqa: E402
import role_c_opportunity_cost as rc                           # noqa: E402

UPSTREAM = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem', 'scripts',
                        'evaluation', 'run_exp_reserve.py')


def extract_upstream(name='_marginal_lambda'):
    """只读地从上游源码 AST 里取出该函数并 exec（不 import 整个模块、不改文件）。"""
    with open(UPSTREAM, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        return None
    ns = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream>', 'exec'), ns)
    return ns[name]


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- ① 与上游**逐点对拍**（忠实移植的证明）----------
    up = extract_upstream()
    if up is None:
        ck('上游 _marginal_lambda 可抽取', False, '未找到函数（路径或改名？）')
    else:
        ck('上游 _marginal_lambda 可抽取（只读 AST）', True)
        rng = random.Random(20261014)
        n_pts, n_mism = 300, 0
        first_bad = None
        for _ in range(n_pts):
            m = rng.randint(0, 8)
            vals = [rng.uniform(0.0, 500.0) for _ in range(m)]
            eng = [rng.uniform(0.001, 40.0) for _ in range(m)]
            R = rng.choice([0.0, 1e-13, rng.uniform(0.0, 300.0), 1e6])
            a = ctl.density_threshold_lambda(vals, eng, R)
            b = float(up(vals, eng, R))
            if abs(a - b) > 1e-12:
                n_mism += 1
                if first_bad is None:
                    first_bad = (vals, eng, R, a, b)
        ck('① 与上游 _marginal_lambda 逐点数值一致（%d 组随机输入）' % n_pts,
           n_mism == 0, '不一致 %d 组，首个=%r' % (n_mism, first_bad))

    # ---------- ② 边界语义 ----------
    ck('② R ≈ 0 ⇒ λ = 0', ctl.density_threshold_lambda([10.0], [1.0], 0.0) == 0.0)
    ck('② 空列表 ⇒ λ = 0', ctl.density_threshold_lambda([], [], 100.0) == 0.0)
    ck('② 全部装得下 ⇒ λ = 0（不预留）',
       ctl.density_threshold_lambda([1.0, 2.0], [1.0, 1.0], 1e6) == 0.0)
    # 手工算：密度 (10/1, 4/1)=10 > 4；R=5 ⇒ 先装 10（剩 4），再装 4（剩 0）⇒ 全装下 ⇒ 0
    ck('② 贪心按密度降序填满 ⇒ 全装下时 λ=0',
       ctl.density_threshold_lambda([10.0, 4.0], [1.0, 1.0], 5.0) == 0.0)
    # R=1.5：装 10/1（剩 0.5），下一个 4/1 装不下 ⇒ λ = 4/1 = 4
    ck('② 装不下时返回**边界密度**（= 第一个装不下的订单的 vals/energy）',
       abs(ctl.density_threshold_lambda([10.0, 4.0], [1.0, 1.0], 1.5) - 4.0) < 1e-12,
       ctl.density_threshold_lambda([10.0, 4.0], [1.0, 1.0], 1.5))

    # ---------- ③ 判决行为（REV − λ·e > 0 ∧ certify）----------
    class _O:
        order_id = 3

    o = _O()
    plan = {'p': 1}
    # 未来订单：密度 10/1 ⇒ R=1.5 时 λ=4；当前 REV=3、e=1 ⇒ 3−4 = −1 ⇒ 拒
    kw = dict(order=o, candidate_plan=plan, budget_B=1.5, revenue_fn=lambda _o: 3.0,
              marginal_energy_fn=lambda _o: 1.0, certify_fn=lambda p: (True, 0.5),
              future_vals=[10.0, 4.0], future_energys=[1.0, 1.0])
    acc, d = ctl.decide_b_density_control(**kw)
    ck('③ B-density：REV=3 < λ·e=4 ⇒ 拒单',
       acc is False and abs(d['lambda_B'] - 4.0) < 1e-12, d)
    acc, d = ctl.decide_b_density_control(**dict(kw, revenue_fn=lambda _o: 5.0))
    ck('③ B-density：REV=5 > 4 ⇒ 接单', acc is True, d)
    acc, d2 = ctl.decide_b_density_control(**dict(kw, certify_fn=lambda p: (False, 9.0)))
    ck('③ B-density：certify 不过 ⇒ 拒单（硬约束优先）', acc is False, d2)

    # ---------- ④ 与 Role C 的**标签区分**（审计可机械分辨）----------
    ck('④ 对照臂 lambda_estimator = density_threshold_knapsack',
       d['lambda_estimator'] == rc.LAMBDA_ESTIMATOR_DENSITY)
    ck('④ 对照臂 arm 字段 = B_density_control（不冒充 Role C）',
       d['arm'] == 'B_density_control')
    # Role C 侧仍然**拒绝**密度阈值标签（护栏未被削弱）
    raised = False
    try:
        rc.decide_candidate(order=o, candidate_plan=plan, budget_B=1.5,
                            myopic_net_fn=lambda _o: 3.0, marginal_energy_fn=lambda _o: 1.0,
                            future_value_fn=lambda b: float(b), certify_fn=lambda p: (True, 0.5),
                            lambda_estimator=rc.LAMBDA_ESTIMATOR_DENSITY)
    except ValueError as e:
        raised = '换名重用' in str(e)
    ck('④ Role C 侧仍拒绝密度阈值 λ（护栏未被本模块削弱）', raised)

    # ---------- ⑤ 两臂可归因性核对 ----------
    # ⚠️ 2026-10-14 D2b：原用例拿 `myopic_net_fn=3.0`（shadow）与 `REV=5`（density）
    #    构造了一对**近视基准不同**的臂，并断言 `matched=True` —— 这正是漏检的示范。
    #    现改为：基准不同 ⇒ **必须**判不可归因；并补一条基准一致的正例。
    a_true, d_shadow = rc.decide_candidate(
        order=o, candidate_plan=plan, budget_B=1.5,
        myopic_net_fn=lambda _o: 3.0, marginal_energy_fn=lambda _o: 1.0,
        future_value_fn=lambda b: b, certify_fn=lambda p: (True, 0.5))
    ok, diffs = ctl.arms_are_matched(d_shadow, d)
    ck('⑤ 近视基准不同（Role C 3.0 vs B-density REV=5）⇒ **不得**判可归因',
       ok is False and any(k == 'myopic_net' for k, _a, _b in diffs), diffs)

    d_shadow_matched = dict(d_shadow)
    d_shadow_matched['myopic_net'] = d['myopic_net']
    # ⚠️ 2026-10-14 用户裁定①：除数值外还须对齐**语义身份** `myopic_base`，
    #    否则"恰好把数调成一样"会被当成"基准相同"。只对齐数值 ⇒ 仍判不可归因
    #    （该反例在 test_role_c_adjudication_20261014.py 中单独断言）。
    d_shadow_matched['myopic_base'] = d['myopic_base']
    ok2, diffs2 = ctl.arms_are_matched(d_shadow_matched, d)
    ck('⑤ 近视基准数值+语义身份都对齐后，其余共享字段一致 + λ 标签不同 ⇒ 可归因',
       ok2 is True, diffs2)

    # ★ 2026-10-14（用户裁定「只替换 λ 估计器」+ 审计 D2b）：走**真实代码路径**的正例 ——
    #   给 B-density 传与 Role C **同一个** myopic_net_fn ⇒ 两臂真单因素 ⇒ 可归因。
    _acc, d_aligned = ctl.decide_b_density_control(
        **dict(kw, myopic_net_fn=lambda _o: 3.0))
    ok3, diffs3 = ctl.arms_are_matched(d_shadow, d_aligned)
    ck('★⑤ B-density 传与 Role C 同一个 myopic_net_fn ⇒ 真单因素 ⇒ 可归因（无需手改 dict）',
       ok3 is True and abs(d_aligned['myopic_net'] - 3.0) < 1e-12, diffs3)
    ck('★⑤ 且诊断里标明基准来源与保真度（不再冒充上游 knn/valuee）',
       '同一基准' in d_aligned['myopic_net_note']
       and '上游口径' in d['myopic_net_note'], (d_aligned['myopic_net_note'],
                                                d['myopic_net_note']))

    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
