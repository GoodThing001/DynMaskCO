#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**真实接口接线样例**（样例，**不执行真实决策**） + **hi/lo 可观测性证明**。

授权：用户 2026-10-14「bce2e988 审查裁定」
      ① 补充决定：**要求提交真实接口的接线代码样例，但本轮不执行真实决策**；
      ② **批准先做分析侧能力边界证明**；**暂不批准**核心函数级白名单。

⚠️ 本模块**只读**被测面：不 import 冻结模块、不跑 env、不跑真实 `decide_fn`。
`wire_phase_map()` 用**桩** `decide_fn` 演示接线（零环境依赖）；
`observability_proof()` 用**真实** `shadow_price_budget` + 合约定注入器，逐场景记录
"wrapper 能直接看到什么"，**不触碰** `role_c_forward.py`。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ['WIRE_PHASES', 'OBSERVABILITY_TIERS', 'wire_phase_map',
           'observability_proof', 'main']

#: 三个必须**分开**的接线阶段（用户要求样例展示）
WIRE_PHASES = (
    'P1_freeze_input_snapshot',   # 决策前：冻结输入身份（严格 schema）
    'P2_call_decide_fn',          # 决策：只调用，不提交
    'P3_classify',                # 决策后：候选 / 拟采纳 / 实际提交 + 异常来源
)

#: 可观测性分层（用户指定）
OBSERVABILITY_TIERS = ('DIRECT', 'INFERRED', 'NOT_VERIFIABLE')


def wire_phase_map():
    """**接线样例的声明式映射**（不执行真实决策）。

    展示：固定输入快照如何进入**旧版**与 **C2 版**、如何提取 `diag['exception_audit']`、
    以及如何区分**候选计划 / 拟采纳计划 / 已经实际提交的计划**。
    """
    return {
        'phase_order': list(WIRE_PHASES),
        'graph': [
            {'phase': 'P1_freeze_input_snapshot',
             'inputs': ['ctx(inst_idx/clock/order)', 'vehicles(+cc 不透明字段)',
                        'frozen plan', 'visible_orders', 'sampler/arm_seed/K'],
             'produces': ['input_snapshot（严格 schema）', 'input_decision_id'],
             'shared_by': ['old', 'c2', 'write_off'],
             'note': ('**同一个**快照对象送给三臂 ⇒ 批内逐元素一致；'
                      '结果类字段**不得**进入该快照（fail-closed）')},
            {'phase': 'P2_call_decide_fn',
             'call': 'accept, diag = decide_fn(order, ctx)',
             'variants': {
                 'old': '冻结旧版 decide_fn（从冻结代码身份启动，**不得**用新版关掉 C2 冒充）',
                 'c2': 'C2 版 decide_fn（diag 内含 exception_audit）',
                 'write_off': 'C2 版 + sidecar 写入关闭（同代码、只丢弃写入）'},
             'extract': {
                 'exception_audit': "diag.get('exception_audit')",
                 'candidate_plan': 'decide_fn 之外由调用方给出的候选计划',
                 'proposed_plan': "diag.get('plan')  ← 拟采纳",
                 'adopted_plan': '**只在真实提交回执可得时**才是实际采纳'},
             'note': ('⚠️ 只调用 `decide_fn()` **不产生提交** ⇒ **最多**只能比较'
                      '"**拟采纳计划**"；**不得**标 `COMMIT_OBSERVED`')},
            {'phase': 'P3_classify',
             'outputs': ['plan_variants(...)', 'classify_fallback(...)',
                         'IndependentCounters.add_exception_event/add_fallback'],
             'commit_evidence_required_for': 'accepted_plan / COMMIT_OBSERVED',
             'note': ('接受动作若无权威提交证据 ⇒ `CommitEvidenceError`（fail-closed）'),
             },
        ],
        'not_executed': ('本样例**不执行**真实决策：`demonstrate_with_stub()` 用桩 '
                         '`decide_fn` 走完三阶段，零环境依赖'),
    }


def demonstrate_with_stub():
    """用**桩** decide_fn 走完三阶段（**不**触碰真实 env，**不**产生真实提交）。"""
    import role_c_sidecar_recorder as sr
    calls = []

    def stub_decide_fn(order, ctx):
        calls.append((order, ctx.get('clock')))
        return False, {'boundary': 'B3_lambda_undefined',
                       'log_fields': {'action': 'reject'},
                       'plan': {'1': [2, 3]},
                       'exception_audit': {'exception_type': 'DecisionTimeout',
                                           'hard_deadline_exception': True,
                                           'budget_point': 'hi', 'event_id': 'exc-1'}}

    ctx = {'inst_idx': 0, 'clock': 3.5}
    snapshot = {'inst_idx': 0, 'clock': 3.5, 'order': 7,
                'vehicles_fingerprint': [[1, 'idle', 0, 0.0, None]],
                'plan': {'1': [2, 3]}, 'frozen': {'1': 0}, 'visible_orders': [2, 3, 4],
                'sampler': 'cond_hist', 'arm_seed': 7002, 'K': 10}
    did = sr.normalized_input_identity(protocol_identity='stub-proto',
                                       decision_point=0, input_snapshot=snapshot)
    frozen = sr.freeze_input(protocol_identity='stub-proto',
                             decision_point=0, input_snapshot=snapshot)
    accept, diag = stub_decide_fn(7, ctx)
    ea = (diag or {}).get('exception_audit')
    # 缺口 A：三个维度**分开**传；决策完成与权威提交不再混为一个布尔
    fb = sr.classify_fallback(diag=diag,
                              decision_outcome_observed=True,
                              authoritative_commit_observed=False,
                              exception_audit=ea)
    # 未提交 ⇒ 不能标 COMMIT_OBSERVED
    pv = sr.plan_variants(plan_before=snapshot['plan'],
                          candidate_plan={'1': [2, 3, 7]},
                          plan_after=snapshot['plan'],
                          commit_observed=False, action=None,
                          decision_outcome_observed=False)
    return {'phases_walked': list(WIRE_PHASES), 'calls': len(calls),
            'input_decision_id': did,
            'frozen_input_sha256': frozen.sha256,
            'frozen_input_unchanged': frozen.assert_unchanged(snapshot),
            'accept': bool(accept),
            'exception_audit_extracted': ea,
            'fallback': fb, 'commit_status': pv['commit_status'],
            'honesty': ('桩调用**没有**真实提交 ⇒ `COMMIT_OBSERVED` 不可用；'
                        '只能比较拟采纳计划'),
            'gap_a_note': ('硬截止异常**已知**但**未**提供决策后实际保留计划的'
                           '权威证据 ⇒ 分类结果为 %s（`fallback_applied=%r`）'
                           '——**不得**写成"发生了终态计划回退"'
                           % (fb['fallback_kind'], fb['fallback_applied']))}


def observability_proof():
    """**hi/lo 可观测性证明**：用**真实**函数，逐场景记录 wrapper 能直接看到什么。"""
    import role_c_opportunity_cost as oc

    class _Injector(object):
        """合约定注入器：`calls` **只在成功返回后**追加（可被 wrapper 直接读长度）。"""

        def __init__(self, raise_on=None, nan_on=None):
            self.calls = []
            self.raise_on = raise_on
            self.nan_on = nan_on

        def __call__(self, b):
            which = 'hi' if not self.calls else 'lo'
            if self.raise_on == which:
                from role_c_timing import DecisionTimeout
                raise DecisionTimeout('synthetic %s' % which)
            val = float('nan') if self.nan_on == which else 0.0
            self.calls.append((float(b), val))
            return val

    scenarios = {
        'normal_two_budget_points': _Injector(),
        'hi_timeout': _Injector(raise_on='hi'),
        'lo_timeout': _Injector(raise_on='lo'),
        'hi_nan_forward_invalid': _Injector(nan_on='hi'),
        'lo_unfinished_then_deadline': _Injector(raise_on='lo'),
    }
    out = {}
    for name, inj in scenarios.items():
        lam, diag = oc.shadow_price_budget(100.0, 1.0, inj)
        ea = (diag or {}).get('exception_audit') or {}
        out[name] = {
            'observable_from_wrapper': {
                'n_successful_budget_returns': len(inj.calls),
                'budget_point_of_exception': ea.get('budget_point'),
                'exception_type': ea.get('exception_type'),
                'hard_deadline_exception': ea.get('hard_deadline_exception'),
                'lambda_B_is_none': lam is None,
                'undefined_flag': (diag or {}).get('undefined'),
                'reason_text': (diag or {}).get('reason'),
            },
            'per_scenario_answers': {
                '两个预算点是否真正被调用': (
                    'DIRECT' if len(inj.calls) >= 2 else
                    ('INFERRED（只知道有 %d 次成功返回）' % len(inj.calls))),
                '返回时是否完成全部场景': 'NOT_VERIFIABLE',
                '异常发生在 hi 还是 lo': ('DIRECT' if ea.get('budget_point') in ('hi', 'lo')
                                          else 'NOT_VERIFIABLE'),
                '哪些只是代理推断': ('`FutureValueFn.calls` 长度只说明'
                                     '"**某个预算点成功返回**"，'
                                     '**不能**解释为"内部所有场景都已完成且无异常"'),
            },
        }
    tiers = {
        'budget_point_evaluation_entry': {
            'tier': 'INFERRED',
            'why': ('wrapper 只能看到"注入器被调用过几次"（`calls` 长度），'
                    '看不到"进入预算点评估"这一事件本身')},
        'budget_point_successful_return': {
            'tier': 'DIRECT',
            'why': '`calls` 长度在**成功返回后**增加 ⇒ 可直接观测'},
        'budget_point_error_exit': {
            'tier': 'DIRECT',
            'why': '异常被 `shadow_price_budget` 捕获并转成 `exception_audit`（含类型/预算点）'},
        'scenario_start': {
            'tier': 'NOT_VERIFIABLE',
            'why': '预算点内部的逐场景开始**不暴露**给 wrapper（除非改动 `role_c_forward.py`）'},
        'scenario_completion': {
            'tier': 'NOT_VERIFIABLE',
            'why': ('`last_diag` 只在 `forward_value` **正常返回后**更新；'
                    '中途异常时它保持旧值/`None` ⇒ 逐场景完成数不可得'),
            'caveat': '**不得**把 `calls` 长度解释为"全部场景已完成"'},
    }
    return {'tiers': tiers, 'scenarios': out,
            'conclusion': {
                'observable_directly': ['预算点成功返回', '预算点异常退出（类型 + 预算点）'],
                'only_inferable': ['预算点评估**进入**'],
                'not_verifiable_on_analysis_side': ['逐场景开始', '逐场景完成',
                                                    '返回时是否完成全部场景'],
                'wording_correction': (
                    '⚠️ 用户 2026-10-14 指定措辞更正：`hi_nan_forward_invalid` 中'
                    '**前瞻价值存在 NaN**，因此**最终 `lambda_B=None`**；'
                    '**不得**把最终 `lambda_B` 写成 NaN。'
                    '上一轮记录中的"`λ_B` 为 NaN"表述**已更正**。'),
                'pre_declared_outputs': ('无法直接观测的字段**必须**写 '
                                         '`NOT_VERIFIABLE`（如 hi/lo 的 '
                                         '`n_scenarios_entered/completed`），'
                                         '**不得**用推断值冒充事实，'
                                         '**不得**为补齐 schema 而改 `role_c_forward.py`'),
                'whitelist_eligibility': ('分析侧**确实无法**拿到逐场景完成情况 ⇒ '
                                          '符合"先证明分析侧无法满足，再申请函数级白名单"'
                                          '的前置条件；用户 2026-10-14 裁定：'
                                          '**只批准起草更窄的函数级观测白名单**，'
                                          '**不批准**立即修改 `role_c_forward.py`'),
            },
            'not_modified': ['role_c_forward.py', 'role_c_timing.py',
                             'role_c_harness.py', 'role_c_opportunity_cost.py']}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='接线样例 + hi/lo 可观测性证明（只读）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rep = {'label': 'P1A-C1C4-wiring-sample-and-hilo-observability-proof',
           'wire_phase_map': wire_phase_map(),
           'stub_walkthrough': demonstrate_with_stub(),
           'observability_proof': observability_proof(),
           'authorization': ('用户 2026-10-14 裁定 ①②：提交接线样例但**不执行真实决策**；'
                             '批准只读能力边界证明；**不**申请核心白名单')}
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({
        'phases': rep['wire_phase_map']['phase_order'],
        'stub_commit_status': rep['stub_walkthrough']['commit_status'],
        'tiers': {k: v['tier'] for k, v in rep['observability_proof']['tiers'].items()},
        'not_verifiable': rep['observability_proof']['conclusion']
                          ['not_verifiable_on_analysis_side']},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
