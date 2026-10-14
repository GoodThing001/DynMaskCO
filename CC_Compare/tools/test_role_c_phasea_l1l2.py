#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1/L2 驱动器的**判定逻辑**回归（本机可跑；不含真实管线与硬件计时）。

为什么必须单独测判定：`aggregate()` 决定"瓶颈到底是谁 / 还是不确定"，
而 §6.2/§6.3 的坑很密（并列、稳定性绑定同一叶子、`P9` 占大头、
有截断就不许判定）。这些**全都是纯逻辑**，可以在本机用合成报告逐条打靶。

覆盖：
  * `i* = argmax_i median_d share_i(d)`；并列取**最小叶子索引**；叶子与 `P9` 并列时取叶子；
  * `median < 0.5` ⇒ `below_threshold` + 分叉①措辞；
  * 稳定性 `5/8 < 6/8` ⇒ 不得判主导；
  * `P9_measured` 最大且 `median ≥ 0.5` ⇒ `inconclusive_attribution_incomplete`；
  * **任一截断 ⇒ 不许判瓶颈**（且**不得**退化成 `below_threshold`）；
  * 记账失败 ⇒ `inconclusive_accounting_error`；
  * 结构性乘数组（`P7a/P7b/P7d`）⇒ `dominant_in_structural_group_slack_undetermined`
    + 分支①原文，且**不声称**其内部计算不可优化；
  * 总上限触及 ⇒ 不确定；
  * A 类截断记录必须带**截断位置**（不得只写超时）；
  * B 类三态 `not_entered` / `completed` / `timed_out` **可区分**；
  * `_digest_of` 与 `role_c_timing.DecisionWorkFn.digest()` **同算法**（digest 门才可比）。

运行：`python CC_Compare/tools/test_role_c_phasea_l1l2.py`
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_instrument as pin                            # noqa: E402
import role_c_phasea_l1l2 as drv                                  # noqa: E402
import role_c_phasea_partition as pp                              # noqa: E402
import role_c_t_series as ts                                      # noqa: E402
import role_c_timing as rt                                        # noqa: E402

ts.ensure_project_paths()

FAILS = []
LABELS = list(pp.LEAF_ORDER) + [pp.UNATTRIBUTED]


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def _shares(**kw):
    s = {l: 0.0 for l in LABELS}
    s.update(kw)
    tot = sum(s.values())
    if tot and abs(tot - 1.0) > 1e-12:
        # 把余量给 P9，保证是合法分区
        s[pp.UNATTRIBUTED] += 1.0 - tot
    return s


def _a(per, **over):
    base = {'class': 'A', 'internal_deadline': drv.DISABLED,
            'per_decision_cap_s': 600.0, 'total_cap_s': 12000.0,
            'enforcement': 'subprocess_sigkill_by_parent',
            'hard_kill_enforced': True, 'total_exceeded': False,
            'n_decision_points': len(per),
            'n_completed': sum(1 for r in per if r.get('status') == 'completed'),
            'n_censored': sum(1 for r in per if r.get('status') == 'censored'),
            'per_decision': per}
    base.update(over)
    return base


def _b(per=None):
    per = per or []
    return {'class': 'B', 'internal_deadline': 10.0, 'per_decision_cap_s': 10.0,
            'enforcement': 'hard_sigalrm', 'n_decision_points': len(per),
            'n_timed_out': 0, 'n_not_entered': 0, 'n_completed': len(per),
            'per_decision': per}


def _ok_dp(i, **sharekw):
    return {'decision_point': i, 'status': 'completed', 'verdict': 'ok',
            'failed_conditions': [], 't_decision': 10.0, 'tolerance_s': 0.1,
            'p9_measured': 0.0, 'p9_residual': 0.0, 'shares': _shares(**sharekw),
            'stages_entered': ['P1'], 'l1': {}, 'wall_s': 10.0}


# ------------------------------------------------------------------ 1. i* 与并列
def test_i_star_and_tie_break():
    print('\n[1] `i* = argmax median share`；并列取**最小叶子索引**')
    shares = [_shares(P7d=0.7), _shares(P7a=0.7), _shares(P7d=0.7), _shares(P7a=0.7),
              _shares(P7d=0.7), _shares(P7a=0.7), _shares(P7d=0.7), _shares(P7a=0.7)]
    pick, med, tied = drv._i_star(shares)
    check('并列 P7a/P7d ⇒ 取 P7a（索引更小）', pick == 'P7a', (pick, med))
    check('并列集合被如实报出', set(tied) == {'P7a', 'P7d'}, tied)
    # 叶子与 P9 并列 ⇒ 取叶子（叶子序列优先）
    s2 = [_shares(**{pp.UNATTRIBUTED: 0.6}), _shares(**{pp.UNATTRIBUTED: 0.6})]
    pick2, _m2, tied2 = drv._i_star(s2)
    check('只有 P9 最大 ⇒ 取 P9（其份额 ≥0.5 时走归因不完整档）',
          pick2 == pp.UNATTRIBUTED, (pick2, tied2))
    s3 = [_shares(P8=0.5, **{pp.UNATTRIBUTED: 0.5}),
          _shares(P8=0.5, **{pp.UNATTRIBUTED: 0.5})]
    pick3, _m3, tied3 = drv._i_star(s3)
    check('叶子与 P9 **同时**并列 ⇒ 取叶子 P8（叶子序列优先）', pick3 == 'P8',
          (pick3, tied3))
    check('中位数是逐决策点取中位数（不是总和）',
          drv._i_star([_shares(P5=0.9), _shares(P5=0.1)])[1]['P5'] == 0.5,
          drv._i_star([_shares(P5=0.9), _shares(P5=0.1)])[1]['P5'])
    return True


# --------------------------------------------------------------- 2. 低于门槛
def test_below_threshold():
    print('\n[2] `median < 0.5` ⇒ below_threshold + 分叉①原文')
    per = [_ok_dp(i, P5=0.2, P6=0.2, P7d=0.2) for i in range(8)]
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('结论 = below_threshold', out['conclusion'] == 'below_threshold',
          out['conclusion'])
    check('没有瓶颈叶子', out['bottleneck'] is None)
    check('给出分叉①原文', '不据此推断所有可能的计算优化均不可行'
          in out.get('fork_1_wording', ''), out.get('fork_1_wording'))
    check('判据里带中位数与稳定性',
          out['criterion']['i_star'] and out['criterion']['stability_hits'] == 8,
          out['criterion'])
    return True


def test_stability_binds_to_same_leaf():
    print('\n[3] 稳定性**绑定同一叶子**：中位数最大但只 5/8 第一 ⇒ 不判主导')
    per = []
    for i in range(8):
        # P5 与 P6 交替当第一，各占 4/8；两者中位数相同（都 0.5）⇒ 并列 ⇒ P5（索引小）。
        # 份额**必须自己配平成 1.0**：否则余量会落到 P9_measured，把中位数最大的位置抢走
        # （第一版夹具就踩了这个坑，于是它测的不是"稳定性"而是"P9 最大"）。
        per.append(_ok_dp(i, P5=0.55 if i % 2 == 0 else 0.45,
                          P6=0.45 if i % 2 == 0 else 0.55))
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('并列取 P5', out['criterion']['i_star'] == 'P5', out['criterion'])
    check('两个叶子中位数都 = 0.5（配平后没有余量落到 P9）',
          abs(out['criterion']['median_share']['P5'] - 0.5) < 1e-12
          and abs(out['criterion']['median_share']['P6'] - 0.5) < 1e-12,
          out['criterion']['median_share'])
    check('稳定性 4/8 < 6/8 ⇒ 不判主导',
          out['conclusion'] == 'below_threshold' and out['bottleneck'] is None,
          (out['conclusion'], out['criterion']['stability_hits']))
    check('理由指向稳定性（而不是份额不足）',
          any('稳定性' in r for r in out['conclusion_reasons']),
          out['conclusion_reasons'])
    return True


# ------------------------------------------------------------- 4. P9 占大头
def test_p9_dominant_is_attribution_incomplete():
    print('\n[4] `P9_measured` 最大且 median ≥ 0.5 ⇒ inconclusive_attribution_incomplete')
    per = [_ok_dp(i, **{pp.UNATTRIBUTED: 0.8, 'P5': 0.1}) for i in range(8)]
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('结论 = inconclusive_attribution_incomplete',
          out['conclusion'] == 'inconclusive_attribution_incomplete',
          out['conclusion'])
    check('**不**写成"无单一主导瓶颈"（不得退化成 below_threshold）',
          out['conclusion'] != 'below_threshold' and out['bottleneck'] is None)
    check('理由写明"漏测吞掉大头 ≠ 没有瓶颈"',
          any('不等于' in r and '瓶颈' in r for r in out['conclusion_reasons']),
          out['conclusion_reasons'])
    return True


# ------------------------------------------------------------------ 5. 截断
def test_any_censored_blocks_bottleneck_verdict():
    print('\n[5] **任一截断 ⇒ 不许判瓶颈**（且不得退化成 below_threshold）')
    per = [_ok_dp(i, P5=0.8) for i in range(7)]
    per.append({'decision_point': 7, 'status': 'censored',
                'inconclusive': 'inconclusive_censored',
                'leaf_at_truncation': 'P7a',
                'scenario_index_at_truncation': 4,
                'n_scenarios_completed_at_truncation': 4,
                'elapsed_s_at_truncation': 600.0, 'wall_s': 600.1})
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('结论 = inconclusive_censored', out['conclusion'] == 'inconclusive_censored',
          out['conclusion'])
    check('**不得**退化成 below_threshold', out['conclusion'] != 'below_threshold')
    check('无瓶颈结论', out['bottleneck'] is None)
    check('理由引用 §5.1.2', any('5.1.2' in r for r in out['conclusion_reasons']),
          out['conclusion_reasons'])
    # 截断位置必须保留在逐决策点记录里
    dp7 = [d for d in out['decision_points'] if d['decision_point'] == 7][0]
    check('截断记录带**位置**（不得只写超时）',
          dp7['leaf_at_truncation'] == 'P7a'
          and dp7['scenario_index_at_truncation'] == 4
          and dp7['n_scenarios_completed_at_truncation'] == 4, dp7)
    return True


def test_accounting_error_blocks_verdict():
    print('\n[6] 记账失败 ⇒ inconclusive_accounting_error（不许判瓶颈）')
    per = [_ok_dp(i, P8=0.9) for i in range(7)]
    bad = _ok_dp(7, P8=0.9)
    bad['verdict'] = 'inconclusive_accounting_error'
    bad['failed_conditions'] = ['2_unattributed_vs_offline']
    per.append(bad)
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('结论 = inconclusive_accounting_error',
          out['conclusion'] == 'inconclusive_accounting_error', out['conclusion'])
    check('无瓶颈结论', out['bottleneck'] is None)
    return True


def test_total_exceeded_blocks_verdict():
    print('\n[7] 触及 12 000 s 总上限 ⇒ 不确定（不许判瓶颈）')
    per = [_ok_dp(i, P8=0.9) for i in range(3)]
    out = drv.aggregate(_a(per, total_exceeded=True), _b(), digest_registered='D')
    check('结论不确定（非 below_threshold）',
          out['conclusion'] in ('inconclusive_censored',
                                'inconclusive_accounting_error')
          and out['conclusion'] != 'below_threshold', out['conclusion'])
    check('理由提到总上限', any('12 000' in r or '上限' in r
                            for r in out['conclusion_reasons']),
          out['conclusion_reasons'])
    return True


# --------------------------------------------------------- 8. 结构性乘数组
def test_structural_group_is_slack_undetermined():
    print('\n[8] 结构性乘数组（P7a/P7b/P7d）⇒ slack_undetermined + 分支①原文')
    for lab in ('P7a', 'P7b', 'P7d'):
        per = [_ok_dp(i, **{lab: 0.75}) for i in range(8)]
        out = drv.aggregate(_a(per), _b(), digest_registered='D')
        check('%s ⇒ dominant_in_structural_group_slack_undetermined' % lab,
              out['conclusion'] == 'dominant_in_structural_group_slack_undetermined',
              out['conclusion'])
        check('%s：给出分支①原文' % lab,
              '不据此推断所有可能的计算优化均不可行'
              in out.get('ratchet_wording', ''), out.get('ratchet_wording'))
        check('%s：**不声称**其内部计算不可优化' % lab,
              '实现级优化余量尚未确定' in ''.join(out['conclusion_reasons']),
              out['conclusion_reasons'])
    per = [_ok_dp(i, P6=0.75) for i in range(8)]
    out = drv.aggregate(_a(per), _b(), digest_registered='D')
    check('非结构性叶子（P6）⇒ dominant_bottleneck_identified',
          out['conclusion'] == 'dominant_bottleneck_identified', out['conclusion'])
    check('非结构性叶子：瓶颈被点名', out['bottleneck'] == 'P6', out['bottleneck'])
    check('非结构性叶子：**不**给分支①收束措辞（仍可进 Phase B 登记）',
          'ratchet_wording' not in out)
    return True


# --------------------------------------------- 9. A 类截断记录 / B 类三态
def _fake_bundle(n=2):
    return {'n_contexts': n, 'contexts': [{'order': i, 'clock': float(i)}
                                          for i in range(n)]}


def test_a_class_censored_record_has_position():
    print('\n[9] A 类硬杀记录：由进度原件恢复**截断位置**（§5.1.2）')
    d = tempfile.mkdtemp()
    try:
        ppath = os.path.join(d, 'p.jsonl')
        with io.open(ppath, 'w', encoding='utf-8', newline='\n') as f:
            for r in ({'kind': 'decision_start'},
                      {'kind': 'heartbeat', 'ordered': 5, 'complete_in_forward': 4,
                       'best_insert_calls': 321},
                      {'kind': 'leaf_enter', 'leaf': 'P7d'},
                      {'kind': 'leaf_enter', 'leaf': 'P7b'}):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        rec = drv._collect_a(3, os.path.join(d, 'no_result.json'), ppath,
                             killed=True, wall_s=600.2, cap_s=600.0)
        check('状态 = censored', rec['status'] == 'censored', rec['status'])
        check('标注 inconclusive_censored',
              rec['inconclusive'] == 'inconclusive_censored')
        check('**给出截断叶子**（最内层活动叶子）',
              rec['leaf_at_truncation'] == 'P7b', rec['leaf_at_truncation'])
        check('给出活动栈尾部', rec['active_stack_tail'] == ['P7d', 'P7b'], rec)
        check('给出场景位置与完成数',
              rec['scenario_index_at_truncation'] == 5
              and rec['n_scenarios_completed_at_truncation'] == 4, rec)
        check('给出截断时刻与 L1 计数',
              rec['elapsed_s_at_truncation'] == 600.0
              and rec['best_insert_calls_at_truncation'] == 321, rec)
        check('写明是**外层**硬杀、不是内部截止',
              '内部截止' in rec['truncation_note'], rec['truncation_note'])
        check('截断记录**不含**份额（不得在半截数据上算份额）',
              'shares' not in rec and 'verdict' not in rec, sorted(rec))
    finally:
        _rmtree(d)
    return True


def test_b_class_three_states_distinguishable():
    print('\n[10] B 类三态必须**可区分**（不得一律填 deadline_hit=false）')
    b = _fake_bundle(3)
    state = {'n': 0}

    def decide_fn(order, ctx):
        state['n'] += 1
        k = int(order)
        if k == 0:                       # 不进前瞻
            return False, {'boundary': 'insertion_infeasible', 'e_o': 1.0,
                           'lambda_B': None, 'certify_ok': False,
                           'log_fields': {'action': 'reject'}, 'forward': None}
        if k == 1:                       # 进前瞻且完成
            with pin.active_session().timer.leaf('P7d'):
                pass
            return True, {'boundary': None, 'e_o': 2.0, 'lambda_B': 0.5,
                          'certify_ok': True,
                          'log_fields': {'action': 'accept'},
                          'forward': {'deadline_hit': False, 'n_scenarios': 10}}
        with pin.active_session().timer.leaf('P7d'):      # 进前瞻且触截止
            pass
        return False, {'boundary': 'B3_lambda_undefined', 'e_o': 3.0,
                       'lambda_B': None, 'certify_ok': False,
                       'log_fields': {'action': 'reject'},
                       'forward': {'deadline_hit': True, 'n_scenarios': 10}}

    rep = drv.run_stage_b(b, decide_fn=decide_fn, per_decision_cap_s=10.0)
    st = [r['evaluation_status'] for r in rep['per_decision']]
    check('三态齐备且互斥', st == ['not_entered', 'completed', 'timed_out'], st)
    check('not_entered：forward_entered=false 且 deadline_hit=false',
          rep['per_decision'][0]['forward_entered'] is False
          and rep['per_decision'][0]['deadline_hit'] is False,
          rep['per_decision'][0])
    check('timed_out：forward_entered=true 且 deadline_hit=true',
          rep['per_decision'][2]['forward_entered'] is True
          and rep['per_decision'][2]['deadline_hit'] is True,
          rep['per_decision'][2])
    check('completed：forward_entered=true 且 deadline_hit=false',
          rep['per_decision'][1]['forward_entered'] is True
          and rep['per_decision'][1]['deadline_hit'] is False,
          rep['per_decision'][1])
    check('**三态没有塌成同一个 deadline_hit=false**',
          len({(r['forward_entered'], r['deadline_hit'])
               for r in rep['per_decision']}) == 3, rep['per_decision'])
    check('逐决策点带 action / boundary / lambda_B / e_o / stages_entered',
          all(k in rep['per_decision'][0] for k in
              ('action', 'boundary', 'lambda_B', 'e_o', 'stages_entered')),
          sorted(rep['per_decision'][0]))
    check('B 类明确标注 enforcement=hard_sigalrm',
          rep['enforcement'] == 'hard_sigalrm', rep['enforcement'])
    check('B 类报告与 A 类**分开**（各自 class 字段）',
          rep['class'] == 'B' and rep['internal_deadline'] == 10.0, rep['class'])
    return True


# ------------------------------------------------------- 11. digest 同算法
def test_digest_matches_timing_module():
    print('\n[11] `_digest_of` 与 `role_c_timing.DecisionWorkFn.digest()` **同算法**')
    contexts = [{'inst_idx': 0, 'clock': 3.5, 'order': 7,
                 'plan': {1: [2, 3], 2: []},
                 'vehicles': [{'vehicle_id': 1, 'status': 'idle', 'current_node': 0,
                               'current_load': 0.0, 'committed_next': None,
                               'committed_finish': 0.0}]},
                {'inst_idx': 0, 'clock': 9.0, 'order': 11,
                 'plan': {1: [], 2: [4]},
                 'vehicles': [{'vehicle_id': 2, 'status': 'idle', 'current_node': 0,
                               'current_load': 0.0, 'committed_next': None,
                               'committed_finish': 0.0}]}]
    mine = drv._digest_of(contexts)
    theirs = rt.DecisionWorkFn(lambda o, c: (False, {}), contexts).digest()
    check('digest 逐字符相同', mine == theirs, (mine, theirs))
    check('digest 是 64 位十六进制', len(mine) == 64 and all(
        c in '0123456789abcdef' for c in mine), mine)
    return True


def _rmtree(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def main():
    print('=' * 78)
    print('L1/L2 驱动器判定逻辑回归（本机；不含真实管线/硬件计时）')
    print('=' * 78)
    test_i_star_and_tie_break()
    test_below_threshold()
    test_stability_binds_to_same_leaf()
    test_p9_dominant_is_attribution_incomplete()
    test_any_censored_blocks_bottleneck_verdict()
    test_accounting_error_blocks_verdict()
    test_total_exceeded_blocks_verdict()
    test_structural_group_is_slack_undetermined()
    test_a_class_censored_record_has_position()
    test_b_class_three_states_distinguishable()
    test_digest_matches_timing_module()
    print('\n' + '=' * 78)
    if FAILS:
        print('DRIVER_TEST_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('DRIVER_TEST_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
