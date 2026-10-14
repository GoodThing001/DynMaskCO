#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**超时语义专项审计**的回归（只读审计 + 轻量独立反例）。

用户 2026-10-14 裁决的一号优先项：先在冻结源码上完成
`DecisionTimeout → shadow_price_budget → B3_lambda_undefined` 的轻量、独立反例审计。

本文件断言两类东西，且**都不重跑真实 Phase A**：

* **源码事实**（机械核验，逐条可失败）：7 条，见 `SOURCE_FACTS`；
* **合成反例**（用**真实** `shadow_price_budget` / `_run_with_limit`）：
  替身前瞻抛 `DecisionTimeout` ⇒ 是否被转成普通 B3、外层是否还看得见超时。

⚠️ 平台差异**必须**分开断言：`_run_with_limit` 只在 POSIX 下才捕获
`DecisionTimeout`；Windows 上它走协作式分支、异常直接穿出。
这**不是**"机制不成立"，而是该接口在非 POSIX 上没有硬中断语义。

⚠️ 归档签名检查是**一致性推断**，不是证据链闭环：本文件同时断言报告里
**必须**带着"需要针对性运行才能定案"的说明，防止把推断写成已确认。

运行：`python CC_Compare/tools/test_role_c_timeout_semantics.py`
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_timeout_semantics_audit as tsa                      # noqa: E402
import role_c_timing as rt                                        # noqa: E402

FAILS = []
ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'P1A_PhaseA_L1L2_B类_原件_2026-10-14.json')


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


POSIX = bool(hasattr(__import__('signal'), 'SIGALRM'))


# ------------------------------------------------------------------ 1. 源码事实
def test_source_facts():
    print('\n[1] 源码事实：逐条机械核验（按**代际预期**判，缺一条即失败）')
    a = tsa.audit_source()
    got = {f['fact']: f['matches_expected'] for f in a['facts']}
    gen = a['generation']
    print('  （被审代际：%s）' % gen)
    for key, _f, _d in tsa.SOURCE_FACTS:
        check('源码事实与代际预期一致：%s' % key, got.get(key) is True, got.get(key))
    check('全部事实与代际预期一致（all_hold）', a['all_hold'] is True, got)
    check('给出代际标记与说明', gen in ('pre-c2', 'c2')
          and bool(a.get('generation_note')), a.get('generation_note'))
    check('审计只读：给出被审文件的 SHA（不改字节）',
          all(v for v in a['files'].values()), a['files'])
    ev = {f['fact']: f['evidence'] for f in a['facts']}
    check('F1：DecisionTimeout 继承 RuntimeError ⇒ 属 Exception',
          ev['F1_decision_timeout_is_exception']['class_decl']
          == 'class DecisionTimeout(RuntimeError)', ev['F1_decision_timeout_is_exception'])
    check('F2：内部截止检查在场景循环内、且在订单循环**之前**',
          ev['F2_deadline_hit_not_independent']['check_is_before_order_loop'] is True,
          ev['F2_deadline_hit_not_independent'])
    # F6 逐代判定：pre-c2 应当"丢弃"；c2 应当"不再丢弃"且**传出去**
    holds6 = {f['fact']: f['holds'] for f in a['facts']}
    if gen == 'c2':
        check('C2 代际：B3 分支**不再**丢弃 `_d0`（F6 如实不成立）',
              holds6['F6_decide_candidate_discards_reason'] is False,
              holds6['F6_decide_candidate_discards_reason'])
        check('C2 代际：F6b 成立（审计信息确实传出到最终 diag）',
              holds6['F6b_c2_passes_exception_audit_through'] is True, holds6)
    else:
        check('pre-c2 代际：B3 分支确实丢弃 `_d0`',
              holds6['F6_decide_candidate_discards_reason'] is True, holds6)
    check('旧事实**未被判失败**（只是逐代读数不同；历史结论建立在旧字节上）',
          all('matches_expected' in f for f in a['facts'])
          and '旧字节' in a['generation_note'])
    return True


def test_decision_timeout_is_runtime_error():
    print('\n[2] `DecisionTimeout` 的类型关系（决定它会不会被宽 except 吃掉）')
    check('issubclass(DecisionTimeout, Exception)',
          issubclass(rt.DecisionTimeout, Exception))
    check('issubclass(DecisionTimeout, RuntimeError)',
          issubclass(rt.DecisionTimeout, RuntimeError))
    check('能被 `except Exception` 捕获（宽分支会吃掉它）',
          _caught_by_broad_except())
    return True


def _caught_by_broad_except():
    try:
        try:
            raise rt.DecisionTimeout('x')
        except Exception:                                         # noqa: BLE001
            return True
    except rt.DecisionTimeout:
        return False


# ------------------------------------------------------------------ 3. 反例
def test_counterexample_inside_forward():
    print('\n[3] 反例（前瞻内抛）：被吞成普通 B3，外层**看不见**超时')
    c = tsa.counterexample_swallow(raise_inside_forward=True, deadline_s=0.02)
    sp = c['shadow_price']
    check('shadow_price_budget **正常返回**（异常没穿出去）',
          sp['returned_normally'] is True, sp)
    check('转成 `undefined=True`', sp['undefined'] is True, sp)
    check('`lambda_B` 变成 None（⇒ B3 边界：强制拒单）', sp['lambda_B'] is None, sp)
    check('reason 保留异常文案（含「硬时限」）',
          '硬时限' in (sp.get('reason') or ''), sp.get('reason'))
    check('**类型名未被保留**（只拼了 `%s` 的字符串）',
          'DecisionTimeout' not in (sp.get('reason') or ''), sp.get('reason'))
    rw = c['run_with_limit']
    check('**外层 `_run_with_limit` 报 `timed_out=False`** ⇒ 超时对上层不可见',
          rw['timed_out'] is False, rw)
    check('外层看到的也是 undefined=True（就是一次普通 B3 拒单）',
          rw.get('undefined') is True, rw)
    check('两次前瞻调用都被吃掉（raised_count=2）',
          c['raised_count'] == 2, c['raised_count'])
    return True


def test_counterexample_outside_forward_control():
    print('\n[4] 对照（前瞻外抛）：异常**不**经过 shadow_price 的宽 except')
    c = tsa.counterexample_swallow(raise_inside_forward=False, deadline_s=0.02)
    sp = c['shadow_price']
    check('shadow_price 正常返回且未受影响（λ=0.0，undefined=False）',
          sp['returned_normally'] is True and sp['undefined'] is False
          and sp['lambda_B'] == 0.0, sp)
    rw = c['run_with_limit']
    if POSIX:
        check('POSIX：`_run_with_limit` 报 `timed_out=True`（吞没只发生在前瞻窗口内）',
              rw.get('timed_out') is True, rw)
    else:
        check('非 POSIX：异常直接穿出（该接口在 Windows 无硬中断语义）',
              rw.get('propagated_out') is True, rw)
        check('非 POSIX：报告显式标注该平台限制',
              '非 POSIX' in (rw.get('note') or ''), rw)
    check('报告标注了本平台是否具备硬中断',
          isinstance(c.get('platform_hard_interrupt_available'), bool), c)
    return True


def test_swallow_is_specific_to_forward_window():
    print('\n[5] 吞没**只**发生在前瞻窗口内（两种情形结论不同）')
    inside = tsa.counterexample_swallow(raise_inside_forward=True, deadline_s=0.02)
    outside = tsa.counterexample_swallow(raise_inside_forward=False, deadline_s=0.02)
    check('前瞻内：shadow_price 吞掉并正常返回',
          inside['shadow_price']['returned_normally'] is True
          and inside['shadow_price']['undefined'] is True)
    check('前瞻外：shadow_price 不受影响（λ 正常算出）',
          outside['shadow_price']['undefined'] is False
          and outside['shadow_price']['lambda_B'] == 0.0)
    check('⇒ 两者的差异**正是"异常是否落在 `future_value_fn` 调用的 try 内"**',
          inside['shadow_price']['undefined'] != outside['shadow_price']['undefined'])
    return True


# ---------------------------------------------------------------- 6. 可恢复性
def test_recoverability_channels():
    print('\n[6] 超时来源能否恢复：逐通道判定')
    ch = {c['channel']: c for c in tsa.recoverability_channels()}
    check('通道①`shadow_price_budget` 自身 diag：**可**恢复（但只靠文案）',
          ch['shadow_price_budget 自己的 diag']['recoverable'] is True)
    check('通道①的局限写明"类型名未保留"',
          '类型名没有被保留' in ch['shadow_price_budget 自己的 diag']['limit'],
          ch['shadow_price_budget 自己的 diag']['limit'])
    check('通道②决策记录：**不可**恢复（F6 丢弃 `_d0`）',
          ch['decide_candidate 的决策记录']['recoverable'] is False)
    check('通道③`_run_with_limit` 返回值：**不可**恢复（F5）',
          ch['_run_with_limit 的返回值']['recoverable'] is False)
    check('通道④`last_diag.deadline_hit`：**不可**恢复（F2+F3）',
          ch['forward_value 的 last_diag.deadline_hit']['recoverable'] is False)
    check('通道⑤进度原件：**部分**（能证"没跑完"，不能证"被 SIGALRM 打断"）',
          ch['进度原件（插桩的 leaf_enter/exit + 心跳）']['recoverable'] == 'partial')
    return True


def test_field_separation():
    print('\n[7] 字段分离：`evaluation_status` vs `timeout_cause`')
    fs = tsa.proposed_field_separation()
    check('`evaluation_status` 三态由**进度**证明',
          fs['evaluation_status']['values'] == ['not_entered', 'completed', 'timed_out']
          and '进度' in fs['evaluation_status']['provable_by'])
    check('`timeout_cause` 需要**异常路径证据**',
          'SIGALRM_decision_timeout' in fs['timeout_cause']['values']
          and '异常' in fs['timeout_cause']['provable_by'])
    check('未取得证据时必须写 `unknown`（不得默认成某一因）',
          fs['timeout_cause'].get('no_evidence_value') == 'unknown'
          and 'unknown' in fs['timeout_cause']['values']
          and '不得默认' in fs['timeout_cause']['meaning'],
          fs['timeout_cause'])
    check('honesty_rule 明确禁止把 `deadline_hit=false` 当"未触发截止"的证据',
          '不得' in fs['honesty_rule'] and 'deadline_hit=false' in fs['honesty_rule'])
    check('给出需要的**新证据**清单（含按类型识别、事件归档、CENSORED）',
          len(fs['required_new_evidence']) >= 3
          and any('CENSORED' in x for x in fs['required_new_evidence']))
    return True


# ------------------------------------------------------------ 8. 归档签名检查
def test_archive_signature_is_inference_only():
    print('\n[8] 归档签名：只作**一致性推断**，报告必须自带"需针对性运行"')
    if not os.path.isfile(ARCHIVE):
        check('归档原件存在', False, ARCHIVE)
        return False
    rep = tsa.archive_signature_check(ARCHIVE)
    check('8 个决策点全部纳入签名检查', rep['n_decision_points'] == 8,
          rep['n_decision_points'])
    c = rep['consistent_with_swallowed_sigalrm']
    check('全部 B3 边界', c['all_boundary_B3'] is True, c)
    check('全部 `deadline_hit=false`', c['all_deadline_hit_false'] is True, c)
    check('全部墙钟略超 10 s', c['all_wall_just_over_10s'] is True, c)
    check('每个决策点**只进入前瞻一次**（本应两次）',
          c['exactly_one_forward_entry_each'] is True, c)
    check('每个决策点只进入 1 个场景', c['exactly_one_scenario_entered_each'] is True, c)
    check('`P7c` 从未进入（0 个场景跑完）',
          c['no_scenario_completed_each'] is True, c)
    check('8/8 分区四条件闭合（截断窗口的份额是合法的）',
          c['all_partitions_closed_ok'] is True, c)
    check('明确标注是**推断**而不是闭环证据',
          '不是**证据链闭环' in rep['explicit_warning']
          or '不是' in rep['explicit_warning'], rep['explicit_warning'])
    check('明确列出"需要针对性运行"才能定案',
          len(rep['requires_targeted_run']) >= 2, rep['requires_targeted_run'])
    check('签名里带 P7b 在**截断窗口内**的份额（并标明是截断窗口）',
          all(x['P7b_share_of_truncated_window'] is not None
              for x in rep['per_decision']), rep['per_decision'][0])
    return True


def test_full_audit_report_is_honest():
    print('\n[9] 完整审计报告：不得出现越界结论（且必须自带禁止性说明）')
    rep = tsa.run_audit(b_report_path=ARCHIVE, deadline_s=0.02)
    txt = json.dumps(rep, ensure_ascii=False)
    sig = rep['archive_signature']
    # ⚠️ 不能只做子串扫描：**禁止性说明本身**就含"已确认被硬超时打断"这几个字。
    #    要检验的是"有没有把它们**当成结论**"，所以看的是**字段语义**。
    check('没有"吞没已确认"这类字段',
          'swallowed_confirmed' not in txt and 'confirmed_swallowed' not in txt)
    check('签名里没有把本批写成已确认的布尔位',
          not any(isinstance(v, bool) and k.endswith('confirmed')
                  for k, v in sig['consistent_with_swallowed_sigalrm'].items()),
          sorted(sig['consistent_with_swallowed_sigalrm']))
    check('归档签名只声明"一致性"（键名即 `consistent_with_…`）',
          list(sig['consistent_with_swallowed_sigalrm'])
          and all(k.startswith(('all_', 'exactly_', 'no_'))
                  for k in sig['consistent_with_swallowed_sigalrm']),
          sorted(sig['consistent_with_swallowed_sigalrm']))
    check('明确写出"不得据此写成已确认"（禁止性说明存在）',
          '不得' in sig['explicit_warning']
          and '已确认被硬超时打断' in sig['explicit_warning'],
          sig['explicit_warning'])
    check('报告带 `requires_targeted_run`', 'requires_targeted_run' in txt)
    check('报告带 `proposed_field_separation`', 'timeout_cause' in txt)
    check('源码事实全过（审计成立的前提）',
          rep['source_audit']['all_hold'] is True)
    return True


def main():
    print('=' * 78)
    print('超时语义专项审计回归（只读 + 轻量反例；不重跑真实 Phase A）')
    print('平台：%s（SIGALRM 可用=%s）' % (sys.platform, POSIX))
    print('=' * 78)
    test_source_facts()
    test_decision_timeout_is_runtime_error()
    test_counterexample_inside_forward()
    test_counterexample_outside_forward_control()
    test_swallow_is_specific_to_forward_window()
    test_recoverability_channels()
    test_field_separation()
    test_archive_signature_is_inference_only()
    test_full_audit_report_is_honest()
    print('\n' + '=' * 78)
    if FAILS:
        print('TIMEOUT_AUDIT_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('TIMEOUT_AUDIT_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
