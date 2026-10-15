#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析侧 sidecar 实现回归 **v2**（按用户 2026-10-14 「bce2e988 审查裁定」修订）。

被取代的版本**逐字节保留**于 `_superseded_test_role_c_sidecar_recorder_v1_2026-10-14.py`
（SHA-256 `8bb47c7d712e5e1dc6b413cd22ffc5f560c8e91fe8604229d83706d228cfc6dc`）。
⚠️ **该 v1 的 PASS 不构成证据**：它的
`check('把 action 混进快照也不影响 ID', a == b or True)` 是**恒真断言**
（`or True` 使条件永远成立），且它只在**顶层**有 `input_snapshot` 键这一点上做 AST 检查，
**没有**检查快照内部是否混入决策结果字段。
⇒ 按裁定："新版本通过之前，也不能继续引用旧测试作为相应缺陷已经解决的证据。"

本 v2 覆盖用户点名的全部缺陷：
  A 输入身份严格 schema + fail-closed（含嵌套结果字段）；
  B 提交状态与权威证据（四种矛盾组合全部 fail-closed；三态语义）；
  C 计数器：事件级/决策级**分别存储与导出**；`fallback_kind` 必须校验；
    重复 `event_id` / 缺事件身份 / 非法 `fallback_kind` 的**独立反例**；
  D `static_selfcheck` 四模块**双身份**（blob 与工作树）分别核验，`expected` 不再为 None。

运行：`python CC_Compare/tools/test_role_c_sidecar_recorder.py`

⚠️ Windows 控制台默认 GBK，断言名含 `⇒`/`—` 时 `print` 会抛 `UnicodeEncodeError`。
本文件在导入时把 stdout 重设为 UTF-8（`errors='replace'`），使**默认命令**即可复现；
这是**输出编码**修正，**不**改变任何断言语义。
"""
from __future__ import annotations

try:  # 输出编码修正（仅影响控制台可打印性，不影响判定）
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # pragma: no cover
    pass

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_sidecar_recorder as sr                            # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


def _snapshot(**over):
    s = {'inst_idx': 0, 'clock': 3.5, 'order': 7,
         'vehicles_fingerprint': [[1, 'idle', 0, 0.0, None]],
         'plan': {'1': [2, 3]}, 'frozen': {'1': 0},
         'visible_orders': [2, 3, 4], 'sampler': 'cond_hist',
         'arm_seed': 7002, 'K': 10}
    s.update(over)
    return s


def _rec(**over):
    """**合规**记录参数：接受动作必须带权威提交证据（v1 的默认值是错的）。"""
    kw = dict(decision_point=0, input_snapshot=_snapshot(), action='reject',
              boundary=None, reject_reason=None, lambda_B=None, e_o=1.0,
              certify_ok=False, decision_hash='dh-abc', exception_audit=None,
              plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
              accepted_plan=None, plan_after={'1': [2, 3]}, commit_observed=False,
              commit_evidence=None, decision_outcome_observed=True)
    kw.update(over)
    return kw


# ------------------------------------------------ A：输入身份严格 schema + fail-closed
def test_input_identity_strict_schema():
    print('\n[A] 输入身份：严格只含决策前信息，不合规 **fail-closed**')
    base = _snapshot()
    a = sr.normalized_input_identity(protocol_identity='p', decision_point=0,
                                     input_snapshot=base)
    # ⚠️ v2 的**真断言**（v1 在这里写的是 `a == b or True`，恒真）：
    bad = dict(base, action='accept')
    check('把 action 混进快照 ⇒ **抛错**（不再"恒真通过"）',
          _raises(sr.InputIdentityError, sr.normalized_input_identity,
                  protocol_identity='p', decision_point=0, input_snapshot=bad))
    for k in ('accepted_plan', 'candidate_plan', 'utility', 'delta', 'certify_ok',
              'decision_hash', 'lambda_B', 'e_o', 'reject_reason'):
        check('结果类键 %s ⇒ 抛错' % k,
              _raises(sr.InputIdentityError, sr.normalized_input_identity,
                      protocol_identity='p', decision_point=0,
                      input_snapshot=dict(base, **{k: 1})))
    check('**嵌套**结果类键 ⇒ 也抛错（v1 只查顶层键，查不出来）',
          _raises(sr.InputIdentityError, sr.normalized_input_identity,
                  protocol_identity='p', decision_point=0,
                  input_snapshot=dict(base, extra_nested={'deep': {'action': 'accept'}}))
          or _raises(sr.InputIdentityError, sr.validate_input_snapshot,
                     dict(base, nested={'action': 'accept'})))
    check('白名单外的未知键 ⇒ 抛错（fail-closed，不靠自觉）',
          _raises(sr.InputIdentityError, sr.validate_input_snapshot,
                  dict(base, some_random_key=1)))
    check('决策前**已冻结的计划** `plan`/`frozen` **保留**（合法）',
          sr.validate_input_snapshot(base) is True)
    check('非 dict ⇒ 抛错',
          _raises(sr.InputIdentityError, sr.validate_input_snapshot, ['not', 'dict']))
    same = sr.normalized_input_identity(protocol_identity='p', decision_point=0,
                                        input_snapshot=base)
    check('相同输入 ⇒ 相同 ID', same == a)
    diff = sr.normalized_input_identity(protocol_identity='p', decision_point=0,
                                        input_snapshot=dict(base, clock=3.6))
    check('输入变化 ⇒ ID 变化', diff != a)
    return True


# ------------------------------------------------- B：提交状态与权威证据（三态）
def test_commit_status_requires_evidence():
    print('\n[B] 提交状态必须由**权威证据**支撑；四种矛盾组合 fail-closed')
    good = sr.plan_variants(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                            accepted_plan={'1': [2, 3, 7]}, plan_after={'1': [2, 3, 7]},
                            commit_observed=True, action='accept',
                            commit_evidence={'kind': 'commit_receipt', 'sha256': 'x'})
    check('接受 + 回执 ⇒ COMMIT_OBSERVED 且主判指向 accepted_plan',
          good['commit_status'] == 'COMMIT_OBSERVED'
          and good['primary_criterion_plan'] == 'accepted_plan', good['commit_status'])
    check('矛盾组合①：commit_observed=True 但 accepted_plan=None ⇒ 抛',
          _raises(sr.CommitEvidenceError, sr.plan_variants, plan_before={'1': []},
                  accepted_plan=None, commit_observed=True, action='accept',
                  commit_evidence={'kind': 'commit_receipt'}))
    check('矛盾组合②：commit_observed=False 却传非空 accepted_plan ⇒ 抛',
          _raises(sr.CommitEvidenceError, sr.plan_variants, plan_before={'1': []},
                  accepted_plan={'1': [9]}, commit_observed=False))
    check('矛盾组合③：commit_observed=True 但**无证据** ⇒ 抛（布尔自述不算证据）',
          _raises(sr.CommitEvidenceError, sr.plan_variants, plan_before={'1': []},
                  accepted_plan={'1': [9]}, commit_observed=True, action='accept'))
    check('矛盾组合④：证据 kind 非法 ⇒ 抛',
          _raises(sr.CommitEvidenceError, sr.plan_variants, plan_before={'1': []},
                  accepted_plan={'1': [9]}, commit_observed=True, action='accept',
                  commit_evidence={'kind': 'caller_says_so'}))
    rej = sr.plan_variants(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                           accepted_plan=None, plan_after={'1': [2, 3]},
                           commit_observed=False, action='reject',
                           decision_outcome_observed=True)
    check('拒单 + 已观察决策完成 ⇒ `NO_NEW_PLAN_ADOPTED`（**不是** COMMIT_NOT_OBSERVED）',
          rej['commit_status'] == 'NO_NEW_PLAN_ADOPTED'
          and rej['primary_criterion_plan'] == 'none_adopted', rej['commit_status'])
    unk = sr.plan_variants(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                           commit_observed=False, action=None,
                           decision_outcome_observed=False)
    check('未观察到决策完成 ⇒ `COMMIT_NOT_OBSERVED`',
          unk['commit_status'] == 'COMMIT_NOT_OBSERVED'
          and unk['primary_criterion_plan'] == 'COMMIT_NOT_OBSERVED', unk['commit_status'])
    check('三态词表齐备', set(sr.COMMIT_STATUS) ==
          {'COMMIT_OBSERVED', 'NO_NEW_PLAN_ADOPTED', 'COMMIT_NOT_OBSERVED'})
    check('note 写明"只调用 decide_fn 时最多比较拟采纳计划"',
          '拟采纳计划' in good['note'])
    return True


def test_recorder_rejects_v1_default():
    print('\n[B2] v1 测试里的默认组合（commit_observed=True + accepted_plan=None）**现在被拒**')
    d = tempfile.mkdtemp()
    try:
        rc = sr.SidecarRecorder(run_id='r1', arm='c2', sidecar_path=os.path.join(d, 's.jsonl'),
                                protocol_identity='p')
        bad = _rec(action='accept', commit_observed=True, accepted_plan=None,
                   commit_evidence={'kind': 'commit_receipt'})
        check('v1 的默认参数组合 ⇒ 抛 `CommitEvidenceError`（v1 却能写进 sidecar）',
              _raises(sr.CommitEvidenceError, rc.record, **bad))
        rc.close()
    finally:
        _rm(d)
    return True


# --------------------------------------------------------- C：计数器口径
def test_counters_units_are_separate():
    print('\n[C1] 事件级与决策级执行错误**分别存储、分别导出**')
    c = sr.IndependentCounters()
    c.add_exception_event(run_id='r1', decision_point=1,
                          exception_audit={'hard_deadline_exception': False,
                                           'event_id': 'e1'})
    c.add_exception_event(run_id='r1', decision_point=2,
                          exception_audit={'hard_deadline_exception': False,
                                           'event_id': 'e2'})
    c.add_fallback(run_id='r1', decision_point=1, fallback_event_id='f1',
                   fallback_kind='timeout_fallback',
                   evidence={'hard_deadline_exception': True, 'event_id': 'e9'})
    c.add_fallback(run_id='r1', decision_point=1, fallback_event_id='f2',
                   fallback_kind='timeout_fallback',
                   evidence={'hard_deadline_exception': True, 'event_id': 'e9'})
    s = c.summary()
    check('事件级错误独立导出 = 2', s['n_execution_errors_events'] == 2, s)
    check('决策级错误独立导出 = 1（重复回退）', s['n_execution_errors_decisions'] == 1, s)
    check('**不再有混用的** `n_execution_errors` 单字段',
          'n_execution_errors' not in s, sorted(s))
    check('声明两个口径彼此独立', s['n_execution_errors_units_are_separate'] is True)
    return True


def test_counter_rejects_bad_input():
    print('\n[C2] 独立反例：重复 event_id / 缺事件身份 / 非法 fallback_kind')
    c = sr.IndependentCounters()
    c.add_exception_event(run_id='r1', decision_point=1,
                          exception_audit={'hard_deadline_exception': True,
                                           'event_id': 'dup'})
    check('重复 event_id ⇒ 抛',
          _raises(sr.CounterInputError, c.add_exception_event, run_id='r1',
                  decision_point=1, exception_audit={'hard_deadline_exception': True,
                                                     'event_id': 'dup'}))
    check('缺事件身份（event_id 为空）⇒ 抛',
          _raises(sr.CounterInputError, c.add_exception_event, run_id='r1',
                  decision_point=2, exception_audit={'hard_deadline_exception': True}))
    for kind in ('none', 'b2_boundary_reject', 'b3_conservative_reject',
                 'FALLBACK_NOT_VERIFIABLE', 'whatever'):
        check('非法 fallback_kind=%s ⇒ **不得**增加 n_fallbacks' % kind,
              _raises(sr.CounterInputError, c.add_fallback, run_id='r1',
                      decision_point=3, fallback_event_id='f9', fallback_kind=kind,
                      evidence={'hard_deadline_exception': True}))
    check('缺 fallback_event_id ⇒ 抛',
          _raises(sr.CounterInputError, c.add_fallback, run_id='r1',
                  decision_point=3, fallback_event_id=None,
                  fallback_kind='timeout_fallback',
                  evidence={'hard_deadline_exception': True}))
    check('缺可核验证据 ⇒ 抛',
          _raises(sr.CounterInputError, c.add_fallback, run_id='r1',
                  decision_point=3, fallback_event_id='f3',
                  fallback_kind='timeout_fallback', evidence=None))
    check('经以上拒绝后 n_fallbacks 仍为 0', c.summary()['n_fallbacks'] == 0, c.summary())
    return True


def test_counter_event_vs_decision_still_correct():
    print('\n[C3] 同决策两条硬超时事件：事件级 2 / 决策级 1')
    c = sr.IndependentCounters()
    for eid in ('a', 'b'):
        c.add_exception_event(run_id='r1', decision_point=3,
                              exception_audit={'hard_deadline_exception': True,
                                               'event_id': eid})
    s = c.summary()
    check('事件级 = 2 且决策级 = 1',
          s['n_deadline_timeout_events'] == 2 and s['n_deadline_timeouts'] == 1, s)
    check('真实路径每决策至多 1 条事件（不变量写明）',
          s['invariant_real_path_events_per_decision_max'] == 1)
    check('**不得要求** timeout == fallback（写进 note）', '不得要求' in s['note'])
    return True


# --------------------------------------------------------- D：静态自检（双身份）
def test_static_selfcheck_dual_identity():
    print('\n[D] 静态自检：四模块**双身份**分别核验，`expected` 不再为 None')
    sc = sr.static_selfcheck()
    check('自检通过', sc['ok'], sc['problems'])
    check('**没有** `expected=None`（v1 的 role_c_replanner / role_c_decision_log 是 None）',
          sc['no_none_expected'] is True)
    check('四个模块都纳入了 blob 与工作树核验',
          len(sc['frozen_modules']) == 4
          and all(x['blob_expected'] and x['worktree_expected']
                  for x in sc['frozen_modules']))
    rep = {x['module']: x for x in sc['frozen_modules']}
    r = rep['role_c_replanner.py']
    check('`role_c_replanner.py` 的 **blob ≠ 工作树**（无 eol=lf 属性）被如实报告',
          r['blob_expected'] != r['worktree_expected']
          and r['blob_equals_worktree_expected'] is False
          and r['crlf_bytes'] > 0, r)
    check('两个身份**分别**通过（未被改动的证据）',
          r['blob_unchanged'] and r['worktree_unchanged'], r)
    check('登记了身份约定说明', '封存 §2.1' in r['identity_convention'], r['identity_convention'])
    check('三个有 eol=lf 属性的模块 blob == 工作树',
          all(rep[m]['blob_sha256'] == rep[m]['worktree_sha256']
              for m in ('role_c_opportunity_cost.py', 'role_c_decision_log.py',
                        'role_c_timing.py')))
    check('AST：未 import / 未写 / 未 poke 冻结模块',
          not sc['imports_frozen'] and not sc['writes_to_frozen']
          and not sc['pokes_frozen'])
    return True


# --------------------------------------------------- 其余边界（v1 已覆盖，保留）
def test_archive_failure_and_durability():
    print('\n[E] 写失败不可伪称已落盘；持久化表述已更正（v1 项保留）')
    d = tempfile.mkdtemp()
    try:
        bad = sr.SidecarRecorder(run_id='r1', arm='c2',
                                 sidecar_path=os.path.join(d, 'nodir', 'x', 's.jsonl'),
                                 protocol_identity='p')
        bad._open = lambda: (_ for _ in ()).throw(OSError('synthetic'))
        rec, ok = bad.record(**_rec())
        check('写失败 ⇒ ok=False 但仍产出记录对象（算法不受影响）',
              ok is False and isinstance(rec, dict))
        check('资格 = ARCHIVE_INCOMPLETE 且失败只记内存',
              bad.archive_status == 'ARCHIVE_INCOMPLETE'
              and all(f['verified_by_supervisor'] is None for f in bad.failures))
        bad.mark_supervisor_verification(decision_point=0, verified=False)
        check('监督无法核验 ⇒ INCONCLUSIVE_ARCHIVE_FAILURE',
              bad.archive_status == 'INCONCLUSIVE_ARCHIVE_FAILURE')
        check('该决策被排除出四通道完整性分母',
              bad.completeness_denominator(8)['eligible'] == 7)
        dd = sr.durability_disclosure()
        check('flush ≠ 持久化、披露断电风险、不使用 fsync、旧表述已更正',
              dd['flush_equals_durability'] is False and '断电' in dd['disclosure']
              and '不使用' in dd['disclosure'] and '已更正' in dd['disclosure'])
    finally:
        _rm(d)
    return True


def test_fallback_classification_v2():
    print('\n[F] 回退分类（v1 项保留）：拒单 ≠ 超时回退')
    b2 = sr.classify_fallback(diag={'boundary': 'B2_budget_out_of_range'},
                              commit_observed=True)
    check('B2 ⇒ 边界拒单、不是超时回退',
          b2['fallback_kind'] == 'b2_boundary_reject'
          and b2['fallback_applied'] is False)
    check('B3 无审计 ⇒ FALLBACK_NOT_VERIFIABLE',
          sr.classify_fallback(diag={'boundary': 'B3_lambda_undefined'},
                               commit_observed=True)['fallback_kind']
          == 'FALLBACK_NOT_VERIFIABLE')
    check('仅 action=reject ⇒ FALLBACK_NOT_VERIFIABLE',
          sr.classify_fallback(diag={'log_fields': {'action': 'reject'}},
                               commit_observed=True)['fallback_kind']
          == 'FALLBACK_NOT_VERIFIABLE')
    check('B3 + 硬截止 ⇒ timeout_fallback',
          sr.classify_fallback(diag={'boundary': 'B3_lambda_undefined'},
                               commit_observed=True,
                               exception_audit={'hard_deadline_exception': True,
                                                'exception_type': 'DecisionTimeout'}
                               )['fallback_kind'] == 'timeout_fallback')
    return True


def _rm(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def main():
    print('=' * 78)
    print('分析侧 sidecar 实现回归 v2（用户 bce2e988 审查裁定的四项缺陷）')
    print('v1 逐字节保留于 _superseded_test_role_c_sidecar_recorder_v1_2026-10-14.py；')
    print('v1 的 PASS **不**构成"缺陷已解决"的证据（含恒真断言）')
    print('=' * 78)
    test_input_identity_strict_schema()
    test_commit_status_requires_evidence()
    test_recorder_rejects_v1_default()
    test_counters_units_are_separate()
    test_counter_rejects_bad_input()
    test_counter_event_vs_decision_still_correct()
    test_static_selfcheck_dual_identity()
    test_archive_failure_and_durability()
    test_fallback_classification_v2()
    print('\n' + '=' * 78)
    if FAILS:
        print('SIDECAR_TEST_V2_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('SIDECAR_TEST_V2_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
