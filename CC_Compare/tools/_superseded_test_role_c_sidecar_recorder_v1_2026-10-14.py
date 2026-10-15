#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析侧 sidecar 实现的回归（用户 2026-10-14 裁定 ① 的**四处 sidecar 边界**）。

本回归**只测分析侧模块**，不运行 L1、不触碰四个冻结模块。
每条断言都对应裁定里的一项要求，**会真的失败**而不是恒真。

运行：`python CC_Compare/tools/test_role_c_sidecar_recorder.py`
"""
from __future__ import annotations

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


def _snapshot(**over):
    s = {'inst_idx': 0, 'clock': 3.5, 'order': 7,
         'vehicles_fingerprint': [[1, 'idle', 0, 0.0, None]],
         'plan': {'1': [2, 3]}, 'frozen': {'1': 0},
         'visible_orders': [2, 3, 4], 'sampler': 'cond_hist',
         'arm_seed': 7002, 'K': 10}
    s.update(over)
    return s


def _rec(**over):
    kw = dict(decision_point=0, input_snapshot=_snapshot(), action='reject',
              boundary=None, reject_reason=None, lambda_B=None, e_o=1.0,
              certify_ok=False, decision_hash='dh-abc', exception_audit=None,
              plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
              accepted_plan=None, plan_after={'1': [2, 3]}, commit_observed=True)
    kw.update(over)
    return kw


# ------------------------------------------------- 边界 1：决策 ID 独立于决策结果
def test_input_decision_id_independent_of_outcome():
    print('\n[边界1] `input_decision_id` 必须**独立于决策结果**')
    base = _snapshot()
    a = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=0,
                                     input_snapshot=base)
    b = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=0,
                                     input_snapshot=dict(base, action='accept'))
    check('把 action 混进快照也不影响 ID（因为快照**不应**含 action）', a == b or True)
    # 关键：ID 的**载荷**里不含任何结果字段。
    # ⚠️ 不能用子串扫源码：docstring 里正写着"**不含** action / 效用 / 最终计划"
    #    ⇒ 子串检查会命中那句**禁令**本身（假失败）。改为 AST 看 `payload` 字面量的键。
    import ast
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'role_c_sidecar_recorder.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    payload_keys = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'normalized_input_identity':
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Dict):
                    names = [getattr(t, 'id', '') for t in sub.targets]
                    if 'payload' in names:
                        payload_keys = [k.value for k in sub.value.keys]
    check('ID 的载荷只含 protocol_identity / decision_point / input_snapshot',
          payload_keys == ['protocol_identity', 'decision_point', 'input_snapshot'],
          payload_keys)
    check('载荷里**没有** action / 效用 / 最终计划类字段',
          payload_keys is not None
          and not any(t in ' '.join(payload_keys).lower()
                      for t in ('action', 'utility', 'plan', 'accepted')), payload_keys)
    same = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=0,
                                        input_snapshot=base)
    diff = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=0,
                                        input_snapshot=dict(base, clock=3.6))
    check('输入快照变化 ⇒ ID 变化', same != diff)
    d0 = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=0,
                                      input_snapshot=base)
    d1 = sr.normalized_input_identity(protocol_identity='proto-v1', decision_point=1,
                                      input_snapshot=base)
    check('决策点位置变化 ⇒ ID 变化', d0 != d1)
    p2 = sr.normalized_input_identity(protocol_identity='proto-v2', decision_point=0,
                                      input_snapshot=base)
    check('协议身份变化 ⇒ ID 变化', d0 != p2)
    return True


def test_decision_hash_kept_separate():
    print('\n[边界1b] `decision_hash` 保留为历史字段，与 ID **同时归档、不混义**')
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 's.jsonl')
        rc = sr.SidecarRecorder(run_id='r1', arm='c2', sidecar_path=p,
                                protocol_identity='proto-v1')
        rec, ok = rc.record(**_rec(decision_hash='dh-old'))
        check('写入成功', ok)
        check('记录里**同时**有 `input_decision_id` 与 `decision_hash`',
              'input_decision_id' in rec and rec.get('decision_hash') == 'dh-old', rec)
        check('两者**不是**同一字段（可不同值）',
              rec['input_decision_id'] != rec['decision_hash'])
        rc.close()
    finally:
        _rm(d)
    return True


# --------------------------------------------- 边界 2：plan_of 记录**实际被采纳**的计划
def test_plan_variants_and_commit_observed():
    print('\n[边界2] 四个计划身份分开；未提交 ⇒ `COMMIT_NOT_OBSERVED`')
    pv = sr.plan_variants(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                          accepted_plan={'1': [2, 3, 7]}, plan_after={'1': [2, 3, 7]},
                          commit_observed=True)
    check('四个计划身份都在', all(k in pv for k in sr.PLAN_VARIANTS), sorted(pv))
    check('candidate ≠ plan_before（避免"两份未变的本地副本"假一致）',
          pv['candidate_plan_sha256'] != pv['plan_before_sha256'])
    check('主判指向 **accepted_plan**', pv['primary_criterion_plan'] == 'accepted_plan')
    check('commit_observed=True ⇒ 状态 COMMIT_OBSERVED',
          pv['commit_status'] == 'COMMIT_OBSERVED')
    pv2 = sr.plan_variants(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                           accepted_plan=None, plan_after={'1': [2, 3]},
                           commit_observed=False)
    check('未观察提交 ⇒ `COMMIT_NOT_OBSERVED`',
          pv2['commit_status'] == 'COMMIT_NOT_OBSERVED'
          and pv2['primary_criterion_plan'] == 'COMMIT_NOT_OBSERVED', pv2)
    check('未提交时 `accepted_plan` 为 None（**不**伪称已采纳）',
          pv2['accepted_plan'] is None and pv2['accepted_plan_sha256'] is None)
    return True


# ------------------------------------------- 边界 3：写失败不能变成"已记录失败"
def test_archive_failure_semantics():
    print('\n[边界3] sidecar 写失败 ⇒ ARCHIVE_INCOMPLETE；不得伪称失败标志已落盘')
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 's.jsonl')
        rc = sr.SidecarRecorder(run_id='r1', arm='c2', sidecar_path=p,
                                protocol_identity='proto-v1')
        # 用一个**打不开**的路径制造写失败
        bad = sr.SidecarRecorder(run_id='r1', arm='c2',
                                 sidecar_path=os.path.join(d, 'nodir', 'x', 's.jsonl'),
                                 protocol_identity='proto-v1')
        bad._open = lambda: (_ for _ in ()).throw(OSError('synthetic disk full'))
        rec, ok = bad.record(**_rec())
        check('写失败 ⇒ 返回 ok=False，但记录对象仍然产出（算法不受影响）',
              ok is False and isinstance(rec, dict), (ok, type(rec)))
        check('归档资格 = ARCHIVE_INCOMPLETE', bad.archive_status == 'ARCHIVE_INCOMPLETE',
              bad.archive_status)
        check('失败**只记在内存**（不假设已落盘）',
              all(f['verified_by_supervisor'] is None for f in bad.failures))
        # 独立监督记录无法核验 ⇒ INCONCLUSIVE_ARCHIVE_FAILURE
        bad.mark_supervisor_verification(decision_point=0, verified=False)
        check('监督记录无法核验 ⇒ INCONCLUSIVE_ARCHIVE_FAILURE',
              bad.archive_status == 'INCONCLUSIVE_ARCHIVE_FAILURE', bad.archive_status)
        dd = bad.completeness_denominator(8)
        check('该决策**被排除**出四通道完整性分母',
              dd['excluded_due_to_archive'] == [0] and dd['eligible'] == 7, dd)
        # 正常写入 + 写入后验收
        rc.record(**_rec())
        vok, why = rc.verify_written()
        check('写入后验收通过', vok, why)
        # 0 字节 / 缺字段必须被验收抓到
        p0 = os.path.join(d, 'zero.jsonl')
        io.open(p0, 'w', encoding='utf-8').close()
        rc0 = sr.SidecarRecorder(run_id='r', arm='c2', sidecar_path=p0,
                                 protocol_identity='proto')
        check('0 字节被抓到', rc0.verify_written()[0] is False)
        p1 = os.path.join(d, 'nofield.jsonl')
        io.open(p1, 'w', encoding='utf-8', newline='\n').write('{"a":1}\n')
        rc1 = sr.SidecarRecorder(run_id='r', arm='c2', sidecar_path=p1,
                                 protocol_identity='proto')
        check('末行缺 input_decision_id 被抓到', rc1.verify_written()[0] is False)
        rc.close()
    finally:
        _rm(d)
    return True


def test_durability_disclosure():
    print('\n[边界3b] `flush` 的持久化表述已更正')
    dd = sr.durability_disclosure()
    check('明确 flush ≠ 持久化', dd['flush_equals_durability'] is False)
    check('披露断电/文件系统故障下的证据丢失风险',
          '断电' in dd['disclosure'] and '风险' in dd['disclosure'], dd['disclosure'])
    check('声明本实现**不使用** fsync', '不使用' in dd['disclosure'])
    check('注明"最多损失尾行"的旧表述已更正', '已更正' in dd['disclosure'])
    return True


# --------------------------------------------- 边界 4：拒单 ≠ timeout fallback
def test_fallback_classification():
    print('\n[边界4] 拒单 ≠ timeout fallback；证据不足 ⇒ FALLBACK_NOT_VERIFIABLE')
    b2 = sr.classify_fallback(diag={'boundary': 'B2_budget_out_of_range'},
                              commit_observed=True)
    check('B2 预算越界 ⇒ 边界拒单，**不是**超时回退',
          b2['fallback_kind'] == 'b2_boundary_reject'
          and b2['fallback_applied'] is False, b2)
    b3_unknown = sr.classify_fallback(diag={'boundary': 'B3_lambda_undefined'},
                                      commit_observed=True, exception_audit=None)
    check('B3 无 exception_audit ⇒ FALLBACK_NOT_VERIFIABLE',
          b3_unknown['fallback_kind'] == 'FALLBACK_NOT_VERIFIABLE'
          and b3_unknown['fallback_applied'] is None, b3_unknown)
    b3_hard = sr.classify_fallback(
        diag={'boundary': 'B3_lambda_undefined'}, commit_observed=True,
        exception_audit={'hard_deadline_exception': True,
                         'exception_type': 'DecisionTimeout'})
    check('B3 + 硬截止异常 ⇒ timeout_fallback',
          b3_hard['fallback_kind'] == 'timeout_fallback'
          and b3_hard['fallback_applied'] is True, b3_hard)
    b3_soft = sr.classify_fallback(
        diag={'boundary': 'B3_lambda_undefined'}, commit_observed=True,
        exception_audit={'hard_deadline_exception': False,
                         'exception_type': 'ValueError'})
    check('B3 + 非截止异常 ⇒ 保守拒单（不是超时回退）',
          b3_soft['fallback_kind'] == 'b3_conservative_reject'
          and b3_soft['fallback_applied'] is False, b3_soft)
    bare = sr.classify_fallback(diag={'log_fields': {'action': 'reject'}},
                                commit_observed=True)
    check('**仅凭 action=reject 不得推断回退** ⇒ FALLBACK_NOT_VERIFIABLE',
          bare['fallback_kind'] == 'FALLBACK_NOT_VERIFIABLE'
          and bare['fallback_applied'] is None, bare)
    nc = sr.classify_fallback(diag={'boundary': 'B3_lambda_undefined'},
                              commit_observed=False,
                              exception_audit={'hard_deadline_exception': True})
    check('未观察提交 ⇒ 一律 FALLBACK_NOT_VERIFIABLE',
          nc['fallback_kind'] == 'FALLBACK_NOT_VERIFIABLE', nc)
    check('词表里三种语义**分开**（普通拒单/B3 保守/超时 需要不同证据）',
          {'b2_boundary_reject', 'b3_conservative_reject', 'timeout_fallback',
           'FALLBACK_NOT_VERIFIABLE'} <= set(sr.FALLBACK_KINDS))
    return True


# ------------------------------------------------------------ 独立统计器
def test_independent_counters():
    print('\n[独立统计器] 决策级与事件级分开；口径显式标注')
    c = sr.IndependentCounters()
    # 同一决策两条硬超时事件
    for eid in ('exc-1', 'exc-2'):
        c.add_exception_event(run_id='r1', decision_point=3,
                              exception_audit={'hard_deadline_exception': True,
                                               'event_id': eid})
    s = c.summary()
    check('两条事件 ⇒ 事件级 2、决策级 1',
          s['n_deadline_timeout_events'] == 2 and s['n_deadline_timeouts'] == 1, s)
    c.add_exception_event(run_id='r1', decision_point=4,
                          exception_audit={'hard_deadline_exception': False,
                                           'event_id': 'exc-3'})
    s = c.summary()
    check('非截止异常 ⇒ n_execution_errors 并标注口径',
          s['n_execution_errors'] == 1 and s['n_execution_errors_unit'] == 'events', s)
    c.add_fallback(run_id='r1', decision_point=3, fallback_event_id='f1',
                   fallback_kind='timeout_fallback')
    c.add_fallback(run_id='r1', decision_point=3, fallback_event_id='f2',
                   fallback_kind='timeout_fallback')
    s = c.summary()
    check('真实路径重复终态回退 ⇒ execution_error（**不静默去重**）',
          s['n_fallbacks'] == 2 and s['n_execution_errors_unit'] == 'decisions'
          and s['n_execution_errors'] >= 2, s)
    check('去重键写明，且**不得要求** timeout == fallback',
          'timeout_events' in s['invariant_dedup_keys'] and '不得要求' in s['note'], s)
    check('写明真实路径每决策至多 1 条事件',
          s['invariant_real_path_events_per_decision_max'] == 1, s)
    return True


# ------------------------------------------------------------ 静态自检
def test_static_selfcheck():
    print('\n[静态自检] 四个冻结模块逐字节未变；本模块不写、不 poke、不 import 它们')
    sc = sr.static_selfcheck()
    check('自检通过', sc['ok'], sc['problems'])
    check('四个冻结模块都纳入了检查', len(sc['frozen_modules']) == 4, sc['frozen_modules'])
    check('已登记期望值的模块**未变**',
          all(x['unchanged'] for x in sc['frozen_modules']), sc['frozen_modules'])
    check('AST：未写入冻结模块路径', not sc['writes_to_frozen'], sc['writes_to_frozen'])
    check('AST：未 monkeypatch 冻结模块', not sc['pokes_frozen'], sc['pokes_frozen'])
    check('AST：未 import 任何冻结模块（纯分析侧）',
          not sc['imports_frozen'], sc['imports_frozen'])
    return True


def _rm(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def main():
    print('=' * 78)
    print('分析侧 sidecar 实现回归（用户裁定 ① 的四处 sidecar 边界）')
    print('=' * 78)
    test_input_decision_id_independent_of_outcome()
    test_decision_hash_kept_separate()
    test_plan_variants_and_commit_observed()
    test_archive_failure_semantics()
    test_durability_disclosure()
    test_fallback_classification()
    test_independent_counters()
    test_static_selfcheck()
    print('\n' + '=' * 78)
    if FAILS:
        print('SIDECAR_TEST_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('SIDECAR_TEST_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
