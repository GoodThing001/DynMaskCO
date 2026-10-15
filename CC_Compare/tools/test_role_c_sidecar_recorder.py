#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析侧 sidecar 实现回归 **v3**（按用户 2026-10-14「`6f32125ff` 独立审查裁定」缺口 A/B/C）。

**版本谱系（逐字节保留，不覆盖）**：
  * v1 `8bb47c7d…`（`_superseded_test_role_c_sidecar_recorder_v1_2026-10-14.py`）
    —— 含恒真断言 `a == b or True`，其 PASS **不构成证据**；
  * v2 `af4bc925…`（`_superseded_test_role_c_sidecar_recorder_v2_2026-10-14.py`）
    —— 报告 60/60，**保留为上一版本身份**；
  * v3（本文件）—— 覆盖本轮三个缺口。

⚠️ **判据更正声明**：v2 的 60/60 是**旧判据**下的记录。
用户本轮裁定要求「观察到硬超时**不**自动等于观察到真实终态计划回退」，
即 `fallback_applied=True` 现在**必须**有决策后保留计划的权威证据。
因此 v2 中依赖旧单一 `commit_observed` 入口、且未提供保留计划证据的回退断言
**在本版本下不再成立**——这是**判据更正**，**不是**"v2 有缺陷"。
**不得**继续用 v2 的 60/60 作为"缺口 A 已解决"或"C1/C3 已独立核验"的证据。

运行：`python CC_Compare/tools/test_role_c_sidecar_recorder.py`
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
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import role_c_sidecar_recorder as sr                      # noqa: E402

_PASS = []
_FAIL = []


def check(name, cond):
    if cond:
        _PASS.append(name)
        print('  ok   %s' % name)
    else:
        _FAIL.append(name)
        print('  FAIL %s' % name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
        return False
    except exc:
        return True
    except Exception:
        return False


def _snap(**over):
    s = {'inst_idx': 0, 'clock': 3.5, 'order': 7,
         'vehicles_fingerprint': [[1, 'idle', 0, 0.0, None]],
         'plan': {'1': [2, 3]}, 'frozen': {'1': 0},
         'visible_orders': [2, 3, 4]}
    s.update(over)
    return s


def _rec(**plans):
    base = dict(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                accepted_plan=None, plan_after={'1': [2, 3]},
                commit_observed=False, action=None)
    base.update(plans)
    return sr.plan_variants(**base)


HARD = {'hard_deadline_exception': True, 'exception_type': 'DecisionTimeout',
        'event_id': 'exc-1'}
TIMEOUT_DIAG = {'boundary': 'B3_lambda_undefined'}


# ------------------------------------------------------- A 决策完成 vs 权威提交
def test_a_decision_completion_vs_commit():
    print('\n[A] 决策完成 vs 权威提交（缺口 A）')

    # A1 硬截止 + 保留计划已核验 ⇒ 真实终态回退
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             exception_audit=HARD,
                             retained_plan={'1': [2, 3]}, plan_before={'1': [2, 3]},
                             retained_plan_evidence={'kind': 'post_decision_plan_readback'})
    check('A1 硬截止 + 保留计划已核验 ⇒ timeout_fallback / applied=True',
          r['fallback_kind'] == 'timeout_fallback' and r['fallback_applied'] is True)

    # A2 **核心**：硬截止但保留计划未核验 ⇒ 不得写 True
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             exception_audit=HARD)
    check('A2 硬截止但**无**保留计划证据 ⇒ 不得 True（新 kind）',
          r['fallback_applied'] is None
          and r['fallback_kind'] == 'b3_hard_deadline_retained_plan_unverified')
    check('A2b 该 kind 指出缺什么', 'retained_plan_verified' in r['requires'])

    # A3 证据种类不在册
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             exception_audit=HARD,
                             retained_plan={'1': [2, 3]}, plan_before={'1': [2, 3]},
                             retained_plan_evidence={'kind': '我自称已核验'})
    check('A3 证据种类不在册 ⇒ 仍不得 True',
          r['fallback_applied'] is None and r['retained_plan_verified'] is False)

    # A4 保留计划与决策前计划不一致
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             exception_audit=HARD,
                             retained_plan={'1': [9]}, plan_before={'1': [2, 3]},
                             retained_plan_evidence={'kind': 'post_decision_plan_readback'})
    check('A4 保留计划 ≠ 决策前计划 ⇒ 不得 True',
          r['fallback_applied'] is None and r['retained_plan_verified'] is False)

    # A5 **回归旧缺陷**：commit 未观测但决策已完成 + 硬截止 + 保留计划
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             authoritative_commit_observed=False,
                             exception_audit=HARD,
                             retained_plan={'1': [2, 3]}, plan_before={'1': [2, 3]},
                             retained_plan_evidence={'kind': 'no_change_commit_receipt'})
    check('A5 未观测到权威提交**不**再早退（旧缺陷的回归）',
          r['fallback_kind'] == 'timeout_fallback'
          and r['authoritative_commit_observed'] is False)

    # A6 连决策完成都没观测到
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=None,
                             exception_audit=HARD,
                             retained_plan={'1': [2, 3]}, plan_before={'1': [2, 3]},
                             retained_plan_evidence={'kind': 'post_decision_plan_readback'})
    check('A6 决策完成未知 ⇒ FALLBACK_NOT_VERIFIABLE（证据再全也不推）',
          r['fallback_kind'] == 'FALLBACK_NOT_VERIFIABLE'
          and r['fallback_applied'] is None)

    # A7 三维度分别导出
    r = sr.classify_fallback(diag=TIMEOUT_DIAG, decision_outcome_observed=True,
                             authoritative_commit_observed=False,
                             exception_audit=HARD)
    check('A7 三个维度**分别**出现在结果中',
          set(['decision_outcome_observed', 'authoritative_commit_observed',
               'retained_plan_verified']) <= set(r))

    # A8/A9 计数不得比分类更宽松
    ic = sr.IndependentCounters()
    check('A8 add_fallback 拒绝「硬截止但保留计划未核验」',
          _raises(sr.CounterInputError, ic.add_fallback, run_id='r', decision_point=0,
                  fallback_event_id='e1', fallback_kind='timeout_fallback',
                  evidence={'hard_deadline_exception': True}))
    ev = dict(HARD)
    ev.update({'retained_plan': {'1': [2, 3]}, 'plan_before': {'1': [2, 3]},
               'retained_plan_evidence': {'kind': 'post_decision_plan_readback'}})
    ic.add_fallback(run_id='r', decision_point=0, fallback_event_id='e1',
                    fallback_kind='timeout_fallback', evidence=ev)
    check('A9 add_fallback 接受「硬截止 + 保留计划已核验」',
          ic.summary().get('n_fallbacks') == 1)
    check('A10 新 kind **不得**被计作真实终态回退',
          _raises(sr.CounterInputError, ic.add_fallback, run_id='r', decision_point=1,
                  fallback_event_id='e2',
                  fallback_kind='b3_hard_deadline_retained_plan_unverified',
                  evidence=ev))


# --------------------------------------------------- B 归档可独立重建输入身份
def test_b_archived_input_identity():
    print('\n[B] 归档可从原始字节独立重建输入身份（缺口 B）')
    d = tempfile.mkdtemp(prefix='sr_v3_b_')
    try:
        rec_path = os.path.join(d, 's.jsonl')
        fr = sr.freeze_input(protocol_identity='proto-1', decision_point=0,
                             input_snapshot=_snap())
        rec = sr.SidecarRecorder(run_id='r', arm='c2', sidecar_path=rec_path,
                                 protocol_identity='proto-1', output_root=d)
        rec_obj, ok = rec.record(decision_point=0, input_snapshot=_snap(), action='reject',
                                 boundary='B3_lambda_undefined', frozen_input=fr,
                                 exception_audit=HARD,
                                 decision_outcome_observed=True,
                                 **_plans_kw())
        rec.close()
        check('B1 写入成功', ok is True)

        line = io.open(rec_path, encoding='utf-8').read().strip()
        arch = json.loads(line)
        check('B2 归档携带完整规范化输入快照',
              isinstance(arch.get('input_snapshot'), dict)
              and arch['input_snapshot'].get('inst_idx') == 0)
        check('B3 归档携带快照 SHA 与协议身份',
              len(arch.get('input_snapshot_sha256') or '') == 64
              and arch.get('protocol_identity') == 'proto-1')

        rc = sr.recompute_archived_input_identity(arch)
        check('B4 **仅凭归档字节**复算输入身份 ⇒ ok', rc.get('ok') is True)
        check('B5 复算出的快照 SHA 与决策 ID 都吻合',
              rc['snapshot_sha_matches'] and rc['decision_id_matches'])

        bad = dict(arch)
        bad['input_snapshot'] = _snap(order=8)
        check('B6 篡改快照后被复算程序识破',
              sr.recompute_archived_input_identity(bad).get('ok') is False)

        old = dict(arch)
        old.pop('input_snapshot')
        old.pop('input_snapshot_sha256')
        rco = sr.recompute_archived_input_identity(old)
        check('B7 缺口 B 之前的旧记录 ⇒ 明确判"无法独立复建"（不假装通过）',
              rco.get('ok') is False and 'input_snapshot' in rco.get('why', ''))

        # 末行必须**就是**目标记录
        okv, why = rec.verify_written(expect_record=rec_obj)
        check('B8 verify_written 证明末行 == 目标记录', okv is True)
        tampered = json.loads(json.dumps(rec_obj))
        tampered['action'] = 'accept'
        okv2, why2 = rec.verify_written(expect_record=tampered)
        check('B9 目标记录不一致 ⇒ 写入后验收失败', okv2 is False)

        # 输入不可变：三臂共享同一冻结对象
        fr.assert_unchanged(_snap())
        check('B10 快照未被修改 ⇒ assert_unchanged 通过', True)
        check('B11 快照被修改 ⇒ InputIdentityError（共享对象不天然不可变）',
              _raises(sr.InputIdentityError, fr.assert_unchanged, _snap(clock=9.9)))

        d2 = tempfile.mkdtemp(prefix='sr_v3_b2_')
        try:
            rec2 = sr.SidecarRecorder(run_id='r', arm='c2',
                                      sidecar_path=os.path.join(d2, 't.jsonl'),
                                      protocol_identity='proto-1', output_root=d2)
            check('B12 record 在写入前重验冻结输入（被改 ⇒ 抛）',
                  _raises(sr.InputIdentityError, rec2.record,
                          decision_point=1, input_snapshot=_snap(clock=1.0),
                          action='reject', frozen_input=fr,
                          decision_outcome_observed=True, **_plans_kw()))
        finally:
            shutil.rmtree(d2, ignore_errors=True)

        # 三臂逐元素一致 ⇒ 同一 input_decision_id；不同 ⇒ 不同
        ids = [sr.normalized_input_identity(protocol_identity='proto-1',
                                           decision_point=0, input_snapshot=_snap())
               for _ in range(3)]
        check('B13 三臂同一快照 ⇒ 同一 input_decision_id（逐元素一致的必要线索）',
              len(set(ids)) == 1)
        other = sr.normalized_input_identity(protocol_identity='proto-1',
                                            decision_point=0,
                                            input_snapshot=_snap(order=8))
        check('B14 任一元素不同 ⇒ 身份不同', other != ids[0])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _plans_kw():
    return dict(plan_before={'1': [2, 3]}, candidate_plan={'1': [2, 3, 7]},
                accepted_plan=None, plan_after={'1': [2, 3]},
                commit_observed=False)


# --------------------------------------------------- C 归档输出路径运行时门
def test_c_output_path_gate():
    print('\n[C] 归档输出路径的**运行时**门（缺口 C）')
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(tools_dir))

    check('C1 缺 output_root ⇒ 构造即抛（不允许隐式推断）',
          _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                  sidecar_path='x.jsonl', protocol_identity='p'))

    d = tempfile.mkdtemp(prefix='sr_v3_c_')
    try:
        check('C2 路径在 output_root 之外 ⇒ 构造即抛',
              _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                      sidecar_path=os.path.join(tempfile.gettempdir(), 'outside.jsonl'),
                      protocol_identity='p', output_root=d))
        check('C3 `..` 逃逸 ⇒ 抛',
              _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                      sidecar_path=os.path.join(d, '..', 'escape.jsonl'),
                      protocol_identity='p', output_root=d))
        check('C4 指向源码/工装目录（禁写根）⇒ 抛',
              _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                      sidecar_path=os.path.join(tools_dir, 'evil.jsonl'),
                      protocol_identity='p', output_root=os.path.dirname(tools_dir)))
        check('C5 指向历史归档原件 ⇒ 抛',
              _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                      sidecar_path=os.path.join(d, 'P1A_PhaseA_L1L2_2026-10-14.json'),
                      protocol_identity='p', output_root=d))
        check('C6 指向 MASKCO_code / 主模型目录 ⇒ 抛',
              _raises(sr.ArchivePathError, sr.SidecarRecorder, run_id='r', arm='c2',
                      sidecar_path=os.path.join(repo, 'MASKCO_code', 'evil.jsonl'),
                      protocol_identity='p', output_root=repo))

        good = os.path.join(d, 'ok.jsonl')
        rec = sr.SidecarRecorder(run_id='r', arm='c2', sidecar_path=good,
                                 protocol_identity='p', output_root=d)
        check('C7 合法路径 ⇒ 过门，且门在**写之前**已执行',
              rec.path_gate['ok'] is True and rec.path_gate['checked_before_write'] is True)
        rec_obj, ok = rec.record(decision_point=0, input_snapshot=_snap(), action='reject',
                                 exception_audit=HARD, decision_outcome_observed=True,
                                 **_plans_kw())
        check('C8 合法路径写入成功', ok is True)
        check('C9 静态自检明确**不**声称解决缺口 C',
              'gap_c_scope_limit' in sr.static_selfcheck()
              and '不得' in sr.static_selfcheck()['gap_c_scope_limit'])
        # 路径替换检测：模拟归档目标身份在两次写入之间被换掉（inode 变化）
        rec._target_identity = (0, 0)
        rec.n_records_before_substitution = rec.n_records
        _, ok2 = rec.record(decision_point=1, input_snapshot=_snap(), action='reject',
                            exception_audit=HARD, decision_outcome_observed=True,
                            **_plans_kw())
        check('C10 归档目标身份被替换（inode 变化）⇒ 写入后验收降级',
              rec.archive_status != 'COMPLETE' and ok2 is False)
        check('C11 降级原因如实记录为路径替换',
              any('path substitution' in f['error'] for f in rec.failures))
        rec.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print('=' * 78)
    print('sidecar 实现回归 v3（缺口 A/B/C）')
    print('=' * 78)
    test_a_decision_completion_vs_commit()
    test_b_archived_input_identity()
    test_c_output_path_gate()
    print('\n' + '=' * 78)
    print('通过 %d / 失败 %d' % (len(_PASS), len(_FAIL)))
    if _FAIL:
        for f in _FAIL:
            print('  FAILED: %s' % f)
        print('SIDECAR_TEST_V3_FAIL')
        return 1
    print('SIDECAR_TEST_V3_PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
