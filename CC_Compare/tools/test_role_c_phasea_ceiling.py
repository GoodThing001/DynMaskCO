#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 专用资源门的校验器回归（用户 2026-10-14 前置门之二：登记侧）。

重点不是"合法登记能过"，而是**每一处口径混淆都必须被拒**：

  * 把在线协议时限 `single_decision_time_limit_s` 搬进 Phase A 顶层 ⇒ 拒
    （用户亲自指出的接口冲突）；
  * A 类写成数值内部截止（而不是 `"disabled"`）⇒ 拒；
  * 12 000 s 漏掉"整个 Phase A 作业"里的任一阶段（尤其 B 类诊断）⇒ 拒；
  * 外层监督器不是硬杀 / 不回收 / 不保留截断位置 ⇒ 拒；
  * 决策点集合被替换、筛选、增加 ⇒ 拒；
  * 对账容差改成"执行后可调"、或把 `P9_measured` 当成应等于 `P9_residual` ⇒ 拒；
  * 试图把 Phase A 登记写到**基线登记**路径上 ⇒ 拒。

运行：`python CC_Compare/tools/test_role_c_phasea_ceiling.py`
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_ceiling as pc                              # noqa: E402
import role_c_resource_ceiling as base                          # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def _valid():
    return pc.build_phasea_registration(
        registered_by='P1A-rolec-phasea-exec-2026-10-14',
        authorization={'source': 'user 2026-10-14 conditional release'})


def _rejects(mutate, why):
    """改一处 ⇒ 必须被拒，且问题串里能看出是哪一类。"""
    reg = _valid()
    mutate(reg)
    ok, problems = pc.validate_phasea_ceiling(reg)
    check('拒：%s' % why, not ok, '竟然通过了')
    return problems


# ------------------------------------------------------------------ 1. 正向
def test_valid_registration_passes():
    print('\n[1] 合法登记通过，且关键值 = 用户给定值')
    reg = _valid()
    ok, problems = pc.validate_phasea_ceiling(reg)
    check('合法：校验通过', ok, problems)
    check('合法：ceiling_id = p1a-rolec-phasea-v1',
          reg['ceiling_id'] == 'p1a-rolec-phasea-v1')
    check('合法：父门 = p1a-rolec-v1', reg['parent_baseline_ceiling_id'] == 'p1a-rolec-v1')
    check('合法：A 类 internal_deadline = "disabled"',
          reg['classes']['A']['internal_deadline'] == 'disabled')
    check('合法：A 类 per_decision_cap_s = 600',
          reg['classes']['A']['per_decision_cap_s'] == 600.0)
    check('合法：B 类 internal_deadline = 冻结 10 s',
          reg['classes']['B']['internal_deadline'] == base.load_frozen_time_limit())
    check('合法：phase_total_cap_s = 12000', reg['phase_total_cap_s'] == 12000.0)
    check('合法：phase_scope 含全部四阶段',
          all(s in reg['phase_scope'] for s in pc.PHASE_SCOPE_REQUIRED),
          reg['phase_scope'])
    check('合法：顶层**不含** single_decision_time_limit_s',
          'single_decision_time_limit_s' not in reg)
    return True


# ------------------------------------------------- 2. 口径混淆（用户指出的接口冲突）
def test_refuses_online_deadline_at_top_level():
    print('\n[2] 口径冲突：在线协议时限不得出现在 Phase A 顶层')
    probs = _rejects(lambda r: r.__setitem__('single_decision_time_limit_s', 600.0),
                     '顶层 single_decision_time_limit_s')
    check('口径：报错文字点明"在线协议时限"',
          any('在线协议时限' in p for p in probs), probs)

    _rejects(lambda r: r['classes']['A'].__setitem__('internal_deadline', 600.0),
             'A 类写成数值内部截止')
    _rejects(lambda r: r['classes']['A'].__setitem__('internal_deadline', 10.0),
             'A 类内部截止=10（把 A 类当 B 类）')
    _rejects(lambda r: r['classes']['B'].__setitem__('internal_deadline', 600.0),
             'B 类内部截止被改成 600（改在线协议）')
    _rejects(lambda r: r['classes']['A'].__setitem__('per_decision_cap_s', 10.0),
             'A 类单点上限被降到 10')
    _rejects(lambda r: r['classes']['A'].__setitem__('per_decision_cap_s', 300.0),
             'A 类单点上限被改成 300（未授权的值）')
    _rejects(lambda r: r['classes']['A'].__setitem__('counts_as_online_evidence', True),
             'A 类被宣称可当在线证据')
    return True


# --------------------------------------------- 3. 12 000 s 必须覆盖整个 Phase A 作业
def test_total_cap_covers_whole_phase_a_job():
    print('\n[3] 12 000 s 必须约束**整个 Phase A 作业**（含准备/L1/L2/B 类诊断）')
    _rejects(lambda r: r.__setitem__('phase_total_cap_s', 600.0),
             '总上限被改成 600')
    _rejects(lambda r: r.__setitem__('phase_total_cap_s', 43200.0),
             '总上限被换成基线门的 43200')
    _rejects(lambda r: r.__setitem__('phase_scope', list(pc.PHASE_SCOPE_REQUIRED)[:3]),
             '缺 B_class_diagnostics（正是用户点名的"不只是 A 类计算循环"）')
    _rejects(lambda r: r.__setitem__('phase_scope', ['L2_partitioned_timing']),
             '只覆盖 L2')
    _rejects(lambda r: r.__setitem__('phase_scope',
                                     list(pc.PHASE_SCOPE_REQUIRED) + ['L3_profiler']),
             '把 L3 也塞进 Phase A 范围（L3 不得自动启动）')
    return True


# ------------------------------------------------------------- 4. 外层监督器与截断
def test_supervision_requirements():
    print('\n[4] 外层监督器：硬杀 + 回收 + 保留截断位置')
    _rejects(lambda r: r['supervision'].__setitem__('mode', 'cooperative'),
             '监督器退化为协作式')
    _rejects(lambda r: r['supervision'].__setitem__('reclaim', False),
             '超时不回收进程')
    _rejects(lambda r: r['supervision'].__setitem__('preserve_truncation_record', False),
             '不保留截断位置')
    _rejects(lambda r: r.__setitem__('supervision', 'hard'),
             'supervision 不是 dict')
    return True


# --------------------------------------------------------- 5. 决策点集合事前固定
def test_decision_points_frozen():
    print('\n[5] 决策点集合：沿用原 M=8，不替换/不筛选/不增加')
    _rejects(lambda r: r['decision_points'].__setitem__('digest', '0' * 64),
             'digest 被换')
    _rejects(lambda r: r['decision_points'].__setitem__('n', 7),
             '点数被减到 7（筛选）')
    _rejects(lambda r: r['decision_points'].__setitem__('n', 12),
             '点数被增到 12（扩大状态数）')
    _rejects(lambda r: r['decision_points'].__setitem__(
        'policy', 'reselect_easier_states'), '策略改成重挑更容易的状态')
    _rejects(lambda r: r['decision_points'].__setitem__(
        'rotation_is_not_independent_state', False),
        '把轮转重复当成独立状态')
    check('决策点：digest 与基线一致',
          _valid()['decision_points']['digest']
          == '5b432f2826ea66622173ecc514378bae8d11013374d4e93ad7940a392b2de3dc')
    return True


# --------------------------------------------------- 6. 对账容差与 P9 语义（裁定 4）
def test_reconciliation_rules_frozen():
    print('\n[6] 对账容差事前写死；P9_measured ≠ P9_residual（用户裁定第 4 项）')
    _rejects(lambda r: r['reconciliation'].__setitem__(
        'tolerance_rule', 'max(50 ms, 5% * T_decision)'), '容差被放宽')
    _rejects(lambda r: r['reconciliation'].__setitem__(
        'adjustable_after_execution', True), '容差被设为"执行后可调"')
    _rejects(lambda r: r['reconciliation'].__setitem__(
        'p9_measured_equals_p9_residual', True), '把 P9_measured 与 residual 混为一谈')
    _rejects(lambda r: r['reconciliation'].__setitem__(
        'p9_measured_is_unattributed_time', False), 'P9_measured 不是未归因时间')
    return True


# ------------------------------------------------------------- 7. 资源事实沿用旧判据
def test_resource_facts_and_occupancy():
    print('\n[7] 资源事实：沿用旧门**同一条**过度订阅不等式（不放宽）')
    _rejects(lambda r: r.__setitem__('workers', 200),
             'workers=200 过度订阅')
    _rejects(lambda r: r.__setitem__('cpu_affinity', [0, 1, 2]),
             '亲和核数少于 workers')
    _rejects(lambda r: r['thread_caps'].__setitem__('OMP_NUM_THREADS', 4),
             'OMP 上限不是 1')
    _rejects(lambda r: r['thread_caps'].pop('MKL_NUM_THREADS'),
             '漏配 MKL 线程上限')
    _rejects(lambda r: r.__setitem__('workers', True),
             'workers 是 bool')
    reg = _valid()
    occ = pc.occupancy_phasea(reg)
    check('占用率：8 × 1 ≤ 96', occ['cores_requested'] <= occ['cpu_count'], occ)
    return True


# ----------------------------------------------------------- 8. 效用泄漏与身份留痕
def test_no_utility_and_identity():
    print('\n[8] 计时登记不得夹带效用；身份必须留痕')
    _rejects(lambda r: r.__setitem__('utility', 123.0), '夹带 utility')
    _rejects(lambda r: r.__setitem__('nested', {'revenue': 1.0}), '嵌套夹带 revenue')
    _rejects(lambda r: r.__setitem__('is_template', True), '模板冒充登记')
    _rejects(lambda r: r.__setitem__('ceiling_id', 'p1a-rolec-v1'),
             '复用基线身份')
    _rejects(lambda r: r.pop('registered_by'), '缺 registered_by')
    _rejects(lambda r: r.__setitem__('authorization', ''), '缺授权来源')
    return True


# ------------------------------------------------------------------ 9. 落盘防线
def test_write_refuses_baseline_path():
    print('\n[9] 落盘：拒绝写到基线登记路径上（防手滑改父门）')
    reg = _valid()
    parent = os.path.join(pc._repo_root(), base.CEILING_REL.replace('/', os.sep))
    try:
        pc.write_phasea_registration(reg, parent)
        check('落盘：拒绝基线路径', False, '竟然接受了基线路径')
    except pc.PhaseACeilingError as exc:
        check('落盘：拒绝基线路径', '不得' in str(exc) or '拒绝写入' in str(exc), str(exc))
    bad = _valid()
    bad['phase_total_cap_s'] = 1.0
    try:
        pc.write_phasea_registration(bad, os.path.join(pc._repo_root(), 'tmp_never.json'))
        check('落盘：校验不过不落盘', False, '竟然落盘了')
    except pc.PhaseACeilingError:
        check('落盘：校验不过不落盘', True)
    return True


# ----------------------------------------------------------- 10. 父门未被改动（只读）
def test_baseline_untouched():
    print('\n[10] 父门 p1a-rolec-v1 只读核验：旧校验器仍通过 + 字节身份在案')
    st = pc.baseline_status()
    check('父门：存在', st['exists'], st)
    check('父门：旧校验器仍判通过', st['legacy_valid'], st['legacy_problems'])
    check('父门：ceiling_id 仍为 p1a-rolec-v1', st.get('ceiling_id') == 'p1a-rolec-v1')
    check('父门：single_decision_time_limit_s 仍为 10.0',
          st.get('single_decision_time_limit_s') == 10.0)
    check('父门：字节身份属于已记录的双身份之一', st['matches_recorded'], st['sha256'])
    check('父门：无问题', not st['problems'], st['problems'])
    d = pc.baseline_recorded_digests()
    check('父门：双身份 **并列保留**（CRLF 与 LF 两个摘要都在案）',
          d['crlf_worktree'].startswith('6d0e57c6')
          and d['git_lf_blob'].startswith('020fc0e7'), d)
    check('父门：当前工作树 = CRLF 原件（= 封存裁定登记的那一个）',
          st['byte_identity'] == 'crlf_worktree'
          and st['sha256'] == d['crlf_worktree'], (st['byte_identity'], st['sha256']))
    return True


# --------------------------------------------------------------- 11. 加载器 fail-closed
def test_loader_fail_closed():
    print('\n[11] 加载器：缺失/未过校验 ⇒ 抛（不返回空 dict）')
    missing = os.path.join(pc._repo_root(), 'CC_Compare', '__no_such_ceiling__.json')
    try:
        pc.load_phasea_ceiling(missing)
        check('加载器：缺失即抛', False, '竟然返回了')
    except pc.PhaseACeilingNotRegistered:
        check('加载器：缺失即抛', True)
    bad = _valid()
    bad['classes']['A']['internal_deadline'] = 600.0
    import io
    import json
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        with io.open(tmp, 'w', encoding='utf-8') as f:
            json.dump(bad, f, ensure_ascii=False)
        try:
            pc.load_phasea_ceiling(tmp)
            check('加载器：未过校验即抛', False, '竟然接受了')
        except pc.PhaseACeilingNotRegistered:
            check('加载器：未过校验即抛', True)
    finally:
        os.remove(tmp)
    return True


def main():
    print('=' * 78)
    print('Phase A 专用资源门校验器回归')
    print('=' * 78)
    test_valid_registration_passes()
    test_refuses_online_deadline_at_top_level()
    test_total_cap_covers_whole_phase_a_job()
    test_supervision_requirements()
    test_decision_points_frozen()
    test_reconciliation_rules_frozen()
    test_resource_facts_and_occupancy()
    test_no_utility_and_identity()
    test_write_refuses_baseline_path()
    test_baseline_untouched()
    test_loader_fail_closed()
    print('\n' + '=' * 78)
    if FAILS:
        print('CEILING_TEST_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('CEILING_TEST_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
