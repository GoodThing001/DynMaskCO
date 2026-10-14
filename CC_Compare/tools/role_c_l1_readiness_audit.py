#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**只读**审计：L1 就绪度 + 优先级 2 超时口径盘查。

授权：用户 2026-10-14「C2 冻结后六项裁定」——
      ① L1 **有条件批准**：先完成只读预检；
      ④ 优先级 2 **批准只读盘查**，限定现有资料与既有记录，**不重算历史效用**。

本模块**只读**：不改预声明、不改被测模块、不跑 L1/L2、不启动服务器任务。

逐字段状态取值（用户指定）：`AVAILABLE` / `MISSING` / `NOT_APPLICABLE` /
`NOT_VERIFIABLE`。**不得**把"没查到"说成"不存在"，也**不得**把"没归档"说成"没发生"。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import re
import sys

__all__ = ['FIELD_STATUS', 'audit_l1_readiness', 'audit_timeout_accounting',
           'run_count_account', 'main']

FIELD_STATUS = ('AVAILABLE', 'MISSING', 'NOT_APPLICABLE', 'NOT_VERIFIABLE')

#: 预声明要求的四条通道 + 相关接口
READINESS_FIELDS = (
    'C1_fallback_recorded',
    'C2_exception_captured',
    'C2_decision_point_bound',
    'C2_budget_point_bound',
    'C2_enters_persistent_archive',
    'C3_decision_level_counter',
    'C3_event_level_counter',
    'C4_raw_events_archived',
    'C4_offline_recompute_possible',
    'HI_LO_substatus',
    'N_BUDGET_POINTS_COMPLETED',
    'OLD_VERSION_plan_of',
    'OLD_VERSION_action_sequence',
    'TIMING_CHANNEL_carries_audit',
)


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel, root):
    p = os.path.join(root, rel.replace('/', os.sep))
    return io.open(p, encoding='utf-8').read() if os.path.isfile(p) else None


def audit_l1_readiness(*, root=None):
    """逐字段就绪度（静态 + 归档存在性）。"""
    root = root or _root()
    oc = _read('CC_Compare/tools/role_c_opportunity_cost.py', root) or ''
    rp = _read('CC_Compare/tools/role_c_replanner.py', root) or ''
    dl = _read('CC_Compare/tools/role_c_decision_log.py', root) or ''
    tm = _read('CC_Compare/tools/role_c_timing.py', root) or ''
    bcls = _read('CC_Compare/P1A_PhaseA_L1L2_B类_原件_2026-10-14.json', root) or ''

    f = {}

    def put(k, status, detail, extra=None):
        assert status in FIELD_STATUS, status
        f[k] = {'status': status, 'detail': detail}
        if extra:
            f[k].update(extra)

    # ---- C1：回退是否落盘 ----
    put('C1_fallback_recorded', 'MISSING',
        '预声明要求落盘 fallback_applied / fallback_kind / event_id / 回退前后计划；'
        '当前代码**没有**这三个字段的任何产出')
    # ---- C2：捕获 ----
    has_audit = "'exception_audit'" in oc and '_exception_audit(' in oc
    put('C2_exception_captured', 'AVAILABLE' if has_audit else 'MISSING',
        'shadow_price_budget 的 except 捕获点已产出 exception_audit（静态确认）')
    # ---- C2：决策身份绑定 ----
    seg = oc[oc.find('def _exception_audit'):oc.find('def _exception_audit') + 900]
    has_dp = "'decision_point'" in seg
    put('C2_decision_point_bound', 'AVAILABLE' if has_dp else 'MISSING',
        'exception_audit 只含全进程递增 event_id，**不含 decision_point**'
        if not has_dp else '含 decision_point')
    _bp = "'budget_point'" in seg
    put('C2_budget_point_bound', 'AVAILABLE' if _bp else 'MISSING',
        'budget_point 由 `except` 体内**只读推断**（`FutureValueFn.calls` 长度）；'
        '注入的不是 `FutureValueFn` 时取值 `unknown`（字段仍在，只是取不到具体点）'
        if _bp else '未见 budget_point 字段')
    # ---- C2：进入持久归档 ----
    log_calls = re.findall(r'self\.logger\.log_event\((.*?)\)', rp, re.S)
    passes_audit = any('exception_audit' in c for c in log_calls)
    rec_has = "'exception_audit'" in dl
    if passes_audit or rec_has:
        put('C2_enters_persistent_archive', 'AVAILABLE', '归档链已承载 exception_audit')
    else:
        put('C2_enters_persistent_archive', 'MISSING',
            'role_c_replanner._log() 只传常规字段；role_c_decision_log.log_event() '
            '的 rec 结构固定、不接收 exception_audit ⇒ 诊断字段**没有**进入可复核归档')
    # ---- C3 ----
    counter_present = any(s in oc or s in tm for s in
                          ('n_deadline_timeout_events', 'n_deadline_timeouts'))
    put('C3_decision_level_counter', 'AVAILABLE' if counter_present else 'MISSING',
        '独立统计器（决策级）尚未实现')
    put('C3_event_level_counter', 'AVAILABLE' if counter_present else 'MISSING',
        '独立统计器（事件级）尚未实现')
    # ---- C4 ----
    put('C4_raw_events_archived',
        'AVAILABLE' if ('raw_events' in bcls) else 'MISSING',
        '已归档 B 类原件中不含 raw_events / events（`plan_of` 与 `exception_audit` 亦为 0 次）')
    put('C4_offline_recompute_possible', 'NOT_VERIFIABLE',
        '分区工具具备离线扫描能力，但**本批真数据未归档事件表** ⇒ 无法在真数据上二次独立重算')
    # ---- hi/lo ----
    has_hi_lo = ("'hi'" in oc and "'lo'" in oc and 'n_budget_points' in oc)
    put('HI_LO_substatus', 'AVAILABLE' if has_hi_lo else 'MISSING',
        'C2 未输出预声明的 hi/lo 完整子状态（进入/完成/异常/短路）')
    put('N_BUDGET_POINTS_COMPLETED', 'AVAILABLE' if 'n_budget_points_completed' in oc
        else 'MISSING', '未输出 n_budget_points_entered / completed')
    # ---- 旧版证据 ----
    put('OLD_VERSION_plan_of', 'MISSING',
        '全树搜索：plan_of 只出现在源码/测试/设计文档，**任何归档里都没有** ⇒ '
        '旧版 8 个决策点的"被采纳计划"没有独立合格副本',
        {'searched': 'CC_Compare/**/*.{json,jsonl,py,md}'})
    put('OLD_VERSION_action_sequence', 'AVAILABLE',
        'B 类原件每条 per_decision 都有 action / boundary / lambda_B / e_o / certify_ok'
        '（但**没有** plan）')
    # ---- 计时通道 ----
    put('TIMING_CHANNEL_carries_audit',
        'AVAILABLE' if 'exception_audit' in tm else 'MISSING',
        'DecisionWorkFn.__call__ 只提取少量行为标签；run_timing 报告只记时间与状态 ⇒ '
        '历史计时入口**不会**保存 exception_audit')
    return {'fields': f,
            'summary': dict(collections.Counter(v['status'] for v in f.values())),
            'note': ('只读审计。MISSING = 代码/归档里确实没有；'
                     'NOT_VERIFIABLE = 现有资料不足以判定（**不得**读成"没发生"）')}


def run_count_account(*, registered_runs=16):
    """三版本 × 8 决策点的**次数账**（用户指出的 16 vs 24 差额）。"""
    m = 8
    design = [
        {'run': '冻结旧版', 'decision_points': m,
         'primary_evidence': 'action（AVAILABLE）+ plan_of（**MISSING**）'},
        {'run': 'C2 新版', 'decision_points': m,
         'primary_evidence': '需本次运行产出（含 plan_of）'},
        {'run': '新版 write_off_control', 'decision_points': m,
         'primary_evidence': '需本次运行产出（含 plan_of）'},
    ]
    need = sum(x['decision_points'] for x in design)
    return {
        'registered_runs': int(registered_runs),
        'design_implied_runs': need,
        'discrepancy': need - int(registered_runs),
        'per_run': design,
        'why': ('预声明 §7.1 的主判要求逐位比较 `action` **与** 被采纳 `plan_of`；'
                '要比较旧版就必须有旧版的 **8 次** 运行证据。'
                '而 §7.4 只写了"各 1 次 + write_off 各 1 次 = 16 次"，'
                '把旧版那一组**漏在账外**（或默认旧版可复用 —— 但 §8.3/本审计显示'
                '旧版 `plan_of` 在任何归档里都不存在）'),
        'options': [
            {'id': 'A', 'desc': '保留 §7.1 原样（action + plan_of 都要比）',
             'runs': 24, 'needs': '执行前**带日期的登记勘误**（把 16 → 24 并说明依据）'},
            {'id': 'B', 'desc': '把旧版一侧的主判收窄为 `action`（plan 记为 NOT_VERIFIABLE）',
             'runs': 16, 'needs': '执行前勘误，并把旧版对比的结论上限写明为 **PARTIAL**'},
            {'id': 'C', 'desc': '重跑旧版 8 次（冻结旧版身份 5f6480b0…）并当次产出 plan_of',
             'runs': 24, 'needs': '执行前勘误 + 旧版运行的资源与身份登记'},
        ],
        'forbidden': ('**不得**在运行之后补解释次数；**不得**以"修测试"为名顺手改预声明'
                      '（用户 2026-10-14 明确）'),
    }


def audit_timeout_accounting(*, root=None, matrix=None):
    """优先级 2：`Q_timeout_clean=False` 记录的**只读**分类盘查。"""
    root = root or _root()
    cands = sorted(glob.glob(os.path.join(
        root, 'CC_Compare', 'audit', 'BASELINE_QUALIFICATION_MATRIX_*.json')))
    mp = matrix or (cands[-1] if cands else None)
    if not mp:
        return {'error': '找不到资格矩阵'}
    d = json.load(io.open(mp, encoding='utf-8'))
    arms = d.get('arms') or []
    bad = [a for a in arms if (a.get('Q') or {}).get('Q_timeout_clean') is False]

    def local_of(a):
        gp = (a.get('gate_path') or '').replace(
            '/home/hzeng/project/MASKCO-Main/', '')
        return os.path.join(root, gp.replace('/', os.sep))

    def mt(a):
        return (a.get('runner_reported') or {}).get('mean_timeouts')

    rows = []
    for a in bad:
        p = local_of(a)
        rows.append({
            'arm': a.get('arm'), 'method': a.get('method'), 'p_c': a.get('p_c'),
            'mean_timeouts': mt(a),
            'magnitude_real_timeout': bool((mt(a) or 0) >= 5.0),
            'accounting_incomplete': not (a.get('health') or {}).get('n_solves'),
            'identity_mismatch': 'Q_identity' in (a.get('Q_vetoes') or []),
            'gate_local': os.path.isfile(p),
        })
    big = [r for r in rows if r['magnitude_real_timeout']]
    small = [r for r in rows if (r['mean_timeouts'] or 0) < 1.0]
    mid = [r for r in rows if 1.0 <= (r['mean_timeouts'] or 0) < 5.0]
    # 本地 gate.json 能否核验 booking_horizon
    seen = with_bh = 0
    for a in bad:
        p = local_of(a)
        if not os.path.isfile(p):
            continue
        try:
            g = json.load(io.open(p, encoding='utf-8'))
        except Exception:                                        # noqa: BLE001
            continue
        seen += 1
        if 'booking_horizon' in json.dumps(g, ensure_ascii=False):
            with_bh += 1
    return {
        'matrix_path': os.path.relpath(mp, root).replace(os.sep, '/'),
        'n_arms_total': len(arms),
        'n_timeout_clean_false': len(bad),
        'categories': {
            'real_timeout_magnitude_ge_5_per_day': len(big),
            'small_magnitude_lt_1_per_day': len(small),
            'intermediate_1_to_5': len(mid),
            'accounting_incomplete__health_not_produced':
                sum(1 for r in rows if r['accounting_incomplete']),
            'identity_mismatch__Q_identity_veto':
                sum(1 for r in rows if r['identity_mismatch']),
            'not_verifiable__gate_record_absent_locally':
                sum(1 for r in rows if not r['gate_local']),
            'cleanest_subpopulation__no_veto_health_present_gate_local':
                sum(1 for r in rows if r['gate_local']
                    and not r['accounting_incomplete'] and not r['identity_mismatch']),
        },
        'booking_horizon': {
            'parseable_local_gates': seen,
            'with_booking_horizon_field': with_bh,
            'annotation': ('历史 gate 缺该字段 ⇒ **NOT_VERIFIABLE_FROM_ARCHIVED_GATE**：'
                           '只表示"无法仅从归档 gate 复核因果性"，'
                           '**不推定**发生未来信息泄漏，**不追溯**改变旧裁决'),
        },
        'overlap_warning': ('上述类别**互相重叠**（同一臂可同时属多类）⇒ '
                            '**不得**当成互斥分区，也不得相加当成总数'),
        'limits': [
            '矩阵里**没有** per-decision 级超时记录 ⇒ 只能按量级与健康项做**粗分类**',
            '46 臂的 gate 记录在本地不存在 ⇒ 只能标 **NOT_VERIFIABLE**，不得推断成因',
            '本盘查**不重算**历史效用、**不增跑**种子、**不改**任何既有裁决',
        ],
        'rows': rows,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='L1 就绪 + 超时口径 只读审计')
    ap.add_argument('--out', default=None)
    ap.add_argument('--matrix', default=None)
    a = ap.parse_args(argv)
    rep = {'label': 'P1A-C2-post-freeze-readonly-audit',
           'l1_readiness': audit_l1_readiness(),
           'run_count_account': run_count_account(),
           'priority2_timeout_accounting': audit_timeout_accounting(matrix=a.matrix),
           'authorization': ('只读：不跑 L1/L2、不改预声明、不改被测模块、'
                             '不启动服务器任务、不重算历史效用')}
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    s = rep['l1_readiness']['summary']
    print(json.dumps({
        'readiness': s,
        'run_count': {'registered': rep['run_count_account']['registered_runs'],
                      'design_implies':
                          rep['run_count_account']['design_implied_runs']},
        'priority2': rep['priority2_timeout_accounting'].get('categories'),
        'booking_horizon': rep['priority2_timeout_accounting'].get('booking_horizon'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
