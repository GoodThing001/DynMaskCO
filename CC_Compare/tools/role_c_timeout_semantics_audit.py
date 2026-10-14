#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**超时语义专项审计**（只读）：`DecisionTimeout` 会不会被吞成普通 B3 拒单？

背景（用户 2026-10-14 Phase A 裁决的一号优先项）
----------------------------------------------
Phase A 本批 B 类 8/8 记录呈现出一种可疑组合：

    boundary = B3_lambda_undefined   ∧   deadline_hit = False   ∧   wall ≈ 10.002 s

即"应该在 10 s 触发硬超时"，但记录里说**没有**触发内部截止。
用户据此提出：`DecisionTimeout` 可能被 `shadow_price_budget()` 的
`except Exception` **吞掉**，转成一个普通的"λ_B 未定义"结果，
于是外层只看到一次**正常返回**，超时计数与拒单原因被混淆。

本模块回答三件事，且**不重跑任何真实 Phase A**：

1. **机制是否成立**（源码事实，机械核验，不是叙述）；
2. **合成反例是否复现**（用**真实**的 `shadow_price_budget` / `_run_with_limit`
   + 抛 `DecisionTimeout` 的替身前瞻，轻量、可复现）；
3. **超时来源还能不能恢复**（逐条通道判定）。

关键区分（用户要求）
--------------------
* `evaluation_status = timed_out`（"前瞻没跑完"）—— **进度可以证明**；
* `timeout_cause = SIGALRM`（"是被硬超时打断的"）—— **需要异常路径证据**。

本模块产出的是**机制证据**；"本批 8 例是否真的走了这条路"仍需事件归档后的
针对性运行确认（本模块给出可检验的签名，并明确标注哪些是推断）。

⚠️ 只读：不修改任何被测模块字节；不跑长任务；不产生任何效用字段。
"""
from __future__ import annotations

import ast
import io
import os
import re
import signal as signal_mod

__all__ = ['FROZEN_FILES', 'SOURCE_FACTS', 'audit_source', 'counterexample_swallow',
           'recoverability_channels', 'proposed_field_separation',
           'archive_signature_check', 'run_audit']

_HERE = os.path.dirname(os.path.abspath(__file__))

#: 本审计只读的**冻结**源码（身份已在封存裁定 §2.1 登记）
FROZEN_FILES = ('role_c_forward.py', 'role_c_opportunity_cost.py',
                'role_c_timing.py', 'role_c_harness.py')


def _read(name):
    with io.open(os.path.join(_HERE, name), encoding='utf-8') as f:
        return f.read()


def _lines(name):
    return _read(name).split('\n')


def _find_def_span(lines, name):
    """返回 `def/class name` 的 `(start_idx, end_idx)`（0-based，end 独占）。"""
    pat = re.compile(r'^(?:def|class) %s\b' % re.escape(name))
    for i, ln in enumerate(lines):
        if pat.match(ln):
            j = i + 1
            while j < len(lines) and not re.match(r'^(def |class )', lines[j]):
                j += 1
            return i, j
    return None, None


def _snippet(name, fn, start=1, end=None):
    lines = _lines(name)
    i, j = _find_def_span(lines, fn)
    if i is None:
        return None, (None, None)
    body = lines[i + (start - 1): j if end is None else i + end]
    return '\n'.join(body), (i + 1, j)


#: 每一条"源码事实"都要**机械核验**：名字、文件、核验函数、说明
SOURCE_FACTS = (
    ('F1_decision_timeout_is_exception',
     'role_c_timing.py',
     '`DecisionTimeout` 继承 `RuntimeError` ⇒ 属于 `Exception`，可被 `except Exception` 捕获'),
    ('F2_deadline_hit_not_independent',
     'role_c_forward.py',
     '`forward_value` 初始化 `deadline_hit=False`，且内部截止**只在每个场景开始前**检查 '
     '⇒ 场景内部的长时间调用不会被该检查观察到'),
    ('F3_last_diag_only_on_normal_return',
     'role_c_forward.py',
     '`FutureValueFn` 只在 `forward_value` **正常返回后**更新 `last_diag`；'
     '中途异常时它保持旧值/初值'),
    ('F4_shadow_price_swallows',
     'role_c_opportunity_cost.py',
     '`shadow_price_budget` 用 `except Exception` 包住两次 `future_value_fn` 调用，'
     '抛错被转成 `undefined=True` 并**正常返回**'),
    ('F5_run_with_limit_needs_propagation',
     'role_c_timing.py',
     '`_run_with_limit` 只有异常**传播出来**才判 `timed_out=True`；'
     '被判函数内部吞掉 ⇒ 外层看到正常返回'),
    ('F6_decide_candidate_discards_reason',
     'role_c_opportunity_cost.py',
     '`decide_candidate` 在 B3 边界分支里**丢弃** `shadow_price_budget` 的诊断（`_d0`），'
     '⇒ 原始抛错信息**不进入**决策记录'),
)


def audit_source():
    """机械核验 `SOURCE_FACTS`。返回 `{'facts': [...], 'all_hold': bool}`。"""
    facts = []

    def add(key, holds, evidence):
        facts.append({'fact': key, 'holds': bool(holds), 'evidence': evidence})

    # ---- F1 ----
    import role_c_timing as rt
    src = _read('role_c_timing.py')
    m = re.search(r'class DecisionTimeout\(([^)]*)\)', src)
    add('F1_decision_timeout_is_exception',
        bool(m) and m.group(1).strip() == 'RuntimeError'
        and issubclass(rt.DecisionTimeout, Exception)
        and issubclass(rt.DecisionTimeout, RuntimeError),
        {'class_decl': m.group(0) if m else None,
         'issubclass_Exception': issubclass(rt.DecisionTimeout, Exception),
         'message_has_hard_deadline_word':
             '硬时限' in str(rt.DecisionTimeout('单决策超过 %.3fs 硬时限 ⇒ 终止并回收' % 10.0))})

    # ---- F2 ----
    body, span = _snippet('role_c_forward.py', 'forward_value')
    init_false = "'deadline_hit': False" in (body or '')
    # 截止检查必须在 `for scen in scenarios:` 循环体内、且在 `for o in _ordered(...)` **之前**
    scen_loop = None
    for i, ln in enumerate((body or '').split('\n')):
        if re.match(r'\s*for scen in scenarios:', ln):
            scen_loop = i
            break
    order_loop = None
    for i, ln in enumerate((body or '').split('\n')):
        if re.match(r'\s*for o in _ordered\(', ln):
            order_loop = i
            break
    dl_check = None
    for i, ln in enumerate((body or '').split('\n')):
        if 'time.monotonic() > deadline' in ln:
            dl_check = i
            break
    indep = (scen_loop is not None and order_loop is not None and dl_check is not None
             and scen_loop < dl_check < order_loop
             and len((body or '').split('\n')[dl_check]) - len(
                 (body or '').split('\n')[dl_check].lstrip()) == 8)
    add('F2_deadline_hit_not_independent', init_false and indep,
        {'init_deadline_hit_false': init_false, 'span_lines': span,
         'scenario_loop_idx': scen_loop, 'deadline_check_idx': dl_check,
         'order_loop_idx': order_loop,
         'check_is_before_order_loop': bool(dl_check is not None and order_loop is not None
                                           and dl_check < order_loop)})

    # ---- F3（精确版）----
    # `FutureValueFn.__init__` 里 `self.last_diag = None`；`__call__` 里**只有一处**赋值，
    # 且位于 `forward_value(...)` 调用**之后**；`__call__` 里**没有** try/except
    # ⇒ 中途异常时 `last_diag` 保持旧值（首次调用时是 `None`）。
    fv_lines = _lines('role_c_forward.py')
    i_init, j_init = _find_def_span(fv_lines, 'FutureValueFn')
    i_call = None
    for k in range(i_init or 0, j_init or 0):
        if re.match(r'\s+def __call__', fv_lines[k]):
            i_call = k
            break
    call_body = '\n'.join(fv_lines[i_call:j_init]) if i_call is not None else ''
    init_body = '\n'.join(fv_lines[i_init:i_call]) if i_call is not None else ''
    fw_call_pos = call_body.find('forward_value(')
    asg_pos = call_body.find('self.last_diag = diag')
    n_asg_call = len(re.findall(r'self\.last_diag\s*=', call_body))
    init_none = 'self.last_diag = None' in init_body
    no_try = 'try:' not in call_body
    f3 = (init_none and n_asg_call == 1 and asg_pos > fw_call_pos >= 0 and no_try)
    add('F3_last_diag_only_on_normal_return', f3,
        {'init_sets_none': init_none, 'n_assignments_in_call': n_asg_call,
         'assignment_after_forward_value': bool(asg_pos > fw_call_pos >= 0),
         'has_try_except_in_call': not no_try,
         'consequence': ('中途异常 ⇒ `last_diag` 保持旧值；首次调用时是 `None` ⇒ '
                         '调用方读到的 "forward 诊断" 可能是**缺失**而不是"观测为假"')})

    # ---- F4 ----
    sp, sp_span = _snippet('role_c_opportunity_cost.py', 'shadow_price_budget')
    try_pos = (sp or '').find('v_hi = float(future_value_fn(b))')
    except_pos = (sp or '').find('except Exception')
    undef = "'undefined': True" in (sp or '')
    add('F4_shadow_price_swallows',
        try_pos >= 0 and except_pos > try_pos and undef,
        {'span_lines': sp_span, 'try_call_pos': try_pos, 'except_pos': except_pos,
         'returns_undefined_true': undef,
         'reason_template': re.search(r"'reason': 'V̂ 求值抛错[^\n]*", sp or '')
         and re.search(r"'reason': 'V̂ 求值抛错[^\n]*", sp or '').group(0)})

    # ---- F5 ----
    rwl, rwl_span = _snippet('role_c_timing.py', '_run_with_limit')
    add('F5_run_with_limit_needs_propagation',
        'except DecisionTimeout:' in (rwl or '')
        and 'return None, True, ' in (rwl or ''),
        {'span_lines': rwl_span,
         'catches_decision_timeout': 'except DecisionTimeout:' in (rwl or ''),
         'returns_timed_out_true': 'return None, True, ' in (rwl or '')})

    # ---- F6（精确版）----
    # 边界分支：从 `if boundary is not None:` 到该分支的 `return False, diag`；
    # 在这段里**不得**出现 `_d0`（即 `shadow_price_budget` 的诊断被丢弃）。
    dc, dc_span = _snippet('role_c_opportunity_cost.py', 'decide_candidate')
    bnd_start = (dc or '').find('if boundary is not None:')
    bnd = ''
    if bnd_start >= 0:
        rest = (dc or '')[bnd_start:]
        end = rest.find('return False, diag')
        bnd = rest[:end + len('return False, diag')] if end >= 0 else rest
    d0_in_boundary = '_d0' in bnd
    lam_diag_from_rule = "'lambda_source': 'boundary_short_circuit'" in bnd
    add('F6_decide_candidate_discards_reason',
        bnd_start >= 0 and lam_diag_from_rule and not d0_in_boundary,
        {'span_lines': dc_span, 'boundary_branch_found': bnd_start >= 0,
         'boundary_branch_lines': bnd.count('\n') + 1,
         'lambda_diag_from_rule': lam_diag_from_rule,
         '_d0_referenced_in_boundary_branch': d0_in_boundary,
         'consequence': ('B3 记录里留下的 `boundary_reason` 是**通用**文案，'
                         '与"V̂ 真的非有限"无法区分；原始抛错信息不进入决策记录')})

    # ---- F7（记录层，非冻结源码）----
    # Phase A 的 B 类记录**没有**归档 `forward` 子诊断 ⇒ 记录里的 `deadline_hit`
    # 无法追溯来源（它可能来自"缺 `forward` 键"的默认取值）。
    drv = _read('role_c_phasea_l1l2.py')
    m = re.search(r"'deadline_hit': bool\(fw\.get\('deadline_hit'\)\)", drv)
    archived = re.search(r"'forward':\s*fw", drv)
    add('F7_record_drops_forward_subdiag',
        bool(m) and not archived,
        {'record_reads_fw_get': bool(m),
         'record_archives_forward_subdiag': bool(archived),
         'consequence': ('归档里 `deadline_hit=False` 有两种不可区分的来源：'
                         '(a) `last_diag` 是 dict 且其 `deadline_hit` 为 false；'
                         '(b) `forward` 键**根本不存在**（`last_diag is None`）'
                         '⇒ 本批 `deadline_hit` 不作为"未触发截止"的证据')})

    return {'facts': facts, 'all_hold': all(f['holds'] for f in facts),
            'files': {n: _sha(os.path.join(_HERE, n)) for n in FROZEN_FILES},
            'note': ('只读审计：机械核验源码事实，不修改任何被审计模块的字节')}


def _sha(p):
    import hashlib
    if not os.path.isfile(p):
        return None
    with io.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------ 合成反例
def counterexample_swallow(*, raise_inside_forward=True, deadline_s=0.05):
    """**轻量合成反例**：用真实 `shadow_price_budget` / `_run_with_limit` 复现吞没路径。

    `raise_inside_forward=True`：替身前瞻抛 `DecisionTimeout`（模拟 SIGALRM 在前瞻内触发）；
    `False`：同一个异常在**前瞻之外**抛（对照，证明吞没**只**发生在前瞻窗口内）。
    """
    import time
    import role_c_opportunity_cost as oc
    import role_c_timing as rt

    raised = {'n': 0}

    def _raise_once():
        raised['n'] += 1
        raise rt.DecisionTimeout('单决策超过 %.3fs 硬时限 ⇒ 终止并回收' % deadline_s)

    if raise_inside_forward:
        def future_value_fn(_b):
            return _raise_once()                     # 在前瞻求值内抛
    else:
        def future_value_fn(_b):
            return 0.0                               # 前瞻本身正常

    out = {}

    def work():
        if not raise_inside_forward:
            _raise_once()                            # 在前瞻**之外**抛
        return oc.shadow_price_budget(100.0, 1.0, future_value_fn)

    # ① 直接调 shadow_price_budget：看它是否把异常转成 undefined
    try:
        lam, diag = oc.shadow_price_budget(100.0, 1.0, future_value_fn)
        out['shadow_price'] = {'returned_normally': True, 'lambda_B': lam,
                               'undefined': diag.get('undefined'),
                               'reason': diag.get('reason')}
    except rt.DecisionTimeout as exc:
        out['shadow_price'] = {'returned_normally': False,
                              'propagated': type(exc).__name__}

    # ② 用**真实** `_run_with_limit` 包住：外层还看不看得见超时？
    #    ⚠️ 平台差异：`_run_with_limit` 只在 **POSIX**（有 SIGALRM/setitimer）下才有
    #    `except DecisionTimeout`；Windows 上它走协作式分支、**不捕获** ⇒
    #    对照组的异常会直接穿出去。这不是"机制不成立"，而是**该接口在非 POSIX 上
    #    根本没有硬中断语义**。两件事分开记。
    posix = hasattr(signal_mod, 'SIGALRM') and hasattr(signal_mod, 'setitimer')
    out['platform_hard_interrupt_available'] = bool(posix)
    try:
        val, timed_out, elapsed, hard = rt._run_with_limit(work, max(0.01, deadline_s))
        out['run_with_limit'] = {'timed_out': bool(timed_out),
                                 'hard_enforcement': bool(hard),
                                 'elapsed_s': elapsed,
                                 'propagated_out': False}
    except rt.DecisionTimeout:
        out['run_with_limit'] = {'propagated_out': True,
                                 'timed_out': None, 'hard_enforcement': False,
                                 'note': ('非 POSIX：`_run_with_limit` 没有 '
                                          '`except DecisionTimeout` ⇒ 异常直接穿出；'
                                          '本平台无法观测"对照组 timed_out=True"')}
        val = None
    if isinstance(val, tuple) and len(val) == 2:
        out['run_with_limit']['lambda_B'] = val[0]
        out['run_with_limit']['undefined'] = (val[1] or {}).get('undefined')
        out['run_with_limit']['reason'] = (val[1] or {}).get('reason')
    out['raised_count'] = raised['n']
    out['deadline_s'] = deadline_s
    out['raise_inside_forward'] = bool(raise_inside_forward)
    return out


def recoverability_channels():
    """**超时来源还能不能恢复**：逐条通道判定（含源码依据）。"""
    return [
        {'channel': 'shadow_price_budget 自己的 diag',
         'recoverable': True,
         'how': "`reason` 里保留了 `'V̂ 求值抛错：%s' % exc`，"
                "而 `DecisionTimeout` 的文案含「硬时限」⇒ 可做子串识别",
         'limit': ('**类型名没有被保留**（只拼了 `exc` 的字符串）⇒ 无法按异常类型判定；'
                   '文案一旦改动即失效')},
        {'channel': 'decide_candidate 的决策记录',
         'recoverable': False,
         'how': '源码事实 F6：B3 边界分支丢弃 `_d0`，'
                '`lambda_diag` 由 `rule` 重建',
         'limit': ('留下的 `boundary_reason` 是**通用**文案'
                   '（"λ_B 未定义（V̂ 非有限/差分未定义）"），'
                   '与"V̂ 真的非有限"**无法区分**')},
        {'channel': '_run_with_limit 的返回值',
         'recoverable': False,
         'how': '源码事实 F5：异常被吞 ⇒ 函数正常返回 ⇒ `timed_out=False`',
         'limit': '外层的超时计数与拒单原因被混淆（正是用户指出的问题）'},
        {'channel': 'forward_value 的 last_diag.deadline_hit',
         'recoverable': False,
         'how': '源码事实 F2+F3：内部截止只在场景开始前检查，且 `last_diag` '
                '只在正常返回后更新',
         'limit': '`deadline_hit=False` **不能**读成"没有触发截止"'},
        {'channel': '进度原件（插桩的 leaf_enter/exit + 心跳）',
         'recoverable': 'partial',
         'how': '能看到"进到哪片叶子、进了几个场景"，'
                '⇒ 可证明 `timed_out`（**前瞻没跑完**）',
         'limit': '**不能**证明"是被 SIGALRM 打断的" ⇒ `timeout_cause` 仍需异常路径证据'},
    ]


def proposed_field_separation():
    """按用户要求把两件事分开：可证的部分 vs 需要异常路径证据的部分。"""
    return {
        'evaluation_status': {
            'values': ['not_entered', 'completed', 'timed_out'],
            'provable_by': '进度原件（进入/退出叶子、进入/完成场景数）',
            'meaning': '前瞻**有没有跑完**；`timed_out` 只表示"没跑完"'},
        'timeout_cause': {
            'values': [None, 'unknown', 'SIGALRM_decision_timeout',
                       'forward_internal_deadline', 'unknown_exception',
                       'budget_exhausted_inside_forward'],
            'provable_by': ('异常路径证据（异常类型 + 抛出点），'
                            '或用 `DecisionTimeout` 的**类型**（而不是文案）在吞没点之前截获'),
            'meaning': ('**为什么**没跑完；未取得证据时**必须**写 `unknown`，'
                        '不得默认成某一因（`None` 只表示"这一栏还没填"）'),
            'no_evidence_value': 'unknown'},
        'required_new_evidence': [
            '在 `shadow_price_budget` 的 `except Exception` 里**按类型**识别 '
            '`DecisionTimeout` 并原样重抛（或至少把 `type(exc).__name__` 写进 diag）',
            '归档原始事件表（`raw_events`/`events`）与 `leaf_counts`，'
            '使通道 C 可在真数据上二次独立重算',
            '记录 `event_stream_complete`：异常或强制截断导致事件缺失时如实标注',
            'A 类未闭合区间标 `CENSORED`，**不得**补写退出事件',
        ],
        'honesty_rule': ('`deadline_hit=false` 在本批**不得**作为"未触发截止"的证据；'
                         '`timeout_cause` 在取得异常路径证据前一律 `unknown`'),
    }


def archive_signature_check(b_report_path):
    """本批归档与"吞没路径"的**一致性**检查（标明哪些是推断）。

    ⚠️ 这是**签名一致性**，**不是**证明：只有事件归档 + 针对性运行才能定案。
    """
    import json
    with io.open(b_report_path, encoding='utf-8') as f:
        b = json.load(f)
    rows = []
    for r in b['per_decision']:
        pt = r.get('partition') or {}
        l1 = r.get('l1') or {}
        stages = r.get('stages_entered') or []
        rows.append({
            'decision_point': r['decision_point'],
            'boundary': r.get('boundary'),
            'deadline_hit_recorded': r.get('deadline_hit'),
            'wall_s': r.get('wall_s'),
            'forward_value_calls': l1.get('forward_value_calls'),
            'scenarios_entered': l1.get('role_c_forward._ordered'),
            'scenarios_completed_counting_proxy':
                l1.get('complete_fn_calls_in_forward'),
            'P7b_exclusive_s': (pt.get('exclusive') or {}).get('P7b'),
            't_decision': pt.get('t_decision'),
            'P7b_share_of_truncated_window':
                (pt.get('shares') or {}).get('P7b'),
            'partition_verdict': pt.get('verdict'),
            'P7c_entered': 'P7c' in stages,
            'certify_calls': l1.get('certify_calls'),
        })
    n = len(rows)
    consistent = {
        'all_boundary_B3': all(x['boundary'] == 'B3_lambda_undefined' for x in rows),
        'all_deadline_hit_false': all(x['deadline_hit_recorded'] is False for x in rows),
        'all_wall_just_over_10s': all(x['wall_s'] and x['wall_s'] > 10.0 for x in rows),
        'exactly_one_forward_entry_each':
            all(x['forward_value_calls'] == 1 for x in rows),
        'exactly_one_scenario_entered_each':
            all(x['scenarios_entered'] == 1 for x in rows),
        'no_scenario_completed_each': all(not x['P7c_entered'] for x in rows),
        'all_partitions_closed_ok': all(x['partition_verdict'] == 'ok' for x in rows),
    }
    return {
        'n_decision_points': n,
        'per_decision': rows,
        'consistent_with_swallowed_sigalrm': consistent,
        'all_consistent': all(consistent.values()),
        'inference_only': [
            ('`forward_value_calls == 1`（本应两次：`b` 与 `b − e_o`）—— '
             '与"第一次前瞻调用中抛错、第二次从未发生"**一致**'),
            ('`scenarios_entered == 1` 且 `P7c` 从未进入 —— '
             '与"在第一个场景内部被打断"**一致**'),
            ('`boundary=B3` 且 `deadline_hit=False` —— '
             '与"异常被吞成 λ 未定义、外层看到正常返回"**一致**'),
        ],
        'requires_targeted_run': [
            '本批**未归档**原始事件表与异常类型 ⇒ 无法区分'
            '"SIGALRM 吞没"与"V̂ 真的非有限"',
            '要定案需：在吞没点按**类型**识别并原样重抛（或记 `type(exc).__name__`），'
            '并用事件归档复算通道 C',
        ],
        'explicit_warning': ('⚠️ 以上是**一致性推断**，**不是**证据链闭环；'
                             '不得据此把本批 8 例写成"已确认被硬超时打断"。'),
    }


def run_audit(*, b_report_path=None, deadline_s=0.05):
    """跑完整审计（源码事实 + 反例 + 可恢复性 + 可选归档签名）。"""
    rep = {'label': 'P1A-RoleC-timeout-semantics-audit',
           'source_audit': audit_source(),
           'counterexample_inside_forward':
               counterexample_swallow(raise_inside_forward=True, deadline_s=deadline_s),
           'counterexample_outside_forward_control':
               counterexample_swallow(raise_inside_forward=False, deadline_s=deadline_s),
           'recoverability': recoverability_channels(),
           'proposed_field_separation': proposed_field_separation()}
    if b_report_path and os.path.isfile(b_report_path):
        rep['archive_signature'] = archive_signature_check(b_report_path)
    return rep


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description='超时语义专项审计（只读）')
    ap.add_argument('--b-report', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--deadline', type=float, default=0.05)
    a = ap.parse_args(argv)
    rep = run_audit(b_report_path=a.b_report, deadline_s=a.deadline)
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({'source_facts_all_hold': rep['source_audit']['all_hold'],
                      'swallow_returned_normally':
                          rep['counterexample_inside_forward']['shadow_price']
                          .get('returned_normally'),
                      'outer_timed_out_inside_forward':
                          rep['counterexample_inside_forward']['run_with_limit']
                          ['timed_out'],
                      'outer_timed_out_control':
                          rep['counterexample_outside_forward_control']
                          ['run_with_limit']['timed_out']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
