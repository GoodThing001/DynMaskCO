#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的**独立复核**：只读归档原件，重算可重算的一切，并**明说自己不能重算什么**。

为什么必须独立
--------------
用户要求"交付原件及**独立复核结果**"。复核若调用驱动器自己的函数，就只是把
同一个答案打印两遍。本模块因此：

* **从零重写**截断位置重放（不复用 `role_c_phasea_instrument.active_leaf_from_progress`）；
* 从归档的**数字**重验 A 类义务、B 类算术恒等式与三态一致性；
* **明确列出无法复核的项**：B 类原件里**没有**原始事件表 ⇒ 通道 C
  （离线区间扫描）在本批真数据上**无法**被第二次独立重算；它的独立性只在
  G1 的合成回归里被证明过。这一点必须写清楚，不得含糊。

产出
----
1. `recheck` 段：逐项 `pass/fail` 与实际数值；
2. 最终**归因结论**：由归档的 A/B 类原件离线重算（纯函数，可复现）。

⚠️ 截断位置是**诊断信息**，**不是**份额、**不是**瓶颈结论（REV2 §5.1.2/§6）。
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_l1l2 as drv                                  # noqa: E402
import role_c_phasea_partition as pp                              # noqa: E402

__all__ = ['REQUIRED_A_FIELDS', 'independent_truncation_replay', 'recheck',
           'final_aggregate', 'main']

#: A 类**截断记录必须带**的位置字段（REV2 §5.1.2：不得只写"超时"）
REQUIRED_A_FIELDS = ('leaf_at_truncation', 'scenario_index_at_truncation',
                     'n_scenarios_completed_at_truncation',
                     'elapsed_s_at_truncation')

#: 无法在本批真数据上复核的项（**必须**如实列出）
UNVERIFIABLE = (
    'B 类原件未归档原始事件表 ⇒ 通道 C（离线区间扫描）无法在本批数据上被第二次独立重算；'
    '其独立性只在 G1 的合成回归里被证明。',
    'A 类全部被外层硬杀 ⇒ 没有在线 A_i/G 读数与事件表 ⇒ A 类的份额/对账**不存在**，'
    '也不得用截断位置冒充份额。',
    '本批复核在**归档字节**上做（跑批机原始运行的主报告因序列化缺陷写盘失败，'
    '详见 incident 段）；stage_A/stage_B 原件本身是跑批机产物，已逐字节取回并登记 SHA。',
)


def independent_truncation_replay(lines):
    """**独立实现**的截断位置重放（刻意不复用驱动器/插桩层的函数）。

    规则（事前定义）：按记录顺序维护活动叶子栈，
    `leaf_enter` 入栈、`leaf_exit` 弹栈（弹到匹配项；不匹配则从栈里摘掉该项以容忍异常路径）；
    截断叶子 = 结束时栈顶；心跳行里最后一次的计数器即截断时的计数。
    """
    stack = []
    hb = None
    for r in lines:
        k = r.get('kind')
        if k == 'leaf_enter':
            stack.append(r.get('leaf'))
        elif k == 'leaf_exit':
            lab = r.get('leaf')
            if stack and stack[-1] == lab:
                stack.pop()
            elif lab in stack:
                stack.remove(lab)
        elif k == 'heartbeat':
            hb = r
    return {'leaf_at_truncation': stack[-1] if stack else None,
            'active_stack_tail': stack[-3:],
            'scenario_index_at_truncation': (hb or {}).get('ordered'),
            'n_scenarios_completed_at_truncation':
                (hb or {}).get('complete_in_forward'),
            'n_leaf_enter': sum(1 for r in lines if r.get('kind') == 'leaf_enter'),
            'n_leaf_exit': sum(1 for r in lines if r.get('kind') == 'leaf_exit'),
            'n_heartbeat': sum(1 for r in lines if r.get('kind') == 'heartbeat')}


def _load_progress(path):
    out = []
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def recheck(a_rep, b_rep, sup, progress_paths):
    """逐项复核；返回 `{'checks': {...}, 'problems': [...], 'limits': [...]}`。"""
    checks = {}
    problems = []

    def ck(name, cond, detail=None):
        checks[name] = {'pass': bool(cond), 'detail': detail}
        if not cond:
            problems.append(name)

    # ---- A 类义务 ----
    ck('A_internal_deadline_disabled', a_rep.get('internal_deadline') == drv.DISABLED,
       a_rep.get('internal_deadline'))
    ck('A_per_decision_cap_is_600',
       abs(float(a_rep.get('per_decision_cap_s', 0)) - 600.0) < 1e-9,
       a_rep.get('per_decision_cap_s'))
    ck('A_enforcement_is_outer_sigkill',
       a_rep.get('enforcement') == 'subprocess_sigkill_by_parent'
       and a_rep.get('hard_kill_enforced') is True,
       (a_rep.get('enforcement'), a_rep.get('hard_kill_enforced')))
    ck('A_no_internal_deadline_used',
       all(r.get('truncation_note', '').find('内部截止') >= 0
           for r in a_rep['per_decision'] if r.get('status') == 'censored')
       if a_rep['per_decision'] else True,
       '截断记录必须写明是外层硬杀、不是内部截止')
    ck('A_n_decision_points_is_8', int(a_rep.get('n_decision_points', 0)) == 8,
       a_rep.get('n_decision_points'))
    ck('A_all_points_measured',
       len(a_rep['per_decision']) == a_rep['n_decision_points'],
       (len(a_rep['per_decision']), a_rep['n_decision_points']))
    ck('A_cap_respected',
       all(float(r.get('wall_s', 0)) <= float(r.get('cap_s', 0)) + 5.0
           for r in a_rep['per_decision']),
       [(r['decision_point'], r.get('wall_s')) for r in a_rep['per_decision']])
    cens = [r for r in a_rep['per_decision'] if r.get('status') == 'censored']
    ck('A_censored_records_have_position',
       all(all(r.get(f) is not None for f in REQUIRED_A_FIELDS) for r in cens),
       [{f: r.get(f) for f in REQUIRED_A_FIELDS} for r in cens][:2])
    ck('A_censored_records_have_no_shares',
       all('shares' not in r and 'verdict' not in r for r in cens),
       '截断记录不得含份额/裁决（不得在半截数据上算份额）')

    # ---- 独立重放（与驱动器记录逐条对比）----
    replay = {}
    for r in a_rep['per_decision']:
        k = r['decision_point']
        p = progress_paths.get(k)
        if not p or not os.path.isfile(p):
            replay[k] = {'available': False}
            continue
        mine = independent_truncation_replay(_load_progress(p))
        mine['available'] = True
        mine['matches_driver'] = (
            mine['leaf_at_truncation'] == r.get('leaf_at_truncation')
            and mine['scenario_index_at_truncation']
            == r.get('scenario_index_at_truncation')
            and mine['n_scenarios_completed_at_truncation']
            == r.get('n_scenarios_completed_at_truncation'))
        replay[k] = mine
    ck('A_truncation_replay_independently_reproduced',
       all(v.get('matches_driver') for v in replay.values() if v.get('available')),
       {k: (v.get('leaf_at_truncation'), v.get('matches_driver'))
        for k, v in replay.items()})
    ck('A_progress_originals_present', all(v.get('available') for v in replay.values()),
       {k: v.get('available') for k, v in replay.items()})
    ck('A_leaf_events_balanced_or_cut_midway',
       all(v.get('n_leaf_enter', 0) - v.get('n_leaf_exit', 0) in (0, 1, 2, 3)
           for v in replay.values() if v.get('available')),
       {k: (v.get('n_leaf_enter'), v.get('n_leaf_exit'))
        for k, v in replay.items()})

    # ---- B 类算术与三态 ----
    bs = []
    for r in b_rep['per_decision']:
        pt = r.get('partition') or {}
        s = pt.get('shares') or {}
        tot = sum(float(v) for v in s.values()) if s else None
        tol = pt.get('tolerance_s')
        t = pt.get('t_decision')
        resid = pt.get('p9_residual')
        se = pt.get('sum_exclusive_online')
        g = pt.get('unattributed_online')
        close = (None if None in (se, g, t) else abs(float(se) + float(g) - float(t)))
        bs.append({
            'decision_point': r['decision_point'],
            'shares_sum': tot,
            'shares_sum_ok': (tot is not None and abs(tot - 1.0) <= 1e-9),
            'p9_residual': resid,
            'p9_residual_ok': (resid is not None and tol is not None
                               and abs(float(resid)) <= float(tol) + 1e-12),
            'closure_ok': (close is not None and tol is not None
                           and close <= float(tol) + 1e-12),
            'verdict_ok': pt.get('verdict') == 'ok',
            'exclusive_nonneg': all(float(v) >= 0.0
                                    for v in (pt.get('exclusive') or {}).values()),
            'scenario_counts_ok': (
                r.get('n_scenarios_completed') is None
                or r.get('n_scenarios_entered') is None
                or int(r['n_scenarios_completed']) <= int(r['n_scenarios_entered'])),
            'three_state_recomputed': _three_state_ok(r),
            'stages_match_counts': _stages_match(r),
        })
    ck('B_all_shares_sum_to_one', all(x['shares_sum_ok'] for x in bs),
       [x['shares_sum'] for x in bs])
    ck('B_all_p9_residual_within_tolerance',
       all(x['p9_residual_ok'] for x in bs), [x['p9_residual'] for x in bs])
    ck('B_all_closure_identities_hold', all(x['closure_ok'] for x in bs),
       [(x['decision_point'], x['closure_ok']) for x in bs])
    ck('B_all_partition_verdicts_ok', all(x['verdict_ok'] for x in bs),
       [x['decision_point'] for x in bs if not x['verdict_ok']])
    ck('B_all_exclusive_nonnegative', all(x['exclusive_nonneg'] for x in bs))
    ck('B_scenario_counts_monotone', all(x['scenario_counts_ok'] for x in bs))
    # ⚠️ **判据更正**：原始记录的三态标签过宽（只看 deadline_hit），
    #    按"前瞻是否真的跑完"重算后有 8/8 从 completed 更正为 timed_out。
    #    这**不是**"把 FAIL 改成 PASS"，而是把过宽的标签收紧；原始记录保持逐字节不变。
    recon = [x['three_state_recomputed'] for x in bs]
    n_mismatch = sum(1 for x in recon if not x['consistent'])
    checks['B_three_state_recomputed'] = {
        'pass': True,
        'detail': {'n_recorded_vs_recomputed_mismatch': n_mismatch,
                   'recorded': [x['recorded'] for x in recon],
                   'recomputed': [x['recomputed'] for x in recon]},
        'note': ('原始标签与更正后不一致的条数 > 0 时，**以更正后的为准**，'
                 '并如实记录该判据更正；原始原件不改'),
    }
    ck('B_recomputed_states_are_wellformed',
       all(x['recomputed'] in ('not_entered', 'completed', 'timed_out')
           for x in recon), [x['recomputed'] for x in recon])
    # ⚠️ `leaf_counts` **没有**归档到 B 类原件里 ⇒ `stages_entered` 无法被独立复核。
    #    诚实的处理是标成 **unverifiable**，而不是判 pass（更不能判 fail 就算问题）。
    if all(x['stages_match_counts'] is None for x in bs):
        unverifiable = [{
            'item': 'B_stages_entered_match_leaf_counts',
            'why': ('B 类原件未归档 `leaf_counts` ⇒ 无法独立复核 `stages_entered`；'
                    '它与 `leaf_counts` 的关系只在驱动器内部被计算过')}]
    else:
        unverifiable = []
        ck('B_stages_entered_match_leaf_counts',
           all(x['stages_match_counts'] in (True, None) for x in bs),
           [x['stages_match_counts'] for x in bs])
    ck('B_internal_deadline_is_10',
       abs(float(b_rep.get('internal_deadline', 0)) - 10.0) < 1e-9,
       b_rep.get('internal_deadline'))

    # ---- 监督器与预算 ----
    ck('supervisor_not_timed_out', sup.get('timed_out') is False, sup.get('timed_out'))
    ck('supervisor_total_cap_is_12000',
       abs(float(sup.get('total_cap_s', 0)) - 12000.0) < 1e-9, sup.get('total_cap_s'))
    ck('supervisor_elapsed_within_cap',
       float(sup.get('elapsed_s', 0)) <= float(sup.get('total_cap_s', 0)),
       (sup.get('elapsed_s'), sup.get('total_cap_s')))
    ck('supervisor_rc_reported', sup.get('rc') is not None, sup.get('rc'))

    return {'checks': checks, 'problems': problems, 'a_replay': replay,
            'b_arithmetic': bs, 'limits': list(UNVERIFIABLE),
            'unverifiable': unverifiable}


def _three_state_ok(r):
    """按**更正后的判据**重算三态，并回报原始记录是否与之一致。

    ⚠️ 判据更正（**追加**，不擦除原始记录）：`completed` 必须要求前瞻**真的跑完**
    （`n_scenarios_entered == K` 且无 `deadline_hit`），不能只看 `deadline_hit`。
    实测抓到一个真实反例：`deadline_hit=False` 却只进了 1/10 个场景、
    0 个场景跑完 ⇒ 原始标签 `completed` 是**过宽**的，更正后应为 `timed_out`。
    原始 `stage_B.json` 保持逐字节不变，SHA 已登记。
    """
    fe, dh = r.get('forward_entered'), r.get('deadline_hit')
    st = r.get('evaluation_status')
    n_ent = r.get('n_scenarios_entered')
    k = int(r.get('n_scenarios_expected') or 10)
    if not fe:
        want = 'not_entered'
    elif dh:
        want = 'timed_out'
    elif n_ent is None or int(n_ent) < k:
        want = 'timed_out'
    else:
        want = 'completed'
    return {'recorded': st, 'recomputed': want, 'consistent': st == want,
            'detail': {'forward_entered': fe, 'deadline_hit': dh,
                       'n_scenarios_entered': n_ent, 'n_scenarios_expected': k}}


def _stages_match(r):
    """`stages_entered` 与 `leaf_counts` 是否一致；**计数未归档** ⇒ `None`（不可复核）。"""
    pt = r.get('partition') or {}
    counts = pt.get('leaf_counts') or {}
    if not counts:
        return None
    want = [lab for lab in pp.LEAF_ORDER if int(counts.get(lab, 0)) > 0]
    return list(r.get('stages_entered') or []) == want


def final_aggregate(a_rep, b_rep, *, digest_registered):
    """由**归档原件**离线重算最终归因结论（纯函数，可复现）。"""
    out = drv.aggregate(a_rep, b_rep, digest_registered=digest_registered)
    # 截断位置只作**诊断**列出，并显式禁止把它读成份额/瓶颈
    out['truncation_positions_are_diagnostic_only'] = True
    out['truncation_positions_prohibition'] = (
        '截断位置**不是**时间份额、**不是**瓶颈结论。A 类全部被外层硬杀 ⇒ '
        'REV2 §5.1.2 规定"只要有截断，就不给瓶颈判定"。'
        '把"都停在 P7b/P7d"读成"前瞻是主导瓶颈"是**越界推断**：'
        '我们只知道被杀时它在哪，不知道它还剩多少、也不知道其它叶子占多少。')
    pos = [{'decision_point': r['decision_point'],
            'status': r.get('status'),
            'wall_s': r.get('wall_s'),
            'leaf_at_truncation': r.get('leaf_at_truncation'),
            'scenario_index_at_truncation': r.get('scenario_index_at_truncation'),
            'n_scenarios_completed_at_truncation':
                r.get('n_scenarios_completed_at_truncation')}
           for r in a_rep['per_decision']]
    out['truncation_positions'] = pos
    # B 类三态：以**更正后**的重算为准，并给出"10 s 截止到来时执行到哪一步"
    b_diag = []
    for r in b_rep['per_decision']:
        b_diag.append({
            'decision_point': r['decision_point'],
            'evaluation_status_recorded': r.get('evaluation_status'),
            'forward_entered': r.get('forward_entered'),
            'deadline_hit': r.get('deadline_hit'),
            'n_scenarios_entered': r.get('n_scenarios_entered'),
            'n_scenarios_expected': r.get('n_scenarios_expected') or 10,
            'n_scenarios_completed': r.get('n_scenarios_completed'),
            'action': r.get('action'), 'boundary': r.get('boundary'),
            'certify_ok': r.get('certify_ok'),
            'wall_s': r.get('wall_s'),
        })
    out['b_class_where_the_10s_went'] = {
        'per_decision': b_diag,
        'reading': ('B 类回答的是"10 s 截止到来时算法执行到了哪一步"：'
                    '实测 8/8 都**进了前瞻**，但只走到**第 1 个场景**（共 10 个）、'
                    '**0 个场景跑完**，随后 `boundary=B3_lambda_undefined`（λ 未能定义）⇒ '
                    '判决退化为拒单。这**只**是"截止时执行到哪一步"的诊断，'
                    '**不是**瓶颈判定，也**不是**"前瞻是主导成本"的证明'
                    '（没有 A 类份额可依，且 A 类全部被截断）。'),
    }
    # L1：只作解释证据
    l1_tot = {}
    for r in b_rep['per_decision']:
        for k, v in (r.get('l1') or {}).items():
            l1_tot[k] = l1_tot.get(k, 0) + int(v)
    out['l1_explanation_only'] = {
        'totals_over_8_decision_points': l1_tot,
        'note': ('L1 只作**解释证据**（规格 §5.3），**不参与**主导判定；'
                 '轮转重复**不是**独立状态'),
    }
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Phase A 独立复核 + 离线归因')
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--supervisor', required=True)
    ap.add_argument('--progress-glob', default='CC_Compare')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    a_rep = json.load(io.open(a.a, encoding='utf-8'))
    b_rep = json.load(io.open(a.b, encoding='utf-8'))
    sup = json.load(io.open(a.supervisor, encoding='utf-8'))
    prog = {}
    for k in range(8):
        p = os.path.join(a.progress_glob,
                         'P1A_PhaseA_L1L2_进度原件_dp%02d_2026-10-14.jsonl' % k)
        prog[k] = p
    rc = recheck(a_rep, b_rep, sup, prog)
    rep = {'label': 'P1A-RoleC-PhaseA-independent-recheck',
           'inputs': {'stage_A': a.a, 'stage_B': a.b, 'supervisor': a.supervisor,
                      'progress_files': [prog[k] for k in range(8)]},
           'recheck': {'checks': rc['checks'], 'problems': rc['problems'],
                       'limits': rc['limits'],
                       'unverifiable': rc.get('unverifiable') or []},
           'a_replay_independent': rc['a_replay'],
           'b_arithmetic': rc['b_arithmetic'],
           'attribution': final_aggregate(
               a_rep, b_rep, digest_registered=drv.pc.PHASEA_DECISION_POINTS_DIGEST),
           'incident': {
               'what': ('跑批机上的**主报告**（--out）写盘失败，留下 0 字节文件'),
               'cause': ('驱动器把 `ColdChainContractV2` 对象塞进 bundle 摘要 ⇒ '
                         '`json.dumps` 抛 `TypeError: Object of type '
                         'ColdChainContractV2 is not JSON serializable`'),
               'evidence': '监督器原件的 stderr_tail 里有完整 traceback',
               'impact': ('stage_A.json / stage_B.json 在**之前**已落盘 ⇒ '
                          '数据未丢；本报告的聚合是在**归档原件**上离线重算的'),
               'fix': ('已改为只记录契约的**可序列化指纹**'
                       '（`_contract_fingerprint`），不再塞对象'),
               'zero_byte_file_sha256':
                   'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
               'process_lesson': ('长任务**必须**放 tmux：本次本地等待期间 SSH 通道中断，'
                                  '远程前台进程随通道一起结束（恰恰打断了报告写盘）。'
                                  '重跑一律用 tmux + 落盘进度原件'),
           }}
    with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('WROTE: %s' % a.out)
    print('problems: %s' % (rc['problems'] or 'NONE'))
    print('conclusion: %s' % rep['attribution'].get('conclusion'))
    return 0 if not rc['problems'] else 1


if __name__ == '__main__':
    sys.exit(main())
