#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 计时工装的**合成反例回归**（用户 2026-10-14 前置门之二）。

为什么不能只测"正常路径"
-----------------------
文档检查器（`_check_phasea_registration_doc.py`）只能证明**文档**结构正确，
**不能**证明时间分区实现正确。用户明确要求：构造**嵌套 / 顺序 / 异常退出 / 重入**
四类反例，验证

    T_decision ≈ Σ_i T_i^exclusive + G

并通过**独立区间扫描**复算。**回归失败时不得通过放宽 `max(5 ms, 1%)` 或重定义
P9 来获得通过。**

本文件的核心是 `test_p7d_gap_not_double_counted`：它把用户 2026-10-14 指出的
"父阶段内部空档被同时计进 `P9`"的缺陷做成**可执行的回归**——同一夹具下，
被废弃的 REV2 初稿规则算出的 `G` **必然大于**修正后的 `G`，且差值恰好等于
父阶段内部空档。修正后若有人把 `G += t_enter − t_last_exit` 写回去，本测试立刻失败。

运行：`python CC_Compare/tools/test_role_c_phasea_partition.py`
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_partition as pp                              # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- 反例辅助
def _rev2_draft_g(raw_events, t_start, t_end):
    """**已被取代的** REV2 初稿 `G` 规则（只作回归见证，**不得**用于测量）。

        G += t_enter_new − t_last_exit        # 进入任何新叶子时都累加

    该规则在**嵌套**下把父阶段内部的空档同时计进 `G` ⇒ `P7d` 与 `P9` 双计。
    本函数存在的唯一目的是证明"修正确实改变了结果"，防止有人改回去。
    """
    g = 0.0
    t_last_exit = t_start
    for kind, _lab, ts in raw_events:
        if kind == 'enter':
            if ts > t_last_exit:
                g += ts - t_last_exit
                t_last_exit = ts
        else:
            t_last_exit = ts
    if t_end > t_last_exit:
        g += t_end - t_last_exit
    return g


# --------------------------------------------------------------------------- 1. 顺序
def test_sequential_leaves():
    """顺序（无嵌套）：两个顶层叶子之间与两端的空档**全部**进 G。"""
    print('\n[1] 顺序反例：无嵌套，空档全部属 G')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P3'):
        pass
    with tm.leaf('P5'):
        pass
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)
    check('顺序：四条件全过', rec['verdict'] == 'ok', rec['failed_conditions'])
    check('顺序：Σexcl = 2 ms', _close(rec['sum_exclusive_online'], 2e-3),
          rec['sum_exclusive_online'])
    check('顺序：G = 3 ms（首尾+中间空档）', _close(on['g'], 3e-3), on['g'])
    check('顺序：T = Σexcl + G', _close(on['t_decision'], 5e-3), on['t_decision'])
    return True


# ------------------------------------------------------- 2. 嵌套（关键回归：P7d 空档）
def test_p7d_gap_not_double_counted():
    """**用户指出的缺陷回归**：父阶段（P7d）内部空档**只能**归 P7d 的 exclusive。

    夹具：`P7d{ P7a ; <2 ms 未插桩空档> ; P7b }`，外层另有 1 ms 前导与 1 ms 尾部空档。
    正确 `G` = 外层 2 ms；初稿规则会算成 7 ms（多算的 5 ms 正是 P7d 内部空档）。
    """
    print('\n[2] 嵌套反例：父(P7d)内部空档不得进 G（用户指出的双计缺陷）')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P7d'):
        with tm.leaf('P7a'):
            pass
        tm.clock()                    # 未插桩工作片段
        tm.clock()
        with tm.leaf('P7b'):
            pass
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)

    # 期望读数（tick_clock 1 ms/次，逐位可复现）
    #   P7d: enter 0.001 → exit 0.008（dur 7 ms）
    #   P7a: 2→3 ms，P7b: 6→7 ms ⇒ child_time 2 ms ⇒ excl(P7d) = 5 ms
    #   外层空档：0→1 ms 与 8→9 ms ⇒ G = 2 ms
    check('嵌套：四条件全过', rec['verdict'] == 'ok', rec['failed_conditions'])
    check('嵌套：excl(P7a)=1 ms', _close(on['exclusive'].get('P7a', 0), 1e-3),
          on['exclusive'].get('P7a'))
    check('嵌套：excl(P7b)=1 ms', _close(on['exclusive'].get('P7b', 0), 1e-3),
          on['exclusive'].get('P7b'))
    check('嵌套：**excl(P7d)=5 ms（含内部 5 ms 空档）**',
          _close(on['exclusive'].get('P7d', 0), 5e-3),
          on['exclusive'].get('P7d'))
    check('嵌套：**G 只含外层 2 ms**', _close(on['g'], 2e-3), on['g'])
    check('嵌套：T = Σexcl + G = 9 ms', _close(on['t_decision'], 9e-3),
          on['t_decision'])

    draft = _rev2_draft_g(on['raw_events'], 0.0, on['t_decision'])
    total_excl = sum(on['exclusive'].values())
    check('嵌套：**初稿规则会多算**（draft G=6 ms > 修正 G=2 ms）',
          draft > on['g'] + 1e-12 and _close(draft, 6e-3), draft)
    # 双计的确切症状：初稿规则下 Σ分区 + G **超过** T_decision ⇒ 条件 ③ 必然失败
    check('嵌套：初稿规则破坏闭合（Σexcl 7 ms + draft G 6 ms = 13 ms > T 9 ms）',
          _close(total_excl + draft, 13e-3) and total_excl + draft > on['t_decision'],
          (total_excl, draft, on['t_decision']))
    check('嵌套：修正后恰好闭合（7 ms + 2 ms = 9 ms）',
          _close(total_excl + on['g'], on['t_decision']),
          (total_excl, on['g'], on['t_decision']))
    # `P9_measured` 是 G；`P9_residual` 是对账误差，**二者不应相等**
    check('嵌套：P9_measured(=G) ≠ P9_residual，且 |residual| ≤ tol',
          rec['p9_residual'] is not None and rec['p9_residual'] <= rec['tolerance_s']
          and not _close(rec['p9_measured'], rec['p9_residual'] or 0.0),
          (rec['p9_measured'], rec['p9_residual']))
    return True


# --------------------------------------------------------------------- 3. 异常退出
def test_exception_exit_closes_frame():
    """异常退出：叶子抛错时帧必须**仍然被收掉**，记账不得失衡。"""
    print('\n[3] 异常退出反例：抛错也要收帧，记账不失衡')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    raised = False
    try:
        with tm.leaf('P8'):
            raise ValueError('synthetic leaf failure')
    except ValueError:
        raised = True
    with tm.leaf('P5'):
        pass
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)
    check('异常：异常仍然向上抛出（不被吞）', raised)
    check('异常：活动栈已清空（无 frame_unbalanced）',
          not [e for e in on['errors'] if e['code'] == 'frame_unbalanced_at_end'],
          on['errors'])
    check('异常：P8 区间仍然闭合且被计入',
          'P8' in dict((l, 1) for l, _, _ in on['events']), on['events'])
    check('异常：四条件全过', rec['verdict'] == 'ok', rec['failed_conditions'])
    check('异常：T = Σexcl + G', _close(rec['sum_exclusive_online'] + on['g'],
                                       on['t_decision']),
          (rec['sum_exclusive_online'], on['g'], on['t_decision']))
    return True


# ------------------------------------------------------------------------ 4. 重入
def test_duplicate_instrumentation_detected():
    """重入（同一叶子在栈上出现两次）= 重复插桩 ⇒ 必须判记账错误（fail-closed）。"""
    print('\n[4] 重入反例：同一叶子重复进栈 ⇒ 判 inconclusive_accounting_error')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P7d'):
        with tm.leaf('P7d'):
            pass
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)
    codes = [e['code'] for e in on['errors']]
    check('重入：在线侧报 duplicate_instrumentation',
          'duplicate_instrumentation' in codes, codes)
    check('重入：离线侧报 duplicate_label_on_stack',
          any('duplicate_label_on_stack' in p
              for p in rec['conditions']['4_structure_invariants']
              ['offline_problems']),
          rec['conditions']['4_structure_invariants']['offline_problems'])
    check('重入：verdict = inconclusive_accounting_error',
          rec['verdict'] == 'inconclusive_accounting_error', rec['verdict'])
    check('重入：条件 ④ 不通过',
          not rec['conditions']['4_structure_invariants']['pass'])
    return True


def test_duplicate_instrumentation_closure_still_holds():
    """重复插桩下 **③ 仍然成立** —— 这正是"③ 不能单独当证据"的实证，靠 ④ 才能抓住。

    这条测试是模块头那句诚实说明的**可执行版本**：条件 ③ 在记账"自洽"时是代数恒等式，
    所以一个真实的插桩 bug（同一叶子装两次）**不会**被 ③ 抓出来，
    必须靠 ④ 的 `duplicate_label_on_stack` / 同叶不重复不变量。
    """
    print('\n[4b] 重复插桩：③ 照样闭合 ⇒ 证明必须依赖 ④（诚实说明的实证）')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P7d'):
        with tm.leaf('P7d'):
            pass
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)
    check('重复插桩：Σexcl = 3 ms（自身仍自洽，未"多算"）',
          _close(sum(on['exclusive'].values()), 3e-3), on['exclusive'])
    check('重复插桩：条件 ③ **通过**（closure 9→5 ms 恒等成立）',
          rec['conditions']['3_closure_consistency']['pass'],
          rec['conditions']['3_closure_consistency'])
    check('重复插桩：条件 ④ **不通过** ⇒ 唯一拦住它的是 ④',
          not rec['conditions']['4_structure_invariants']['pass'])
    check('重复插桩：verdict 仍为 inconclusive_accounting_error',
          rec['verdict'] == 'inconclusive_accounting_error', rec['verdict'])
    # inclusive 因重复插桩虚高：再次证明 inclusive 不得用于份额
    check('重复插桩：inclusive(P7d) 虚高（= 4 ms > exclusive 3 ms）⇒ inclusive 不可用于份额',
          _close(on['inclusive'].get('P7d', 0), 4e-3), on['inclusive'].get('P7d'))
    return True


# ------------------------------------------------------------- 5. 独立区间扫描复算
def test_offline_scan_independent_recompute():
    """通道 C 必须**独立**重算并与在线通道一致（含"事件按 exit 序落盘"的情形）。"""
    print('\n[5] 离线区间扫描：独立算法复算，处理乱序事件表')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P7d'):
        with tm.leaf('P7a'):
            pass
        tm.clock()
        with tm.leaf('P7c'):
            pass
    with tm.leaf('P8'):
        pass
    tm.end_decision()
    on = tm.result()
    off = pp.offline_scan(on['events'], on['t_decision'],
                          raw_events=on['raw_events'])
    check('离线：结构不变量全过', off['ok'], off['problems'])
    check('离线：G 与在线一致', _close(off['g'], on['g']),
          (off['g'], on['g']))
    check('离线：Σexcl 与在线一致',
          _close(sum(off['exclusive'].values()), sum(on['exclusive'].values())),
          (sum(off['exclusive'].values()), sum(on['exclusive'].values())))
    # 事件表是**按 exit 顺序**落盘的（子先于父），离线必须能自己重排
    enters = [e[1] for e in on['events']]
    check('离线：输入事件表确实非 enter 升序（证明重排逻辑被走到）',
          enters != sorted(enters), enters)
    check('离线：用到的是与在线不同的算法',
          '互不派生' in off['algorithm'], off['algorithm'])
    return True


# --------------------------------------------------------------- 6. 结构不变量检测
def test_offline_detects_unbalanced_and_overlap():
    """离线扫描必须抓到"未配对 / LIFO 违约 / 区间相交 / 越界包含"。"""
    print('\n[6] 结构不变量：坏账必须被抓到')
    bad_raw = [('enter', 'P1', 0.0), ('enter', 'P2', 1.0), ('exit', 'P1', 2.0)]
    off = pp.offline_scan([('P1', 0.0, 2.0)], 3.0, raw_events=bad_raw)
    check('坏账：LIFO 违约被抓到',
          any('lifo_violation' in p for p in off['problems']), off['problems'])

    off2 = pp.offline_scan([('P1', 0.0, 2.0), ('P2', 1.0, 3.0)], 3.0)
    check('坏账：相交的极大区间被抓到',
          any('overlap' in p or 'not_contained' in p for p in off2['problems']),
          off2['problems'])

    off3 = pp.offline_scan([('P1', 0.0, 1.0)], 2.0,
                           raw_events=[('exit', 'P1', 1.0)])
    check('坏账：无 enter 的 exit 被抓到',
          any('unmatched_exit' in p for p in off3['problems']), off3['problems'])

    off4 = pp.offline_scan([('P1', 0.0, 1.0)], 2.0,
                           raw_events=[('enter', 'P1', 0.0)])
    check('坏账：未闭合 enter 被抓到',
          any('unmatched_enter' in p for p in off4['problems']), off4['problems'])
    return True


def test_online_detects_unbalanced_exit_and_lifo():
    """在线侧：exit 无对应 enter / LIFO 违约 ⇒ 记错。"""
    print('\n[7] 在线侧坏账：unbalanced_exit 与 lifo_violation')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    tm._exit('P5')                     # 故意：无 enter 的 exit
    with tm.leaf('P8'):
        tm._exit('P5')                 # 故意：栈顶是 P8 却退 P5
    tm.end_decision()
    on = tm.result()
    rec = pp.reconcile(on)
    codes = [e['code'] for e in on['errors']]
    check('在线坏账：unbalanced_exit 被抓到', 'unbalanced_exit' in codes, codes)
    check('在线坏账：lifo_violation 被抓到', 'lifo_violation' in codes, codes)
    check('在线坏账：verdict = inconclusive_accounting_error',
          rec['verdict'] == 'inconclusive_accounting_error', rec['verdict'])
    return True


# ------------------------------------------------------------ 8. 容差与份额语义
def test_tolerance_rule_and_shares():
    """容差**事前写死**为 `max(5 ms, 1%)`；份额分母**唯一**是 T_decision。"""
    print('\n[8] 容差规则与份额分母')
    check('容差：小 T 时取下限 5 ms',
          _close(pp.reconciliation_tolerance(0.100), 5e-3),
          pp.reconciliation_tolerance(0.100))
    check('容差：大 T 时取 1%',
          _close(pp.reconciliation_tolerance(10.0), 0.1),
          pp.reconciliation_tolerance(10.0))
    check('容差：None ⇒ 下限', _close(pp.reconciliation_tolerance(None), 5e-3))

    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P3'):
        pass
    tm.end_decision()
    rec = pp.reconcile(tm.result())
    s = rec['shares']
    check('份额：含 P9_measured 项', pp.UNATTRIBUTED in s, sorted(s))
    check('份额：总和 ≈ 1（互斥分区 + 未归因）', _close(sum(s.values()), 1.0, 1e-9),
          sum(s.values()))
    check('份额：分母唯一 = T_decision（P3 = 1 ms / 3 ms）',
          _close(s['P3'], 1e-3 / 3e-3, 1e-9), s['P3'])
    return True


def test_tie_break_by_lowest_leaf_index():
    """并列最大值按**最小叶子索引**裁决（REV2 §6.2 事前写死）。"""
    print('\n[9] 并列最大值裁决 = 最小叶子索引')
    check('裁决：P7a 与 P8 并列 ⇒ P7a', pp.tie_break_leaf(['P8', 'P7a']) == 'P7a')
    check('裁决：P3 与 P1 并列 ⇒ P1', pp.tie_break_leaf(['P3', 'P1']) == 'P1')
    check('裁决：无候选 ⇒ None', pp.tie_break_leaf(['P7', 'zzz']) is None)
    check('裁决：P7 不是叶子（无索引）', pp.leaf_index('P7') is None)
    return True


# ------------------------------------------------------------- 10. 仪器开销（自开销）
def test_overhead_reports_empty_instrument():
    """必须能测"空仪器"自开销（规格 §3），且**不**据此调整容差。"""
    print('\n[10] 仪器自开销：空仪器 vs 挂满分区')

    def bare():
        tm = pp.PartitionTimer()
        tm.start_decision()
        s = 0
        for i in range(200):
            s += i
        tm.end_decision()
        return tm.g + s * 1e-9

    def wrapped():
        tm = pp.PartitionTimer()
        tm.start_decision()
        with tm.leaf('P3'):
            s = 0
            for i in range(200):
                s += i
        with tm.leaf('P5'):
            pass
        tm.end_decision()
        return s * 1e-9

    rep = pp.measure_overhead(bare, wrapped, repeats=15, warmup=3)
    check('开销：报了 bare', rep['bare_s'] and rep['bare_s']['n'] == 15, rep['bare_s'])
    check('开销：报了 wrapped', rep['wrapped_s'] is not None, rep['wrapped_s'])
    check('开销：给了 overhead_s / ratio（可正可负，如实报）',
          rep['overhead_s'] is not None and rep['overhead_ratio'] is not None,
          (rep['overhead_s'], rep['overhead_ratio']))
    check('开销：明确写"不得据此调容差"', '不得' in rep['note'], rep['note'])
    return True


# ------------------------------------------------------------------ 11. 随机嵌套
def test_random_nested_closure_identity():
    """随机（但定种子）嵌套序列：③ 恒等必须成立、①② 必须一致、份额和为 1。"""
    print('\n[11] 随机嵌套：恒等与独立复算一致')
    import random
    rng = random.Random(20261014)
    leaves = list(pp.LEAF_ORDER)
    for trial in range(30):
        tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
        tm.start_decision()

        def body(depth):
            if depth > 3:
                return
            for _ in range(rng.randint(0, 3)):
                tm.clock()                     # 未插桩空档
                lab = rng.choice(leaves)
                if lab in [f[0] for f in tm._stack]:
                    continue
                with tm.leaf(lab):
                    body(depth + 1)
        body(0)
        tm.end_decision()
        on = tm.result()
        rec = pp.reconcile(on)
        if rec['verdict'] != 'ok':
            check('随机嵌套 trial=%d 四条件' % trial, False,
                  rec['failed_conditions'])
            return False
        if not _close(sum(rec['shares'].values()), 1.0, 1e-9):
            check('随机嵌套 trial=%d 份额和=1' % trial, False,
                  sum(rec['shares'].values()))
            return False
    check('随机嵌套 30 次：全过 + 份额和=1', True)
    return True


# --------------------------------------------------------------------- 11b. 一致性
def test_shares_are_mutually_exclusive():
    """份额是**互斥**的：Σ_i share_i + share(P9) = 1，且不含父节点重复计入。"""
    print('\n[12] 互斥性：父(P7d)与其子(P7a/P7b)不得同时出现在份额里')
    tm = pp.PartitionTimer(clock=pp.tick_clock(1e-3))
    tm.start_decision()
    with tm.leaf('P7d'):
        with tm.leaf('P7a'):
            pass
        tm.clock()
        with tm.leaf('P7b'):
            pass
    tm.end_decision()
    on = tm.result()
    check('互斥：P7d 的 exclusive **已扣除**子区间 ⇒ 与子相加不超 1',
          _close(sum(on['exclusive'].values()) + on['g'], on['t_decision']),
          (sum(on['exclusive'].values()), on['g'], on['t_decision']))
    inc = pp.instrument_inclusive(on['events'])
    check('互斥：inclusive(P7d)=6 ms > exclusive(P7d)=4 ms（inclusive 不可用于份额）',
          _close(inc['P7d'], 6e-3) and _close(on['exclusive']['P7d'], 4e-3),
          (inc, on['exclusive']))
    return True


def main():
    print('=' * 78)
    print('Phase A 计时工装合成反例回归（用户 2026-10-14 前置门之二）')
    print('=' * 78)
    test_sequential_leaves()
    test_p7d_gap_not_double_counted()
    test_exception_exit_closes_frame()
    test_duplicate_instrumentation_detected()
    test_duplicate_instrumentation_closure_still_holds()
    test_offline_scan_independent_recompute()
    test_offline_detects_unbalanced_and_overlap()
    test_online_detects_unbalanced_exit_and_lifo()
    test_tolerance_rule_and_shares()
    test_tie_break_by_lowest_leaf_index()
    test_overhead_reports_empty_instrument()
    test_random_nested_closure_identity()
    test_shares_are_mutually_exclusive()
    print('\n' + '=' * 78)
    if FAILS:
        print('REGRESSION_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('REGRESSION_PASS: 全部合成反例通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
