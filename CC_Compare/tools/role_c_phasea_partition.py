#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的**互斥时间分区**工装：三通道独立计量 + 离线复算（REV2 §4.3）。

为什么需要独立的一层
-------------------
规格 §4.1 要求用**包装器**实现 exclusive/inclusive，且**不改生产代码**。本模块就是
那层包装器基础设施。它只做时间分区，**不碰**任何效用字段。

REV2 §4.3 定义了三条互不派生的通道：

  * **通道 A**（在线，exit 侧）：每叶 `A_i += (t_exit − t_enter) − child_time`；
  * **通道 B**（在线，enter 侧）：栈管理器的空档累加器 `G`；
  * **通道 C**（离线）：由落盘的原始事件表用**独立算法**重算。

⚠️ **用户 2026-10-14 指出的关键缺陷（本模块修的就是它）**
--------------------------------------------------------
REV2 初稿把 `G` 写成：

    G += t_enter_new − t_last_exit        # 进入任何新叶子时都累加

这在**嵌套**下会**重复计量**：父阶段（如 `P7d`）仍然活动时，其内部两个子叶子之间的
空档**既**属于父的 exclusive time，**又**被加进 `G` ⇒ `P7d` 与 `P9` 双计。

**正确语义（本模块实现）**：`G` **只**累计"**栈上没有任何被计量叶子活动**"的区间；
父阶段内部的剩余时间归**父的 exclusive**，**不得**同时进 `G`。实现上不需要
`t_last_exit`：只在 `活动数 由 0 变非 0` 与 `由 非 0 变 0` 的两个边界上收支。

⚠️ **关于 ③ 的诚实说明（不得当成独立证据）**
------------------------------------------
验收 ③ `|Σ A_i + G − T_decision| ≤ tol` 在**记账完全正确**时是**代数恒等**的
（Σ exclusive 恰等于"深度 1 区间之并"的测度，而 `G` 恰是其补集）。因此 ③ 的作用
**不是**独立测量，而是抓**实现层故障**：帧不平衡、异常路径漏 pop、多线程交错、
时钟非单调。**独立性来自 ① 与 ②**（通道 C 用另一套算法重算）。这一点必须如实
写进报告，不得宣称 ③ 提供了独立证据。

⚠️ 同一叶子**不得**在栈上同时出现两次（视为**重复插桩**）⇒ 记账错误。
"""
from __future__ import annotations

import contextlib

__all__ = [
    'LEAF_ORDER', 'UNATTRIBUTED', 'DEFAULT_TOL_FLOOR_S', 'DEFAULT_TOL_FRACTION',
    'PartitionAccountingError', 'PartitionTimer', 'offline_scan', 'reconcile',
    'reconciliation_tolerance', 'tick_clock', 'instrument_inclusive',
    'leaf_index', 'tie_break_leaf', 'measure_overhead',
]

#: 事前定义的互斥叶子顺序（**判定并列最大值时按此索引裁决**，REV2 §6.2）。
#: ⚠️ `P7` 不在其中：它被完全拆成 `P7a/P7b/P7c/P7d`，**不是**叶子。
LEAF_ORDER = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6',
              'P7a', 'P7b', 'P7c', 'P7d', 'P8')

#: 未归因分区（`P9_measured := G`）。它不是"被包装的叶子"，而是**空档累加器**。
UNATTRIBUTED = 'P9_measured'

#: 对账容差：`max(5 ms, 1% · T_decision)`（REV2 §4.3；**执行后不得调整**）。
DEFAULT_TOL_FLOOR_S = 5e-3
DEFAULT_TOL_FRACTION = 0.01

#: 用于比较的浮点零（仅内部判"是否同一时刻"，不参与验收容差）
_EPS = 1e-15

_LEAF_INDEX = {lab: i for i, lab in enumerate(LEAF_ORDER)}


def leaf_index(label):
    """叶子的**事前索引**（并列最大值按最小索引裁决，REV2 §6.2）。"""
    return _LEAF_INDEX.get(label)


def tie_break_leaf(labels):
    """并列最大时取**索引最小**的叶子（REV2 §6.2 事前写死）。"""
    cand = [l for l in labels if l in _LEAF_INDEX]
    if not cand:
        return None
    return min(cand, key=lambda l: _LEAF_INDEX[l])


class PartitionAccountingError(RuntimeError):
    """记账错误 ⇒ 本次决策判 `inconclusive_accounting_error`（fail-closed）。"""

    def __init__(self, code, detail=''):
        super(PartitionAccountingError, self).__init__('%s: %s' % (code, detail))
        self.code = code
        self.detail = detail


def tick_clock(step=1e-3, start=0.0):
    """**确定性时钟**：每次调用前进 `step` 秒（合成回归用；可逐位复现）。

    为什么需要：真实 `perf_counter` 下"对账是否通过"会随机器抖动，
    合成反例必须**逐位确定**，否则回归本身不可复现。
    """
    state = {'t': float(start)}

    def clock():
        v = state['t']
        state['t'] = v + float(step)
        return v
    clock.state = state
    return clock


class PartitionTimer(object):
    """一次决策的互斥分区计时器（通道 A + 通道 B）。

    用法::

        tm = PartitionTimer()
        tm.start_decision()
        with tm.leaf('P3'):
            ...
        with tm.leaf('P7d'):
            with tm.leaf('P7a'):
                ...
        tm.end_decision()
        online = tm.result()          # A_i / G / 事件表
        rec = reconcile(online)       # 加通道 C，给四条件验收

    `clock` 可注入（默认 `time.perf_counter`）；合成回归用 `tick_clock()`。

    ⚠️ **同一叶子在栈上出现两次 = 重复插桩 ⇒ 无条件记错**（不存在"合法递归"开关：
    叶子集合里没有任何一个叶子会递归调用自身；放宽它只会让一个真实的插桩 bug 静默过关）。
    """

    def __init__(self, clock=None, *, leaf_order=LEAF_ORDER, on_event=None):
        if clock is None:
            import time
            clock = time.perf_counter
        self.clock = clock
        self.leaf_order = tuple(leaf_order)
        #: `on_event(kind, label, t)`：可选的**进度钩子**（`enter`/`exit`）。
        #: 用途：A 类被外层监督器硬杀时，从最后一条钩子记录恢复**截断位置**
        #: （REV2 §5.1.2 要求，不得只写"超时"）。钩子必须**不抛**、**快**、
        #: 且**不得**改变分区语义（它只观察）。
        self.on_event = on_event
        self._stack = []                 # 活动帧：[label, t_enter, child_time]
        self._exclusive = {}
        self._inclusive = {}
        self._events = []                # (label, t_enter, t_exit) —— 通道 C 的输入
        self._raw = []                   # ('enter'|'exit', label, ts) —— 原始事件流
        self._g = 0.0                    # 通道 B：真正未覆盖区间
        self._free_since = None          # 活动数为 0 ⇒ 从该时刻起算未覆盖
        self._t_start = None
        self._t_end = None
        self._errors = []                # [(code, detail)]
        self._counts = {}
        self._finished = False

    # ------------------------------------------------------------------ 生命周期
    def start_decision(self):
        """开启计量分母 `T_decision`（规格 §3：`perf_counter` 环绕一次 `decide_fn`）。"""
        if self._t_start is not None:
            raise PartitionAccountingError('already_started', '同一次决策重复 start')
        self._t_start = self.clock()
        self._free_since = self._t_start     # 决策一开始即为"未覆盖"
        return self._t_start

    def end_decision(self):
        """关闭计量分母。若此时仍有活动帧（异常路径漏 pop）⇒ 记记账错误。"""
        if self._t_start is None:
            raise PartitionAccountingError('not_started', '未 start 就 end')
        if self._finished:
            raise PartitionAccountingError('already_finished', '重复 end_decision')
        t = self.clock()
        self._t_end = t
        if self._stack:
            self._note('frame_unbalanced_at_end',
                       '决策结束时仍有 %d 个活动帧：%s'
                       % (len(self._stack), [f[0] for f in self._stack]))
        elif self._free_since is not None:
            self._g += t - self._free_since
            self._free_since = None
        self._finished = True
        return t

    # ------------------------------------------------------------------ 叶子
    @contextlib.contextmanager
    def leaf(self, label):
        """把一段计算登记为叶子 `label`（**异常也会正确收帧**：`finally`）。"""
        self._enter(label)
        try:
            yield self
        finally:
            self._exit(label)

    def _enter(self, label):
        if label not in _LEAF_INDEX:
            raise PartitionAccountingError(
                'unknown_leaf', '%r 不是事前定义的叶子（%s）' % (label, LEAF_ORDER))
        t = self.clock()
        if not self._stack:
            # ⚠️ **只有活动数为 0 时**才把"到此刻为止"的时间收进 G。
            #    父阶段内部（栈非空）的空档归**父的 exclusive**，不进 G。
            if self._free_since is not None:
                self._g += t - self._free_since
            self._free_since = None
        elif label in [f[0] for f in self._stack]:
            detail = ('叶子 %s 已在栈上 %s ⇒ **重复插桩**（同一叶子不得同时在栈上出现两次）'
                      % (label, [f[0] for f in self._stack]))
            self._note('duplicate_instrumentation', detail)
        self._stack.append([label, t, 0.0])
        self._raw.append(('enter', label, t))
        self._counts[label] = self._counts.get(label, 0) + 1
        self._emit('enter', label, t)

    def _exit(self, label):
        t = self.clock()
        self._raw.append(('exit', label, t))
        self._emit('exit', label, t)
        if not self._stack:
            self._note('unbalanced_exit',
                       '叶子 %s 退出时活动栈为空（exit 无对应 enter）' % label)
            return
        frame = self._stack.pop()
        if frame[0] != label:
            self._note('lifo_violation',
                       '退出 %s 但栈顶是 %s ⇒ 破坏 LIFO' % (label, frame[0]))
            return
        dur = t - frame[1]
        self._inclusive[label] = self._inclusive.get(label, 0.0) + dur
        self._exclusive[label] = self._exclusive.get(label, 0.0) + (dur - frame[2])
        self._events.append((label, frame[1], t))
        if self._stack:
            self._stack[-1][2] += dur            # 记入**直接父**的 child_time
        else:
            self._free_since = t                 # 活动数回到 0 ⇒ 重新开始计未覆盖

    def _note(self, code, detail=''):
        self._errors.append((code, detail))

    def _emit(self, kind, label, t):
        """调进度钩子。**钩子异常一律吞掉**：观测不得改变被测行为。"""
        if self.on_event is None:
            return
        try:
            self.on_event(kind, label, t)
        except Exception:                                        # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 读数
    def result(self):
        """在线读数（通道 A + 通道 B）。**不含效用**。"""
        t_decision = None
        if self._t_start is not None and self._t_end is not None:
            t_decision = self._t_end - self._t_start
        return {
            't_decision': t_decision,
            'exclusive': dict(self._exclusive),
            'inclusive': dict(self._inclusive),
            'events': list(self._events),
            'raw_events': list(self._raw),
            'g': self._g,
            'unattributed': self._g,
            'leaf_counts': dict(self._counts),
            'errors': self._errors_as_dicts(),
            'leaf_order': list(self.leaf_order),
            'unattributed_label': UNATTRIBUTED,
            'note': ('在线通道 A（exit 侧 exclusive 累加器）与通道 B（活动数 0↔非 0 边界'
                     '上的空档累加器 G）。⚠️ G **只**在活动数为 0 时累加 ⇒ 父阶段内部'
                     '的空档归父的 exclusive，**不与 G 重复计量**。'),
        }

    def _errors_as_dicts(self):
        return [{'code': c, 'detail': d} for c, d in self._errors]

    @property
    def exclusive(self):
        return dict(self._exclusive)

    @property
    def g(self):
        return self._g

    @property
    def events(self):
        return list(self._events)

    @property
    def raw_events(self):
        return list(self._raw)

    @property
    def errors(self):
        return self._errors_as_dicts()


# ---------------------------------------------------------------------- 通道 C
def _check_raw_events(raw, problems):
    """原始事件流的不变量：enter/exit 配对 + 时间戳单调不降 + LIFO + 无重叠活动。"""
    stack = []
    last_ts = None
    for kind, lab, ts in raw or ():
        if last_ts is not None and ts < last_ts - _EPS:
            problems.append('timestamp_not_monotone:%s@%.9g' % (lab, ts))
        last_ts = ts
        if kind == 'enter':
            if lab in [x[0] for x in stack]:
                problems.append('duplicate_label_on_stack:%s' % lab)
            stack.append((lab, ts))
        elif kind == 'exit':
            if not stack:
                problems.append('unmatched_exit:%s' % lab)
            else:
                top = stack.pop()
                if top[0] != lab:
                    problems.append('lifo_violation:exit=%s top=%s' % (lab, top[0]))
        else:
            problems.append('unknown_event_kind:%r' % (kind,))
    if stack:
        problems.append('unmatched_enter:%s' % [x[0] for x in stack])


def offline_scan(events, t_decision, *, leaf_order=LEAF_ORDER, raw_events=None):
    """**通道 C**：由原始事件表用**独立算法**重算 `Σ A_i` 与 `G`（REV2 §4.3）。

    与在线累加器**互不派生**：

      * 在线 exclusive 是"退出时减 child_time"的**流式**累加；
      * 本函数先按 `enter_ts` 重排、再用**栈重放**建立父子关系，
        用"逐事件时长减去**直接子**时长"给出 `A_i^offline`；
      * `G^offline` 用"深度 1（极大）区间**并的测度**"从 `T_decision` 里取补集
        —— 这是一个**几何/区间**算法，不是流式累加。

    结构不变量（任一条不通过 ⇒ `inconclusive_accounting_error`）：
      LIFO / enter-exit 配对 / 时间戳单调不降 / 同一叶子不得同时在栈上 /
      子区间必须落在父区间内 / 极大区间两两不相交。

    返回 `{'ok', 'problems', 'exclusive', 'g', 'n_events', 'n_maximal', ...}`。
    """
    problems = []
    if raw_events is not None:
        _check_raw_events(raw_events, problems)

    ev = list(events or [])
    for lab, a, b in ev:
        if b < a - _EPS:
            problems.append('interval_negative:%s' % lab)
        if lab not in _LEAF_INDEX:
            problems.append('unknown_leaf:%s' % lab)

    # 按 enter_ts 升序重排（合法的嵌套下，父的 enter 必早于子）。
    # 同一 enter 时刻并列时，长区间（exit 更晚）优先 ⇒ 父先于子。
    order = sorted(range(len(ev)), key=lambda i: (ev[i][1], -ev[i][2], i))

    stack = []
    children = {}
    maximal = []
    for i in order:
        lab, a, b = ev[i]
        while stack and ev[stack[-1]][2] <= a + _EPS:
            stack.pop()
        if stack:
            parent = stack[-1]
            pa, pb = ev[parent][1], ev[parent][2]
            if not (a >= pa - _EPS and b <= pb + _EPS):
                problems.append('not_contained_in_parent:%s(parent=%s)'
                                % (lab, ev[parent][0]))
            children.setdefault(parent, []).append(i)
        else:
            maximal.append(i)
        if any(ev[k][0] == lab for k in stack):
            problems.append('duplicate_label_on_stack:%s' % lab)
        stack.append(i)

    segs = sorted((ev[i][1], ev[i][2]) for i in maximal)
    for k in range(1, len(segs)):
        if segs[k][0] < segs[k - 1][1] - _EPS:
            problems.append('maximal_intervals_overlap:%d' % k)

    excl = {}
    for i, (lab, a, b) in enumerate(ev):
        own = (b - a) - sum(ev[j][2] - ev[j][1] for j in children.get(i, ()))
        excl[lab] = excl.get(lab, 0.0) + own
    covered = sum(b - a for a, b in segs)
    g = None if t_decision is None else float(t_decision) - covered

    return {'ok': not problems, 'problems': problems, 'exclusive': excl, 'g': g,
            'n_events': len(ev), 'n_maximal': len(maximal), 'covered_s': covered,
            'maximal_intervals': [[ev[i][0], ev[i][1], ev[i][2]] for i in maximal],
            'leaf_order': list(leaf_order),
            'algorithm': ('按 enter_ts 栈重放 + 直接子区间扣除（exclusive）；'
                          '极大区间并的补集（G）—— **与在线流式累加器互不派生**')}


def reconciliation_tolerance(t_decision, *, floor_s=DEFAULT_TOL_FLOOR_S,
                             fraction=DEFAULT_TOL_FRACTION):
    """对账容差 `max(5 ms, 1% · T_decision)`（REV2 §4.3；**执行后不得调整**）。"""
    if t_decision is None:
        return float(floor_s)
    return max(float(floor_s), float(fraction) * abs(float(t_decision)))


def reconcile(online, *, t_decision=None, floor_s=DEFAULT_TOL_FLOOR_S,
              fraction=DEFAULT_TOL_FRACTION, leaf_order=LEAF_ORDER):
    """四条件验收（REV2 §4.3）：把通道 A/B 与通道 C 对账。

    ① `|Σ A_i − Σ A_i^offline| ≤ tol`（通道 A vs C，不同算法）
    ② `|G − G^offline| ≤ tol`（通道 B vs C）
    ③ `|Σ A_i + G − T_decision| ≤ tol`（**实现层一致性**；见模块头的诚实说明）
    ④ 结构不变量（LIFO / 配对 / 单调 / 包含 / 同叶不重复 / 极大区间不相交）

    返回 `{'verdict', 'conditions', ...}`；`verdict ∈ {'ok',
    'inconclusive_accounting_error'}`（fail-closed）。
    """
    td = online.get('t_decision') if t_decision is None else t_decision
    tol = reconciliation_tolerance(td, floor_s=floor_s, fraction=fraction)
    off = offline_scan(online.get('events') or [], td, leaf_order=leaf_order,
                       raw_events=online.get('raw_events'))
    a_sum = sum(float(v) for v in (online.get('exclusive') or {}).values())
    a_off = sum(float(v) for v in (off['exclusive'] or {}).values())
    g_on = online.get('g')
    g_off = off['g']
    d1 = abs(a_sum - a_off)
    d2 = None if (g_on is None or g_off is None) else abs(float(g_on) - float(g_off))
    d3 = None if td is None else abs(a_sum + float(g_on or 0.0) - float(td))
    on_err = online.get('errors') or []
    cond = {
        '1_exclusive_vs_offline': {'pass': d1 <= tol, 'abs_gap_s': d1},
        '2_unattributed_vs_offline': {'pass': (d2 is not None) and d2 <= tol,
                                      'abs_gap_s': d2},
        '3_closure_consistency': {'pass': (d3 is not None) and d3 <= tol,
                                  'abs_gap_s': d3},
        '4_structure_invariants': {'pass': bool(off['ok']) and not on_err,
                                   'offline_problems': off['problems'],
                                   'online_errors': on_err},
    }
    failed = [k for k, v in cond.items() if not v['pass']]
    return {
        'verdict': 'ok' if not failed else 'inconclusive_accounting_error',
        'failed_conditions': failed,
        'conditions': cond,
        'tolerance_s': tol,
        'tolerance_rule': ('max(%.0e s, %.4g · T_decision) —— 事前写死，'
                           '执行后不得调整' % (floor_s, fraction)),
        't_decision': td,
        'sum_exclusive_online': a_sum,
        'sum_exclusive_offline': a_off,
        'unattributed_online': g_on,
        'unattributed_offline': g_off,
        # `P9_residual`：**对账误差**，正常应接近零。⚠️ 它**不等于** `P9_measured`(=G)。
        'p9_residual': d3,
        'p9_measured': g_on,
        'shares': _shares(online.get('exclusive') or {}, g_on, td, leaf_order),
        'offline': off,
        'honesty_note': ('条件 ③ 在记账完全正确时是代数恒等式 ⇒ 它抓的是**实现层故障**'
                         '（帧不平衡 / 异常路径漏 pop / 多线程交错 / 时钟非单调），'
                         '**不是**独立证据；独立性来自条件 ①②（通道 C 用另一套算法）。'),
    }


def _shares(exclusive, g, t_decision, leaf_order):
    """互斥份额：分母**唯一**是 `T_decision`（规格 §3），`P9_measured` 也占一份。"""
    if not t_decision:
        return {}
    out = {}
    for lab in leaf_order:
        out[lab] = float(exclusive.get(lab, 0.0)) / float(t_decision)
    out[UNATTRIBUTED] = float(g or 0.0) / float(t_decision)
    return out


def measure_overhead(bare_fn, wrapped_fn=None, *, repeats=25, warmup=3, clock=None):
    """**仪器自开销**（规格 §3 明文要求）：先跑"空仪器"，再跑挂满分区的版本。

    `bare_fn`  = 只挂总计时（空仪器）；`wrapped_fn` = 挂上全部分区叶子。
    若 `wrapped_fn` 为 None，则只报 `bare_fn` 的耗时（并明确标注"未给挂满版本"）。

    返回 `{'bare_s', 'wrapped_s', 'overhead_s', 'overhead_ratio', 'n', 'note'}`。
    ⚠️ 开销是**诊断量**，**不得**用它去改 §4.3 的容差（容差事前写死）。
    """
    clk = clock or __import__('time').perf_counter

    def _run(fn):
        for _ in range(int(warmup)):
            fn()
        xs = []
        for _ in range(int(repeats)):
            t0 = clk()
            fn()
            xs.append(clk() - t0)
        xs.sort()
        n = len(xs)
        return {'n': n, 'mean_s': sum(xs) / n if n else None,
                'p50_s': xs[n // 2] if n else None, 'max_s': xs[-1] if n else None}

    bare = _run(bare_fn)
    out = {'bare_s': bare, 'wrapped_s': None, 'overhead_s': None,
           'overhead_ratio': None, 'warmup': int(warmup), 'repeats': int(repeats),
           'note': ('"空仪器"自开销：只挂总计时、不挂分区。开销只作诊断，'
                    '**不得**据此调整 §4.3 对账容差。')}
    if wrapped_fn is not None:
        wrapped = _run(wrapped_fn)
        out['wrapped_s'] = wrapped
        if bare['p50_s']:
            out['overhead_s'] = wrapped['p50_s'] - bare['p50_s']
            out['overhead_ratio'] = out['overhead_s'] / bare['p50_s']
    return out


def instrument_inclusive(events, leaf_order=LEAF_ORDER):
    """由事件表算 **inclusive**（定位热点用；⚠️ 同级 inclusive **可重叠**，
    **不得**用于份额分解或 50% 判据）。"""
    inc = {}
    for lab, a, b in events or ():
        inc[lab] = inc.get(lab, 0.0) + (b - a)
    return {lab: inc.get(lab, 0.0) for lab in leaf_order if lab in inc}
