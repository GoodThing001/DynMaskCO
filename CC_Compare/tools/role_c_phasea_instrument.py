#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的**叶子插桩层**：把 §4.2 的互斥分区映射到真实调用点（不改生产代码）。

依据与边界
---------
* 规格 §4.1 要求 exclusive/inclusive 用**包装器**实现，且**不改生产代码**
  ⇒ 本模块用"打补丁 + 退出时**逐字节还原**"的方式，把被测模块的若干模块级名字
  换成计数/计时包装器。
* 映射**逐条对应** REV2 §4.2 表，**不新增、不改名、不合并**：

  | 叶子 | 绑定的真实调用点 |
  |---|---|
  | `P1` | `role_c_local_search.state_from_vehicles` |
  | `P2` | `role_c_harness.sample_scenarios` |
  | `P3` | `role_c_harness.build_space_bundle` |
  | `P4` | `role_c_harness.make_order_energy_fn` / `make_order_revenue_fn` / `make_order_penalty_fn` / `make_c0_complete_fn` 的**构造** |
  | `P5` | **前瞻之外**的 `ls.best_insert` + `ls.improve`（即"当前订单"的搜索） |
  | `P6` | `role_c_harness._insertion_fuel`（含其整算全场总距离） |
  | `P7a` | `forward_value` 内的 `ls.best_insert` |
  | `P7b` | `forward_value` 内的 `ls.improve` |
  | `P7c` | `forward_value` 内的 `complete_fn` |
  | `P7d` | `forward_value` 自身（场景循环/记账/排序等 P7 的剩余部分） |
  | `P8` | `certify_fn`（真实 `certify_plan`）—— 由调用方注入包装好的 `certify_plan_fn` |

* ⚠️ **`P6` 是"砍点"**：`_insertion_fuel` 内部还会调 `ls.best_insert`（harness:609），
  按 §4.2 那属于 `P6`"含其整算全场总距离"的工作量，**不得**再计一次 `P5`
  （否则父与子同时计入份额，破坏互斥）。⇒ 进入 `P6` 时**抑制**内层叶子标注。
* ⚠️ **`P5` / `P7a` 用同一个函数**（`ls.best_insert`）；用 `in_forward` 深度区分，
  **不是**靠调用栈猜测。
* ⚠️ **L1 计数只作解释证据**（规格 §5.3）：`route_feasible` / `route_dist` 等**只计数，
  不标注叶子**；**不得**参与主导判定。

不做的事
--------
* 不修改任何被测模块的**字节**（`_PATCH_TARGETS` 只替换模块属性，退出时还原）；
* 不产生任何效用字段（本层只记时间与计数）。
"""
from __future__ import annotations

import functools

import role_c_phasea_partition as pp

__all__ = ['LEAF_BINDINGS', 'SUPPRESSING_LEAVES', 'InstrumentationError',
           'LeafInstrumentation', 'patch_table', 'active_leaf_from_progress',
           'leaves_entered', 'measure_progress_overhead', 'certify_proxy',
           'active_session']

#: 当前生效的插桩会话（单线程顺序跑决策 ⇒ 不需要线程局部）。
#: 用途：`certify_proxy` 在**构建期**就被注入 `build_decide_fn`，运行期它要找到
#: 当下的 `PartitionTimer`。用模块级"活动会话"而不是闭包捕获，
#: 是因为 `decide_fn` 的构建早于任何一次决策。
_ACTIVE = None
_REAL_CERTIFY = None


def active_session():
    return _ACTIVE


def _real_certify():
    """懒加载真实 `certify_plan`（不依赖 import 期就有项目路径）。"""
    global _REAL_CERTIFY
    if _REAL_CERTIFY is None:
        try:
            import role_c_t_series as ts
            ts.ensure_project_paths()
        except Exception:                                        # noqa: BLE001
            pass
        from run_exp_energy_c0 import certify_plan
        _REAL_CERTIFY = certify_plan
    return _REAL_CERTIFY


def certify_proxy(*a, **kw):
    """`P8` 的注入点：在**构建期**传给 `build_decide_fn(certify_plan_fn=...)`。

    为什么走"显式注入"而不是替换模块属性：`build_decide_fn` 在缺省时才从
    `run_exp_energy_c0` 导入 `certify_plan`；显式传入是公开参数，比替换模块属性
    更干净、也更难出错。没有活动会话时**原样调用**真实函数（构建期跑一天就是这种情况）。
    """
    fn = _real_certify()
    if _ACTIVE is None:
        return fn(*a, **kw)
    _ACTIVE.l1['certify_calls'] = _ACTIVE.l1.get('certify_calls', 0) + 1
    with _ACTIVE.timer.leaf('P8'):
        return fn(*a, **kw)

#: 叶子 → 绑定的真实调用点（人类可读，供报告与审计）
LEAF_BINDINGS = {
    'P1': 'role_c_local_search.state_from_vehicles',
    'P2': 'role_c_harness.sample_scenarios',
    'P3': 'role_c_harness.build_space_bundle',
    'P4': ('role_c_harness.make_order_energy_fn / make_order_revenue_fn / '
           'make_order_penalty_fn / make_c0_complete_fn（构造）'),
    'P5': 'ls.best_insert + ls.improve（前瞻之外，当前订单）',
    'P6': 'role_c_harness._insertion_fuel（含整算全场总距离）',
    'P7a': 'forward_value 内的 ls.best_insert',
    'P7b': 'forward_value 内的 ls.improve',
    'P7c': 'forward_value 内的 complete_fn',
    'P7d': 'role_c_forward.forward_value 自身（场景循环/记账/排序）',
    'P8': 'certify_fn（真实 certify_plan；由调用方注入包装）',
}

#: 进入这些叶子时**抑制**内层叶子标注（父/子不得同时计入份额）
SUPPRESSING_LEAVES = ('P6',)

#: L1 只计数、不标注叶子的调用点（规格 §5.3：只作解释证据）
L1_COUNTERS = (
    'role_c_local_search.route_feasible',
    'role_c_local_search.route_dist',
    'role_c_local_search.normalize_state',
    'role_c_forward._ordered',
)

#: 落进度里程碑的叶子。**全部叶子都落**（含 `P7a/P7b/P7c`）——
#: 若只落粗粒度叶子，被硬杀时只能报出 `P7d`，丢掉"到底停在哪片叶子"，
#: 而 REV2 §5.1.2 要求保留**截断位置**。代价是每决策约 1e3–1e4 行小记录；
#: `flush()` 只把数据交给 OS 页缓存（SIGKILL 后仍可读回），**不做 fsync**。
#: ⚠️ 该 I/O 落在被测区间内 ⇒ 必须在报告里如实给出"开/关进度记录"的耗时差。
_LOGGED_LEAVES = pp.LEAF_ORDER


def active_leaf_from_progress(lines):
    """由进度原件重放**当时的活动栈**（供 A 类硬杀后恢复截断位置）。

    返回 `{'leaf_at_truncation', 'active_stack_tail', 'n_scenarios_completed',
    'ordered_at_truncation', 'heartbeat'}`；无可用行时各字段为 `None`。
    REV2 §5.1.2 要求"**不得**只写超时" ⇒ 这个重放就是给出位置的那一步。
    """
    stack = []
    hb = None
    for r in lines or ():
        k = r.get('kind')
        if k == 'leaf_enter':
            stack.append(r.get('leaf'))
        elif k == 'leaf_exit':
            lab = r.get('leaf')
            if stack and stack[-1] == lab:
                stack.pop()
            elif lab in stack:
                stack.remove(lab)          # 容忍异常路径留下的错位
        elif k == 'heartbeat':
            hb = r
    return {
        'leaf_at_truncation': stack[-1] if stack else None,
        'active_stack_tail': list(stack[-3:]),
        'n_scenarios_completed': (hb or {}).get('complete_in_forward'),
        'ordered_at_truncation': (hb or {}).get('ordered'),
        'best_insert_calls_at_truncation': (hb or {}).get('best_insert_calls'),
        'last_heartbeat': hb,
    }


class InstrumentationError(RuntimeError):
    """插桩层自身的问题（不是被测行为）。"""


def patch_table():
    """返回 `[(module_name, attr, leaf_or_None, kind)]` —— 便于测试与审计打印。"""
    rows = [
        ('role_c_local_search', 'state_from_vehicles', 'P1', 'leaf'),
        ('role_c_harness', 'sample_scenarios', 'P2', 'leaf'),
        ('role_c_harness', 'build_space_bundle', 'P3', 'leaf'),
        ('role_c_harness', 'make_order_energy_fn', 'P4', 'leaf'),
        ('role_c_harness', 'make_order_revenue_fn', 'P4', 'leaf'),
        ('role_c_harness', 'make_order_penalty_fn', 'P4', 'leaf'),
        ('role_c_harness', 'make_c0_complete_fn', 'P4+wraps_return', 'leaf_factory'),
        ('role_c_local_search', 'best_insert', 'P5/P7a', 'leaf_conditional'),
        ('role_c_local_search', 'improve', 'P5/P7b', 'leaf_conditional'),
        ('role_c_harness', '_insertion_fuel', 'P6', 'leaf'),
        ('role_c_forward', 'forward_value', 'P7d', 'leaf_forward'),
    ]
    for dotted in L1_COUNTERS:
        mod, attr = dotted.split('.')
        rows.append((mod, attr, None, 'counter'))
    return rows


class LeafInstrumentation(object):
    """把 `PartitionTimer` 接到真实调用点上的上下文管理器（可重入不安全，按需用）。

    用法::

        with LeafInstrumentation(progress_path='/tmp/p.jsonl') as inst:
            inst.begin(subject)                 # 一次决策
            accept, diag = decide_fn(order, ctx)
            rec = inst.end()                    # 本次决策的在线读数 + 验收
        inst.l1                                  # L1 计数（只作解释证据）
    """

    def __init__(self, *, clock=None, on_event=None, progress_path=None,
                 progress_every=200):
        self.timer = pp.PartitionTimer(clock=clock, on_event=self._on_event)
        self._user_on_event = on_event
        self._progress_path = progress_path
        self._progress_every = int(progress_every)
        self._progress_fh = None
        self._events_since_flush = 0
        self._saved = []                     # [(module, attr, original)]
        self._in_forward = 0
        self._suppress = 0
        self._patching = False
        self.l1 = {}
        self.subject = None
        self.marks = []                      # 供截断恢复的**粗粒度**里程碑

    # ------------------------------------------------------------ 生命周期
    def __enter__(self):
        global _ACTIVE
        self._install()
        _ACTIVE = self
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        self._restore()
        self._close_progress()
        return False

    # ------------------------------------------------------------ 补丁
    def _install(self):
        if self._patching:
            raise InstrumentationError('已在插桩状态，不得重复 install')
        import role_c_forward as fwd
        import role_c_harness as hn
        import role_c_local_search as ls
        mods = {'role_c_local_search': ls, 'role_c_harness': hn, 'role_c_forward': fwd}
        # (模块, 属性名, **包装器工厂**)：工厂要吃下**原函数**并返回包装器。
        factories = [
            (ls, 'state_from_vehicles', self._wrap_leaf('P1')),
            (hn, 'sample_scenarios', self._wrap_leaf('P2')),
            (hn, 'build_space_bundle', self._wrap_leaf('P3')),
            (hn, 'make_order_energy_fn', self._wrap_leaf('P4')),
            (hn, 'make_order_revenue_fn', self._wrap_leaf('P4')),
            (hn, 'make_order_penalty_fn', self._wrap_leaf('P4')),
            (hn, 'make_c0_complete_fn', self._wrap_complete_factory()),
            (ls, 'best_insert', self._wrap_best_insert()),
            (ls, 'improve', self._wrap_improve()),
            (hn, '_insertion_fuel', self._wrap_leaf('P6', suppress=True)),
            (fwd, 'forward_value', self._wrap_forward()),
        ]
        for dotted in L1_COUNTERS:
            mod_name, attr = dotted.split('.')
            factories.append((mods[mod_name], attr, self._wrap_counter(dotted)))
        for mod, attr, factory in factories:
            if not hasattr(mod, attr):
                raise InstrumentationError('%s.%s 不存在 ⇒ 插桩映射与源码不符'
                                           % (mod.__name__, attr))
            original = getattr(mod, attr)
            self._saved.append((mod, attr, original))
            setattr(mod, attr, factory(original))
        self._patching = True

    def _restore(self):
        for mod, attr, original in reversed(self._saved):
            setattr(mod, attr, original)
        self._saved = []
        self._patching = False

    # ------------------------------------------------------------ 包装器工厂
    def _wrap_leaf(self, label, *, suppress=False):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                if self._suppress and not suppress:
                    return fn(*a, **kw)
                if suppress:
                    self._suppress += 1
                    try:
                        with self.timer.leaf(label):
                            return fn(*a, **kw)
                    finally:
                        self._suppress -= 1
                with self.timer.leaf(label):
                    return fn(*a, **kw)
            wrapper.__phasea_leaf__ = label
            return wrapper
        return deco

    def _wrap_complete_factory(self):
        """`make_c0_complete_fn` 的**构造**记 P4；返回的闭包在**前瞻内**记 P7c。

        前瞻外调用该闭包（若发生）**不标注叶子** ⇒ 时间落入 `P9_measured`（未归因），
        而不是被硬塞进某个叶子。这比"给它编一个叶子名"更诚实。
        """
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                with self.timer.leaf('P4'):
                    inner = fn(*a, **kw)

                @functools.wraps(inner)
                def g(*ia, **ikw):
                    self.l1['complete_fn_calls'] = \
                        self.l1.get('complete_fn_calls', 0) + 1
                    if self._in_forward:
                        self.l1['complete_fn_calls_in_forward'] = \
                            self.l1.get('complete_fn_calls_in_forward', 0) + 1
                        with self.timer.leaf('P7c'):
                            return inner(*ia, **ikw)
                    return inner(*ia, **ikw)
                g.__phasea_wrapped_complete__ = True
                return g
            wrapper.__phasea_leaf__ = 'P4'
            return wrapper
        return deco

    def _wrap_best_insert(self):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(ctx, state, o, *a, **kw):
                self._count('best_insert_calls')
                n = _candidate_positions(state)
                self.l1['best_insert_candidate_positions'] = \
                    self.l1.get('best_insert_candidate_positions', 0) + n
                if self._suppress:
                    return fn(ctx, state, o, *a, **kw)
                label = 'P7a' if self._in_forward else 'P5'
                with self.timer.leaf(label):
                    return fn(ctx, state, o, *a, **kw)
            return wrapper
        return deco

    def _wrap_improve(self):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(ctx, state, *a, **kw):
                self._count('improve_calls')
                if self._suppress:
                    return fn(ctx, state, *a, **kw)
                label = 'P7b' if self._in_forward else 'P5'
                with self.timer.leaf(label):
                    return fn(ctx, state, *a, **kw)
            return wrapper
        return deco

    def _wrap_forward(self):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                self._count('forward_value_calls')
                self._in_forward += 1
                try:
                    with self.timer.leaf('P7d'):
                        return fn(*a, **kw)
                finally:
                    self._in_forward -= 1
            return wrapper
        return deco

    def _wrap_counter(self, dotted):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                self._count(dotted)
                return fn(*a, **kw)
            return wrapper
        return deco

    def _count(self, key, n=1):
        self.l1[key] = self.l1.get(key, 0) + n

    def wrap_certify_fn(self, fn):
        """把真实 `certify_plan` 包成 `P8` 叶子，供注入 `build_decide_fn(certify_plan_fn=…)`。

        为什么由调用方注入而不是打补丁：`build_decide_fn` 在**缺省**时才从
        `run_exp_energy_c0` 导入 `certify_plan`；显式传入是**公开参数**，属于正常用法，
        比替换模块属性更干净、也更难出错。
        """
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            self._count('certify_calls')
            with self.timer.leaf('P8'):
                return fn(*a, **kw)
        wrapper.__phasea_leaf__ = 'P8'
        return wrapper

    # ------------------------------------------------------------ 一次决策
    def begin(self, subject=None):
        self.subject = subject
        self.marks = []
        self.timer.start_decision()
        self._mark('decision_start')
        return self

    def end(self, **kw):
        self.timer.end_decision()
        online = self.timer.result()
        rec = pp.reconcile(online, **kw)
        rec['subject'] = self.subject
        rec['l1'] = dict(self.l1)
        rec['stages_entered'] = leaves_entered(online.get('leaf_counts') or {})
        rec['marks_tail'] = self.marks[-1] if self.marks else None
        return rec

    # ------------------------------------------------------------ 进度/里程碑
    def _mark(self, kind, **extra):
        rec = {'kind': kind}
        rec.update(extra)
        self.marks.append(rec)
        self._write(rec)

    def _on_event(self, kind, label, t):
        # 细粒度事件**不逐条落盘**（I/O 会污染耗时）：只在**叶子级**粗粒度上记
        # `leaf_enter`/`leaf_exit`（供硬杀后重放"当时活动栈"=截断位置），
        # 并每 `progress_every` 条刷一次**带 L1 快照**的心跳。
        self._events_since_flush += 1
        if self._user_on_event is not None:
            try:
                self._user_on_event(kind, label, t)
            except Exception:                                    # noqa: BLE001
                pass
        if label in _LOGGED_LEAVES:
            self._mark('leaf_%s' % kind, leaf=label, t=t)
        if self._events_since_flush >= self._progress_every:
            self._events_since_flush = 0
            self._mark('heartbeat', t=t, events=len(self.timer.events),
                       ordered=self.l1.get('role_c_forward._ordered', 0),
                       complete_in_forward=self.l1.get(
                           'complete_fn_calls_in_forward', 0),
                       best_insert_calls=self.l1.get('best_insert_calls', 0))

    def _write(self, rec):
        if not self._progress_path:
            return
        import io
        import json
        if self._progress_fh is None:
            self._progress_fh = io.open(self._progress_path, 'a', encoding='utf-8',
                                        newline='\n')
        try:
            self._progress_fh.write(json.dumps(rec, ensure_ascii=False,
                                               sort_keys=True) + '\n')
            self._progress_fh.flush()
        except Exception:                                        # noqa: BLE001
            pass

    def _close_progress(self):
        if self._progress_fh is not None:
            try:
                self._progress_fh.close()
            except Exception:                                    # noqa: BLE001
                pass
            self._progress_fh = None


def leaves_entered(leaf_counts):
    """本次决策**进入过**的叶子（按 §4.2 的事前顺序）—— 供 B 类的 `stages_entered`。"""
    return [lab for lab in pp.LEAF_ORDER if int((leaf_counts or {}).get(lab, 0)) > 0]


def measure_progress_overhead(*, n_events=4000, repeats=7, warmup=1):
    """**进度记录的自开销**："不开/开进度文件"两次同样的事件流的耗时差。

    为什么必须单独测：`leaf_enter`/`leaf_exit` 的写入落在**被测区间内**，
    它属于"仪器自开销"，必须在报告里如实给出（规格 §3）。
    ⚠️ 开销只作诊断，**不得**据此调整 §4.3 的容差。
    """
    import os
    import tempfile
    import time as _time

    def run(progress_path):
        tm = pp.PartitionTimer(on_event=None)
        inst = LeafInstrumentation.__new__(LeafInstrumentation)
        inst.timer = tm
        inst._user_on_event = None
        inst._progress_path = progress_path
        inst._progress_every = max(1, n_events // 4)
        inst._progress_fh = None
        inst._events_since_flush = 0
        inst._saved = []
        inst._in_forward = 0
        inst._suppress = 0
        inst._patching = False
        inst.l1 = {}
        inst.subject = None
        inst.marks = []
        tm.on_event = inst._on_event
        try:
            tm.start_decision()
            for i in range(int(n_events)):
                lab = pp.LEAF_ORDER[i % len(pp.LEAF_ORDER)]
                with tm.leaf(lab):
                    pass
            tm.end_decision()
        finally:
            inst._close_progress()
        return tm.result()

    def timed(progress_path):
        xs = []
        for _ in range(int(warmup) + int(repeats)):
            t0 = _time.perf_counter()
            run(progress_path)
            xs.append(_time.perf_counter() - t0)
        xs = sorted(xs[int(warmup):])
        n = len(xs)
        return {'n': n, 'p50_s': xs[n // 2] if n else None,
                'min_s': xs[0] if n else None, 'max_s': xs[-1] if n else None}

    off = timed(None)
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    try:
        on = timed(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    d = (on['p50_s'] - off['p50_s']) if (on['p50_s'] and off['p50_s']) else None
    return {'n_events': int(n_events), 'progress_off_s': off, 'progress_on_s': on,
            'overhead_s': d,
            'overhead_per_event_s': (d / n_events) if d is not None else None,
            'note': ('进度记录自开销（落在被测区间内）。只作诊断，'
                     '**不得**据此调整 §4.3 对账容差。')}


def _candidate_positions(state):
    """`best_insert(ctx, state, o)` 会被考察的插入位数（**输入的函数**，不改行为）。

    = Σ_vehicles (len(route) + 1)。这是 L1 口径的"候选数"，
    **只作解释证据**（规格 §5.3），**不参与**主导判定。
    """
    try:
        return int(sum(len(list(s.get('route') or ())) + 1 for s in state.values()))
    except Exception:                                            # noqa: BLE001
        return 0
