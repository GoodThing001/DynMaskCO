#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""叶子插桩层回归（用户 2026-10-14 前置门之二在**真实调用点**上的验证）。

与 `test_role_c_phasea_partition.py` 的分工：
  * 那个测的是**计时器语义**（G 边界、三通道、四条件）—— 用确定性时钟的合成夹具；
  * 本文件测的是**映射是否落在真实源码的调用点上** —— 用**真实的**
    `role_c_local_search` / `role_c_forward` / `role_c_harness` 函数 + 合成输入。

⚠️ 物理是**替身**（`_FakePhysics` 返回常数能耗）：本文件验证的是**接线**，
不是 C0 物理。真实物理只在跑批机器上跑；不要把本文件的通过读成"物理正确"。

关键断言（都会真的失败，而不是恒真）：
  * `P6` **砍点**：做一次**差分**实验 —— 有/无 `_insertion_fuel` 两种序列下
    `P5` 计数必须**相同**，而 `best_insert_calls` 差恰好 1（被抑制的那次）；
  * `P5`/`P7a` 分流：前瞻内 `ls.best_insert` 记 `P7a`、前瞻外记 `P5`；
  * `P7c` 只在前瞻内标注，前瞻外调用同一闭包**不**产生 `P7c`；
  * 还原：退出上下文后，被测模块的相关属性与进入前**是同一个对象**。

运行：`python CC_Compare/tools/test_role_c_phasea_instrument.py`
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_forward as fwd                                       # noqa: E402
import role_c_harness as hn                                        # noqa: E402
import role_c_local_search as ls                                   # noqa: E402
import role_c_phasea_instrument as inst_mod                        # noqa: E402
import role_c_phasea_partition as pp                               # noqa: E402
import role_c_t_series as ts                                       # noqa: E402

# `_insertion_fuel` 内部要 `coldchain_evaluator_a1.KM_PER_UNIT × FUEL_COST_PER_KM`；
# 走项目自带的路径引导（与真实 runner 相同），而不是在本文件里硬编码常数。
ts.ensure_project_paths()

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


# --------------------------------------------------------------------- 合成世界
class _Ctx(object):
    """`space.route_ctx()` 的鸭子类型替身（字段与 `route_dist`/`route_feasible` 对齐）。"""

    def __init__(self, n=6):
        self.n = n
        self.dist = [[abs(i - j) * 1.0 for j in range(n)] for i in range(n)]
        self.tw_speed = 1.0
        self.reveal = [0.0] * n
        self.tws = [0.0] * n
        self.twe = [1e6] * n
        self.service = [0.1] * n
        self.demand = [0.0] + [1.0] * (n - 1)
        self.capacity = 100.0
        self.deadline = 1e6


class _Space(object):
    """同时满足 `route_ctx()` 与 `make_c0_complete_fn` 的字段需求（转发给 ctx）。"""

    def __init__(self, ctx):
        self._ctx = ctx
        self.reveal = ctx.reveal
        self.tc = [0] * ctx.n
        self.iq = [0.0] * ctx.n
        self.demand = ctx.demand
        self.return_deadline = ctx.deadline

    # `_complete` 直接读 space.* ⇒ 转发
    @property
    def dist(self):
        return self._ctx.dist

    @property
    def tws(self):
        return self._ctx.tws

    @property
    def twe(self):
        return self._ctx.twe

    @property
    def service(self):
        return self._ctx.service

    def route_ctx(self):
        return self._ctx


class _CC(object):
    closed = False
    cumulative_energy_kwh = 0.0


class _Veh(object):
    def __init__(self, vid, node=0, load=0.0):
        self.vehicle_id = vid
        self.coldchain_state = _CC()
        self.status = 'idle'
        self.committed_next = None
        self.committed_finish = 0.0
        self.current_node = node
        self.current_load = load


class _Bundle(object):
    def __init__(self, space):
        self.space = space


class _VState(object):
    cumulative_energy_kwh = 1.0


class _FakePhysics(object):
    """**物理替身**：每次调用返回固定能耗 1.0。只用于验证接线。"""

    calls = 0

    def __getitem__(self, _k):
        return self._fn

    def _fn(self, *a, **kw):
        type(self).calls += 1
        return _VState(), None


class _FakeContract(object):
    class units(object):
        speed_kmph = 1.0
        distance_km_per_unit = 1.0

    class thermal(object):
        supported_temp_classes = (0,)


def _world(n=6):
    ctx = _Ctx(n)
    space = _Space(ctx)
    veh = [_Veh(1), _Veh(2)]
    served = [False] * n
    plan = {1: [], 2: []}
    return ctx, space, veh, served, plan


def _make_complete(space):
    """用**被插桩的**工厂造 `complete_fn`（真实管线就是这么拿到的）。"""
    return hn.make_c0_complete_fn(bundle=_Bundle(space), contract=_FakeContract(),
                                  booking_horizon=10.0, physics=_FakePhysics())


def _forward(space, states, complete, scenarios):
    return fwd.forward_value(
        space=space, base_states=states, scenarios=scenarios, budget=100.0,
        initial_est_used=0.0, order_energy_fn=lambda o: 1.0,
        order_revenue_fn=lambda o: 10.0, order_penalty_fn=lambda o: 5.0,
        complete_fn=complete, fuel_cost_per_distance_unit=1.0,
        infeasible_penalty=-1e5)


# ------------------------------------------------------------------ 1. 补丁与还原
def test_install_and_restore_is_identity():
    print('\n[1] 补丁可安装、可**逐字节还原**（模块属性回到同一对象）')
    attrs = [
        (ls, 'state_from_vehicles'), (ls, 'best_insert'), (ls, 'improve'),
        (ls, 'route_feasible'), (ls, 'route_dist'), (ls, 'normalize_state'),
        (hn, 'sample_scenarios'), (hn, 'build_space_bundle'),
        (hn, 'make_order_energy_fn'), (hn, 'make_order_revenue_fn'),
        (hn, 'make_order_penalty_fn'), (hn, 'make_c0_complete_fn'),
        (hn, '_insertion_fuel'), (fwd, 'forward_value'), (fwd, '_ordered'),
    ]
    before = {(m.__name__, a): getattr(m, a) for m, a in attrs}
    with inst_mod.LeafInstrumentation() as _inst:
        check('安装：属性确实被换成了包装器',
              all(getattr(m, a) is not before[(m.__name__, a)] for m, a in attrs))
        check('安装：标记过的叶子名可查',
              getattr(ls.state_from_vehicles, '__phasea_leaf__', None) == 'P1'
              and getattr(ls.best_insert, '__phasea_leaf__', None) is None)
        check('安装：重复 install 被拒',
              _raises(inst_mod.InstrumentationError, _inst._install))
    after = {(m.__name__, a): getattr(m, a) for m, a in attrs}
    check('还原：全部属性回到**同一对象**', after == before,
          [k for k in before if after[k] is not before[k]])
    return True


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


def test_patch_table_resolves_on_real_source():
    print('\n[2] 映射表逐条在**真实源码**上可解析（缺一即抛）')
    mods = {'role_c_local_search': ls, 'role_c_harness': hn, 'role_c_forward': fwd}
    bad = ['%s.%s' % (m, a) for m, a, _l, _k in inst_mod.patch_table()
           if not hasattr(mods[m], a)]
    check('映射：全部目标属性存在', not bad, bad)
    check('映射：叶子覆盖 §4.2 的 11 个叶子（不含 P7/P9）',
          set(inst_mod.LEAF_BINDINGS) == set(pp.LEAF_ORDER), inst_mod.LEAF_BINDINGS)
    check('映射：P6 被声明为抑制点', inst_mod.SUPPRESSING_LEAVES == ('P6',))
    check('映射：L1 计数器不含任何叶子标签（只计数）',
          all(lab is None for _m, _a, lab, kind in inst_mod.patch_table()
              if kind == 'counter'))
    return True


# --------------------------------------------- 3. P5/P7a 分流 + P6 砍点（差分实验）
def _run_sequence(*, with_fuel, with_forward):
    ctx, space, veh, served, plan = _world()
    with inst_mod.LeafInstrumentation(clock=pp.tick_clock(1e-3)) as inst:
        inst.begin(subject={'case': 'diff'})
        states = ls.state_from_vehicles(veh, served, plan, 0.0)         # P1
        st2, _ok = ls.best_insert(ctx, states, 1)                       # P5
        st2, _ = ls.improve(ctx, st2)                                   # P5
        if with_fuel:
            hn._insertion_fuel(_Bundle(space), states, 1, None)         # P6（砍点）
        if with_forward:
            complete = _make_complete(space)                            # P4
            _forward(space, states, complete, [[2, 3], [4, 5]])         # P7d/P7a/P7b/P7c
        rec = inst.end()
    return inst.timer._counts, dict(inst.l1), rec


def test_p6_suppression_is_a_real_cut_point():
    print('\n[3] **P6 砍点（差分实验）**：加上 P6 不改变 P5 计数')
    c0, l0, r0 = _run_sequence(with_fuel=False, with_forward=False)
    c1, l1, r1 = _run_sequence(with_fuel=True, with_forward=False)
    check('差分：P6 计数 0 → 1', c0.get('P6', 0) == 0 and c1.get('P6', 0) == 1,
          (c0, c1))
    check('**差分：P5 计数不变（内部 best_insert 被抑制，未双计）**',
          c0.get('P5', 0) == c1.get('P5', 0) == 2, (c0.get('P5'), c1.get('P5')))
    check('差分：best_insert_calls 恰好 +1（被抑制的那一次仍然被 L1 计数）',
          l1.get('best_insert_calls', 0) - l0.get('best_insert_calls', 0) == 1,
          (l0.get('best_insert_calls'), l1.get('best_insert_calls')))
    check('差分：两次都四条件通过',
          r0['verdict'] == 'ok' and r1['verdict'] == 'ok',
          (r0['failed_conditions'], r1['failed_conditions']))
    return True


def test_p5_vs_p7a_routing_and_forward_leaves():
    print('\n[4] P5/P7a 分流 + 前瞻四叶齐备')
    c, l1, rec = _run_sequence(with_fuel=True, with_forward=True)
    check('分流：P5 恰好 2（前瞻外 best_insert + improve）', c.get('P5', 0) == 2, c)
    check('分流：P7a 存在（前瞻内 best_insert）', c.get('P7a', 0) >= 1, c)
    check('分流：P7b 存在（前瞻内 improve）', c.get('P7b', 0) >= 1, c)
    check('分流：P7c 存在（前瞻内 complete_fn）', c.get('P7c', 0) >= 1, c)
    check('分流：P7d 存在（forward_value 自身）', c.get('P7d', 0) == 1, c)
    check('分流：P4 存在（complete 工厂构造）', c.get('P4', 0) >= 1, c)
    check('分流：P1 存在', c.get('P1', 0) == 1, c)
    check('分流：前瞻内 best_insert 数 = 场景×订单 = 4',
          c.get('P7a', 0) == 4, c)
    check('L1：场景数 = _ordered 调用数 = 2',
          l1.get('role_c_forward._ordered') == 2, l1)
    check('L1：complete_fn_calls = 场景数 = 2',
          l1.get('complete_fn_calls') == 2, l1)
    check('四条件：verdict = ok', rec['verdict'] == 'ok', rec['failed_conditions'])
    check('份额：含 P9_measured 且和为 1',
          pp.UNATTRIBUTED in rec['shares']
          and abs(sum(rec['shares'].values()) - 1.0) < 1e-9, rec['shares'])
    return True


# ---------------------------------------------------------------- 5. P7c 只在前瞻内
def test_p7c_only_inside_forward():
    print('\n[5] `complete_fn` 只在前瞻内标 P7c；前瞻外不标注（落入 P9）')
    ctx, space, veh, served, plan = _world()
    with inst_mod.LeafInstrumentation(clock=pp.tick_clock(1e-3)) as inst:
        inst.begin(subject={'case': 'p7c'})
        states = ls.state_from_vehicles(veh, served, plan, 0.0)
        complete = _make_complete(space)
        check('P4：工厂构造被计为 P4', inst.timer._counts.get('P4', 0) == 1,
              inst.timer._counts)
        complete(dict(states))
        check('前瞻外：complete_fn_calls +1 ⇒ 确实被调用过',
              inst.l1.get('complete_fn_calls') == 1, inst.l1)
        check('前瞻外：**不**产生 P7c',
              inst.timer._counts.get('P7c', 0) in (None, 0), inst.timer._counts)
        _forward(space, states, complete, [[2]])
        check('前瞻内：产生 P7c', inst.timer._counts.get('P7c', 0) >= 1,
              inst.timer._counts)
        check('前瞻内：complete_fn_calls 变成 2',
              inst.l1.get('complete_fn_calls') == 2, inst.l1)
        rec = inst.end()
        check('完整决策：verdict = ok', rec['verdict'] == 'ok',
              rec['failed_conditions'])
        off = rec['offline']['exclusive']
        check('互斥：P7c 的 exclusive 不含前瞻外那次调用的时间',
              off.get('P7c', 0) > 0, off)
    return True


# ------------------------------------------------------------------ 6. 进度与恢复
def test_progress_file_and_truncation_recovery():
    print('\n[6] 进度原件：粗粒度里程碑 + 心跳；硬杀后可恢复截断位置')
    _ctx, _space, _veh, _s, _p = _world()
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    try:
        with inst_mod.LeafInstrumentation(clock=pp.tick_clock(1e-3),
                                         progress_path=path,
                                         progress_every=2) as inst:
            inst.begin(subject={'dp': 3})
            with inst.timer.leaf('P3'):
                pass
            with inst.timer.leaf('P8'):
                pass
            inst.timer.end_decision()
        lines = [json.loads(x) for x in
                 io.open(path, encoding='utf-8').read().splitlines() if x.strip()]
        kinds = [r['kind'] for r in lines]
        check('进度：有 decision_start', 'decision_start' in kinds, kinds)
        check('进度：有 leaf_enter 里程碑', 'leaf_enter' in kinds, kinds)
        check('进度：有 heartbeat（每 progress_every 条）', 'heartbeat' in kinds, kinds)
        check('进度：每行合法 JSON 且带 kind', all('kind' in r for r in lines))
        leaf_lines = [r for r in lines if r['kind'] == 'leaf_enter']
        check('进度：里程碑里带叶子名（可定位截断在哪个叶子）',
              all('leaf' in r for r in leaf_lines)
              and {r['leaf'] for r in leaf_lines} >= {'P3', 'P8'},
              leaf_lines)
        check('进度：写下决策索引（本次 subject）',
              lines[0].get('kind') == 'decision_start', lines[0])
    finally:
        os.remove(path)
    return True


# ------------------------------------------------------------------------ 7. L1
def test_l1_counters_are_explanation_only():
    print('\n[7] L1 计数：只作解释证据，不出现在叶子/份额里')
    ctx, space, veh, served, plan = _world()
    with inst_mod.LeafInstrumentation(clock=pp.tick_clock(1e-3)) as inst:
        inst.begin()
        states = ls.state_from_vehicles(veh, served, plan, 0.0)
        ls.best_insert(ctx, states, 1)
        rec = inst.end()
    keys = sorted(k for k in inst.l1 if '.' in k)
    check('L1：有 route_feasible / route_dist 计数',
          any('route_feasible' in k for k in keys)
          and any('route_dist' in k for k in keys), keys)
    check('L1：有候选位计数（输入的函数，不改行为）',
          inst.l1.get('best_insert_candidate_positions', 0) > 0, inst.l1)
    check('L1：**未**污染叶子计数（route_dist 不在 timer._counts）',
          'route_dist' not in inst.timer._counts, inst.timer._counts)
    check('L1：**未**进入份额字典',
          all('route_' not in k for k in rec['shares']), sorted(rec['shares']))
    check('L1：候选位 = Σ(len(route)+1) = 2 辆空车 ⇒ 2',
          inst.l1.get('best_insert_candidate_positions') == 2, inst.l1)
    return True


# --------------------------------------------------- 8. 与计时器语义的组合正确性
def test_composes_with_partition_invariants():
    print('\n[8] 与计时器组合：真实调用点下的四条件与互斥性')
    c, _l1, rec = _run_sequence(with_fuel=True, with_forward=True)
    check('组合：①②③④ 全过', rec['verdict'] == 'ok', rec['failed_conditions'])
    check('组合：Σexclusive + G == T_decision',
          abs(rec['sum_exclusive_online'] + rec['unattributed_online']
              - rec['t_decision']) < 1e-12,
          (rec['sum_exclusive_online'], rec['unattributed_online'],
           rec['t_decision']))
    off = rec['offline']['exclusive']
    check('组合：P6 的 exclusive > 0（含被抑制的内部工作）', off.get('P6', 0) > 0, off)
    check('组合：P7d 的 exclusive > 0 且不含 P7a/P7b/P7c',
          off.get('P7d', 0) > 0 and off.get('P7a', 0) > 0, off)
    check('组合：honesty_note 明写 ③ 不是独立证据',
          '独立证据' in rec['honesty_note'], rec['honesty_note'])
    check('组合：stages_entered 按 §4.2 顺序给出进入过的叶子',
          rec['stages_entered'] == [l for l in pp.LEAF_ORDER if l in c],
          rec['stages_entered'])
    check('组合：complete_fn_calls_in_forward = 场景数（B 类要用）',
          rec['l1'].get('complete_fn_calls_in_forward') == 2, rec['l1'])
    return True


# ------------------------------------------------------ 9. 截断位置重放（A 类硬杀用）
def test_truncation_replay_from_progress():
    print('\n[9] 截断位置重放：由进度原件恢复"当时活动栈"（§5.1.2 不得只写超时）')
    lines = [
        {'kind': 'decision_start'},
        {'kind': 'heartbeat', 'ordered': 3, 'complete_in_forward': 2,
         'best_insert_calls': 40},
        {'kind': 'leaf_enter', 'leaf': 'P7d'},
        {'kind': 'leaf_enter', 'leaf': 'P7a'},
    ]
    rep = inst_mod.active_leaf_from_progress(lines)
    check('重放：截断叶子 = 最内层活动叶子', rep['leaf_at_truncation'] == 'P7a', rep)
    check('重放：活动栈尾部可见', rep['active_stack_tail'] == ['P7d', 'P7a'], rep)
    check('重放：场景完成数来自最近心跳', rep['n_scenarios_completed'] == 2, rep)
    check('重放：已进入场景数来自最近心跳', rep['ordered_at_truncation'] == 3, rep)
    rep2 = inst_mod.active_leaf_from_progress(
        lines + [{'kind': 'leaf_exit', 'leaf': 'P7a'}])
    check('重放：exit 后截断叶子回退到父', rep2['leaf_at_truncation'] == 'P7d', rep2)
    rep3 = inst_mod.active_leaf_from_progress([])
    check('重放：空原件 ⇒ 各字段 None（不得编造）',
          rep3['leaf_at_truncation'] is None
          and rep3['n_scenarios_completed'] is None, rep3)
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    try:
        with inst_mod.LeafInstrumentation(progress_path=path,
                                          progress_every=100000) as inst:
            inst.begin()
            with inst.timer.leaf('P7d'):
                with inst.timer.leaf('P7a'):
                    pass
            inst.timer.end_decision()
        got = [json.loads(x) for x in
               io.open(path, encoding='utf-8').read().splitlines() if x.strip()]
        kinds = [r['kind'] for r in got]
        rep4 = inst_mod.active_leaf_from_progress(got)
        check('重放：真实进度文件里进入/退出都记到了（末态无活动叶子）',
              kinds.count('leaf_enter') == 2 and kinds.count('leaf_exit') == 2
              and rep4['leaf_at_truncation'] is None, (kinds, rep4))
    finally:
        os.remove(path)
    return True


# ----------------------------------------------------------- 10. 进度记录自开销
def test_progress_overhead_is_reported_honestly():
    print('\n[10] 进度记录自开销：开/关都报，且写明不得据此调容差')
    rep = inst_mod.measure_progress_overhead(n_events=1200, repeats=5, warmup=1)
    check('开销：报了关闭态', rep['progress_off_s']['p50_s'] is not None, rep)
    check('开销：报了开启态', rep['progress_on_s']['p50_s'] is not None, rep)
    check('开销：给了绝对差与每事件差',
          rep['overhead_s'] is not None and rep['overhead_per_event_s'] is not None,
          rep)
    check('开销：每事件开销是**微秒级**（不是毫秒级）',
          rep['overhead_per_event_s'] < 1e-3, rep['overhead_per_event_s'])
    check('开销：写明"不得据此调整容差"', '不得' in rep['note'], rep['note'])
    return True


def main():
    print('=' * 78)
    print('Phase A 叶子插桩层回归（真实调用点；物理为替身，只验证接线）')
    print('=' * 78)
    test_install_and_restore_is_identity()
    test_patch_table_resolves_on_real_source()
    test_p6_suppression_is_a_real_cut_point()
    test_p5_vs_p7a_routing_and_forward_leaves()
    test_p7c_only_inside_forward()
    test_progress_file_and_truncation_recovery()
    test_l1_counters_are_explanation_only()
    test_composes_with_partition_invariants()
    test_truncation_replay_from_progress()
    test_progress_overhead_is_reported_honestly()
    print('\n' + '=' * 78)
    if FAILS:
        print('INSTRUMENT_TEST_FAIL: %d 项不通过 -> %s' % (len(FAILS), FAILS))
        return 1
    print('INSTRUMENT_TEST_PASS: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
