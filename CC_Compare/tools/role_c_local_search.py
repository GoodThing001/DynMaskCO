#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**独立**局部搜索（设计规格 §2.2，用户 2026-10-14 裁定**路线 A**）。

为什么是独立模块（而不是复用现成 LS）
------------------------------------
规格 §2.2.1 的只读能力审计结论：项目内**没有任何现成 LS 同时满足「三项能力齐备 ∧
与对照独立」** —— 能力齐备的 `scenario_saa._local_search_u` **长在对照类
`SaaReplanner` 里**（且只在 `shadow_mode='ls'` 的影子空间生效），不在对照类里的
`rolling_optimizer._local_search` **缺显式已承诺锁定**（全文件 0 处 `frozen`）。

用户裁定走**路线 A**：新写独立模块。本模块因此**结构上**满足 §2.2.1 的四条前提：

  * **不 import、不继承对照血统**：本文件不 import `scenario_saa` /
    `rolling_optimizer` / `C0ReserveReplanner` / `SaaReplanner`（由
    `test_role_c_local_search.py` 的**源码级独立性与目标纯度检查**机械验证）。
  * **① 显式已承诺锁定**：`frozen` 是**必需字段**（不是 `.get('frozen', 0)` 的
    缺省值！）。对照侧 Q-01 修复（`scenario_saa.py:881-886`）记录过一个真实事故：
    复制状态时丢失 `frozen` ⇒ 预检恒为 0 ⇒ 未来订单被插进**不可撤销的在途
    committed leg**。本模块把"缺失 `frozen`"直接判为 `LSStateError`，让这类 bug
    **不可能复现**。移除侧与插入侧**双侧**尊重，且每次移动后断言
    `route[:frozen]` 前缀逐位不变（`frozen_prefix_preserved`）。
  * **② 真实车辆锚点**：每条路线从 `(cur, cur_time, load)` 出发——在途车从
    `committed_finish` 续行、载重含在途单；可行性用**真锚点**算（不是从 depot 重算）。
  * **③ 跨车移动**：relocate 的 `src_vid × dst_vid` 全枚举 + swap（跨车/同车）。

邻域、顺序、停止条件、随机源
--------------------------
规格 §2.2.1 前提③要求"源码 + 邻域顺序 + 停止条件 + 随机源冻结并计入
`provider_dependency_sha256`"。因此：

  * **邻域顺序固定**：`relocate` 先（`src_vid` 升序 × `i` 升序 × `dst_vid` 升序 ×
    `pos` 升序），`relocate` 无改进才做 `swap`（`a_vid`/`i`/`b_vid`/`j` 升序）；
    **首次改进即采纳**（first-improvement）后重新开始本趟。
  * **停止条件固定且确定**：`max_passes`（默认 2，与对照侧同量级）+ `max_moves`
    （默认 4000，硬上限）。**与 `time.time()` 无关** ⇒ 同一输入必得同一输出。
  * **随机源：无**。本模块不 import `random`/`numpy.random`，不构造任何 RNG
    （`certificate.rng_used = False`）。
  * `deadline`（可选，单调时钟）只是**熔断**：一旦触发，`certificate.deadline_hit=True`
    —— 该次运行**不再可复现**，调用方必须据此降级而不是当正常结果用。

目标纯度（禁止混用距离代价与 `coldchain_cost`）
--------------------------------------------
用户边界："**禁止混用距离代价与 `coldchain_cost`**"。本模块的改进判据是**纯行驶
距离**（`route_dist` = Σ 车际距离 + 最后一段回仓距离），`coldchain_cost` /
`energy_kwh` / `p_c` / 收入**完全不出现**在目标里。这不是文风问题：`energy_kwh`
是**硬约束**（由 `certify_plan` 把关），把它折进 LS 目标会让 `C − B` 混淆
（规格 §2.1.2）。行为级验证见 `test_..._objective_is_pure_distance`：把候选状态的
能量/成本相关输入全部扰动，LS 输出**必须逐位不变**。

用法
----
    st = state_from_vehicles(vehicles, served_mask, plan, clock)   # 真实锚点 + frozen
    ctx = RouteContext(dist=env.dist_mat[inst_idx], tws=..., twe=..., service=...,
                       demand=..., reveal=..., capacity=..., tw_speed=...,
                       deadline=float(env.tw_end[inst_idx, 0]))
    st2, cert = improve(ctx, st, certify_fn=lambda s: certify_plan(..., plan_of(s), ...))
    plan = plan_of(st2)        # 交给 certify_plan / 落盘

⚠️ 本模块**不读效用、不跑实验、不构造场景**；它在给定状态上做优化并返回证书。
场景集合上的前瞻评估由 Role C 的 `decide_fn` 负责（§2.2）。
"""
from __future__ import annotations

import hashlib
import json
import time

__all__ = [
    'NEIGHBORHOOD_ID', 'SOURCE_SPEC', 'SOURCE_SPEC_SHA256',
    'LSStateError', 'LSError',
    'RouteContext', 'LSCertificate',
    'state_from_vehicles', 'plan_of', 'frozen_prefixes', 'normalize_state',
    'route_dist', 'route_feasible', 'best_insert', 'improve',
    'ablate_to_myopic_scope',
]

# ---------------------------------------------------------------------------
# 邻域规格（**冻结**：任何改动都必须换代 —— 它会进 provider_dependency_sha256）
# ---------------------------------------------------------------------------
NEIGHBORHOOD_ID = 'role-c-ls-v1'

SOURCE_SPEC = {
    'neighborhood_id': NEIGHBORHOOD_ID,
    'route': 'A',                        # 用户裁定：新写独立 LS 模块
    'lineage': 'independent',            # 不 import / 不继承对照血统
    'operators': ('relocate', 'swap'),
    'operator_order': 'relocate_then_swap',
    'insertion': 'best_distance_increment_over_(vid,pos); pos>=frozen; '
                 'first_lexicographic_minimum; no_crossing_frozen_prefix',
    'acceptance': 'first_improvement',
    'objective': 'pure_travel_distance',  # 不含 energy / coldchain_cost / revenue
    'objective_depends_on': ('dist',),
    'objective_forbidden': ('energy_kwh', 'coldchain_cost', 'revenue', 'p_c'),
    'neighborhood_order': ('src_vid', 'src_index', 'dst_vid', 'dst_pos'),
    'swap_order': ('vid_a', 'index_a', 'vid_b', 'index_b'),
    'stopping': {'max_passes': 2, 'max_moves': 4000, 'time': 'fuse_only'},
    'rng': None,
    'frozen_semantics': 'required_field; prefix_len = frozen; both_sides; asserted',
    'feasibility': ('capacity', 'time_window', 'reveal_aware_departure',
                    'return_deadline', 'exactly_once'),
    'certify': 'returned plan must pass injected certify_fn; modes=final|each_move; '
               'final-mode failure falls back to incumbent and records it',
    'indexing': '0 = depot',
}


def _spec_sha256() -> str:
    blob = json.dumps(SOURCE_SPEC, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


SOURCE_SPEC_SHA256 = _spec_sha256()


class LSStateError(ValueError):
    """状态结构不合法（缺 `frozen`、类型错、路线含重复单等）。fail-closed。"""


class LSError(RuntimeError):
    """搜索过程不变量被破坏（如冻结前缀被改动、最终计划未过认证）。"""


# ---------------------------------------------------------------------------
# 输入容器
# ---------------------------------------------------------------------------
class RouteContext:
    """**env 无关**的路线上下文（纯数据，便于合成夹具单测）。

    参数
    ----
    dist : 2D 索引（`dist[a][b]`，索引 `0` = depot）—— 单位任意，LS 只比大小。
    tws, twe : 各节点时间窗起/止（索引 0 = depot）。
    service : 各节点服务时长。
    demand : 各节点需求（索引 0 = depot，应为 0）。
    reveal : 各节点揭示时刻（未揭示订单须等到 reveal 才能**出发**）。
    capacity, tw_speed, deadline : 容量上限 / 行驶速度（距离单位每小时）/ 回仓截止。
    """

    __slots__ = ('dist', 'tws', 'twe', 'service', 'demand', 'reveal',
                 'capacity', 'tw_speed', 'deadline')

    def __init__(self, *, dist, tws, twe, service, demand, reveal,
                 capacity, tw_speed, deadline):
        self.dist = dist
        self.tws = tws
        self.twe = twe
        self.service = service
        self.demand = demand
        self.reveal = reveal
        self.capacity = float(capacity)
        self.tw_speed = float(tw_speed)
        self.deadline = float(deadline)
        if not (self.tw_speed > 0.0):
            raise LSStateError('tw_speed 必须 > 0（否则行驶时间无定义）')


class LSCertificate:
    """一次搜索的**自证**记录（判据全在里面，调用方不必信本模块的"改进了"）。"""

    __slots__ = ('neighborhood_id', 'source_spec_sha256', 'passes',
                 'moves_applied', 'evaluations', 'dist_before', 'dist_after',
                 'frozen_prefix_preserved', 'deadline_hit', 'rng_used',
                 'certified', 'certify_mode', 'fallback_to_incumbent',
                 'moved_orders')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def improved(self):
        return bool(self.dist_after is not None and self.dist_before is not None
                    and self.dist_after < self.dist_before - 1e-9)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):  # pragma: no cover - 便于人工查看
        return 'LSCertificate(%s)' % json.dumps(self.to_dict(), ensure_ascii=False,
                                                sort_keys=True)


# ---------------------------------------------------------------------------
# 状态：真实锚点 + **必需** frozen
# ---------------------------------------------------------------------------
REQUIRED_STATE_KEYS = ('cur', 'cur_time', 'load', 'route', 'frozen')
# **不透明透传**字段：本模块**不读也不解释**它们的语义，只负责"别弄丢"。
# `cc` = 车辆的冷链状态（`coldchain_state`）。⚠️ 它必须透传，否则**终局 C0 完成**会从零
# 起算而不是从车辆真实状态续行 ⇒ 累计能耗与在途货被丢掉，算出的能耗系统性偏低，
# 既与对照 `_complete_u` 不一致、也与真实 `certify_plan` 不一致。
OPTIONAL_STATE_KEYS = ('cc',)


def normalize_state(raw):
    """校验并规范化状态。（**不改原对象**，返回新的 dict/list。）

    与对照侧 `scenario_saa` 的 `s.get('frozen', 0)` 的**关键差别**：本模块要求
    `frozen` **必须显式存在**。缺省 0 正是 Q-01 事故的成因（`scenario_saa.py:881-886`），
    此处把它变成硬错误。
    """
    if not isinstance(raw, dict):
        raise LSStateError('状态必须是 dict(vid -> 记录)，得到 %r' % (type(raw).__name__,))
    out = {}
    for vid, s in raw.items():
        if not isinstance(s, dict):
            raise LSStateError('车辆 %r 的记录必须是 dict' % (vid,))
        missing = [k for k in REQUIRED_STATE_KEYS if k not in s]
        if missing:
            raise LSStateError(
                '车辆 %r 缺字段 %s —— 其中 `frozen` 是**必需**字段：'
                '缺省为 0 会让搜索越过不可撤销的在途 committed leg'
                '（对照侧 Q-01 真实事故）。' % (vid, missing))
        frozen = s['frozen']
        if isinstance(frozen, bool) or not isinstance(frozen, int):
            raise LSStateError('车辆 %r 的 frozen 必须是 int，得到 %r'
                               % (vid, frozen))
        route = [int(o) for o in s['route']]
        if frozen < 0 or frozen > len(route):
            raise LSStateError('车辆 %r 的 frozen=%d 超出路线长度 %d'
                               % (vid, frozen, len(route)))
        if len(set(route)) != len(route):
            raise LSStateError('车辆 %r 的路线含重复订单（恰好一次护栏）：%r'
                               % (vid, route))
        if 0 in route:
            raise LSStateError('车辆 %r 的路线含 depot(0)：路线只装订单' % (vid,))
        out[vid] = dict(cur=int(s['cur']), cur_time=float(s['cur_time']),
                        load=float(s['load']), route=route, frozen=int(frozen))
        # 不透明透传（不校验、不解释语义）：见 `OPTIONAL_STATE_KEYS`
        for k in OPTIONAL_STATE_KEYS:
            if k in s:
                out[vid][k] = s[k]
    return out


def state_from_vehicles(vehicles, served_mask, plan, clock):
    """由真实车辆列表构造 LS 状态（**本模块自己的实现**，非复用对照侧）。

    语义（与项目 strict-online 口径一致，独立重写）：
      * `closed` / `returning` 车**退出**搜索（不可再服务未来订单）；
      * 在途 `committed` 车：从 `committed_finish` 续行，`route=[committed_next]+tail`，
        前 **1** 位冻结（`frozen=1`）——在途 leg 不可撤销；
      * 其余车：锚点 `(current_node, clock, current_load)`，无冻结位；
      * 未出车车**即使空计划也保留**（可接未来订单，且不产生幻影预冷）；
      * `cc`（冷链状态）**原样透传**（`OPTIONAL_STATE_KEYS`）—— 终局完成必须从车辆真实
        状态**续行**，否则累计能耗/在途货被丢掉（对照 `_complete_u:834` 是
        `state = cc if cc is not None else dispatch(...)`）。
    """
    if plan is None:
        plan = {}
    st = {}
    for v in vehicles:
        vid = int(v.vehicle_id)
        cc = getattr(v, 'coldchain_state', None)
        status = getattr(v, 'status', None)
        if cc is not None and (getattr(cc, 'closed', False) or status == 'returning'):
            continue
        tail = [int(o) for o in plan.get(vid, []) if not served_mask[o]]
        cnext = getattr(v, 'committed_next', None)
        if cc is not None and status == 'committed' and cnext not in (None, 0):
            route = [int(cnext)] + [o for o in tail if o != int(cnext)]
            cur, t = int(cnext), float(v.committed_finish)
            load = float(v.current_load)
            frozen = 1
        elif cc is not None:
            cur, t = int(v.current_node), float(clock)
            load = float(v.current_load)
            route, frozen = tail, 0
        else:
            cur, t, load, route, frozen = 0, float(clock), 0.0, tail, 0
        st[vid] = dict(cur=cur, cur_time=t, load=load, route=route, frozen=frozen,
                       cc=cc)
    return st


def plan_of(state):
    """状态 → 计划 `{vid: [orders]}`（交给 `certify_plan` / 落盘）。"""
    return {int(vid): list(s['route']) for vid, s in state.items()}


def frozen_prefixes(state):
    """`{vid: tuple(route[:frozen])}` —— 用于移动前后逐位比对。"""
    return {int(vid): tuple(s['route'][:s['frozen']]) for vid, s in state.items()}


# ---------------------------------------------------------------------------
# 纯距离目标 + 可行性（**真锚点**）
# ---------------------------------------------------------------------------
def _d(ctx, a, b):
    return float(ctx.dist[int(a)][int(b)])


def route_dist(ctx, cur, route):
    """路线行驶距离 = 锚点→首单 + 段间 + 末单→depot。**唯一**目标。"""
    if not route:
        return 0.0
    d = _d(ctx, cur, route[0])
    for a, b in zip(route, route[1:]):
        d += _d(ctx, a, b)
    d += _d(ctx, route[-1], 0)
    return d


def route_feasible(ctx, cur, cur_time, load, route):
    """硬可行性预检（容量 / 时间窗 / 揭示感知出发 / 回仓截止 / 恰好一次）。

    与对照侧 `_route_feasible_u` **同语义**但**独立重写**（这是路线 A 的意义所在）：
      * **揭示感知出发**：未揭示订单在**出发地**等到 reveal 才行驶
        （`depart = max(t, reveal[o])`）—— 写成"在目的地等待"会低估到达时间，
        让预检放行而完成阶段 TW 全崩（对照侧 `:435-437` 记录的根因修复）；
      * 到达早于 `tws` ⇒ 等到 `tws`；晚于 `twe+1e-6` ⇒ 不可行；
      * 容量留 `1e-4` 保守余量（影子预检严格于物理 transition 的 `_EPS=1e-9`）；
      * 回仓 `t + D[c,0]/speed <= deadline + 1e-6`。
    """
    if len(set(route)) != len(route):
        return False
    inv = 1.0 / ctx.tw_speed
    t, c, l = float(cur_time), int(cur), float(load)
    for o in route:
        d = _d(ctx, c, o)
        rv = float(ctx.reveal[o])
        depart = t if t >= rv else rv
        arr = depart + d * inv
        if arr < float(ctx.tws[o]):
            arr = float(ctx.tws[o])
        if arr > float(ctx.twe[o]) + 1e-6:
            return False
        l += float(ctx.demand[o])
        if l > ctx.capacity - 1e-4:
            return False
        t = arr + float(ctx.service[o])
        c = int(o)
    ret = t + _d(ctx, c, 0) * inv
    return ret <= ctx.deadline + 1e-6


def _total_dist(ctx, state):
    return sum(route_dist(ctx, s['cur'], s['route']) for s in state.values())


def best_insert(ctx, state, o):
    """把订单 `o` 插到**最优可行位**（最小距离增量），返回 `(新状态, 是否插入)`。

    * 枚举顺序**固定**：`vid` 升序 → `pos` 升序；取**首个**达到最小增量的位置
      （严格 `< best − 1e-9` 才替换）⇒ 同一输入必得同一结果，无 RNG。
    * **插入侧尊重 `frozen`**：`pos` 从 `s['frozen']` 起（不可越过在途 committed leg）。
    * **不改动输入**：返回新的状态 dict。
    * 无处可插（全部不可行）⇒ 返回 `(未改动的副本, False)`，由调用方决定拒单语义。
    """
    incumbent = normalize_state(state)
    o = int(o)
    for s in incumbent.values():
        if o in s['route']:
            raise LSStateError('订单 %d 已在车辆路线上（恰好一次护栏）' % o)
    work = {vid: dict(s, route=list(s['route'])) for vid, s in incumbent.items()}
    best = None
    for vid in sorted(work.keys()):
        s = work[vid]
        base = route_dist(ctx, s['cur'], s['route'])
        for pos in range(s['frozen'], len(s['route']) + 1):
            trial = s['route'][:pos] + [o] + s['route'][pos:]
            if not route_feasible(ctx, s['cur'], s['cur_time'], s['load'], trial):
                continue
            incr = route_dist(ctx, s['cur'], trial) - base
            if best is None or incr < best[0] - 1e-9:
                best = (incr, vid, pos)
    if best is None:
        return work, False
    _, vid, pos = best
    r = work[vid]['route']
    work[vid]['route'] = r[:pos] + [o] + r[pos:]
    return work, True


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def _try_relocate(ctx, state, ev, certify_fn):
    """relocate：把 `src[i]` 移到 `dst` 的 `pos`。首次改进即返回 True。

    同车情形统一按"**先移除、再插入**"语义枚举插入位（`pos` 是移除后列表的下标），
    覆盖与对照侧 `:720-725` 等价（含"移回原位"这一无操作会被 `pos == i` 跳过）。
    """
    vids = sorted(state.keys())
    for src in vids:
        s = state[src]
        for i in range(s['frozen'], len(s['route'])):
            o = s['route'][i]
            base = route_dist(ctx, s['cur'], s['route'])
            for dst in vids:
                d = state[dst]
                droute = d['route']
                for pos in range(d['frozen'], len(droute) + 1):
                    ev[0] += 1
                    if src == dst:
                        rest = s['route'][:i] + s['route'][i + 1:]
                        if pos == i:
                            continue                      # 无操作
                        trial = rest[:pos] + [o] + rest[pos:]
                        if not route_feasible(ctx, s['cur'], s['cur_time'],
                                              s['load'], trial):
                            continue
                        new_d = route_dist(ctx, s['cur'], trial)
                        if new_d >= base - 1e-9:
                            continue
                        if not _certify_ok(certify_fn, state, src, trial):
                            continue
                        s['route'] = trial
                        return True
                    else:
                        trial_src = s['route'][:i] + s['route'][i + 1:]
                        trial_dst = droute[:pos] + [o] + droute[pos:]
                        if not route_feasible(ctx, s['cur'], s['cur_time'],
                                              s['load'], trial_src):
                            continue
                        if not route_feasible(ctx, d['cur'], d['cur_time'],
                                              d['load'], trial_dst):
                            continue
                        old_d = base + route_dist(ctx, d['cur'], droute)
                        new_d = (route_dist(ctx, s['cur'], trial_src)
                                 + route_dist(ctx, d['cur'], trial_dst))
                        if new_d >= old_d - 1e-9:
                            continue
                        if not _certify_pair_ok(certify_fn, state, src, trial_src,
                                                dst, trial_dst):
                            continue
                        s['route'] = trial_src
                        d['route'] = trial_dst
                        return True
    return False


def _try_swap(ctx, state, ev, certify_fn):
    """swap：两条路线上的两个订单互换（跨车 / 同车）。首次改进即返回 True。"""
    vids = sorted(state.keys())
    for a_vid in vids:
        ra = state[a_vid]['route']
        for i in range(state[a_vid]['frozen'], len(ra)):
            for b_vid in vids:
                rb = state[b_vid]['route']
                for j in range(state[b_vid]['frozen'], len(rb)):
                    if a_vid == b_vid and i >= j:
                        continue
                    ev[0] += 1
                    sa, sb = state[a_vid], state[b_vid]
                    na, nb = list(sa['route']), list(sb['route'])
                    oa, ob = na[i], nb[j]
                    na[i], nb[j] = ob, oa
                    if na == sa['route'] and nb == sb['route']:
                        continue
                    if not route_feasible(ctx, sa['cur'], sa['cur_time'],
                                          sa['load'], na):
                        continue
                    if not route_feasible(ctx, sb['cur'], sb['cur_time'],
                                          sb['load'], nb):
                        continue
                    old_d = (route_dist(ctx, sa['cur'], sa['route'])
                             + route_dist(ctx, sb['cur'], sb['route']))
                    new_d = (route_dist(ctx, sa['cur'], na)
                             + route_dist(ctx, sb['cur'], nb))
                    if new_d >= old_d - 1e-9:
                        continue
                    if not _certify_pair_ok(certify_fn, state, a_vid, na,
                                            b_vid, nb):
                        continue
                    sa['route'], sb['route'] = na, nb
                    return True
    return False


def _certify_ok(certify_fn, state, vid, route):
    """单车上限试认证（`certify_fn=None` ⇒ 不额外认证，由 `improve` 末端统一认证）。"""
    if certify_fn is None:
        return True
    ctx_view = _state_view(state, {vid: route})
    ok, _ = certify_fn(ctx_view)
    return bool(ok)


def _certify_pair_ok(certify_fn, state, vid_a, route_a, vid_b, route_b):
    if certify_fn is None:
        return True
    ok, _ = certify_fn(_state_view(state, {vid_a: route_a, vid_b: route_b}))
    return bool(ok)


def _state_view(state, override):
    """浅拷贝状态并覆盖若干车的路线（不动原对象；保留 cur/cur_time/load/frozen）。"""
    out = {vid: dict(s) for vid, s in state.items()}
    for vid, route in override.items():
        out[vid] = dict(out[vid])
        out[vid]['route'] = list(route)
    return out


def improve(ctx, state, *, certify_fn=None, certify_mode='final', max_passes=2,
            max_moves=4000, deadline=None, ablate_operators=None):
    """在 `state` 上做 relocate → swap 局部搜索，返回 `(新状态, LSCertificate)`。

    参数
    ----
    certify_fn : 可选 `fn(state) -> (ok, energy)`。返回值里的**能量被丢弃**（只取
        bool）—— 目标不含能量，见模块头"目标纯度"。
    certify_mode : 认证**时机**，两条都是 fail-closed：
        * `'final'`（缺省）：移动只用路线可行性预检把关，**返回的计划**在末端认证一次；
          未过 ⇒ **回退 incumbent**（并确认 incumbent 自身可认证），
          证书记 `certified=False, fallback_to_incumbent=True`。
          规格 §2.2 的字面要求正是"任何**计划**须过 `certify_plan`"；且真实
          `certify_plan` 要做整条计划的 C0 全模拟，逐移动调用在 10 s/决策的预算下
          不现实 ⇒ 这是缺省。
        * `'each_move'`：**每次候选移动**都先试认证，通过才采纳（严格但昂贵；适合
          小规模严格核验）。此模式下最终状态必然已认证 ⇒ 不会触发回退。
    max_passes, max_moves : **确定的**停止条件（不依赖墙钟）。
    deadline : 可选单调时钟熔断；触发即置 `deadline_hit=True`（该次运行不可复现）。
    ablate_operators : 消融钩子（如 `()` 关闭全部邻域 = 恒等算子，或 `('swap',)`
        只留 swap）。给 `()` 时输出必等于输入 —— 这是**机制消融**而非"动作不同"。

    ⚠️ **不读效用**：本函数既不读 `J`、也不读 `energy_kwh` / `coldchain_cost`，
    目标只有 `route_dist`。
    """
    if certify_mode not in ('final', 'each_move'):
        raise LSStateError("certify_mode 只能是 'final' 或 'each_move'，得到 %r"
                           % (certify_mode,))
    if certify_mode == 'final':
        per_move_certify = None      # 末端统一认证
    else:
        per_move_certify = certify_fn
    if ablate_operators is None:
        ops = ('relocate', 'swap')
    else:
        ops = tuple(ablate_operators)
        bad = [o for o in ops if o not in ('relocate', 'swap')]
        if bad:
            raise LSStateError('未知消融算子 %r（只支持 relocate / swap）' % (bad,))
    if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 0:
        raise LSStateError('max_passes 必须是非负 int，得到 %r' % (max_passes,))
    if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0:
        raise LSStateError('max_moves 必须是非负 int，得到 %r' % (max_moves,))

    incumbent = normalize_state(state)
    before_prefix = frozen_prefixes(incumbent)
    work = {vid: dict(s, route=list(s['route'])) for vid, s in incumbent.items()}
    dist_before = _total_dist(ctx, work)

    ev = [0]
    moves = 0
    passes = 0
    hit = False
    moved_orders = []
    improved = True
    while improved and passes < max_passes and moves < max_moves:
        improved = False
        passes += 1
        while moves < max_moves:
            if deadline is not None and time.monotonic() > deadline:
                hit = True
                break
            done = False
            if 'relocate' in ops:
                before = {vid: list(s['route']) for vid, s in work.items()}
                done = _try_relocate(ctx, work, ev, per_move_certify)
                if done:
                    _record_move(before, work, moved_orders)
                    moves += 1
                    improved = True
            if not done and 'swap' in ops:
                before = {vid: list(s['route']) for vid, s in work.items()}
                done = _try_swap(ctx, work, ev, per_move_certify)
                if done:
                    _record_move(before, work, moved_orders)
                    moves += 1
                    improved = True
            if not done:
                break
        if hit:
            break

    after_prefix = frozen_prefixes(work)
    preserved = after_prefix == before_prefix
    if not preserved:
        raise LSError('冻结前缀被改动：%r → %r（搜索不得越过不可撤销的在途 leg）'
                      % (before_prefix, after_prefix))

    dist_after = _total_dist(ctx, work)
    cert = LSCertificate(
        neighborhood_id=NEIGHBORHOOD_ID, source_spec_sha256=SOURCE_SPEC_SHA256,
        passes=passes, moves_applied=moves, evaluations=ev[0],
        dist_before=dist_before, dist_after=dist_after,
        frozen_prefix_preserved=True, deadline_hit=hit, rng_used=False,
        certified=None, certify_mode=certify_mode, fallback_to_incumbent=False,
        moved_orders=tuple(moved_orders))
    return _finalize(ctx, incumbent, work, cert, certify_fn)


def _record_move(before, after, moved_orders):
    for vid, old in before.items():
        new = after[vid]['route']
        if old != new:
            moved_orders.append((int(vid), tuple(old), tuple(new)))


def _finalize(ctx, incumbent, work, cert, certify_fn):
    """末端认证 + 回退。返回 `(最终状态, cert)`。"""
    if certify_fn is None:
        cert.certified = None          # 未注入认证 ⇒ 明确记为"未认证"，不假装通过
        return work, cert
    ok, _ = certify_fn({vid: dict(s) for vid, s in work.items()})
    if ok:
        cert.certified = True
        return work, cert
    # 搜索后不可认证 ⇒ 回退 incumbent（并确认 incumbent 自身可认证）
    ok0, _ = certify_fn({vid: dict(s) for vid, s in incumbent.items()})
    if not ok0:
        raise LSError(
            '搜索后计划未过 certify_fn，且 incumbent 本身也未过 —— '
            '这不是 LS 能修的问题（调用方给的起点就不可认证）。'
            '请检查 state_from_vehicles 的锚点/frozen 构造。')
    cert.certified = False
    cert.fallback_to_incumbent = True
    cert.dist_after = _total_dist(ctx, incumbent)
    return {vid: dict(s, route=list(s['route'])) for vid, s in incumbent.items()}, cert


def ablate_to_myopic_scope(state, *, reveal, clock):
    """**近视消融**（§2.2 `--ablate-lookahead` 的路线侧对应物，只做只读切分）。

    前瞻与近视的区别在**候选订单集合**：前瞻会用场景内的未来订单评估后果，近视只在
    **当前已揭示**订单上评估。本函数把 `state` 里 `reveal > clock` 的订单**切**出去
    （返回 `(myopic_state, removed)`），供调用方构造近视臂。

    ⚠️ 它**只切分、不评估**：验收判据（"恒等于无前瞻版本 ⇒ 判实现缺陷 FAIL"；
    "一致率升高或降低都只记录、不赋方向"）由上层审计负责（规格 §2.2 修正稿）。
    被切出的订单若位于 `frozen` 前缀内 ⇒ 抛 `LSStateError`（在途 leg 不可撤销，
    不允许"近视"掉它）。
    """
    out, removed = {}, []
    for vid in sorted(state.keys()):
        s = state[vid]
        keep, drop = [], []
        for idx, o in enumerate(s['route']):
            (keep if float(reveal[o]) <= float(clock) + 1e-9 else drop).append((idx, o))
        if any(idx < s['frozen'] for idx, _ in drop):
            raise LSStateError(
                '车辆 %r 的未揭示订单落在 frozen 前缀内（%r）—— 在途 committed leg '
                '不可撤销，不允许被"近视"切掉。' % (vid, drop))
        out[vid] = dict(s, route=[o for _, o in keep])
        removed.extend((int(vid), int(o)) for _, o in drop)
    return out, tuple(removed)
