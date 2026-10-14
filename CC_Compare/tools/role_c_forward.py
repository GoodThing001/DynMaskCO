#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**前瞻评估** `Ĵ(b)`：在**共享场景集合**上、用 Role C **自己的** LS 求续行价值。

本模块只做**编排**：环境相关的部分全部注入（与 `role_c_opportunity_cost.decide_candidate`
同一风格），因此可完全离线单测、不 import 项目 env、不跑实验。

它在 Role C 里的位置（规格 §2.1 + §2.2）
--------------------------------------
```
decide_candidate(myopic_net_fn=…, marginal_energy_fn=…,
                 future_value_fn=<本模块产出的 Ĵ>, certify_fn=…)
        └─ shadow_price_budget(B, e_o, future_value_fn) → λ_B = clip((Ĵ(B) − Ĵ(B−e_o))/e_o, 0, ∞)
```
⇒ **`Ĵ` 在本规格里只有一个用途：估计 `λ_B`**（§2.1.1 写法 (i)）。
**不得**把 `Ĵ` 当作第二个价值项再加进接单判据（那正是被禁用的写法 (iii)）。

⚠️ **`Ĵ` 结构上不可能重复计入本单收益**：本函数的参数表里**没有**当前决策订单的收益
（`revenue_o`）这个入口 —— `plan_rev` 只由**状态里的订单**（既有计划 ∪ 场景内的未来订单）
用注入的 `order_revenue_fn` 算出。这是"接口不可表达"级别的保证，不是靠注释约定
（§2.1.1 要求"`Ĵ` 必须只含未来（不含 `revenue_o`）"）。

#### `inner_gate` 两档的**预算通道**（2026-10-14 实现时定死；规格 §2.2.2）

用户裁定"两档都实现、主档事前定死"。实现时必须回答一个规格没写的问题：
**去掉 `est` 闸门之后，预算还从哪个通道进 `Ĵ`？** 若"只是把闸门删掉"，则 `Ĵ` 完全不依赖
`b` ⇒ `v_hi == v_lo` ⇒ `λ_B ≡ 0` ⇒ 该档**退化成 A 版**，不构成有信息的消融。故定死为：

| 档 | 离散 `est` 闸门 | 终局能耗 | 与对照的关系 |
|---|---|---|---|
| `'on'`（**主档**） | ✅ 有（`est_used + e > b` ⇒ 拒该场景订单并计 `p_c`） | **不**罚超额（与对照 `scenario_saa.py:929` 一致） | 与 `cond_hist+SAA` **严格单因素**：场景集合、预算规则、承诺语义全同，**只换路线优化器** |
| `'off'`（机制消融） | ✗ 无 | **硬约束** `energy ≤ b`（真实协议口径），违反 ⇒ 有限不可行罚 | **两处**不同（闸门 + 终局能耗口径）⇒ 只可作机制消融，**不得**顶替主档 |

为什么 `'off'` 必须补终局硬约束：`certify_plan ≤ B` 才是协议对真实决策的**真约束**；
`est` 闸门只是对照的前瞻启发式。去掉启发式、改用真约束，才使 `Ĵ(b)` 重新依赖 `b`
（预算越少 ⇒ 越容易触碰硬约束 ⇒ `Ĵ` 越低），`λ_B` 才有定义。

⚠️ 另一条**必须照抄**的对照语义（`scenario_saa.py:923-928` 的根因修复）：
**终局能耗超标本身不是"不可行"**。对照曾对超标场景置 `−1e6`，结果逐场景淹没
`plan_rev/fuel/rej_loss`，40 天实测使投票退化为"`p_c>0` 全接 / `p_c=0` 全拒"的空转。
故 `'on'` 档**只**对 `energy is None`（TW/容量/返仓不可行）置有限不可行罚。
`'off'` 档的终局硬约束用**同一个有限罚**（不是 `−1e6`）。

#### 其余口径

* 场景内订单按 **`reveal` 升序**处理（与对照 `future_policy='reveal'` 同口径；
  对照的 `'density'` 顺序是另一个已登记因子，不在本模块默认里）。
* **确定性**：排序键 `(reveal, node)`，LS 无 RNG ⇒ 同一输入必得同一 `Ĵ`。
* **超时熔断**：超 `deadline` ⇒ 返回 `NaN` 并把 `deadline_hit=True` 写进诊断。
  `shadow_price_budget` 会把非有限 `V̂` 判为 **B3 未定义** ⇒ 事前边界规则**强制拒单**，
  正是对照的"超时降级 = 保持已认证计划 + 拒单"语义（fail-closed，不静默用半成品值）。
* **同场景集合**：`scenarios` 是**调用方传入**的同一对象 ⇒ 两档（乃至两臂）结构上共享；
  `scenario_digest()` 把它写成闭包哈希，便于在 T 系列里**证明**两臂用的是同一集合。
"""
from __future__ import annotations

import hashlib
import json
import time

import role_c_local_search as ls

__all__ = [
    'INNER_GATES', 'INNER_GATE_PRIMARY', 'INNER_GATE_SPEC', 'FORWARD_SPEC',
    'FORWARD_SPEC_SHA256',
    'ForwardError', 'ForwardSpace', 'scenario_digest', 'forward_value',
    'FutureValueFn',
]

INNER_GATES = ('on', 'off')
INNER_GATE_PRIMARY = 'on'                 # 规格 §2.2.2：**事前锁定**，不得按结果切换

INNER_GATE_SPEC = {
    'on': {'discrete_est_gate': True, 'terminal_energy_cap': False,
           'role': 'primary: strict single-factor vs cond_hist+SAA '
                   '(only the route optimizer differs)'},
    'off': {'discrete_est_gate': False, 'terminal_energy_cap': True,
            'role': 'mechanism ablation only: real protocol cap instead of the '
                    'control\'s pre-screen; MUST NOT replace the primary'},
}

FORWARD_SPEC = {
    'module': 'role_c_forward',
    'value': 'J_hat(b) = future-only continuation value over the SHARED scenario set',
    'used_for': 'lambda_B estimation ONLY (spec 2.1.1 formulation (i))',
    'forbidden': 'adding J_hat as a second value term (formulation (iii))',
    'future_order': 'reveal ascending, tie-break by node index',
    'route_search': 'role_c_local_search (route A, independent lineage)',
    'insertion': 'best_insert (min distance increment, pos >= frozen)',
    'polish': 'improve (relocate -> swap, first improvement, bounded, no RNG)',
    'terminal_infeasible': 'energy is None (TW/capacity/return) OR '
                           '(inner_gate=off AND energy > b) => finite penalty',
    'terminal_over_budget_when_gate_on': 'NOT penalised (mirrors scenario_saa '
                                         'root-cause fix; the est gate is the budget rule)',
    'rng': None,
    'timeout': 'return NaN => shadow_price_budget B3 undefined => forced reject',
    'revenue_o_channel': 'structurally absent from the signature',
}


def _spec_sha256(spec):
    blob = json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


FORWARD_SPEC_SHA256 = _spec_sha256(FORWARD_SPEC)


class ForwardError(ValueError):
    """前瞻评估的输入不合法（不可行罚非有限、预算为负、场景为空等）。fail-closed。"""


class ForwardSpace(object):
    """前瞻用的统一索引空间（**env 无关**；索引 `0` = depot）。

    `dist` 为 2D 索引（`dist[a][b]`），单位与 `fuel_cost_per_distance_unit` 一致。
    `return_deadline` 是回仓截止时刻（与对照的 `env.tw_end[inst_idx, 0]` 同义）。
    """

    __slots__ = ('dist', 'tws', 'twe', 'service', 'demand', 'reveal',
                 'capacity', 'tw_speed', 'return_deadline', 'tc', 'iq')

    def __init__(self, *, dist, tws, twe, service, demand, reveal,
                 capacity, tw_speed, return_deadline, tc=None, iq=None):
        self.dist = dist
        self.tws = tws
        self.twe = twe
        self.service = service
        self.demand = demand
        self.reveal = reveal
        self.capacity = float(capacity)
        self.tw_speed = float(tw_speed)
        self.return_deadline = float(return_deadline)
        # `tc` / `iq` 只供**终局 C0 完成**（物理）使用；路线搜索（LS）不读它们
        # —— 这正是 §2.1.2「能量只作硬约束、不进目标」的接口体现。
        self.tc = tc
        self.iq = iq

    def route_ctx(self):
        """本模块的 LS 上下文（`role_c_local_search.RouteContext`）。"""
        return ls.RouteContext(dist=self.dist, tws=self.tws, twe=self.twe,
                               service=self.service, demand=self.demand,
                               reveal=self.reveal, capacity=self.capacity,
                               tw_speed=self.tw_speed, deadline=self.return_deadline)


def scenario_digest(scenarios):
    """场景集合的**闭包哈希**（**双重顺序无关**）：用来证明两档/两臂用的是同一集合。

    场景之间与场景**内部**都排序 —— 因为前瞻按 `reveal` 升序处理场景内订单，
    输入列表的排列不影响 `Ĵ`，digest 也就不该受影响（否则同一集合会有多个 digest，
    "同场景集合"的证明反而变得不可用）。
    """
    rows = sorted('|'.join(str(o) for o in sorted(int(x) for x in scen))
                  for scen in scenarios)
    blob = '\n'.join(rows)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _assert_disjoint_from_plan(base_states, scenarios):
    """场景订单必须与**既有计划**不相交，否则 `best_insert` 会因"恰好一次"抛错。

    ⚠️ 为什么要在**入口**显式拦：若让 `LSStateError` 直接冒泡，`shadow_price_budget`
    会把它吞成 `undefined=True`（B3）⇒ 看起来像"预算稀缺导致强制拒单"，而实际上是
    **调用方的 id 空间错误**（场景订单复用了计划里的订单号）。把静默降级变成响亮标错。
    """
    in_plan = set()
    for s in base_states.values():
        in_plan.update(int(o) for o in s['route'])
    for scen in scenarios:
        dup = sorted(set(int(o) for o in scen) & in_plan)
        if dup:
            raise ForwardError(
                '场景订单与既有计划重叠：%r。这几乎总是调用方的 id 空间错误'
                '（场景订单必须用与真实订单不同的编号，如对照的临时编号空间）。'
                '若放任它抛 LSStateError，会被 shadow_price_budget 吞成 B3 "λ_B 未定义"'
                '⇒ 伪装成"预算稀缺⇒强制拒单"而掩盖真实缺陷。' % (dup,))


def _rev_ordered(scen, reveal):
    return sorted((int(o) for o in scen), key=lambda o: (float(reveal[o]), o))


def forward_value(*, space, base_states, scenarios, budget, initial_est_used,
                  order_energy_fn, order_revenue_fn, order_penalty_fn,
                  complete_fn, fuel_cost_per_distance_unit, infeasible_penalty,
                  inner_gate=INNER_GATE_PRIMARY, ls_kwargs=None, deadline=None):
    """在**给定**场景集合上算 `Ĵ(b)`。返回 `(value, diag)`。

    参数（除 `space` 外全部注入，故可离线单测）
    ----------------------------------------
    space : `ForwardSpace`（统一索引空间；`0` = depot）。
    base_states : LS 状态（`role_c_local_search.state_from_vehicles` 的产物）。
    scenarios : **同一**场景集合（list[list[node_idx]]）—— 两档/两臂必须传**同一对象**。
    budget : 本评估点的预算 `b`（`shadow_price_budget` 会用 `B` 与 `B − e_o` 调两次）。
    initial_est_used : 起点已占用的边际能耗估计（调用方按与对照**同一公式**算好；
        能耗记账是共享物理，不是 Role C 的设计选择）。
    order_energy_fn(node) / order_revenue_fn(node) / order_penalty_fn(node) :
        场景内订单的边际能耗 `e_o` / 收入 `R_o` / 拒单罚 `p_c`。
    complete_fn(state) -> (energy, dist) : 终局**真实 C0** 完成（TW/容量/返仓不可行 ⇒
        `(None, None)`）。注入以便离线测试；真实 runner 绑定项目 `transition_segment` 口径。
    fuel_cost_per_distance_unit : `KM_PER_UNIT × FUEL_COST_PER_KM`（**边际行驶燃料**口径；
        §2.1 要求写死为边际，不得用分摊/整日燃料）。
    infeasible_penalty : 场景不可行时的罚。**必须有限**（代码强制；`NaN`/`±inf` 会被拒），
        否则均值被污染、`Ĵ` 失去排序意义。**量级由调用方负责**：对照记录过一个真实事故
        （`scenario_saa.py:923-928`）—— 对终局超标场景置 `−1e6` 会淹没
        `plan_rev/fuel/rej_loss`，40 天实测使投票退化成"`p_c>0` 全接 / `p_c=0` 全拒"空转。
    inner_gate : `'on'`（主档）/ `'off'`（机制消融）—— 见模块头表格。
    deadline : 可选单调时钟熔断；超出 ⇒ `value = NaN` 且 `diag['deadline_hit']=True`。

    ⚠️ **不含 `revenue_o` 入口**（§2.1.1）：参数表里没有当前决策订单的收益。
    """
    if inner_gate not in INNER_GATES:
        raise ForwardError('inner_gate 只能是 %s，得到 %r'
                           % (' / '.join(INNER_GATES), inner_gate))
    pen = float(infeasible_penalty)
    if pen != pen or pen in (float('inf'), float('-inf')):
        raise ForwardError('infeasible_penalty 必须有限（得到 %r）：`−1e6`/`−inf` 会淹没'
                           '场景相关的价值信息（对照 §923-928 记录的 40 天空转事故）' % (pen,))
    b = float(budget)
    if b < 0.0:
        raise ForwardError('budget 不得为负：%r' % (b,))
    scenarios = [list(s) for s in scenarios]
    if not scenarios:
        raise ForwardError('场景集合为空 ⇒ Ĵ 无定义（不得用空集合冒充已评估）')
    _assert_disjoint_from_plan(ls.normalize_state(base_states), scenarios)
    fcu = float(fuel_cost_per_distance_unit)
    ctx = space.route_ctx()
    kw = dict(ls_kwargs or {})
    cap_on = INNER_GATE_SPEC[inner_gate]['terminal_energy_cap']

    diag = {'inner_gate': inner_gate, 'budget': b, 'n_scenarios': len(scenarios),
            'scenario_digest': scenario_digest(scenarios),
            'forward_spec_sha256': FORWARD_SPEC_SHA256,
            'neighborhood_id': ls.NEIGHBORHOOD_ID,
            'source_spec_sha256': ls.SOURCE_SPEC_SHA256,
            'deadline_hit': False, 'n_infeasible_scenario': 0,
            'n_gate_reject': 0, 'n_insert_reject': 0, 'n_future_accept': 0,
            'value': None}

    total = 0.0
    for scen in scenarios:
        if deadline is not None and time.monotonic() > deadline:
            diag['deadline_hit'] = True
            diag['value'] = float('nan')
            return float('nan'), diag
        st = {vid: dict(s, route=list(s['route']))
              for vid, s in ls.normalize_state(base_states).items()}
        est_used = float(initial_est_used)
        rej_loss = 0.0
        for o in _rev_ordered(scen, space.reveal):
            e = float(order_energy_fn(o))
            if INNER_GATE_SPEC[inner_gate]['discrete_est_gate'] \
                    and est_used + e > b + 1e-9:
                rej_loss += float(order_penalty_fn(o))
                diag['n_gate_reject'] += 1
                continue
            st2, ok = ls.best_insert(ctx, st, o)
            if not ok:
                rej_loss += float(order_penalty_fn(o))
                diag['n_insert_reject'] += 1
                continue
            # §2.2：Role C 用**自己的**前瞻路线优化（LS）打磨插入后的计划
            st2, _cert = ls.improve(ctx, st2, **kw)
            st = st2
            est_used += e
            diag['n_future_accept'] += 1
        energy, dist = complete_fn(st)
        over_cap = (cap_on and energy is not None and float(energy) > b + 1e-9)
        if energy is None or over_cap:
            diag['n_infeasible_scenario'] += 1
            total += pen
            continue
        plan_rev = sum(float(order_revenue_fn(o)) for s in st.values()
                       for o in s['route'])
        total += plan_rev - float(dist) * fcu - rej_loss
    value = total / float(len(scenarios))
    diag['value'] = value
    return value, diag


class FutureValueFn(object):
    """可调用对象 `Ĵ(b)`，供 `shadow_price_budget(budget, e_o, future_value_fn)` 注入。

    用法：
        fvf = FutureValueFn(**common_kwargs)      # 除 budget / initial_est_used 外全给
        lam, lam_diag = shadow_price_budget(B, e_o, fvf)
        fvf.last_diag                             # 最近一次评估的诊断（供 T2a/T2b 接线）

    `initial_est_used` 若给了**函数**则以 `b` 为参数求值（因为"已占用"随预算点不变，
    但调用方可能想按 `b` 缩放待命预留）；给常数则直接用。
    """

    def __init__(self, *, space, base_states, scenarios, order_energy_fn,
                 order_revenue_fn, order_penalty_fn, complete_fn,
                 fuel_cost_per_distance_unit, infeasible_penalty,
                 initial_est_used, inner_gate=INNER_GATE_PRIMARY,
                 ls_kwargs=None, deadline=None):
        if inner_gate not in INNER_GATES:
            raise ForwardError('inner_gate 只能是 %s，得到 %r'
                               % (' / '.join(INNER_GATES), inner_gate))
        self.space = space
        self.base_states = base_states
        self.scenarios = scenarios
        self.order_energy_fn = order_energy_fn
        self.order_revenue_fn = order_revenue_fn
        self.order_penalty_fn = order_penalty_fn
        self.complete_fn = complete_fn
        self.fuel_cost_per_distance_unit = fuel_cost_per_distance_unit
        self.infeasible_penalty = infeasible_penalty
        self.initial_est_used = initial_est_used
        self.inner_gate = inner_gate
        self.ls_kwargs = ls_kwargs
        self.deadline = deadline
        self.calls = []
        self.last_diag = None
        # 构造期就拦"场景订单与计划重叠"：runner 在**装配阶段**失败，
        # 而不是在每个决策点静默退化成 B3 强制拒单。
        _assert_disjoint_from_plan(ls.normalize_state(base_states),
                                   [list(s) for s in scenarios])

    def __call__(self, b):
        ies = self.initial_est_used
        ies = ies(float(b)) if callable(ies) else float(ies)
        value, diag = forward_value(
            space=self.space, base_states=self.base_states,
            scenarios=self.scenarios, budget=float(b), initial_est_used=ies,
            order_energy_fn=self.order_energy_fn,
            order_revenue_fn=self.order_revenue_fn,
            order_penalty_fn=self.order_penalty_fn, complete_fn=self.complete_fn,
            fuel_cost_per_distance_unit=self.fuel_cost_per_distance_unit,
            infeasible_penalty=self.infeasible_penalty,
            inner_gate=self.inner_gate, ls_kwargs=self.ls_kwargs,
            deadline=self.deadline)
        self.calls.append((float(b), value))
        self.last_diag = diag
        return value

    def scenario_digest(self):
        return scenario_digest(self.scenarios)
