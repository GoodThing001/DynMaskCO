#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**装配层**：共享场景流 + 统一索引空间 + 项目物理绑定 + `decide_fn` 组装。

这一层回答的是"零件怎么接起来"，而**不是**"机制是什么"：
机制在 `role_c_local_search`（路线，路线 A）/ `role_c_forward`（`Ĵ`）/
`role_c_opportunity_cost`（`myopic_net`、`λ_B`、写法 (i)）里，本模块不重新定义它们。

## 1. 场景流为什么是"两臂**可证明**共用"的（不是约定）

用户边界要求"两臂共用同一场景集合"。逐行核对 `scenario_saa.py` 后可以把它变成**定理**，
而不是靠约定（这很重要：约定会在某次重构后静默失效）：

* `CondHistoricalSampler.sample(snapshot, rng, k)` **只读** `snapshot.clock` 与
  `snapshot.orders` 的 `reveal / temp_class / x / y` 四个字段（`_feat`，`:162-171`）；
  `snapshot.accepted / rejected / vehicles / energy_used / budget / capacity` **一律不读**。
* `CondHistoricalSampler.sample` 的随机性**只**来自 `rng.integers(0, len(kn))`（`:183`），
  而对照侧 `rng = np.random.default_rng([arm_seed, inst_idx, round(clock*1000)])`（`:1002`）。
* `_sync_id_map`（`:306-313`）按 `demands > 0 且 reveal_time <= clock` 选"已揭示"订单
  —— **与接单/拒单决策无关**；`_real_order`（`:323-331`）只读 env 数组。
  ⇒ `_feat(snapshot.orders, clock)` 是 `(env, inst_idx, clock)` 的**纯函数**。

**推论**：同一实例、同一 `clock`、同一 sampler 实例、同一 `arm_seed`、同一 `K`
⇒ 两臂抽到的场景集合**逐字段相同**（`rng` 序列一致、近邻排序一致、`argsort` 确定）。
`build_space` 与决策只在**抽完之后**才分叉，故不污染抽样。
本模块的 `scenario_digest` 把该集合写成哈希，装配期可对两臂**对拍**。

⚠️ **冻结配置的一个缺口（2026-10-14 只读审计发现，未擅自改冻结文件）**：
`P1A_CONTROL_FROZEN.json` 的 `frozen_config` **没有 `arm_seed`**，而它**完整决定**场景流。
实际取值可从**产出该冻结证据的 runner** 读出：`run_a1_step2_gate.py:51`
`ARM_SEEDS = {"uncond_hist": 7001, "cond_hist": 7002, "explicit_feat": 7003, ...}`
（`diag_local_timing.py:23` 一致）⇒ **主对照 `cond_hist` 的 `arm_seed = 7002`**，
**不是** `SaaReplanner` 的默认值 `7001`。若谁按默认值构造对照，会拿到**另一条场景流**
且没有任何报错。因此本模块把 `arm_seed` 定为**必需的关键字参数（无默认值）** ——
让"用错种子"在装配期就变成 `TypeError`，而不是静默换掉场景集合。
（是否把该字段补进冻结文件属**预注册变更**，由用户裁定；本模块不代改。）

## 2. 与对照的边界（哪些能 import，哪些绝不能）

规格 §2.2.1 前提①要求 Role C 的路线搜索**不 import / 不继承对照血统**。本模块把该边界
写成**可机械检查的清单**（`CONTROL_ENGINE_SYMBOLS` + `assert_no_control_engine_use`）：
可以共享的是**协议部件**（`CondHistoricalSampler` 场景抽样、`certify_plan` 认证、
C0 物理 `transition_segment` 等），**绝不能**碰的是对照的**决策引擎**
（`SaaReplanner` 及其 `_saa_decide` / `_sim_scenario` / `_local_search_u` / `_insert_u` /
`_complete_u` / `_sim_states` / `_build_space` 等）。后者若被调用，`C − B` 就不再单因素。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import role_c_forward as fwd
import role_c_local_search as ls
import role_c_opportunity_cost as oc

__all__ = [
    'HarnessError', 'SAMPLER_RNG_CONVENTION', 'CONTROL_ENGINE_SYMBOLS',
    'ROLE_C_DEPENDENCY_ROLES', 'RevealedOrder', 'SpaceBundle',
    'scenario_rng', 'revealed_order_view', 'sample_scenarios',
    'build_space_bundle', 'assert_no_control_engine_use',
    'make_order_energy_fn', 'make_order_revenue_fn', 'make_order_penalty_fn',
    'make_c0_complete_fn', 'initial_est_used', 'build_decide_fn',
]

# 对照的**决策引擎**符号：Role C 的代码路径里不得出现（场景抽样/认证/物理**不在此列**）
CONTROL_ENGINE_SYMBOLS = (
    'SaaReplanner', '_saa_decide', '_sim_scenario', '_local_search_u', '_insert_u',
    '_greedy_insert_u', '_complete_u', '_sim_states', '_build_space', '_build_snapshot',
    '_route_feasible_u', '_route_dist_u', '_try_insert_certified', 'overage_price',
)

# `provider_dependency_sha256` 闭包的**角色申报**（每条都必须写明"为什么进来"）
ROLE_C_DEPENDENCY_ROLES = {
    'CC_Compare/tools/role_c_local_search.py':
        'decision: independent route search (spec 2.2.1 route A)',
    'CC_Compare/tools/role_c_forward.py':
        'decision: forward value J_hat for lambda_B (spec 2.1.1)',
    'CC_Compare/tools/role_c_opportunity_cost.py':
        'decision: myopic_net + lambda_B + formulation (i)',
    'CC_Compare/tools/role_c_replanner.py':
        'wiring: online replanner skeleton (transactional, causal fail-closed)',
    'CC_Compare/tools/role_c_harness.py':
        'wiring: shared scenario stream + space + physics binding',
    'CC_Compare/tools/role_c_controls.py':
        'control arms: B-density / B-shadow (same identity required)',
    'CC_Compare/tools/role_c_decision_log.py':
        'evidence: write-time-validated decision log (T2-T7 inputs)',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/scenario_saa.py':
        'shared_sampler_only: CondHistoricalSampler + make_c0_contract_v2 / ScenarioOrder '
        '(the control decision engine in the same file is NOT used)',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_exp_energy_c0.py':
        'shared_protocol: certify_plan (hard C0 budget) + c0_marginal_energy',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py':
        'shared_physics: transition_segment / create_vehicle_state / dispatch_vehicle',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/coldchain_evaluator_a1.py':
        'shared_objective: REV / FUEL_COST_PER_KM / KM_PER_UNIT',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py':
        'shared_protocol: StrictOnlineEnv event engine',
}

SAMPLER_RNG_CONVENTION = ("np.random.default_rng([arm_seed, inst_idx, "
                          "int(round(clock * 1000))])   # scenario_saa.py:1002")


class HarnessError(ValueError):
    """装配层输入不合法。fail-closed。"""


@dataclass(frozen=True)
class RevealedOrder:
    """已揭示订单视图。**只**需 `CondHistoricalSampler._feat` 读的四个字段正确
    （`reveal / temp_class / x / y`）；其余字段为完整性保留，抽样不读。"""
    oid: int
    reveal: float
    x: float
    y: float
    demand: float
    temp_class: int
    tw_start: float
    tw_end: float
    service_time: float


class SpaceBundle(object):
    """统一索引空间 + 场景节点清单（索引 `n_real + i` 对应 `scen_nodes[i]`）。"""

    __slots__ = ('space', 'scen_nodes', 'n_real', 'scen_indices')

    def __init__(self, space, scen_nodes, n_real, scen_indices):
        self.space = space
        self.scen_nodes = scen_nodes
        self.n_real = n_real
        self.scen_indices = scen_indices


# ---------------------------------------------------------------------------
# 场景流
# ---------------------------------------------------------------------------
def scenario_rng(*, arm_seed, inst_idx, clock):
    """对照侧**同一条** rng 构造（`scenario_saa.py:1002`）。

    ⚠️ `arm_seed` 是**必需的关键字参数、没有默认值**：冻结配置漏记了它，而它决定整条
    场景流；给它默认值就等于允许"静默换掉场景集合"。用错种子必须**报错**，不是悄悄生效。
    """
    if arm_seed is None or isinstance(arm_seed, bool) or not isinstance(arm_seed, int):
        raise HarnessError(
            'arm_seed 必须显式给出 int（当前 %r）。冻结配置 P1A_CONTROL_FROZEN.json 的 '
            'frozen_config **没有记录**该字段，但主对照 cond_hist 的实际取值是 **7002**'
            '（run_a1_step2_gate.py:51 的 ARM_SEEDS；diag_local_timing.py:23 一致），'
            '**不是** SaaReplanner 的默认 7001。它完整决定场景流 ⇒ 必须显式传入。'
            % (arm_seed,))
    return np.random.default_rng([int(arm_seed), int(inst_idx),
                                  int(round(float(clock) * 1000))])


def revealed_order_view(env, inst_idx, clock):
    """已揭示订单视图（**独立实现**，不调用对照的 `_sync_id_map` / `_real_order`）。

    判定与对照 `_sync_id_map` 同义：`demands > 0 且 reveal_time <= clock + 1e-6`，
    按真实索引升序 ⇒ 与对照的 `_feat` 输入是**同一个多重集**（这是"同场景集合"的前提）。
    """
    dem = env.demands[inst_idx]
    rev = env.reveal_time[inst_idx]
    tc = env.dataset['temp_class'][inst_idx]
    out = []
    for i in range(1, int(env.num_nodes)):
        if float(dem[i]) > 0.0 and float(rev[i]) <= float(clock) + 1e-6:
            out.append(RevealedOrder(
                oid=int(i), reveal=float(rev[i]),
                x=float(env.coords[inst_idx, i, 0]),
                y=float(env.coords[inst_idx, i, 1]),
                demand=float(dem[i]), temp_class=int(tc[i]),
                tw_start=float(env.tw_start[inst_idx, i]),
                tw_end=float(env.tw_end[inst_idx, i]),
                service_time=float(env.service_time[inst_idx, i])))
    return out


def sample_scenarios(*, sampler, env, inst_idx, clock, arm_seed, K):
    """在 `(inst_idx, clock)` 上抽 `K` 个场景（与对照**同 rng、同 sampler、同 K**）。

    返回 `(scenarios, diag)`。`diag['scenario_digest']` 供两臂装配期对拍。
    """
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise HarnessError('K 必须是 >=1 的 int，得到 %r' % (K,))
    snap = _Snapshot(clock=float(clock), orders=tuple(revealed_order_view(env, inst_idx, clock)))
    rng = scenario_rng(arm_seed=arm_seed, inst_idx=inst_idx, clock=clock)
    scenarios = sampler.sample(snap, rng, int(K))
    diag = {'arm_seed': int(arm_seed), 'K': int(K), 'clock': float(clock),
            'rng_convention': SAMPLER_RNG_CONVENTION,
            'n_revealed_orders': len(snap.orders),
            'info_max_reveal': max((o.reveal for o in snap.orders), default=float(clock)),
            'scenario_digest': fwd.scenario_digest(
                [[getattr(o, 'oid', i) for i, o in enumerate(scen)] for scen in scenarios]),
            'n_scenarios': len(scenarios)}
    return scenarios, diag


class _Snapshot(object):
    """**最小**快照：只提供 `CondHistoricalSampler` 真正读的两个字段。

    不构造 `accepted/rejected/vehicles/energy_used/budget/capacity` —— 它们对
    `CondHistoricalSampler.sample` 是死字段；照着对照的完整 `VisibleSnapshot` 抄一遍
    只会把对照的数据结构拖进 Role C 的依赖，而**不增加任何**场景流一致性。
    """
    __slots__ = ('clock', 'orders')

    def __init__(self, clock, orders):
        self.clock = float(clock)
        self.orders = tuple(orders)


# ---------------------------------------------------------------------------
# 统一索引空间
# ---------------------------------------------------------------------------
def build_space_bundle(*, env, inst_idx, scenarios, contract):
    """把真实节点 + 场景节点拼成统一索引空间（`0` = depot，`n_real + i` = 场景单）。

    与对照 `_build_space` **同语义**（同一套坐标→距离/TW/需求口径），但**独立实现**：
    场景流共享不等于"复用对照代码"，两者必须同口径，否则 `C − B` 里混入了空间口径差。
    """
    from coldchain_contract import QualityConfigV2

    scen_nodes = [o for scen in scenarios for o in scen]
    n_real = int(env.coords.shape[1])
    coords = np.asarray(env.coords[inst_idx], dtype=np.float64)
    if scen_nodes:
        scoords = np.array([[float(o.x), float(o.y)] for o in scen_nodes], dtype=np.float64)
        coords_all = np.vstack([coords, scoords])
    else:
        coords_all = coords
    diff = coords_all[:, None, :] - coords_all[None, :, :]
    is_v2 = isinstance(contract.quality, QualityConfigV2)
    iq_real = np.array([float(contract.quality.initial_value[int(c)]) if is_v2 else 1.0
                        for c in env.dataset['temp_class'][inst_idx]], dtype=np.float64)
    iq_scen = np.array(
        [float(contract.quality.initial_value[int(o.temp_class)]) if is_v2 else 1.0
         for o in scen_nodes], dtype=np.float64)
    iq = np.concatenate([iq_real, iq_scen]) if scen_nodes else iq_real
    tc = np.concatenate(
        [np.asarray(env.dataset['temp_class'][inst_idx], np.int64),
         np.array([int(o.temp_class) for o in scen_nodes], np.int64)]) if scen_nodes \
        else np.asarray(env.dataset['temp_class'][inst_idx], np.int64)
    space = fwd.ForwardSpace(
        dist=np.sqrt((diff ** 2).sum(-1)),
        tws=np.concatenate([np.asarray(env.tw_start[inst_idx], np.float64),
                            np.array([float(o.tw_start) for o in scen_nodes], np.float64)])
        if scen_nodes else np.asarray(env.tw_start[inst_idx], np.float64),
        twe=np.concatenate([np.asarray(env.tw_end[inst_idx], np.float64),
                            np.array([float(o.tw_end) for o in scen_nodes], np.float64)])
        if scen_nodes else np.asarray(env.tw_end[inst_idx], np.float64),
        service=np.concatenate([np.asarray(env.service_time[inst_idx], np.float64),
                                np.array([float(o.service_time) for o in scen_nodes],
                                         np.float64)])
        if scen_nodes else np.asarray(env.service_time[inst_idx], np.float64),
        demand=np.concatenate([np.asarray(env.demands[inst_idx], np.float64),
                               np.array([float(o.demand) for o in scen_nodes], np.float64)])
        if scen_nodes else np.asarray(env.demands[inst_idx], np.float64),
        reveal=np.concatenate([np.asarray(env.reveal_time[inst_idx], np.float64),
                               np.array([float(o.reveal) for o in scen_nodes], np.float64)])
        if scen_nodes else np.asarray(env.reveal_time[inst_idx], np.float64),
        capacity=float(env.capacity), tw_speed=float(env.tw_speed),
        return_deadline=float(env.tw_end[inst_idx, 0]), tc=tc, iq=iq)
    idx = {id(o): n_real + i for i, o in enumerate(scen_nodes)}
    scen_indices = [[idx[id(o)] for o in scen] for scen in scenarios]
    return SpaceBundle(space, scen_nodes, n_real, scen_indices)


def assert_no_control_engine_use(modules):
    """**机械**检查：给定模块对象里不得出现对照决策引擎的符号。

    `modules` 为 `{名字: 模块对象}`。返回被污染的 `[(模块, 符号), ...]`（空 = 干净）。
    场景抽样 / `certify_plan` / C0 物理**不**在禁用清单里（它们是共享协议部件）。
    """
    bad = []
    for name, mod in modules.items():
        for sym in CONTROL_ENGINE_SYMBOLS:
            if hasattr(mod, sym):
                bad.append((name, sym))
    return bad


# ---------------------------------------------------------------------------
# 物理 / 目标绑定
# ---------------------------------------------------------------------------
def make_order_energy_fn(*, space, cooling_share, contract,
                         energy_pricing='marginal'):
    """`e_o = c0_marginal_energy(...)`（与既有实现**同一物理口径**）。

    ⚠️ **必须按 `space` 取温区/需求，不能按 `env` 数组取**：场景订单的索引是
    `n_real + i`，**超出** `env.dataset['temp_class'][inst_idx]` 的长度。第一版按 `env`
    索引 ⇒ 场景节点 `IndexError` ⇒ 异常被 `shadow_price_budget` 吞成 B3「λ 未定义」
    ⇒ 全部强制拒单，**而且看起来像机制在正常工作**（这类静默降级正是要防的）。
    `space.tc` / `space.demand` 已把真实节点与场景节点拼在一起，是唯一正确的索引源。

    `energy_pricing='marginal'`（冻结值）⇒ 只计 door/COP + precool，固定待命另行按
    `standby_orders` 预留（见 `initial_est_used`）；`'amortized'` 则把 `cooling_share`
    直接并进每单。
    """
    from run_exp_energy_c0 import c0_marginal_energy

    share = float(cooling_share) if energy_pricing != 'marginal' else 0.0
    tc, dem = space.tc, space.demand
    return lambda o: float(c0_marginal_energy(int(tc[int(o)]), float(dem[int(o)]),
                                              share, contract))


def make_order_revenue_fn(*, space):
    from coldchain_evaluator_a1 import REV
    tc = space.tc
    return lambda o: float(REV[int(tc[int(o)])])


def make_order_penalty_fn(*, space, reject_penalty):
    tc = space.tc
    return lambda o: float(reject_penalty[int(tc[int(o)])])


def make_c0_complete_fn(*, bundle, contract, booking_horizon, physics=None):
    """终局完成：路线 + **WAIT 到 `booking_horizon`** + 返仓，真实 C0 能耗与距离。

    返回 `fn(state) -> (energy | None, dist | None)`；`(None, None)` 表示 TW/容量/返仓
    **真实不可行**。返回**能耗与距离**而不是布尔，是为了让调用方（而不是本模块）决定
    "能耗超标算不算不可行" —— 那由 `inner_gate` 的预算通道决定（规格 §2.2.2），
    且在 `'on'` 档必须**不**罚（照抄对照 `scenario_saa.py:923-928` 的根因修复）。

    与对照 `_complete_u` **同语义、独立实现**（同一套共享物理）。
    """
    if physics is None:
        from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                                     transition_segment)
        physics = {'transition_segment': transition_segment,
                   'create_vehicle_state': create_vehicle_state,
                   'dispatch_vehicle': dispatch_vehicle}
    transition_segment = physics['transition_segment']
    space = bundle.space
    speed = float(contract.units.speed_kmph) / float(contract.units.distance_km_per_unit)
    zones = (True,) * len(contract.thermal.supported_temp_classes)
    wait_until = None if booking_horizon is None else float(booking_horizon)
    deadline = float(space.return_deadline)

    def _complete(state):
        total_e = 0.0
        total_d = 0.0
        for s in state.values():
            if not s['route'] and int(s['cur']) == 0:
                continue                      # 未出车且空计划：不 dispatch、不产生幻影预冷
            vstate = physics['dispatch_vehicle'](
                physics['create_vehicle_state'](contract), contract)
            cur, t = int(s['cur']), float(s['cur_time'])
            d_acc = 0.0
            for o in s['route']:
                d = float(space.dist[cur][int(o)])
                rv = float(space.reveal[int(o)])
                depart = t if t >= rv else rv      # 揭示感知出发（未揭示不得提前行驶）
                arrive = depart + d / speed
                sstart = arrive if arrive >= float(space.tws[int(o)]) \
                    else float(space.tws[int(o)])
                if sstart > float(space.twe[int(o)]) + 1e-6:
                    return None, None
                sfinish = sstart + float(space.service[int(o)])
                try:
                    vstate, _ = transition_segment(
                        vstate, depart_time=depart, arrival_time=arrive,
                        service_finish=sfinish, served_customer=int(o),
                        active_zone_mask=zones, contract=contract,
                        order_quantity=float(space.demand[int(o)]),
                        order_temp_class=int(space.tc[int(o)]),
                        initial_quality=float(space.iq[int(o)]),
                        segment_distance_units=d)
                except ValueError:
                    return None, None             # 物理边界硬违反 ⇒ 该场景不可行
                d_acc += d
                t = sfinish
                cur = int(o)
            if wait_until is not None and wait_until > t + 1e-9:
                vstate, _ = transition_segment(
                    vstate, depart_time=t, arrival_time=wait_until,
                    service_finish=wait_until, served_customer=None,
                    active_zone_mask=zones, contract=contract,
                    segment_distance_units=0.0)
                t = wait_until
            d = float(space.dist[cur][0])
            arrive = t + d / speed
            if arrive > deadline + 1e-6:
                return None, None
            vstate, _ = transition_segment(
                vstate, depart_time=t, arrival_time=arrive, service_finish=arrive,
                served_customer=None, active_zone_mask=zones, contract=contract,
                return_to_depot=True, segment_distance_units=d)
            total_e += float(vstate.cumulative_energy_kwh)
            total_d += d_acc + d
        return total_e, total_d

    return _complete


def initial_est_used(*, env, inst_idx, vehicles, plan, served_mask, order_energy_fn,
                     cooling_share, energy_pricing='marginal', standby_orders=None):
    """起点已占用能耗估计（与对照 `_sim_scenario` 同口径的**共享记账**）。

    = 真实已耗（`coldchain_state.cumulative_energy_kwh`）+ 当前计划逐单 `est`
      + （`marginal` 且给了 `standby_orders`）`cooling_share × standby_orders` 全天预留。
    """
    used = sum(float(v.coldchain_state.cumulative_energy_kwh) for v in vehicles
               if getattr(v, 'coldchain_state', None) is not None)
    for v in vehicles:
        for o in plan.get(int(v.vehicle_id), []):
            if not served_mask[int(o)]:
                used += float(order_energy_fn(int(o)))
    if energy_pricing == 'marginal' and standby_orders:
        used += float(cooling_share) * float(standby_orders)
    return used


# ---------------------------------------------------------------------------
# decide_fn 组装
# ---------------------------------------------------------------------------
def build_decide_fn(*, sampler, arm_seed, K, contract, cooling_share, booking_horizon,
                    budget, inner_gate=fwd.INNER_GATE_PRIMARY, ls_kwargs=None,
                    energy_pricing='marginal', standby_orders=None, time_limit=10.0,
                    infeasible_penalty=-1e5, certify_plan_fn=None,
                    ablate_opportunity_cost=False, ablate_lookahead=False,
                    ablate_scenarios=None):
    """组装 `RoleCReplanner` 需要的 `decide_fn(order, ctx) -> (accept, diag)`。

    流程（严格按规格 §2.4 顺序：候选评估 → 认证 → 提交；提交由 `RoleCReplanner` 做）：
      1. 由 `ctx` 的真实车辆/计划构造 LS 状态（真锚点 + 显式 `frozen`）；
      2. 抽**共享**场景集合（同 sampler / 同 `arm_seed` / 同 `K` / 同 rng 构造），
         并记录 `scenario_digest`；
      3. 构造统一索引空间与物理绑定；
      4. 候选计划 = `best_insert` + `improve`（Role C **自己的**路线搜索）；
      5. `decide_candidate`：`myopic_net − λ_B·e_o > 0 ∧ certify_ok`（写法 (i)）。

    ⚠️ `inner_gate` 缺省 = `'on'`（规格 §2.2.2 事前锁定的主档），**不得**按结果切换。

    **消融钩子（T2/T3/T4 需要，缺省全关）**：
      * `ablate_opportunity_cost=True` ⇒ 保留场景前瞻但 `λ_B ≡ 0`（A–F 表的 **A 版**）；
      * `ablate_lookahead=True` ⇒ 不做前瞻 ⇒ `Ĵ` 无从估计 ⇒ `λ_B` **必然也只能是 0**
        （与 A 版机制不同：A 版仍持有场景机制）；
      * `ablate_scenarios` ∈ {None, `'k1'`, `'shuffle'`}：`'k1'` 把场景数截到 **1**；
        `'shuffle'` 把场景内未来订单改为**降序**处理。两者都**不是**无操作 ——
        单纯重排"取平均"的场景列表是**恒等变换**，拿它当消融会得到"消融前后完全相同"的
        假 FAIL，故这里用真正会改变贪心打包的两个变体。
    """
    if inner_gate not in fwd.INNER_GATES:
        raise HarnessError('inner_gate 只能是 %s' % (' / '.join(fwd.INNER_GATES),))
    if ablate_scenarios not in (None, 'k1', 'shuffle'):
        raise HarnessError("ablate_scenarios 只能是 None / 'k1' / 'shuffle'，得到 %r"
                           % (ablate_scenarios,))
    if not isinstance(arm_seed, int) or isinstance(arm_seed, bool):
        raise HarnessError('arm_seed 必须是显式 int（见 scenario_rng 的说明），得到 %r'
                           % (arm_seed,))
    if certify_plan_fn is None:
        from run_exp_energy_c0 import certify_plan as certify_plan_fn
    last = {}
    K_eff = 1 if ablate_scenarios == 'k1' else int(K)
    future_order = 'reveal_desc' if ablate_scenarios == 'shuffle' else 'reveal_asc'

    def decide_fn(order, ctx):
        env, inst_idx = ctx['env'], ctx['inst_idx']
        clock = float(ctx['clock'])
        vehicles, served_mask = ctx['vehicles'], ctx['served_mask']
        plan = {int(k): list(v) for k, v in (ctx.get('plan') or {}).items()}
        t0 = time.monotonic()
        deadline = t0 + float(time_limit)
        # ① 真锚点 + 显式 frozen 的状态
        states = ls.state_from_vehicles(vehicles, served_mask, plan, clock)
        # ② 共享场景集合
        scenarios, sdiag = sample_scenarios(sampler=sampler, env=env, inst_idx=inst_idx,
                                            clock=clock, arm_seed=arm_seed, K=K_eff)
        # ③ 空间 + 物理
        bundle = build_space_bundle(env=env, inst_idx=inst_idx, scenarios=scenarios,
                                    contract=contract)
        energy_fn = make_order_energy_fn(space=bundle.space, cooling_share=cooling_share,
                                        contract=contract, energy_pricing=energy_pricing)
        rev_fn = make_order_revenue_fn(space=bundle.space)
        pen_fn = make_order_penalty_fn(space=bundle.space,
                                       reject_penalty=ctx.get('reject_penalty')
                                       or {0: 5.0, 1: 10.0, 2: 15.0})
        complete = make_c0_complete_fn(bundle=bundle, contract=contract,
                                       booking_horizon=booking_horizon)
        # ④ 候选计划（Role C 自己的路线搜索）
        st2, inserted = ls.best_insert(bundle.space.route_ctx(), states, int(order))
        if not inserted:
            # ⚠️ **接单必须可兑现**：无处可插 ⇒ 该单不可能被服务。若这里仍让它走完
            # `decide_candidate` 并接受，就会记录一个「已接受但不在任何计划里」的承诺
            # ⇒ `evaluate_trace_a1` 的 accepted ⊆ served 直接判协议违规。
            # 对照的处理相同：`scenario_saa.py:956-958` **先**判插入失败即拒单、再跑 SAA。
            # 因此这里是**硬短路**（不估 λ、不认证）：计划未变，"认证 incumbent"没有意义。
            diag = {'boundary': 'insertion_infeasible',
                    'boundary_reason': '无任何可行插入位 ⇒ 该单无法被服务 ⇒ 强制拒单'
                                       '（避免"已接受但不可服务"的承诺）',
                    'e_o': float(energy_fn(int(order))),
                    'lambda_B': None, 'opportunity_cost': None,
                    'net_after_opportunity_cost': None,
                    'myopic_net': None, 'certify_ok': False, 'certify_energy': None,
                    'lambda_diag': {'lambda_source': 'boundary_short_circuit',
                                    'undefined': False},
                    'lambda_estimator': oc.LAMBDA_ESTIMATOR_SHADOW,
                    'myopic_base': oc.MYOPIC_BASE_COMPONENTS,
                    'budget_B': float(budget), 'ablate_lookahead': False,
                    'reject_reason': 'insertion_infeasible',
                    'formulation': '(i) myopic_net − λ_B·e_o > 0 ∧ certify_ok',
                    'plan': None, 'day_id': inst_idx,
                    'info_max_reveal': sdiag['info_max_reveal'],
                    'scenario': sdiag, 'inner_gate': inner_gate,
                    'arm_seed': int(arm_seed), 'elapsed_s': time.monotonic() - t0,
                    'log_fields': {'order_id': getattr(order, 'order_id', None),
                                   'action': 'reject', 'certified': False,
                                   'certify_energy': None}}
            last['diag'] = diag
            return False, diag
        st2, _cert = ls.improve(bundle.space.route_ctx(), st2, **(ls_kwargs or {}))
        cand_plan = ls.plan_of(st2)

        def certify_one(p):
            ok, e = certify_plan_fn(env, inst_idx, clock, vehicles, served_mask,
                                   {int(k): list(v) for k, v in p.items()},
                                   contract, float(budget))
            return bool(ok), e

        def myopic_net_of(o):
            o = int(o)
            return oc.myopic_net(revenue=rev_fn(o),
                                 fuel=_insertion_fuel(bundle, states, o, energy_fn),
                                 reject_loss=pen_fn(o))

        fvf = fwd.FutureValueFn(
            space=bundle.space, base_states=states, scenarios=bundle.scen_indices,
            order_energy_fn=energy_fn, order_revenue_fn=rev_fn,
            order_penalty_fn=pen_fn, complete_fn=complete,
            fuel_cost_per_distance_unit=_fuel_per_unit(),
            infeasible_penalty=infeasible_penalty,
            initial_est_used=lambda _b: initial_est_used(
                env=env, inst_idx=inst_idx, vehicles=vehicles, plan=plan,
                served_mask=served_mask, order_energy_fn=energy_fn,
                cooling_share=cooling_share, energy_pricing=energy_pricing,
                standby_orders=standby_orders),
            inner_gate=inner_gate, ls_kwargs=ls_kwargs, deadline=deadline,
            future_order=future_order)
        acc, diag = oc.decide_candidate(
            order=int(order), candidate_plan=cand_plan, budget_B=float(budget),
            myopic_net_fn=myopic_net_of, marginal_energy_fn=energy_fn,
            future_value_fn=fvf, certify_fn=certify_one,
            lambda_estimator=oc.LAMBDA_ESTIMATOR_SHADOW,
            myopic_base=oc.MYOPIC_BASE_COMPONENTS,
            ablate_opportunity_cost=bool(ablate_opportunity_cost),
            ablate_lookahead=bool(ablate_lookahead))
        diag['plan'] = cand_plan if acc else None
        diag['day_id'] = inst_idx
        diag['info_max_reveal'] = sdiag['info_max_reveal']
        diag['scenario'] = sdiag
        diag['inner_gate'] = inner_gate
        diag['arm_seed'] = int(arm_seed)
        diag['ablate_opportunity_cost'] = bool(ablate_opportunity_cost)
        diag['ablate_lookahead'] = bool(ablate_lookahead)
        diag['ablate_scenarios'] = ablate_scenarios
        # 前瞻评估的**自有诊断**（场景数/接受拒单计数/信息边界/规格哈希），
        # 供 T2b/T4 统计"场景机制是否真的被跑到"；不含任何效用字段。
        diag['forward'] = fvf.last_diag
        diag['elapsed_s'] = time.monotonic() - t0
        last['diag'] = diag
        return bool(acc), diag

    decide_fn.last_diag = last
    return decide_fn


# `KM_PER_UNIT × FUEL_COST_PER_KM` —— 懒取，避免在 import 期就依赖项目路径
_FUEL_PER_UNIT = None


def _fuel_per_unit():
    global _FUEL_PER_UNIT
    if _FUEL_PER_UNIT is None:
        from coldchain_evaluator_a1 import FUEL_COST_PER_KM, KM_PER_UNIT
        _FUEL_PER_UNIT = float(KM_PER_UNIT) * float(FUEL_COST_PER_KM)
    return _FUEL_PER_UNIT


def _insertion_fuel(bundle, states, o, energy_fn):
    """`fuel_o` 的**边际**口径（§2.1 要求写死为边际，不得用分摊/整日燃料）。

    = `fuel_cost_per_distance_unit × Δdistance(插入 o)`；若无处可插 ⇒ `Δ=0`（不虚增燃料，
    该单本来也接不了，`certify_plan` 会否决）。
    """
    ctx = bundle.space.route_ctx()

    def total(st):
        return sum(ls.route_dist(ctx, s['cur'], s['route']) for s in st.values())

    base = total(states)
    st2, ok = ls.best_insert(ctx, states, o)
    if not ok:
        return 0.0
    return max(0.0, total(st2) - base) * _fuel_per_unit()
