"""A-v1 步骤 2：三采样器共用的因果 SAA 决策闭环（无训练条件信息门）。

按 A-v1步骤12锁定协议 §2.5：
  sampler.sample(snapshot: VisibleSnapshot, rng, k) -> list[FutureScenario]
  - snapshot 由在线环境用已揭示信息构造（无 env/inst_idx/未来订单/原始数组索引）；
  - FutureScenario = 数量可变、可为空的未来订单集合（临时编号，揭示时间晚于当前时钟）；
  - 三臂共用同一下游决策器（接/拒 SAA 同批场景 + 贪心插入 + 真实 C0 硬认证 + A-v1 效用）；
  - 场景评估不修改真实环境，最终动作才提交。

三臂只替换 sampler：uncond_hist（无条件历史）/ cond_hist（条件历史 k-NN）/ explicit_feat（显式特征预测）。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from coldchain_evaluator_a1 import REV, FUEL_COST_PER_KM, KM_PER_UNIT
from coldchain_contract import QualityConfigV2
from coldchain_state import create_vehicle_state, dispatch_vehicle, transition_segment
from run_exp_energy_c0 import C0ReserveReplanner, c0_marginal_energy

SERVICE_H = 0.05
SPOT_CENTERS = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]], dtype=float)

# 影子场景不可行（能耗超预算 / 返仓超截止）的效用占位（2026-09-26 审计 T4）：
# 有限大负值，避免 −inf 传染使「期望效用 SAA」退化为「任一失败即否决」。
# 语义：一个失败场景不再把候选总分变 −inf；失败按严重不可行计罚（多失败按次数累积）。
INFEASIBLE_SCENARIO_PENALTY = -1e6


def make_c0_contract_v2():
    """A-v1 生产口径 v2 合同：units=(20, 1.0, 30)，使 contract 速度 == env tw_speed(1.5)。

    注意：旧 `ColdChainContractV2()` 裸默认 units=(1.0,1.0,1.0) 与 env 不一致（certify 按 1.0 走），
    本门统一用此工厂。
    """
    from coldchain_contract import ColdChainContractV2, UnitScale
    from run_exp_reserve import SPEED_KMH

    contract = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=KM_PER_UNIT, hours_per_time_unit=1.0,
                      speed_kmph=SPEED_KMH)
    provenance = list(contract.parameter_provenance)
    for i, p in enumerate(provenance):
        if p.parameter_path == 'units.distance_km_per_unit':
            provenance[i] = replace(p, sensitivity_high=50.0)
    contract = replace(contract, units=units, parameter_provenance=tuple(provenance))
    contract.validate()
    return contract


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioOrder:
    """订单（真实已揭示 / 场景未来预测共用）。oid 为临时编号：已揭示>0，场景未来<0。"""
    oid: int
    reveal: float
    x: float
    y: float
    demand: float
    temp_class: int
    tw_start: float
    tw_end: float
    service_time: float


FutureScenario = list  # list[ScenarioOrder]


@dataclass(frozen=True)
class VisibleVehicle:
    vid: int
    status: str
    node: int            # 临时编号（depot=0）；committed 车=committed_next
    ready_time: float
    load: float
    committed_next: int  # 临时编号或 None
    plan_tail: tuple     # 未执行计划尾部（临时编号）


@dataclass(frozen=True)
class VisibleSnapshot:
    clock: float
    orders: tuple                # 已揭示订单（ScenarioOrder，oid=临时编号）
    accepted: frozenset          # 已接受临时编号
    rejected: frozenset          # 已拒绝临时编号
    vehicles: tuple              # VisibleVehicle
    energy_used: float           # 已耗 C0 能耗
    budget: float                # 剩余预算 = budget
    booking_horizon: float
    capacity: float


def build_history(dataset):
    """训练集 → 每天全部订单（ScenarioOrder，oid 保留原始 idx，仅供 samplers 内部使用）。"""
    hist = []
    N = dataset['coords'].shape[0]
    dem = dataset['demands']
    for i in range(N):
        day = []
        for o in range(1, dem.shape[1]):
            if dem[i, o] > 0:
                day.append(ScenarioOrder(
                    oid=int(o), reveal=float(dataset['reveal_time'][i, o]),
                    x=float(dataset['coords'][i, o, 0]), y=float(dataset['coords'][i, o, 1]),
                    demand=float(dem[i, o]), temp_class=int(dataset['temp_class'][i, o]),
                    tw_start=float(dataset['tw_start'][i, o]),
                    tw_end=float(dataset['tw_end'][i, o]),
                    service_time=float(dataset['service_time'][i, o])))
        hist.append(day)
    return hist


def build_history_times_classes(dataset):
    """compute_budget_and_dwell 需要的 [(times, classes), ...] 格式。"""
    out = []
    dem = dataset['demands']
    for i in range(dataset['coords'].shape[0]):
        mask = dem[i, 1:] > 0
        out.append((dataset['reveal_time'][i, 1:][mask],
                    dataset['temp_class'][i, 1:][mask].astype(int)))
    return out


def _scenario_from_day(day, clock, start_oid=-1_000_000):
    return [replace(o, oid=start_oid - j) for j, o in enumerate(day)
            if o.reveal > clock + 1e-6]


# --------------------------------------------------------------------------- #
# 三个采样器（只读 VisibleSnapshot + 训练历史）
# --------------------------------------------------------------------------- #
class UncondHistoricalSampler:
    """臂 1：无条件历史场景。从历史日无差别重采样，取其 reveal>clock 的未来段。"""
    name = "uncond_hist"

    def __init__(self, history):
        self.history = history

    def sample(self, snapshot, rng, k):
        n = len(self.history)
        out = []
        for _ in range(k):
            day = self.history[int(rng.integers(0, n))]
            out.append(_scenario_from_day(day, snapshot.clock))
        return out


class CondHistoricalSampler:
    """臂 2：条件历史场景。按可见状态（早期订单类/空间桶计数 + 时钟）选相似历史日，取其未来段。"""
    name = "cond_hist"

    def __init__(self, history, n_neighbors=10):
        self.history = history
        self.n_neighbors = n_neighbors

    def _feat(self, orders, clock):
        f = np.zeros(7)
        for o in orders:
            if o.reveal > clock + 1e-6:
                continue
            f[o.temp_class] += 1.0
            b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
            f[3 + b] += 1.0
        f[6] = float(clock)
        return f

    def sample(self, snapshot, rng, k):
        x = self._feat(snapshot.orders, snapshot.clock)
        h = np.array([self._feat(d, snapshot.clock) for d in self.history])
        mu, sd = h.mean(0), h.std(0) + 1e-6
        hn = (h - mu) / sd
        xn = (x - mu) / sd
        d = ((hn - xn) ** 2).sum(1)
        kn = np.argsort(d)[:max(1, min(self.n_neighbors, len(d)))]
        out = []
        for _ in range(k):
            day = self.history[int(kn[int(rng.integers(0, len(kn)))])]
            out.append(_scenario_from_day(day, snapshot.clock))
        return out


class ExplicitFeatureSampler:
    """臂 3：简单显式特征预测器。按时钟桶回归未来「类×空间桶」单元计数（历史经验均值 + Poisson），
    时间/坐标/需求从历史经验分布采样。"""
    name = "explicit_feat"

    def __init__(self, history, clock_bin=0.5):
        self.history = history
        self.clock_bin = clock_bin
        self._fit()

    def _fit(self):
        bins = np.arange(0.0, 16.0 + self.clock_bin, self.clock_bin)
        self.bins = bins
        cell_mean = defaultdict(list)         # (c, b, clock_bin_idx) -> list[future count]
        self.time_pool = defaultdict(list)    # 'all' -> future reveal times
        self.bucket_coords = defaultdict(list)  # b -> coords
        self.bucket_demand = defaultdict(list)  # (c, b) -> demands
        for day in self.history:
            for o in day:
                b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
                self.bucket_coords[b].append((o.x, o.y))
                self.bucket_demand[(o.temp_class, b)].append(o.demand)
                self.time_pool['all'].append(o.reveal)
            for bi, cut in enumerate(bins):
                for c in range(3):
                    for b in range(3):
                        nf = sum(1 for o in day if o.reveal > cut + 1e-6
                                 and o.temp_class == c and self._bucket_of(o) == b)
                        cell_mean[(c, b, bi)].append(nf)
        self.cell_mean = {k: float(np.mean(v)) for k, v in cell_mean.items()}
        self.time_pool = {k: np.array(v) for k, v in self.time_pool.items()}
        self.bucket_coords = {k: np.array(v) for k, v in self.bucket_coords.items()}
        self.bucket_demand = {k: np.array(v) for k, v in self.bucket_demand.items()}

    def _bucket_index(self, clock):
        return int(min(float(clock), 16.0 - 1e-9) / self.clock_bin)

    def _bucket_of(self, o):
        return int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))

    def sample(self, snapshot, rng, k):
        bi = self._bucket_index(snapshot.clock)
        pool_all = self.time_pool['all']
        future_pool = pool_all[pool_all > snapshot.clock + 1e-6]
        out = []
        for _ in range(k):
            orders = []
            for c in range(3):
                for b in range(3):
                    lam = self.cell_mean.get((c, b, bi), 0.0)
                    if lam <= 0 or len(future_pool) == 0:
                        continue
                    n = int(rng.poisson(lam))
                    for _ in range(n):
                        t = float(future_pool[int(rng.integers(0, len(future_pool)))])
                        x, y = self.bucket_coords[b][int(rng.integers(0, len(self.bucket_coords[b])))]
                        dem_pool = self.bucket_demand.get((c, b))
                        demand = (float(dem_pool[int(rng.integers(0, len(dem_pool)))])
                                  if dem_pool is not None and len(dem_pool) else 2.0)
                        orders.append(ScenarioOrder(
                            oid=-(1_000_000 + len(orders) + 1), reveal=t, x=float(x), y=float(y),
                            demand=demand, temp_class=c, tw_start=0.0,
                            tw_end=t + float(rng.uniform(2.0, 4.0)),
                            service_time=SERVICE_H))
            out.append(orders)
        return out


# --------------------------------------------------------------------------- #
# 共用下游决策器：SAA 接/拒（真实 C0 硬认证 + A-v1 效用），三臂只换 sampler
# --------------------------------------------------------------------------- #
class SaaReplanner(C0ReserveReplanner):
    """每次揭示：同批 K 个场景 → 对每个新订单做 接(插入+认证+SAA) vs 拒(SAA−拒绝损失) 比较。

    超时降级（统一）：保持当前已认证计划 + 拒绝该订单，计 timeouts。
    """

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 sampler, reject_penalty, K=10, time_limit=10.0, arm_seed=7001,
                 hist=None, k=10):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='myopic', hist=hist, k=k)
        self.sampler = sampler
        self.reject_penalty = dict(reject_penalty)
        self.K = K
        self.time_limit = float(time_limit)
        self.arm_seed = int(arm_seed)
        self._id_map = {}
        self._next_oid = 0
        self.timeouts = 0

    # ---- 临时编号映射（每实例重置） ----
    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self._id_map = {}
            self._next_oid = 0
            self.timeouts = 0

    def _sync_id_map(self, env, inst_idx, clock):
        revealed = [i for i in range(1, env.num_nodes)
                    if env.demands[inst_idx, i] > 0
                    and env.reveal_time[inst_idx, i] <= clock + 1e-6]
        for i in revealed:
            if i not in self._id_map:
                self._next_oid += 1
                self._id_map[i] = self._next_oid

    def _map_node(self, n):
        if n is None or n == 0:
            return n
        return self._id_map.get(int(n), int(n))

    def _class_of(self, env, inst_idx, o):
        return int(env.dataset['temp_class'][inst_idx, o])

    def _real_order(self, env, inst_idx, o, oid=None):
        return ScenarioOrder(
            oid=int(o) if oid is None else int(oid),
            reveal=float(env.reveal_time[inst_idx, o]),
            x=float(env.coords[inst_idx, o, 0]), y=float(env.coords[inst_idx, o, 1]),
            demand=float(env.demands[inst_idx, o]),
            temp_class=int(env.dataset['temp_class'][inst_idx, o]),
            tw_start=float(env.tw_start[inst_idx, o]), tw_end=float(env.tw_end[inst_idx, o]),
            service_time=float(env.service_time[inst_idx, o]))

    def _build_snapshot(self, env, inst_idx, clock, vehicles, served_mask):
        orders = tuple(self._real_order(env, inst_idx, i, self._id_map[i])
                       for i in sorted(self._id_map, key=lambda x: self._id_map[x]))
        vehs = tuple(VisibleVehicle(
            vid=v.vehicle_id, status=v.status, node=self._map_node(v.current_node),
            ready_time=float(v.ready_time), load=float(v.current_load),
            committed_next=self._map_node(v.committed_next),
            plan_tail=tuple(self._map_node(x) for x in self._plan.get(v.vehicle_id, [])
                            if not served_mask[x])) for v in vehicles)
        energy_used = sum(float(v.coldchain_state.cumulative_energy_kwh)
                          for v in vehicles if v.coldchain_state is not None)
        return VisibleSnapshot(
            clock=float(clock), orders=orders,
            accepted=frozenset(self._map_node(x) for x in self._accepted),
            rejected=frozenset(self._map_node(x) for x in self._rejected),
            vehicles=vehs, energy_used=float(energy_used),
            budget=float(self.budget), booking_horizon=float(self.booking_horizon),
            capacity=float(self.capacity))

    # ---- 统一索引空间上的影子模拟（不改真实环境） ----
    class _SimSpace:
        """统一索引：0=depot，1..n_real-1=真实节点（原 idx），n_real..=场景订单。"""
        __slots__ = ("D", "reveal", "tws", "twe", "st", "demand", "tc", "iq", "n_real")

    def _build_space(self, env, inst_idx, scenarios):
        contract = self.contract
        scen_orders = [o for s in scenarios for o in s]
        n_real = env.coords.shape[1]
        coords = np.asarray(env.coords[inst_idx], dtype=np.float64)
        if scen_orders:
            scoords = np.array([[o.x, o.y] for o in scen_orders], dtype=np.float64)
            coords_all = np.vstack([coords, scoords])
        else:
            coords_all = coords
        diff = coords_all[:, None, :] - coords_all[None, :, :]
        is_v2 = isinstance(contract.quality, QualityConfigV2)
        iq_real = np.array([float(contract.quality.initial_value[int(c)]) if is_v2 else 1.0
                            for c in env.dataset['temp_class'][inst_idx]], dtype=np.float64)
        sp = self._SimSpace()
        sp.D = np.sqrt((diff ** 2).sum(-1))
        sp.n_real = n_real
        sp.reveal = np.concatenate([np.asarray(env.reveal_time[inst_idx], np.float64),
                                    np.array([o.reveal for o in scen_orders], np.float64)])
        sp.tws = np.concatenate([np.asarray(env.tw_start[inst_idx], np.float64),
                                 np.array([o.tw_start for o in scen_orders], np.float64)])
        sp.twe = np.concatenate([np.asarray(env.tw_end[inst_idx], np.float64),
                                 np.array([o.tw_end for o in scen_orders], np.float64)])
        sp.st = np.concatenate([np.asarray(env.service_time[inst_idx], np.float64),
                                np.array([o.service_time for o in scen_orders], np.float64)])
        sp.demand = np.concatenate([np.asarray(env.demands[inst_idx], np.float64),
                                    np.array([o.demand for o in scen_orders], np.float64)])
        sp.tc = np.concatenate([np.asarray(env.dataset['temp_class'][inst_idx], np.int64),
                                np.array([o.temp_class for o in scen_orders], np.int64)])
        sp.iq = np.concatenate([iq_real,
                                np.array([float(contract.quality.initial_value[o.temp_class])
                                          if is_v2 else 1.0 for o in scen_orders], np.float64)])
        return sp, scen_orders

    def _sim_states(self, env, inst_idx, clock, vehicles, served_mask, plan):
        """影子起点状态 —— 与真实 `certify_plan` 同口径（2026-09-26 审计 T3 对账）：
        committed 车从 current_node / clock 继续（含 committed_next 段），未 dispatch 车从 depot 出发。"""
        st = {}
        for v in vehicles:
            cc = v.coldchain_state
            tail = [o for o in plan.get(v.vehicle_id, []) if not served_mask[o]]
            if cc is not None and v.status == 'committed' and v.committed_next not in (None, 0):
                route = [v.committed_next] + [o for o in tail if o != v.committed_next]
                cur, t, load = int(v.current_node), float(clock), float(v.current_load)
            elif cc is not None:
                # dispatched 未 committed（ready/returning）：从当前位置继续（空 route 会 WAIT+返仓，与认证一致）
                cur, t, load = int(v.current_node), float(clock), float(v.current_load)
                route = tail
            else:
                # 未 dispatch：只有有计划的车才参与（从 depot 出发）
                if not tail:
                    continue
                cur, t, load = 0, float(clock), 0.0
                route = tail
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=route,
                                    cc=v.coldchain_state)
        return st

    def _route_feasible_u(self, space, env, cur, cur_time, load, route, deadline):
        D = space.D
        tws, twe, st_ = space.tws, space.twe, space.st
        dem = space.demand
        inv = 1.0 / env.tw_speed
        cap = self.capacity
        t, c, l = cur_time, cur, load
        for o in route:
            d = D[c, o]
            arr = t + d * inv
            if arr < space.reveal[o]:   # 2026-09-26 审计 T2：预检按未来订单揭示时刻推进
                arr = space.reveal[o]
            if arr < tws[o]:
                arr = tws[o]
            if arr > twe[o] + 1e-6:
                return False
            l += dem[o]
            if l > cap + 1e-9:   # 与 coldchain_state._EPS=1e-9 对齐：避免边界浮点噪声触发合同硬检查
                return False
            t = arr + st_[o]
            c = o
        ret = t + D[c, 0] * inv
        return ret <= deadline + 1e-6

    def _greedy_insert_u(self, space, env, o, st, deadline):
        for vid, s in st.items():
            route = s['route']
            for pos in range(len(route) + 1):
                trial = route[:pos] + [o] + route[pos:]
                if self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                          s['load'], trial, deadline):
                    s['route'] = trial
                    return True
        return False

    def _complete_u(self, space, env, inst_idx, clock, st, wait_until):
        """完成影子计划：真实 C0 能耗 + 距离；WAIT 到 booking_end + 返仓；返仓过 depot 截止=不可行。"""
        contract = self.contract
        speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
        zones = (True,) * len(contract.thermal.supported_temp_classes)
        deadline = float(env.tw_end[inst_idx, 0])
        D = space.D
        total_energy = 0.0
        total_dist = 0.0
        for s in st.values():
            cc = s['cc']
            if cc is not None and cc.closed:
                total_energy += float(cc.cumulative_energy_kwh)
                continue
            if not s['route'] and s['cur'] == 0:
                if cc is not None:
                    total_energy += float(cc.cumulative_energy_kwh)
                continue
            state = cc if cc is not None else dispatch_vehicle(create_vehicle_state(contract), contract)
            cur, t = s['cur'], s['cur_time']
            d_acc = 0.0
            for o in s['route']:
                d = float(D[cur, o])
                depart = t if t >= space.reveal[o] else float(space.reveal[o])
                arrive = depart + d / speed
                sstart = arrive if arrive >= space.tws[o] else float(space.tws[o])
                if sstart > float(space.twe[o]) + 1e-6:   # 2026-09-26 审计 T2：完成阶段复查每单截止
                    return None, None
                sfinish = sstart + float(space.st[o])
                try:
                    state, _ = transition_segment(
                        state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
                        served_customer=int(o), active_zone_mask=zones, contract=contract,
                        order_quantity=float(space.demand[o]), order_temp_class=int(space.tc[o]),
                        initial_quality=float(space.iq[o]), segment_distance_units=d)
                except ValueError:
                    return None, None  # 影子边界硬违反 → 该场景不可行
                d_acc += d
                t = sfinish
                cur = o
            if wait_until is not None and wait_until > t + 1e-9:
                state, _ = transition_segment(
                    state, depart_time=t, arrival_time=wait_until, service_finish=wait_until,
                    served_customer=None, active_zone_mask=zones, contract=contract,
                    segment_distance_units=0.0)
                t = wait_until
            d = float(D[cur, 0])
            arrive = t + d / speed
            if arrive > deadline + 1e-6:
                return None, None
            state, _ = transition_segment(
                state, depart_time=t, arrival_time=arrive, service_finish=arrive,
                served_customer=None, active_zone_mask=zones, contract=contract,
                return_to_depot=True, segment_distance_units=d)
            total_energy += float(state.cumulative_energy_kwh)
            total_dist += d_acc + d
        return total_energy, total_dist

    def _sim_scenario(self, env, inst_idx, clock, st, scen_idx, space):
        """影子模拟：myopic 未来策略（est 预算门槛 + 贪心插入）→ 终局 C0 硬预算 + A-v1 效用。

        2026-09-26 审计修正：①未来接单收入只经 `plan_rev` 计一次（去掉 served_rev 双计）；
        ②不可行场景返回有限 `INFEASIBLE_SCENARIO_PENALTY`（无 −inf 传染）；
        ③未来接单门槛以真实已耗能耗（cc.cumulative）+ 剩余计划逐单 est 为基准。
        """
        st = {vid: dict(cur=s['cur'], cur_time=s['cur_time'], load=s['load'],
                        route=list(s['route']), cc=s['cc']) for vid, s in st.items()}
        deadline = float(env.tw_end[inst_idx, 0])
        est_used = sum(float(s['cc'].cumulative_energy_kwh) for s in st.values()
                       if s['cc'] is not None)
        est_used += sum(self._est_energy(env, inst_idx, o)
                        for s in st.values() for o in s['route'])
        rej_loss = 0.0
        for o in sorted(scen_idx, key=lambda i: space.reveal[i]):
            tc = int(space.tc[o])
            e = c0_marginal_energy(tc, float(space.demand[o]), self.cooling_share, self.contract)
            if est_used + e > self.budget + 1e-9:
                rej_loss += self.reject_penalty[tc]
                continue
            if not self._greedy_insert_u(space, env, o, st, deadline):
                rej_loss += self.reject_penalty[tc]
                continue
            est_used += e
        energy, dist_units = self._complete_u(space, env, inst_idx, clock, st, self.booking_horizon)
        if energy is None or energy > self.budget + 1e-9:
            return INFEASIBLE_SCENARIO_PENALTY
        plan_rev = sum(REV[int(space.tc[o])] for s in st.values() for o in s['route'])
        fuel = dist_units * KM_PER_UNIT * FUEL_COST_PER_KM
        return plan_rev - fuel - rej_loss

    # ---- 在线决策 ----
    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        if time.time() - t0 > self.time_limit:   # 2026-09-26 审计 T5：采样时间计入每决策时限
            self._rejected.add(o)
            self.timeouts += 1
            return
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        try:
            insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles, served_mask, clock)
        except ValueError:
            insert_ok = False  # 认证路径边界硬违反（如恰好满容浮点噪声）→ 视为不可行拒绝
        if not insert_ok:
            self._rejected.add(o)
            return
        acc_plan = {k: list(v) for k, v in self._plan.items()}
        space, scen_orders = self._build_space(env, inst_idx, scenarios)
        idx_of = {id(x): space.n_real + i for i, x in enumerate(scen_orders)}
        scen_indices = [[idx_of[id(x)] for x in scen] for scen in scenarios]
        acc_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, acc_plan)
        self._plan = saved_plan
        rej_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, saved_plan)
        pc_o = self.reject_penalty[self._class_of(env, inst_idx, o)]
        acc_u = 0.0
        rej_u = 0.0
        for scen_idx in scen_indices:
            if time.time() - t0 > self.time_limit:
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                return
            acc_u += self._sim_scenario(env, inst_idx, clock, acc_states, scen_idx, space)
            rej_u += self._sim_scenario(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
        if acc_u > rej_u:
            self._plan = acc_plan
            self._accepted.add(o)
        else:
            self._plan = saved_plan
            self._rejected.add(o)

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        if not new_set:
            return
        self._sync_id_map(env, inst_idx, clock)
        snap = self._build_snapshot(env, inst_idx, clock, vehicles, served_mask)
        rng = np.random.default_rng([self.arm_seed, int(inst_idx), int(round(clock * 1000))])
        # 2026-09-26 审计 T5：决策时钟从快照构造起算，场景生成（sampler.sample）计入每决策 10s
        t0 = time.time()
        scenarios = self.sampler.sample(snap, rng, self.K)
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            self._saa_decide(env, inst_idx, clock, vehicles, served_mask, o,
                             scenarios, t0)
