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


class SoftKNNHistoricalSampler:
    """S3-6a（2026-10-03 注册，规格《S3-6a软k-NN_正式规格预声明_2026-10-03.md》）：软 k-NN
    高斯核权重（不收缩）。w_i ∝ exp(−d_i/σ²)，d = 7 维 z 标准化平方距离（与
    CondHistoricalSampler 同特征口径）；对**全部**历史日软加权采样（无硬 top-k 截断，
    有放回），取 reveal>clock 未来段。σ² = sigma2_mult × h_med，h_med = 训练日两两
    （7 维 z 标准化、clock=0 参考）平方距离中位数（**训练日选定**，gate 日不调）。
    n_eff=(Σw)²/Σw² 逐决策记录。不改 Uncond/Cond/Explicit 采样器行为。"""

    name = "soft_knn"

    def __init__(self, history, sigma2_mult=1.0, n_neighbors=10):
        self.history = history
        self.sigma2_mult = float(sigma2_mult)
        self.n_neighbors = int(n_neighbors)
        self._calibrate_bandwidth()
        self.last_n_eff = None
        self.n_eff_sum = 0.0
        self.n_samples = 0

    @staticmethod
    def _feat(orders, clock):
        f = np.zeros(7)
        for o in orders:
            if o.reveal > clock + 1e-6:
                continue
            f[o.temp_class] += 1.0
            b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
            f[3 + b] += 1.0
        f[6] = float(clock)
        return f

    def _calibrate_bandwidth(self):
        """h_med：训练日两两（clock=0 参考）z 标准化平方距离的中位数（正距离口径，
        重复计数向量日距 0 不使带宽塌缩）；σ² = mult × h_med。"""
        H = np.array([self._feat(d, 0.0) for d in self.history])
        mu, sd = H.mean(0), H.std(0) + 1e-6
        Hn = (H - mu) / sd
        d = ((Hn[:, None, :] - Hn[None, :, :]) ** 2).sum(-1)
        n = d.shape[0]
        if n < 2:
            self.sigma2 = 1.0
            self.h_med = 1.0
            return
        off = np.triu_indices(n, 1)
        vals = d[off]
        pos = vals[vals > 1e-9]
        self.h_med = float(np.median(pos)) if len(pos) else 1.0
        self.sigma2 = self.sigma2_mult * max(self.h_med, 1e-12)

    def _weights(self, snapshot):
        x = self._feat(snapshot.orders, snapshot.clock)
        H = np.array([self._feat(d, snapshot.clock) for d in self.history])
        mu, sd = H.mean(0), H.std(0) + 1e-6
        Hn = (H - mu) / sd
        xn = (x - mu) / sd
        d = ((Hn - xn) ** 2).sum(1)
        # log 域稳定化：σ→0 时最大权重仍为 1，无下溢/除零
        log_w = -d / max(self.sigma2, 1e-300)
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        w = w / w.sum()
        return w

    def sample(self, snapshot, rng, k):
        w = self._weights(snapshot)
        n_eff = float((w.sum() ** 2) / (w ** 2).sum())
        self.last_n_eff = n_eff
        self.n_eff_sum += n_eff
        self.n_samples += 1
        idx = rng.choice(len(self.history), size=k, p=w, replace=True)
        return [_scenario_from_day(self.history[int(i)], snapshot.clock) for i in idx]


# --------------------------------------------------------------------------- #
# 共用下游决策器：SAA 接/拒（真实 C0 硬认证 + A-v1 效用），三臂只换 sampler
# --------------------------------------------------------------------------- #
class SaaReplanner(C0ReserveReplanner):
    """每次揭示：同批 K 个场景 → 对每个新订单做 接(插入+认证+SAA) vs 拒(SAA−拒绝损失) 比较。

    超时降级（统一）：保持当前已认证计划 + 拒绝该订单，计 timeouts。
    """

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 sampler, reject_penalty, K=10, time_limit=10.0, arm_seed=7001,
                 hist=None, k=10, shadow_mode='greedy', energy_pricing='amortized',
                 standby_orders=None, future_policy='reveal', overage_price=None,
                 incr_eval=False, anytime_vote=False, anytime_early_stop=True):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='myopic', hist=hist, k=k)
        self.sampler = sampler
        self.reject_penalty = dict(reject_penalty)
        self.K = K
        self.time_limit = float(time_limit)
        self.arm_seed = int(arm_seed)
        self.shadow_mode = shadow_mode  # 战役 L1：'greedy'|'ls'（影子未来插入路由强度）
        # 战役 L2b：影子能耗定价。'amortized'=每单摊待命分摊（旧）；'marginal'=只计边际
        # （door/COP+precool），固定待命按 standby_orders（训练日均单量）×cooling_share 预留。
        self.energy_pricing = energy_pricing
        self.standby_orders = standby_orders
        # 战役 L2c：影子未来策略接单顺序。'reveal'=按揭示时刻（旧）；'density'=按价值密度
        # （REV/边际能耗）降序（场景内后见之明打包，逼近预算约束下的最优接单集；在线决策仍因果）。
        self.future_policy = future_policy
        # 战役 S3-3：影子终局超额连续价格（元/kWh）。None=旧行为（终局超预算不扣罚）。
        # 完成阶段 WAIT/返仓待命能耗令真实终局能耗系统高于影子 est 累计；对终局超额量
        # （energy − B）+ 连续计价，把预算稀缺信号在场景内平滑化。不得恢复全场景 −1e6。
        self.overage_price = (None if overage_price is None else float(overage_price))
        # 战役 S3-4（2026-10-02）：增量评价预过滤开关。True=ls 候选先用 O(1) 路线摘要
        # （前向最早出发/后向最晚出发/累计载重）预筛，预筛通过者仍走原全量
        # _route_feasible_u 终检——预筛只跳过「可证明不可行」候选（保守 +1e-9 余量），
        # 故接受集合与原实现完全一致（等价加速，不改变数值与选择）。默认 False=旧行为。
        self.incr_eval = bool(incr_eval)
        # 战役 S3-4-D3（2026-10-03，注册件 S3-4-D3_anytime部分投票_正式规格预声明）：
        # anytime 部分投票——超时语义变更（默认 False=旧行为：保计划+拒单+计次）。
        # True=场景顺序随机化（固定种子、臂无关 → CRN）+ 逐场景累进累计统计量 +
        # 可选先验早停 + 时限到时按已完成场景子集提交（S_m>0 接单）。计数口径：
        # timeouts 保留原义（0 场景完成才计）；partial_commits=部分投票提交事件数；
        # early_stops=先验早停事件数。deadline=∞ 且早停关闭时与旧路径逐位一致。
        self.anytime_vote = bool(anytime_vote)
        self.anytime_early_stop = bool(anytime_early_stop)
        self._id_map = {}
        self._next_oid = 0
        self.timeouts = 0
        self.partial_commits = 0
        self.early_stops = 0

    # ---- 临时编号映射（每实例重置） ----
    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self._id_map = {}
            self._next_oid = 0
            self.timeouts = 0
            self.partial_commits = 0
            self.early_stops = 0

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
        """影子起点状态（2026-09-28 A-06 修订，按真实事件推进）：
        - committed 车：在途段不可撤销——从 committed_finish 继续（cur=committed_next、
          t=committed_finish、载重含在途单），route=[committed_next]+tail 且前 1 位冻结（frozen=1）；
        - returning / closed 车：不可再服务未来订单，退出影子车队；
        - 未 dispatch 车：从 depot 出发（空计划车也保留，供未来订单插入）。"""
        st = {}
        for v in vehicles:
            cc = v.coldchain_state
            if cc is not None and (cc.closed or v.status == 'returning'):
                continue   # 已返仓关闭 / 返仓中：不可再服务
            tail = [o for o in plan.get(v.vehicle_id, []) if not served_mask[o]]
            if cc is not None and v.status == 'committed' and v.committed_next not in (None, 0):
                route = [v.committed_next] + [o for o in tail if o != v.committed_next]
                cur, t = int(v.committed_next), float(v.committed_finish)
                load = float(v.current_load)
                frozen = 1   # 在途段冻结：插入/relocate/swap 不得越过
            elif cc is not None:
                cur, t, load = int(v.current_node), float(clock), float(v.current_load)
                route = tail
                frozen = 0
            else:
                # Q-02（2026-09-29 核查修复）：未出车车**即使空计划也保留**（可接未来订单）。
                # _complete_u 对 cc=None 且空路线不 dispatch、不产生派车/制冷费（无幻影预冷）；
                # 只有未来实际插入订单后才 dispatch。否则 accept/reject 两分支可能因
                # 「某车是否刚获得计划」使用不同的影子车队，扭曲资源价值。
                cur, t, load = 0, float(clock), 0.0
                route = tail
                frozen = 0
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=route,
                                    cc=v.coldchain_state, frozen=frozen)
        return st

    def _route_feasible_u(self, space, env, cur, cur_time, load, route, deadline):
        D = space.D
        tws, twe, st_ = space.tws, space.twe, space.st
        dem = space.demand
        inv = 1.0 / env.tw_speed
        cap = self.capacity
        t, c, l = cur_time, cur, load
        if len(set(route)) != len(route):   # 恰好一次护栏：影子搜索不得产生重复订单
            return False
        for o in route:
            d = D[c, o]
            # 2026-09-27 根因修复：与 _complete_u 同语义——未揭示订单在【出发地】等到 reveal 再行驶
            # （旧写法 max(t+d*inv, reveal, tws) 是在目的地等待，低估到达时间 → 预检放行、
            # 完成阶段 TW 全崩 → 每决策全场景 −1e6、投票退化）。
            depart = t if t >= space.reveal[o] else float(space.reveal[o])
            arr = depart + d * inv
            if arr < tws[o]:
                arr = tws[o]
            if arr > twe[o] + 1e-6:
                return False
            l += dem[o]
            if l > cap - 1e-4:   # 保守容量余量 1e-4：影子预检严格于物理 transition（_EPS=1e-9）
                return False
            t = arr + st_[o]
            c = o
        ret = t + D[c, 0] * inv
        return ret <= deadline + 1e-6

    def _greedy_insert_u(self, space, env, o, st, deadline):
        for vid, s in st.items():
            route = s['route']
            for pos in range(s.get('frozen', 0), len(route) + 1):   # A-06：冻结段不可插
                trial = route[:pos] + [o] + route[pos:]
                if self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                          s['load'], trial, deadline):
                    s['route'] = trial
                    return True
        return False

    def _route_dist_u(self, space, cur, route):
        """影子路线总距离（含返仓边）。"""
        D = space.D
        d = 0.0
        c = cur
        for o in route:
            d += D[c, o]
            c = o
        return d + D[c, 0]

    # ---- 战役 S3-4（2026-10-02）：O(1) 路线摘要与保守预筛（等价加速，默认关闭） ----
    # 预筛只拒绝「可证明不可行」的候选（-1e-9 保守余量）；预筛通过者仍走原
    # _route_feasible_u 全量终检——接受集合与旧实现完全一致，仅省掉大部分 O(m) 重放。

    def _rebuild_summary(self, space, env, s, deadline):
        """前向 T（各位置出发时刻）/后向 L（各位置最晚可行出发时刻）/累计载重。
        语义与 _route_feasible_u 一致（出发地等待 reveal、TW 裁剪、容量余量 1e-4）。
        后缀单调性：出发时刻整体推迟 δ 时，后缀到达最多推迟 δ（reveal 等待与 TW 裁剪
        只会吸收延迟），故后缀可行 ⟺ 新出发 ≤ L + 1e-9。"""
        route = s['route']
        D, inv = space.D, 1.0 / env.tw_speed
        tws, twe, st_ = space.tws, space.twe, space.st
        reveal, dem = space.reveal, space.demand
        m = len(route)
        T = [0.0] * m
        cload = [0.0] * m
        t, c = s['cur_time'], s['cur']
        l = s['load']
        for i, o in enumerate(route):
            depart = t if t >= reveal[o] else float(reveal[o])
            arr = depart + D[c, o] * inv
            if arr < tws[o]:
                arr = float(tws[o])
            l += dem[o]
            T[i] = arr + st_[o]
            cload[i] = l
            t, c = T[i], o
        L = [0.0] * m
        if m:
            L[m - 1] = deadline - D[route[m - 1], 0] * inv
        NEG = -1e300
        for i in range(m - 2, -1, -1):
            o, nxt = route[i], route[i + 1]
            d = D[o, nxt]
            cand = []
            # 区间 A：t_i >= reveal[nxt] 且无 TW 裁剪（arr = t_i + d*inv）
            vA = min(L[i + 1] - d * inv - st_[nxt], twe[nxt] + 1e-6 - d * inv)
            if vA >= reveal[nxt] - 1e-9:
                cand.append(vA)
            # 区间 B：t_i >= reveal[nxt] 且 arr 被 tws 裁剪（arr = tws[nxt]）
            if tws[nxt] + st_[nxt] <= L[i + 1] + 1e-9 and tws[nxt] <= twe[nxt] + 1e-6:
                clip_upper = tws[nxt] - d * inv - 1e-9
                if clip_upper >= reveal[nxt] - 1e-9:
                    cand.append(clip_upper)
            # 区间 C：t_i < reveal[nxt]（出发地等待揭示，arr 固定）
            fixed = max(float(reveal[nxt]) + d * inv, float(tws[nxt]))
            if fixed <= twe[nxt] + 1e-6 and fixed + st_[nxt] <= L[i + 1] + 1e-9:
                cand.append(float(reveal[nxt]) - 1e-9)
            L[i] = max(cand) if cand else NEG
        # v2 距离摘要：pre[i]=从 cur 到 route[i] 的前缀距离；total=含返仓边的全长
        pre = [0.0] * m
        dacc = 0.0
        c = s['cur']
        for i, o in enumerate(route):
            dacc += D[c, o]
            pre[i] = dacc
            c = o
        total = dacc + (D[c, 0] if m else D[s['cur'], 0])
        s['sum'] = dict(T=T, L=L, cload=cload, total_load=l, route_len=m,
                        pre=pre, total=total)

    def _precheck_ins(self, space, env, s, o, pos, deadline):
        """插入 o 于 route[pos] 前的 O(1) 可行预筛（保守：只拒绝可证明不可行者）。"""
        route = s['route']
        sm = s.get('sum')
        m = sm['route_len']
        D, inv = space.D, 1.0 / env.tw_speed
        tws, twe, st_ = space.tws, space.twe, space.st
        reveal, dem = space.reveal, space.demand
        cap = self.capacity
        if pos == 0:
            a, t_a, load_pre = s['cur'], s['cur_time'], s['load']
        else:
            a, t_a, load_pre = route[pos - 1], sm['T'][pos - 1], sm['cload'][pos - 1]
        depart = t_a if t_a >= reveal[o] else float(reveal[o])
        arr_o = depart + D[a, o] * inv
        if arr_o < tws[o]:
            arr_o = float(tws[o])
        if arr_o > twe[o] + 1e-6:
            return False
        load_o = load_pre + dem[o]
        if load_o > cap - 1e-4:
            return False
        t_o = arr_o + st_[o]
        if pos < m:
            b = route[pos]
            depart_b = t_o if t_o >= reveal[b] else float(reveal[b])
            arr_b = depart_b + D[o, b] * inv
            if arr_b < tws[b]:
                arr_b = float(tws[b])
            if arr_b > twe[b] + 1e-6:
                return False
            t_b_new = arr_b + st_[b]
            if t_b_new > sm['L'][pos] + 1e-9:   # 后缀最晚出发约束（+1e-9 保守余量）
                return False
            if load_o + (sm['cload'][m - 1] - sm['cload'][pos - 1]) > cap - 1e-4:
                return False
        else:
            if t_o + D[o, 0] * inv > deadline + 1e-6:
                return False
        return True

    def _precheck_replace(self, space, env, s, o_new, i, deadline):
        """把 route[i] 替换为 o_new 的 O(1) 可行预筛（swap 用；保守）。"""
        route = s['route']
        sm = s.get('sum')
        m = sm['route_len']
        D, inv = space.D, 1.0 / env.tw_speed
        tws, twe, st_ = space.tws, space.twe, space.st
        reveal, dem = space.reveal, space.demand
        cap = self.capacity
        if i == 0:
            a, t_a, load_pre = s['cur'], s['cur_time'], s['load']
        else:
            a, t_a, load_pre = route[i - 1], sm['T'][i - 1], sm['cload'][i - 1]
        depart = t_a if t_a >= reveal[o_new] else float(reveal[o_new])
        arr_o = depart + D[a, o_new] * inv
        if arr_o < tws[o_new]:
            arr_o = float(tws[o_new])
        if arr_o > twe[o_new] + 1e-6:
            return False
        load_o = load_pre + dem[o_new]
        if load_o > cap - 1e-4:
            return False
        if load_o + (sm['cload'][m - 1] - sm['cload'][i]) > cap - 1e-4:
            return False
        t_o = arr_o + st_[o_new]
        if i + 1 < m:
            b = route[i + 1]
            depart_b = t_o if t_o >= reveal[b] else float(reveal[b])
            arr_b = depart_b + D[o_new, b] * inv
            if arr_b < tws[b]:
                arr_b = float(tws[b])
            if arr_b > twe[b] + 1e-6:
                return False
            t_b_new = arr_b + st_[b]
            if t_b_new > sm['L'][i + 1] + 1e-9:
                return False
        else:
            if t_o + D[o_new, 0] * inv > deadline + 1e-6:
                return False
        return True

    def _delta_ins(self, space, s, o, pos):
        """O(1) 距离增量：o 插入 route[pos] 前（Δ = 新全长 − 旧全长，边差分语义）。"""
        route = s['route']
        D = space.D
        a = s['cur'] if pos == 0 else route[pos - 1]
        b = route[pos] if pos < len(route) else 0
        return float(D[a, o] + D[o, b] - D[a, b])

    def _delta_rem(self, space, s, i):
        """O(1) 距离增量：移除 route[i]（Δ = 新全长 − 旧全长，边差分语义）。"""
        route = s['route']
        D = space.D
        a = s['cur'] if i == 0 else route[i - 1]
        b = route[i + 1] if i + 1 < len(route) else 0
        return float(D[a, b] - D[a, route[i]] - D[route[i], b])

    def _delta_replace(self, space, s, o_new, i):
        """O(1) 距离增量：route[i] 换成 o_new。"""
        route = s['route']
        D = space.D
        a = s['cur'] if i == 0 else route[i - 1]
        b = route[i + 1] if i + 1 < len(route) else 0
        return float(D[a, o_new] + D[o_new, b] - D[a, route[i]] - D[route[i], b])

    def _insert_u(self, space, env, o, st, deadline):
        """未来订单插入：greedy=首个可行位；ls=全车全位最优（最小车队总距离增量，A-06 冻结位不可插）。
        S3-4 v2：incr_eval 下候选排序用 O(1) 预筛 + O(1) 边差分 Δ，选中后做旧式全量终检；
        终检失败（数值边界）→ 回退旧式全扫描（正确性兜底，接受集合在数值边界外与旧实现一致）。"""
        if self.shadow_mode == 'greedy':
            return self._greedy_insert_u(space, env, o, st, deadline)
        best = None
        for vid, s in st.items():
            route = s['route']
            fz = s.get('frozen', 0)
            if self.incr_eval:
                self._rebuild_summary(space, env, s, deadline)
            for pos in range(fz, len(route) + 1):   # A-06：冻结段（在途 committed leg）不可插
                if self.incr_eval:
                    if not self._precheck_ins(space, env, s, o, pos, deadline):
                        continue   # S3-4：可证明不可行 → 跳过
                    delta = self._delta_ins(space, s, o, pos)
                    key = delta
                    if best is None or key < best[0]:
                        best = (key, vid, pos)
                else:
                    trial = route[:pos] + [o] + route[pos:]
                    if self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                              s['load'], trial, deadline):
                        # 2026-09-28 修正（核查 A-L1）：比较插入增量而非整条路线成本
                        incr = (self._route_dist_u(space, s['cur'], trial)
                                - self._route_dist_u(space, s['cur'], route))
                        if best is None or incr < best[0]:
                            best = (incr, vid, pos)
        if best is None:
            return False
        if self.incr_eval:
            _, vid, pos = best
            s = st[vid]
            trial = s['route'][:pos] + [o] + s['route'][pos:]
            if not self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                          s['load'], trial, deadline):
                best = None   # v2 终检失败 → 旧式全扫描兜底
                for vid, s in st.items():
                    route = s['route']
                    for pos in range(s.get('frozen', 0), len(route) + 1):
                        trial = route[:pos] + [o] + route[pos:]
                        if self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                                  s['load'], trial, deadline):
                            incr = (self._route_dist_u(space, s['cur'], trial)
                                    - self._route_dist_u(space, s['cur'], route))
                            if best is None or incr < best[0]:
                                best = (incr, vid, pos)
                if best is None:
                    return False
        _, vid, pos = best
        r = st[vid]['route']
        st[vid]['route'] = r[:pos] + [o] + r[pos:]
        if self.incr_eval:
            self._rebuild_summary(space, env, st[vid], deadline)
        return True

    def _local_search_u(self, space, env, st, deadline, max_passes=2):
        """战役 L1：影子内 relocate（跨车）+ swap 有限局部搜索（纯距离改进，可行性硬检）。"""
        if self.shadow_mode != 'ls':
            return
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            vids = list(st.keys())
            if self.incr_eval:
                for s in st.values():
                    self._rebuild_summary(space, env, s, deadline)
            for src_vid in vids:
                s = st[src_vid]
                route = s['route']
                for i in range(s.get('frozen', 0), len(route)):   # A-06：冻结订单不可移动
                    o = route[i]
                    base = self._route_dist_u(space, s['cur'], route)
                    for dst_vid in vids:
                        d = st[dst_vid]
                        droute = d['route']
                        for pos in range(d.get('frozen', 0), len(droute) + 1):   # A-06：冻结位不可插
                            if src_vid == dst_vid:
                                # 同车 relocate：必须先从 route 去掉 o 再插入（否则订单重复）
                                rest = route[:i] + route[i + 1:]
                                trial_dst = rest[:pos] + [o] + rest[pos:]
                                if pos == i or pos == i + 1:
                                    continue
                            else:
                                trial_src = route[:i] + route[i + 1:]
                                trial_dst = droute[:pos] + [o] + droute[pos:]
                            if self.incr_eval and src_vid != dst_vid:
                                if not self._precheck_ins(space, env, d, o, pos, deadline):
                                    continue   # S3-4：跨车 relocate O(1) 预筛目标车
                                delta = (self._delta_ins(space, d, o, pos)
                                         + self._delta_rem(space, s, i))
                                if delta >= -1e-9:
                                    continue   # S3-4：边差分不改进 → 跳过
                            if not self._route_feasible_u(space, env, d['cur'], d['cur_time'],
                                                          d['load'], trial_dst, deadline):
                                continue
                            if src_vid == dst_vid:
                                if self._route_dist_u(space, d['cur'], trial_dst) < base - 1e-9:
                                    s['route'] = trial_dst
                                    route = trial_dst
                                    if self.incr_eval:
                                        self._rebuild_summary(space, env, s, deadline)
                                    improved = True
                                    break
                            else:
                                if not self._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                                              s['load'], trial_src, deadline):
                                    continue
                                new_d = (self._route_dist_u(space, s['cur'], trial_src) +
                                         self._route_dist_u(space, d['cur'], trial_dst))
                                old_d = (base + self._route_dist_u(space, d['cur'], droute))
                                if new_d < old_d - 1e-9:
                                    s['route'] = trial_src
                                    d['route'] = trial_dst
                                    route = trial_src
                                    if self.incr_eval:
                                        self._rebuild_summary(space, env, s, deadline)
                                        self._rebuild_summary(space, env, d, deadline)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # swap：两订单交换（跨车），只接受可行且总距离改进（A-06：冻结位不可交换）
                for a_vid in vids:
                    for i in range(st[a_vid].get('frozen', 0), len(st[a_vid]['route'])):
                        for b_vid in vids:
                            for j in range(st[b_vid].get('frozen', 0), len(st[b_vid]['route'])):
                                if a_vid == b_vid and i >= j:
                                    continue
                                sa, sb = st[a_vid], st[b_vid]
                                ra, rb = list(sa['route']), list(sb['route'])
                                oa, ob = ra[i], rb[j]
                                ra[i], rb[j] = ob, oa
                                if ra == sa['route'] and rb == sb['route']:
                                    continue
                                if self.incr_eval and a_vid != b_vid:
                                    if not self._precheck_replace(space, env, sa, ob, i, deadline):
                                        continue
                                    if not self._precheck_replace(space, env, sb, oa, j, deadline):
                                        continue
                                    delta = (self._delta_replace(space, sa, ob, i)
                                             + self._delta_replace(space, sb, oa, j))
                                    if delta >= -1e-9:
                                        continue   # S3-4：边差分不改进 → 跳过
                                if not self._route_feasible_u(space, env, sa['cur'], sa['cur_time'],
                                                              sa['load'], ra, deadline):
                                    continue
                                if not self._route_feasible_u(space, env, sb['cur'], sb['cur_time'],
                                                              sb['load'], rb, deadline):
                                    continue
                                old_d = (self._route_dist_u(space, sa['cur'], sa['route']) +
                                         self._route_dist_u(space, sb['cur'], sb['route']))
                                new_d = (self._route_dist_u(space, sa['cur'], ra) +
                                         self._route_dist_u(space, sb['cur'], rb))
                                if new_d < old_d - 1e-9:
                                    sa['route'], sb['route'] = ra, rb
                                    if self.incr_eval:
                                        self._rebuild_summary(space, env, sa, deadline)
                                        self._rebuild_summary(space, env, sb, deadline)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

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
        # Q-01（2026-09-29 核查修复）：复制状态必须保留 frozen 标记——
        # 旧实现丢失 frozen 后，_insert_u/_local_search_u 的 s.get('frozen',0) 恒为 0，
        # 未来预测订单可插入/越过不可撤销的在途 committed leg（探针：route[1,2] frozen=1
        # 完成入口变 [3,1,2]）。
        st = {vid: dict(cur=s['cur'], cur_time=s['cur_time'], load=s['load'],
                        route=list(s['route']), cc=s['cc'], frozen=s.get('frozen', 0))
              for vid, s in st.items()}
        deadline = float(env.tw_end[inst_idx, 0])
        # 战役 L2b：marginal 定价下每单只计 door/COP+precool；固定待命按全天预留一次
        share = self.cooling_share if self.energy_pricing == 'amortized' else 0.0
        est_used = sum(float(s['cc'].cumulative_energy_kwh) for s in st.values()
                       if s['cc'] is not None)
        for s in st.values():
            for o in s['route']:
                est_used += c0_marginal_energy(
                    int(env.dataset['temp_class'][inst_idx, o]),
                    float(env.demands[inst_idx, o]), share, self.contract)
        if self.energy_pricing == 'marginal' and self.standby_orders:
            est_used += self.cooling_share * float(self.standby_orders)
        rej_loss = 0.0
        # 战役 L2c：density 顺序 = 按价值密度降序打包（场景内后见之明）；reveal 顺序 = 旧行为
        if self.future_policy == 'density':
            def _density(i):
                e_i = c0_marginal_energy(int(space.tc[i]), float(space.demand[i]),
                                         share, self.contract)
                return REV[int(space.tc[i])] / max(e_i, 1e-9)
            scen_order = sorted(scen_idx, key=lambda i: -_density(i))
        else:
            scen_order = sorted(scen_idx, key=lambda i: space.reveal[i])
        for o in scen_order:
            tc = int(space.tc[o])
            e = c0_marginal_energy(tc, float(space.demand[o]), share, self.contract)
            if est_used + e > self.budget + 1e-9:
                rej_loss += self.reject_penalty[tc]
                continue
            if not self._insert_u(space, env, o, st, deadline):
                rej_loss += self.reject_penalty[tc]
                continue
            est_used += e
            if self.shadow_mode == 'ls':
                self._local_search_u(space, env, st, deadline)
        energy, dist_units = self._complete_u(space, env, inst_idx, clock, st, self.booking_horizon)
        # 2026-09-26 修复（助手，用户授权「你自己改」）：只对「真正不可行」（TW/容量/返仓，
        # energy is None）置 INFEASIBLE_SCENARIO_PENALTY。能耗超标不再置惩罚——
        # est 门槛（上面的循环）才是未来策略的预算规则，影子计划把未来一天的订单塞满后
        # 真实能耗几乎恒超 B，逐场景置 -1e6 会把场景相关的价值信息（plan_rev/fuel/rej_loss）
        # 全部淹没，使投票退化为「p_c>0 全接 / p_c=0 全拒」的空转（40 天实测 cond≡uncond≡myopic）。
        # 真实 C0 预算仍由实际接单步骤的 certify_plan 强制，不因本行放宽而失效。
        if energy is None:
            return INFEASIBLE_SCENARIO_PENALTY
        plan_rev = sum(REV[int(space.tc[o])] for s in st.values() for o in s['route'])
        fuel = dist_units * KM_PER_UNIT * FUEL_COST_PER_KM
        u = plan_rev - fuel - rej_loss
        # 战役 S3-3（2026-09-29）：终局超额连续价格——完成阶段真实能耗 energy 超过预算 B 时，
        # 按超额量 × overage_price 连续扣罚（替代「终局超预算不扣罚」的旧行为）。
        # 注意边界：不得恢复全场景 −1e6（那会使投票退化）；只在 est 硬门槛之外补上
        # 完成阶段被低估的待命能耗差额，价格尺度 = 元/kWh。
        if self.overage_price is not None and energy > self.budget + 1e-9:
            u -= self.overage_price * (energy - self.budget)
        return u

    # ---- 在线决策 ----
    def _anytime_perm(self, inst_idx, o, n):
        """D3：场景访问顺序的固定乱序。种子只依赖 (inst_idx, 订单映射 id)——臂无关，
        故三臂同事件同顺序（CRN）；不同订单不同顺序（避免固定序的系统偏向）。"""
        return list(np.random.default_rng(
            [0xA771, int(inst_idx), int(self._id_map.get(o, o))]).permutation(n))

    @staticmethod
    def _partial_vote_accept(acc_vals, rej_vals, done):
        """D3：部分投票提交规则 = 旧 SAA 统计量在已完成子集上的原序累加（acc_u > rej_u）。
        显式左到右 +=（不用 sum()），保证 done 全 True 时与旧路径浮点逐位一致。"""
        acc_u = 0.0
        rej_u = 0.0
        for a, r, d in zip(acc_vals, rej_vals, done):
            if d:
                acc_u += a
                rej_u += r
        return acc_u > rej_u

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        """A-02：单调时钟 + 绝对 deadline；采样/构造/插入/认证/投票全读剩余预算；提交前复查。
        D3（anytime_vote=True）：场景乱序访问；时限到时按已完成子集提交（S_m>0 接单，
        0 场景完成才退回旧语义：保计划+拒单+计 timeouts）；可选先验早停（m≥4 且
        |mean|>2×SE）。anytime_vote=False 路径与旧实现逐位不变。"""
        deadline = t0 + self.time_limit
        if time.perf_counter() > deadline:
            self._rejected.add(o)
            self.timeouts += 1
            return
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        try:
            insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles, served_mask, clock)
        except ValueError:
            insert_ok = False
            self._plan = {k: list(v) for k, v in saved_plan.items()}   # A-01：异常出口同样回滚
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
        if not self.anytime_vote:
            acc_u = 0.0
            rej_u = 0.0
            for scen_idx in scen_indices:
                if time.perf_counter() > deadline:
                    self._plan = saved_plan
                    self._rejected.add(o)
                    self.timeouts += 1
                    return
                acc_u += self._sim_scenario(env, inst_idx, clock, acc_states, scen_idx, space)
                rej_u += self._sim_scenario(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
            if time.perf_counter() > deadline:   # A-02：最后提交前复查（最后一轮超时不得接单）
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                return
            if acc_u > rej_u:
                self._plan = acc_plan
                self._accepted.add(o)
            else:
                self._plan = saved_plan
                self._rejected.add(o)
            return
        # ---- D3 anytime 分支 ----
        n = len(scen_indices)
        perm = self._anytime_perm(inst_idx, o, n)
        acc = [0.0] * n
        rej = [0.0] * n
        done = [False] * n
        m = 0
        for j in perm:
            if time.perf_counter() > deadline:
                break   # 部分投票提交，不回退丢弃
            scen_idx = scen_indices[j]
            acc[j] = self._sim_scenario(env, inst_idx, clock, acc_states, scen_idx, space)
            rej[j] = self._sim_scenario(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
            done[j] = True
            m += 1
            if self.anytime_early_stop and m >= 4:
                dvals = [acc[i] - rej[i] for i in range(n) if done[i]]
                mean = sum(dvals) / m
                se = (np.var(dvals) / m) ** 0.5
                if abs(mean) > 2.0 * se:
                    self.early_stops += 1
                    break
        if m == 0:
            self._plan = saved_plan
            self._rejected.add(o)
            self.timeouts += 1
            return
        if m < n:
            self.partial_commits += 1
        if self._partial_vote_accept(acc, rej, done):
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
        # A-02：决策时钟从快照构造起算（含采样）；单调 perf_counter。多单同揭示：
        # 采样计入首个决策，后续决策各自从入口起算（与外部驱动逐单计时一致）。
        event_t0 = time.perf_counter()
        self._sync_id_map(env, inst_idx, clock)
        snap = self._build_snapshot(env, inst_idx, clock, vehicles, served_mask)
        rng = np.random.default_rng([self.arm_seed, int(inst_idx), int(round(clock * 1000))])
        scenarios = self.sampler.sample(snap, rng, self.K)
        for idx, o in enumerate(sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i])):
            t0 = event_t0 if idx == 0 else time.perf_counter()
            self._saa_decide(env, inst_idx, clock, vehicles, served_mask, o,
                             scenarios, t0)
