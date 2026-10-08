"""真 3b · 步骤3：C0 对齐能耗评价器（预算/预测/终局统一到 C0 物理合同）。

与 run_exp_energy.py（UA·ΔT·dwell 代理）的区别：
  - **终局能耗**：不再用代理公式逐单累加，而是挂 `coldchain_contract` 跑 env 的真实 C0 物理
    （`transition_segment` 的逐车温区制冷 + 开门事件 + 预冷），取 trace 的
    `final_coldchain_state.cumulative_energy_kwh` 之和。
  - **预算 B**：= (1−ρ) × 训练集「全接（持久预留可服务的全部单）」的 C0 能耗均值。
  - **接受当刻预测**：C0 边际估计 = cooling_power_kw[c]·mean_dwell[c] + door_heat/COP
    + precool_energy，其中 mean_dwell 从训练集 C0 全接 trace 按温区估计。
  - 复用步骤2的持久预留（接受当刻锁单、plan() 只回放、时序 max(ready_time,clock)），
    使承诺违约为 0。

diurnal_amplitude_c 置 0，使 C0 制冷能耗退化为 cooling_power×时长（时间无关），
从而预算/预测/终局三者完全同口径（否则预测的纯 dwell 与终局的昼夜加权时长不一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SIM = os.path.join(_SCRIPTS, 'simulation')
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)
_COLD = os.path.join(_SCRIPTS, 'coldchain')
if _COLD not in sys.path:
    sys.path.insert(0, _COLD)

from strict_online_env import StrictOnlineEnv, Replanner
from coldchain_contract import (ColdChainContract, default_pilot_contract,
                                UnitScale, replace, ParameterProvenance, QualityConfigV2)
from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                             transition_segment, compute_cop)
from run_exp_reserve import (REV, PVAL, K_EFF, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT,
                             FUEL_COST_PER_KM, generate_dataset, _stat, _marginal_lambda)


def make_c0_contract():
    contract = default_pilot_contract()
    thermal = replace(contract.thermal, diurnal_amplitude_c=0.0)
    units = UnitScale(distance_km_per_unit=KM_PER_UNIT, hours_per_time_unit=1.0,
                      speed_kmph=SPEED_KMH)
    # 放宽 units.distance_km_per_unit 的 provenance 上界：本数据集坐标单位≈20km（与
    # 评价器其余 KM_PER_UNIT=20 一致），是数据集尺度选择，非物理参数漂移。
    provenance = list(contract.parameter_provenance)
    for i, p in enumerate(provenance):
        if p.parameter_path == 'units.distance_km_per_unit':
            provenance[i] = replace(p, sensitivity_high=50.0)
    contract = replace(contract, units=units, thermal=thermal,
                       parameter_provenance=tuple(provenance))
    contract.validate()
    return contract


def c0_energy_from_traces(traces):
    E = 0.0
    for t in traces:
        if t.final_coldchain_state is not None:
            E += float(t.final_coldchain_state.cumulative_energy_kwh)
    return E


def c0_marginal_energy(c, q, cooling_share, contract):
    """接受当刻 C0 边际估计：全温区制冷分摊 + 开门/COP + 预冷。

    制冷是车辆级持续成本（dispatch 后全温区 ~Σcooling_power kW/h 直到 return），分摊到每单
    （cooling_share，从训练集全接 trace 校准）；预冷是订单级（0.15·qty·ΔT_c，class 越高越大）。
    """
    th = contract.thermal
    door = th.door_heat_kwh[c] / compute_cop(th.target_temperature_c[c], contract)
    precool = q * th.precool_energy_per_unit_per_c * max(
        0.0, th.ambient_temperature_c - th.target_temperature_c[c])
    return cooling_share + door + precool


def c0_route_energy(route, dist, tws, st_, rev, dem, tc, contract):
    """离线路线 C0 能耗（transition_segment 真实物理，与 env 同口径）。"""
    state = create_vehicle_state(contract)
    state = dispatch_vehicle(state, contract)
    speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
    cur, t = 0, 0.0
    zones = (True,) * len(contract.thermal.supported_temp_classes)
    for o in route:
        d = float(dist[cur, o])
        depart = max(t, float(rev[o]))
        arrive = depart + d / speed
        sstart = max(arrive, float(tws[o]))
        sfinish = sstart + float(st_[o])
        state, _ = transition_segment(
            state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
            served_customer=int(o), active_zone_mask=zones, contract=contract,
            order_quantity=float(dem[o]), order_temp_class=int(tc[o]),
            initial_quality=1.0, segment_distance_units=d)
        cur = o
        t = sfinish
    d = float(dist[cur, 0])
    arrive = t + d / speed
    state, _ = transition_segment(
        state, depart_time=t, arrival_time=arrive, service_finish=arrive,
        served_customer=None, active_zone_mask=zones, contract=contract,
        return_to_depot=True, segment_distance_units=d)
    return float(state.cumulative_energy_kwh)


def c0_completion_energy(state, cur_node, cur_time, route, dist, tws, st_, rev, dem, tc,
                         contract, wait_until=None):
    """从当前 coldchain_state（已 dispatched）继续服务 route + WAIT + 返仓的 C0 能耗（含已消耗）。

    硬预算认证判据：完成当前 plan 的总能耗 ≤ B。
    从真实车辆状态接续（state 已含 committed leg 的 cargo/能耗），route 是未执行订单；
    wait_until 是返仓前的 WAIT 截止时刻（预约截止 booking_end）——服务完 route 后若早于它，
    就 WAIT 制冷到该时刻再返仓（P0-C WAIT 语义）。
    """
    speed = contract.units.speed_kmph / contract.units.distance_km_per_unit
    zones = (True,) * len(contract.thermal.supported_temp_classes)
    is_v2 = isinstance(contract.quality, QualityConfigV2)
    cur, t = cur_node, cur_time
    for o in route:
        d = float(dist[cur, o])
        depart = max(t, float(rev[o]))
        arrive = depart + d / speed
        sstart = max(arrive, float(tws[o]))
        sfinish = sstart + float(st_[o])
        iq = float(contract.quality.initial_value[int(tc[o])]) if is_v2 else 1.0
        state, _ = transition_segment(
            state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
            served_customer=int(o), active_zone_mask=zones, contract=contract,
            order_quantity=float(dem[o]), order_temp_class=int(tc[o]),
            initial_quality=iq, segment_distance_units=d)
        cur = o
        t = sfinish
    if wait_until is not None and wait_until > t + 1e-9:
        state, _ = transition_segment(
            state, depart_time=t, arrival_time=wait_until, service_finish=wait_until,
            served_customer=None, active_zone_mask=zones, contract=contract,
            segment_distance_units=0.0)
        t = wait_until
    d = float(dist[cur, 0])
    arrive = t + d / speed
    state, _ = transition_segment(
        state, depart_time=t, arrival_time=arrive, service_finish=arrive,
        served_customer=None, active_zone_mask=zones, contract=contract,
        return_to_depot=True, segment_distance_units=d)
    return float(state.cumulative_energy_kwh)


def certify_plan(env, inst_idx, clock, vehicles, served_mask, plan, contract, B):
    """硬预算认证：完成当前 plan 的真实 C0 能耗 ≤ B（含 committed leg + WAIT 到 booking_end + 返仓）。

    plan = {vid: 未执行订单 route}。返回 (feasible, total_energy_kwh)。
    从真实车辆 coldchain_state 接续（保留 committed leg 与已消耗能耗）。
    """
    dist = env.dist_mat[inst_idx]
    tws = env.tw_start[inst_idx]
    st_ = env.service_time[inst_idx]
    rev = env.reveal_time[inst_idx]
    dem = env.demands[inst_idx]
    tc = env.dataset['temp_class'][inst_idx]
    wait_until = env.booking_horizon if env.booking_horizon is not None else None
    total = 0.0
    for v in vehicles:
        cc = v.coldchain_state
        if cc is not None and cc.closed:
            total += float(cc.cumulative_energy_kwh)
            continue
        route = [o for o in plan.get(v.vehicle_id, []) if not served_mask[o]]
        if cc is not None:
            # dispatched 且未 closed：从当前状态继续
            if v.status == 'committed' and v.committed_next not in (None, 0):
                # committed_next 已在途（将从 current_node 驶向它），plan 里须排除它避免重复
                route = [v.committed_next] + [
                    o for o in plan.get(v.vehicle_id, [])
                    if not served_mask[o] and o != v.committed_next]
                cur_node = v.current_node
            elif v.status == 'ready':
                cur_node = v.current_node
            else:
                cur_node = v.current_node if v.current_node != 0 else 0
            cur_time = float(clock)
            if not route and cur_node == 0:
                total += float(cc.cumulative_energy_kwh)
                continue
            total += c0_completion_energy(cc, cur_node, cur_time, route, dist, tws, st_,
                                          rev, dem, tc, contract, wait_until)
        else:
            # 未 dispatch：从当前 clock 出发（reveal 后才可能 dispatch）
            if not route:
                continue
            state = dispatch_vehicle(create_vehicle_state(contract), contract)
            total += c0_completion_energy(state, 0, float(clock), route, dist, tws, st_,
                                          rev, dem, tc, contract, wait_until)
    return total <= B + 1e-9, total


class C0ReserveReplanner(Replanner):
    """持久预留 + C0 能耗预算的接受/拒绝 replanner（步骤2 + 步骤3）。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 mode='myopic', hist=None, k=10):
        self.budget = budget
        self.capacity = capacity
        self.booking_horizon = booking_horizon
        self.contract = contract
        self.cooling_share = cooling_share  # float：全温区制冷分摊（kWh/单）
        self.mode = mode
        self.hist = hist
        self.k = k
        self._accepted = set()
        self._rejected = set()
        self._plan = {}
        self._status_inst = -1

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = int(inst_idx)
            self._accepted = set()
            self._rejected = set()
            self._plan = {}

    def _est_energy(self, env, inst_idx, o):
        c = int(env.dataset['temp_class'][inst_idx, o])
        q = float(env.demands[inst_idx, o])
        return c0_marginal_energy(c, q, self.cooling_share, self.contract)

    def _vehicle_serve_state(self, env, inst_idx, v, clock):
        if v.status == 'committed':
            nxt = v.committed_next
            if nxt in (None, 0):
                return None
            return (int(nxt), float(v.committed_finish),
                    float(v.current_load) + float(env.demands[inst_idx, nxt]))
        if v.status == 'ready':
            return (int(v.current_node), max(float(v.ready_time), float(clock)), float(v.current_load))
        if v.status == 'idle':
            return (0, max(float(v.ready_time), float(clock)), 0.0)
        return None

    def _route_feasible(self, env, inst_idx, cur, cur_time, load, route):
        t, c, l = cur_time, cur, load
        for o in route:
            d = env.dist_mat[inst_idx, c, o]
            arr = max(t + d / env.tw_speed, env.tw_start[inst_idx, o])
            if arr > env.tw_end[inst_idx, o] + 1e-6:
                return False
            l += env.demands[inst_idx, o]
            if l > self.capacity - 1e-4:   # 2026-09-28：保守容量余量 1e-4——接单侧检查必须严格于
                return False               # 物理 transition（_EPS=1e-9），杜绝 float32 边界载重执行期崩溃
            t = arr + env.service_time[inst_idx, o]
            c = o
        ret = t + env.dist_mat[inst_idx, c, 0] / env.tw_speed
        return ret <= env.tw_end[inst_idx, 0] + 1e-6

    def _plan_start_states(self, env, inst_idx, vehicles, served_mask, clock):
        st = {}
        for v in vehicles:
            pos = self._vehicle_serve_state(env, inst_idx, v, clock)
            if pos is None:
                continue
            cur, t, load = pos
            # committed 车的 committed_next 已在 cur/load 计入（在途），route 须排除它，避免重复计载重
            committed = v.committed_next if v.status == 'committed' else None
            route = [o for o in self._plan.get(v.vehicle_id, [])
                     if not served_mask[o] and o != committed]
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=route)
        return st

    def _try_insert(self, env, inst_idx, o, vehicles, served_mask, clock):
        st = self._plan_start_states(env, inst_idx, vehicles, served_mask, clock)
        best_v, best_pos, best_incr = None, None, float('inf')
        for vid, s in st.items():
            route = s['route']
            cur = s['cur']
            for pos in range(len(route) + 1):
                pred = cur if pos == 0 else route[pos - 1]
                succ = 0 if pos == len(route) else route[pos]
                incr = (env.dist_mat[inst_idx, pred, o] + env.dist_mat[inst_idx, o, succ]
                        - env.dist_mat[inst_idx, pred, succ])
                if incr >= best_incr:
                    continue
                if self._route_feasible(env, inst_idx, cur, s['cur_time'], s['load'],
                                        route[:pos] + [o] + route[pos:]):
                    best_v, best_pos, best_incr = vid, pos, incr
        if best_v is None:
            return False
        st[best_v]['route'].insert(best_pos, o)
        self._plan = {vid: list(s['route']) for vid, s in st.items() if s['route']}
        return True

    def _try_insert_certified(self, env, inst_idx, o, vehicles, served_mask, clock):
        """贪心插入 o 并用真实 C0 能耗硬认证预算（certify_plan）。认证失败/异常回滚计划（A-01）。"""
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        if not self._try_insert(env, inst_idx, o, vehicles, served_mask, clock):
            return False
        try:
            ok, _ = certify_plan(env, inst_idx, clock, vehicles, served_mask, self._plan,
                                 self.contract, self.budget)
        except ValueError:
            ok = False   # 认证中途异常（物理硬违反）：同样按不可行处理并回滚
        if not ok:
            self._plan = saved_plan
            return False
        return True

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        t_cut = min(float(clock), 3.0)
        m = env.dataset['reveal_time'][inst_idx] <= t_cut
        feat = np.zeros(3)
        for c in range(3):
            feat[c] = np.sum(env.dataset['temp_class'][inst_idx][m] == c)
        hfeat = np.array([np.array([np.sum(d[1][d[0] <= t_cut] == c) for c in range(3)])
                          for d in self.hist])
        hn = (hfeat - hfeat.mean(0)) / (hfeat.std(0) + 1e-6)
        x = (feat - hfeat.mean(0)) / (hfeat.std(0) + 1e-6)
        d = ((hn - x) ** 2).sum(1)
        kn = np.argsort(d)[:self.k]
        rem = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o2) for o2 in self._accepted]))
        lams = []
        for h in kn:
            ht, hc = self.hist[h]
            f = ht > clock
            fv = np.array([REV[int(c)] for c in hc[f]])
            fe = np.array([c0_marginal_energy(int(c), 2.0, self.cooling_share, self.contract)
                           for c in hc[f]])
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams))

    def _uncond_lookahead_lambda(self, env, inst_idx, clock):
        """无条件动态前瞻门槛：用全部训练日未来订单算边际 value/e（不按当天早期计数选邻居）。

        返回 lam：接受当前订单当且仅当 REV/e > lam（更值得占用预算）。hist 为空返回 0（不预留）。
        """
        if not self.hist:
            return 0.0
        rem = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o2) for o2 in self._accepted]))
        rev_by_class = np.array([REV[c] for c in range(3)], dtype=float)
        fe_by_class = np.array(
            [c0_marginal_energy(c, 2.0, self.cooling_share, self.contract) for c in range(3)],
            dtype=float)
        lams = []
        for ht, hc in self.hist:
            f = ht > clock
            if f.sum() == 0:
                continue
            fv = rev_by_class[hc[f]]
            fe = fe_by_class[hc[f]]
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams)) if lams else 0.0

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        remaining = self.budget - float(np.sum(
            [self._est_energy(env, inst_idx, o) for o in self._accepted]))
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(env, inst_idx, o)
            if e > remaining + 1e-9:
                self._rejected.add(o)
                continue
            if self.mode == 'knn' and self.hist is not None:
                lam = self._lookahead_lambda(env, inst_idx, clock, o)
                if REV[int(env.dataset['temp_class'][inst_idx, o])] <= e * lam + 1e-9:
                    self._rejected.add(o)
                    continue
            if not self._try_insert(env, inst_idx, o, vehicles, served_mask, clock):
                self._rejected.add(o)
                continue
            self._accepted.add(o)
            remaining -= e

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self._reset_if_new(inst_idx)
        active = [v for v in vehicles if v.status in ('idle', 'ready')
                  and (replan_ids is None or v.vehicle_id in replan_ids)]
        has_future = clock < self.booking_horizon - 1e-6
        for v in active:
            route = [o for o in self._plan.get(v.vehicle_id, []) if not served_mask[o]]
            if route:
                # 不含 [0]：服务完 route 后触发 plan_exhaustion 再决定 WAIT（有未来 reveal）或返回
                v.mutable_suffix = route
            else:
                v.mutable_suffix = [] if (v.current_node != 0 and has_future) else [0]

    def export_state(self):
        return {'version': 'c0-reserve-v1', 'budget': self.budget, 'mode': self.mode,
                'accepted': sorted(self._accepted), 'rejected': sorted(self._rejected),
                'plan': {str(k): list(v) for k, v in self._plan.items()},
                'status_inst': self._status_inst}

    def restore_state(self, state):
        self.budget = state['budget']
        self.mode = state['mode']
        self._accepted = set(state['accepted'])
        self._rejected = set(state['rejected'])
        self._plan = {int(k): list(v) for k, v in state.get('plan', {}).items()}
        self._status_inst = state['status_inst']


def evaluate_trace_c0(inst_idx, env, traces, dataset, accepted_set, B, contract):
    tc = dataset['temp_class'][inst_idx]
    demand = dataset['demands'][inst_idx]
    pval = np.array([PVAL[i] for i in tc])
    revenue = np.array([REV[i] for i in tc])
    k_eff = np.array([K_EFF[i] for i in tc])
    tw_end = dataset['tw_end'][inst_idx]
    depot_deadline = float(tw_end[0])

    served_list = [s.node for t in traces for s in t.services]
    served_set = set(served_list)
    failures = []
    if len(served_list) != len(served_set):
        failures.append('duplicate_service')
    for t in traces:
        load = 0.0
        for s in t.services:
            load += float(demand[s.node])
            if load > env.capacity + 1e-6:
                failures.append('capacity_violation')
                break
        if 'capacity_violation' in failures:
            break
    energy = c0_energy_from_traces(traces)
    budget_violated = energy > B + 1e-9
    if accepted_set - served_set:
        failures.append('commitment_violation')
    if served_set - accepted_set:
        failures.append('served_not_accepted')
    for t in traces:
        for s in t.services:
            if s.depart_time < float(dataset['reveal_time'][inst_idx, s.node]) - 1e-6:
                failures.append('served_before_reveal')
                break
        if 'served_before_reveal' in failures:
            break
    for t in traces:
        if not t.services:
            continue
        if t.return_arrival is None:
            failures.append('missing_return'); break
        if t.return_arrival > depot_deadline + 1e-6:
            failures.append('depot_return_violation'); break
        for s in t.services:
            if s.arrival_time > float(tw_end[s.node]) + 1e-6:
                failures.append('customer_tw_violation'); break
        if any(f in failures for f in ('missing_return', 'depot_return_violation', 'customer_tw_violation')):
            break

    if failures:
        return float('nan'), energy, set(failures), budget_violated
    total_quality = 0.0
    total_distance_km = 0.0
    for t in traces:
        if not t.services:
            continue
        d_units = 0.0
        for s in t.services:
            d_units += float(env.dist_mat[inst_idx, s.prev_node, s.node])
        d_units += float(env.dist_mat[inst_idx, t.services[-1].node, 0])
        total_distance_km += d_units * KM_PER_UNIT
        ret = float(t.return_arrival)
        for s in t.services:
            dwell = max(0.0, ret - float(s.service_finish))
            o = s.node
            total_quality += pval[o] * demand[o] * (1.0 - np.exp(-k_eff[o] * dwell))
    profit = float(np.sum([revenue[o] for o in served_set])) - total_quality - total_distance_km * FUEL_COST_PER_KM
    return profit, energy, set(), budget_violated


def run_policy(test_ds, B, capacity, num_vehicles, contract, cooling_share, mode, hist, k):
    replanner = C0ReserveReplanner(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
                                   contract=contract, cooling_share=cooling_share, mode=mode, hist=hist, k=k)
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    profits, energies, fail_sets, budget_flags, n_served = [], [], [], [], []
    for i in range(test_ds['coords'].shape[0]):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        profit, energy, fails, budget_violated = evaluate_trace_c0(
            i, env, traces, test_ds, accepted_i, B, contract)
        profits.append(profit)
        energies.append(energy)
        fail_sets.append(fails)
        budget_flags.append(budget_violated)
        n_served.append(len([s.node for t in traces for s in t.services]))
    return (np.array(profits), np.array(energies), fail_sets,
            np.array(budget_flags, dtype=bool), np.array(n_served))


def clairvoyant_profit_c0(test_ds, i, B, capacity, num_vehicles, contract, cooling_share):
    tc = test_ds['temp_class'][i]
    dem = test_ds['demands'][i]
    tw = test_ds['tw_end'][i]
    tws = test_ds['tw_start'][i]
    st_ = test_ds['service_time'][i]
    rev = test_ds['reveal_time'][i]
    coords = test_ds['coords'][i]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    orders = [o for o in range(1, tc.shape[0]) if dem[o] > 0]
    ratio = {o: REV[int(tc[o])] / max(c0_marginal_energy(int(tc[o]), dem[o], cooling_share, contract), 1e-9)
             for o in orders}
    orders.sort(key=lambda o: -ratio[o])

    accepted = []
    rem = B
    routes = [[] for _ in range(num_vehicles)]
    for o in orders:
        e = c0_marginal_energy(int(tc[o]), dem[o], cooling_share, contract)
        if e > rem + 1e-9:
            continue
        best = None
        for vi, route in enumerate(routes):
            for pos in range(len(route) + 1):
                trial = route[:pos] + [o] + route[pos:]
                if _seq_feasible(trial, dem, tw, tws, st_, dist, capacity, rev):
                    best = (vi, pos)
                    break
            if best is not None:
                break
        if best is None:
            continue
        vi, pos = best
        routes[vi].insert(pos, o)
        accepted.append(o)
        rem -= e

    served = set(accepted)
    energy = sum(c0_route_energy(r, dist, tws, st_, rev, dem, tc, contract) for r in routes if r)
    total_quality = 0.0
    total_dist = 0.0
    for route in routes:
        if not route:
            continue
        cur = 0
        t = 0.0
        service_finish = {}
        for o in route:
            d = dist[cur, o]
            arr = max(t + d / (SPEED_KMH / KM_PER_UNIT), tws[o], rev[o])
            t = arr + st_[o]
            service_finish[o] = t
            cur = o
            total_dist += d * KM_PER_UNIT
        total_dist += dist[cur, 0] * KM_PER_UNIT
        ret = t + dist[cur, 0] / (SPEED_KMH / KM_PER_UNIT)
        for o in route:
            dwell = max(0.0, ret - service_finish[o])
            total_quality += PVAL[int(tc[o])] * dem[o] * (1.0 - np.exp(-K_EFF[int(tc[o])] * dwell))
    profit = sum(REV[int(tc[o])] for o in served) - total_quality - total_dist * FUEL_COST_PER_KM
    feasible = (energy <= B + 1e-9)
    return profit, feasible, energy


def _seq_feasible(route, dem, tw, tws, st_, dist, capacity, rev):
    cur, t, load = 0, 0.0, 0.0
    for o in route:
        d = dist[cur, o]
        arr = max(t + d / (SPEED_KMH / KM_PER_UNIT), tws[o], rev[o])
        if arr > tw[o] + 1e-6:
            return False
        load += dem[o]
        if load > capacity - 1e-4:   # 保守容量余量（同上）
            return False
        t = arr + st_[o]
        cur = o
    ret = t + dist[cur, 0] / (SPEED_KMH / KM_PER_UNIT)
    return ret <= tw[0] + 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--test-instances", type=int, default=40)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    contract = make_c0_contract()
    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)

    # 预算 B + cooling_share：训练集「全接」C0 能耗（持久预留可服务的全部单）。
    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    cooling_share0 = 2.0  # 全接时 budget=inf，est 不用于决策，占位即可
    train_energy = np.zeros(args.train_instances)
    total_orders = 0
    n_veh_used = 0
    for i in range(args.train_instances):
        rp = C0ReserveReplanner(budget=float('inf'), capacity=args.capacity,
                                booking_horizon=BOOKING_HORIZON, contract=contract,
                                cooling_share=cooling_share0, mode='myopic', hist=hist, k=args.k)
        env = StrictOnlineEnv(train_ds, capacity=args.capacity, num_vehicles=args.num_vehicles,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        traces, _ = env.run(i)
        train_energy[i] = c0_energy_from_traces(traces)
        for t in traces:
            if t.final_coldchain_state is None:
                continue
            n_veh_used += 1
            total_orders += len(t.services)
    B = (1.0 - args.rho) * float(train_energy.mean())
    sum_cooling = float(sum(contract.thermal.cooling_power_kw))
    # 保守：每车运行时长上界 = booking_horizon（WAIT 到预约截止），制冷分摊 = 全温区功率 × 上界 / 每车单数
    mean_orders_per_vehicle = total_orders / max(n_veh_used, 1)
    cooling_share = sum_cooling * BOOKING_HORIZON / max(mean_orders_per_vehicle, 1)

    p_my, e_my, f_my, b_my, ns_my = run_policy(test_ds, B, args.capacity, args.num_vehicles,
                                               contract, cooling_share, 'myopic', hist, args.k)
    p_kn, e_kn, f_kn, b_kn, ns_kn = run_policy(test_ds, B, args.capacity, args.num_vehicles,
                                               contract, cooling_share, 'knn', hist, args.k)

    p_cl, e_cl, b_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit_c0(test_ds, i, B, args.capacity, args.num_vehicles,
                                                   contract, cooling_share)
        p_cl.append(prof)
        e_cl.append(energy)
        b_cl.append(not feas)

    def fail_counts(f_sets):
        c = Counter()
        for f in f_sets:
            for r in f:
                c[r] += 1
        return dict(c)

    def summarize(p, f_sets, b_flags):
        hard_ok = np.array([len(f) == 0 for f in f_sets])
        return {
            "feasible_rate": float(np.mean(hard_ok & ~b_flags)),
            "hard_feasible_rate": float(np.mean(hard_ok)),
            "budget_violation_count": int(np.sum(b_flags)),
            "commitment_violation_count": int(fail_counts(f_sets).get('commitment_violation', 0)),
            "fail_counts": fail_counts(f_sets),
            "profit_budget_feasible": _stat(p[hard_ok & ~b_flags], args.seed),
            "profit_hard_feasible": _stat(p[hard_ok], args.seed),
        }

    my_hard_ok = np.array([len(f) == 0 for f in f_my])
    kn_hard_ok = np.array([len(f) == 0 for f in f_kn])
    cl_budget_ok = ~np.array(b_cl, dtype=bool)
    both_hard_ok = my_hard_ok & kn_hard_ok

    per_instance = []
    for i in range(args.test_instances):
        per_instance.append({
            "i": int(i),
            "myopic_profit": (float(p_my[i]) if np.isfinite(p_my[i]) else None),
            "myopic_hard_feasible": bool(my_hard_ok[i]),
            "myopic_budget_violated": bool(b_my[i]),
            "myopic_fails": sorted(f_my[i]),
            "knn_profit": (float(p_kn[i]) if np.isfinite(p_kn[i]) else None),
            "knn_hard_feasible": bool(kn_hard_ok[i]),
            "knn_budget_violated": bool(b_kn[i]),
            "knn_fails": sorted(f_kn[i]),
            "clairvoyant_profit": (float(p_cl[i]) if np.isfinite(p_cl[i]) else None),
            "clairvoyant_budget_violated": bool(b_cl[i]),
            "both_hard_feasible": bool(both_hard_ok[i]),
            "knn_minus_myopic": (float(p_kn[i] - p_my[i])
                                if (np.isfinite(p_kn[i]) and np.isfinite(p_my[i])) else None),
        })

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "train_accept_all_c0_energy_mean": float(train_energy.mean()),
                   "cooling_share_kwh_per_order": float(cooling_share)},
        "myopic": {"n_served_mean": float(np.mean(ns_my)), **summarize(p_my, f_my, b_my)},
        "knn": {"n_served_mean": float(np.mean(ns_kn)), **summarize(p_kn, f_kn, b_kn)},
        "clairvoyant": {
            "feasible_rate": float(np.mean(cl_budget_ok)),
            "budget_violation_count": int(np.sum(b_cl)),
            "profit_budget_feasible": _stat(np.array(p_cl)[cl_budget_ok], args.seed),
            "profit_hard_feasible": _stat(np.array(p_cl), args.seed),
        },
        "per_instance": per_instance,
        "note": "步骤3：能耗统一到 C0 物理合同（transition_segment 逐车温区制冷+开门+预冷，diurnal=0）；"
                "预算 B=(1−ρ)×训练全接 C0 能耗均值；接受预测=全温区制冷分摊(cooling_share)+开门/COP+预冷；"
                "终局= trace cumulative_energy_kwh；复用步骤2持久预留（违约清零）。",
    }
    if both_hard_ok.sum() > 0:
        report["knn_minus_myopic_paired_hard_feasible"] = _stat(p_kn[both_hard_ok] - p_my[both_hard_ok], args.seed)
        report["paired_n_hard_feasible"] = int(both_hard_ok.sum())
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
