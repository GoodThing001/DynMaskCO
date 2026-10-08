"""真 3b · 实验 2：把逐单能耗 E_class 换成物理路由能耗。

在 run_causal_accept_gate.py（v4）基础上，唯一改动是能耗模型：
  - 逐单 E_class（固定 D_MEAN 距离代理）→ 物理路由耦合能耗
      E_refrig = Σ_车 [ UA·ΔT_class·(在途+服务时长) + E_door·n_开门 ] / COP
    （UA=0.03 kW/K, ΔT=[7,21,43] K, E_door=0.056 kWh, COP=1.5）
  - 终局评价：per-order dwell（service_finish → 返仓时长），同一公式。
  - 接受时 route_time 未知 → 用"边际插入能耗"估计 = 插入该单使某车 route_time 增加
    （空车 depot→o→depot 返仓段时长 dist(o,0)/speed）+ 一次开门 E_door。
  - 预算 B = (1−ρ) × 历史"全接"路由能耗均值（NN 最优插入路由所有可行单）。

四项闭环（因果 booking_horizon / 承诺持久 / 独立认证 / 失败计分）与 NN 最优插入路由保留。
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

from strict_online_env import StrictOnlineEnv, Replanner

REV = {0: 10.0, 1: 20.0, 2: 30.0}
DT = {0: 7.0, 1: 21.0, 2: 43.0}
K_EFF = {0: 4.53e-3, 1: 1.3e-3, 2: 2.23e-5}
PVAL = {0: 1.0, 1: 1.3, 2: 2.0}
UA, E_DOOR, COP = 0.03, 0.056, 1.5
SPEED_KMH, KM_PER_UNIT, SERVICE_H = 30.0, 20.0, 0.05
FUEL_COST_PER_KM = 0.1  # 燃油成本（value/km），使距离成本 ~10% 收益
BOOKING_HORIZON = 16.0
DEPOT = np.array([0.5, 0.5])


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def order_energy(dwell_h, temp_class):
    """物理路由能耗（同一公式，接受时与终局共用）：(UA·ΔT_class·dwell + E_door)/COP。"""
    return (UA * DT[int(temp_class)] * dwell_h + E_DOOR) / COP


def marginal_order_energy(dist_units, temp_class):
    """naive 边际插入能耗（返仓段 dist(o,0)/speed）：仅作参考，会 ~10× 低估终局 dwell。"""
    return order_energy(dist_units / (SPEED_KMH / KM_PER_UNIT), temp_class)


def _route_time(dist_units):
    return dist_units / (SPEED_KMH / KM_PER_UNIT)


def routes_dwells(routes, dist, tws, st_, tc, rev=None):
    """终局物理路由能耗：Σ_车 Σ_o [UA·ΔT_class(o)·dwell(o) + E_door]/COP，
    dwell(o) = 返仓时刻 − o 的 service_finish（在途+服务时长，含后续停站）。
    返回 (energy, list_of_(class, dwell))。
    rev（reveal_time）：物理可执行口径，arrival 下界加 rev[o]（本数据集 tw_start≡0）。
    """
    dwells = []
    for route in routes:
        if not route:
            continue
        cur, t = 0, 0.0
        sf = {}
        for o in route:
            d = dist[cur, o]
            arr = max(t + _route_time(d), tws[o], (rev[o] if rev is not None else tws[o]))
            t = arr + st_[o]
            sf[o] = t
            cur = o
        ret = t + _route_time(dist[cur, 0])
        for o in route:
            dwells.append((int(tc[o]), max(0.0, ret - sf[o])))
    energy = sum(UA * DT[c] * d + E_DOOR for c, d in dwells) / COP
    return energy, dwells


def routes_energy(routes, dist, tws, st_, tc, rev=None):
    return routes_dwells(routes, dist, tws, st_, tc, rev)[0]


def route_energy_from_traces(inst_idx, traces, dataset):
    """从执行 trace 算终局物理路由能耗（与 routes_energy 同一公式）。"""
    tc = dataset['temp_class'][inst_idx]
    E = 0.0
    for t in traces:
        if not t.services or t.return_arrival is None:
            continue
        ret = float(t.return_arrival)
        for s in t.services:
            dwell = max(0.0, ret - float(s.service_finish))
            E += UA * DT[int(tc[s.node])] * dwell + E_DOOR
    return E / COP


def _seq_feasible(route, dem, tw, tws, st_, dist, capacity, rev=None):
    cur, t, load = 0, 0.0, 0.0
    for o in route:
        d = dist[cur, o]
        arr = max(t + _route_time(d), tws[o], (rev[o] if rev is not None else tws[o]))
        if arr > tw[o] + 1e-6:
            return False
        load += dem[o]
        if load > capacity + 1e-6:
            return False
        t = arr + st_[o]
        cur = o
    ret = t + _route_time(dist[cur, 0])
    return ret <= tw[0] + 1e-6


def accept_all_route(ds, i, capacity, num_vehicles):
    """历史"全接"路由：接受所有可行单（NN 最优插入）。返回 (routes, dist, tws, st_, tc, rev)。
    物理可执行口径：_seq_feasible 的 arrival 下界加 reveal_time。"""
    tc = ds['temp_class'][i]
    dem = ds['demands'][i]
    tw = ds['tw_end'][i]
    tws = ds['tw_start'][i]
    st_ = ds['service_time'][i]
    rev = ds['reveal_time'][i]
    coords = ds['coords'][i]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    orders = [o for o in range(1, tc.shape[0]) if dem[o] > 0]
    orders.sort(key=lambda o: tws[o])  # EDD 确定性
    routes = [[] for _ in range(num_vehicles)]
    for o in orders:
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
    return routes, dist, tws, st_, tc, rev


def accept_all_route_energy(ds, i, capacity, num_vehicles):
    """历史"全接"路由能耗（物理路由能耗，arrival 下界含 reveal_time）。"""
    routes, dist, tws, st_, tc, rev = accept_all_route(ds, i, capacity, num_vehicles)
    return routes_energy(routes, dist, tws, st_, tc, rev)


def generate_dataset(n_instances, n_orders, seed):
    rng = np.random.default_rng(seed)
    insts = []
    for _ in range(n_instances):
        z = float(rng.normal())
        beta = np.array([2.0, -0.5, -0.5])
        base = np.array([0.2, 0.4, 0.4])
        weights = _softmax(np.log(base) + beta * z)
        peak = float(np.clip(11.0 + 4.0 * z, 9.0, 19.0))
        total_per_h = 200.0 * np.exp(0.3 * z) / 16.0
        lam_max = total_per_h * 2.2
        n = int(rng.poisson(lam_max * 16.0))
        times = rng.uniform(0.0, 16.0, n)
        prof = 1.0 + 1.2 * np.exp(-((times - peak) ** 2) / (2.0 * 1.5 ** 2))
        keep = rng.uniform(size=n) < (prof * total_per_h / lam_max)
        times = times[keep]
        if len(times) > n_orders:
            times = times[rng.choice(len(times), n_orders, replace=False)]
        n_act = times.size
        centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
        spot = rng.choice(3, size=n_act, p=weights)
        coords = centers[spot] + rng.normal(0.0, 0.12, size=(n_act, 2))
        class_probs = np.array([[0.1, 0.3, 0.6], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
        cls = np.array([rng.choice(3, p=class_probs[s]) for s in spot])
        demand = rng.uniform(1.0, 3.0, n_act)

        N = n_orders
        coords_all = np.zeros((N + 1, 2)); coords_all[0] = [0.5, 0.5]
        coords_all[1:1 + n_act] = coords
        coords_all[1 + n_act:] = rng.uniform(0.0, 1.0, (N - n_act, 2))
        demands_all = np.zeros(N + 1); demands_all[1:1 + n_act] = demand
        tw_start_all = np.zeros(N + 1)
        tw_end_all = np.full(N + 1, 22.0)
        tw_end_all[1:1 + n_act] = times + rng.uniform(2.0, 4.0, n_act)
        service_all = np.full(N + 1, SERVICE_H); service_all[0] = 0.0
        reveal_all = np.full(N + 1, 1e6)
        reveal_all[1:1 + n_act] = times
        temp_class_all = np.zeros(N + 1, dtype=np.int32); temp_class_all[1:1 + n_act] = cls
        insts.append(dict(coords=coords_all.astype(np.float32),
                          demands=demands_all.astype(np.float32),
                          tw_start=tw_start_all.astype(np.float32),
                          tw_end=tw_end_all.astype(np.float32),
                          service_time=service_all.astype(np.float32),
                          reveal_time=reveal_all.astype(np.float32),
                          temp_class=temp_class_all))
    keys = ['coords', 'demands', 'tw_start', 'tw_end', 'service_time', 'reveal_time', 'temp_class']
    return {k: np.stack([i[k] for i in insts], axis=0) for k in keys}


def _marginal_lambda(vals, energys, R):
    if len(vals) == 0 or R <= 1e-12:
        return 0.0
    order = np.argsort(-(np.asarray(vals) / np.maximum(np.asarray(energys), 1e-12)))
    rem = R
    for i in order:
        if energys[i] <= rem + 1e-12:
            rem -= energys[i]
        else:
            return float(vals[i] / energys[i])
    return 0.0


class AcceptRejectReplanner(Replanner):
    def __init__(self, budget, capacity, booking_horizon, mode='myopic', hist=None, k=10,
                 mean_dwell=None):
        self.budget = budget
        self.capacity = capacity
        self.booking_horizon = booking_horizon
        self.mode = mode
        self.hist = hist  # list of (times, classes, coords) per train day
        self.k = k
        self.mean_dwell = mean_dwell or {0: 0.0, 1: 0.0, 2: 0.0}  # 接受时 dwell 估计（h）
        self._accepted = set()
        self._rejected = set()
        self._status_inst = -1

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = int(inst_idx)
            self._accepted = set()
            self._rejected = set()

    def _est_energy(self, c):
        """接受时的能耗估计：同一公式 (UA·ΔT·dwell + E_door)/COP，dwell 用历史均值。"""
        return order_energy(self.mean_dwell[int(c)], int(c))

    def _accepted_energy(self, env, inst_idx):
        return float(np.sum([self._est_energy(env.dataset['temp_class'][inst_idx, o])
                             for o in self._accepted]))

    def _can_serve(self, env, inst_idx, o, vehicles):
        deadline = float(env.tw_end[inst_idx, o])
        for v in vehicles:
            if v.status == 'closed':
                continue
            t = float(v.ready_time) if v.status in ('idle', 'ready') else (
                float(v.committed_finish) if v.committed_finish is not None else float(v.ready_time))
            arr = t + env.dist_mat[inst_idx, v.current_node, o] / env.tw_speed
            arr = max(arr, env.tw_start[inst_idx, o])
            if arr <= deadline + 1e-6 and float(v.current_load) + env.demands[inst_idx, o] <= self.capacity + 1e-6:
                return True
        return False

    def _lookahead_lambda(self, env, inst_idx, clock, o):
        """k-NN 未来边际 λ：用 morning 特征找 k 个历史天，取 clock 后未来订单的 value/energy 分数边际。
        energy 用同一物理公式的接受时估计（mean_dwell，非 naive 返仓段）。"""
        # 因果口径（2026-09-24 修正）：当前时刻 clock 只能看到 reveal_time <= clock 的订单；
        # 历史天用相同时间截面 min(clock, 3h) 匹配，避免把未揭示订单算进特征。
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
        rem = self.budget - self._accepted_energy(env, inst_idx)
        lams = []
        for h in kn:
            ht, hc, _hcoord = self.hist[h]
            f = ht > clock
            fv = np.array([REV[int(c)] for c in hc[f]])
            fe = np.array([self._est_energy(c) for c in hc[f]])
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams))

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        tc = env.dataset['temp_class']
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        remaining = self.budget - self._accepted_energy(env, inst_idx)
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = self._est_energy(tc[inst_idx, o])
            if e > remaining + 1e-9 or not self._can_serve(env, inst_idx, o, vehicles):
                self._rejected.add(o)
                continue
            if self.mode == 'knn' and self.hist is not None:
                lam = self._lookahead_lambda(env, inst_idx, clock, o)
                if REV[int(tc[inst_idx, o])] <= e * lam + 1e-9:
                    self._rejected.add(o)
                    continue
            self._accepted.add(o)
            remaining -= e

    def _route_feasible(self, env, inst_idx, cur, cur_time, load, route):
        t, c, l = cur_time, cur, load
        for o in route:
            d = env.dist_mat[inst_idx, c, o]
            arr = max(t + d / env.tw_speed, env.tw_start[inst_idx, o])
            if arr > env.tw_end[inst_idx, o] + 1e-6:
                return False
            l += env.demands[inst_idx, o]
            if l > self.capacity + 1e-6:
                return False
            t = arr + env.service_time[inst_idx, o]
            c = o
        ret = t + env.dist_mat[inst_idx, c, 0] / env.tw_speed
        return ret <= env.tw_end[inst_idx, 0] + 1e-6

    def _recompute_end(self, env, inst_idx, s):
        t, c, l = s['cur_time'], s['cur'], s['load']
        for o in s['route']:
            d = env.dist_mat[inst_idx, c, o]
            t = max(t + d / env.tw_speed, env.tw_start[inst_idx, o]) + env.service_time[inst_idx, o]
            l += env.demands[inst_idx, o]
            c = o
        s['cur_time'], s['cur'], s['load'] = t, c, l

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        to_assign = [int(i) for i in visible_ids if not served_mask[i]
                     and int(i) not in reserved and int(i) in self._accepted]
        to_assign.sort(key=lambda i: env.tw_end[inst_idx, i])  # EDD

        active = [v for v in vehicles if v.status in ('idle', 'ready')
                  and (replan_ids is None or v.vehicle_id in replan_ids)]
        st = {v.vehicle_id: dict(cur=v.current_node, cur_time=v.ready_time,
                                 load=float(v.current_load), route=[])
              for v in active}

        for o in to_assign:
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
                continue
            st[best_v]['route'].insert(best_pos, o)
            self._recompute_end(env, inst_idx, st[best_v])

        has_future = clock < self.booking_horizon - 1e-6
        for v in active:
            s = st[v.vehicle_id]
            if s['route']:
                v.mutable_suffix = s['route'] + [0]
            else:
                v.mutable_suffix = [] if (v.current_node != 0 and has_future) else [0]

    def export_state(self):
        return {'version': 'exp-energy-v1', 'budget': self.budget, 'mode': self.mode,
                'accepted': sorted(self._accepted), 'rejected': sorted(self._rejected),
                'status_inst': self._status_inst}

    def restore_state(self, state):
        self.budget = state['budget']
        self.mode = state['mode']
        self._accepted = set(state['accepted'])
        self._rejected = set(state['rejected'])
        self._status_inst = state['status_inst']


def evaluate_trace(inst_idx, env, traces, dataset, accepted_set, B):
    """返回 (profit, energy, hard_failures, budget_violated)。

    hard_failures = 运营硬约束违约（承诺/容量/TW/重复服务等），不含能耗预算。
    budget_violated = 终局物理路由能耗 > B（绿色预算）。
    profit 只要无 hard failure 就返回（即使 budget 违约），用于回答"头腔是否仍存在"。
    """
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

    energy = route_energy_from_traces(inst_idx, traces, dataset)
    budget_violated = energy > B + 1e-9

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


def _stat(x, seed):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float('nan'), "ci_lo": float('nan'), "ci_hi": float('nan'), "n": 0}
    n = len(x)
    r = np.random.default_rng(seed + 777)
    mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
    return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(mm, 2.5)),
            "ci_hi": float(np.percentile(mm, 97.5)), "n": int(n)}


def clairvoyant_profit(test_ds, i, B, capacity, num_vehicles, mean_dwell):
    """离线 clairvoyant 参考：按 value/energy（物理公式的 mean_dwell 估计）贪心接受 + NN 插入路由。
    终局能耗 = 物理路由能耗（routes_energy）。返回 (profit, feasible, energy)。
    物理可执行口径（2026-09-24 修正）：arrival 下界加 reveal_time。"""
    tc = test_ds['temp_class'][i]
    dem = test_ds['demands'][i]
    tw = test_ds['tw_end'][i]
    tws = test_ds['tw_start'][i]
    st_ = test_ds['service_time'][i]
    rev = test_ds['reveal_time'][i]
    coords = test_ds['coords'][i]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    orders = [o for o in range(1, tc.shape[0]) if dem[o] > 0]
    ratio = {o: REV[int(tc[o])] / max(order_energy(mean_dwell[int(tc[o])], int(tc[o])), 1e-12)
             for o in orders}
    orders.sort(key=lambda o: -ratio[o])

    accepted = []
    rem = B
    routes = [[] for _ in range(num_vehicles)]
    for o in orders:
        e = order_energy(mean_dwell[int(tc[o])], int(tc[o]))
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
    energy = routes_energy(routes, dist, tws, st_, tc, rev)
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
            arr = max(t + _route_time(d), tws[o], rev[o])
            t = arr + st_[o]
            service_finish[o] = t
            cur = o
            total_dist += d * KM_PER_UNIT
        total_dist += dist[cur, 0] * KM_PER_UNIT
        ret = t + _route_time(dist[cur, 0])
        for o in route:
            dwell = max(0.0, ret - service_finish[o])
            total_quality += PVAL[int(tc[o])] * dem[o] * (1.0 - np.exp(-K_EFF[int(tc[o])] * dwell))
    profit = sum(REV[int(tc[o])] for o in served) - total_quality - total_dist * FUEL_COST_PER_KM
    feasible = (energy <= B + 1e-9)
    return profit, feasible, energy


def run_policy(test_ds, train_ds, B, capacity, num_vehicles, mode, hist, k, mean_dwell):
    replanner = AcceptRejectReplanner(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
                                      mode=mode, hist=hist, k=k, mean_dwell=mean_dwell)
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=None, booking_horizon=BOOKING_HORIZON)
    profits, energies, fail_sets, budget_flags, n_served = [], [], [], [], []
    for i in range(test_ds['coords'].shape[0]):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        profit, energy, fails, budget_violated = evaluate_trace(i, env, traces, test_ds, accepted_i, B)
        profits.append(profit)
        energies.append(energy)
        fail_sets.append(fails)
        budget_flags.append(budget_violated)
        n_served.append(len([s.node for t in traces for s in t.services]))
    return (np.array(profits), np.array(energies), fail_sets,
            np.array(budget_flags, dtype=bool), np.array(n_served))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--test-instances", type=int, default=60)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)

    # 预算 B = (1−ρ) × 历史"全接"路由能耗均值（物理路由能耗）；同时算接受时 mean_dwell。
    train_route_energy = np.zeros(args.train_instances)
    dwell_by_class = {c: [] for c in range(3)}
    for i in range(args.train_instances):
        routes, dist, tws, st_, tc, rev = accept_all_route(train_ds, i, args.capacity, args.num_vehicles)
        energy, dwells = routes_dwells(routes, dist, tws, st_, tc, rev)
        train_route_energy[i] = energy
        for c, d in dwells:
            dwell_by_class[c].append(d)
    B = (1.0 - args.rho) * float(train_route_energy.mean())
    mean_dwell = {c: (float(np.mean(dwell_by_class[c])) if dwell_by_class[c] else 0.0)
                  for c in range(3)}

    # k-NN 历史（train 天：(reveal_time, class, coords)，只含真实订单，排除 depot + dummy）
    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        coords = train_ds['coords'][i][1:][mask]
        hist.append((times, classes, coords))

    p_my, e_my, f_my, b_my, ns_my = run_policy(test_ds, train_ds, B, args.capacity, args.num_vehicles, 'myopic', hist, args.k, mean_dwell)
    p_kn, e_kn, f_kn, b_kn, ns_kn = run_policy(test_ds, train_ds, B, args.capacity, args.num_vehicles, 'knn', hist, args.k, mean_dwell)

    p_cl, e_cl, b_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit(test_ds, i, B, args.capacity, args.num_vehicles, mean_dwell)
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
        "budget": {"B": float(B),
                   "train_accept_all_route_energy_mean": float(train_route_energy.mean()),
                   "mean_dwell_h": {str(c): float(mean_dwell[c]) for c in range(3)}},
        "myopic": {"n_served_mean": float(np.mean(ns_my)), **summarize(p_my, f_my, b_my)},
        "knn": {"n_served_mean": float(np.mean(ns_kn)), **summarize(p_kn, f_kn, b_kn)},
        "clairvoyant": {
            "feasible_rate": float(np.mean(cl_budget_ok)),
            "budget_violation_count": int(np.sum(b_cl)),
            "profit_budget_feasible": _stat(np.array(p_cl)[cl_budget_ok], args.seed),
            "profit_hard_feasible": _stat(np.array(p_cl), args.seed),
        },
        "per_instance": per_instance,
        "note": "实验2：能耗由逐单 E_class 换成物理路由能耗（UA·ΔT·dwell + E_door）/COP；"
                "接受时 dwell 用历史 mean_dwell_per_class 估计（同一公式，naive 返仓段 dist(o,0)/speed "
                "会 ~10x 低估终局 dwell 导致全盘违约，已弃用）；B=(1−ρ)×历史全接路由能耗均值；"
                "profit_hard_feasible=运营硬约束可行（不含预算）下的利润，用于回答头腔是否仍在；"
                "feasible_rate=硬约束+预算都可行；"
                "2026-09-24 因果口径修正：k-NN 前瞻特征用 min(clock,3h) 时间截面，"
                "路由（含全接基线/clairvoyant 参考）arrival 下界加 reveal_time。",
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
