"""真 3b · 因果承诺评价器 v5：容量/TW 感知接受判据（修 k-NN 过度承诺）。

在 run_causal_accept_gate.py (v4) 基础上修改，唯一实质改动 = 接受判据从
「松 _can_serve」（只查"某车最早可服务时刻≤截止"，忽略该车 committed leg 的位置/载重、
忽略 depot-return deadline）改为「紧 _can_serve_tight」：

  对每辆车重建"下一次可自由派车"状态 (cur, t, load)：
    - committed 车：cur = committed_next（而非过时的 current_node），
      t = committed_finish，load += demand[committed_next]（committed leg 的 pickup）；
    - ready 车：cur = current_node, t = ready_time, load = current_load；
    - idle 车：cur = 0, t = ready_time, load = 0；
    - returning / closed / committed_next==0(返仓)：不可服务。
  然后复用 _route_feasible([o]) 校验：TW + 容量 + depot-return deadline（tw_end[0]）。

v4 的松判据有两处错位 + 一处漏检，导致 k-NN 接受过多、承诺违约：
  (1) committed 车用 current_node（出发原点）算距离，而非 committed_next（落点）；
  (2) committed 车 load 漏加 committed_next 的 pickup，容量低估；
  (3) 漏检 depot-return deadline（与 GreedyReplanner P0-B0 同源）。

四项闭环保留（因果 booking_horizon / 承诺 accepted 持久 / 独立认证 / 失败计分）。
其余（NN 最优插入路由、myopic/k-NN/clairvoyant、逐单能耗 E_class）与 v4 一致。
--serve-check {tight,loose} 可选回退到 v4 松判据，便于对照。
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
D_MEAN_KM = 0.5 * KM_PER_UNIT
E_CLASS = {c: (UA * DT[c] * (D_MEAN_KM / SPEED_KMH) + E_DOOR) / COP for c in range(3)}
BOOKING_HORIZON = 16.0


def _softmax(w):
    w = np.asarray(w, dtype=np.float64)
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


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


def instance_full_energy(dataset, i):
    tc = dataset['temp_class'][i]
    dem = dataset['demands'][i]
    return float(np.sum([E_CLASS[int(tc[o])] for o in range(1, tc.shape[0]) if dem[o] > 0]))


def _morning_feat(day, tau=3.0):
    times, cls, locs = day
    m = times <= tau
    feat = np.zeros(6)
    for c in range(3):
        feat[3 + c] = np.sum(cls[m] == c)
    if m.sum() > 0:
        d = np.linalg.norm(locs[m][:, None, :] - np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])[None, :, :], axis=2)
        nearest = d.argmin(1)
        for s in range(3):
            feat[s] = np.sum(nearest == s)
    return feat


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
                 serve_check='tight'):
        self.budget = budget
        self.capacity = capacity
        self.booking_horizon = booking_horizon
        self.mode = mode
        self.hist = hist  # list of (times, classes) per train day
        self.k = k
        self.serve_check = serve_check  # 'tight' (v5) | 'loose' (v4)
        self._accepted = set()
        self._rejected = set()
        self._status_inst = -1

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = int(inst_idx)
            self._accepted = set()
            self._rejected = set()

    def _vehicle_serve_state(self, env, inst_idx, v):
        """车辆"下一次可自由派车"的 (cur, earliest_depart, load)，或 None（不可再服务）。"""
        if v.status == 'committed':
            nxt = v.committed_next
            if nxt in (None, 0):
                return None  # 正在返仓或无 committed leg，不能中途接新单
            return (int(nxt), float(v.committed_finish),
                    float(v.current_load) + float(env.demands[inst_idx, nxt]))
        if v.status == 'ready':
            return (int(v.current_node), float(v.ready_time), float(v.current_load))
        if v.status == 'idle':
            return (0, float(v.ready_time), 0.0)
        return None  # returning / closed

    def _can_serve_tight(self, env, inst_idx, o, vehicles):
        """紧判据：把 o 插到某车"committed leg 之后"的可行位置，仍 TW/容量/depot-return 可行。"""
        for v in vehicles:
            st = self._vehicle_serve_state(env, inst_idx, v)
            if st is None:
                continue
            cur, t, load = st
            if self._route_feasible(env, inst_idx, cur, t, load, [o]):
                return True
        return False

    def _fleet_start_states(self, env, inst_idx, vehicles):
        """所有可用车辆（含 committed leg 之后）的下一次自由派车起点。"""
        st = {}
        for v in vehicles:
            pos = self._vehicle_serve_state(env, inst_idx, v)
            if pos is None:
                continue
            cur, t, load = pos
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=[])
        return st

    def _assign_all(self, env, inst_idx, pending, start_states):
        """EDD + NN 最优插入，把 pending 贪心分配到 start_states；返回 (assigned, dropped)。"""
        dropped = []
        for o in pending:
            best_v, best_pos, best_incr = None, None, float('inf')
            for vid, s in start_states.items():
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
                dropped.append(o)
            else:
                start_states[best_v]['route'].insert(best_pos, o)
                self._recompute_end(env, inst_idx, start_states[best_v])
        return len(dropped) == 0, dropped

    def _can_serve_capacity(self, env, inst_idx, vehicles, served_mask, o):
        """容量/TW 感知（collective）：车队在 committed legs 之上能否再服务
        {accepted-unserved-uncommitted} ∪ {o}（EDD + NN 插入贪心，与 plan() 同款路由器）。"""
        reserved = env.get_reserved_customers(vehicles)
        pending = sorted([p for p in (self._accepted | {o})
                          if not served_mask[p] and p not in reserved],
                         key=lambda i: env.tw_end[inst_idx, i])
        fleet = self._fleet_start_states(env, inst_idx, vehicles)
        ok, _ = self._assign_all(env, inst_idx, pending, fleet)
        return ok

    def _can_serve(self, env, inst_idx, clock, o, vehicles, served_mask):
        if self.serve_check == 'capacity':
            return self._can_serve_capacity(env, inst_idx, vehicles, served_mask, o)
        if self.serve_check == 'tight':
            return self._can_serve_tight(env, inst_idx, o, vehicles)
        # ---- v4 松判据（保留作对照）----
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
        """k-NN 未来边际 λ：用 morning 特征找 k 个历史天，取时刻 clock 后未来订单的 value/energy 分数边际。"""
        # 因果口径（2026-09-24 修正）：当前时刻 clock 只能看到 reveal_time <= clock 的订单；
        # 历史天用相同时间截面 min(clock, 3h) 匹配，避免把未揭示订单算进特征。
        t_cut = min(float(clock), 3.0)
        m = env.dataset['reveal_time'][inst_idx] <= t_cut
        feat = np.zeros(3)
        for c in range(3):
            feat[c] = np.sum(env.dataset['temp_class'][inst_idx][m] == c)
        # 历史天特征（同一时间截面）
        hfeat = np.array([np.array([np.sum(d[1][d[0] <= t_cut] == c) for c in range(3)]) for d in self.hist])
        hn = (hfeat - hfeat.mean(0)) / (hfeat.std(0) + 1e-6)
        x = (feat - hfeat.mean(0)) / (hfeat.std(0) + 1e-6)
        d = ((hn - x) ** 2).sum(1)
        kn = np.argsort(d)[:self.k]
        rem = self.budget - float(np.sum([E_CLASS[int(env.dataset['temp_class'][inst_idx, o2])] for o2 in self._accepted]))
        lams = []
        for h in kn:
            ht, hc = self.hist[h]
            f = ht > clock
            fv = np.array([REV[int(c)] for c in hc[f]])
            fe = np.array([E_CLASS[int(c)] for c in hc[f]])
            lams.append(_marginal_lambda(fv, fe, rem))
        return float(np.mean(lams))

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        tc = env.dataset['temp_class']
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        accepted_energy = float(np.sum([E_CLASS[int(tc[inst_idx, o])] for o in self._accepted]))
        remaining = self.budget - accepted_energy
        for o in sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i]):
            e = E_CLASS[int(tc[inst_idx, o])]
            if e > remaining + 1e-9 or not self._can_serve(env, inst_idx, clock, o, vehicles, served_mask):
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
        return {'version': 'accept-reject-v5', 'budget': self.budget, 'mode': self.mode,
                'serve_check': self.serve_check,
                'accepted': sorted(self._accepted), 'rejected': sorted(self._rejected),
                'status_inst': self._status_inst}

    def restore_state(self, state):
        self.budget = state['budget']
        self.mode = state['mode']
        self.serve_check = state.get('serve_check', 'tight')
        self._accepted = set(state['accepted'])
        self._rejected = set(state['rejected'])
        self._status_inst = state['status_inst']


def evaluate_trace(inst_idx, env, traces, dataset, accepted_set, B):
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
    energy = float(np.sum([E_CLASS[int(tc[o])] for o in served_set]))
    if energy > B + 1e-9:
        failures.append('budget_violation')
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
        return float('nan'), energy, set(failures)
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
    return profit, energy, set()


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


def clairvoyant_profit(test_ds, i, B, capacity, num_vehicles):
    """离线 clairvoyant 参考：按 value/energy 贪心接受 + NN 插入路由，返回 (profit, feasible, energy)。

    物理可执行口径（2026-09-24 修正）：reveal_time 是最早可服务时刻（本数据集 tw_start≡0）。
    """
    tc = test_ds['temp_class'][i]
    dem = test_ds['demands'][i]
    tw = test_ds['tw_end'][i]
    tws = test_ds['tw_start'][i]
    st_ = test_ds['service_time'][i]
    rev = test_ds['reveal_time'][i]
    coords = test_ds['coords'][i]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    orders = [o for o in range(1, tc.shape[0]) if dem[o] > 0]
    ratio = {o: REV[int(tc[o])] / E_CLASS[int(tc[o])] for o in orders}
    orders.sort(key=lambda o: -ratio[o])

    accepted = []
    rem = B
    # 简单路由：每单插到最佳可行位置（单序列，多车用 round-robin 近似）
    routes = [[] for _ in range(num_vehicles)]
    for o in orders:
        e = E_CLASS[int(tc[o])]
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

    # 评估（简化：距离 + 品质，与在线口径一致）
    served = set(accepted)
    energy = sum(E_CLASS[int(tc[o])] for o in served)
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
        if load > capacity + 1e-6:
            return False
        t = arr + st_[o]
        cur = o
    ret = t + dist[cur, 0] / (SPEED_KMH / KM_PER_UNIT)
    return ret <= tw[0] + 1e-6


def run_policy(test_ds, train_ds, B, capacity, num_vehicles, mode, hist, k, serve_check='tight'):
    replanner = AcceptRejectReplanner(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
                                      mode=mode, hist=hist, k=k, serve_check=serve_check)
    env = StrictOnlineEnv(test_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=None, booking_horizon=BOOKING_HORIZON)
    profits, energies, fail_sets, n_served = [], [], [], []
    for i in range(test_ds['coords'].shape[0]):
        traces, served_mask = env.run(i)
        accepted_i = set(replanner._accepted)
        profit, energy, fails = evaluate_trace(i, env, traces, test_ds, accepted_i, B)
        profits.append(profit)
        energies.append(energy)
        fail_sets.append(fails)
        n_served.append(len([s.node for t in traces for s in t.services]))
    return np.array(profits), np.array(energies), fail_sets, np.array(n_served)


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
    ap.add_argument("--serve-check", type=str, default='tight', choices=['tight', 'loose', 'capacity'])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)
    train_full = np.array([instance_full_energy(train_ds, i) for i in range(args.train_instances)])
    B = (1.0 - args.rho) * float(train_full.mean())

    # k-NN 历史（train 天：(reveal_time, class)，只含真实订单，排除 depot + dummy padding）
    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    sc = args.serve_check
    p_my, e_my, f_my, ns_my = run_policy(test_ds, train_ds, B, args.capacity, args.num_vehicles, 'myopic', hist, args.k, sc)
    p_kn, e_kn, f_kn, ns_kn = run_policy(test_ds, train_ds, B, args.capacity, args.num_vehicles, 'knn', hist, args.k, sc)

    p_cl, e_cl, f_cl = [], [], []
    for i in range(args.test_instances):
        prof, feas, energy = clairvoyant_profit(test_ds, i, B, args.capacity, args.num_vehicles)
        p_cl.append(prof if feas else float('nan'))
        e_cl.append(energy)
        f_cl.append(set() if feas else {'clairvoyant_infeasible'})

    def fail_counts(f_sets):
        c = Counter()
        for f in f_sets:
            for r in f:
                c[r] += 1
        return dict(c)

    def feasible_rate(f_sets):
        return float(np.mean([len(f) == 0 for f in f_sets]))

    my_ok = np.array([len(f) == 0 for f in f_my])
    kn_ok = np.array([len(f) == 0 for f in f_kn])
    cl_ok = np.array([len(f) == 0 for f in f_cl])
    both_ok = my_ok & kn_ok

    per_instance = []
    for i in range(args.test_instances):
        per_instance.append({
            "i": int(i),
            "myopic_profit": (float(p_my[i]) if np.isfinite(p_my[i]) else None),
            "myopic_feasible": bool(my_ok[i]),
            "myopic_fails": sorted(f_my[i]),
            "knn_profit": (float(p_kn[i]) if np.isfinite(p_kn[i]) else None),
            "knn_feasible": bool(kn_ok[i]),
            "knn_fails": sorted(f_kn[i]),
            "clairvoyant_profit": (float(p_cl[i]) if np.isfinite(p_cl[i]) else None),
            "clairvoyant_feasible": bool(cl_ok[i]),
            "clairvoyant_fails": sorted(f_cl[i]),
            "both_feasible": bool(both_ok[i]),
            "knn_minus_myopic": (float(p_kn[i] - p_my[i])
                                if (np.isfinite(p_kn[i]) and np.isfinite(p_my[i])) else None),
        })

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "budget": {"B": float(B), "train_full_energy_mean": float(train_full.mean())},
        "myopic": {"feasible_rate": feasible_rate(f_my), "feasible_n": int(my_ok.sum()),
                   "fail_counts": fail_counts(f_my),
                   "profit": _stat(p_my, args.seed), "n_served_mean": float(np.mean(ns_my))},
        "knn": {"feasible_rate": feasible_rate(f_kn), "feasible_n": int(kn_ok.sum()),
                "fail_counts": fail_counts(f_kn),
                "profit": _stat(p_kn, args.seed), "n_served_mean": float(np.mean(ns_kn))},
        "clairvoyant": {"feasible_rate": feasible_rate(f_cl), "feasible_n": int(cl_ok.sum()),
                        "fail_counts": fail_counts(f_cl),
                        "profit": _stat(np.array(p_cl), args.seed)},
        "per_instance": per_instance,
        "note": "v5：容量/TW 感知接受判据（serve_check=%s）；四项闭环保留；能耗逐单 E_class 简化；"
                "2026-09-24 因果口径修正：k-NN 前瞻特征用 min(clock,3h) 时间截面、clairvoyant 参考加 reveal_time。" % sc,
    }
    if both_ok.sum() > 0:
        report["knn_minus_myopic_paired_both_feasible"] = _stat(p_kn[both_ok] - p_my[both_ok], args.seed)
        report["paired_n_both_feasible"] = int(both_ok.sum())
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
