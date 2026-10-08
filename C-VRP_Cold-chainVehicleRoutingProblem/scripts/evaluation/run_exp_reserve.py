"""真 3b · 步骤2：持久预留（reservation）的因果承诺接受 —— 把承诺违约清零。

与 run_exp_capacity.py（v5）的区别（这是本脚本的实质）：
  1. **持久计划 self._plan**（vid -> 接受订单路由）：接受当刻把订单**锁进**某辆车的路线
     （NN 最优插入 + 全路线 feasibility 认证），之后 plan() 只**回放**该计划，绝不丢弃。
     修复了「fresh-greedy 每次揭示重路由、把已接受订单挤掉」的违约根因。
  2. **时序一致**：`_vehicle_serve_state` 对 ready/idle 用 `max(ready_time, clock)`，
     与 env 的 `prepare_decision_point` 口径一致（on_reveal 在 prepare 之前被调用，
     旧代码用了未 bump 的 ready_time，乐观了等待时间）。
  3. 接受判据 = 预算内 + (k-NN 价值门槛) + **插入持久计划可行**。插入可行 iff 存在某个
     位置使全路线 TW/容量/depot-return 都满足（min-incr 贪心找可行位，找不到即拒绝）。

其余（NN 插入、myopic/k-NN/clairvoyant、逐单 E_class 能耗、四项闭环、逐日输出）与 v5 一致。
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
FUEL_COST_PER_KM = 0.1
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


class ReserveReplanner(Replanner):
    """持久预留 replanner：接受当刻把订单锁进持久计划，plan() 只回放，绝不丢弃。"""

    def __init__(self, budget, capacity, booking_horizon, mode='myopic', hist=None, k=10):
        self.budget = budget
        self.capacity = capacity
        self.booking_horizon = booking_horizon
        self.mode = mode
        self.hist = hist
        self.k = k
        self._accepted = set()
        self._rejected = set()
        self._plan = {}  # vid -> 接受订单路由（post-commit / from-current）
        self._status_inst = -1

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            self._status_inst = int(inst_idx)
            self._accepted = set()
            self._rejected = set()
            self._plan = {}

    def _vehicle_serve_state(self, env, inst_idx, v, clock):
        """车辆下一次可自由派车的 (cur, earliest_depart, load)；ready/idle 用 max(ready_time, clock)。"""
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
            if l > self.capacity + 1e-6:
                return False
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
            route = [o for o in self._plan.get(v.vehicle_id, []) if not served_mask[o]]
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=route)
        return st

    def _try_insert(self, env, inst_idx, o, vehicles, served_mask, clock):
        """把 o 锁进持久计划（NN 最优插入到某车路线），全路线可行则返回 True 并更新 self._plan。"""
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
            [E_CLASS[int(env.dataset['temp_class'][inst_idx, o2])] for o2 in self._accepted]))
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
            if e > remaining + 1e-9:
                self._rejected.add(o)
                continue
            if self.mode == 'knn' and self.hist is not None:
                lam = self._lookahead_lambda(env, inst_idx, clock, o)
                if REV[int(tc[inst_idx, o])] <= e * lam + 1e-9:
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
        return {'version': 'reserve-v1', 'budget': self.budget, 'mode': self.mode,
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


def run_policy(test_ds, B, capacity, num_vehicles, mode, hist, k):
    replanner = ReserveReplanner(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
                                 mode=mode, hist=hist, k=k)
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
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    train_ds = generate_dataset(args.train_instances, args.n_orders, args.seed)
    test_ds = generate_dataset(args.test_instances, args.n_orders, args.seed + 1)
    train_full = np.array([instance_full_energy(train_ds, i) for i in range(args.train_instances)])
    B = (1.0 - args.rho) * float(train_full.mean())

    hist = []
    for i in range(args.train_instances):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    p_my, e_my, f_my, ns_my = run_policy(test_ds, B, args.capacity, args.num_vehicles, 'myopic', hist, args.k)
    p_kn, e_kn, f_kn, ns_kn = run_policy(test_ds, B, args.capacity, args.num_vehicles, 'knn', hist, args.k)

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
        "note": "reserve-v1（步骤2）：持久预留 self._plan，接受当刻锁单、plan() 只回放不丢弃；"
                "ready/idle 时序用 max(ready_time, clock)；k-NN 前瞻特征 min(clock,3h) 时间截面；"
                "clairvoyant 参考加 reveal_time。",
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
