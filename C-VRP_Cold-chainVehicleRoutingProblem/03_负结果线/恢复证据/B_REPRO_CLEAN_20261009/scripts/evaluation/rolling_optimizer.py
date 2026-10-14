"""A-v1 滚动联合优化 replanner（第 4 步主强对照）。

语义（用户定）：
  - 每次揭示，在时限内联合优化「新订单接受/拒绝 + 尚未执行路线尾部」。
  - 锁定：已执行路段（committed_next）、在途承诺、此前已接受订单（不回滚）。
  - 候选可行性检查时间窗/容量/C0 能耗；找不到更好可行方案回退上一份已认证计划。
  - 非预知：只用当前可见订单。

参数：reject_penalty p_c（默认 (5,10,15)），time_limit（秒）。

接受/拒绝流程（2026-09-24 统一口径）：
  - 无条件未来预算门槛：value/e <= lam 的订单预拒绝（lam 用训练集未来订单边际 value/e），
    为未来高价值订单留预算；e = 逐单 C0 边际 est，仅作门槛比值分母与候选排序，**不是可行性认证**。
  - 候选可行性 = 真实 C0 `certify_plan` 硬认证（时间窗/容量在 _route_feasible，能耗在 certify_plan）。
  - 搜索 = 新单接/拒（≤8 枚举 / >8 按 REV/e 贪心生成候选）+ 贪心插入 + relocate/swap 尾部局部搜索。
"""
from __future__ import annotations

import time

import numpy as np

from coldchain_evaluator_a1 import REV, FUEL_COST_PER_KM
from run_exp_energy_c0 import C0ReserveReplanner, certify_plan

KM_PER_UNIT = 20.0


class RollingOptimizeReplanner(C0ReserveReplanner):
    """滚动联合优化：时限内枚举新单接受/拒绝组合 + 贪心插入，选最高 A-v1 效用。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 reject_penalty, time_limit, hist=None, k=10):
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         mode='myopic', hist=hist, k=k)
        self.reject_penalty = dict(reject_penalty)
        self.time_limit = float(time_limit)

    def _start_states(self, env, inst_idx, vehicles, served_mask, clock, plan):
        """vid -> {cur, cur_time, load, route}；committed 车的 route 排除 committed_next。"""
        st = {}
        for v in vehicles:
            pos = self._vehicle_serve_state(env, inst_idx, v, clock)
            if pos is None:
                continue
            cur, t, load = pos
            committed = v.committed_next if v.status == 'committed' else None
            route = [x for x in plan.get(v.vehicle_id, [])
                     if not served_mask[x] and x != committed]
            st[v.vehicle_id] = dict(cur=cur, cur_time=t, load=load, route=route)
        return st

    def _route_util(self, env, inst_idx, accepted, rejected, st):
        """A-v1 效用 = 收入 − 燃油 − 拒绝损失。燃油从每辆车当前位置（cur）出发计。"""
        rev = sum(REV[int(env.dataset['temp_class'][inst_idx, o])] for o in accepted)
        dist_units = 0.0
        for s in st.values():
            prev = s['cur']
            for o in s['route']:
                dist_units += float(env.dist_mat[inst_idx, prev, o])
                prev = o
            dist_units += float(env.dist_mat[inst_idx, prev, 0])
        fuel = dist_units * KM_PER_UNIT * FUEL_COST_PER_KM
        reject = sum(self.reject_penalty[int(env.dataset['temp_class'][inst_idx, o])]
                     for o in rejected)
        return rev - fuel - reject

    def _greedy_insert_into(self, env, inst_idx, o, st):
        """把 o 贪心插入 st 的最优可行位置（原地改 st）。返回是否成功。"""
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
        return True

    def _route_distance(self, env, inst_idx, cur, route):
        d = 0.0
        prev = cur
        for o in route:
            d += float(env.dist_mat[inst_idx, prev, o])
            prev = o
        d += float(env.dist_mat[inst_idx, prev, 0])
        return d

    def _total_distance(self, env, inst_idx, st):
        return sum(self._route_distance(env, inst_idx, s['cur'], s['route']) for s in st.values())

    def _local_search(self, env, inst_idx, st, t0):
        """最小 relocate + swap 局部搜索（车内 + 跨车未执行订单），减少总距离，超时返回。"""
        improved = True
        while improved and time.time() - t0 < self.time_limit:
            improved = self._relocate(env, inst_idx, st, t0)
            if improved:
                continue
            improved = self._swap(env, inst_idx, st, t0)
        return st

    def _relocate(self, env, inst_idx, st, t0):
        vids = list(st.keys())
        for vid in vids:
            route = st[vid]['route']
            for i, o in enumerate(route):
                for vid2 in vids:
                    if time.time() - t0 > self.time_limit:
                        return False
                    route2 = st[vid2]['route']
                    for j in range(len(route2) + 1):
                        if vid == vid2 and j in (i, i + 1):
                            continue
                        if vid == vid2:
                            removed = route[:i] + route[i + 1:]
                            j_adj = j - 1 if j > i else j
                            new_pairs = [(vid, removed[:j_adj] + [o] + removed[j_adj:])]
                        else:
                            new_pairs = [(vid, route[:i] + route[i + 1:]),
                                         (vid2, route2[:j] + [o] + route2[j:])]
                        ok = all(self._route_feasible(env, inst_idx, st[v]['cur'], st[v]['cur_time'],
                                                      st[v]['load'], nr) for v, nr in new_pairs)
                        if not ok:
                            continue
                        old_d = self._total_distance(env, inst_idx, st)
                        saved = {v: st[v]['route'] for v, _ in new_pairs}
                        for v, nr in new_pairs:
                            st[v]['route'] = nr
                        if self._total_distance(env, inst_idx, st) < old_d - 1e-9:
                            return True
                        for v, r0 in saved.items():
                            st[v]['route'] = r0
        return False

    def _swap(self, env, inst_idx, st, t0):
        items = [(vid, i, st[vid]['route'][i]) for vid in st for i in range(len(st[vid]['route']))]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if time.time() - t0 > self.time_limit:
                    return False
                vid1, i1, o1 = items[a]
                vid2, i2, o2 = items[b]
                if vid1 == vid2:
                    new_route = st[vid1]['route'][:]
                    new_route[i1], new_route[i2] = new_route[i2], new_route[i1]
                    if not self._route_feasible(env, inst_idx, st[vid1]['cur'], st[vid1]['cur_time'],
                                                st[vid1]['load'], new_route):
                        continue
                    old_d = self._total_distance(env, inst_idx, st)
                    saved = st[vid1]['route']
                    st[vid1]['route'] = new_route
                    if self._total_distance(env, inst_idx, st) < old_d - 1e-9:
                        return True
                    st[vid1]['route'] = saved
                else:
                    new1 = st[vid1]['route'][:]; new2 = st[vid2]['route'][:]
                    new1[i1] = o2; new2[i2] = o1
                    ok = self._route_feasible(env, inst_idx, st[vid1]['cur'], st[vid1]['cur_time'],
                                              st[vid1]['load'], new1)
                    ok = ok and self._route_feasible(env, inst_idx, st[vid2]['cur'],
                                                     st[vid2]['cur_time'], st[vid2]['load'], new2)
                    if not ok:
                        continue
                    old_d = self._total_distance(env, inst_idx, st)
                    saved1 = st[vid1]['route']; saved2 = st[vid2]['route']
                    st[vid1]['route'] = new1; st[vid2]['route'] = new2
                    if self._total_distance(env, inst_idx, st) < old_d - 1e-9:
                        return True
                    st[vid1]['route'] = saved1; st[vid2]['route'] = saved2
        return False

    def _optimize(self, env, inst_idx, clock, vehicles, served_mask, new_set):
        """联合优化 new_set 的接受/拒绝组合，选最高效用。超时回退。"""
        orders = sorted(new_set, key=lambda i: env.reveal_time[inst_idx, i])
        n = len(orders)
        t0 = time.time()
        base_plan = {k: list(v) for k, v in self._plan.items()}
        best = None  # (util, plan, acc, rej)

        def evaluate(acc):
            """对接受集合 acc 贪心插入 + 局部搜索，用真实 C0 能耗硬认证预算。"""
            st = self._start_states(env, inst_idx, vehicles, served_mask, clock, base_plan)
            for o in sorted(acc, key=lambda i: env.tw_end[inst_idx, i]):
                if not self._greedy_insert_into(env, inst_idx, o, st):
                    return None, None
            self._local_search(env, inst_idx, st, t0)
            plan = {vid: list(s['route']) for vid, s in st.items() if s['route']}
            ok, _ = certify_plan(env, inst_idx, clock, vehicles, served_mask, plan,
                                 self.contract, self.budget)
            if not ok:
                return None, None
            rej = set(orders) - acc
            util = self._route_util(env, inst_idx, self._accepted | acc, self._rejected | rej, st)
            return util, plan

        if n <= 8:
            for mask in range(1 << n):
                if time.time() - t0 > self.time_limit:
                    break
                acc = {orders[i] for i in range(n) if mask >> i & 1}
                util, plan = evaluate(acc)
                if util is None:
                    continue
                if best is None or util > best[0]:
                    best = (util, plan, acc, set(orders) - acc)
        else:
            ratio = sorted(orders, key=lambda i: -(REV[int(env.dataset['temp_class'][inst_idx, i])]
                                                   / max(self._est_energy(env, inst_idx, i), 1e-9)))
            acc = set()
            for o in ratio:
                if time.time() - t0 > self.time_limit:
                    break
                candidate = acc | {o}
                util, plan = evaluate(candidate)
                if util is None:
                    continue
                if best is None or util > best[0]:
                    best = (util, plan, candidate, set(orders) - candidate)
                acc = candidate
            # 也评估「全拒绝」和「贪心逐单」之外的纯全拒绝基线
            util, plan = evaluate(set())
            if util is not None and (best is None or util > best[0]):
                best = (util, plan, set(), set(orders))

        if best is None:
            return dict(base_plan), set(), set(orders), 0.0
        return best[1], best[2], best[3], best[0]

    def on_reveal(self, env, inst_idx, clock, vehicles, served_mask, visible_ids):
        self._reset_if_new(inst_idx)
        reserved = env.get_reserved_customers(vehicles)
        new_set = set(int(i) for i in visible_ids if not served_mask[i]
                      and int(i) not in reserved and int(i) not in self._accepted
                      and int(i) not in self._rejected)
        if not new_set:
            return
        # 无条件未来预算门槛：value/e <= lam 的订单预拒绝（为未来高价值订单留预算），
        # 其余交给现有搜索（枚举/贪心插入 + relocate/swap + certify_plan 硬认证）。
        lam = self._uncond_lookahead_lambda(env, inst_idx, clock)
        reserve_rej = set()
        for o in new_set:
            e = self._est_energy(env, inst_idx, o)
            if REV[int(env.dataset['temp_class'][inst_idx, o])] <= e * lam + 1e-9:
                reserve_rej.add(o)
        candidates = new_set - reserve_rej
        self._rejected |= reserve_rej
        plan, acc, rej, _ = self._optimize(env, inst_idx, clock, vehicles, served_mask, candidates)
        self._plan = plan
        self._accepted |= acc
        self._rejected |= rej
