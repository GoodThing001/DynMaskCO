# -*- coding: utf-8 -*-
"""A1 强动态对照两臂（2026-10-03，设计件《A1强动态对照_助手实施设计_2026-10-03.md》）。

- `ConsensusReplanner`（合议臂）：逐场景投票（d_i = u_acc − u_rej > 0，rej 侧已含 −pc_o）、
  多数接单、平票拒绝（与旧「保计划」同向）；插入共用 `_try_insert_certified`（C0 硬认证）；
  超时语义沿用协议（保计划+拒单+计 timeouts）；报告 SAA 完成率计数。
- `RolloutVReplanner`（rollout 臂）：h=2 截断影子 + 终端剩余价值 V（14 维固定特征线性模型）；
  截断口径 = 未来池只取 reveal ≤ clock+h 的单；截断点用 `_complete_u` 算「不再插入」的完成值
  u_cutoff，再加 V(s_after)。V 权重来自训练日因果回放回归（train_a1_rollout_v.py）。
- 两臂均不改 `SaaReplanner` 既有路径语义；只新增计数与覆盖 `_saa_decide`。
"""
from __future__ import annotations

import time

import numpy as np

from coldchain_evaluator_a1 import FUEL_COST_PER_KM, KM_PER_UNIT, REV
from run_exp_energy_c0 import c0_marginal_energy
from scenario_saa import INFEASIBLE_SCENARIO_PENALTY, SaaReplanner

V_FEATURE_VERSION = 1
V_FEATURE_DIM = 14


class ConsensusReplanner(SaaReplanner):
    """A1 臂 1：场景合议接单（多数票；平票拒）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scen_completed_sum = 0
        self.n_decisions = 0

    def _reset_if_new(self, inst_idx):
        if inst_idx != self._status_inst:
            super()._reset_if_new(inst_idx)
            self.scen_completed_sum = 0
            self.n_decisions = 0

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
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
        votes_accept = 0
        m = 0
        for scen_idx in scen_indices:
            if time.perf_counter() > deadline:
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                return
            acc_u = self._sim_scenario(env, inst_idx, clock, acc_states, scen_idx, space)
            rej_u = self._sim_scenario(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
            m += 1
            if acc_u > rej_u:
                votes_accept += 1
        if time.perf_counter() > deadline:   # 提交前复查（同 A-02）
            self._plan = saved_plan
            self._rejected.add(o)
            self.timeouts += 1
            return
        self.scen_completed_sum += m
        self.n_decisions += 1
        if votes_accept * 2 > m:   # 多数票接单；平票拒（保守）
            self._plan = acc_plan
            self._accepted.add(o)
        else:
            self._plan = saved_plan
            self._rejected.add(o)

    @property
    def saa_completion_rate(self):
        if self.n_decisions == 0:
            return None
        return float(self.scen_completed_sum) / (self.K * self.n_decisions)


def a1_v_features(space, env, inst_idx, clock, st, est_used, budget, capacity, h):
    """14 维固定特征（设计件 §2，V_FEATURE_VERSION=1）。

    [0:3]=每车 rem_time/rem_cap/geo 的 mean；[3:6]=max；[6:9]=var（跨车方差显式）；
    [9]=B−est_used（剩余能耗 slack，车队级标量）；[10]=n_active（有路线或 frozen 的车数）；
    [11]=n_remaining（场景池中 reveal>clock+h 的剩余单数）；[12]=其需求总和；
    [13]=clock。
    """
    deadline = float(env.tw_end[inst_idx, 0])
    inv = 1.0 / env.tw_speed
    D = space.D
    feats = []
    n_active = 0
    n_remaining = 0
    tot_remaining = 0.0
    for o in range(int(space.n_real), int(space.reveal.shape[0])):
        if float(space.reveal[o]) > float(clock) + float(h) + 1e-9:
            n_remaining += 1
            tot_remaining += float(space.demand[o])
    for s in st.values():
        cur = int(s['cur'])
        t = float(s['cur_time'])
        if s.get('route') or s.get('frozen', 0):
            n_active += 1
        rem_time = deadline - t - D[cur, 0] * inv
        geo = 0.0
        for o in range(int(space.n_real), int(space.reveal.shape[0])):
            if float(space.reveal[o]) <= float(clock) + float(h) + 1e-9:
                continue
            depart = t if t >= float(space.reveal[o]) else float(space.reveal[o])
            arr = depart + D[cur, o] * inv
            if arr < float(space.tws[o]):
                arr = float(space.tws[o])
            if arr <= float(space.twe[o]) + 1e-6:
                geo += 1.0 / (1.0 + D[cur, o])
        feats.append([max(0.0, rem_time),
                      max(0.0, float(capacity) - float(s['load'])),
                      float(geo)])
    if feats:
        A = np.asarray(feats, dtype=np.float64)
        per = np.concatenate([A.mean(0), A.max(0), A.var(0)])
    else:
        per = np.zeros(9, dtype=np.float64)
    rest = np.array([float(budget) - float(est_used), float(n_active),
                     float(n_remaining), float(tot_remaining), float(clock)],
                    dtype=np.float64)
    return np.concatenate([per, rest])


class RolloutVReplanner(SaaReplanner):
    """A1 臂 2：h=2 截断影子 + 终端剩余价值 V（线性 14 维特征模型）。

    v_ckpt=None → V≡0（纯截断影子，供测试/烟测）；v_ckpt 为 train_a1_rollout_v.py 产物 npz
    （含 w、b、f_mean、f_std、feature_version、identity 字段）。V 模型预载不入决策时限
    （规格允许，如实声明）。
    """

    def __init__(self, *args, v_ckpt=None, h=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.overage_price is not None:
            raise ValueError('RolloutVReplanner 不支持 overage_price（h2 截断与 S3-3 超额价格'
                             '口径不兼容；A1 臂只用 campaign 标准配置 marginal/no-overage）')
        self.h = float(h)
        self._v_w = None
        self._v_b = 0.0
        self._v_mean = None
        self._v_std = None
        if v_ckpt is not None:
            z = np.load(v_ckpt)
            if int(z.get('feature_version', -1)) != V_FEATURE_VERSION:
                raise ValueError('V 权重特征版本不匹配：%s vs %d'
                                 % (z.get('feature_version'), V_FEATURE_VERSION))
            self._v_w = np.asarray(z['w'], dtype=np.float64)
            self._v_b = float(z['b'])
            self._v_mean = np.asarray(z['f_mean'], dtype=np.float64)
            self._v_std = np.asarray(z['f_std'], dtype=np.float64)

    def _v_predict(self, f):
        if self._v_w is None:
            return 0.0
        fs = (np.asarray(f, dtype=np.float64) - self._v_mean) / np.maximum(self._v_std, 1e-12)
        return float(self._v_b + float(np.dot(self._v_w, fs)))

    def _shadow_insert_loop(self, space, env, inst_idx, clock, st, scen_idx, est_used,
                            rej_loss, share, h):
        """h=2 截断影子插入循环（与 _sim_scenario 同门槛/顺序口径，只处理 reveal≤clock+h）。
        返回 (est_used, rej_loss)。"""
        if self.future_policy == 'density':
            def _density(i):
                e_i = c0_marginal_energy(int(space.tc[i]), float(space.demand[i]),
                                         share, self.contract)
                return REV[int(space.tc[i])] / max(e_i, 1e-9)
            scen_order = sorted(scen_idx, key=lambda i: -_density(i))
        else:
            scen_order = sorted(scen_idx, key=lambda i: space.reveal[i])
        for o in scen_order:
            if float(space.reveal[o]) > float(clock) + h + 1e-9:
                continue   # h=2 截断：窗口外单交给终端 V
            tc = int(space.tc[o])
            e = c0_marginal_energy(tc, float(space.demand[o]), share, self.contract)
            if est_used + e > self.budget + 1e-9:
                rej_loss += self.reject_penalty[tc]
                continue
            if not self._insert_u(space, env, o, st, float(env.tw_end[inst_idx, 0])):
                rej_loss += self.reject_penalty[tc]
                continue
            est_used += e
            if self.shadow_mode == 'ls':
                self._local_search_u(space, env, st, float(env.tw_end[inst_idx, 0]))
        return est_used, rej_loss

    def _h2_shadow_state(self, env, inst_idx, clock, st, scen_idx, space):
        """h=2 截断影子（深拷贝状态 + 窗口内插入循环 + 无插入完成）。
        返回 (st2, est_used, rej_loss, energy, dist_units)；完成不可行 → None。"""
        st2 = {vid: dict(cur=s['cur'], cur_time=s['cur_time'], load=s['load'],
                         route=list(s['route']), cc=s['cc'], frozen=s.get('frozen', 0))
               for vid, s in st.items()}
        share = self.cooling_share if self.energy_pricing == 'amortized' else 0.0
        est_used = sum(float(s['cc'].cumulative_energy_kwh) for s in st2.values()
                       if s['cc'] is not None)
        for s in st2.values():
            for o in s['route']:
                est_used += c0_marginal_energy(
                    int(env.dataset['temp_class'][inst_idx, o]),
                    float(env.demands[inst_idx, o]), share, self.contract)
        if self.energy_pricing == 'marginal' and self.standby_orders:
            est_used += self.cooling_share * float(self.standby_orders)
        rej_loss = 0.0
        est_used, rej_loss = self._shadow_insert_loop(
            space, env, inst_idx, clock, st2, scen_idx, est_used, rej_loss, share, self.h)
        energy, dist_units = self._complete_u(space, env, inst_idx, clock, st2,
                                              self.booking_horizon)
        if energy is None:
            return None
        return st2, est_used, rej_loss, energy, dist_units

    def _sim_scenario_h2(self, env, inst_idx, clock, st, scen_idx, space):
        """h=2 截断影子 + V(s_after)。u = u_cutoff + V；u_cutoff = 截断点完成值
        （plan_rev − fuel − rej_loss，_complete_u 无插入完成）；完成不可行 → 同全影子
        INFEASIBLE_SCENARIO_PENALTY。"""
        res = self._h2_shadow_state(env, inst_idx, clock, st, scen_idx, space)
        if res is None:
            return INFEASIBLE_SCENARIO_PENALTY
        st2, est_used, rej_loss, _energy, dist_units = res
        plan_rev = sum(REV[int(space.tc[o])] for s in st2.values() for o in s['route'])
        fuel = dist_units * KM_PER_UNIT * FUEL_COST_PER_KM
        u_cutoff = plan_rev - fuel - rej_loss
        f = a1_v_features(space, env, inst_idx, clock, st2, est_used, self.budget,
                          self.capacity, self.h)
        return u_cutoff + self._v_predict(f)

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
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
            self._plan = {k: list(v) for k, v in saved_plan.items()}
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
            if time.perf_counter() > deadline:
                self._plan = saved_plan
                self._rejected.add(o)
                self.timeouts += 1
                return
            acc_u += self._sim_scenario_h2(env, inst_idx, clock, acc_states, scen_idx, space)
            rej_u += self._sim_scenario_h2(env, inst_idx, clock, rej_states, scen_idx, space) - pc_o
        if time.perf_counter() > deadline:
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
