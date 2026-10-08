"""A-v1 SAA 下游校准变体（2026-09-26，用户授权「下游校准修复 + 重跑步骤2门」）。

只改 SaaReplanner 的影子场景评估 `_sim_scenario`，不改 scenario_saa.py（步骤 2 归档源码
身份不变）。实际接单步骤的真实 C0 certify（_try_insert_certified）在所有模式下不变。

模式：
  orig  —— 原版（影子计划真实能耗 > B → 分支置 -inf；est 门槛 = 未来策略预算规则）
  fixA  —— 影子计划能耗超标不再置 -inf（TW/容量/返仓不可行仍 -inf）：
           未来策略的预算规则就是 est 门槛，真实 C0 认证只在实际接单步骤执行；
           双分支共享同一近似 → 机会成本比较不被「代理不可行」污染。
  fixC  —— 保持 orig 的 -inf，但把未来策略的 est 门槛乘以 est_scale（校准到
           真实边际能耗 ≈ 9.2-10.4 kWh/单，即 est_scale≈2.5）：
           影子计划不再「按 est 允许 ~129 单、按真实能耗只有 ~68 单」而系统性爆炸。
"""
from __future__ import annotations

from run_exp_energy_c0 import c0_marginal_energy
from coldchain_evaluator_a1 import REV, FUEL_COST_PER_KM, KM_PER_UNIT
from scenario_saa import SaaReplanner


class CalibratedSaaReplanner(SaaReplanner):
    """SAA 下游校准子类：sim_mode ∈ {orig, fixA, fixC} + est_scale。"""

    def __init__(self, budget, capacity, booking_horizon, contract, cooling_share,
                 sampler, reject_penalty, K=10, time_limit=10.0, arm_seed=7001,
                 sim_mode='fixA', est_scale=1.0):
        if sim_mode not in ('orig', 'fixA', 'fixC'):
            raise ValueError('sim_mode must be orig|fixA|fixC')
        super().__init__(budget, capacity, booking_horizon, contract, cooling_share,
                         sampler, reject_penalty, K=K, time_limit=time_limit,
                         arm_seed=arm_seed)
        self.sim_mode = sim_mode
        self.est_scale = float(est_scale)

    def _sim_scenario(self, env, inst_idx, clock, st, scen_idx, space):
        st = {vid: dict(cur=s['cur'], cur_time=s['cur_time'], load=s['load'],
                        route=list(s['route']), cc=s['cc']) for vid, s in st.items()}
        deadline = float(env.tw_end[inst_idx, 0])
        est_used = sum(self._est_energy(env, inst_idx, o)
                       for s in st.values() for o in s['route'])
        served_rev = 0.0
        rej_loss = 0.0
        for o in sorted(scen_idx, key=lambda i: space.reveal[i]):
            tc = int(space.tc[o])
            e = (c0_marginal_energy(tc, float(space.demand[o]),
                                    self.cooling_share, self.contract)
                 * self.est_scale)
            if est_used + e > self.budget + 1e-9:
                rej_loss += self.reject_penalty[tc]
                continue
            if not self._greedy_insert_u(space, env, o, st, deadline):
                rej_loss += self.reject_penalty[tc]
                continue
            est_used += e
            served_rev += REV[tc]
        energy, dist_units = self._complete_u(space, env, inst_idx, clock, st,
                                              self.booking_horizon)
        if energy is None:
            return float('-inf')   # 真正不可行（TW/容量/返仓）——任何模式都置 -inf
        if self.sim_mode in ('orig', 'fixC') and energy > self.budget + 1e-9:
            return float('-inf')
        # fixA：能耗超标不置 -inf —— est 规则即未来策略的预算语义
        plan_rev = sum(REV[int(space.tc[o])] for s in st.values() for o in s['route'])
        fuel = dist_units * KM_PER_UNIT * FUEL_COST_PER_KM
        return plan_rev + served_rev - fuel - rej_loss
