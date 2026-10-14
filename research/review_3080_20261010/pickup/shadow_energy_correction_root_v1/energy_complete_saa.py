"""Scoped resource-ledger correction; old SAA source/results remain unchanged."""
import copy, math
from scenario_saa import *
from scenario_saa import SaaReplanner as OriginalSAA
class EnergyCompleteSAA(OriginalSAA):
    def _sim_states(self,env,inst_idx,clock,vehicles,served_mask,plan):
        states=super()._sim_states(env,inst_idx,clock,vehicles,served_mask,plan)
        fixed_energy=0.;fixed_distance=0.;entries=[]
        for v in vehicles:
            if v.vehicle_id in states or v.coldchain_state is None:continue
            state=copy.deepcopy(v.coldchain_state)
            if v.status=='returning' and not state.closed:
                finish=float(v.return_finish)
                if not math.isfinite(finish) or finish<float(clock)-1e-6:
                    raise ValueError('Invalid immutable return deadline')
                distance=float(env.dist_mat[inst_idx,v.current_node,0])
                state,_=transition_segment(state,depart_time=float(clock),arrival_time=finish,service_finish=finish,
                    served_customer=None,active_zone_mask=(True,)*len(self.contract.thermal.supported_temp_classes),
                    contract=self.contract,return_to_depot=True,segment_distance_units=distance)
                fixed_distance+=distance
            elif not state.closed:
                raise ValueError('Excluded vehicle has no immutable closed/returning accounting')
            fixed_energy+=float(state.cumulative_energy_kwh)
            entries.append({'vehicle_id':int(v.vehicle_id),'status':v.status,'fixed_energy_kwh':float(state.cumulative_energy_kwh)})
        self._fixed_unavailable_energy=fixed_energy
        self._fixed_unavailable_distance=fixed_distance
        self.unavailable_ledger=entries
        return states
    def _complete_u(self,space,env,inst_idx,clock,st,wait_until):
        energy,distance=super()._complete_u(space,env,inst_idx,clock,st,wait_until)
        if energy is None:return None,None
        return energy+self._fixed_unavailable_energy,distance+self._fixed_unavailable_distance
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
        est_used += self._fixed_unavailable_energy
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
