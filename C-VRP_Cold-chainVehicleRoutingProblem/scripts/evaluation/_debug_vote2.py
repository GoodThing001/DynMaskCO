"""临时诊断2：打印 acc/rej 影子计划的真实能耗与订单数（找边界差）。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, 'coldchain'), os.path.join(_ROOT, 'simulation'),
          os.path.join(_ROOT, 'evaluation')):
    if p not in sys.path:
        sys.path.insert(0, p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import SaaReplanner, CondHistoricalSampler, build_history, make_c0_contract_v2
from strict_online_env import StrictOnlineEnv

LOG = []
_orig = SaaReplanner._saa_decide


def patched(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
    if len(LOG) < 2:
        saved_plan = {k: list(v) for k, v in self._plan.items()}
        try:
            insert_ok = self._try_insert_certified(env, inst_idx, o, vehicles,
                                                   served_mask, clock)
        except ValueError:
            insert_ok = False
        if insert_ok:
            acc_plan = {k: list(v) for k, v in self._plan.items()}
            space, scen_orders = self._build_space(env, inst_idx, scenarios)
            idx_of = {id(x): space.n_real + i for i, x in enumerate(scen_orders)}
            scen_indices = [[idx_of[id(x)] for x in scen] for scen in scenarios]
            acc_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, acc_plan)
            self._plan = saved_plan
            rej_states = self._sim_states(env, inst_idx, clock, vehicles, served_mask, saved_plan)
            for label, states in (('acc', acc_states), ('rej', rej_states)):
                scen_idx = scen_indices[0]
                st = {vid: dict(cur=s['cur'], cur_time=s['cur_time'], load=s['load'],
                                route=list(s['route']), cc=s['cc'])
                      for vid, s in states.items()}
                deadline = float(env.tw_end[inst_idx, 0])
                est_used = sum(float(s['cc'].cumulative_energy_kwh)
                               for s in st.values() if s['cc'] is not None)
                est_used += sum(self._est_energy(env, inst_idx, x)
                                for s in st.values() for x in s['route'])
                n_fut = 0
                for x in sorted(scen_idx, key=lambda i: space.reveal[i]):
                    tc = int(space.tc[x])
                    e = self._est_energy(env, inst_idx, x) if False else None
                    e = self._est_scenario_energy(x, space, env, inst_idx)
                    if est_used + e > self.budget + 1e-9:
                        break
                    if not self._greedy_insert_u(space, env, x, st, deadline):
                        break
                    est_used += e
                    n_fut += 1
                energy, dist = self._complete_u(space, env, inst_idx, clock, st,
                                                self.booking_horizon)
                print('%s branch: plan_orders=%d future_crammed=%d est_used=%.1f '
                      'real_energy=%s B=%.1f' % (
                          label, sum(len(s['route']) for s in st.values()) - n_fut,
                          n_fut, est_used, None if energy is None else round(energy, 1),
                          self.budget), flush=True)
    return _orig(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)


# 注入一个 est 帮助函数（与 _sim_scenario 相同口径）
def _est_scenario_energy(self, x, space, env, inst_idx):
    from run_exp_energy_c0 import c0_marginal_energy
    tc = int(space.tc[x])
    return c0_marginal_energy(tc, float(space.demand[x]), self.cooling_share, self.contract)


SaaReplanner._est_scenario_energy = _est_scenario_energy
SaaReplanner._saa_decide = patched

contract = make_c0_contract_v2()
hist = build_history(generate_dataset(200, 200, 20260925))
ds = add_v2_initial_quality(generate_dataset(1, 200, 20260926), contract)
sampler = CondHistoricalSampler(hist, n_neighbors=10)
rp = SaaReplanner(budget=710.6548152249378, capacity=50.0,
                  booking_horizon=BOOKING_HORIZON, contract=contract,
                  cooling_share=3.7208946347769882, sampler=sampler,
                  reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, K=10,
                  time_limit=10.0, arm_seed=7002)
env = StrictOnlineEnv(ds, capacity=50.0, num_vehicles=15,
                      tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                      coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
traces, served = env.run(0)
print('accepted:', len(rp._accepted), 'rejected:', len(rp._rejected))
