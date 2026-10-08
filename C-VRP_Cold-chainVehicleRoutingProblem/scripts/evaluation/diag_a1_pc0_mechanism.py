# -*- coding: utf-8 -*-
"""L2c p_c=0 零接单机制的逐决策诊断：每决策 acc/rej 两侧场景和、真不可行(−1e6)数、margin。"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, build_history,
                          build_history_times_classes, make_c0_contract_v2,
                          INFEASIBLE_SCENARIO_PENALTY)
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926
P_C0 = {0: 0.0, 1: 0.0, 2: 0.0}


class Spy(SaaReplanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.decisions = []

    def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
        rec = dict(o=int(o), clock=float(clock), acc_sum=0.0, rej_sum=0.0,
                   n_infeas_acc=0, n_infeas_rej=0, n_scen=len(scenarios))
        orig = self._sim_scenario
        call_i = [0]

        def wrapped(env2, i2, clk, st, scen_idx, space):
            u = orig(env2, i2, clk, st, scen_idx, space)
            if call_i[0] < len(scenarios):
                rec['acc_sum'] += u
                rec['n_infeas_acc'] += (u == INFEASIBLE_SCENARIO_PENALTY)
            else:
                rec['rej_sum'] += u
                rec['n_infeas_rej'] += (u == INFEASIBLE_SCENARIO_PENALTY)
            call_i[0] += 1
            return u

        self._sim_scenario = wrapped
        try:
            super()._saa_decide(env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)
        finally:
            self._sim_scenario = orig
        rec['decision'] = 'accept' if o in self._accepted else 'reject'
        rec['margin'] = rec['acc_sum'] - rec['rej_sum']
        self.decisions.append(rec)


def main():
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, TRAIN_SEED,
                                                   0.6, hist_tc, k=10)
    dev_ds = add_v2_initial_quality(generate_dataset(60, 200, DEV_SEED), contract)
    sub = {k: v[0:1] for k, v in dev_ds.items()}
    rp = Spy(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON, contract=contract,
             cooling_share=cooling_share, sampler=UncondHistoricalSampler(hist),
             reject_penalty=P_C0, K=10, time_limit=10.0, arm_seed=7001,
             energy_pricing='marginal', future_policy='density')
    env = StrictOnlineEnv(sub, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.run(0)
    ds = rp.decisions
    print(f"B={B:.1f} cooling_share={cooling_share:.2f} n_decisions={len(ds)} "
          f"accepted={sum(1 for d in ds if d['decision']=='accept')}")
    for d in ds[:12]:
        print(f"  o={d['o']:3d} t={d['clock']:5.2f} {d['decision']:6s} "
              f"acc_sum={d['acc_sum']:12.1f} rej_sum={d['rej_sum']:12.1f} "
              f"margin={d['margin']:10.1f} infeas(acc/rej)={d['n_infeas_acc']}/{d['n_infeas_rej']}")
    # 统计
    acc_i = [d for d in ds if d['decision'] == 'accept']
    rej_i = [d for d in ds if d['decision'] == 'reject']
    print("accept margins: mean %.1f (n=%d)" % (np.mean([d['margin'] for d in acc_i]), len(acc_i)))
    print("reject margins: mean %.1f (n=%d)" % (np.mean([d['margin'] for d in rej_i]), len(rej_i)))
    print("rej decisions with infeas_acc>0:", sum(1 for d in rej_i if d['n_infeas_acc'] > 0),
          "/", len(rej_i))
    print("rej decisions with infeas_rej>0:", sum(1 for d in rej_i if d['n_infeas_rej'] > 0))
    print("acc decisions with infeas_acc>0:", sum(1 for d in acc_i if d['n_infeas_acc'] > 0))


if __name__ == "__main__":
    main()
