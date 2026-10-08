# -*- coding: utf-8 -*-
"""影子超预算诊断（核查 §4.5）：记录每场景 shadow_energy/B、超额量、接受数、acc−rej 超额差、
真不可行（energy=None）数，以及真实接单侧的拒绝原因分布。"""
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
                          build_history_times_classes, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

TRAIN_SEED = 20260925
DEV_SEED = 20260926
P_C = {0: 5.0, 1: 10.0, 2: 15.0}


class BudgetSpy(SaaReplanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log = []

    def _complete_u(self, space, env, inst_idx, clock, st, wait_until):
        e, d = super()._complete_u(space, env, inst_idx, clock, st, wait_until)
        self.log.append(dict(energy=e, dist=d,
                             n_orders=sum(len(s['route']) for s in st.values()),
                             n_veh=len(st)))
        return e, d


def main():
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, TRAIN_SEED,
                                                   0.6, hist_tc, k=10)
    dev_ds = add_v2_initial_quality(generate_dataset(60, 200, DEV_SEED), contract)
    sub = {k: v[0:1] for k, v in dev_ds.items()}
    rp = BudgetSpy(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON, contract=contract,
                   cooling_share=cooling_share, sampler=UncondHistoricalSampler(hist),
                   reject_penalty=P_C, K=10, time_limit=10.0, arm_seed=7001,
                   energy_pricing='marginal')
    env = StrictOnlineEnv(sub, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    env.run(0)
    log = rp.log
    en = np.array([x['energy'] for x in log if x['energy'] is not None], float)
    n_none = sum(1 for x in log if x['energy'] is None)
    over = en[en > B + 1e-9]
    print(f"B={B:.1f} | shadow completions={len(log)} (acc+rej 各半) | energy=None={n_none} "
          f"({n_none/len(log):.1%})")
    print(f"energy: mean={en.mean():.1f} p50={np.percentile(en,50):.1f} "
          f"p90={np.percentile(en,90):.1f} max={en.max():.1f} | over_budget={len(over)} "
          f"({len(over)/max(len(en),1):.1%})")
    if len(over):
        print(f"excess: mean={float((over-B).mean()):.1f} p50={np.percentile(over-B,50):.1f} "
              f"p90={np.percentile(over-B,90):.1f} max={float((over-B).max()):.1f}")
    n_ord = np.array([x['n_orders'] for x in log], float)
    print(f"shadow packed orders: mean={n_ord.mean():.1f} max={n_ord.max():.0f}")
    print(f"decisions: accepted={len(rp._accepted)} rejected={len(rp._rejected)} "
          f"timeouts={rp.timeouts}")
    print("done")


if __name__ == "__main__":
    main()
