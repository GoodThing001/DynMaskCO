# -*- coding: utf-8 -*-
"""本地 1 天计时诊断：Q 修订后 reveal 臂（marginal/K=10/10s）无负载下单日耗时，
与服务器高负载下 ~13-14 分钟/天对比，判断慢因构成（Q-02 车队扩大 vs 负载）。"""
import os
import sys
import time

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history,
                          build_history_times_classes, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}
ARM_SEEDS = {"uncond_hist": 7001, "cond_hist": 7002, "explicit_feat": 7003}


def main():
    contract = make_c0_contract_v2()
    t0 = time.time()
    train_ds = generate_dataset(200, 200, 20260925)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, 20260925,
                                                   0.6, hist_tc, k=10)
    print('budget setup %.1fs (B=%.1f, share=%.3f)' % (time.time() - t0, B, cooling_share))
    gate_ds = add_v2_initial_quality(generate_dataset(1, 200, 20260926), contract)
    for name, sampler in [("cond_hist", CondHistoricalSampler(hist, n_neighbors=10)),
                          ("uncond_hist", UncondHistoricalSampler(hist)),
                          ("explicit_feat", ExplicitFeatureSampler(hist))]:
        rp = SaaReplanner(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                          contract=contract, cooling_share=cooling_share, sampler=sampler,
                          reject_penalty=P_C, K=10, time_limit=10.0,
                          arm_seed=ARM_SEEDS[name], shadow_mode='greedy',
                          energy_pricing='marginal', future_policy='reveal')
        env = StrictOnlineEnv(gate_ds, capacity=50.0, num_vehicles=15,
                              tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                              coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
        t1 = time.time()
        env.run(0)
        dt = time.time() - t1
        print('%s: day0 %.1fs, timeouts=%d, accepted=%d rejected=%d'
              % (name, dt, rp.timeouts, len(rp._accepted), len(rp._rejected)))


if __name__ == '__main__':
    main()
