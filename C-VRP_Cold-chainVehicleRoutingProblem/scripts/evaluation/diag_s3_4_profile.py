# -*- coding: utf-8 -*-
"""S3-4 决策耗时分解诊断（只读）：ls+incr 口径下把单日每决策耗时按
采样器 / _insert_u / _local_search_u / _complete_u / 其余 分解，定位 85 分钟/天
的构成。用 time_limit=2s 压缩运行（比例可读，绝对值×5≈10s 口径）。"""
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
from scenario_saa import (SaaReplanner, CondHistoricalSampler, build_history,
                          build_history_times_classes, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}


def main():
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, 20260925)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, 15, 20260925,
                                                   0.6, hist_tc, k=10)
    gate_ds = add_v2_initial_quality(generate_dataset(1, 200, 20260926), contract)
    rp = SaaReplanner(budget=B, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=cooling_share,
                      sampler=CondHistoricalSampler(hist, n_neighbors=10),
                      reject_penalty=P_C, K=10, time_limit=2.0, arm_seed=7002,
                      shadow_mode='ls', energy_pricing='marginal',
                      future_policy='reveal', incr_eval=True)
    acc = {'ins': 0.0, 'ls': 0.0, 'comp': 0.0, 'sim': 0.0}
    _ins = rp._insert_u
    _ls = rp._local_search_u
    _comp = rp._complete_u
    _sim = rp._sim_scenario

    def t_ins(*a, **k):
        t = time.perf_counter()
        r = _ins(*a, **k)
        acc['ins'] += time.perf_counter() - t
        return r

    def t_ls(*a, **k):
        t = time.perf_counter()
        r = _ls(*a, **k)
        acc['ls'] += time.perf_counter() - t
        return r

    def t_comp(*a, **k):
        t = time.perf_counter()
        r = _comp(*a, **k)
        acc['comp'] += time.perf_counter() - t
        return r

    def t_sim(*a, **k):
        t = time.perf_counter()
        r = _sim(*a, **k)
        acc['sim'] += time.perf_counter() - t
        return r

    rp._insert_u = t_ins
    rp._local_search_u = t_ls
    rp._complete_u = t_comp
    rp._sim_scenario = t_sim
    env = StrictOnlineEnv(gate_ds, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    t0 = time.perf_counter()
    env.run(0)
    wall = time.perf_counter() - t0
    print('wall=%.1fs timeouts=%d accepted=%d rejected=%d' % (
        wall, rp.timeouts, len(rp._accepted), len(rp._rejected)))
    print('insert_u=%.1fs  local_search=%.1fs  complete_u=%.1fs  sim_scenario_total=%.1fs'
          % (acc['ins'], acc['ls'], acc['comp'], acc['sim']))


if __name__ == '__main__':
    main()
