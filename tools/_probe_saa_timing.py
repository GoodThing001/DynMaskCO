# -*- coding: utf-8 -*-
"""A-v1 步骤 2 计时探针：1 个门判定日，K=10，量单天耗时/决策耗时/超时数。"""
import os
import sys
import time

import numpy as np

_SCRIPTS = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\scripts"
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history, make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

P_C = {0: 5.0, 1: 10.0, 2: 15.0}

contract = make_c0_contract_v2()
t0 = time.time()
hist = build_history(generate_dataset(200, 200, 20260925))
print("history built:", round(time.time() - t0, 1), "s", flush=True)
gate = add_v2_initial_quality(generate_dataset(3, 200, 20260926), contract)

for name, sampler in [
        ("uncond_hist", UncondHistoricalSampler(hist)),
        ("cond_hist", CondHistoricalSampler(hist)),
        ("explicit_feat", ExplicitFeatureSampler(hist))]:
    rp = SaaReplanner(budget=709.0, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                      contract=contract, cooling_share=3.71, sampler=sampler,
                      reject_penalty=P_C, K=10, time_limit=10.0, arm_seed=7001)
    env = StrictOnlineEnv(gate, capacity=50.0, num_vehicles=15, tw_speed=SPEED_KMH / KM_PER_UNIT,
                          replanner=rp, coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    t0 = time.time()
    traces, _ = env.run(0)
    dt = time.time() - t0
    print(f"[{name}] one day: {dt:.1f}s | decisions-timeouts: {rp.timeouts} | "
          f"accepted: {len(rp._accepted)} rejected: {len(rp._rejected)}", flush=True)
