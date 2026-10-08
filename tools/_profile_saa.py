# -*- coding: utf-8 -*-
"""Profile one SAA day (reduced size) to find the hot spots."""
import cProfile
import io
import os
import pstats
import sys

_SCRIPTS = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\scripts"
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, build_history,
                          make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

contract = make_c0_contract_v2()
hist = build_history(generate_dataset(30, 60, 20260925))
gate = add_v2_initial_quality(generate_dataset(1, 60, 20260926), contract)
rp = SaaReplanner(budget=700.0, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                  contract=contract, cooling_share=3.71,
                  sampler=UncondHistoricalSampler(hist),
                  reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, K=3,
                  time_limit=120.0, arm_seed=7001)
env = StrictOnlineEnv(gate, capacity=50.0, num_vehicles=8, tw_speed=SPEED_KMH / KM_PER_UNIT,
                      replanner=rp, coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)

pr = cProfile.Profile()
pr.enable()
traces, _ = env.run(0)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(28)
print(s.getvalue())
