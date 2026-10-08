# -*- coding: utf-8 -*-
"""S3-4 微基准（只读诊断，非正式判据）：同一随机影子状态/同一操作序列，
ls 旧实现 vs incr_eval 增量预筛的耗时对比。"""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, 'coldchain'), os.path.join(_ROOT, 'simulation'),
           os.path.join(_ROOT, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_saa import (SaaReplanner, UncondHistoricalSampler, ScenarioOrder,
                          build_history, make_c0_contract_v2)
from run_exp_reserve import generate_dataset, BOOKING_HORIZON
from coldchain_evaluator_a1 import REV


def main():
    rng = np.random.default_rng(777)
    contract = make_c0_contract_v2()
    sampler = UncondHistoricalSampler(build_history(generate_dataset(5, 30, 20260925)))
    n_real = 60
    coords = rng.uniform(0.1, 0.9, (1, n_real, 2)).astype(np.float32)
    coords[0, 0] = [0.5, 0.5]
    demands = rng.uniform(1, 12, (1, n_real)).astype(np.float32)
    demands[0, 0] = 0
    reveal = np.full((1, n_real), 1e6, np.float32)
    reveal[0, 1:] = rng.uniform(0, 8, n_real - 1)
    tw_end = np.minimum(22.0, reveal + rng.uniform(0.3, 4.0, (1, n_real))).astype(np.float32)
    tw_end[0, 0] = 22.0
    tw_start = np.zeros((1, n_real), np.float32)
    service = np.full((1, n_real), 0.05, np.float32)
    service[0, 0] = 0
    tc = rng.integers(0, 3, (1, n_real)).astype(np.int32)
    ds = {'coords': coords, 'demands': demands, 'tw_start': tw_start, 'tw_end': tw_end,
          'service_time': service, 'reveal_time': reveal, 'temp_class': tc}

    class Env:
        tw_speed = 30.0 / 1.0

        def __init__(self, d):
            self.tw_end = d['tw_end']
            self.coords = d['coords']
            self.dataset = d
            self.demands = d['demands']
            self.tw_start = d['tw_start']
            self.service_time = d['service_time']
            self.reveal_time = d['reveal_time']

    env = Env(ds)
    fut = [ScenarioOrder(oid=-(j + 1), reveal=float(rng.uniform(0, 10)),
                         x=float(rng.uniform(0.1, 0.9)), y=float(rng.uniform(0.1, 0.9)),
                         demand=float(rng.uniform(1, 12)), temp_class=int(rng.integers(0, 3)),
                         tw_start=0.0, tw_end=float(rng.uniform(10, 22)),
                         service_time=0.05) for j in range(40)]
    import copy
    pool = list(range(1, n_real))
    rng.shuffle(pool)
    st_base = {}
    for vid in range(6):
        k = int(rng.integers(1, 5))
        st_base[vid] = dict(cur=0, cur_time=0.0, load=0.0,
                            route=list(pool[vid * 4:vid * 4 + k]), cc=None, frozen=0)
    st_base[6] = dict(cur=int(pool[24]), cur_time=float(rng.uniform(0, 2)),
                      load=float(rng.uniform(0, 8)), route=list(pool[25:27]),
                      cc=None, frozen=1)
    res = {}
    for incr in (False, True):
        rp = SaaReplanner(budget=1e9, capacity=50.0, booking_horizon=BOOKING_HORIZON,
                          contract=contract, cooling_share=2.0, sampler=sampler,
                          reject_penalty={0: 5.0, 1: 10.0, 2: 15.0}, K=10,
                          time_limit=60.0, arm_seed=7001, shadow_mode='ls',
                          incr_eval=incr)
        space, _ = rp._build_space(env, 0, [fut])
        deadline = float(tw_end[0, 0])
        st = copy.deepcopy(st_base)
        t0 = time.perf_counter()
        for j in range(len(fut)):
            o = n_real + j
            rp._insert_u(space, env, o, st, deadline)
            rp._local_search_u(space, env, st, deadline)
        dt = time.perf_counter() - t0
        key = 'incr' if incr else 'old'
        res[key] = dict(seconds=round(dt, 3),
                        final_routes=[list(s['route']) for s in st.values()])
    assert res['old']['final_routes'] == res['incr']['final_routes'], 'routes diverged'
    speedup = res['old']['seconds'] / max(res['incr']['seconds'], 1e-9)
    print('BENCH old=%.3fs incr=%.3fs speedup=%.1fx (routes identical)'
          % (res['old']['seconds'], res['incr']['seconds'], speedup))


if __name__ == '__main__':
    main()
