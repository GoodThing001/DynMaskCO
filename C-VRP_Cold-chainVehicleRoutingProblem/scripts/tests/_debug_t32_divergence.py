# -*- coding: utf-8 -*-
"""调试：复现 T32 trial 20 的首个分歧点（v2 vs old ls 的哪个动作先分叉）。"""
import copy
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'tests'))
for _p in (_ROOT, os.path.join(_ROOT, 'coldchain'), os.path.join(_ROOT, 'simulation'),
           os.path.join(_ROOT, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_a1_saa_correctness as T
from scenario_saa import ScenarioOrder

rng = np.random.default_rng(20261002)
rp_off, _ = T._mk_replanner(shadow_mode='ls', incr_eval=False)
rp_on, _ = T._mk_replanner(shadow_mode='ls', incr_eval=True)

# 复现 T32 的 rng 消耗序列直至 trial 20（trial 0..19 各消耗：_mk_random_day + fut + st + 30×(2 次 rng.integers)）
for trial in range(21):
    n_real = 30
    tight = (trial % 2 == 1)
    ds = T._mk_random_day.__wrapped__ if False else None
    # 直接复制 T32 内的生成代码（rng 序列完全一致）
    n = 1 + (n_real - 1)
    coords = rng.uniform(0.2, 0.8, size=(1, n, 2)).astype(np.float32)
    coords[0, 0] = [0.5, 0.5]
    demands = rng.uniform(1.0, 12.0, size=(1, n)).astype(np.float32)
    demands[0, 0] = 0.0
    tw_end = rng.uniform(12.0, 22.0, size=(1, n)).astype(np.float32)
    tw_end[0, 0] = 22.0
    service = np.full((1, n), 0.05, np.float32)
    service[0, 0] = 0.0
    reveal = np.full((1, n), 1e6, np.float32)
    reveal[0, 1:] = rng.uniform(0.0, 6.0, n_real - 1)
    if tight:
        width = rng.uniform(0.3, 2.5, n_real - 1)
        tw_end[0, 1:] = np.minimum(22.0, reveal[0, 1:] + width).astype(np.float32)
    tc = rng.integers(0, 3, size=(1, n)).astype(np.int32)
    tw_start = np.zeros((1, n), np.float32)
    ds = {'coords': coords, 'demands': demands, 'tw_start': tw_start,
          'tw_end': tw_end, 'service_time': service,
          'reveal_time': reveal, 'temp_class': tc}
    env = T._FakeEnv(ds)
    fut = [ScenarioOrder(oid=-(j + 1), reveal=float(rng.uniform(2, 10)),
                         x=float(rng.uniform(0.2, 0.8)), y=float(rng.uniform(0.2, 0.8)),
                         demand=float(rng.uniform(1, 6)), temp_class=int(rng.integers(0, 3)),
                         tw_start=0.0, tw_end=float(rng.uniform(14, 22)),
                         service_time=0.05) for j in range(12)]
    space, _scen = rp_off._build_space(env, 0, [fut])
    deadline = float(env.tw_end[0, 0])
    pool = list(range(1, n_real))
    rng.shuffle(pool)
    st = {}
    for vid in range(4):
        k = int(rng.integers(0, 4))
        st[vid] = dict(cur=0, cur_time=0.0, load=0.0,
                       route=list(pool[vid * 3:vid * 3 + k]), cc=None, frozen=0)
    st[4] = dict(cur=int(pool[12]), cur_time=float(rng.uniform(0, 2)),
                 load=float(rng.uniform(0, 8)), route=list(pool[13:15]),
                 cc=None, frozen=1)
    for _ in range(30):   # oracle 循环的 rng 消耗
        vid = int(rng.integers(0, len(st)))
        s = st[vid]
        rp_on._rebuild_summary(space, env, s, deadline)
        o = n_real + int(rng.integers(0, len(fut)))
        pos = int(rng.integers(s.get('frozen', 0), len(s['route']) + 1))
        rp_off._route_feasible_u(space, env, s['cur'], s['cur_time'],
                                 s['load'], s['route'][:pos] + [o] + s['route'][pos:], deadline)
        rp_on._precheck_ins(space, env, s, o, pos, deadline)
        if s['route']:
            i = int(rng.integers(s.get('frozen', 0), len(s['route'])))
            o_new = n_real + int(rng.integers(0, len(fut)))
            rep = list(s['route'])
            rep[i] = o_new
            rp_off._route_feasible_u(space, env, s['cur'], s['cur_time'], s['load'], rep, deadline)
            rp_on._precheck_replace(space, env, s, o_new, i, deadline)

    if trial != 20:
        continue
    st_a = copy.deepcopy(st)
    st_b = copy.deepcopy(st)
    for j in range(len(fut)):
        o = n_real + j
        ok_a = rp_off._insert_u(space, env, o, st_a, deadline)
        ok_b = rp_on._insert_u(space, env, o, st_b, deadline)
        print('j=%d ins ok %s/%s' % (j, ok_a, ok_b))
        for vid in st:
            if st_a[vid]['route'] != st_b[vid]['route']:
                print('  DIVERGE after insert j=%d vid=%d' % (j, vid))
                print('   old:', st_a[vid]['route'])
                print('   new:', st_b[vid]['route'])
                sys.exit(0)
        rp_off._local_search_u(space, env, st_a, deadline)
        rp_on._local_search_u(space, env, st_b, deadline)
        for vid in st:
            if st_a[vid]['route'] != st_b[vid]['route']:
                print('  DIVERGE after ls j=%d vid=%d' % (j, vid))
                print('   old:', st_a[vid]['route'])
                print('   new:', st_b[vid]['route'])
                sys.exit(0)
    print('no divergence in trial 20 replay')
