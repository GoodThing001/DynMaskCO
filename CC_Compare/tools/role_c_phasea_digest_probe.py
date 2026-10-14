#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的**决策点身份探针**：在跑批机器上实测固定决策点集合的 digest。

为什么单独一个脚本（而不改 `role_c_timing.py`）
------------------------------------------------
`role_c_timing.py` 是封存裁定 §2.1 登记的 **15 个计时模块之一**（raw SHA 已登记）。
为了把 digest 变成"可核验的产物"，最省事的做法是给它加一个 `--digest-out`；
但那会**改掉它的字节** ⇒ 那份登记的源码身份不再描述未来的运行。
⇒ 本脚本**只读调用** `role_c_timing.build_real_work_fn`（不改任何被测/已登记模块），
把实测 digest 与**构建事实**一起写成产物，供 `run_phasea_gates.py` 的 G4 门消费。

⚠️ 本脚本**不产生任何耗时结论**（不做 profiling、不跑 A/B 类观测）。
它只说一件事：**这次测的，是不是事前固定那 8 个决策点**。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_timing as rt                                          # noqa: E402

#: G4 消费的**构建事实**白名单（键必须在产物里且值必须与冻结配置一致）
BUILD_FACT_KEYS = ('n_orders', 'n_vehicles', 'arm_seed', 'K', 'inner_gate',
                   'gate_seed', 'train_seed', 'n_contexts', 'inst_idx')

FROZEN_EXPECT = {'n_orders': 200, 'n_vehicles': 15, 'arm_seed': 7002, 'K': 10,
                 'inner_gate': 'on', 'gate_seed': 20260926, 'train_seed': 20260925,
                 'n_contexts': 8, 'inst_idx': 0}


def _sha_of_module(rel):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
    if not os.path.isfile(p):
        return None
    with io.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def probe(*, dataset_cache=None, **kw):
    """搭一次真实管线、取 digest，返回产物 dict（**不含任何效用/耗时结论**）。"""
    t0 = time.perf_counter()
    params = dict(FROZEN_EXPECT)
    params.pop('n_contexts')
    params.update(kw or {})
    built = rt.build_real_work_fn(dataset_cache=dataset_cache, **params)
    wf = built['work_fn']
    n_ctx = len(wf)
    art = {
        'label': 'P1A-RoleC-PhaseA-decision-point-identity-probe',
        'probed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'host': {'platform': sys.platform, 'python': sys.version.split()[0],
                 'cpu_count': os.cpu_count(),
                 'affinity': (sorted(os.sched_getaffinity(0))
                              if hasattr(os, 'sched_getaffinity') else None)},
        'thread_caps': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'decision_points_digest': wf.digest(),
        'n_contexts': int(n_ctx),
        'build_facts': {k: built.get(k) for k in BUILD_FACT_KEYS},
        'anchor_check': built.get('anchor_check'),
        'dataset_source': built.get('dataset_source'),
        'budget_B': built.get('budget'),
        'cooling_share': built.get('cooling_share'),
        'probe_wall_s': time.perf_counter() - t0,
        'source_sha256': {m: _sha_of_module(m) for m in
                          ('role_c_timing.py', 'role_c_harness.py',
                           'role_c_forward.py', 'role_c_local_search.py',
                           'role_c_identity.py', 'role_c_t_series.py',
                           'role_c_phasea_digest_probe.py')},
        'not_a_timing_conclusion': True,
        'note': ('本产物只说"这次测的是哪 8 个决策点"；**不含**任何耗时结论，'
                 '**不得**当作 A/B 类观测或在线可执行性证据。'),
    }
    return art


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Phase A 决策点身份探针（不产生耗时结论）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset-cache', default='/tmp/rolec_train_ds.npz')
    a = ap.parse_args(argv)
    art = probe(dataset_cache=a.dataset_cache)
    p = os.path.abspath(a.out)
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(art, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('WROTE: %s' % p)
    print(json.dumps({'decision_points_digest': art['decision_points_digest'],
                      'n_contexts': art['n_contexts'],
                      'build_facts': art['build_facts'],
                      'dataset_source': art['dataset_source'],
                      'anchor_check': art['anchor_check'],
                      'probe_wall_s': art['probe_wall_s']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
