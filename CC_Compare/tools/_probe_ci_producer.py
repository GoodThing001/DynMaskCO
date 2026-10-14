#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时探针（只读）：找出主模型配对块 CI 的**实际生产者参数**。

背景：主模型配对块的 `mean/ci_*` 有三个候选生产者：
  1. `run_a1_step2_gate.paired_all_days` —— 按 **a_rows 原始顺序**，用 `run_exp_reserve._stat`
  2. `diag_a3_adjudicate._stat_exact` —— 按 **range(40) 升序**，用自己的 `default_rng(seed+777)`
  3. `collect_compare_table._pair` —— 按 **day_id 排序**（本次复刻所用）
三者 bootstrap 都是按位置抽样 ⇒ 顺序/种子不同则 CI 不同、均值相同。

本探针在 (顺序 × 种子) 网格上搜索能**逐位复现**落盘 CI 的组合，用于把"未复现"
变成"已定位"或"确认无法复现"。**只读**，不改任何 gate。
"""
import glob
import json
import os
import sys
import io

import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EXT = r'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
DATA = r'/home/hzeng/project/MASKCO-Main/CC_Compare/results/a1_compare_table/dataset.json'
SEEDS = [20260926, 20260930, 20261001, 20260925, 20260927, 20260928, 777, 0, 1]


def stat(x, seed):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    r = np.random.default_rng(int(seed) + 777)
    mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
    return float(np.mean(x)), float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    want = []
    for r in d['main_model']:
        arm = r.get('arm')
        blk = (r.get('main_comparison') or {}).get('%s_minus_cond_hist' % arm)
        if blk and blk.get('mean') is not None:
            want.append((r['gate'], r['p_c'], arm, blk))
    print('候选配对块 %d 个' % len(want))
    hits = {}
    for gate, pc, arm, blk in want:
        gp = os.path.join(EXT, *gate.split('/'))
        if not os.path.isfile(gp):
            continue
        g = json.load(open(gp, encoding='utf-8'))
        blk_pd = None
        for _k, yb in (g.get('results') or {}).items():
            if isinstance(yb, dict) and isinstance(yb.get(pc), dict):
                blk_pd = yb[pc].get('per_day')
                break
        if not blk_pd:
            continue
        ra, rb = blk_pd.get(arm), blk_pd.get('cond_hist')
        if not ra or not rb:
            continue
        mb = {r['i']: r['utility'] for r in rb}
        order_a = [r['utility'] - mb[r['i']] for r in ra]
        ma = {r['i']: r['utility'] for r in ra}
        order_s = [ma[i] - mb[i] for i in sorted(ma)]
        found = None
        for name, arr in (('a_rows', order_a), ('sorted', order_s)):
            for s in SEEDS:
                m, lo, hi = stat(arr, s)
                if (abs(lo - blk['ci_lo']) < 1e-6 and abs(hi - blk['ci_hi']) < 1e-6):
                    found = (name, s)
                    break
            if found:
                break
        hits[(gate.split('/')[-2], pc, arm)] = found
    ok = sum(1 for v in hits.values() if v)
    print('逐位复现成功 %d / %d' % (ok, len(hits)))
    for k, v in sorted(hits.items()):
        print('  %-44s %-14s %-10s -> %s' % (k[0], k[1], k[2], v or '**未找到**'))


if __name__ == '__main__':
    main()
