#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读探针：把**两种预先存在**的 bootstrap 定义在**全部配对块**上分别算出并对照落盘值。

用户 2026-10-14 要求（P1-D 段）：
  「**只读**检查原生产者的 bootstrap 实现/随机数初始化/采样顺序/分位数算法，
   若确有两种预先固定的 bootstrap 定义则**分别记录**，**不得事后选择更有利的区间**。」

源码事实（只读核对，`run_exp_reserve.py:353-362`）：

    def _stat(x, seed):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        n = len(x)
        r = np.random.default_rng(seed + 777)                       # 随机数初始化
        mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])  # 2000 次**顺序**抽样
        return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(mm, 2.5)),
                "ci_hi": float(np.percentile(mm, 97.5)), "n": int(n)}        # method='linear'

两种定义**共用同一个 `_stat`**（同初始化、同次数、同分位数算法），差别**只在输入向量顺序**
（`r.integers(0, n, n)` 按**位置**抽 ⇒ 顺序敏感）：

  · **定义 P（生产者顺序，规范）**：`run_a1_step2_gate.paired_all_days` 用 `for r in a_rows`
    （`:207`）取对 ⇒ 差向量 = `a_rows` 原始顺序。**这是落盘 `mean/ci_*` 的真实来源。**
  · **定义 S（day_id 升序）**：`collect_compare_table._pair` 与
    `diag_a3_adjudicate._stat_exact`（`range(40)`）按 day_id 排序后取对。

⇒ 均值对顺序**不变**；CI 对顺序**敏感**。本探针在**所有**配对块上同时输出 P 与 S，
  逐块判定落盘值命中哪一个，并汇总。**只读**：不改任何落盘值、不替换任何已发布 CI。

用法（扩展根下）：
    python ../CC_Compare/tools/_probe_bootstrap_defs.py [--data <dataset.json>]
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WS = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'scripts', 'evaluation'))

import check_pairing_identity as cpi                          # noqa: E402
from run_exp_reserve import _stat                              # noqa: E402

# ⚠️ **不能**自己再 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`：
# `check_pairing_identity` 被 import 时**已经**绑了一个 wrapper 到同一个 buffer，
# 两个 wrapper 套同一 buffer，先建的那个被 GC 时会**关掉底层 buffer**
# ⇒ 之后 print 报 `ValueError: I/O operation on closed file`（2026-10-14 实测再次踩到）。
# `reconfigure` 就地改编码，不新建 wrapper。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                              # noqa: BLE001
    pass


def _diffs(pd_, arm, base):
    ra, rb = pd_.get(arm) or [], pd_.get(base) or []
    b_by_i = {r['i']: r['utility'] for r in rb}
    P = [r['utility'] - b_by_i[r['i']] for r in ra]
    S = [u for _i, u in sorted((r['i'], r['utility'] - b_by_i[r['i']]) for r in ra)]
    return P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='CC_Compare/results/a1_compare_table/dataset.json')
    a = ap.parse_args()
    dp = a.data if os.path.isabs(a.data) else os.path.join(ROOT, *a.data.split('/'))
    data = json.load(open(dp, encoding='utf-8'))

    n = same_order = 0
    hitP = hitS = hitNeither = 0
    differ, unmatched = [], []
    max_ci_gap = 0.0
    for gate, pc, arm, d, _i in cpi._pairs(data):
        gp = gate if os.path.isabs(gate) else os.path.join(WS, *str(gate).split('/'))
        pd_, gseed, src = cpi._per_day_of(gp, pc)
        if pd_ is None or not isinstance(gseed, int):
            continue
        P, S = _diffs(pd_, arm, 'cond_hist')
        if len(P) != 40:
            continue
        n += 1
        stP, stS = _stat(np.array(P, float), gseed), _stat(np.array(S, float), gseed)
        gap = max(abs(stP['ci_lo'] - stS['ci_lo']), abs(stP['ci_hi'] - stS['ci_hi']))
        max_ci_gap = max(max_ci_gap, gap)
        if P == S:
            same_order += 1
        else:
            differ.append((gate, pc, arm, gseed, stP, stS, gap, d))
        mP = abs(stP['mean'] - d['mean']) < 1e-6 and abs(stP['ci_lo'] - d['ci_lo']) < 1e-6
        mS = abs(stS['mean'] - d['mean']) < 1e-6 and abs(stS['ci_lo'] - d['ci_lo']) < 1e-6
        if mP and not mS:
            hitP += 1
        elif mS and not mP:
            hitS += 1
        elif mP and mS:
            hitP += 1
            hitS += 1
        else:
            hitNeither += 1
            unmatched.append((gate, pc, arm, gseed, stP, stS, d))

    print('=== 两种 bootstrap 定义在全部配对块上的对照（只读）===')
    print('可判定配对块 = %d' % n)
    print('  顺序相同（P == S）           = %d' % same_order)
    print('  顺序不同（P != S）           = %d' % len(differ))
    print('  CI 下界/上界最大差（P vs S）  = %.6f' % max_ci_gap)
    print('  落盘值命中定义 P（生产者序）  = %d' % hitP)
    print('  落盘值命中定义 S（day_id 序） = %d' % hitS)
    print('  两者皆不命中                 = %d' % hitNeither)
    if differ:
        print('--- 顺序**不同**的块（P 与 S 的 CI 并列记录；**不择优**）---')
        for gate, pc, arm, gs, stP, stS, gap, d in differ[:20]:
            print('  %s %s %s seed=%d' % (os.path.basename(str(gate).split('/')[-2]),
                                           pc, arm, gs))
            print('    落盘            [%.6f, %.6f]' % (d['ci_lo'], d['ci_hi']))
            print('    P(生产者序)      [%.6f, %.6f]' % (stP['ci_lo'], stP['ci_hi']))
            print('    S(day_id 序)     [%.6f, %.6f]   差=%.6f' % (stS['ci_lo'], stS['ci_hi'], gap))
    if unmatched:
        print('--- **两者皆不命中**的块（须在报告中如实列出）---')
        for gate, pc, arm, gs, stP, stS, d in unmatched[:20]:
            print('  %s %s %s seed=%d' % (os.path.basename(str(gate).split('/')[-2]),
                                           pc, arm, gs))
            print('    落盘 [%.6f, %.6f]  P [%.6f, %.6f]  S [%.6f, %.6f]'
                  % (d['ci_lo'], d['ci_hi'], stP['ci_lo'], stP['ci_hi'],
                     stS['ci_lo'], stS['ci_hi']))
    print('\n结论：两种定义**均已预先固定在代码中**；上表并列记录，**未**据此替换任何落盘 CI。')


if __name__ == '__main__':
    main()
