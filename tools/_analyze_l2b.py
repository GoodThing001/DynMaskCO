# -*- coding: utf-8 -*-
"""修正口径对比分析（核查 §4.1 修订版）。

- 显式输入两个 gate.json；seed 身份一致才允许按 day_id 配对（跨 seed 拒绝）；
- 统一 bootstrap 随机流（seed 固定）按天配对；
- 跨批次（不同实现）差值只标「混合改动诊断」；同批次内 cond−uncond 才是可裁决比较；
- 输出 served/效用/分解 + 可裁决标记（A-03 语义）。
"""
import argparse
import json
import sys

import numpy as np

BOOT_SEED = 20260928


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"MISSING: {path}")
        sys.exit(1)


def paired(a_rows, b_rows):
    """按 day_id 显式配对；缺失日显式记录；day-clustered bootstrap（统一随机流）。"""
    b_by_i = {r['i']: r['utility'] for r in b_rows}
    pa, pb, missing = [], [], []
    for r in a_rows:
        if r['i'] not in b_by_i:
            missing.append(('b_missing_day', int(r['i'])))
            continue
        if r['utility'] is None or b_by_i[r['i']] is None:
            missing.append(('missing_utility', int(r['i'])))
            continue
        pa.append(r['utility'])
        pb.append(b_by_i[r['i']])
    if not pa:
        return dict(mean=None, ci_lo=None, ci_hi=None, n=0,
                    adjudicable=False, missing=missing)
    d = np.array(pa, float) - np.array(pb, float)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), n=n,
                n_paired=len(pa), adjudicable=(len(missing) == 0 and len(pa) == len(a_rows)),
                missing=missing)


def show_gate(d, tag):
    g = d.get('gate', d.get('results'))
    seeds = d.get('seeds', {})
    print(f"########## {tag} (seeds={seeds}) ##########")
    for pn in g:
        blk = g[pn]
        if not isinstance(blk, dict) or 'per_day' not in blk:
            continue
        print(f"=== {pn} ===")
        for arm in ('uncond_hist', 'cond_hist', 'explicit_feat'):
            if arm not in blk['per_day']:
                continue
            rows = blk['per_day'][arm]
            s = np.mean([r['n_served'] for r in rows if r.get('n_served') is not None])
            u = np.array([r['utility'] for r in rows if r.get('utility') is not None], float)
            arm_sum = blk.get('arms', {}).get(arm, {})
            print(f"  {arm}: served {s:.1f} | utility {u.mean():.1f} | "
                  f"adjudicable={arm_sum.get('adjudicable')} | "
                  f"timeouts {arm_sum.get('mean_timeouts')} | feas {arm_sum.get('hard_feasible_rate')}")
        pd = blk['per_day']
        if 'cond_hist' in pd and 'uncond_hist' in pd:
            st = paired(pd['cond_hist'], pd['uncond_hist'])
            print(f"  [same-batch] cond−uncond: {st['mean']:.1f} [{st['ci_lo']:.1f},{st['ci_hi']:.1f}] "
                  f"adjudicable={st['adjudicable']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='gate.json A')
    ap.add_argument('--b', required=True, help='gate.json B（跨批次差值仅作混合改动诊断）')
    args = ap.parse_args()
    da, db = load(args.a), load(args.b)
    sa = int(da.get('seeds', {}).get('gate', -1))
    sb = int(db.get('seeds', {}).get('gate', -1))
    show_gate(da, args.a)
    show_gate(db, args.b)
    if sa != -1 and sa != sb:
        print(f"[identity] seed mismatch ({sa} vs {sb}) → 跨批次配对拒绝（A-04）")
        return
    print("########## 跨批次（仅混合改动诊断，不可归因单一杠杆） ##########")
    ga, gb = da.get('gate', da.get('results')), db.get('gate', db.get('results'))
    for pn in ga:
        if pn not in gb or 'per_day' not in ga[pn] or 'per_day' not in gb[pn]:
            continue
        for arm in ('uncond_hist', 'cond_hist', 'explicit_feat'):
            if arm not in ga[pn]['per_day'] or arm not in gb[pn]['per_day']:
                continue
            st = paired(ga[pn]['per_day'][arm], gb[pn]['per_day'][arm])
            print(f"  {pn} {arm} (A−B): {st['mean']:.1f} [{st['ci_lo']:.1f},{st['ci_hi']:.1f}] "
                  f"adjudicable={st['adjudicable']}")


if __name__ == '__main__':
    main()
