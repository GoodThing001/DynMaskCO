"""A-v1 下游消融 2×2 汇总（router × vote）——从各 gate.json 读取逐日行，输出矩阵表。

用法（结果齐全后）：
  python scripts/evaluation/_analyze_ablation.py --out results/a1_external_dev/ablation_summary.json
臂来源：
  greedy_novote      abl_greedy_novote/gate.json
  greedy_vote_cond   abl_greedy_vote/gate.json（应与步骤2 cond_hist 逐位一致）
  ortools_novote     ortools_v2/gate.json（= ortools_accept）
  ortools_vote_cond  abl_ot_vote/gate.json
  pyvrp_novote       pyvrp_v2/gate.json（= pyvrp_accept）
  pyvrp_vote_cond    abl_pv_vote/gate.json
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.normpath(os.path.join(_HERE, '..', '..', 'results', 'a1_external_dev'))

ARM_FILES = {
    'greedy_novote': 'abl_greedy_novote/gate.json',
    'greedy_vote_cond': 'abl_greedy_vote_cond_v2/gate.json',
    'ortools_novote': 'ortools_v2/gate.json',
    'ortools_vote_cond': 'abl_ortools_vote_cond_v2/gate.json',
    'pyvrp_novote': 'pyvrp_v2/gate.json',
    'pyvrp_vote_cond': 'abl_pyvrp_vote_cond_v2/gate.json',
}


def load_rows(path, pname):
    d = json.load(open(os.path.join(_RES, path), encoding='utf-8'))
    r = d['results']['20260926'][pname]
    key = [k for k in r['arms'] if k not in ('cond_hist', 'uncond_hist', 'explicit_feat')][0]
    rows = sorted(r['per_day'], key=lambda x: x['i'])
    return key, rows, r['arms'][key]


def _stat(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {'mean': None, 'ci_lo': None, 'ci_hi': None, 'n': 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        means.append(x[idx].mean())
    means = np.array(means)
    return {'mean': float(x.mean()), 'ci_lo': float(np.percentile(means, 2.5)),
            'ci_hi': float(np.percentile(means, 97.5)), 'n': n}


def paired(a_rows, b_rows, seed):
    b_by_i = {r['i']: r['utility'] for r in b_rows}
    diff = [r['utility'] - b_by_i[r['i']] for r in a_rows
            if r['i'] in b_by_i and r['utility'] is not None and b_by_i[r['i']] is not None]
    return _stat(diff, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    summary = {}
    for pname in ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']:
        arms = {}
        rows_by_arm = {}
        for arm, path in ARM_FILES.items():
            try:
                key, rows, summ = load_rows(path, pname)
            except Exception as e:  # noqa: BLE001
                print('skip %s (%s): %s' % (arm, pname, e))
                continue
            arms[arm] = dict(
                utility=_stat([r['utility'] for r in rows], 20260926),
                served_mean=float(np.mean([r['n_served'] for r in rows])),
                rejected_mean=float(np.mean([r['n_rejected'] for r in rows])),
                energy_mean=float(np.mean([r['energy_kwh'] for r in rows])),
                hard_feasible=float(summ['hard_feasible_rate']),
                timeouts=float(summ['mean_timeouts']))
            rows_by_arm[arm] = rows
        # 配对差矩阵
        pdiff = {}
        names = sorted(rows_by_arm)
        for a in names:
            for b in names:
                if a != b:
                    pdiff['%s_minus_%s' % (a, b)] = paired(
                        rows_by_arm[a], rows_by_arm[b], 20260926)
        # 2×2 关键配对
        key_pairs = ['greedy_novote_minus_greedy_vote_cond',
                     'ortools_novote_minus_ortools_vote_cond',
                     'pyvrp_novote_minus_pyvrp_vote_cond',
                     'ortools_novote_minus_greedy_novote',
                     'pyvrp_novote_minus_greedy_novote',
                     'greedy_novote_minus_ortools_vote_cond']
        summary[pname] = {'arms': arms, 'paired': {k: pdiff[k] for k in key_pairs},
                          'paired_all': pdiff}
    json.dump(summary, open(args.out, 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
