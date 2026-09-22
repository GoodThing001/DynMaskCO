"""固定状态质量分析：读取 per_state.json，计算部署均值、实例级配对 bootstrap CI、
胜/平/负、失败率与认证失败率。

配对比较按实例聚类 bootstrap（同一实例多事件相关，不把事件当独立样本）。
部署语义：失败状态用 J0 代替（保留 P0），不静默删除。
"""
import argparse
import json
import os

import numpy as np


def _Jd(r, m):
    """部署 J：成功取 J_vis，失败取 J0。"""
    v = r[m]['det']
    return v['J'] if not v['fail'] else r['J0']


def _inst_mean(rows, m):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(_Jd(r, m))
    return [float(np.mean(v)) for v in per.values()]


def _paired_boot(rows, mA, mB, n_boot=2000, seed=0):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(_Jd(r, mA) - _Jd(r, mB))
    inst_diffs = [float(np.mean(v)) for v in per.values()]
    mean = float(np.mean(inst_diffs))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(inst_diffs), size=len(inst_diffs))
        boots.append(float(np.mean([inst_diffs[i] for i in idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _fail_rate(rows, m):
    n = sum(1 for r in rows if r[m]['det']['fail'])
    return n, n / max(len(rows), 1)


def _cert_fail_rate(rows, m):
    n = sum(1 for r in rows if r[m]['det']['cert_ok'] is False)
    return n, n / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-state', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.per_state) as f:
        rows = json.load(f)

    summary = {
        'n_states': len(rows),
        'n_instances': len(set(r['inst'] for r in rows)),
        'deployed_mean': {m: float(np.mean(_inst_mean(rows, m))) for m in ('R', 'Mpre', 'Mtrained')},
    }
    for mA, mB in (('Mtrained', 'Mpre'), ('Mtrained', 'R'), ('Mpre', 'R')):
        mean, lo, hi = _paired_boot(rows, mA, mB)
        summary[f'{mA}_minus_{mB}'] = {'mean': mean, 'ci_lo': lo, 'ci_hi': hi}

    wins = ties = losses = 0
    for r in rows:
        d = _Jd(r, 'Mtrained') - _Jd(r, 'R')
        if d < -1e-9:
            wins += 1
        elif d > 1e-9:
            losses += 1
        else:
            ties += 1
    summary['Mtrained_vs_R'] = {'win': wins, 'tie': ties, 'loss': losses}

    summary['fail_rate'] = {m: _fail_rate(rows, m) for m in ('R', 'Mpre', 'Mtrained')}
    summary['cert_fail_rate'] = {m: _cert_fail_rate(rows, m) for m in ('R', 'Mpre', 'Mtrained')}

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
