"""固定状态质量分析：读取 per_state.json，计算候选质量 + 接受后质量两套指标。

两套语义：
  - candidate：失败兜底 J0（成功但变差仍计候选 J_vis）——即「失败兜底后的候选质量」。
  - accepted：真实在线接受规则——候选完整认证通过时取 min(J0, J_cand)，否则 J0。

接受规则对齐在线搜索（best 从 P0 起，仅 J < J0 才替换）。配对比较按实例聚类
bootstrap（同一实例多事件相关）。best-of-8 仅作辅助，不能与 R 一次重构直接宣称公平。
"""
import argparse
import json
import os

import numpy as np

METHODS = ('R', 'Mpre', 'Mtrained')


def _cand_J(r, m):
    v = r[m]['det']
    return v['J'] if not v['fail'] else r['J0']


def _accept_J(r, m):
    v = r[m]['det']
    if v['fail'] or v['cert_ok'] is False:
        return r['J0']
    return min(r['J0'], v['J'])


def _is_accepted(r, m):
    v = r[m]['det']
    return (not v['fail']) and (v['cert_ok'] is True) and (v['J'] < r['J0'] - 1e-9)


def _inst_mean(rows, m, jfn):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(jfn(r, m))
    return [float(np.mean(v)) for v in per.values()]


def _paired_boot(rows, mA, mB, jfn, n_boot=2000, seed=0):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(jfn(r, mA) - jfn(r, mB))
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


def _accept_rate(rows, m):
    n = sum(1 for r in rows if _is_accepted(r, m))
    return n, n / max(len(rows), 1)


def _samples_agg(rows, m):
    """8 次采样：候选均值（非失败）、接受后 best-of-8（实例级均值）。"""
    per_cand = {}
    per_best8 = {}
    n_fail = 0
    n_total = 0
    for r in rows:
        inst = r['inst']
        cands = []
        accepts = []
        for s in r[m]['samples']:
            n_total += 1
            if s['fail']:
                n_fail += 1
                accepts.append(r['J0'])
                continue
            cands.append(s['J'])
            aj = r['J0'] if s['cert_ok'] is False else min(r['J0'], s['J'])
            accepts.append(aj)
        if cands:
            per_cand.setdefault(inst, []).append(float(np.mean(cands)))
        per_best8.setdefault(inst, []).append(float(min(accepts)))
    cand_mean = [float(np.mean(v)) for v in per_cand.values()] if per_cand else []
    best8_mean = [float(np.mean(v)) for v in per_best8.values()]
    return {
        'mean_candidate': float(np.mean(cand_mean)) if cand_mean else None,
        'best_of_8_accepted': float(np.mean(best8_mean)),
        'fail_rate': float(n_fail / max(n_total, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-state', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.per_state) as f:
        rows = json.load(f)

    j0_mean = float(np.mean([r['J0'] for r in rows]))

    summary = {
        'n_states': len(rows),
        'n_instances': len(set(r['inst'] for r in rows)),
        'J0_mean': j0_mean,
        'candidate_mean': {m: float(np.mean(_inst_mean(rows, m, _cand_J))) for m in METHODS},
        'accepted_mean': {m: float(np.mean(_inst_mean(rows, m, _accept_J))) for m in METHODS},
        'improvement_vs_J0': {
            m: float(j0_mean - np.mean(_inst_mean(rows, m, _accept_J))) for m in METHODS
        },
        'acceptance_rate': {m: _accept_rate(rows, m) for m in METHODS},
    }

    summary['candidate_paired'] = {}
    summary['accepted_paired'] = {}
    for mA, mB in (('Mtrained', 'Mpre'), ('Mtrained', 'R'), ('Mpre', 'R')):
        summary['candidate_paired'][f'{mA}_minus_{mB}'] = dict(
            zip(('mean', 'ci_lo', 'ci_hi'), _paired_boot(rows, mA, mB, _cand_J)))
        summary['accepted_paired'][f'{mA}_minus_{mB}'] = dict(
            zip(('mean', 'ci_lo', 'ci_hi'), _paired_boot(rows, mA, mB, _accept_J)))

    wins = ties = losses = 0
    for r in rows:
        d = _accept_J(r, 'Mtrained') - _accept_J(r, 'R')
        if d < -1e-9:
            wins += 1
        elif d > 1e-9:
            losses += 1
        else:
            ties += 1
    summary['Mtrained_vs_R_accepted'] = {'win': wins, 'tie': ties, 'loss': losses}

    summary['fail_rate'] = {m: _fail_rate(rows, m) for m in METHODS}
    summary['cert_fail_rate'] = {m: _cert_fail_rate(rows, m) for m in METHODS}
    summary['samples'] = {'Mpre': _samples_agg(rows, 'Mpre'),
                          'Mtrained': _samples_agg(rows, 'Mtrained')}

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
