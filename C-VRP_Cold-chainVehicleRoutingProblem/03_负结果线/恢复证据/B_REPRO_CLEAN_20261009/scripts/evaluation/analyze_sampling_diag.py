"""采样诊断：读 per_state.json，统计 greedy vs 采样分布的候选潜力。

对 M-pre / M-trained 分别统计：
  - greedy（deterministic argmax）接受后质量；
  - best-of-1/2/4/8（前 k 个采样取 min 接受后 J）；
  - 胜 R 比例（greedy / best-of-8）；
  - 重复率（8 采样去重 + 与 greedy 重复比例）；
  - 失败率。
按实例聚类配对 bootstrap。best-of-8 仅作候选潜力诊断，不代表对 R 的公平胜出。
"""
import argparse
import json

import numpy as np

METHODS = ('Mpre', 'Mtrained')


def _acc(v, J0):
    """接受后 J：认证通过取 min(J0, J)，否则 J0。"""
    if v['fail'] or v['cert_ok'] is False:
        return J0
    return min(J0, v['J'])


def _acc_det(r, m):
    return _acc(r[m]['det'], r['J0'])


def _acc_sample(r, m, i):
    return _acc(r[m]['samples'][i], r['J0'])


def _inst_mean(rows, fn):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(fn(r))
    return [float(np.mean(v)) for v in per.values()]


def _paired_boot(rows, fA, fB, n_boot=2000, seed=0):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(fA(r) - fB(r))
    inst_diffs = [float(np.mean(v)) for v in per.values()]
    mean = float(np.mean(inst_diffs))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(inst_diffs), size=len(inst_diffs))
        boots.append(float(np.mean([inst_diffs[i] for i in idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}


def _win_r_rate(rows, fn_m):
    n = sum(1 for r in rows if fn_m(r) < _acc_det(r, 'R') - 1e-9)
    return n, n / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-state', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.per_state) as f:
        rows = json.load(f)

    R_acc = float(np.mean(_inst_mean(rows, lambda r: _acc_det(r, 'R'))))

    out = {
        'n_states': len(rows),
        'n_instances': len(set(r['inst'] for r in rows)),
        'R_greedy_accepted_mean': R_acc,
    }

    for m in METHODS:
        def det_fn(r):
            return _acc_det(r, m)

        def best_k_fn(k):
            return lambda r: min(_acc_sample(r, m, i) for i in range(min(k, len(r[m]['samples']))))

        res = {
            'greedy_accepted_mean': float(np.mean(_inst_mean(rows, det_fn))),
            'best_of_1': float(np.mean(_inst_mean(rows, best_k_fn(1)))),
            'best_of_2': float(np.mean(_inst_mean(rows, best_k_fn(2)))),
            'best_of_4': float(np.mean(_inst_mean(rows, best_k_fn(4)))),
            'best_of_8': float(np.mean(_inst_mean(rows, best_k_fn(8)))),
        }
        res['win_R_greedy'] = _win_r_rate(rows, det_fn)
        res['win_R_best8'] = _win_r_rate(rows, best_k_fn(8))

        # 重复率 + 失败率
        dup_rates = []
        greedy_dup_rates = []
        n_fail = 0
        n_total = 0
        for r in rows:
            samples = r[m]['samples']
            hashes = [s['hash'] for s in samples if s['hash'] is not None and not s['fail']]
            n_total += len(samples)
            n_fail += sum(1 for s in samples if s['fail'])
            n_uniq = len(set(hashes))
            dup_rates.append(1.0 - n_uniq / max(len(samples), 1))
            det_h = r[m]['det'].get('hash')
            if det_h is not None:
                greedy_dup_rates.append(sum(1 for h in hashes if h == det_h) / max(len(samples), 1))
            else:
                greedy_dup_rates.append(0.0)
        res['dup_rate_mean'] = float(np.mean(dup_rates))
        res['greedy_dup_rate_mean'] = float(np.mean(greedy_dup_rates))
        res['fail_rate'] = float(n_fail / max(n_total, 1))
        out[m] = res

    # 配对 bootstrap（实例聚类）
    out['paired'] = {
        'Mtrained_best8_minus_greedy': _paired_boot(
            rows, lambda r: min(_acc_sample(r, 'Mtrained', i) for i in range(len(r['Mtrained']['samples']))),
            lambda r: _acc_det(r, 'Mtrained')),
        'Mtrained_best8_minus_R': _paired_boot(
            rows, lambda r: min(_acc_sample(r, 'Mtrained', i) for i in range(len(r['Mtrained']['samples']))),
            lambda r: _acc_det(r, 'R')),
        'Mpre_best8_minus_R': _paired_boot(
            rows, lambda r: min(_acc_sample(r, 'Mpre', i) for i in range(len(r['Mpre']['samples']))),
            lambda r: _acc_det(r, 'R')),
        'Mtrained_best8_minus_Mpre_best8': _paired_boot(
            rows, lambda r: min(_acc_sample(r, 'Mtrained', i) for i in range(len(r['Mtrained']['samples']))),
            lambda r: min(_acc_sample(r, 'Mpre', i) for i in range(len(r['Mpre']['samples'])))),
    }

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
