"""互补性分析：模型提议（搜索／采样）是否发现距离搜索遗漏的完整改善。

只读已有两层搜索的 per_state.json（run_sgbs_fixed_state.py 产出），不重跑、不重生成候选。

对每个状态 s：
  model_best(s)   = min(Mtrained_sgbs(s), Mtrained_greedy7(s))
  J_union(s)      = min(dist_sgbs(s), model_best(s))
  gain_union(s)   = dist_sgbs(s) - J_union(s)   (>= 0，按定义)

回答：模型平均虽输，是否在部分状态/实例上找到距离搜索未找到的完整改善；
改善的分量、分布、以及 TRAIN/CAL 迁移是否一致。不构成算法胜出证据。
"""
import argparse
import json
import os

import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def _inst_agg(rows, key):
    per = {}
    for r in rows:
        per.setdefault(r['inst'], []).append(r[key])
    return [float(np.mean(v)) for v in per.values()]


def _clustered_mean_ci(rows, key, n_boot=2000, seed=0):
    """按实例聚类的配对 bootstrap：对每实例均值差再做实例级重抽样。"""
    d = _inst_agg(rows, key)
    mean = float(np.mean(d))
    rng = np.random.default_rng(seed)
    b = [float(np.mean([d[i] for i in rng.integers(0, len(d), size=len(d))]))
         for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}


def analyze(path, eps=1e-9):
    rows = _load(path)
    for r in rows:
        r['model_best'] = min(r['Mtrained_sgbs'], r['Mtrained_greedy7'])
        r['J_union'] = min(r['dist_sgbs'], r['model_best'])
        r['gain_union'] = r['dist_sgbs'] - r['J_union']
        r['model_beats'] = r['model_best'] < r['dist_sgbs'] - eps
        r['sgbs_beats'] = r['Mtrained_sgbs'] < r['dist_sgbs'] - eps
        r['greedy7_beats'] = r['Mtrained_greedy7'] < r['dist_sgbs'] - eps
        r['sgbs_best'] = r['Mtrained_sgbs'] <= r['Mtrained_greedy7']

    n_states = len(rows)
    n_inst = len(set(r['inst'] for r in rows))
    gains = np.array([r['gain_union'] for r in rows], dtype=float)

    # 逐实例联合均值（先每实例取 min 后的 J 均值，再平均；等价于 gain_union 的实例聚合）
    union_inst = _inst_agg(rows, 'J_union')
    dist_inst = _inst_agg(rows, 'dist_sgbs')

    # 实例级：该实例内模型是否在任何状态改善，以及实例内最大改善
    inst_beats = {}
    inst_max_gain = {}
    inst_mean_gain = {}
    for r in rows:
        i = r['inst']
        inst_beats.setdefault(i, False)
        inst_beats[i] = inst_beats[i] or r['model_beats']
        inst_max_gain[i] = max(inst_max_gain.get(i, 0.0), r['gain_union'])
        inst_mean_gain.setdefault(i, []).append(r['gain_union'])
    inst_mean_gain = {i: float(np.mean(v)) for i, v in inst_mean_gain.items()}

    # 有改善状态的改善幅度分布
    pos = gains[gains > eps]
    pos_states_idx = [i for i, r in enumerate(rows) if r['gain_union'] > eps]

    out = {
        'n_states': n_states,
        'n_instances': n_inst,
        'accepted_mean': {
            'dist_sgbs': float(np.mean(_inst_agg(rows, 'dist_sgbs'))),
            'Mtrained_sgbs': float(np.mean(_inst_agg(rows, 'Mtrained_sgbs'))),
            'Mtrained_greedy7': float(np.mean(_inst_agg(rows, 'Mtrained_greedy7'))),
            'J_union': float(np.mean(union_inst)),
        },
        'gain_union': _clustered_mean_ci(rows, 'gain_union'),
        'gain_union_per_state_quantiles': {
            'min': float(np.min(gains)), 'p25': float(np.percentile(gains, 25)),
            'p50': float(np.percentile(gains, 50)), 'p75': float(np.percentile(gains, 75)),
            'p90': float(np.percentile(gains, 90)), 'max': float(np.max(gains)),
        },
        'positive_gain': {
            'n_states': int(np.sum(gains > eps)),
            'frac_states': float(np.mean(gains > eps)),
            'n_instances': int(sum(1 for v in inst_beats.values() if v)),
            'frac_instances': float(np.mean(list(inst_beats.values()))),
        },
        'positive_gain_magnitude': {
            'mean_over_positive_states': float(np.mean(pos)) if len(pos) else 0.0,
            'median_over_positive_states': float(np.median(pos)) if len(pos) else 0.0,
            'max_over_positive_states': float(np.max(pos)) if len(pos) else 0.0,
            'sum_over_positive_states': float(np.sum(pos)),
        },
        'threshold_frac': {
            '>=0.005': float(np.mean(gains >= 0.005)),
            '>=0.01': float(np.mean(gains >= 0.01)),
            '>=0.02': float(np.mean(gains >= 0.02)),
            '>=0.05': float(np.mean(gains >= 0.05)),
        },
        'contributor': {
            'n_sgbs_beats': int(np.sum([r['sgbs_beats'] for r in rows])),
            'n_greedy7_beats': int(np.sum([r['greedy7_beats'] for r in rows])),
            'n_sgbs_strictly_better_than_greedy7': int(np.sum(
                [r['Mtrained_sgbs'] < r['Mtrained_greedy7'] - eps for r in rows])),
            'n_greedy7_strictly_better_than_sgbs': int(np.sum(
                [r['Mtrained_greedy7'] < r['Mtrained_sgbs'] - eps for r in rows])),
        },
        'instance_max_gain': {
            'mean': float(np.mean(list(inst_max_gain.values()))),
            'max': float(np.max(list(inst_max_gain.values()))),
        },
        'instance_mean_gain': {
            'mean': float(np.mean(list(inst_mean_gain.values()))),
            'max': float(np.max(list(inst_mean_gain.values()))),
            'min': float(np.min(list(inst_mean_gain.values()))),
        },
        'per_instance': [
            {'inst': i, 'beats': inst_beats[i], 'mean_gain': inst_mean_gain[i],
             'max_gain': inst_max_gain[i]}
            for i in sorted(inst_beats)
        ],
        'positive_states': [
            {'inst': rows[i]['inst'], 'event': rows[i]['event'],
             'dist_sgbs': rows[i]['dist_sgbs'], 'Mtrained_sgbs': rows[i]['Mtrained_sgbs'],
             'Mtrained_greedy7': rows[i]['Mtrained_greedy7'],
             'model_best': rows[i]['model_best'], 'gain': rows[i]['gain_union']}
            for i in pos_states_idx
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-state', nargs='+', required=True,
                    help='per_state.json 路径（可多个 split）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for p in args.per_state:
        name = os.path.basename(os.path.dirname(p))
        r = analyze(p)
        results[name] = r
        print(f"===== {name} =====")
        print(json.dumps({k: v for k, v in r.items() if k not in ('per_instance', 'positive_states')},
                         indent=2))
        print()

    with open(os.path.join(args.out, 'complementarity.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved: {args.out}/complementarity.json")


if __name__ == '__main__':
    main()
