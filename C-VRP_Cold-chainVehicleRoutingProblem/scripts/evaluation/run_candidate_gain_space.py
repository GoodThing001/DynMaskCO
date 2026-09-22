"""候选收益空间（hindsight 参照，几乎零成本，不训练、不用模型）。

对每个 context，用已有完整标签计算
    g* = max(0, max_a [J_KEEP - J_a])
只在「有效监督」（supervision_mask：有终局 outcome、service_ok、无协议错误）的候选中计算，
并标明标签覆盖情况。它是当前候选集与既定 continuation 下的 hindsight 上限参照，不是在线
可用策略。

判读：
  - g* 接近零 → 当前样本几乎没有可学收益；
  - g* 明显为正、模型净收益仍约零 → 存在可利用但未学到的选择空间（转向状态/路线表征）。

用法（服务器）：
    python scripts/evaluation/run_candidate_gain_space.py \
        --cal-teacher-dir results/m0_scale/cal --cal-data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --dev-teacher-dir results/m0_scale/dev_check --dev-data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --out results/m0_scale/gstar
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from coldchain_teacher_dataset import load_teacher_dataset


def _g_star_rows(ds):
    """每个 context 的 g* 与标签覆盖；无有效监督的 context g* = None。"""
    rows = []
    for ctx in ds.contexts:
        cands = ds.candidates_by_context[ctx['context_id']]
        sup = ds.supervision_mask(cands)
        gains = []
        for c, s in zip(cands, sup):
            if s and c.get('delta_vs_keep') is not None:
                gains.append(-float(c['delta_vs_keep']))
        rows.append({
            'inst_idx': int(ctx['inst_idx']),
            'event_id': int(ctx['event_id']),
            'n_candidates': len(cands),
            'n_supervised': int(sum(sup)),
            'g_star': (max(0.0, max(gains)) if gains else None),
            'best_delta_vs_keep': (-max(gains) if gains else None),
        })
    return rows


def _candidate_categories(ds):
    """四类候选计数：区分「不合法（正常无标签）」与「合法但真正缺标签」。

    1) illegal：当前约束下不合法（枚举负例），不参与在线选择，不需要成本标签；
    2) legal_supervised：合法 + 终局服务完整 + 标签有效（成本监督）；
    3) legal_service_fail：合法 + 终局服务失败（单列失败，不能冒充零收益）；
    4) legal_missing_or_protocol_error：合法 + 结果缺失或协议错误（真正需解释的标签缺口）。
    """
    counts = {'illegal': 0, 'legal_supervised': 0, 'legal_service_fail': 0,
              'legal_missing_or_protocol_error': 0}
    for ctx in ds.contexts:
        cands = ds.candidates_by_context[ctx['context_id']]
        legal = ds.legal_mask(cands)
        sup = ds.supervision_mask(cands)
        for c, l, s in zip(cands, legal, sup):
            if not l:
                counts['illegal'] += 1
            elif s:
                counts['legal_supervised'] += 1
            else:
                has_outcome = c.get('outcome') is not None
                service_ok = bool(c.get('service_ok', False))
                protocol_error = bool(c.get('protocol_error', False))
                if has_outcome and not service_ok and not protocol_error:
                    counts['legal_service_fail'] += 1
                else:
                    counts['legal_missing_or_protocol_error'] += 1
    total = sum(counts.values())
    n_legal = total - counts['illegal']
    counts['total'] = int(total)
    counts['n_legal'] = int(n_legal)
    counts['supervised_over_legal'] = (counts['legal_supervised'] / n_legal) if n_legal else None
    counts['supervised_over_all'] = (counts['legal_supervised'] / total) if total else None
    return counts


def _summarize(rows):
    g = [r['g_star'] for r in rows if r['g_star'] is not None]
    n_ctx = len(rows)
    n_supervised_ctx = len(g)
    n_supervised_cand = int(sum(r['n_supervised'] for r in rows))
    n_cand = int(sum(r['n_candidates'] for r in rows))
    g_pos = [x for x in g if x > 0.0]
    return {
        'n_contexts': n_ctx,
        'n_contexts_with_supervision': n_supervised_ctx,
        'n_supervised_candidates': n_supervised_cand,
        'n_candidates': n_cand,
        'label_coverage_contexts': (n_supervised_ctx / n_ctx) if n_ctx else None,
        'label_coverage_candidates': (n_supervised_cand / n_cand) if n_cand else None,
        'mean_g_star': float(np.mean(g)) if g else None,
        'median_g_star': float(np.median(g)) if g else None,
        'p90_g_star': float(np.percentile(g, 90)) if g else None,
        'max_g_star': float(np.max(g)) if g else None,
        'frac_zero_g_star': (sum(1 for x in g if x <= 1e-9) / len(g)) if g else None,
        'frac_positive_g_star': (len(g_pos) / len(g)) if g else None,
        'mean_g_star_when_positive': float(np.mean(g_pos)) if g_pos else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal-teacher-dir', required=True)
    ap.add_argument('--cal-data', required=True)
    ap.add_argument('--dev-teacher-dir', required=True)
    ap.add_argument('--dev-data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cal_ds = load_teacher_dataset(args.cal_teacher_dir, data_path=args.cal_data)
    dev_ds = load_teacher_dataset(args.dev_teacher_dir, data_path=args.dev_data)
    cal_rows = _g_star_rows(cal_ds)
    dev_rows = _g_star_rows(dev_ds)
    cal_cats = _candidate_categories(cal_ds)
    dev_cats = _candidate_categories(dev_ds)

    out = {'cal': _summarize(cal_rows), 'dev': _summarize(dev_rows),
           'cal_per_context': cal_rows, 'dev_per_context': dev_rows,
           'cal_candidate_categories': cal_cats, 'dev_candidate_categories': dev_cats}
    with open(os.path.join(args.out, 'gstar.json'), 'w') as f:
        json.dump(out, f, indent=2)

    for name, s, cats in [('CAL', out['cal'], cal_cats), ('DEV', out['dev'], dev_cats)]:
        print(f"\n=== {name} candidate gain space g* ===")
        print(f"  contexts={s['n_contexts']} with_supervision={s['n_contexts_with_supervision']} "
              f"label_cov(ctx)={_f(s['label_coverage_contexts'])}")
        print(f"  mean g*={_f(s['mean_g_star'])} median={_f(s['median_g_star'])} "
              f"p90={_f(s['p90_g_star'])} max={_f(s['max_g_star'])}")
        print(f"  frac g*=0: {_f(s['frac_zero_g_star'])}  frac g*>0: {_f(s['frac_positive_g_star'])}  "
              f"mean g*|g*>0: {_f(s['mean_g_star_when_positive'])}")
        print(f"  candidate categories: illegal={cats['illegal']} "
              f"legal_supervised={cats['legal_supervised']} "
              f"legal_service_fail={cats['legal_service_fail']} "
              f"legal_missing_or_protocol_error={cats['legal_missing_or_protocol_error']}")
        print(f"    supervised/legal={_f(cats['supervised_over_legal'])}  "
              f"supervised/all={_f(cats['supervised_over_all'])}")
    print(f"\nsaved: {args.out}/gstar.json")


def _f(x):
    return 'nan' if x is None else f'{x:.4f}'


if __name__ == '__main__':
    main()
