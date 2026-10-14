"""对比方法总表配图（论文/汇报用）。

从 `dataset.json` 生成：
  左图：各方法效用（p_c=0 档 + 主档）横向条形，含 C1 cond_hist 基线参考线；
  右图：相对 C1 cond_hist 的配对差（95% CI），负值=不如基线。

用法：
    python CC_Compare/tools/render_compare_figure.py \
        --data CC_Compare/results/a1_compare_table/dataset.json \
        --outdir CC_Compare/results/a1_compare_table
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 中文标签在无中文字体的环境会退化为方块；统一用英文短标签保证可移植。
SHORT = {
    'routefinder': 'RouteFinder', 'cada': 'CaDA', 'mvmoe': 'MVMoE',
    'pomo': 'POMO', 'attention': 'AttentionModel', 'symnco': 'Sym-NCO',
    'sgbs': 'SGBS', 'deepaco': 'DeepACO', 'omnivrp': 'Omni-VRP',
    'lih': 'LIH', 'rrnco': 'RRNCO', 'ortools': 'OR-Tools', 'pyvrp': 'PyVRP',
    'myopic': 'rule: myopic', 'valuee20': 'rule: value/energy λ=2',
    'valuee30': 'rule: value/energy λ=3',
}


def load_rows(path):
    d = json.load(open(path, encoding='utf-8'))
    by = {}
    RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1, 'a1_ext_extra': 2}
    for r in d['arms']:
        if not r.get('formal'):
            continue
        by.setdefault(r['method'], {}).setdefault(r['p_c'], []).append(r)

    rows = []
    for k, per in by.items():
        def pick(pc):
            c = per.get(pc) or []
            return sorted(c, key=lambda x: RANK.get(x['group'], 9))[0] if c else None
        a0, am = pick('p_c=0'), pick('p_c=(5,10,15)')
        if not a0 and not am:
            continue
        base = a0 or am
        rows.append(dict(
            key=k, label=SHORT.get(k, base.get('arm') or k),
            u0=(a0 or {}).get('reported', {}).get('utility'),
            um=(am or {}).get('reported', {}).get('utility'),
            d0=((a0 or {}).get('repaired_vs_baseline') or {}).get('cond_hist'),
            dm=((am or {}).get('repaired_vs_baseline') or {}).get('cond_hist'),
        ))
    rows.sort(key=lambda r: -(r['u0']['mean'] if r['u0'] else -1e18))
    return d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    data, rows = load_rows(args.data)
    based = data['baseline_arms']
    c0 = based['p_c=0|cond_hist']['utility']
    cm = based['p_c=(5,10,15)|cond_hist']['utility']

    labels = [r['label'] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.2, 0.36 * len(rows) + 1.8)))

    ax = axes[0]
    h = 0.38
    v0 = [(r['u0'] or {}).get('mean') for r in rows]
    vm = [(r['um'] or {}).get('mean') for r in rows]
    e0 = [((r['u0'] or {}).get('mean', 0) - (r['u0'] or {}).get('ci_lo', 0))
          if r['u0'] else 0 for r in rows]
    em = [((r['um'] or {}).get('mean', 0) - (r['um'] or {}).get('ci_lo', 0))
          if r['um'] else 0 for r in rows]
    ax.barh(y + h / 2, [v if v is not None else 0 for v in v0], height=h,
            xerr=e0, color='#4C72B0', label='p_c = 0')
    ax.barh(y - h / 2, [v if v is not None else 0 for v in vm], height=h,
            xerr=em, color='#DD8452', label='main tier p_c=(5,10,15)')
    ax.axvline(c0['mean'], color='#4C72B0', ls='--', lw=1.4,
               label='C1 cond_hist (p_c=0)')
    ax.axvline(cm['mean'], color='#DD8452', ls='--', lw=1.4,
               label='C1 cond_hist (main)')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('A-v1 utility  (revenue − fuel − reject loss)')
    ax.set_title('Utility by method (40-day dev set, seed 20260926)', fontsize=11)
    ax.grid(axis='x', alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[1]
    d0 = [(r['d0'] or {}).get('mean') for r in rows]
    d0lo = [((r['d0'] or {}).get('mean', 0) - (r['d0'] or {}).get('ci_lo', 0))
            if r['d0'] else 0 for r in rows]
    dm = [(r['dm'] or {}).get('mean') for r in rows]
    dmlo = [((r['dm'] or {}).get('mean', 0) - (r['dm'] or {}).get('ci_lo', 0))
            if r['dm'] else 0 for r in rows]
    ax.barh(y + h / 2, [v if v is not None else 0 for v in d0], height=h,
            xerr=d0lo, color='#4C72B0')
    ax.barh(y - h / 2, [v if v is not None else 0 for v in dm], height=h,
            xerr=dmlo, color='#DD8452')
    ax.axvline(0, color='k', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel('paired difference vs C1 cond_hist (95% CI)')
    ax.set_title('Paired gain over the C1 scenario-SAA baseline', fontsize=11)
    ax.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    out = os.path.join(args.outdir, 'A-v1对比方法效用图.png')
    fig.savefig(out, dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    main()
