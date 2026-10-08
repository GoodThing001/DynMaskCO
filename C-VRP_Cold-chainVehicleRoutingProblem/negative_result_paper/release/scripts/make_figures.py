"""B 方向论文三张主图（matplotlib，v1 统一物理版）。

Fig1 配对效应：从 results/09_统一主表_v1/ 的机器可读 JSON 读取（不再硬编码）。
Fig2 掩码客户车辆归属保留率（execution_trace，v1 物理，标签 "assignment changed / not retained"）。
Fig3 未来评分逐实例配对差异（②b 修复后采样器 + v1 物理，N=8；clairvoyant 为同候选集特权上界）。

运行：cd C-VRP_Cold-chainVehicleRoutingProblem && python negative_result_paper/scripts/make_figures.py
产物：negative_result_paper/figures/fig{1,2,3}_*.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # negative_result_paper
OUT = os.path.join(ROOT, 'figures')
V1 = os.path.join(ROOT, 'results', '09_unified_v1')
os.makedirs(OUT, exist_ok=True)


def inst_clustered_boot(per_inst, n_boot=2000, seed=0):
    means = np.array([float(np.mean(v)) for v in per_inst])
    rng = np.random.default_rng(seed)
    boots = [float(np.mean([means[i] for i in rng.integers(0, len(means), size=len(means))]))
             for _ in range(n_boot)]
    return float(np.mean(means)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------------- #
# Fig1 配对效应（机器可读：固定状态 7 行 / 闭环 2 行，评价层级分开）
# --------------------------------------------------------------------------- #
def fig1():
    fixed_rows = ['42 train', '42 cal', '43 train', '43 cal', '44 train', '44 cal', 'heldout(42)']
    keys = ['s42_train', 's42_cal', 's43_train', 's43_cal', 's44_train', 's44_cal', 'heldout']
    cl_rows = ['42', '43']

    cand_R, cand_M, acc_R, acc_M = [], [], [], []
    for k in keys:
        with open(os.path.join(V1, f'{k}.analysis.json')) as f:
            d = json.load(f)
        cand_R.append(d['candidate_paired']['Mtrained_minus_R']['mean'])
        cand_M.append(d['candidate_paired']['Mtrained_minus_Mpre']['mean'])
        a_r = d['accepted_paired']['Mtrained_minus_R']
        a_m = d['accepted_paired']['Mtrained_minus_Mpre']
        acc_R.append((a_r['mean'], a_r['ci_lo'], a_r['ci_hi']))
        acc_M.append((a_m['mean'], a_m['ci_lo'], a_m['ci_hi']))

    cl_R, cl_M = [], []
    for s in ('42', '43'):
        with open(os.path.join(V1, f'step5_heldout_s{s}.summary.json')) as f:
            d = json.load(f)
        cl_R.append((d['R_vs_Mtrained_mean'], d['R_vs_Mtrained_ci'][0], d['R_vs_Mtrained_ci'][1]))
        cl_M.append((d['Mpre_vs_Mtrained_mean'], d['Mpre_vs_Mtrained_ci'][0], d['Mpre_vs_Mtrained_ci'][1]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='row')

    def forest(ax, rows, cand, acc, title, legend=False):
        y = np.arange(len(rows))
        ax.plot(cand, y, 'o', color='tab:blue', ms=5, label='candidate (no CI)')
        for yi, (m, lo, hi) in enumerate(acc):
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='s', color='tab:orange',
                        ms=5, capsize=2, label='accepted' if yi == 0 else None)
        ax.axvline(0, color='gray', ls='--', lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        if legend:
            ax.legend(fontsize=8, loc='lower right')

    def forest_cl(ax, rows, cl, title):
        y = np.arange(len(rows))
        for yi, (m, lo, hi) in enumerate(cl):
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='D', color='tab:green',
                        ms=6, capsize=2)
        ax.axvline(0, color='gray', ls='--', lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)

    forest(axes[0, 0], fixed_rows, cand_R, acc_R, '(a) fixed-state M-trained − R', legend=True)
    forest(axes[0, 1], fixed_rows, cand_M, acc_M, '(b) fixed-state M-trained − Mpre')
    forest_cl(axes[1, 0], cl_rows, cl_R, '(c) closed-loop M-trained − R (heldout 16)')
    forest_cl(axes[1, 1], cl_rows, cl_M, '(d) closed-loop M-trained − Mpre (heldout 16)')

    for ax in axes.flatten():
        ax.set_xlabel('paired Δ (M-trained − ·)')
    fig.suptitle('Fig 1: paired effects — fixed-state (a,b) vs closed-loop (c,d) are '
                 'different evaluation levels and instance sets', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig1_paired_effects.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig2 掩码客户车辆归属保留率
# --------------------------------------------------------------------------- #
def fig2():
    d = json.load(open(os.path.join(ROOT, 'results/06_机制实验/execution_trace_cal/per_state.json')))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for tag, color in [('dg', 'tab:blue'), ('exh', 'tab:orange')]:
        dist = {0: 0, 1: 0, 2: 0}
        pos = []
        for r in d:
            x = r.get(tag)
            if not x:
                continue
            dist[x['n_persisted']] += 1
            for p in x.get('cand_pos', {}).values():
                pos.append(p)
        n_states = sum(dist.values())
        frac = [dist[k] / n_states for k in (0, 1, 2)]
        xpos = np.arange(3) + (0.1 if tag == 'dg' else -0.1)
        axes[0].bar(xpos, frac, width=0.2, color=color, label=f'{tag} (n={n_states})')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(['0/2 both\nnot retained', '1/2 one\nretained', '2/2 both\nretained'])
    axes[0].set_ylabel('fraction of states')
    axes[0].set_title('(a) mask-customer vehicle assignment retention')
    axes[0].legend(fontsize=8)

    for tag, color in [('dg', 'tab:blue'), ('exh', 'tab:orange')]:
        pos = []
        for r in d:
            x = r.get(tag)
            if not x:
                continue
            pos.extend(x.get('cand_pos', {}).values())
        pos = np.array(pos)
        bins = np.arange(0, 12) - 0.5
        axes[1].hist(pos, bins=bins, color=color, alpha=0.6, label=tag, density=True)
    axes[1].axvline(0.5, color='gray', ls='--', lw=1)
    axes[1].set_xlabel('position of mask customer in candidate suffix (0 = first)')
    axes[1].set_ylabel('density')
    axes[1].set_title('(b) candidate suffix position of mask customers')
    axes[1].legend(fontsize=8)

    fig.suptitle('Fig 2: mask-customer vehicle assignment retention (this probe measures '
                 'vehicle assignment only)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig2_retention.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig3 未来评分逐实例配对差异（②b 修复后采样器，v1 物理，N=8）
# --------------------------------------------------------------------------- #
def fig3():
    d = json.load(open(os.path.join(ROOT, 'results/06_机制实验/future_sampling_oracle_v1/per_state.json')))
    per_inst = {}
    n_skip = 0
    for s in d:
        g = s['gain']
        if g.get('clairvoyant') is None or g.get('dist_full') is None or \
           g.get('dist_attr') is None or g.get('myopic') is None:
            n_skip += 1
            continue
        inst = s['inst']
        per_inst.setdefault(inst, []).append({
            'full': g['clairvoyant'] - g['dist_full'],
            'attr': g['clairvoyant'] - g['dist_attr'],
            'myopic': g['clairvoyant'] - g['myopic'],
        })
    print(f'fig3: {len(d) - n_skip}/{len(d)} states usable ({n_skip} skipped for None gain)')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    keys = ['full', 'attr', 'myopic']
    labels = {'full': 'clairvoyant − dist_full', 'attr': 'clairvoyant − dist_attr',
              'myopic': 'clairvoyant − myopic'}
    colors = {'full': 'tab:blue', 'attr': 'tab:orange', 'myopic': 'tab:red'}

    insts = sorted(per_inst.keys())
    for k in keys:
        xs = [np.mean([s[k] for s in per_inst[i]]) for i in insts]
        axes[0].scatter(insts, xs, color=colors[k], label=labels[k], s=25)
    axes[0].axhline(0, color='gray', ls='--', lw=1)
    axes[0].set_xlabel('instance')
    axes[0].set_ylabel('paired gain difference')
    axes[0].set_title('(a) per-instance paired difference (true-future gain)')
    axes[0].legend(fontsize=8)

    ys, yerrs = [], []
    for k in keys:
        vals = [[s[k] for s in per_inst[i]] for i in insts]
        mean, lo, hi = inst_clustered_boot(vals)
        ys.append(mean)
        yerrs.append([mean - lo, hi - mean])
    xs = np.arange(len(keys))
    axes[1].bar(xs, ys, yerr=np.array(yerrs).T, color=[colors[k] for k in keys],
                capsize=4, alpha=0.8)
    axes[1].axhline(0, color='gray', ls='--', lw=1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([labels[k] for k in keys], fontsize=8, rotation=15)
    axes[1].set_ylabel('paired gain difference (95% CI)')
    axes[1].set_title('(b) instance-clustered mean ± 95% CI')

    fig.suptitle('Fig 3: clairvoyant − sampling scoring (paired, N=8, fixed sampler, v1 objective)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, 'fig3_paired_gain.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig3 per-key mean/lo/hi:')
    for k in keys:
        vals = [[s[k] for s in per_inst[i]] for i in insts]
        m, lo, hi = inst_clustered_boot(vals)
        print(f'  {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print(f'saved figures to {OUT}')
