"""CAL 离线 margin 扫描 + 固定 DEV 矩阵（去编号消融配套评价）。

对给定 feature-only-local checkpoint，在 CAL 上离线扫描接受阈值 τ（J 单位），并固定
CAL 选择规则选出 τ*（只用 CAL，不看 DEV）；在 DEV 上报告 τ=0 / τ* 以及全网格（探索性
附表，不用于选择）。

接受规则：选出最高分合法修改动作 a，仅当 ĝ(a)=s·[f(a)−f(KEEP/DEFER)] > τ 时接受，否则
KEEP/DEFER。KEEP/DEFER 用真实评分（不人工置零）。

服务失败与标签不可用**不算零收益**：单独计数，接受后不计入收益分母；只要非零，
完整策略净收益标为 `service_feasible=false` / `label_complete=false`，并另报有效标签的
条件均值。每实例结果与跨实例聚合都落盘（不只看跨实例均值）。

每个 checkpoint 记录：CAL 选择规则、CAL 选出的 τ、DEV 上 τ=0、DEV 上该固定 τ。
DEV 全网格保留为探索性附表，不从其中挑最优阈值当主结果。

用法（服务器）：
    python scripts/evaluation/run_margin_scan.py \
        --ckpt results/m0_scale/probe_local_n64_s42/local.ckpt \
        --cal-teacher-dir results/m0_scale/cal --cal-data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --dev-teacher-dir results/m0_scale/dev_check --dev-data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --deindex \
        --out results/m0_scale/margin_n64_deindex_s42
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert',
          'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

import jax.numpy as jnp
from flax import nnx

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        extract_candidate_local_features,
                                        ACTION_FEAT_DIM, LOCAL_VAL_DIM, LOCAL_VALID_DIM)
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM
from train_feature_only_local import LocalHead, IN_DIM, build_dataset

TAU_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, float('inf')]
# 固定 CAL 选择规则（只看 CAL，不看 DEV）：在 service_feasible 且 label_complete 且数值有限的
# 配置中，最大化 CAL 实例等权净收益，平局取更小 τ。τ=∞ 选到只表示「保持 baseline」。
CAL_SELECTION_RULE = ('argmax over CAL grid of instance-equal net_gain, '
                      'restricted to service_feasible & label_complete & finite, '
                      'tie-break smaller tau')


def _score(graphdef, params, local, ctx, av, avd, lv, lvd):
    h = nnx.merge(graphdef, params)
    return h(jnp.asarray(ctx), jnp.asarray(av), jnp.asarray(avd),
             jnp.asarray(lv), jnp.asarray(lvd))


def _load(ckpt_path, seed=42):
    import pickle
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    head = LocalHead(IN_DIM, rngs=seed)
    graphdef, _ = nnx.split(head)
    return graphdef, ck['params'], ck['s']


def _score_all(graphdef, params, t):
    C = t['context_vecs'].shape[0]
    out = []
    for i in range(C):
        s = _score(graphdef, params, True, t['context_vecs'][i:i+1], t['action_vals'][i:i+1],
                   t['action_valid'][i:i+1], t['local_vals'][i:i+1], t['local_valid'][i:i+1])
        out.append(np.asarray(s)[0])
    return np.stack(out)


def _context_decision(t, scores, i, s_scale):
    """返回 (g_hat, accept_gain, status)：status ∈ {keep, beneficial, neutral, harmful,
    service_fail, label_missing}。"""
    legal = t['legal'][i]
    sup = t['supervision'][i]
    is_pseudo = t['is_pseudo'][i]
    delta = t['delta'][i]
    service_ok = t['service_ok'][i]
    s = scores[i]

    # KEEP/DEFER 基准分数（真实评分）
    keep_idx = np.flatnonzero(legal & is_pseudo)
    if keep_idx.size > 0:
        keep_score = float(s[keep_idx[0]])
    else:
        keep_score = float(s[np.flatnonzero(legal)[0]])

    # 最高分非 KEEP 修改动作
    mod_idx = np.flatnonzero(legal & ~is_pseudo)
    if mod_idx.size == 0:
        return None, None, 'keep'
    best = mod_idx[int(np.argmax(s[mod_idx]))]
    g_hat = s_scale * (float(s[best]) - keep_score)
    if g_hat <= 0:
        return g_hat, 0.0, 'keep'   # 不满足 τ=0，最低门槛都不改
    # 该动作的实际后果
    if not sup[best]:
        if not service_ok[best]:
            return g_hat, None, 'service_fail'
        return g_hat, None, 'label_missing'
    d = delta[best]
    if d < -1e-9:
        return g_hat, -d, 'beneficial'
    if d > 1e-9:
        return g_hat, -d, 'harmful'
    return g_hat, 0.0, 'neutral'


def _empty_inst():
    return {'n_contexts': 0, 'n_keep': 0, 'n_accept': 0,
            'n_accept_beneficial': 0, 'n_accept_neutral': 0, 'n_accept_harmful': 0,
            'n_accept_service_fail': 0, 'n_accept_label_missing': 0,
            'gain_sum': 0.0, 'accept_gain_sum': 0.0, 'regret_sum': 0.0, 'harm_loss': []}


def _context_g_star(delta, supervision):
    """每 context 的 hindsight 最优单步收益 g* = max(0, max_a[-(delta_a)])（只在有效标签候选上）。"""
    sup = np.asarray(supervision)
    d = np.asarray(delta, np.float64)
    g = np.zeros(d.shape[0], np.float64)
    for i in range(d.shape[0]):
        m = sup[i]
        if m.any():
            g[i] = max(0.0, float(np.max(-d[i][m])))
    return g


def _scan(t, scores, s_scale):
    C = len(t['instance_ids'])
    decisions = [_context_decision(t, scores, i, s_scale) for i in range(C)]
    g_star = _context_g_star(t['delta'], t['supervision'])
    insts = np.unique(t['instance_ids'])
    rows = {}
    per_instance = {}
    for tau in TAU_GRID:
        per_inst = {int(inst): _empty_inst() for inst in insts}
        for i in range(C):
            g_hat, gain, status = decisions[i]
            inst = int(t['instance_ids'][i])
            p = per_inst[inst]
            p['n_contexts'] += 1
            if status == 'keep':
                p['n_keep'] += 1
                p['gain_sum'] += 0.0
                p['regret_sum'] += g_star[i]
                continue
            # status != keep 意味着 g_hat > 0；是否接受取决于 τ
            if g_hat is None or g_hat <= tau:
                p['n_keep'] += 1
                p['gain_sum'] += 0.0
                p['regret_sum'] += g_star[i]
                continue
            p['n_accept'] += 1
            if status == 'beneficial':
                p['n_accept_beneficial'] += 1
                p['gain_sum'] += gain
                p['accept_gain_sum'] += gain
                p['regret_sum'] += g_star[i] - gain
            elif status == 'neutral':
                p['n_accept_neutral'] += 1
                p['gain_sum'] += 0.0
                p['accept_gain_sum'] += 0.0
                p['regret_sum'] += g_star[i]
            elif status == 'harmful':
                p['n_accept_harmful'] += 1
                p['gain_sum'] += gain
                p['accept_gain_sum'] += gain
                p['regret_sum'] += g_star[i] - gain
                p['harm_loss'].append(gain)
            elif status == 'service_fail':
                p['n_accept_service_fail'] += 1
                # 不计收益/regret（未定义），单独计数
            elif status == 'label_missing':
                p['n_accept_label_missing'] += 1
                # 不计收益/regret（未定义），单独计数
        rows[str(tau)] = _aggregate(per_inst, tau)
        per_instance[str(tau)] = per_inst
    return rows, per_instance


def _aggregate(per_inst, tau):
    def _sum(k):
        return int(sum(v[k] for v in per_inst.values()))
    n_contexts = _sum('n_contexts')
    n_keep = _sum('n_keep')
    n_accept = _sum('n_accept')
    n_beneficial = _sum('n_accept_beneficial')
    n_neutral = _sum('n_accept_neutral')
    n_harmful = _sum('n_accept_harmful')
    n_service_fail = _sum('n_accept_service_fail')
    n_label_missing = _sum('n_accept_label_missing')
    n_valid_accept = n_beneficial + n_neutral + n_harmful
    gain_sum = float(sum(v['gain_sum'] for v in per_inst.values()))
    accept_gain_sum = float(sum(v['accept_gain_sum'] for v in per_inst.values()))
    regret_sum = float(sum(v['regret_sum'] for v in per_inst.values()))
    harm_losses = [l for v in per_inst.values() for l in v['harm_loss']]

    # 「全部 context 净收益」：有效标签的接受动作计 -delta，KEEP/拒绝计 0；service_fail /
    # label_missing 接受动作不计收益（单独计数），故当它们非零时此数把未定义结果当作 0，
    # 需配合 service_feasible / label_complete 判读。
    # 主口径：实例等权 G = (1/N) Σ_i (1/C_i) Σ_c g_ic；辅口径：context 混合均值 Σg / ΣC。
    net_gain_context_pooled = (gain_sum / n_contexts) if n_contexts else 0.0
    per_inst_gain = [v['gain_sum'] / v['n_contexts'] if v['n_contexts'] else 0.0
                     for v in per_inst.values()]
    net_gain_instance_mean = float(np.mean(per_inst_gain)) if per_inst_gain else 0.0
    # 有效标签部分的条件均值（排除 service_fail / label_missing 接受动作）。
    n_scored = n_contexts - n_service_fail - n_label_missing
    net_gain_valid = (gain_sum / n_scored) if n_scored else None

    # regret = g* − 实际单步收益（g* 是 hindsight 最优单步收益参照）。
    regret_context_pooled = (regret_sum / n_scored) if n_scored else None
    per_inst_regret = [v['regret_sum'] / (v['n_contexts'] - v['n_accept_service_fail']
                                         - v['n_accept_label_missing'])
                       if (v['n_contexts'] - v['n_accept_service_fail']
                           - v['n_accept_label_missing']) else None
                       for v in per_inst.values()]
    per_inst_regret = [r for r in per_inst_regret if r is not None]
    regret_instance_mean = (float(np.mean(per_inst_regret)) if per_inst_regret else None)

    return {
        'tau': tau,
        'n_contexts': n_contexts,
        'n_accept': n_accept,
        'n_keep': n_keep,
        'n_accept_beneficial': n_beneficial,
        'n_accept_neutral': n_neutral,
        'n_accept_harmful': n_harmful,
        'n_accept_service_fail': n_service_fail,
        'n_accept_label_missing': n_label_missing,
        'n_valid_accept': n_valid_accept,
        'accept_coverage': (n_accept / n_contexts) if n_contexts else 0.0,
        'avg_gain_after_accept': (accept_gain_sum / n_valid_accept) if n_valid_accept else None,
        'harmful_accept_ratio': (n_harmful / n_accept) if n_accept else None,
        'net_gain_instance_mean': net_gain_instance_mean,
        'net_gain_context_pooled': net_gain_context_pooled,
        'net_gain_valid_label_conditional': net_gain_valid,
        'regret_instance_mean': regret_instance_mean,
        'regret_context_pooled': regret_context_pooled,
        'service_feasible': n_service_fail == 0,
        'label_complete': n_label_missing == 0,
        'harm_mean_loss': float(np.mean(harm_losses)) if harm_losses else 0.0,
        'harm_max_loss': float(np.max(harm_losses)) if harm_losses else 0.0,
    }


def select_tau_cal(cal_rows):
    """固定 CAL 选择规则（只看 CAL，平局取更小 τ）。

    1) 只在服务完整(service_feasible)且被选动作标签完整(label_complete)且数值有限的配置中选择；
    2) 在固定网格上最大化 CAL 实例等权净收益 net_gain_instance_mean；
    3) 平局取更小 τ（更接近 τ=0，即更激进接受）；
    4) τ=∞ 保留，但选到它只表示「保持 baseline」，不算方法改善（调用方标记）。
    """
    best = None
    for tau in TAU_GRID:
        r = cal_rows[str(tau)]
        if not r['service_feasible'] or not r['label_complete']:
            continue
        g = r['net_gain_instance_mean']
        if g is None or not np.isfinite(g):
            continue
        if best is None or g > best[0] + 1e-12:
            best = (g, tau)
    return best[1] if best is not None else None


def _inst_row(p):
    """把一个 instance 的累计计数转成 JSON-ready 行（含派生指标）。"""
    n_ctx = p['n_contexts']
    n_accept = p['n_accept']
    n_valid = p['n_accept_beneficial'] + p['n_accept_neutral'] + p['n_accept_harmful']
    return {
        'n_contexts': int(n_ctx),
        'n_accept': int(n_accept),
        'n_keep': int(p['n_keep']),
        'n_accept_beneficial': int(p['n_accept_beneficial']),
        'n_accept_neutral': int(p['n_accept_neutral']),
        'n_accept_harmful': int(p['n_accept_harmful']),
        'n_accept_service_fail': int(p['n_accept_service_fail']),
        'n_accept_label_missing': int(p['n_accept_label_missing']),
        'accept_coverage': (float(n_accept) / n_ctx) if n_ctx else 0.0,
        'net_gain_all_contexts': (float(p['gain_sum']) / n_ctx) if n_ctx else 0.0,
        'avg_gain_after_accept': (float(p['accept_gain_sum']) / n_valid) if n_valid else None,
    }


def _per_instance_json(per_inst):
    """per_inst: {tau_str: {inst_id: _empty_inst_dict}} → JSON-ready。"""
    out = {}
    for tau, m in per_inst.items():
        out[str(tau)] = {str(k): _inst_row(v) for k, v in m.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cal-teacher-dir', required=True)
    ap.add_argument('--cal-data', required=True)
    ap.add_argument('--dev-teacher-dir', required=True)
    ap.add_argument('--dev-data', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--deindex', action='store_true',
                    help='去编号消融：用屏蔽编号通道后的输入重建特征')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    graphdef, params, s_scale = _load(args.ckpt, args.seed)

    cal_ds = load_teacher_dataset(args.cal_teacher_dir, data_path=args.cal_data)
    cal_t = build_dataset(cal_ds, dict(np.load(args.cal_data)), args.capacity,
                          deindex=args.deindex)
    dev_ds = load_teacher_dataset(args.dev_teacher_dir, data_path=args.dev_data)
    dev_t = build_dataset(dev_ds, dict(np.load(args.dev_data)), args.capacity,
                          deindex=args.deindex)

    cal_scores = _score_all(graphdef, params, cal_t)
    dev_scores = _score_all(graphdef, params, dev_t)

    cal_rows, cal_per_inst = _scan(cal_t, cal_scores, s_scale)
    dev_rows, dev_per_inst = _scan(dev_t, dev_scores, s_scale)

    tau_star = select_tau_cal(cal_rows)
    keep_baseline = (tau_star is not None and float(tau_star) == float('inf'))
    cal_star = cal_rows[str(tau_star)] if tau_star is not None else None
    dev_at_0 = dev_rows['0.0']
    dev_at_star = dev_rows[str(tau_star)] if tau_star is not None else None

    with open(os.path.join(args.out, 'margin_scan.json'), 'w') as f:
        json.dump({
            's_scale': s_scale,
            'deindex': args.deindex,
            'cal_selection_rule': CAL_SELECTION_RULE,
            'tau_grid': [str(t) for t in TAU_GRID],
            'cal_selected_tau': tau_star,
            'cal_selected_tau_is_keep_baseline': keep_baseline,
            'cal': cal_rows,
            'dev': dev_rows,
            'cal_selected': {'cal': cal_star, 'dev_at_tau0': dev_at_0,
                             'dev_at_selected_tau': dev_at_star},
            'per_instance': {'cal': _per_instance_json(cal_per_inst),
                             'dev': _per_instance_json(dev_per_inst)},
        }, f, indent=2)

    print(f"\n=== margin scan (s={s_scale:.4f}, deindex={args.deindex}) ===")
    print(f"CAL selection rule: {CAL_SELECTION_RULE}")
    print(f"CAL selected tau* = {tau_star}"
          f"{'  (keep baseline)' if keep_baseline else ''}\n")
    print(f"{'tau':>7s} | {'CAL gain':>9s} {'cov':>6s} {'ben/neu/harm':>14s} "
          f"{'srv/label':>10s} | {'DEV gain':>9s} {'cov':>6s} {'ben/neu/harm':>14s}")
    for tau in TAU_GRID:
        cr, dr = cal_rows[str(tau)], dev_rows[str(tau)]
        print(f"{tau:7.4g} | {cr['net_gain_instance_mean']:9.4f} {cr['accept_coverage']:6.3f} "
              f"{cr['n_accept_beneficial']}/{cr['n_accept_neutral']}/{cr['n_accept_harmful']:>3d}    "
              f"{cr['n_accept_service_fail']}/{cr['n_accept_label_missing']:>3d}     "
              f"| {dr['net_gain_instance_mean']:9.4f} {dr['accept_coverage']:6.3f} "
              f"{dr['n_accept_beneficial']}/{dr['n_accept_neutral']}/{dr['n_accept_harmful']}")

    print(f"\n=== fixed CAL rule summary (primary = instance-equal net gain) ===")
    for name, r in [('CAL@tau*', cal_star), ('DEV@tau=0', dev_at_0), ('DEV@tau*', dev_at_star)]:
        if r is None:
            print(f"  {name:10s}: (no valid config)")
            continue
        print(f"  {name:10s}: net_gain={r['net_gain_instance_mean']:+.4f} "
              f"(ctx-pooled {r['net_gain_context_pooled']:+.4f}) "
              f"cov={r['accept_coverage']:.3f} "
              f"avg_gain_accept={_fmt(r['avg_gain_after_accept'])} "
              f"harm_ratio={_fmt(r['harmful_accept_ratio'])} "
              f"service_fail={r['n_accept_service_fail']} label_missing={r['n_accept_label_missing']}")
    print(f"saved: {args.out}/margin_scan.json")


def _fmt(x):
    return 'nan' if x is None else f'{x:.4f}'


if __name__ == '__main__':
    main()
