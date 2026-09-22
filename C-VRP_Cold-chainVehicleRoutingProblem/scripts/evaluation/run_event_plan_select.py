"""event_plan_v2：同状态选择能力表（固定 CAL/TRAIN baseline 状态，共享候选池与反事实标签）。

离线读取 export data.json（含 g_B 标签 + A/B 特征 + distance/proxy_J），对各选择器在
同一状态、同一候选池上比较选择收益 / regret / 有害接受 / KEEP 拒绝 / 输入碰撞。

选择器：
  KEEP        — 恒保持（gain=0）；
  S_distance  — 选完整计划距离最小者（含返仓段；更短才接受）；
  S_proxy     — 选可见物理投影 J 最小者（J(P0)-J(P) > 0 才接受）；
  A / B       — learned 评分器 s·(f(P)-f(P0)) > 0 选 argmax；
  后验最佳    — 用真实未来 g_B 的离线参照（max(0, max g_B)），不可在线。

TRAIN 与 CAL 分开报告；实例/事件平衡（逐事件一个 chosen gain，不做候选支配）。

用法：
    python scripts/evaluation/run_event_plan_select.py \
        --data results/m0_scale/event_plan_v2_export/data.json \
        --ckpt-A results/m0_scale/event_plan_v2_A_s42/model.ckpt \
        --ckpt-B results/m0_scale/event_plan_v2_B_s42/model.ckpt \
        --out results/m0_scale/select_train
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from event_plan_replanner import load_plan_scorer

_NEAR_ZERO = 1e-9


def _selector_choice(rows, selector, scorer):
    """对每个事件返回 chosen 候选的 g_B（KEEP=0）与 chosen_idx（None=KEEP）。"""
    chosen = []
    for row in rows:
        cands = row['cands']
        if not cands:
            chosen.append((0.0, None))
            continue
        gbs = np.asarray([c['gB'] for c in cands], np.float64)
        if selector == 'KEEP':
            chosen.append((0.0, None))
        elif selector == 'distance':
            ds = np.asarray([c['distance'] for c in cands], np.float64)
            keep_d = float(row['keep_distance'])
            i = int(np.argmin(ds))
            if ds[i] < keep_d - _NEAR_ZERO:
                chosen.append((float(gbs[i]), i))
            else:
                chosen.append((0.0, None))
        elif selector == 'proxy':
            js = np.asarray([c['proxy_J'] for c in cands], np.float64)
            keep_j = float(row['keep_proxy_J'])
            i = int(np.argmin(js))
            if js[i] < keep_j - _NEAR_ZERO:
                chosen.append((float(gbs[i]), i))
            else:
                chosen.append((0.0, None))
        elif selector in ('A', 'B'):
            key = 'plan_delta' if selector == 'A' else 'consequence'
            keep_key = 'keep_plan_delta' if selector == 'A' else 'keep_consequence'
            ctx = np.asarray(row['context'], np.float32)
            keep_feat = np.asarray(row[keep_key], np.float32)
            feats = np.stack([np.asarray(c[key], np.float32) for c in cands])
            raw = np.asarray(scorer.score(ctx, keep_feat, feats))
            gains = scorer.s * raw
            i = int(np.argmax(gains))
            if float(gains[i]) > _NEAR_ZERO:
                chosen.append((float(gbs[i]), i))
            else:
                chosen.append((0.0, None))
        else:
            raise ValueError(selector)
    return chosen


def _metrics(rows, chosen, name):
    n = len(rows)
    gains = np.asarray([g for g, _ in chosen], np.float64)
    bests = np.asarray([max(0.0, max([c['gB'] for c in r['cands']], default=0.0))
                        for r in rows], np.float64)
    n_accept = sum(1 for g, i in chosen if i is not None)
    n_harmful = sum(1 for g, i in chosen if i is not None and g < -_NEAR_ZERO)
    n_nearzero = sum(1 for g, i in chosen if i is not None and abs(g) <= _NEAR_ZERO)
    # KEEP 拒绝：后验最佳是 KEEP（bests==0）但选择器接受了非 KEEP，且该选择无益（g<=0）
    n_keep_reject = sum(1 for (g, i), b in zip(chosen, bests)
                        if b <= _NEAR_ZERO and i is not None and g <= _NEAR_ZERO)
    return {
        'name': name,
        'n_events': n,
        'mean_gain': float(gains.mean()),
        'mean_best': float(bests.mean()),
        'mean_regret': float((bests - gains).mean()),
        'n_accept': int(n_accept),
        'n_harmful': int(n_harmful),
        'n_nearzero': int(n_nearzero),
        'n_keep_reject': int(n_keep_reject),
        'harmful_rate': float(n_harmful / max(n, 1)),
        'nearzero_rate': float(n_nearzero / max(n, 1)),
    }


def _collisions(rows, representation):
    """A/B 表示输入碰撞：同一事件内输入（四舍五入）相同但 g_B 不同的候选组数。"""
    key = 'plan_delta' if representation == 'A' else 'consequence'
    groups = 0
    for row in rows:
        by_input = {}
        for c in row['cands']:
            k = tuple(round(float(x), 6) for x in c[key])
            by_input.setdefault(k, set()).add(round(float(c['gB']), 6))
        groups += sum(1 for v in by_input.values() if len(v) > 1)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt-A', default=None)
    ap.add_argument('--ckpt-B', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.data) as f:
        data = json.load(f)
    rows = data['rows']

    scorerA = load_plan_scorer(args.ckpt_A) if args.ckpt_A else None
    scorerB = load_plan_scorer(args.ckpt_B) if args.ckpt_B else None

    results = []
    results.append(_metrics(rows, _selector_choice(rows, 'KEEP', None), 'KEEP'))
    results.append(_metrics(rows, _selector_choice(rows, 'distance', None), 'S_distance'))
    results.append(_metrics(rows, _selector_choice(rows, 'proxy', None), 'S_proxy'))
    if scorerA is not None:
        results.append(_metrics(rows, _selector_choice(rows, 'A', scorerA), 'A_learned'))
    if scorerB is not None:
        results.append(_metrics(rows, _selector_choice(rows, 'B', scorerB), 'B_learned'))

    summary = {
        'n_events': len(rows),
        'collision_A': _collisions(rows, 'A'),
        'collision_B': _collisions(rows, 'B'),
        'selectors': results,
    }
    with open(os.path.join(args.out, 'selection.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"=== same-state selection (n_events={len(rows)}) ===")
    print(f"  collision A={summary['collision_A']} B={summary['collision_B']}")
    for r in results:
        print(f"  {r['name']:12s} gain={r['mean_gain']:+.4f} best={r['mean_best']:+.4f} "
              f"regret={r['mean_regret']:.4f} accept={r['n_accept']} "
              f"harmful={r['n_harmful']} nearzero={r['n_nearzero']} "
              f"keep_reject={r['n_keep_reject']}")
    print(f"saved: {args.out}/selection.json")


if __name__ == '__main__':
    main()
