"""
R1.7-3 State / Leverage Analysis —— 模型在哪类状态最强、哪类最易犯错。

基于 R1.7-2 的 event_level.csv，做组合切片（acceptance × improve/harm × structure divergence），
并做高杠杆集中度（accepted events 的收益贡献 Top 1%/5%/10% 占比）。

回答（导师 §20）：
  模型的价值是否集中在少数 high-branching / reveal-heavy / high-pending 状态？
  这直接指导 B1 expert dataset oversampling / directional scorer / preference hard negatives。

用法:
    python scripts/analysis/proposal_state_leverage_audit.py \
        --event_level results/r1_7/proposal_structure/event_level.csv \
        --eps 1e-6 --out results/r1_7/state_leverage
"""

import sys, os, argparse, csv
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)


def _read_events(path):
    events = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            e = {
                'instance_id': int(r['instance_id']),
                'clock': float(r['clock']),
                'replan_reason': r['replan_reason'],
                'candidate_accepted': int(r['candidate_accepted']),
                'selection_reason': r['selection_reason'],
                'inc_cost': float(r['inc_cost']),
                'cand_cost': float(r['cand_cost']),
                'cost_delta': float(r['cost_delta']),
                'service_delta': int(r['service_delta']),
                'exact_route_equal': int(r['exact_route_equal']),
                'edge_jaccard': float(r['edge_jaccard']),
                'initial_actionable_count': int(r['initial_actionable_count']),
                'visible_pending_count': int(r['visible_pending_count']),
            }
            events.append(e)
    return events


def _slice_stats(rows, eps):
    if not rows:
        return None
    n = len(rows)
    same_service = [e for e in rows if e['service_delta'] == 0]
    improve = [e for e in same_service if e['cost_delta'] < -eps]
    harm = [e for e in same_service if e['cost_delta'] > eps]
    neutral = [e for e in same_service if abs(e['cost_delta']) <= eps]
    return {
        'n': n,
        'acceptance_rate': float(np.mean([e['candidate_accepted'] for e in rows])),
        'improve_rate': len(improve) / n if n else 0.0,
        'harm_rate': len(harm) / n if n else 0.0,
        'neutral_rate': len(neutral) / n if n else 0.0,
        'more_service_rate': float(np.mean([e['service_delta'] > 0 for e in rows])),
        'less_service_rate': float(np.mean([e['service_delta'] < 0 for e in rows])),
        'structure_divergence_rate': float(np.mean([1 - e['exact_route_equal'] for e in rows])),
        'mean_edge_jaccard': float(np.mean([e['edge_jaccard'] for e in rows])),
    }


COLS = ['group', 'n', 'acceptance_rate', 'improve_rate', 'harm_rate', 'neutral_rate',
        'more_service_rate', 'less_service_rate', 'structure_divergence_rate',
        'mean_edge_jaccard']


def _write_table(path, groups):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for label, rows in groups:
            s = _slice_stats(rows, 1e-6)
            if s is None:
                continue
            w.writerow([label, s['n'],
                        f"{s['acceptance_rate']:.4f}", f"{s['improve_rate']:.4f}",
                        f"{s['harm_rate']:.4f}", f"{s['neutral_rate']:.4f}",
                        f"{s['more_service_rate']:.4f}", f"{s['less_service_rate']:.4f}",
                        f"{s['structure_divergence_rate']:.4f}",
                        f"{s['mean_edge_jaccard']:.4f}"])


def _high_leverage(events):
    """高杠杆集中度：lower_cost accepted 的局部收益贡献 Top 1%/5%/10% 占比。"""
    lc = [e for e in events
          if e['candidate_accepted'] and e['selection_reason'] == 'lower_cost_same_service']
    if not lc:
        return None
    gains = sorted([(e['inc_cost'] - e['cand_cost']) for e in lc], reverse=True)
    total = sum(gains)
    def cum(pct):
        k = max(1, int(np.ceil(len(gains) * pct)))
        return sum(gains[:k]) / total if total > 0 else float('nan')
    return {
        'n_lower_cost': len(lc),
        'total_gain': total,
        'top1_pct': cum(0.01),
        'top5_pct': cum(0.05),
        'top10_pct': cum(0.10),
    }


def main():
    parser = argparse.ArgumentParser(description='R1.7-3 State/Leverage Analysis')
    parser.add_argument('--event_level', type=str, required=True)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    events = _read_events(args.event_level)
    os.makedirs(args.out, exist_ok=True)
    print(f"=== R1.7-3 State/Leverage Analysis ===")
    print(f"  total candidate events = {len(events)}")

    # 1. by replan_reason
    _write_table(os.path.join(args.out, 'by_reason.csv'), [
        (r, [e for e in events if e['replan_reason'] == r])
        for r in ['initial', 'reveal', 'plan_exhaustion']
    ])

    # 2. by actionable count
    _write_table(os.path.join(args.out, 'by_actionable_count.csv'), [
        ('1', [e for e in events if e['initial_actionable_count'] == 1]),
        ('2-3', [e for e in events if 2 <= e['initial_actionable_count'] <= 3]),
        ('4-7', [e for e in events if 4 <= e['initial_actionable_count'] <= 7]),
        ('8+', [e for e in events if e['initial_actionable_count'] >= 8]),
    ])

    # 3. by visible pending
    _write_table(os.path.join(args.out, 'by_pending_count.csv'), [
        ('1-5', [e for e in events if 1 <= e['visible_pending_count'] <= 5]),
        ('6-10', [e for e in events if 6 <= e['visible_pending_count'] <= 10]),
        ('11-20', [e for e in events if 11 <= e['visible_pending_count'] <= 20]),
        ('20+', [e for e in events if e['visible_pending_count'] >= 21]),
    ])

    # 4. by episode progress（按实例内最大 clock 归一化）
    max_clock = {}
    for e in events:
        iid = e['instance_id']
        max_clock[iid] = max(max_clock.get(iid, 0.0), e['clock'])
    _write_table(os.path.join(args.out, 'by_progress.csv'), [
        ('0-25%', [e for e in events if e['clock'] / max_clock[e['instance_id']] < 0.25]),
        ('25-50%', [e for e in events if 0.25 <= e['clock'] / max_clock[e['instance_id']] < 0.5]),
        ('50-75%', [e for e in events if 0.5 <= e['clock'] / max_clock[e['instance_id']] < 0.75]),
        ('75-100%', [e for e in events if e['clock'] / max_clock[e['instance_id']] >= 0.75]),
    ])

    # 5. by selection reason（accepted vs rejected 的结构差异）
    _write_table(os.path.join(args.out, 'by_selection_reason.csv'), [
        ('accepted', [e for e in events if e['candidate_accepted']]),
        ('more_service', [e for e in events if e['selection_reason'] == 'more_service']),
        ('lower_cost', [e for e in events if e['selection_reason'] == 'lower_cost_same_service']),
        ('rejected', [e for e in events if not e['candidate_accepted']]),
    ])

    # 6. 高杠杆集中度（lower_cost accepted 的局部收益）
    hl = _high_leverage(events)
    hl_path = os.path.join(args.out, 'high_leverage.csv')
    with open(hl_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        if hl is None:
            w.writerow(['n_lower_cost', 0])
        else:
            w.writerow(['n_lower_cost', hl['n_lower_cost']])
            w.writerow(['total_local_gain', f"{hl['total_gain']:.4f}"])
            w.writerow(['top1_pct_gain', f"{hl['top1_pct']:.4f}"])
            w.writerow(['top5_pct_gain', f"{hl['top5_pct']:.4f}"])
            w.writerow(['top10_pct_gain', f"{hl['top10_pct']:.4f}"])

    print(f"  output dir: {args.out}")
    print(f"  files: by_reason / by_actionable_count / by_pending_count / by_progress / "
          f"by_selection_reason / high_leverage")


if __name__ == '__main__':
    main()
