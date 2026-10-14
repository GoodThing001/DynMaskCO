"""
Proposal Quality Audit (R1.6) — 分析 H2（raw）candidate_exists=True 事件。

回答：模型提出的 candidate 相对 NN incumbent 是 improve / neutral / harmful 各占多少？
按 4 个维度切片：replan_reason / initial_actionable_count / visible_pending_count / episode progress。

输入：`run_r1_5_model_utility.py` 输出的 `event_audit.csv`。
用法:
    python scripts/analysis/proposal_quality_audit.py \
        --audit_csv results/r1_5_model_utility/clean_seed42_val128/event_audit.csv \
        --method model --eps 1e-6
"""

import sys, os, argparse, csv
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)


def _stats(rows):
    """给定 candidate_exists 且 service-equivalent 的行，返回 improve/neutral/harm 统计。"""
    if not rows:
        return None
    deltas = np.array([r['dc'] for r in rows], dtype=np.float64)
    eps = rows[0]['eps']
    improve = float(np.mean(deltas < -eps))
    neutral = float(np.mean(np.abs(deltas) <= eps))
    harm = float(np.mean(deltas > eps))
    improve_deltas = deltas[deltas < -eps]
    harm_deltas = deltas[deltas > eps]
    return {
        'n': len(rows),
        'improve_rate': improve,
        'neutral_rate': neutral,
        'harm_rate': harm,
        'mean_dc': float(np.mean(deltas)),
        'median_dc': float(np.median(deltas)),
        'mean_improve': float(np.mean(improve_deltas)) if len(improve_deltas) else float('nan'),
        'mean_harm': float(np.mean(harm_deltas)) if len(harm_deltas) else float('nan'),
    }


def _print_block(title, s):
    if s is None:
        print(f"\n[{title}]  (无 service-equivalent candidate 事件)")
        return
    print(f"\n[{title}]  n={s['n']}")
    print(f"  improve={s['improve_rate']:.1%}  neutral={s['neutral_rate']:.1%}  "
          f"harm={s['harm_rate']:.1%}")
    print(f"  mean_ΔC={s['mean_dc']:+.4f}  median_ΔC={s['median_dc']:+.4f}")
    print(f"  mean_improve(ΔC<0)={s['mean_improve']:+.4f}  "
          f"mean_harm(ΔC>0)={s['mean_harm']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='R1.6 Proposal Quality Audit')
    parser.add_argument('--audit_csv', type=str, required=True)
    parser.add_argument('--method', type=str, default='model',
                        help='要审计的 method（默认 model=H2 raw）')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='neutral 阈值（浮点噪声大可 1e-5）')
    args = parser.parse_args()

    all_rows = []
    with open(args.audit_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['method'] != args.method:
                continue
            if r.get('candidate_exists') != '1':
                continue
            # service-equivalent：candidate 与 incumbent 服务客户数相同
            inc_n = int(r['inc_n_planned'])
            cand_n = int(r['candidate_n_planned'])
            if inc_n != cand_n:
                continue
            try:
                inc_cost = float(r['inc_cost'])
                cand_cost = float(r['candidate_cost'])
                clock = float(r['clock'])
                actionable = int(r['initial_actionable_count'])
                visible = int(r['visible_pending_count'])
            except (ValueError, KeyError, TypeError):
                continue
            if np.isnan(inc_cost) or np.isnan(cand_cost):
                continue
            all_rows.append({
                'dc': cand_cost - inc_cost,
                'replan_reason': r['replan_reason'],
                'actionable': actionable,
                'visible': visible,
                'clock': clock,
                'eps': args.eps,
            })

    print(f"=== Proposal Quality Audit（{args.method}，candidate_exists & service-equivalent）===")
    print(f"  eps={args.eps}  total_events={len(all_rows)}")

    if not all_rows:
        print("  无可用事件。检查 --audit_csv 是否来自 run_r1_5_model_utility.py 的新版 event_audit。")
        return

    max_clock = max(r['clock'] for r in all_rows)

    # 0. overall
    _print_block("overall", _stats(all_rows))

    # 1. by replan_reason
    print("\n--- 切片 1：by replan_reason ---")
    for reason in ['initial', 'reveal', 'plan_exhaustion']:
        rows = [r for r in all_rows if r['replan_reason'] == reason]
        _print_block(reason, _stats(rows))

    # 2. by initial_actionable_count
    print("\n--- 切片 2：by initial_actionable_count ---")
    for label, lo, hi in [('2-3', 2, 3), ('4-7', 4, 7), ('8+', 8, 10**9)]:
        rows = [r for r in all_rows if lo <= r['actionable'] <= hi]
        _print_block(label, _stats(rows))

    # 3. by visible_pending_count
    print("\n--- 切片 3：by visible_pending_count ---")
    for label, lo, hi in [('1-5', 1, 5), ('6-10', 6, 10), ('11-20', 11, 20), ('20+', 21, 10**9)]:
        rows = [r for r in all_rows if lo <= r['visible'] <= hi]
        _print_block(label, _stats(rows))

    # 4. by episode progress (clock 分位)
    print("\n--- 切片 4：by episode progress（clock 分位）---")
    for label, lo, hi in [('0-25%', 0.0, 0.25), ('25-50%', 0.25, 0.5),
                          ('50-75%', 0.5, 0.75), ('75-100%', 0.75, 1.01)]:
        rows = [r for r in all_rows if lo <= r['clock'] / max_clock < hi]
        _print_block(label, _stats(rows))


if __name__ == '__main__':
    main()
