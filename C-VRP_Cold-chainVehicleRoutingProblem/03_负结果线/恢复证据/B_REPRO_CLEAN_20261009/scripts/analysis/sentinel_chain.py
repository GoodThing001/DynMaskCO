"""
Sentinel Causal Chain —— 追踪 H3-G s43 唯一 incomplete instance 的 first-divergence 因果链。

导师 §七/§八：sentinel 要画一条「从哪一次 locally accepted recourse 开始，H3-G 走到最终
incomplete trajectory」的因果链，而不是只盯最终 unserved 出现的前一个 event。

输入：s43 的 event_audit.csv（含 inc_first/candidate_first/executed_first/selection_reason）。
输出：instance 98 的 divergence trace（每个「H3-G 与 NN 分叉」的决策点 + 后续 cascade）。

注意：完整 causal chain（具体哪个客户漏服务、哪辆车位置分叉）需 proposal_trace.jsonl（未同步），
本脚本先用 event_audit.csv 定位 first-divergence + 分叉序列。

用法:
    python scripts/analysis/sentinel_chain.py \
        --audit results/r1_5_model_utility/r1_7_h3g_s43_trace/event_audit.csv \
        --instance 98
"""

import sys, os, argparse, csv
from collections import Counter

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)


def main():
    parser = argparse.ArgumentParser(description='Sentinel causal chain (H3-G s43)')
    parser.add_argument('--audit', type=str, required=True)
    parser.add_argument('--instance', type=int, default=98)
    args = parser.parse_args()

    rows = []
    with open(args.audit, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r['instance_id']) == args.instance:
                rows.append(r)
    rows.sort(key=lambda r: (int(r['event_id']), int(r['vehicle_id'])))

    print(f"=== Sentinel instance {args.instance}（{len(rows)} decisions）===")

    # 1. 第一个「执行与 NN 分叉」的决策点（executed_first != inc_first）
    first_div = None
    for r in rows:
        if r['executed_first'] != r['inc_first']:
            first_div = r
            break
    if first_div:
        print(f"\n[first divergence] event={first_div['event_id']} clock={first_div['clock']} "
              f"vehicle={first_div['vehicle_id']} reason={first_div['replan_reason']} "
              f"sel={first_div['selection_reason']} accepted={first_div['candidate_accepted']}")
        print(f"  NN first={first_div['inc_first']} -> executed={first_div['executed_first']} "
              f"(cand first={first_div['candidate_first']})")
        print(f"  inc_cost={first_div['inc_cost']} cand_cost={first_div['candidate_cost']} "
              f"inc_n={first_div['inc_n_planned']} cand_n={first_div['candidate_n_planned']} "
              f"actionable={first_div['initial_actionable_count']}")

    # 2. 全部分叉序列（executed_first != inc_first）
    diverged = [r for r in rows if r['executed_first'] != r['inc_first']]
    print(f"\n[divergence trace] {len(diverged)}/{len(rows)} 个决策点执行了与 NN 不同的首动作：")
    for r in diverged[:30]:
        mark = "ACCEPT" if r['candidate_accepted'] == '1' else "reject"
        print(f"  e={r['event_id']:>3} clk={r['clock']:>7} v={r['vehicle_id']:>2} "
              f"{r['replan_reason']:<10} {r['selection_reason']:<20} {mark:<6} "
              f"NN {r['inc_first']:>2} -> exec {r['executed_first']:>2}")

    # 3. 接受原因分布（instance 98 内）
    accepted = [r for r in rows if r['candidate_accepted'] == '1']
    print(f"\n[accepted] {len(accepted)} accepted candidates in instance {args.instance}:")
    print(f"  selection_reason: {dict(Counter(r['selection_reason'] for r in accepted))}")
    print(f"  replan_reason:    {dict(Counter(r['replan_reason'] for r in accepted))}")

    # 4. 提示：完整因果链需 proposal_trace.jsonl（含 full route + current_node）
    print("\n[note] 完整 causal chain（具体哪个客户漏服务、车辆位置如何分叉）需 "
          "r1_7_h3g_s43_trace/proposal_trace.jsonl；当前 event_audit 只有 first-action + cost。")


if __name__ == '__main__':
    main()
