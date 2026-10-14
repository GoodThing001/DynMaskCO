"""行为汇总：从 ONE/FULL gate 的 instances/*.json 聚合决策日志（⑤ 归因诊断第四步）。

统计（每个 FULL 策略）：
  - 接受数、被接受的不同客户数、评分拒绝/认证拒绝数；
  - 同客户跨事件重复迁移次数（同一 customer 被 accept >1 次）；
  - 接受但当前计划距离增加的比例（预测改善 vs 实际 d_plan）；
  - 认证拒绝原因直方图；
  - event_plans 中 learned 修改未持久化的比例（learned_after != baseline_after 且下一事件 baseline_before == 该事件 baseline_after）。

用法（服务器）：
    python scripts/evaluation/run_behavior_summary.py --gate-dir results/m0_scale/onefull_F_H_s42 \
        --out results/m0_scale/onefull_F_H_s42/behavior_summary.json
"""
import argparse
import json
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    insts = sorted(os.listdir(os.path.join(args.gate_dir, 'instances')))
    agg = {'n_instances': 0, 'n_accept': 0, 'n_score_reject': 0, 'n_cert_reject': 0,
           'n_distinct_customers': 0, 'customer_accept_counts': {},
           'n_accept_dist_increased': 0, 'n_accept_with_dplan': 0,
           'cert_reject_reasons': Counter(), 'n_learned_change_unpersisted': 0,
           'n_learned_change': 0, 'n_accept_customers_repeat': 0}
    per_inst = []
    for fn in insts:
        with open(os.path.join(args.gate_dir, 'instances', fn)) as f:
            d = json.load(f)
        log = d.get('full_decision_log', [])
        ev = d.get('full_event_plans', [])
        accepts = [r for r in log if r['result'] == 'accept']
        cust_counts = Counter(r['customer'] for r in accepts)
        agg['n_instances'] += 1
        agg['n_accept'] += len(accepts)
        agg['n_score_reject'] += sum(1 for r in log if r['result'] == 'score_reject')
        agg['n_cert_reject'] += sum(1 for r in log if r['result'] == 'cert_reject')
        agg['n_distinct_customers'] += len(cust_counts)
        for c, k in cust_counts.items():
            agg['customer_accept_counts'][str(c)] = agg['customer_accept_counts'].get(str(c), 0) + k
            if k > 1:
                agg['n_accept_customers_repeat'] += 1
        for r in accepts:
            if 'd_plan_before' in r and 'd_plan_after' in r:
                agg['n_accept_with_dplan'] += 1
                if r['d_plan_after'] > r['d_plan_before'] + 1e-9:
                    agg['n_accept_dist_increased'] += 1
        for r in log:
            if r['result'] == 'cert_reject' and 'reject_reasons' in r:
                for k in r['reject_reasons']:
                    agg['cert_reject_reasons'][k] += 1
        # learned 修改持久化：某事件 learned_after != baseline_after，且下一事件 baseline_before == 该事件 baseline_after
        for j in range(len(ev)):
            if ev[j]['learned_after'] != ev[j]['baseline_after']:
                agg['n_learned_change'] += 1
                if j + 1 < len(ev) and ev[j + 1]['baseline_before'] == ev[j]['baseline_after']:
                    agg['n_learned_change_unpersisted'] += 1
        per_inst.append({'inst': d.get('inst_idx'), 'n_accept': len(accepts),
                         'n_distinct': len(cust_counts),
                         'n_cert_reject': agg['n_cert_reject']})

    agg['cert_reject_reasons'] = dict(agg['cert_reject_reasons'])
    agg['accept_dist_increased_frac'] = (agg['n_accept_dist_increased'] / agg['n_accept_with_dplan']
                                         if agg['n_accept_with_dplan'] else None)
    agg['learned_change_unpersisted_frac'] = (agg['n_learned_change_unpersisted'] / agg['n_learned_change']
                                              if agg['n_learned_change'] else None)
    agg['repeat_customer_frac'] = (agg['n_accept_customers_repeat'] / agg['n_distinct_customers']
                                   if agg['n_distinct_customers'] else None)
    agg['per_instance'] = per_inst
    with open(args.out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != 'per_instance'}, indent=2))
    print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
