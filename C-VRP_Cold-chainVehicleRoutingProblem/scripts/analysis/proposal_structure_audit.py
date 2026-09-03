"""
Proposal Structure Audit（R1.7-2）—— 分析 H2-G proposal trace 的候选结构。

回答：模型 candidate 相对 NN incumbent 是「精确复制」还是「等成本但结构不同的 alternative」？
拆开 94.4% 的「candidate 与 NN 局部 cost 相同」现象。

输入：`run_r1_5_model_utility.py --save_proposal_trace` 生成的 `proposal_trace.jsonl`
      （每 decision 一行 JSON，含 current_node + 完整 incumbent/candidate/executed route）。

用法:
    python scripts/analysis/proposal_structure_audit.py \
        --trace results/r1_5_model_utility/r1_7_h3g/proposal_trace.jsonl \
        --method model_guarded --eps 1e-6 \
        --out results/r1_7/proposal_structure
"""

import sys, os, argparse, csv, json
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, _CVRPTW)


# ── route helpers（route 均以 current_node 开头）────────────────────────────
def to_suffix(current_node, route):
    """suffix = route 去掉开头的 current_node（仅第一个，保留末尾 depot 0）。"""
    if route is None:
        return []
    nodes = [int(n) for n in route]
    if nodes and nodes[0] == int(current_node):
        nodes = nodes[1:]
    return nodes


def suffix_edges(current_node, route):
    """有向边集合（去掉开头 current_node，保留末尾 depot 返回边）。"""
    nodes = to_suffix(current_node, route)
    edges = set()
    prev = int(current_node)
    for n in nodes:
        edges.add((prev, n))
        prev = n
    return edges


def first_action(current_node, route):
    nodes = to_suffix(current_node, route)
    return nodes[0] if nodes else 0


def action_type(current_node, route):
    if route is None:
        return "NONE"
    nodes = to_suffix(current_node, route)
    if len(nodes) == 0:
        return "WAIT"
    if nodes[0] == 0:
        return "CLOSE"
    return "SERVE"


def num_planned_customers(current_node, route):
    return sum(1 for n in to_suffix(current_node, route) if n != 0)


def edge_jaccard(ei, ec):
    union = len(ei | ec)
    return len(ei & ec) / union if union > 0 else 1.0


# ── per-event metric ────────────────────────────────────────────────────────
def compute_event(t, eps):
    current = int(t['current_node'])
    inc = t['incumbent_route']
    cand = t['candidate_route']
    if cand is None:
        return None  # candidate_exists=False，不进结构审计

    inc = [int(n) for n in inc]
    cand = [int(n) for n in cand]
    ei = suffix_edges(current, inc)
    ec = suffix_edges(current, cand)

    inc_cost = float(t['inc_cost'])
    cand_cost = float(t['candidate_cost'])
    same_cost = abs(cand_cost - inc_cost) <= eps

    return {
        'instance_id': int(t['instance_id']),
        'event_id': int(t['event_id']),
        'vehicle_id': int(t['vehicle_id']),
        'clock': float(t['clock']),
        'replan_reason': t['replan_reason'],
        'current_node': current,
        'candidate_accepted': bool(t['candidate_accepted']),
        'selection_reason': t['selection_reason'],
        'inc_action_type': action_type(current, inc),
        'cand_action_type': action_type(current, cand),
        'inc_first': first_action(current, inc),
        'cand_first': first_action(current, cand),
        'first_equal': first_action(current, inc) == first_action(current, cand),
        'inc_cost': inc_cost,
        'cand_cost': cand_cost,
        'cost_delta': cand_cost - inc_cost,
        'same_cost': same_cost,
        'inc_n_planned': num_planned_customers(current, inc),
        'cand_n_planned': num_planned_customers(current, cand),
        'service_delta': num_planned_customers(current, cand) - num_planned_customers(current, inc),
        'exact_route_equal': inc == cand,
        'edge_jaccard': edge_jaccard(ei, ec),
        'same_cost_diff_route': same_cost and (inc != cand),
        'initial_actionable_count': int(t['initial_actionable_count']),
        'visible_pending_count': int(t['visible_pending_count']),
    }


# ── aggregate ───────────────────────────────────────────────────────────────
def _agg(rows):
    if not rows:
        return None
    return {
        'n': len(rows),
        'exact_route_equal_rate': float(np.mean([r['exact_route_equal'] for r in rows])),
        'first_action_equal_rate': float(np.mean([r['first_equal'] for r in rows])),
        'mean_edge_jaccard': float(np.mean([r['edge_jaccard'] for r in rows])),
        'median_edge_jaccard': float(np.median([r['edge_jaccard'] for r in rows])),
        'same_cost_rate': float(np.mean([r['same_cost'] for r in rows])),
        'same_cost_diff_route_rate': float(np.mean([r['same_cost_diff_route'] for r in rows])),
        'mean_service_delta': float(np.mean([r['service_delta'] for r in rows])),
        'more_service_rate': float(np.mean([r['service_delta'] > 0 for r in rows])),
        'less_service_rate': float(np.mean([r['service_delta'] < 0 for r in rows])),
        'same_service_rate': float(np.mean([r['service_delta'] == 0 for r in rows])),
    }


def _fmt(x):
    return 'nan' if x is None else f"{x:.4f}"


EVENT_COLUMNS = [
    'instance_id', 'event_id', 'vehicle_id', 'clock', 'replan_reason', 'current_node',
    'candidate_accepted', 'selection_reason',
    'inc_action_type', 'cand_action_type', 'inc_first', 'cand_first', 'first_equal',
    'inc_cost', 'cand_cost', 'cost_delta', 'same_cost',
    'inc_n_planned', 'cand_n_planned', 'service_delta',
    'exact_route_equal', 'edge_jaccard', 'same_cost_diff_route',
    'initial_actionable_count', 'visible_pending_count',
]

AGG_COLUMNS = [
    'group', 'n', 'exact_route_equal_rate', 'first_action_equal_rate',
    'mean_edge_jaccard', 'median_edge_jaccard', 'same_cost_rate',
    'same_cost_diff_route_rate', 'mean_service_delta',
    'more_service_rate', 'less_service_rate', 'same_service_rate',
]


def _write_agg_csv(path, groups):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(AGG_COLUMNS)
        for label, rows in groups:
            a = _agg(rows)
            if a is None:
                continue
            w.writerow([label, a['n'],
                        _fmt(a['exact_route_equal_rate']), _fmt(a['first_action_equal_rate']),
                        _fmt(a['mean_edge_jaccard']), _fmt(a['median_edge_jaccard']),
                        _fmt(a['same_cost_rate']), _fmt(a['same_cost_diff_route_rate']),
                        _fmt(a['mean_service_delta']),
                        _fmt(a['more_service_rate']), _fmt(a['less_service_rate']),
                        _fmt(a['same_service_rate'])])


def main():
    parser = argparse.ArgumentParser(description='R1.7-2 Proposal Structure Audit')
    parser.add_argument('--trace', type=str, required=True)
    parser.add_argument('--method', type=str, default='model_guarded',
                        help='要审计的 method（默认 model_guarded=H2-G）')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='same-cost 阈值')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    events = []
    with open(args.trace, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get('method') != args.method:
                continue
            if t.get('candidate_route') is None:
                continue  # candidate_exists=False
            e = compute_event(t, args.eps)
            if e is not None:
                events.append(e)

    os.makedirs(args.out, exist_ok=True)
    print(f"=== Proposal Structure Audit（{args.method}）===")
    print(f"  candidate_exists events = {len(events)}")

    if not events:
        print("  无 candidate 事件。检查 --trace 是否来自 --save_proposal_trace 运行。")
        return

    # event_level.csv
    event_path = os.path.join(args.out, 'event_level.csv')
    with open(event_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(EVENT_COLUMNS)
        for e in events:
            w.writerow([e['instance_id'], e['event_id'], e['vehicle_id'],
                        f"{e['clock']:.3f}", e['replan_reason'], e['current_node'],
                        int(e['candidate_accepted']), e['selection_reason'],
                        e['inc_action_type'], e['cand_action_type'],
                        e['inc_first'], e['cand_first'], int(e['first_equal']),
                        f"{e['inc_cost']:.4f}", f"{e['cand_cost']:.4f}",
                        f"{e['cost_delta']:+.4f}", int(e['same_cost']),
                        e['inc_n_planned'], e['cand_n_planned'], e['service_delta'],
                        int(e['exact_route_equal']), f"{e['edge_jaccard']:.4f}",
                        int(e['same_cost_diff_route']),
                        e['initial_actionable_count'], e['visible_pending_count']])

    # summary.csv
    summary_path = os.path.join(args.out, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(AGG_COLUMNS)
        a = _agg(events)
        w.writerow(['overall', a['n'],
                    _fmt(a['exact_route_equal_rate']), _fmt(a['first_action_equal_rate']),
                    _fmt(a['mean_edge_jaccard']), _fmt(a['median_edge_jaccard']),
                    _fmt(a['same_cost_rate']), _fmt(a['same_cost_diff_route_rate']),
                    _fmt(a['mean_service_delta']),
                    _fmt(a['more_service_rate']), _fmt(a['less_service_rate']),
                    _fmt(a['same_service_rate'])])

    # by_acceptance.csv
    accepted = [e for e in events if e['candidate_accepted']]
    rejected = [e for e in events if not e['candidate_accepted']]
    more_service_accepted = [e for e in accepted if e['selection_reason'] == 'more_service']
    lower_cost_accepted = [e for e in accepted if e['selection_reason'] == 'lower_cost_same_service']
    _write_agg_csv(os.path.join(args.out, 'by_acceptance.csv'), [
        ('all', events), ('accepted', accepted), ('rejected', rejected),
        ('more_service_accepted', more_service_accepted),
        ('lower_cost_accepted', lower_cost_accepted),
    ])

    # by_reason.csv
    _write_agg_csv(os.path.join(args.out, 'by_reason.csv'), [
        (r, [e for e in events if e['replan_reason'] == r])
        for r in ['initial', 'reveal', 'plan_exhaustion']
    ])

    # by_actionable_count.csv
    _write_agg_csv(os.path.join(args.out, 'by_actionable_count.csv'), [
        ('1', [e for e in events if e['initial_actionable_count'] == 1]),
        ('2-3', [e for e in events if 2 <= e['initial_actionable_count'] <= 3]),
        ('4-7', [e for e in events if 4 <= e['initial_actionable_count'] <= 7]),
        ('8+', [e for e in events if e['initial_actionable_count'] >= 8]),
    ])

    # by_pending_count.csv
    _write_agg_csv(os.path.join(args.out, 'by_pending_count.csv'), [
        ('1-5', [e for e in events if 1 <= e['visible_pending_count'] <= 5]),
        ('6-10', [e for e in events if 6 <= e['visible_pending_count'] <= 10]),
        ('11-20', [e for e in events if 11 <= e['visible_pending_count'] <= 20]),
        ('20+', [e for e in events if e['visible_pending_count'] >= 21]),
    ])

    # by_progress.csv（clock 分位，按实例内最大 clock 归一化）
    max_clock_by_inst = {}
    for e in events:
        iid = e['instance_id']
        max_clock_by_inst[iid] = max(max_clock_by_inst.get(iid, 0.0), e['clock'])
    _write_agg_csv(os.path.join(args.out, 'by_progress.csv'), [
        ('0-25%', [e for e in events if e['clock'] / max_clock_by_inst[e['instance_id']] < 0.25]),
        ('25-50%', [e for e in events if 0.25 <= e['clock'] / max_clock_by_inst[e['instance_id']] < 0.5]),
        ('50-75%', [e for e in events if 0.5 <= e['clock'] / max_clock_by_inst[e['instance_id']] < 0.75]),
        ('75-100%', [e for e in events if e['clock'] / max_clock_by_inst[e['instance_id']] >= 0.75]),
    ])

    print(f"  output dir: {args.out}")
    print(f"  files: event_level.csv / summary.csv / by_acceptance.csv / by_reason.csv "
          f"/ by_actionable_count.csv / by_pending_count.csv / by_progress.csv")


if __name__ == '__main__':
    main()
