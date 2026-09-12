"""
P0-R3 统一 service-first Gate 统计（02 手册 §P0-R3 / evaluation_contract §1-2）。

所有 comparator 结果必须：
  1. 先输出 per-instance service matrix（complete / unserved / duplicate / TW / cap / return）；
  2. paired cost 只对「同一实例且同一 hard outcome（service-equivalent）」的多路计算；
  3. incomplete / hard-infeasible 的方法不进 complete cost ranking；
  4. 所有 delta 使用同一 paired subset；
  5. 同时输出 unconditional lexicographic win/tie/loss。

本模块是 B0 / VAL32 comparator 的共享统计原语；不依赖 JAX / OR-Tools。
"""
import hashlib
import numpy as np

# 与 evaluate_execution_trace 返回 dict 的硬指标字段一一对应。
HARD_OUTCOME_FIELDS = ['n_unserved', 'n_duplicate', 'tw_feasible',
                       'capacity_feasible', 'depot_return_feasible']


def hard_outcome(m):
    """lexicographic hard outcome（service-first，小 = 好）。

    (unserved, duplicate, tw_viol, cap_viol, return_viol)。complete 由前两项决定，
    故不单独编码。用于 service-equivalent 配对。
    """
    return (
        int(m['n_unserved']),
        int(m['n_duplicate']),
        int(not m['tw_feasible']),
        int(not m['capacity_feasible']),
        int(not m['depot_return_feasible']),
    )


def service_equivalent_idx(results, methods, n):
    """返回所有方法 hard outcome 完全一致的实例索引（service-equivalent subset）。"""
    idx = []
    for i in range(n):
        base = hard_outcome(results[methods[0]][i])
        if all(hard_outcome(results[m][i]) == base for m in methods[1:]):
            idx.append(i)
    return idx


def paired_bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05):
    """逐实例 delta/interaction 的 paired bootstrap CI（mean）。"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, size=values.size, replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def win_tie_loss(deltas, tie_margin=1e-9):
    """unconditional win/tie/loss（deltas = a - b；负 = a 赢，正 = a 输）。"""
    w = t = l = 0
    for d in deltas:
        if d < -tie_margin:
            w += 1
        elif d > tie_margin:
            l += 1
        else:
            t += 1
    return w, t, l


def _set_hash(s):
    """确定性 set hash（sorted + sha256 前 16 hex），不依赖 PYTHONHASHSEED。"""
    return hashlib.sha256(repr(tuple(sorted(int(x) for x in s))).encode()).hexdigest()[:16]


def aggregate_solver_status(log, inst_idx):
    """把 replanner 的 coverage log 聚合成 per-instance 摘要（通用：TSPTW / greedy-seq 任一 arm）。

    returned_set 语义 = 该 arm 最终 plan 实际 route 的客户集（TSPTW 最优序，或 greedy 幸存集）。
    TSPTW arm 在 unavailable 时 fallback 到 greedy，客户仍被服务 → returned_set == assigned_set。
    因此 n_dropped 只统计「assigned 但未被该 arm route」的客户（greedy-seq 的 silent drop）。

    n_partial = returned_set != assigned_set 的调用数（TSPTW arm 应为 0；greedy-seq arm 记录真实 drop）。
    """
    entries = [e for e in log if e['inst_idx'] == int(inst_idx)]
    n = len(entries)
    assigned_union = set().union(*(e['assigned_set'] for e in entries)) if entries else set()
    returned_union = set().union(*(e['returned_set'] for e in entries)) if entries else set()

    n_optimal = sum(1 for e in entries if e['status'] == 'optimal')
    n_unavailable = sum(1 for e in entries if e['status'].startswith('unavailable'))
    n_k_over8 = sum(1 for e in entries if e['status'] == 'unavailable_k_over8')
    n_infeasible = sum(1 for e in entries if e['status'] == 'unavailable_infeasible')
    n_empty = sum(1 for e in entries if e['status'] == 'empty')
    n_partial = sum(1 for e in entries if set(e['returned_set']) != set(e['assigned_set']))
    n_dropped = len(assigned_union - returned_union)

    if n == 0:
        solver_status = 'no_solves'
    elif n_dropped == 0:
        solver_status = 'no_drop'
    else:
        solver_status = 'has_drop'

    coverage = (len(returned_union) / len(assigned_union)) if assigned_union else float('nan')
    return {
        'solver_status': solver_status,
        'n_solve_calls': n,
        'n_optimal': n_optimal,
        'n_unavailable': n_unavailable,
        'n_k_over8': n_k_over8,
        'n_infeasible': n_infeasible,
        'n_empty': n_empty,
        'n_partial': n_partial,
        'n_dropped': n_dropped,
        'assigned_union_hash': _set_hash(assigned_union),
        'returned_union_hash': _set_hash(returned_union),
        'coverage': float(coverage),
    }


def write_service_matrix(path, results, methods, n):
    """写 per-instance service matrix（02 §P0-R2 要求列，先 service 后 cost）。"""
    import csv, os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cols = ['instance_id', 'method', 'complete', 'n_unserved', 'n_duplicate',
            'tw_feasible', 'capacity_feasible', 'depot_return_feasible',
            'distance_cost', 'vehicle_count', 'runtime']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            for m in methods:
                r = results[m][i]
                w.writerow([
                    i, m,
                    int(r['complete']), int(r['n_unserved']), int(r['n_duplicate']),
                    int(r['tw_feasible']), int(r['capacity_feasible']),
                    int(r['depot_return_feasible']),
                    f"{r['distance_cost']:.4f}", int(r['vehicle_count']),
                    f"{r.get('runtime', float('nan')):.4f}",
                ])


def write_solver_status_csv(path, log, n, method='TSPTW'):
    """写 per-instance solver coverage CSV（某 comparator arm 的 assigned/returned 覆盖）。"""
    import csv, os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cols = ['instance_id', 'method', 'solver_status', 'n_solve_calls', 'n_optimal', 'n_unavailable',
            'n_k_over8', 'n_infeasible', 'n_empty', 'n_partial', 'n_dropped',
            'assigned_union_hash', 'returned_union_hash', 'coverage']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            s = aggregate_solver_status(log, i)
            w.writerow([
                i, method, s['solver_status'], s['n_solve_calls'], s['n_optimal'],
                s['n_unavailable'], s['n_k_over8'], s['n_infeasible'], s['n_empty'],
                s['n_partial'], s['n_dropped'], s['assigned_union_hash'],
                s['returned_union_hash'], f"{s['coverage']:.4f}",
            ])
