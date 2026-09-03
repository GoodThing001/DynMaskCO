"""
R1.7-4 Guard Long-Horizon Audit —— fork incumbent vs candidate at decision states，
比较 long-horizon outcome（service-first lexicographic）。

回答：当前 local guard 的 accept/reject 是否真正转化为长期 episode 收益？

核心指标（区分 service harm vs cost harm）：
  - service harm：candidate 分支更多 unserved / TW / capacity / duplicate（严重 failure）
  - cost harm：service 相同但 candidate 分支 distance 更高（guard myopia）
  - Guard Regret：R_e = J(guard choice) - min(J_I, J_C)（service-equivalent distance）

支持 checkpoint + resume（每 N forks 落盘 progress.json），分层采样（--stratified N）。

用法:
    # 100-fork stratified validation
    python scripts/analysis/guard_long_horizon_audit.py \
        --data ... --ckpt ... --trace ... \
        --capacity 50 --beam_width 16 --decode_seed 42 \
        --stratified 100 --out results/r1_7/guard_long_horizon_100

    # 全量（checkpoint + resume）
    python scripts/analysis/guard_long_horizon_audit.py \
        --data ... --ckpt ... --trace ... \
        --capacity 50 --beam_width 16 --decode_seed 42 \
        --checkpoint_every 100 --out results/r1_7/guard_long_horizon
"""

import sys, os, argparse, csv, json, time
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'decoding'))

from online_decode import build_maskco_replanner
from strict_online_env import StrictOnlineEnv
from authoritative_evaluator import evaluate_execution_trace


def load_model(ckpt_path):
    from flax import nnx
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError(f"load_ckpt 返回空 params: {ckpt_path}")
    return nnx.merge(nnx.graphdef(model), params)


def route_to_suffix(current_node, route):
    return [int(n) for n in route if int(n) != int(current_node)]


def service_tuple(m):
    return (m['n_unserved'], m['tw_late_count'], m['n_duplicate'],
            int(not m['capacity_feasible']))


def detailed_compare(c, i, eps):
    """service-first 分层次比较。返回 service_better/service_worse/cost_better/cost_worse/equivalent。"""
    sc, si = service_tuple(c), service_tuple(i)
    if sc < si:
        return 'service_better'
    if sc > si:
        return 'service_worse'
    if c['distance_cost'] < i['distance_cost'] - eps:
        return 'cost_better'
    if c['distance_cost'] > i['distance_cost'] + eps:
        return 'cost_worse'
    return 'equivalent'


def guard_regret(c, i, accepted):
    """R_e = J(guard choice) - min(J_I, J_C)，service-equivalent distance 口径。"""
    if service_tuple(c) != service_tuple(i):
        return float('nan')
    if accepted:
        return max(0.0, c['distance_cost'] - i['distance_cost'])
    return max(0.0, i['distance_cost'] - c['distance_cost'])


def _bucket_actionable(n):
    if n <= 1:
        return '1'
    if n <= 3:
        return '2-3'
    if n <= 7:
        return '4-7'
    return '8+'


def _bucket_pending(n):
    if n <= 5:
        return '1-5'
    if n <= 10:
        return '6-10'
    if n <= 20:
        return '11-20'
    return '20+'


def stratified_sample(forks, n):
    """按 (accepted, actionable bucket) 分层 round-robin 采样 n 个。"""
    groups = {}
    for fk in forks:
        key = (fk['accepted'], _bucket_actionable(fk['initial_actionable_count']))
        groups.setdefault(key, []).append(fk)
    sampled, i = [], 0
    group_lists = [groups[k] for k in sorted(groups)]
    while len(sampled) < n:
        added = False
        for gl in group_lists:
            if i < len(gl):
                sampled.append(gl[i])
                added = True
            if len(sampled) >= n:
                break
        if not added:
            break
        i += 1
    return sampled


def main():
    parser = argparse.ArgumentParser(description='R1.7-4 Guard Long-Horizon Audit')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--trace', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=16)
    parser.add_argument('--decode_seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--max_forks', type=int, default=None)
    parser.add_argument('--stratified', type=int, default=None,
                        help='分层采样 n 个（覆盖 accepted/rejected × actionable bucket）')
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())
    model = load_model(args.ckpt)
    # H2-G policy；enable_local_counterfactual=False（fork 下游不需要 shuffle counterfactual，
    # 它只用于 local_first_divergence 审计，不影响 executed trajectory，省一半 compute）。
    replanner = build_maskco_replanner(
        dataset, args.capacity, model, tw_max, 1.0, args.beam_width,
        decode_seed=args.decode_seed, incumbent_builder='nn', logit_mode='real',
        enable_local_counterfactual=False, selection_mode='guarded')
    env = StrictOnlineEnv(dataset, args.capacity, tw_speed=1.0, num_vehicles=25,
                          replanner=replanner)

    def _run_branch(inst_idx, force_suffix):
        replanner.reset_audit()
        traces, served_mask = env.run(inst_idx, force_suffix=force_suffix)
        return evaluate_execution_trace(
            traces, env.coords[inst_idx], env.tw_start[inst_idx], env.tw_end[inst_idx],
            env.service_time[inst_idx], env.demands[inst_idx], env.capacity,
            speed=env.tw_speed, dist_mat=env.dist_mat[inst_idx])

    # 读 trace → fork states
    forks = []
    bad_lines = 0
    with open(args.trace, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue  # 跳过损坏行（服务器 trace 可能尾部截断），不整跑崩溃
            if t.get('candidate_route') is None:
                continue
            inc_suffix = route_to_suffix(t['current_node'], t['incumbent_route'])
            cand_suffix = route_to_suffix(t['current_node'], t['candidate_route'])
            accepted = bool(t['candidate_accepted'])
            if not accepted and inc_suffix == cand_suffix:
                continue  # exact NN copy，跳过
            forks.append({
                'instance_id': int(t['instance_id']),
                'event_id': int(t['event_id']),
                'vehicle_id': int(t['vehicle_id']),
                'replan_reason': t['replan_reason'],
                'selection_reason': t['selection_reason'],
                'accepted': accepted,
                'incumbent_suffix': inc_suffix,
                'candidate_suffix': cand_suffix,
                'initial_actionable_count': int(t['initial_actionable_count']),
                'visible_pending_count': int(t['visible_pending_count']),
                'clock': float(t['clock']),
            })

    if args.stratified is not None:
        forks = stratified_sample(forks, args.stratified)
    elif args.max_forks is not None:
        forks = forks[:args.max_forks]

    os.makedirs(args.out, exist_ok=True)
    fork_csv = os.path.join(args.out, 'fork_level.csv')
    progress_json = os.path.join(args.out, 'progress.json')

    start_idx = 0
    if args.resume and os.path.exists(progress_json):
        start_idx = json.load(open(progress_json, encoding='utf-8'))['completed']
        print(f"resume from fork index {start_idx}")

    print(f"=== R1.7-4 Guard Long-Horizon Audit ===")
    print(f"  fork states = {len(forks)}（start={start_idx}），skipped malformed lines = {bad_lines}")

    FORK_COLS = ['instance_id', 'event_id', 'vehicle_id', 'replan_reason', 'selection_reason',
                 'accepted', 'initial_actionable_count', 'visible_pending_count', 'clock',
                 'i_unserved', 'c_unserved', 'i_distance', 'c_distance', 'distance_delta',
                 'verdict', 'guard_regret', 'runtime_i_s', 'runtime_c_s']

    mode = 'a' if (args.resume and os.path.exists(fork_csv)) else 'w'
    with open(fork_csv, mode, newline='') as f:
        w = csv.writer(f)
        if mode == 'w':
            w.writerow(FORK_COLS)

    n_done = 0
    t_start = time.time()
    for k in range(start_idx, len(forks)):
        fk = forks[k]
        key = (fk['event_id'], fk['vehicle_id'])

        t0 = time.time()
        m_i = _run_branch(fk['instance_id'], {key: fk['incumbent_suffix']})
        rt_i = time.time() - t0

        t0 = time.time()
        m_c = _run_branch(fk['instance_id'], {key: fk['candidate_suffix']})
        rt_c = time.time() - t0

        verdict = detailed_compare(m_c, m_i, args.eps)
        regret = guard_regret(m_c, m_i, fk['accepted'])

        with open(fork_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                fk['instance_id'], fk['event_id'], fk['vehicle_id'], fk['replan_reason'],
                fk['selection_reason'], int(fk['accepted']),
                fk['initial_actionable_count'], fk['visible_pending_count'],
                f"{fk['clock']:.3f}", m_i['n_unserved'], m_c['n_unserved'],
                f"{m_i['distance_cost']:.4f}", f"{m_c['distance_cost']:.4f}",
                f"{m_c['distance_cost'] - m_i['distance_cost']:+.4f}",
                verdict, _fmt(regret), f"{rt_i:.2f}", f"{rt_c:.2f}"])

        n_done += 1
        if (k + 1) % args.checkpoint_every == 0:
            json.dump({'completed': k + 1, 'total': len(forks), 'elapsed_s': time.time() - t_start},
                      open(progress_json, 'w', encoding='utf-8'))
            print(f"  checkpoint: {k+1}/{len(forks)} done, "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    json.dump({'completed': len(forks), 'total': len(forks), 'elapsed_s': time.time() - t_start},
              open(progress_json, 'w', encoding='utf-8'))
    print(f"  done: {n_done} forks, elapsed={time.time()-t_start:.0f}s")

    # 聚合（从 fork_level.csv 读）
    rows = []
    with open(fork_csv, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def _agg(rs):
        if not rs:
            return None
        n = len(rs)
        verdicts = [r['verdict'] for r in rs]
        regrets = [float(r['guard_regret']) for r in rs if r['guard_regret'] not in ('nan', '')]
        return {
            'n': n,
            'beneficial_rate': float(np.mean([v in ('service_better', 'cost_better') for v in verdicts])),
            'harmful_rate': float(np.mean([v in ('service_worse', 'cost_worse') for v in verdicts])),
            'service_worse_rate': float(np.mean([v == 'service_worse' for v in verdicts])),
            'cost_worse_rate': float(np.mean([v == 'cost_worse' for v in verdicts])),
            'equivalent_rate': float(np.mean([v == 'equivalent' for v in verdicts])),
            'mean_guard_regret': float(np.mean(regrets)) if regrets else float('nan'),
            'p95_guard_regret': float(np.percentile(regrets, 95)) if regrets else float('nan'),
            'max_guard_regret': float(np.max(regrets)) if regrets else float('nan'),
        }

    COLS = ['group', 'n', 'beneficial_rate', 'harmful_rate', 'service_worse_rate',
            'cost_worse_rate', 'equivalent_rate', 'mean_guard_regret',
            'p95_guard_regret', 'max_guard_regret']

    def _write(path, groups):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for label, rs in groups:
                a = _agg(rs)
                if a is None:
                    continue
                w.writerow([label, a['n'], f"{a['beneficial_rate']:.4f}", f"{a['harmful_rate']:.4f}",
                            f"{a['service_worse_rate']:.4f}", f"{a['cost_worse_rate']:.4f}",
                            f"{a['equivalent_rate']:.4f}", f"{a['mean_guard_regret']:.4f}",
                            f"{a['p95_guard_regret']:.4f}", f"{a['max_guard_regret']:.4f}"])

    accepted = [r for r in rows if r['accepted'] == '1']
    rejected = [r for r in rows if r['accepted'] == '0']
    _write(os.path.join(args.out, 'summary.csv'), [
        ('overall', rows), ('accepted', accepted), ('rejected', rejected),
    ])
    _write(os.path.join(args.out, 'by_selection_reason.csv'), [
        ('more_service', [r for r in rows if r['selection_reason'] == 'more_service']),
        ('lower_cost', [r for r in rows if r['selection_reason'] == 'lower_cost_same_service']),
        ('not_better', [r for r in rows if r['selection_reason'] == 'not_better']),
        ('less_service', [r for r in rows if r['selection_reason'] == 'less_service']),
    ])
    _write(os.path.join(args.out, 'by_reason.csv'), [
        (r_, [x for x in rows if x['replan_reason'] == r_])
        for r_ in ['initial', 'reveal', 'plan_exhaustion']
    ])
    _write(os.path.join(args.out, 'by_actionable_count.csv'), [
        ('1', [x for x in rows if _bucket_actionable(int(x['initial_actionable_count'])) == '1']),
        ('2-3', [x for x in rows if _bucket_actionable(int(x['initial_actionable_count'])) == '2-3']),
        ('4-7', [x for x in rows if _bucket_actionable(int(x['initial_actionable_count'])) == '4-7']),
        ('8+', [x for x in rows if _bucket_actionable(int(x['initial_actionable_count'])) == '8+']),
    ])
    _write(os.path.join(args.out, 'by_pending_count.csv'), [
        ('1-5', [x for x in rows if _bucket_pending(int(x['visible_pending_count'])) == '1-5']),
        ('6-10', [x for x in rows if _bucket_pending(int(x['visible_pending_count'])) == '6-10']),
        ('11-20', [x for x in rows if _bucket_pending(int(x['visible_pending_count'])) == '11-20']),
        ('20+', [x for x in rows if _bucket_pending(int(x['visible_pending_count'])) == '20+']),
    ])

    print(f"  output dir: {args.out}")
    print(f"  files: fork_level.csv / summary.csv / by_selection_reason.csv / by_reason.csv "
          f"/ by_actionable_count.csv / by_pending_count.csv / progress.json")


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'nan'
    return f"{x:.4f}"


if __name__ == '__main__':
    main()
