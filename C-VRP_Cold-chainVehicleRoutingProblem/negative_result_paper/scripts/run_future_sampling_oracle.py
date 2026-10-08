"""②b · 未来信息 vs 评价时域：三评分固定状态对照（分布期望评分新增）。

在同一决策点状态 + 候选集上，对每个候选动作计算三种评分，比较三者选出动作的
真实终局增益，回答 H1（myopic 代理差、分布可学）vs H2（头腔依赖真实未来实现）：

  - ① myopic           ：J_vis（evaluate_visible_plan，无未来）
  - ② 分布期望 full    ：mean_n J_terminal(a | 重采样未来：属性 + reveal 时序)
  - ② 分布期望 attr    ：mean_n J_terminal(a | 重采样未来：仅属性，保留真实 reveal 时序)
  - ③ clairvoyant      ：J_terminal(a | 真实未来)（rollout_action）

判读（对每个状态，各 scorer 选出动作的真实终局增益 gain = J0 − J_terminal(pick)）：
  gain_dist ≈ gain_clairvoyant >> gain_myopic → H1（评价时域/代理差，分布可学）
  gain_dist ≈ gain_myopic       << gain_clairvoyant → H2（未来信息主导）

未来重采样：对未揭示客户 {c : reveal_time[c] > clock 且未 served}，按
generate_coldchain_data.generate_coldchain_instance 的生成分布重采样（demand 与生成器一致
= integers(1,11) = 1..10），就地改写 env.coords/demands/tw/service/temp_class/reveal_time
并重算 dist_mat 行/列。full 模式 reveal_time ~ U(clock, max(0, tw_end − dist_depot − service))
（拒绝早于 clock 的不可行 tw，重试 ≤200 次，失败则保留该客户原属性；attr 模式保留真实 reveal_time）。
**同一状态的所有候选动作共享同一组未来场景（common random numbers）**，避免候选间评分差异被
未来场景采样噪声污染。

用法（本地，纯 NumPy + JF1-H continuation）：
    PYTHONPATH=scripts python negative_result_paper/scripts/run_future_sampling_oracle.py \
        --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --max-instances 8 --snapshots-per-inst 2 --customers-per-snapshot 2 \
        --n-samples 4 --mode both --out negative_result_paper/results/future_sampling_oracle
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
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from action_contract import apply_action, enumerate_actions_from_plans
from coldchain_contract import default_pilot_contract, apply_objective_profile, ObjectiveProfile
from recourse_snapshot import capture_recourse_snapshot, resume_from_snapshot
from counterfactual_teacher import (_eval, _snapshot_with_force, rollout_baseline, rollout_action)
from sequential_oracle import customer_order_key, decision_pool, mutable_vehicle_ids
from run_causal_headroom import _incumbent_plans_and_vis, _myopic_J

TEMP_DIST = [0.4, 0.35, 0.25]
HORIZON = 24.0
DEPOT = np.array([0.5, 0.5], dtype=np.float32)


def _spearman(x, y):
    def _avg_rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind='mergesort')
        s = a[order]
        r = np.empty(len(a), dtype=float)
        i, n = 0, len(a)
        while i < n:
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float('nan')
    rx, ry = _avg_rank(x), _avg_rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _sample_future(env, inst_idx, clock, served_mask, mode, rng, counters=None):
    """采样一组未来属性（对未揭示客户按生成分布重采样）。

    返回 {customer_id: dict(coord, demand, temp, tw_start, tw_end, service, reveal)}。
    demand 与生成器一致（capacity=50 时 integers(1,11) = 1..10）。full 模式
    reveal ~ U(clock, max(0, tw_end − dist_depot − service))，拒绝早于 clock 的不可行
    tw（重试 ≤2000 次）；仍未命中则**整单保留原属性**（不写入 futures，_apply_future 对该
    客户零改动），杜绝"属性已重采样但 reveal 保留旧值"的不一致（投稿前审计 P0-2 修复）。
    attr 模式保留真实 reveal_time（设计行为）。counters 统计：n_unrevealed=未揭示客户数、
    n_exhausted_old=前 200 次全失败数（旧代码不一致路径触发数）、n_kept_whole=full 整单
    保留数。
    """
    N = env.num_nodes
    futures = {}
    if counters is None:
        counters = {'n_unrevealed': 0, 'n_exhausted_old': 0, 'n_kept_whole': 0}
    for c in range(1, N):
        if bool(served_mask[c]):
            continue
        if env.reveal_time[inst_idx, c] <= clock + 1e-6:
            continue  # 已揭示（过去）——固定，不重采样
        counters['n_unrevealed'] += 1
        coord = demand = temp = tw_start = tw_end = service = None
        reveal = None
        for attempt in range(2000):
            coord = rng.uniform(0.02, 0.98, 2).astype(np.float32)
            demand = float(rng.integers(1, 11))  # 生成器 capacity=50 时为 1..10
            temp = int(rng.choice(3, p=TEMP_DIST))
            dist_depot = float(np.hypot(coord[0] - DEPOT[0], coord[1] - DEPOT[1]))
            center = dist_depot * 10.0 + rng.uniform(0.0, HORIZON * 0.4)
            width = HORIZON * 0.2 * rng.uniform(0.5, 1.5)
            tw_start = max(0.0, center - width / 2)
            tw_end = min(HORIZON, center + width / 2)
            service = 0.1 + demand * 0.05
            if mode == 'full':
                max_reveal = max(0.0, tw_end - dist_depot - service)
                if max_reveal > clock + 1e-6:
                    reveal = rng.uniform(clock, max_reveal)
                    break
                if attempt == 199 and reveal is None:
                    counters['n_exhausted_old'] += 1
            else:
                break  # attr：单次采样，reveal=None 保留真实揭示时间（设计行为）
        if mode == 'full' and reveal is None:
            counters['n_kept_whole'] += 1
            continue  # 修复：full 拒绝采样未命中 → 整单保留（_apply_future 零改动）
        futures[c] = {'coord': coord, 'demand': demand, 'temp': temp,
                      'tw_start': tw_start, 'tw_end': tw_end, 'service': service,
                      'reveal': reveal}
    return futures


def _apply_future(env, inst_idx, futures):
    """把一组未来属性就地写入 env 并重算 dist_mat 行/列。"""
    N = env.num_nodes
    for c, f in futures.items():
        env.coords[inst_idx, c] = f['coord']
        env.demands[inst_idx, c] = f['demand']
        env.tw_start[inst_idx, c] = f['tw_start']
        env.tw_end[inst_idx, c] = f['tw_end']
        env.service_time[inst_idx, c] = f['service']
        env.temp_class[inst_idx, c] = f['temp']
        env.initial_quality[inst_idx, c] = 1.0
        if f['reveal'] is not None:
            env.reveal_time[inst_idx, c] = f['reveal']
        for j in range(N):
            d = float(np.hypot(env.coords[inst_idx, c, 0] - env.coords[inst_idx, j, 0],
                               env.coords[inst_idx, c, 1] - env.coords[inst_idx, j, 1]))
            env.dist_mat[inst_idx, c, j] = d
            env.dist_mat[inst_idx, j, c] = d


def _dist_expected_cost(dataset, env, snapshot, plans, action, continuation, contract,
                        capacity, num_vehicles, futures_samples):
    """候选动作在共享未来场景下的分布期望终局成本。返回 (mean, std)。

    futures_samples 是同一状态所有候选动作共享的 N 组未来（common random numbers），
    保证候选间评分差异不被"未来场景采样噪声"污染。每组未来用 dataset 副本建 env 后
    _apply_future 就地重写未揭示客户，避免污染共享 dataset。
    """
    inst_idx = int(snapshot['instance_id'])
    mutable_ids = mutable_vehicle_ids(snapshot)
    new_plans = apply_action(plans, action, allowed_vehicle_ids=mutable_ids)
    snap2 = _snapshot_with_force(env, snapshot, new_plans, mutable_ids=mutable_ids)
    costs = []
    for futures in futures_samples:
        dataset_n = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in dataset.items()}
        renv2 = StrictOnlineEnv(dataset_n, capacity, 1.0, num_vehicles,
                                replanner=make_continuation(), coldchain_contract=contract)
        _apply_future(renv2, inst_idx, futures)
        traces, served_mask = renv2.run_resumed(snap2)
        out = _eval(renv2, inst_idx, traces, 'coldchain', served_mask=served_mask)
        costs.append(float(out['coldchain_cost']))
    return float(np.mean(costs)), float(np.std(costs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=8)
    ap.add_argument('--snapshots-per-inst', type=int, default=2)
    ap.add_argument('--customers-per-snapshot', type=int, default=2)
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--mode', choices=['full', 'attr', 'both', 'none'], default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    with open(args.objective_profile) as f:
        d = json.load(f)
    profile = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                               quality_scale=float(d['quality_scale']),
                               energy_scale=float(d['energy_scale']),
                               lambda_quality=float(d['lambda_quality']),
                               lambda_energy=float(d['lambda_energy']),
                               scale_source=d.get('scale_source', 'pilot'),
                               dev_statistics=d.get('dev_statistics'))
    contract = apply_objective_profile(default_pilot_contract(), profile)
    objective = contract.objective
    rng = np.random.default_rng(args.seed)
    modes = ['full', 'attr'] if args.mode == 'both' else ([] if args.mode == 'none' else [args.mode])

    states = []
    counters_total = {'n_unrevealed': 0, 'n_exhausted_old': 0, 'n_kept_whole': 0}
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=make_continuation(), coldchain_contract=contract)
        snaps = []

        def hook(e, i, clk, eid, rid, veh, tr, sm, ac):
            if sum(1 for s in snaps if s['instance_id'] == int(i)) < args.snapshots_per_inst:
                snaps.append(capture_recourse_snapshot(e, i, clk, eid, rid, veh, tr, sm, ac))

        env.snapshot_hook = hook
        env.run(inst)

        for snap in snaps:
            renv = resume_from_snapshot(dataset, snap, make_continuation(), args.capacity,
                                        1.0, args.num_vehicles, coldchain_contract=contract)
            inst_idx = int(snap['instance_id'])
            plans, vis = _incumbent_plans_and_vis(renv, snap, renv.replanner, contract, objective)
            mutable_ids = mutable_vehicle_ids(snap)
            pool = sorted(decision_pool(snap), key=customer_order_key(renv, inst_idx))

            base = rollout_baseline(renv, snap, 'coldchain', objective)
            J0 = float(base['coldchain_cost'])
            J0_myopic = _myopic_J(vis, plans, contract, objective)

            # 先收集所有可行候选动作（未来场景按状态共享，需一次性枚举）
            cand_list = []
            for customer in pool[:args.customers_per_snapshot]:
                cands, _ = enumerate_actions_from_plans(renv, inst_idx, plans, customer,
                                                        allowed_vehicle_ids=mutable_ids)
                for cand in cands:
                    if cand.feasible:
                        cand_list.append((customer, cand))

            # 每个 mode 采样 N 组共享未来场景（common random numbers）
            clock = float(snap['clock'])
            served = snap['served_mask']
            futures_by_mode = {m: [_sample_future(renv, inst_idx, clock, served, m, rng,
                                                   counters_total)
                                   for _ in range(args.n_samples)]
                               for m in modes}

            rows = []
            for customer, cand in cand_list:
                new_plans = apply_action(plans, cand.action, allowed_vehicle_ids=mutable_ids)
                jm = _myopic_J(vis, new_plans, contract, objective)
                co, _ = rollout_action(renv, snap, cand.action, renv.replanner, plans,
                                       objective='coldchain',
                                       allowed_vehicle_ids=mutable_ids,
                                       mutable_ids=mutable_ids)
                cc = float(co['coldchain_cost'])
                row = {'customer': int(customer),
                       'action_id': cand.action.action_id(),
                       'myopic': jm, 'clairvoyant': cc}
                for m in modes:
                    mean, std = _dist_expected_cost(
                        dataset, renv, snap, plans, cand.action, renv.replanner,
                        contract, args.capacity, args.num_vehicles, futures_by_mode[m])
                    row[f'dist_{m}'] = mean
                    row[f'dist_{m}_std'] = std
                rows.append(row)

            def _pick(key):
                finite = [(i, r[key]) for i, r in enumerate(rows)
                          if r.get(key) is not None and np.isfinite(r[key])]
                if not finite:
                    return None
                return min(finite, key=lambda x: x[1])[0]

            pick = {k: _pick(k) for k in
                    ['myopic', 'clairvoyant'] + [f'dist_{m}' for m in modes]}
            gain = {}
            for k, idx in pick.items():
                gain[k] = (J0 - rows[idx]['clairvoyant']) if idx is not None else None

            states.append({'inst': inst_idx, 'event': int(snap['event_id']),
                           'J0': J0, 'J0_myopic': J0_myopic,
                           'n_candidates': len(rows), 'rows': rows,
                           'pick': pick, 'gain': gain})
            _fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
            print(f"  [inst {inst_idx}/evt {snap['event_id']}] n_cand={len(rows)} "
                  f"gain myopic={_fmt(gain['myopic'])} clair={_fmt(gain['clairvoyant'])} "
                  + ' '.join(f"{m}={_fmt(gain[f'dist_{m}'])}" for m in modes), flush=True)

    # 聚合：各 scorer 选出动作的真实终局增益（实例聚类 bootstrap）
    def _inst_agg(key):
        per = {}
        for s in states:
            v = s['gain'].get(key)
            if v is None:
                continue
            per.setdefault(s['inst'], []).append(v)
        return [float(np.mean(v)) for v in per.values()]

    def _ci(vals, n_boot=2000, seed=0):
        if not vals:
            return {'mean': None, 'ci_lo': None, 'ci_hi': None}
        mean = float(np.mean(vals))
        rngb = np.random.default_rng(seed)
        b = [float(np.mean([vals[i] for i in rngb.integers(0, len(vals), size=len(vals))]))
             for _ in range(n_boot)]
        return {'mean': mean, 'ci_lo': float(np.percentile(b, 2.5)),
                'ci_hi': float(np.percentile(b, 97.5))}

    # 评分一致性：dist vs clairvoyant、myopic vs clairvoyant（候选内 Spearman，pooled）
    def _pooled_spearman(key_a, key_b):
        xs, ys = [], []
        for s in states:
            for r in s['rows']:
                if (r.get(key_a) is not None and r.get(key_b) is not None
                        and np.isfinite(r[key_a]) and np.isfinite(r[key_b])):
                    xs.append(r[key_a])
                    ys.append(r[key_b])
        return _spearman(xs, ys)

    summary = {
        'mode': args.mode, 'n_samples': args.n_samples,
        'n_states': len(states), 'n_instances': len(set(s['inst'] for s in states)),
        'n_candidates_total': sum(s['n_candidates'] for s in states),
        'gain_mean': {'myopic': _ci(_inst_agg('myopic')),
                      'clairvoyant': _ci(_inst_agg('clairvoyant')),
                      **{f'dist_{m}': _ci(_inst_agg(f'dist_{m}')) for m in modes}},
        'spearman_vs_clairvoyant': {'myopic': _pooled_spearman('myopic', 'clairvoyant'),
                                    **{f'dist_{m}': _pooled_spearman(f'dist_{m}', 'clairvoyant')
                                       for m in modes}},
        'sampler_counters': counters_total,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(states, f, indent=2, default=str)

    print("\n=== future-info vs horizon (fixed-state 3-scorer) ===")
    print(json.dumps({'gain_mean': summary['gain_mean'],
                      'spearman_vs_clairvoyant': summary['spearman_vs_clairvoyant'],
                      'sampler_counters': summary['sampler_counters']},
                     indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
