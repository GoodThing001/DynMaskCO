"""③ 固定状态 SGBS 公平对照：单次 mask=2 修复，多种提议 + 有限搜索。

方法（每个决策点同一公开状态、同一 mask）：
  - R：regret-2 单次重建（参照）；
  - Mpre_greedy / Mtrained_greedy：确定性 argmax（1 次补全）；
  - Mpre_sgbs / Mtrained_sgbs / dist_sgbs：beam=2 分支 + 贪心补全（≤3 次补全）；
  - Mtrained_greedy7：greedy + 7 次独立采样（8 次补全，盲采样对照）。

接受后 J = min(J0, J_cand)（完整认证通过且可行有限），否则 J0。报告墙钟与补全次数。
使用缓存 score_fn（已验证与未缓存 bit-identical）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from action_contract import build_vehicle_plans, plan_hash, apply_action
from cc_lns_replanner import _select_one_mask, _plan_suffixes, _reconstruct
from dynmaskco_cc_context import mask_candidate_pool, validate_mask_scope
from dynmaskco_cc_graph import build_visible_index
from mpre_policy import stepwise_sample, enumerate_legal_actions, validate_repair_plan
from mpre import load_cvrp_model
from mtrained_replanner import (load_trained_model, _make_extract, mpre_score_fn_cached,
                                mtrained_score_fn_cached)
from visible_state import build_visible_state, evaluate_visible_plan
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile)
from run_fixed_state_quality import Probe, _eval_plan, _load_profile


# --------------------------------------------------------------------------- #
# 提议 + 有限搜索
# --------------------------------------------------------------------------- #

def _model_propose(score_fn, extract_fn, env, inst_idx, P, remaining, mutable_ids, K):
    legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
    if not legal:
        return []
    state = extract_fn(P, remaining, legal)
    scores = np.asarray(score_fn(state))
    order = np.argsort(-scores)
    return [legal[i] for i in order[:K]]


def _distance_propose(env, inst_idx, P, remaining, mutable_ids, K):
    legal = enumerate_legal_actions(env, inst_idx, P, remaining, mutable_ids)
    legal.sort(key=lambda x: (x[2].incremental_distance, x[1].action_id()))
    return legal[:K]


def _greedy_complete(propose, env, inst_idx, P, remaining, mutable_ids):
    rem = list(remaining)
    while rem:
        acts = propose(env, inst_idx, P, rem, mutable_ids, 1)
        if not acts:
            return None
        c, a, _cand = acts[0]
        P = apply_action(P, a, allowed_vehicle_ids=mutable_ids)
        rem.remove(c)
    return P


def sgbs_search(propose, env, inst_idx, partial, mask_set, mutable_ids, beam=2,
                max_per_parent=2, max_completions=8, apply_fn=apply_action,
                hash_fn=plan_hash):
    """逐层 beam 搜索：greedy + 每层 beam 分支、每父≤max_per_parent 动作、去重。

    返回 (completions, stats)。stats 记 n_unique / n_total / n_fail（greedy_complete 返回 None）。
    """
    seen = set()
    completions = []
    n_total = 0
    n_fail = 0

    def record(P):
        nonlocal n_total, n_fail
        n_total += 1
        if P is None:
            n_fail += 1
            return
        h = hash_fn(P)
        if h not in seen:
            seen.add(h)
            completions.append(P)

    def greedy_complete(P, remaining):
        rem = list(remaining)
        while rem:
            acts = propose(env, inst_idx, P, rem, mutable_ids, 1)
            if not acts:
                return None
            c, a, _cand = acts[0]
            P = apply_fn(P, a, allowed_vehicle_ids=mutable_ids)
            rem.remove(c)
        return P

    record(greedy_complete(partial, list(mask_set)))

    frontier = [(partial, list(mask_set))]
    while frontier and len(completions) < max_completions:
        nxt = []
        for (P, rem) in frontier:
            if not rem:
                continue
            acts = propose(env, inst_idx, P, rem, mutable_ids, max_per_parent)
            for c, a, _cand in acts:
                P2 = apply_fn(P, a, allowed_vehicle_ids=mutable_ids)
                rem2 = [x for x in rem if x != c]
                record(greedy_complete(P2, rem2))
                nxt.append((P2, rem2))
                if len(completions) >= max_completions:
                    break
            if len(completions) >= max_completions:
                break
        frontier = nxt[:beam]

    return completions, {'n_unique': len(completions), 'n_total': n_total, 'n_fail': n_fail}


def _eval_best(vis, plans, P0, mask_set, protected, contract, objective, J0):
    """从若干补全里选接受后 J 最优者；返回 (J, n_completions)。失败/认证不过 → J0。"""
    best_J = J0
    n = 0
    for P in plans:
        if P is None:
            continue
        n += 1
        ok, _ = validate_repair_plan(P, P0, mask_set, protected)
        if not ok:
            continue
        r = _eval_plan(vis, P, contract, objective)
        if r is None:
            continue
        best_J = min(best_J, float(r.J_vis))
    return best_J, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--split', choices=['train', 'cal'], required=True)
    ap.add_argument('--max-instances', type=int, default=16)
    ap.add_argument('--n-samples', type=int, default=7)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile)
    contract = default_pilot_contract()
    eff = apply_objective_profile(contract, profile)
    tw_max = float(dataset['tw_end'][:, 0].max())
    objective = eff.objective

    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)
    mpre_score = mpre_score_fn_cached(backbone)
    mtrained_score = mtrained_score_fn_cached(model)

    probe = Probe(eff, args.capacity, max_per_instance=2)
    for inst in range(args.max_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=probe, coldchain_contract=eff)
        env.run(inst)
    states = probe.states
    print(f"collected {len(states)} states over {args.max_instances} instances ({args.split})",
          flush=True)

    rows = []
    total_t = {}
    for st in states:
        env = st['env']
        inst_idx = st['inst']
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in st['visible_ids'] if int(i) != 0])
        extract_fn = _make_extract(env, inst_idx, st['clock'], st['vehicles'], st['served_mask'],
                                   st['visible_ids'], args.capacity, tw_max,
                                   node_to_local, local_to_node, Nv, st['P0'], st['mutable_ids'])
        vis = st['vis']
        partial = st['partial']
        mask_set = list(st['mask_set'])
        mutable_ids = st['mutable_ids']
        P0 = st['P0']
        protected = st['protected']
        J0 = st['J0']

        mpre_prop = lambda env, i, P, rem, mut, K: _model_propose(
            mpre_score, extract_fn, env, i, P, rem, mut, K)
        mtr_prop = lambda env, i, P, rem, mut, K: _model_propose(
            mtrained_score, extract_fn, env, i, P, rem, mut, K)
        dist_prop = _distance_propose

        row = {'inst': inst_idx, 'event': st['event'], 'J0': J0}
        t = {}

        # R（regret-2 单次）
        t0 = time.perf_counter()
        P_r = _reconstruct(env, inst_idx, partial, mask_set, mutable_ids)
        row['R'], _ = _eval_best(vis, [P_r], P0, mask_set, protected, eff, objective, J0)
        t['R'] = time.perf_counter() - t0

        # greedy
        for name, score_fn in (('Mpre_greedy', mpre_score), ('Mtrained_greedy', mtrained_score)):
            t0 = time.perf_counter()
            P, _s, _r, fail = stepwise_sample(score_fn, extract_fn, env, inst_idx, partial,
                                              mask_set, mutable_ids, deterministic=True)
            row[name], _ = _eval_best(vis, [None if fail is not None else P], P0, mask_set,
                                      protected, eff, objective, J0)
            t[name] = time.perf_counter() - t0

        # SGBS（逐层 beam=2，每父≤2 动作，≤8 补全）
        for name, prop in (('Mpre_sgbs', mpre_prop), ('Mtrained_sgbs', mtr_prop),
                           ('dist_sgbs', dist_prop)):
            t0 = time.perf_counter()
            completions, sstats = sgbs_search(prop, env, inst_idx, partial, mask_set,
                                              mutable_ids, beam=2)
            row[name], _ = _eval_best(vis, completions, P0, mask_set, protected, eff, objective,
                                      J0)
            t[name] = time.perf_counter() - t0
            row[name + '_n_unique'] = sstats['n_unique']
            row[name + '_n_total'] = sstats['n_total']
            row[name + '_n_fail'] = sstats['n_fail']

        # greedy + n_samples 独立采样
        t0 = time.perf_counter()
        plans = []
        P, _s, _r, fail = stepwise_sample(mtrained_score, extract_fn, env, inst_idx, partial,
                                          mask_set, mutable_ids, deterministic=True)
        plans.append(None if fail is not None else P)
        for si in range(args.n_samples):
            P, _s, _r, fail = stepwise_sample(mtrained_score, extract_fn, env, inst_idx, partial,
                                              mask_set, mutable_ids, deterministic=False,
                                              rng=np.random.default_rng(si))
            plans.append(None if fail is not None else P)
        row['Mtrained_greedy7'], _ = _eval_best(vis, plans, P0, mask_set, protected, eff,
                                                objective, J0)
        t['Mtrained_greedy7'] = time.perf_counter() - t0

        rows.append(row)
        for k, v in t.items():
            total_t[k] = total_t.get(k, 0.0) + v
        print(f"  [inst {inst_idx}/evt {st['event']}] J0={J0:.4f} R={row['R']:.4f} "
              f"greedy={row['Mtrained_greedy']:.4f} sgbs={row['Mtrained_sgbs']:.4f} "
              f"dist={row['dist_sgbs']:.4f} g7={row['Mtrained_greedy7']:.4f}", flush=True)

    methods = ['R', 'Mpre_greedy', 'Mtrained_greedy', 'Mpre_sgbs', 'Mtrained_sgbs',
               'dist_sgbs', 'Mtrained_greedy7']

    def _inst_mean(m):
        per = {}
        for r in rows:
            per.setdefault(r['inst'], []).append(r[m])
        return [float(np.mean(v)) for v in per.values()]

    def _paired(mA, mB, n_boot=2000, seed=0):
        per = {}
        for r in rows:
            per.setdefault(r['inst'], []).append(r[mA] - r[mB])
        d = [float(np.mean(v)) for v in per.values()]
        mean = float(np.mean(d))
        rng = np.random.default_rng(seed)
        b = [float(np.mean([d[i] for i in rng.integers(0, len(d), size=len(d))]))
             for _ in range(n_boot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'mean': mean, 'ci_lo': float(lo), 'ci_hi': float(hi)}

    summary = {
        'split': args.split, 'n_states': len(rows), 'n_instances': len(set(r['inst'] for r in rows)),
        'accepted_mean': {m: float(np.mean(_inst_mean(m))) for m in methods},
        'wallclock_total_s': {m: float(total_t.get(m, 0.0)) for m in methods},
        'completion_stats': {
            name: {
                'n_unique_total': int(sum(r.get(f'{name}_n_unique', 0) for r in rows)),
                'n_total': int(sum(r.get(f'{name}_n_total', 0) for r in rows)),
                'n_fail_total': int(sum(r.get(f'{name}_n_fail', 0) for r in rows)),
            } for name in ('Mpre_sgbs', 'Mtrained_sgbs', 'dist_sgbs')
        },
    }
    for mA, mB in (('Mtrained_sgbs', 'Mtrained_greedy'), ('Mtrained_sgbs', 'R'),
                   ('Mtrained_sgbs', 'Mtrained_greedy7'), ('Mtrained_sgbs', 'dist_sgbs'),
                   ('Mtrained_sgbs', 'Mpre_sgbs'), ('Mtrained_greedy7', 'R')):
        summary[f'{mA}_minus_{mB}'] = _paired(mA, mB)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_state.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in summary.items() if k != 'per_state'}, indent=2))
    print(f"saved: {args.out}/summary.json + per_state.json")


if __name__ == '__main__':
    main()
