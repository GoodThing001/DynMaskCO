"""Is the measured prize actually LEARNABLE? The gate that must pass before any training.

Round 6 measured an oracle prize that grows with subproblem size: at n>=12 the customer decision has
headroom above 0.9 and a mean achievable plan-distance saving of about 0.0133 (max 0.137). But that
prize was defined by an ORACLE completion - the option that turns out best once the rest of the
reconstruction is greedily finished. A learned term cannot compute that; it can only score what it
sees at decision time. Oracle headroom is therefore an UPPER BOUND on what any scoring function
could get, not evidence that a scoring function can get any of it.

This module closes that gap with the cheapest decisive test: can the oracle-optimal customer be
predicted, out of sample, from features available at the decision instant, better than the base
rule already does? If a linear probe cannot beat the regret rule on held-out instances, then no
amount of training on these features will convert the oracle prize into a real gain, and the honest
next action is to record the line as negative rather than to spend a training cycle.

Protocol discipline:
  * splits are BY INSTANCE, so a probe is never scored on a decision point from an instance whose
    other states it saw. Within-state candidate leakage would make the probe look clairvoyant.
  * baselines are reported alongside: the base regret rule, the plain immediate-delta rule, and the
    uniform base rate.
  * features are strictly decision-time visible: no completed-plan quantity is used as an input.

Diagnostic only: nothing is trained to convergence, no arm is compared, no utility is improved, and
no interval is a registered test. No impossibility, equivalence, attribution or stable-gain claim.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

FEATURES = ['demand', 'temp0', 'temp1', 'temp2', 'tw_width', 'depot_dist', 'best_gap_min',
            'best_immediate_delta', 'regret', 'n_positions', 'route_len']


def build_state(env, inst, orders, mask_fraction, mask_seed):
    """Return the decision-time candidate table for one subproblem state."""
    from pickup_operator_headroom import local_nodes, plan_distance, complete_greedily
    coords, loc = local_nodes(env, inst, orders)
    inv = {v: k for k, v in loc.items()}
    n = len(orders)
    nodes = n + 1
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    from delivery_maskco import partial_adjacency
    mask = set(map(int, np.random.default_rng(
        np.random.SeedSequence([mask_seed, n, inst])).choice(
        np.arange(1, nodes), size=max(1, int(math.ceil(mask_fraction * n))), replace=False)))
    adj = partial_adjacency([[loc[o] for o in orders]], mask, nodes)
    route = [loc[o] for o in orders if loc[o] not in mask]
    remaining = sorted(mask)
    if len(remaining) < 2:
        return None
    rows = []
    for i in remaining:
        opts = []
        for at in range(len(route) + 1):
            a = route[at - 1] if at else 0
            b = route[at] if at < len(route) else 0
            if adj[a, b] > 0:
                continue
            delta = dist[a, i] + dist[i, b] - (0.0 if a == b else dist[a, b])
            opts.append((float(delta), at))
        if not opts:
            continue
        opts.sort()
        regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
        r2 = list(route)
        r2.insert(opts[0][1], i)
        done = complete_greedily(coords, loc, r2, [x for x in remaining if x != i], adj, dist)
        if done is None:
            continue
        gid = inv[i]
        tc = int(env.temp_class[inst, gid])
        rows.append({
            'customer': gid,
            'oracle_plan_distance': float(plan_distance(coords, [done])),
            'regret': float(regret) if np.isfinite(regret) else 1e9,
            'best_immediate_delta': float(opts[0][0]),
            'n_positions': int(len(opts)),
            'features': {
                'demand': float(env.demands[inst, gid]),
                'temp0': 1.0 if tc == 0 else 0.0,
                'temp1': 1.0 if tc == 1 else 0.0,
                'temp2': 1.0 if tc == 2 else 0.0,
                'tw_width': float(env.tw_end[inst, gid] - env.tw_start[inst, gid]),
                'depot_dist': float(dist[0, i]),
                'best_gap_min': float(min(dist[0, i], dist[i, 0])),
                'best_immediate_delta': float(opts[0][0]),
                'regret': float(regret) if np.isfinite(regret) else 1e9,
                'n_positions': float(len(opts)),
                'route_len': float(len(route)),
            }})
    if len(rows) < 2:
        return None
    best = min(rows, key=lambda r: (r['oracle_plan_distance'], r['customer']))
    by_regret = max(rows, key=lambda r: (r['regret'], -r['customer']))
    by_delta = min(rows, key=lambda r: (r['best_immediate_delta'], r['customer']))
    n_c = len(rows)
    x = np.array([[r['features'][f] for f in FEATURES] for r in rows], dtype=np.float64)
    return {
        'n_candidates': n_c,
        'X': x,
        'y': int(next(k for k, r in enumerate(rows) if r['customer'] == best['customer'])),
        'regret_pick': int(next(k for k, r in enumerate(rows)
                                if r['customer'] == by_regret['customer'])),
        'delta_pick': int(next(k for k, r in enumerate(rows)
                               if r['customer'] == by_delta['customer'])),
        'regret_win': float(by_regret['oracle_plan_distance']
                            - best['oracle_plan_distance']),
        'customers': [r['customer'] for r in rows],
    }


def fit_probe(Xtr, ytr):
    """Tiny multinomial logistic probe: one shared weight vector scored per candidate, softmax
    over the candidates of each state. Enough to detect whether ANY linear signal exists."""
    d = Xtr[0].shape[-1]
    w = np.zeros(d)
    for _ in range(400):
        grads = np.zeros(d)
        for X, y in zip(Xtr, ytr):
            Z = X @ w
            Z = Z - Z.max()
            p = np.exp(Z) / np.exp(Z).sum()
            hot = np.zeros_like(p)
            hot[y] = 1.0
            grads += X.T @ (p - hot)
        w -= 0.5 * grads / max(1, len(Xtr))
    return w


def evaluate(w, Xte, yte):
    correct = 0
    for X, y in zip(Xte, yte):
        if int(np.argmax(X @ w)) == y:
            correct += 1
    return correct / max(1, len(Xte))


def main(args):
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    scripts = Path(args.scripts).resolve()
    from pickup_operator_headroom import bootstrap
    bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = add_v2_initial_quality(generate_dataset(args.instances, 200, 20260924), contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        m = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][m],
                     train_ds['temp_class'][i][1:][m].astype(int)))

    def make_env():
        rp = UncondDynamicReplanner(budget=budget, capacity=50.0,
                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                    cooling_share=args.cooling_share, hist=hist, k=10)
        return StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=15,
                               tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                               coldchain_contract=contract,
                               booking_horizon=BOOKING_HORIZON), rp

    states = []
    for inst in range(args.instances):
        env, rp = make_env()
        got = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock) or len(got) >= args.states_per_instance:
                return
            pool = [int(o) for o in range(1, env_.num_nodes)
                    if not served_mask[o]
                    and float(env_.reveal_time[inst_idx, o]) <= clock + 1e-6]
            if len(pool) >= max(args.sizes):
                got.append({'inst': inst, 'clock': float(clock),
                            'pool': sorted(pool)[:args.pool_cap]})

        env.snapshot_hook = hook
        env.run(inst)
        states.extend(got)

    # build samples per size, keeping the instance label so splits can be made by instance
    samples = {}
    for st in states:
        env, _ = make_env()
        for n in args.sizes:
            if len(st['pool']) < n:
                continue
            b = build_state(env, st['inst'], st['pool'][:n], args.mask_fraction, args.mask_seed)
            if b is None:
                continue
            b['inst'] = st['inst']
            samples.setdefault(n, []).append(b)

    results = {}
    for n, ss in sorted(samples.items()):
        insts = sorted({s['inst'] for s in ss})
        if len(insts) < 3:
            results[str(n)] = {'n_samples': len(ss), 'note': 'too few instances to split'}
            continue
        test_insts = set(insts[-max(1, len(insts) // 3):])
        tr = [s for s in ss if s['inst'] not in test_insts]
        te = [s for s in ss if s['inst'] in test_insts]
        if not tr or not te:
            results[str(n)] = {'n_samples': len(ss), 'note': 'empty split'}
            continue
        w = fit_probe([s['X'] for s in tr], [s['y'] for s in tr])
        probe_acc = evaluate(w, [s['X'] for s in te], [s['y'] for s in te])
        regret_acc = float(np.mean([s['regret_pick'] == s['y'] for s in te]))
        delta_acc = float(np.mean([s['delta_pick'] == s['y'] for s in te]))
        base_rate = float(np.mean([1.0 / s['n_candidates'] for s in te]))
        prize = float(np.mean([s['regret_win'] for s in te]))
        results[str(n)] = {
            'n_samples': len(ss), 'n_train': len(tr), 'n_test': len(te),
            'n_test_instances': len(test_insts),
            'mean_candidates': float(np.mean([s['n_candidates'] for s in ss])),
            'uniform_base_rate': base_rate,
            'regret_rule_accuracy': regret_acc,
            'immediate_delta_rule_accuracy': delta_acc,
            'linear_probe_out_of_sample_accuracy': float(probe_acc),
            'probe_beats_regret_rule': bool(probe_acc > regret_acc),
            'mean_oracle_prize_foregone_by_regret_rule': prize,
            'probe_weights': {f: float(v) for f, v in zip(FEATURES, w)},
        }

    result = {
        'kind': 'pickup_prize_learnability_gate',
        'question': ('can the oracle-optimal customer be predicted out of sample from '
                     'decision-time-visible features better than the base regret rule already '
                     'predicts it?'),
        'features_used': FEATURES,
        'sizes': list(args.sizes), 'per_size': results,
        'split_protocol': ('splits are BY INSTANCE; a probe is never scored on a decision from an '
                           'instance whose other states it was fitted on, because candidates '
                           'within one state share the whole reconstructed plan and would leak'),
        'reading': ('If the probe does not beat the regret rule out of sample, the oracle prize '
                    'measured earlier is not reachable by any scoring function over these '
                    'features, and training on them is pointless. If it does beat the rule, the '
                    'margin AND the remaining foregone prize say how much a trained term could '
                    'aim for.'),
        'what_this_is_not': [
            'not a method: the probe is a diagnostic, not a deployed scorer, and it is not '
            'trained to convergence',
            'the oracle uses plan distance, a pure-geometry proxy, not A-v1 utility',
            'descriptive only; no interval here is a registered test',
            'no impossibility, equivalence, attribution or stable-gain claim: a null result '
            'speaks to THESE features on THESE states only'],
        'mask_fraction': args.mask_fraction, 'mask_seed': args.mask_seed,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=18)
    ap.add_argument('--states-per-instance', type=int, default=3)
    ap.add_argument('--min-clock', type=float, default=2.0)
    ap.add_argument('--sizes', type=int, nargs='+', default=[8, 12, 16])
    ap.add_argument('--pool-cap', type=int, default=24)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--mask-seed', type=int, default=2026120202)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
