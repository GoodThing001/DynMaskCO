"""Does the local-win prize grow with the declared subproblem size?

Round 5 established, at 4-6 orders on one vehicle, that the insertion-position decision has almost
no headroom and that moving the learned term to the customer decision multiplies the OPPORTUNITY
rate about fivefold while leaving the PRIZE at ~0.003 distance units. That makes subproblem SIZE
the suspect, and this module tests it directly: the same headroom measurement repeated at
controlled sizes, so the decision "keep redesigning the operator" versus "stop training this line"
can be made on a measured scaling curve instead of on a guess.

Design, and the confound it avoids. The live day only offers whatever suffix length a vehicle
happens to have (3-5 orders), which cannot probe larger sizes. So the subproblem is built from the
pool of revealed, unserved orders at a real decision point and truncated to size n by ORDER ID.
Order ids are arbitrary with respect to geometry in this generator, so id-prefixes do not
systematically select near or far customers; the module records the mean pairwise distance of each
subset anyway so a geometry confound would be visible rather than hidden.

For each size it reports, for BOTH candidate decisions, the headroom rate (how often the base
rule's own currency disagrees with the completed-plan optimum) and the prize (how much plan
distance the plan-optimum choice would actually save).

Diagnostic only: nothing is trained, no arm is compared, the plan-distance proxy is pure geometry
rather than A-v1 utility, no interval is a registered test, and no impossibility, equivalence,
attribution or stable-gain claim is made. A flat curve speaks to THIS operator form on THESE
states; it cannot close other operators or methods.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def main(args):
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    scripts = Path(args.scripts).resolve()
    from pickup_operator_headroom import (bootstrap, local_nodes, plan_distance,
                                          complete_greedily)
    bootstrap(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from delivery_maskco import partial_adjacency
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

    decisions = []
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
            if len(pool) >= args.sizes[0]:
                got.append({'inst': inst, 'clock': float(clock), 'pool': sorted(pool)})

        env.snapshot_hook = hook
        env.run(inst)
        decisions.extend(got)

    rows = []
    for st in decisions:
        pool = st['pool'][:args.pool_cap]
        for n in args.sizes:
            if len(pool) < n:
                continue
            orders = pool[:n]
            env = None
            # rebuild a standalone env just to read coordinates (arrays are not mutated by a run)
            env, _ = make_env()
            coords, loc = local_nodes(env, st['inst'], orders)
            inv = {v: k for k, v in loc.items()}
            nodes = n + 1
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            off = ~np.eye(nodes, dtype=bool)
            mask = set(map(int, np.random.default_rng(
                np.random.SeedSequence([args.mask_seed, n, st['inst']])).choice(
                np.arange(1, nodes), size=max(1, int(math.ceil(args.mask_fraction * n))),
                replace=False)))
            adj = partial_adjacency([[loc[o] for o in orders]], mask, nodes)
            route = [loc[o] for o in orders if loc[o] not in mask]
            remaining = sorted(mask)

            # ---- position-decision headroom, for the customer the rule would insert ----
            per_customer = []
            for i in remaining:
                opts = []
                for at in range(len(route) + 1):
                    a = route[at - 1] if at else 0
                    b = route[at] if at < len(route) else 0
                    if adj[a, b] > 0:
                        continue
                    delta = dist[a, i] + dist[i, b] - (0.0 if a == b else dist[a, b])
                    opts.append((float(delta), at))
                if opts:
                    opts.sort()
                    regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
                    per_customer.append((regret, i, opts))
            pos_prize = pos_rate = cust_prize = cust_rate = None
            if per_customer:
                _, i0, opts0 = max(per_customer, key=lambda r: (r[0], -r[1]))
                if len(opts0) > 1:
                    cand = []
                    for delta, at in opts0:
                        r2 = list(route)
                        r2.insert(at, i0)
                        done = complete_greedily(coords, loc, r2,
                                                 [x for x in remaining if x != i0], adj, dist)
                        if done is not None:
                            cand.append((float(delta), at, plan_distance(coords, [done])))
                    if len(cand) > 1:
                        bd = min(cand, key=lambda c: (c[0], c[1]))
                        bp = min(cand, key=lambda c: (c[2], c[1]))
                        pos_rate = int(bd[1] != bp[1])
                        pos_prize = float(bd[2] - bp[2])
                cust = []
                for regret, i, opts in per_customer:
                    r2 = list(route)
                    r2.insert(opts[0][1], i)
                    done = complete_greedily(coords, loc, r2,
                                             [x for x in remaining if x != i], adj, dist)
                    if done is not None:
                        cust.append((regret, i, plan_distance(coords, [done])))
                if len(cust) > 1:
                    br = max(cust, key=lambda c: (c[0], -c[1]))
                    bp = min(cust, key=lambda c: (c[2], c[1]))
                    cust_rate = int(br[1] != bp[1])
                    cust_prize = float(br[2] - bp[2])
            pw = dist[np.triu_indices(nodes, 1)]
            rows.append({'n': int(n), 'inst': st['inst'], 'clock': st['clock'],
                         'n_revealed_pool': len(pool),
                         'mean_pairwise_distance': float(pw.mean()) if len(pw) else None,
                         'position_headroom': pos_rate, 'position_prize': pos_prize,
                         'customer_headroom': cust_rate, 'customer_prize': cust_prize})

    per_size = {}
    for n in args.sizes:
        rs = [r for r in rows if r['n'] == n]
        pr = [r['position_prize'] for r in rs if r['position_prize'] is not None]
        cr = [r['customer_prize'] for r in rs if r['customer_prize'] is not None]
        phr = [r['position_headroom'] for r in rs if r['position_headroom'] is not None]
        chr_ = [r['customer_headroom'] for r in rs if r['customer_headroom'] is not None]
        per_size[str(n)] = {
            'n_decisions': len(rs),
            'mean_pairwise_distance': (float(np.mean([r['mean_pairwise_distance'] for r in rs]))
                                       if rs else None),
            'position_headroom_rate': float(np.mean(phr)) if phr else None,
            'position_mean_prize': float(np.mean(pr)) if pr else None,
            'position_max_prize': float(np.max(pr)) if pr else None,
            'customer_headroom_rate': float(np.mean(chr_)) if chr_ else None,
            'customer_mean_prize': float(np.mean(cr)) if cr else None,
            'customer_max_prize': float(np.max(cr)) if cr else None,
        }
    result = {
        'kind': 'pickup_local_win_prize_scaling',
        'sizes': list(args.sizes), 'per_size': per_size,
        'mask_fraction': args.mask_fraction, 'mask_seed': args.mask_seed,
        'pool_cap': args.pool_cap,
        'per_decision': rows,
        'reading': ('If the prize grows with n, the operator should be enlarged and only then '
                    'retrained; if it stays flat, no scoring function at any size tested here has '
                    'a meaningful prize and this line should be closed rather than retuned.'),
        'confound_note': ('subsets are id-prefixes of the revealed pool; ids are arbitrary with '
                          'respect to geometry in this generator, and mean_pairwise_distance is '
                          'reported per size so a geometry confound would be visible'),
        'what_this_is_not': [
            'not a method and not an arm comparison: nothing is trained or improved',
            'the proxy is pure plan distance, not A-v1 utility',
            'descriptive only; no interval is a registered test',
            'no impossibility, equivalence, attribution or stable-gain claim: a flat curve speaks '
            'to THIS operator form on THESE states only'],
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps(per_size, indent=2, default=str))
    print(json.dumps({'wall_s': result['wall_s']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--instances', type=int, default=10)
    ap.add_argument('--states-per-instance', type=int, default=3)
    ap.add_argument('--min-clock', type=float, default=2.0)
    ap.add_argument('--sizes', type=int, nargs='+', default=[4, 6, 8, 12, 16])
    ap.add_argument('--pool-cap', type=int, default=24)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--mask-seed', type=int, default=2026120202)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
