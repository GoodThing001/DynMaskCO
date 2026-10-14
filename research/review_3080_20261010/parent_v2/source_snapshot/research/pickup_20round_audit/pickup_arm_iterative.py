"""P-MASKCO-ARM-ITER-v1: Stage A - does ITERATION add anything over ONE-SHOT reconstruction?

WHY: the registered panels (v1/v2) describe a MaskCO mask-and-reconstruct REFINEMENT LOOP, but both
archived pilots were ONE-SHOT single-decision interventions. The untested part of the treatment is
therefore iteration. This module measures it directly, cheaply, with already-qualified machinery.

FEASIBILITY FACT that makes this cheap and safe: `pickup_maskco_arm.propose` builds the subproblem
purely from the `order_ids` list it is handed (`partial_adjacency([[loc[o] for o in order_ids]], ...)`)
and reads `env` only for coordinates/demands. So a refinement round is just
`propose(..., order_ids=previous_permutation, seed=next_seed, ...)` - no env surgery, no writeback
into a live plan, and the installation path is the SAME force_suffix path the qualified pilots used.

DESIGN per captured decision point and per mode:
  round 0 = the baseline's own recorded suffix order
  round r = propose(round r-1's order, seed = mask_seed + r), installed with force_suffix, whole day
            run causally, FULL A-v1 utility evaluated
  acceptance = keep a round's permutation only if its full-day utility strictly exceeds the best so
            far (this IS the registered accept-only-on-certified-improvement rule, evaluated exactly)
  decoder calls = 1 per round per mode, so the arms stay matched at R calls

It ALSO surveys the subproblem size actually available per vehicle suffix, because node n13 located the
oracle prize at n>=12 while the pilots ran n=3 and n=4-5. If the per-vehicle suffix regime cannot
reach n>=12, that is a decisive statement about the panel as specified.

SCOPE: bounded engineering/mechanism gate on fresh instances. Descriptive only, no alpha. It cannot
establish or refute a stable online gain, MaskCO attribution, impossibility or equivalence, and any
mean above zero is far below the >= 20 utility/day main gate.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap_paths(scripts):
    scripts = Path(scripts).resolve()
    here = Path(__file__).resolve().parent
    for p in (here, scripts / 'delivery', scripts / 'evaluation', scripts / 'coldchain',
              scripts / 'simulation'):
        sys.path.insert(0, str(p))
    return scripts


def place(perm, template, slots):
    out = list(template)
    for slot, val in zip(slots, perm):
        out[slot] = val
    return out


def main(args):
    started = time.perf_counter()
    scripts = bootstrap_paths(args.scripts)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from delivery_maskco import MaskCORouteProvider
    from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH, generate_dataset)
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv
    from pickup_maskco_arm import propose, MODES

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = ColdChainContractV2()
    contract.validate()
    budget = float(args.budget)
    p_c = {0: 5.0, 1: 10.0, 2: 15.0}

    train_ds = generate_dataset(1, 200, args.data_seed + 1)
    test_ds = add_v2_initial_quality(generate_dataset(args.instances, 200, args.data_seed), contract)
    dem = train_ds['demands'][0]
    m = dem[1:] > 0
    hist = [(train_ds['reveal_time'][0][1:][m], train_ds['temp_class'][0][1:][m].astype(int))]

    def make_env():
        rp = UncondDynamicReplanner(budget=budget, capacity=50.0,
                                    booking_horizon=BOOKING_HORIZON, contract=contract,
                                    cooling_share=args.cooling_share, hist=hist, k=10)
        return StrictOnlineEnv(test_ds, capacity=50.0, num_vehicles=15,
                               tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                               coldchain_contract=contract,
                               booking_horizon=BOOKING_HORIZON), rp

    def ev(traces, inst, rp):
        return evaluate_trace_a1(traces, {k: v[inst] for k, v in test_ds.items()}, contract,
                                 set(rp._accepted), set(rp._rejected), budget, p_c)

    # ---------------------------------------------------------------- capture + size survey
    states = []
    size_hist = {}
    for inst in range(args.instances):
        env, rp = make_env()
        got = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock):
                return
            for v in vehicles:
                tmpl = [int(x) for x in (v.mutable_suffix or [])]
                slots = [k for k, x in enumerate(tmpl) if x != 0 and not served_mask[x]
                         and float(env_.reveal_time[inst_idx, x]) <= clock + 1e-6]
                n = len(slots)
                size_hist[n] = size_hist.get(n, 0) + 1
                if args.min_orders <= n <= args.max_orders and len(got) < args.states_per_instance:
                    got.append({'inst': inst, 'clock': float(clock), 'event_id': int(event_id),
                                'vehicle_id': int(v.vehicle_id), 'template': tmpl, 'slots': slots,
                                'orders': [tmpl[k] for k in slots], 'env': env_})
                    return

        env.snapshot_hook = hook
        base_traces, _ = env.run(inst)
        base_eval = ev(base_traces, inst, rp)
        for g in got:
            g['baseline_utility'] = float(base_eval['utility'])
        states.extend(got)

    provider = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root)
    provider.prewarm(args.max_orders + 1, 50.0)

    # ---------------------------------------------------------------- iterative loop
    rows = []
    for st in states:
        inst, key = st['inst'], (st['event_id'], st['vehicle_id'])
        for mode in MODES:
            order = list(st['orders'])
            best_seq = list(st['template'])
            best_u = st['baseline_utility']
            per_round = []
            for r in range(args.rounds):
                perm, info = propose(provider, st['env'], inst, order, mode, args.mask_fraction,
                                     args.mask_seed + r, args.guide_weight)
                seq = place(perm, st['template'], st['slots'])
                e2, r2 = make_env()
                try:
                    tr2, _ = e2.run(inst, force_suffix={key: seq})
                    e_ = ev(tr2, inst, r2)
                    u = float(e_['utility'])
                    hard = bool(e_['hard_feasible'])
                except Exception as exc:                                  # noqa: BLE001
                    u, hard, e_ = None, False, {}
                    info['failure'] = '%s: %s' % (type(exc).__name__, exc)
                accepted = bool(u is not None and hard and u > best_u + 1e-10)
                per_round.append({'round': r, 'mask_seed': args.mask_seed + r, 'utility': u,
                                  'hard': hard, 'accepted': accepted,
                                  'delta_vs_baseline': (None if u is None else u - st['baseline_utility']),
                                  'decoder_calls': info['decoder_calls'],
                                  'mask_extended': info['mask_extended_to_all_for_insertability'],
                                  'rel_to_prev': (None if u is None else u - per_round[-1]['utility']
                                                  if per_round and per_round[-1]['utility'] is not None
                                                  else None)})
                if accepted:
                    best_u, best_seq = u, seq
                    order = list(perm)
            rows.append({'inst': inst, 'clock': st['clock'], 'mode': mode,
                         'n_subproblem': len(st['orders']), 'orders': list(st['orders']),
                         'baseline_utility': st['baseline_utility'],
                         'best_utility': best_u,
                         'best_delta_vs_baseline': best_u - st['baseline_utility'],
                         'n_rounds_accepted': sum(1 for x in per_round if x['accepted']),
                         'decoder_calls_total': sum(x['decoder_calls'] for x in per_round),
                         'per_round': per_round})

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.asarray(vals, dtype=np.float64)
        return {'n': int(a.size), 'mean': float(a.mean()),
                'n_positive': int((a > 1e-10).sum()), 'n_zero': int((abs(a) <= 1e-10).sum()),
                'n_negative': int((a < -1e-10).sum())}

    per_mode = {}
    for mode in MODES:
        rs = [r for r in rows if r['mode'] == mode]
        per_mode[mode] = {
            'n': len(rs),
            'mean_best_delta_vs_doing_nothing': float(np.mean(
                [r['best_delta_vs_baseline'] for r in rs])) if rs else None,
            'versus_doing_nothing': stat([r['best_delta_vs_baseline'] for r in rs]),
            'n_states_accepting_any_round': int(sum(1 for r in rs if r['n_rounds_accepted'] > 0)),
            'mean_round1_delta': float(np.mean(
                [r['per_round'][0]['delta_vs_baseline'] for r in rs
                 if r['per_round'] and r['per_round'][0]['delta_vs_baseline'] is not None]))
            if rs else None,
            'decoder_calls_total': int(sum(r['decoder_calls_total'] for r in rs)),
        }

    def paired(a_mode, b_mode):
        a = {(r['inst'], r['clock']): r for r in rows if r['mode'] == a_mode}
        b = {(r['inst'], r['clock']): r for r in rows if r['mode'] == b_mode}
        ks = sorted(set(a) & set(b))
        d = [a[k]['best_utility'] - b[k]['best_utility'] for k in ks]
        if not d:
            return None
        return {'n_pairs': len(d), 'mean': float(np.mean(d)),
                'a_better': int(sum(1 for x in d if x > 1e-10)),
                'a_worse': int(sum(1 for x in d if x < -1e-10)),
                'tied': int(sum(1 for x in d if abs(x) <= 1e-10))}

    result = {
        'kind': 'pickup_stage_a_iterative_vs_oneshot',
        'protocol_version': 'P-MASKCO-ARM-ITER-v1',
        'stage': 'A (cheap gate; not the panel)',
        'parent_registration': 'results/research_20261010_continuation_v2/pickup/'
                               'P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json',
        'data_seed': args.data_seed, 'mask_seed': args.mask_seed, 'rounds': args.rounds,
        'mask_fraction': args.mask_fraction, 'guide_weight': args.guide_weight,
        'budget': budget, 'n_instances': args.instances, 'n_states': len(states),
        'subproblem_size_survey': {str(k): v for k, v in sorted(size_hist.items())},
        'max_subproblem_size_available': (max(size_hist) if size_hist else None),
        'n_states_at_or_above_12': int(sum(v for k, v in size_hist.items() if k >= 12)),
        'per_mode': per_mode,
        'paired_after_iteration': {'maskco_minus_recon_only': paired('maskco', 'recon_only'),
                                   'recon_only_minus_discard': paired('recon_only', 'discard')},
        'rows': rows,
        'what_this_measures': [
            'whether repeated mask-and-reconstruct rounds at one decision point improve the '
            'whole-day A-v1 utility over a single round (the untested part of the treatment)',
            'every mode against DOING NOTHING, not only against each other',
            'the subproblem size actually available per vehicle suffix, since the oracle prize was '
            'located at n>=12 while the archived pilots ran n=3 and n=4-5'],
        'what_this_does_not_measure': [
            'the online policy that reconstructs at EVERY decision point across a day; this iterates '
            'at ONE captured decision point per state',
            'no strong_rolling arm, no cond_hist arm, no acceptance policy over a whole day',
            'DESCRIPTIVE only, no alpha; cannot support stable gain, attribution, impossibility or '
            'equivalence, and the >= 20 utility/day main gate is NOT addressed'],
        'stable_online_gain_established': False, 'maskco_attributable': False,
        'pickup_impossibility_proven': False, 'equivalence_established': False,
        'input_hashes': {
            'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'arm_sha256': hashlib.sha256(Path(Path(__file__).resolve().parent /
                                              'pickup_maskco_arm.py').read_bytes()).hexdigest()},
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_states', 'max_subproblem_size_available', 'n_states_at_or_above_12',
                       'per_mode', 'paired_after_iteration', 'wall_s')}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--instances', type=int, default=4)
    ap.add_argument('--states-per-instance', type=int, default=2)
    ap.add_argument('--min-clock', type=float, default=6.0)
    ap.add_argument('--min-orders', type=int, default=3)
    ap.add_argument('--max-orders', type=int, default=12)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--guide-weight', type=float, default=1.0)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--data-seed', type=int, default=2026120201)
    ap.add_argument('--mask-seed', type=int, default=2026120231)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    main(ap.parse_args())
