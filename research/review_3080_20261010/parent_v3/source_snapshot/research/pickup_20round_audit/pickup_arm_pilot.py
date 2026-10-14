"""Bounded pickup development pilot for the masked-reconstruction arm (NOT the registered panel).

The registered panel `P-LIVE-DEV-PANEL-v1` needs a full online policy with an acceptance rule and
matched budgets across five arms. This pilot answers one narrower question first, because it can
invalidate everything else cheaply:

    on the same real mid-day state, with the SAME mask schedule and the SAME one decoder call,
    does installing the guide-scored reconstruction produce a better A-v1 outcome than
    installing the distance-only reconstruction?

Design. For each captured decision point the baseline's own recorded suffix is the reference.
Each arm proposes a permutation of exactly the subproblem order set; the harness rewrites that
permutation into the baseline template's own order slots and installs it with force_suffix, then
the whole day runs causally and the full A-v1 utility is evaluated. Both the RAW candidate effect
and the effect under the online accept-only-if-better rule are reported, because they answer
different questions.

Scope and honesty. This is an exploratory pilot on a handful of engineered states, so every
interval here is DESCRIPTIVE ONLY and carries no alpha. It cannot establish a stable online gain,
MaskCO attribution, impossibility or equivalence. A mean above zero is far below the program's
>= 20 utility/day main gate and must never be reported as reaching it.
"""
import argparse
from copy import deepcopy
import hashlib
import json
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

    train_ds = generate_dataset(1, 200, 20260923)
    test_ds = add_v2_initial_quality(generate_dataset(args.instances, 200, 20260924), contract)
    hist = []
    for i in range(1):
        dem = train_ds['demands'][i]
        m = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][m], train_ds['temp_class'][i][1:][m].astype(int)))

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

    states = []
    for inst in range(args.instances):
        env, rp = make_env()
        got = []

        def hook(env_, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
                 all_customers):
            if clock < float(args.min_clock) or len(got) >= args.states_per_instance:
                return
            for v in vehicles:
                tmpl = [int(x) for x in (v.mutable_suffix or [])]
                slots = [k for k, x in enumerate(tmpl) if x != 0 and not served_mask[x]
                         and float(env_.reveal_time[inst_idx, x]) <= clock + 1e-6]
                if args.min_orders <= len(slots) <= args.max_orders:
                    got.append({'inst': inst, 'clock': float(clock), 'event_id': int(event_id),
                                'vehicle_id': int(v.vehicle_id), 'template': tmpl, 'slots': slots,
                                'orders': [tmpl[k] for k in slots],
                                'env': env_, 'served': np.array(served_mask, copy=True)})
                    return

        env.snapshot_hook = hook
        base_traces, _ = env.run(inst)
        base_eval = ev(base_traces, inst, rp)
        base_u = float(base_eval['utility'])
        base_accept = set(rp._accepted)
        for gi, g in enumerate(got):
            g['baseline_utility'] = base_u
            g['baseline_accept'] = base_accept
            states.append(g)

    provider = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root)
    provider.prewarm(args.max_orders + 1, 50.0)

    rows = []
    for st in states:
        inst = st['inst']
        key = (st['event_id'], st['vehicle_id'])
        for mode in MODES:
            for s in range(args.pilot_seed, args.pilot_seed + args.seeds_per_mode):
                perm, info = propose(provider, st['env'], inst, st['orders'], mode,
                                     args.mask_fraction, s, args.guide_weight)
                seq = place(perm, st['template'], st['slots'])
                e2, r2 = make_env()
                try:
                    tr2, _ = e2.run(inst, force_suffix={key: seq})
                    e_ = ev(tr2, inst, r2)
                    raw = float(e_['utility']) - st['baseline_utility']
                    hard = bool(e_['hard_feasible'])
                except Exception as exc:                       # noqa: BLE001
                    raw, hard, e_ = None, False, {}
                    info['failure'] = '%s: %s' % (type(exc).__name__, exc)
                rows.append({'inst': inst, 'clock': st['clock'], 'mode': mode, 'mask_seed': s,
                             'permutation': perm, 'raw_delta_utility': raw,
                             'hard_feasible': hard, 'accepted_delta_utility':
                                 (max(0.0, raw) if raw is not None else None),
                             'mask_extended': info['mask_extended_to_all_for_insertability'],
                             'decoder_calls': info['decoder_calls'],
                             'is_identity_to_template':
                                 bool(seq == list(st['template'])),
                             'baseline_utility': st['baseline_utility'],
                             'utility': float(e_['utility']) if e_ else None})

    def paired(mode_a, mode_b):
        a = {(r['inst'], r['mask_seed']): r for r in rows if r['mode'] == mode_a}
        b = {(r['inst'], r['mask_seed']): r for r in rows if r['mode'] == mode_b}
        d = [a[k]['raw_delta_utility'] - b[k]['raw_delta_utility']
             for k in sorted(set(a) & set(b))
             if a[k]['raw_delta_utility'] is not None and b[k]['raw_delta_utility'] is not None]
        if not d:
            return None
        d = np.asarray(d, dtype=np.float64)
        ci = None
        if len(d) >= 2:
            rng = np.random.default_rng(args.bootstrap_seed)
            draws = d[rng.integers(0, len(d), size=(args.bootstrap_repetitions, len(d)))].mean(axis=1)
            ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        return {'n_pairs': int(len(d)), 'mean': float(d.mean()), 'ci95': ci,
                'n_positive': int((d > 1e-10).sum()), 'n_zero': int((abs(d) <= 1e-10).sum()),
                'n_negative': int((d < -1e-10).sum())}

    summary = {}
    for mode in MODES:
        vals = [r['raw_delta_utility'] for r in rows if r['mode'] == mode
                and r['raw_delta_utility'] is not None]
        acc = [r['accepted_delta_utility'] for r in rows if r['mode'] == mode
               and r['accepted_delta_utility'] is not None]
        summary[mode] = {
            'n': len(vals),
            'mean_raw_delta_utility': float(np.mean(vals)) if vals else None,
            'n_raw_positive': int(sum(1 for v in vals if v > 1e-10)),
            'n_raw_zero': int(sum(1 for v in vals if abs(v) <= 1e-10)),
            'n_raw_negative': int(sum(1 for v in vals if v < -1e-10)),
            'mean_accepted_delta_utility': float(np.mean(acc)) if acc else None,
            'n_hard_feasible': int(sum(1 for r in rows if r['mode'] == mode
                                       and r['hard_feasible'])),
            'n_identity_to_template': int(sum(1 for r in rows if r['mode'] == mode
                                             and r['is_identity_to_template'])),
            'decoder_calls_total': int(sum(r['decoder_calls'] for r in rows
                                           if r['mode'] == mode)),
            'n_mask_extended': int(sum(1 for r in rows if r['mode'] == mode
                                       and r['mask_extended']))}
    contrasts = {'maskco_minus_recon_only': paired('maskco', 'recon_only'),
                 'recon_only_minus_discard': paired('recon_only', 'discard')}
    result = {
        'kind': 'pickup_masked_reconstruction_bounded_development_pilot',
        'protocol_version': 'P-MASKCO-ARM-PILOT-v1',
        'parent_protocol': 'P-LIVE-DEV-PANEL-v1 (registered, not yet run)',
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'n_states': len(states), 'n_rows': len(rows),
        'states': [{k: v for k, v in s.items() if k not in ('env', 'served', 'baseline_accept')}
                   for s in states],
        'per_mode': summary,
        'paired_contrasts': contrasts,
        'mask_fraction': args.mask_fraction, 'guide_weight': args.guide_weight,
        'pilot_seed': args.pilot_seed, 'seeds_per_mode': args.seeds_per_mode,
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'rows': rows,
        'program_main_gate_restated': ('the pickup MAIN gate requires >= 20 utility/day versus '
                                       'cond_hist with a paired CI lower bound > 0, plus a '
                                       'positive mean versus the same-initial-value '
                                       'reconstruction-only control. This pilot has no cond_hist '
                                       'arm and few states, so it cannot address the main gate.'),
        'what_this_measures': [
            'whether the guide-scored reconstruction differs from the distance-only '
            'reconstruction on identical states, masks, seeds and decoder calls'],
        'what_this_does_not_measure': [
            'no cond_hist arm, no strong rolling controller, no acceptance policy over a day',
            'few engineered states: every interval is DESCRIPTIVE, carries no alpha, and cannot '
            'support a stable-online-gain, attribution, impossibility or equivalence claim',
            'the >= 20 utility/day main gate is NOT addressed by this pilot'],
        'stable_online_gain_established': False, 'maskco_attributable': False,
        'pickup_impossibility_proven': False, 'equivalence_established': False,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({'n_states': result['n_states'], 'n_rows': result['n_rows'],
                      'per_mode': summary, 'paired_contrasts': contrasts,
                      'wall_s': result['wall_s']}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--instances', type=int, default=3)
    ap.add_argument('--states-per-instance', type=int, default=1)
    ap.add_argument('--min-clock', type=float, default=2.0)
    ap.add_argument('--min-orders', type=int, default=3)
    ap.add_argument('--max-orders', type=int, default=3)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--guide-weight', type=float, default=1.0)
    ap.add_argument('--pilot-seed', type=int, default=2026120202)
    ap.add_argument('--seeds-per-mode', type=int, default=4)
    ap.add_argument('--budget', type=float, default=710.6548152249378)
    ap.add_argument('--cooling-share', type=float, default=3.7125478804082377)
    ap.add_argument('--bootstrap-seed', type=int, default=2026120212)
    ap.add_argument('--bootstrap-repetitions', type=int, default=2000)
    main(ap.parse_args())
