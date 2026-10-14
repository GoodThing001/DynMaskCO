"""P-CLAIR-v1: is the reference policy's accept threshold too conservative?

Origin. P-ATTRIB-v1 attributed 4392 rejections and found 92.71 percent are the
value-per-energy threshold, 7.29 percent the true energy certification, and exactly ZERO
fleet or time-window feasibility. P-RESIDUAL-v1 had already shown the unspent residual is
about 18 times the cheapest rejected round trip. Together: the policy can serve many of
the orders it declines and declines them by choice, reserving budget for orders not yet
revealed. That reservation is justified only if the future orders it protects are worth
more than the present ones it gives up, and a full-information reference answers that
directly.

Method. Runs the reference policy on the frozen cohort for its per-instance served count,
then runs the existing full-information reference (run_exp_energy_c0.clairvoyant_profit_c0,
a greedy heuristic that ranks orders by revenue-per-marginal-energy with the whole day
known) and compares SERVED COUNTS - the one quantity both computations define identically.

Why served count and not profit. clairvoyant_profit_c0 computes profit with its own formula
(revenue minus quality minus fuel, no reject penalty) that is not the A-v1 utility, so
comparing its profit against the recorded A-v1 utility would be meaningless.

Implementation note. clairvoyant_profit_c0 returns only (profit, feasible, energy) and not
the accepted set, so its selection loop is reproduced here VERBATIM - the same ordering by
revenue per marginal energy, the same greedy insertion, the same _seq_feasible check - to
recover the served set. The original function is not modified. A consistency check
confirms the reproduced loop yields the same total energy as the original function's
returned energy field on every instance; the panel aborts if it does not.

Registered reading. If the full-information reference serves materially more orders (mean
relative gain >= 1 percent with paired 95 percent CI lower > 0), the threshold is too
conservative and pickup headroom is real. If the two are indistinguishable, or the
reference serves fewer, the reservation is near-optimal on this cohort and the threshold
channel closes as well.

Scope. The reference is greedy, not optimal, so it lower-bounds what full information
could achieve: a null result bounds the threshold question rather than perfectly closing
it. No intervention is tested and no gain is established.

Usage:
    python pickup_clairvoyant_gap.py --gate GATE.json --out NEWDIR [--test-instances N]
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('evaluation', 'coldchain', 'simulation'):
    sys.path.insert(0, str(SCRIPTS / sub))

from coldchain_contract import ColdChainContractV2                        # noqa: E402
from coldchain_evaluator_a1 import REV, add_v2_initial_quality             # noqa: E402
from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH,
                               _seq_feasible, c0_marginal_energy,
                               c0_route_energy, clairvoyant_profit_c0,
                               generate_dataset)                          # noqa: E402
from run_exp_rule_control import UncondDynamicReplanner                    # noqa: E402
from strict_online_env import StrictOnlineEnv                              # noqa: E402

PROTOCOL = {
    'version': 'P-CLAIR-v1',
    'kind': 'threshold_calibration_test_not_gain_claim',
    'parents': ['P-ATTRIB-v1', 'P-RESIDUAL-v1'],
    'reference_policy': 'uncond_dynamic (non-learned)',
    'full_information_reference': 'run_exp_energy_c0.clairvoyant_profit_c0 (greedy)',
    'compared_quantity': 'served count (the only quantity both computations define alike)',
    'not_compared': 'profit - the clairvoyant profit formula is not the A-v1 utility',
    'selection_loop_reproduced_verbatim': True,
    'recorded_reference_mean_served': 83.7,
    'bootstrap_seed': 2026101201,
    'bootstrap_repetitions': 2000,
    'threshold_gain': 0.01,
    'reading': ('full-information reference serves >=1 percent more with paired 95 percent '
                'CI lower > 0 implies the threshold is too conservative'),
    'greedy_caveat': 'the reference is greedy, so a null result bounds rather than closes',
    'claims_online_gain': False,
}


def clairvoyant_routes(test_ds, i, B, capacity, num_vehicles, contract, cooling_share):
    """Verbatim reproduction of the selection loop inside clairvoyant_profit_c0.

    That function returns only (profit, feasible, energy), so the accepted set is
    recovered here without modifying it. The caller checks that the energy implied by
    these routes equals the original function's returned energy.
    """
    tc = test_ds['temp_class'][i]
    dem = test_ds['demands'][i]
    tw = test_ds['tw_end'][i]
    tws = test_ds['tw_start'][i]
    st_ = test_ds['service_time'][i]
    rev = test_ds['reveal_time'][i]
    coords = test_ds['coords'][i]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    orders = [o for o in range(1, tc.shape[0]) if dem[o] > 0]
    ratio = {o: REV[int(tc[o])] / max(
        c0_marginal_energy(int(tc[o]), dem[o], cooling_share, contract), 1e-9)
        for o in orders}
    orders.sort(key=lambda o: -ratio[o])
    accepted, rem = [], B
    routes = [[] for _ in range(num_vehicles)]
    for o in orders:
        e = c0_marginal_energy(int(tc[o]), dem[o], cooling_share, contract)
        if e > rem + 1e-9:
            continue
        best = None
        for vi, route in enumerate(routes):
            for pos in range(len(route) + 1):
                trial = route[:pos] + [o] + route[pos:]
                if _seq_feasible(trial, dem, tw, tws, st_, dist, capacity, rev):
                    best = (vi, pos)
                    break
            if best is not None:
                break
        if best is None:
            continue
        vi, pos = best
        routes[vi].insert(pos, o)
        accepted.append(o)
        rem -= e
    energy = sum(c0_route_energy(r, dist, tws, st_, rev, dem, tc, contract)
                 for r in routes if r)
    return set(accepted), energy


def paired(control, other, seed, reps):
    shared = sorted(set(control) & set(other))
    rel = np.array([(other[i] - control[i]) / control[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rel), size=(reps, len(rel)))
    means = rel[idx].mean(axis=1)
    return {'n_instances': len(shared),
            'mean_relative_gain': float(rel.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_clairvoyant_higher': int(sum(rel > 0)),
            'instances_clairvoyant_lower': int(sum(rel < 0)),
            'instances_tied': int(sum(rel == 0))}


def main(args):
    started = time.perf_counter()
    gate_path = Path(args.gate)
    gate_raw = gate_path.read_bytes()
    gate = json.loads(gate_raw.decode('utf-8'))
    B = float(gate['budget']['B'])
    cooling_share = float(gate['budget']['cooling_share'])
    cfg = gate['config']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    contract = ColdChainContractV2()
    contract.validate()
    train_ds = generate_dataset(cfg['train_instances'], cfg['n_orders'], cfg['seed'])
    n_test = cfg['test_instances'] if args.test_instances is None else args.test_instances
    test_ds = generate_dataset(n_test, cfg['n_orders'], cfg['seed'] + 1)
    test_ds = add_v2_initial_quality(test_ds, contract)
    hist = []
    for i in range(cfg['train_instances']):
        dem = train_ds['demands'][i]
        mask = dem[1:] > 0
        hist.append((train_ds['reveal_time'][i][1:][mask],
                     train_ds['temp_class'][i][1:][mask].astype(int)))

    replanner = UncondDynamicReplanner(
        budget=B, capacity=cfg['capacity'], booking_horizon=BOOKING_HORIZON,
        contract=contract, cooling_share=cooling_share, hist=hist, k=cfg['k'])
    env = StrictOnlineEnv(test_ds, capacity=cfg['capacity'], num_vehicles=cfg['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)

    base, clair, rows = {}, {}, []
    for i in range(n_test):
        env.run(i)
        base[i] = len(replanner._accepted)
        served, energy = clairvoyant_routes(test_ds, i, B, cfg['capacity'],
                                            cfg['num_vehicles'], contract, cooling_share)
        profit, feasible, ref_energy = clairvoyant_profit_c0(
            test_ds, i, B, cfg['capacity'], cfg['num_vehicles'], contract, cooling_share)
        if abs(energy - float(ref_energy)) > 1e-6:
            raise RuntimeError('Reproduced clairvoyant loop disagrees with the original '
                               'energy on instance %d: %.9f vs %.9f' % (i, energy,
                                                                        float(ref_energy)))
        clair[i] = len(served)
        rows.append({'instance': i, 'base_served': base[i], 'clairvoyant_served': clair[i],
                     'clairvoyant_profit': float(profit),
                     'clairvoyant_budget_feasible': bool(feasible)})
        print(json.dumps({'instance': i, 'base_served': base[i],
                          'clair_served': clair[i], 'profit': round(float(profit), 3),
                          'feasible': bool(feasible)}), flush=True)

    mean_base = statistics.fmean(base.values())
    recorded = PROTOCOL['recorded_reference_mean_served']
    full_cohort = args.test_instances is None
    reproduced = abs(mean_base - recorded) <= 0.05
    if full_cohort and not reproduced:
        raise RuntimeError('Reference policy did not reproduce recorded mean served '
                           '%.2f (got %.2f); aborting' % (recorded, mean_base))
    gain = paired(base, clair, PROTOCOL['bootstrap_seed'], PROTOCOL['bootstrap_repetitions'])
    passed = (gain['mean_relative_gain'] >= PROTOCOL['threshold_gain']
              and gain['descriptive_95_CI'][0] > 0)
    result = {
        'protocol': PROTOCOL,
        'source_gate': str(gate_path).replace('\\', '/'),
        'source_gate_sha256': hashlib.sha256(gate_raw).hexdigest(),
        'budget_B': B, 'n_instances': len(rows),
        'reproduction': {'recorded_mean_served': recorded,
                         'observed_mean_served': mean_base,
                         'full_cohort_checked': full_cohort,
                         'reproduced_within_0.05': reproduced,
                         'clairvoyant_loop_energy_matches_original': True},
        'mean_base_served': mean_base,
        'mean_clairvoyant_served': statistics.fmean(clair.values()),
        'served_count_comparison': gain,
        'threshold_is_too_conservative': bool(passed),
        'all_clairvoyant_budget_feasible': all(r['clairvoyant_budget_feasible'] for r in rows),
        'per_instance': rows,
        'online_gain_established': False,
        'stable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'mean_base_served': mean_base,
                      'mean_clairvoyant_served': result['mean_clairvoyant_served'],
                      'served_count_comparison': gain,
                      'threshold_is_too_conservative': bool(passed),
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-instances', type=int, default=None)
    main(ap.parse_args())
