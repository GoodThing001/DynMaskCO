"""P-RESIDUAL-v1: is the unspent energy budget an indivisibility artefact or a choice?

Origin. P-BUDGET-v1 showed, by arithmetic over the frozen A-v1 gate, that the C0 energy
budget covers 40.0 percent of an unconditional day while the reference policy stops at
90.7 percent utilisation, leaving a residual worth about 10.27 percent of served orders
as an average-based upper bound. It also showed the gain is not efficiency but the
residual itself. That leaves one question, which this panel answers directly:

    after the reference policy stops, can ANY single rejected order still be served
    inside the leftover budget?

If yes, the residual is a strategic choice - the policy is holding budget back - and the
10.27 percent is only reachable by changing the decision rule. If no single order fits,
the residual is indivisible and only a multi-order bundle could use it, which is the
premise of the registered node n6.

Method. Rebuilds the frozen A-v1 configuration exactly (same dataset generator, seed,
order count, fleet, capacity, booking horizon), takes the energy budget B and the
cooling share from the recorded gate rather than recomputing the training-side
calibration, runs the reference policy uncond_dynamic, and for every rejected order
evaluates the true C0 energy of a dedicated single-order round trip with
c0_route_energy - the same transition_segment physics the environment uses - plus an
explicit time-window and depot-return check that c0_route_energy itself does not perform.

Reproduction check. The mean served count must reproduce the recorded 83.7; the panel
aborts rather than reports if it does not.

Scope, stated before the numbers. The single-order round trip is evaluated from t = 0
with a fresh vehicle, which is OPTIMISTIC: a real extra dispatch would start late in the
day, so it would accumulate at least as much standby refrigeration and would face
tighter time windows. A finding that no order fits under the optimistic evaluation
therefore implies none fits in reality, but a finding that some order fits does not by
itself prove the policy could have served it - it only proves the residual is not
physically indivisible at the single-order granularity.

Usage:
    python pickup_residual_census.py --gate GATE.json --out NEWDIR [--test-instances N]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('evaluation', 'coldchain', 'simulation'):
    sys.path.insert(0, str(SCRIPTS / sub))

from coldchain_contract import ColdChainContractV2                 # noqa: E402
from coldchain_evaluator_a1 import add_v2_initial_quality          # noqa: E402
from run_exp_energy_c0 import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH,
                               c0_energy_from_traces, c0_route_energy,
                               generate_dataset)                   # noqa: E402
from run_exp_rule_control import UncondDynamicReplanner            # noqa: E402
from strict_online_env import StrictOnlineEnv                      # noqa: E402

PROTOCOL = {
    'version': 'P-RESIDUAL-v1',
    'kind': 'feasibility_census_of_the_unspent_energy_budget',
    'parent': 'P-BUDGET-v1',
    'question': ('can any single rejected order still be served inside the leftover '
                 'budget after the reference policy stops?'),
    'generator': 'run_exp_energy_c0.generate_dataset',
    'policy': 'uncond_dynamic (non-learned reference)',
    'energy_function': 'run_exp_energy_c0.c0_route_energy (transition_segment physics)',
    'budget_source': 'recorded A-v1 gate, not recomputed',
    'optimistic_because': ('single-order round trip evaluated from t=0 with a fresh '
                           'vehicle; a real extra dispatch starts late in the day and '
                           'faces at least as much standby refrigeration and tighter windows'),
    'claims_online_gain': False,
    'recorded_reference_mean_served': 83.7,
}


def single_route_checks(o, dist, tws, tw_end, st_, rev, horizon, speed):
    """Time-window and depot-return feasibility that c0_route_energy does not test."""
    depart = max(0.0, float(rev[o]))
    arrive = depart + float(dist[0, o]) / speed
    sstart = max(arrive, float(tws[o]))
    sfinish = sstart + float(st_[o])
    if sfinish > float(tw_end[o]) + 1e-9:
        return False
    back = sfinish + float(dist[o, 0]) / speed
    return back <= horizon + 1e-9


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
        times = train_ds['reveal_time'][i][1:][mask]
        classes = train_ds['temp_class'][i][1:][mask].astype(int)
        hist.append((times, classes))

    replanner = UncondDynamicReplanner(
        budget=B, capacity=cfg['capacity'], booking_horizon=BOOKING_HORIZON,
        contract=contract, cooling_share=cooling_share, hist=hist, k=cfg['k'])
    env = StrictOnlineEnv(test_ds, capacity=cfg['capacity'], num_vehicles=cfg['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=replanner,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    speed = SPEED_KMH / KM_PER_UNIT

    rows = []
    for i in range(n_test):
        traces, _ = env.run(i)
        used = c0_energy_from_traces(traces)
        residual = B - used
        rejected = sorted(int(o) for o in replanner._rejected)
        accepted = sorted(int(o) for o in replanner._accepted)
        dist = np.asarray(env.dist_mat[i], float)
        tws = np.asarray(test_ds['tw_start'][i], float)
        tw_end = np.asarray(test_ds['tw_end'][i], float)
        st_ = np.asarray(test_ds['service_time'][i], float)
        rev = np.asarray(test_ds['reveal_time'][i], float)
        dem = np.asarray(test_ds['demands'][i], float)
        tc = np.asarray(test_ds['temp_class'][i], int)
        horizon = float(tw_end.max())
        fits, costs = [], []
        for o in rejected:
            e = c0_route_energy([o], dist, tws, st_, rev, dem, tc, contract)
            ok = single_route_checks(o, dist, tws, tw_end, st_, rev, horizon, speed)
            costs.append((e, o, ok))
            if ok and e <= residual + 1e-9:
                fits.append(o)
        costs.sort()
        rows.append({
            'instance': i, 'served': len(accepted), 'rejected': len(rejected),
            'energy_used': used, 'residual': residual,
            'residual_fraction': residual / B,
            'rejected_single_route_min_energy': costs[0][0] if costs else None,
            'min_cost_order_feasible': bool(costs and costs[0][2]),
            'n_rejected_orders_that_fit_residual': len(fits),
            'residual_over_min_single_cost': (residual / costs[0][0]) if costs else None,
        })
        print(json.dumps({'instance': i, 'served': len(accepted),
                          'rejected': len(rejected), 'residual': round(residual, 3),
                          'min_rejected_single_cost': (round(costs[0][0], 3)
                                                       if costs else None),
                          'n_fit': len(fits)}), flush=True)

    mean_served = statistics.fmean(r['served'] for r in rows)
    recorded = PROTOCOL['recorded_reference_mean_served']
    full_cohort = args.test_instances is None
    reproduced = abs(mean_served - recorded) <= 0.05
    # The reproduction check is only meaningful on the full registered cohort: a
    # truncated run compares a subset mean against the recorded full-cohort mean.
    if full_cohort and not reproduced:
        raise RuntimeError('Reference policy did not reproduce the recorded mean served '
                           '%.2f (got %.2f); aborting' % (recorded, mean_served))

    n_any = sum(1 for r in rows if r['n_rejected_orders_that_fit_residual'] > 0)
    result = {
        'protocol': PROTOCOL,
        'source_gate': str(gate_path).replace('\\', '/'),
        'source_gate_sha256': hashlib.sha256(gate_raw).hexdigest(),
        'budget_B': B, 'cooling_share': cooling_share, 'n_instances': len(rows),
        'reproduction': {'recorded_mean_served': recorded,
                         'observed_mean_served': mean_served,
                         'full_cohort_checked': full_cohort,
                         'reproduced_within_0.05': reproduced},
        'aggregate': {
            'mean_served': statistics.fmean(r['served'] for r in rows),
            'mean_rejected': statistics.fmean(r['rejected'] for r in rows),
            'mean_residual': statistics.fmean(r['residual'] for r in rows),
            'mean_residual_fraction': statistics.fmean(r['residual_fraction'] for r in rows),
            'mean_min_rejected_single_cost': statistics.fmean(
                r['rejected_single_route_min_energy'] for r in rows
                if r['rejected_single_route_min_energy'] is not None),
            'instances_with_at_least_one_order_fitting': n_any,
            'instances_total': len(rows),
            'total_orders_that_fit': sum(r['n_rejected_orders_that_fit_residual']
                                         for r in rows),
        },
        'verdict': ('residual_is_strategic_not_indivisible' if n_any > 0
                    else 'residual_is_indivisible_at_single_order_granularity'),
        'per_instance': rows,
        'online_gain_established': False,
        'stable_gain_established': False,
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'reproduction': result['reproduction'],
                      'aggregate': result['aggregate'],
                      'verdict': result['verdict'],
                      'wall_s': round(result['wall_s'], 1)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-instances', type=int, default=None)
    main(ap.parse_args())
