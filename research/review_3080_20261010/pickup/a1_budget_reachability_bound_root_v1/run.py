"""Pickup A-v1: a valid relaxation upper bound on the daily utility, paired with recorded controls.

Question: how much utility headroom exists above the strongest recorded non-learning
control, under the frozen A-v1 contract?  The only reference used until now
("clairvoyant") is a greedy heuristic that rules beat, so it bounds nothing.

Design (every relaxation is one-sided, so the result is an upper bound on the
best achievable utility):
  R1 servability  : in any schedule order i can only be served at or after
                    max(straight-line travel time, reveal_time), and service must
                    finish by its time window.  Straight-line distance is a lower
                    bound on any real travel distance.
  R2 energy       : servicing i necessarily charges at least its pre-cooling
                    energy q_i * 0.15 * (25 - target_c) plus a non-negative door
                    term, so the served set must satisfy sum(e_lower) <= B.
  R3 vehicle time : servicing i consumes at least one out-and-back trip plus the
                    loading service time, so sum(tau) <= num_vehicles * horizon.
The daily utility is revenue - fuel - reject_loss <= revenue, so the fractional
knapsack over each of R2/R3 (and their minimum) bounds the utility.

The recorded control arms are used as a falsification oracle: any arm-day whose
utility, served count, revenue or accumulated vehicle time violates the
corresponding bound invalidates that relaxation and is reported as such.
"""
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
sys.path.insert(0, str(ROOT / 'tools'))
for sub in ('scripts', 'scripts/evaluation', 'scripts/coldchain', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / sub))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import numpy as np

import engineering_stage_closure_v1 as seal

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))


def save(name, value):
    seal.save(HERE / name, value)


def rev_of(cls):
    return float(REG['objective']['revenue_by_class'][str(int(cls))])


def e_lower(q, cls):
    th = REG['objective']
    delta = max(0.0, th['ambient_c'] - th['target_c'][int(cls)])
    return float(q) * th['precool_energy_per_unit_per_c'] * delta + th['door_heat_kwh'][int(cls)]


def frac_knapsack(values, costs, cap):
    """Fractional knapsack = valid upper bound on the 0/1 knapsack."""
    order = sorted(range(len(values)), key=lambda i: (-(values[i] / costs[i]) if costs[i] > 0 else 0.0))
    total = 0.0
    left = float(cap)
    for i in order:
        if values[i] <= 0:
            continue
        c = costs[i]
        if c <= 0:
            total += values[i]
            continue
        if c <= left:
            total += values[i]
            left -= c
        else:
            total += values[i] * (left / c)
            break
    return total


def make_contract():
    """The frozen A-v1 contract, rebuilt from the unchanged contract module only."""
    from dataclasses import replace
    from coldchain_contract import ColdChainContractV2, UnitScale
    contract = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=float(REG['frozen']['km_per_unit']),
                      hours_per_time_unit=1.0, speed_kmph=float(REG['frozen']['speed_kmh']))
    provenance = list(contract.parameter_provenance)
    for i, p in enumerate(provenance):
        if p.parameter_path == 'units.distance_km_per_unit':
            provenance[i] = replace(p, sensitivity_high=50.0)
    contract = replace(contract, units=units, parameter_provenance=tuple(provenance))
    contract.validate()
    return contract


def main():
    seal.initialize()
    import run_exp_reserve as R
    from run_identity import dataset_identity, dataset_meta_sha256
    from coldchain_evaluator_a1 import add_v2_initial_quality

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    # ---- frozen dataset and contract identity ---------------------------------
    contract = make_contract()
    gate_ds = add_v2_initial_quality(R.generate_dataset(REG['frozen']['gate_instances'],
                                                        REG['frozen']['n_orders'],
                                                        REG['frozen']['gate_seed']), contract)
    data_sha = dataset_identity(gate_ds)
    meta_sha = dataset_meta_sha256(gate_ds)
    ck('frozen_dataset_identity_matches_recorded', data_sha == REG['frozen']['expected_data_sha256'])
    ck('frozen_dataset_meta_matches_recorded', meta_sha == REG['frozen']['expected_meta_sha256'])
    ck('frozen_contract_identity_matches_recorded', contract.contract_hash == REG['frozen']['expected_contract_sha256'])

    controls_path = ROOT / REG['frozen']['recorded_controls']
    ck('recorded_controls_sha256',
       hashlib.sha256(controls_path.read_bytes()).hexdigest() == REG['frozen']['recorded_controls_sha256'])
    controls = json.loads(controls_path.read_text(encoding='utf-8'))
    per_day = controls['gate'][REG['objective']['tier']]['per_day']

    # ---- per-day bound ---------------------------------------------------------
    th = REG['objective']
    depot = np.asarray(th['depot_coord'], np.float64)
    km_per_unit = float(REG['frozen']['km_per_unit'])
    speed = float(REG['frozen']['speed_kmh'])
    horizon = float(REG['frozen']['booking_horizon'])
    vehicles = int(REG['frozen']['num_vehicles'])
    service_h = float(REG['frozen']['service_time_h'])
    B = float(REG['frozen']['budget_B'])

    days = []
    for i in range(REG['frozen']['gate_instances']):
        coords = np.asarray(gate_ds['coords'][i], np.float64)
        q = np.asarray(gate_ds['demands'][i], np.float64)
        cls = np.asarray(gate_ds['temp_class'][i], np.int64)
        reveal = np.asarray(gate_ds['reveal_time'][i], np.float64)
        tw_end = np.asarray(gate_ds['tw_end'][i], np.float64)
        universe = [o for o in range(1, len(q)) if q[o] > 0]
        rec = dict(i=i, n_universe=len(universe))
        for variant, use_horizon in (('tw_only', False), ('tw_and_horizon', True)):
            feas, values, energy, tau, dist = [], [], [], [], []
            for o in universe:
                d_km = km_per_unit * float(np.linalg.norm(coords[o] - depot))
                t = d_km / speed
                start = max(t, float(reveal[o]))
                ok = start + service_h <= float(tw_end[o])
                if use_horizon:
                    ok = ok and (start + service_h + t <= horizon)
                if not ok:
                    continue
                feas.append(o)
                values.append(rev_of(cls[o]))
                energy.append(e_lower(q[o], cls[o]))
                tau.append(2.0 * t + service_h)
                dist.append(d_km)
            rec[variant] = dict(
                reachable=len(feas),
                reachable_revenue=float(np.sum(values)),
                energy_lower_total=float(np.sum(energy)),
                vehicle_hours_lower_total=float(np.sum(tau)),
                total_distance_lower_km=float(np.sum(dist)),
                lp_energy_bound=frac_knapsack(values, energy, B),
                lp_time_bound=frac_knapsack(values, tau, vehicles * horizon),
                lp_min_bound=min(frac_knapsack(values, energy, B),
                                 frac_knapsack(values, tau, vehicles * horizon)),
            )
        days.append(rec)

    # ---- recorded arms as a falsification oracle --------------------------------
    arms = list(per_day)
    violations = OrderedDict()
    for arm in arms:
        rows = per_day[arm]
        for variant in ('tw_only', 'tw_and_horizon'):
            v = dict(utility_gt_bound=[], served_gt_reachable=[], revenue_gt_reachable=[], time_gt_capacity=[])
            for r, row in zip(days, rows):
                d = r[variant]
                if float(row['utility']) > d['lp_min_bound'] + 1e-9:
                    v['utility_gt_bound'].append(int(r['i']))
                if int(row['n_served']) > d['reachable']:
                    v['served_gt_reachable'].append(int(r['i']))
                if float(row['revenue']) > d['reachable_revenue'] + 1e-9:
                    v['revenue_gt_reachable'].append(int(r['i']))
                if d['vehicle_hours_lower_total'] > vehicles * horizon + 1e-9:
                    v['time_gt_capacity'].append(int(r['i']))
            violations.setdefault(arm, {})[variant] = v

    for arm in arms:
        for variant in ('tw_only', 'tw_and_horizon'):
            v = violations[arm][variant]
            tag = '%s:%s' % (arm, variant)
            ck('no_recorded_utility_above_bound:' + tag, not v['utility_gt_bound'])
            ck('no_recorded_served_count_above_reachable:' + tag, not v['served_gt_reachable'])
            ck('no_recorded_revenue_above_reachable:' + tag, not v['revenue_gt_reachable'])
            ck('per_day_vehicle_hours_within_capacity:' + tag, not v['time_gt_capacity'])

    # ---- the bracket -----------------------------------------------------------
    bracket = {}
    for variant in ('tw_only', 'tw_and_horizon'):
        ub = [r[variant]['lp_min_bound'] for r in days]
        entry = dict(mean_upper_bound=float(np.mean(ub)), min_upper_bound=float(np.min(ub)),
                     max_upper_bound=float(np.max(ub)),
                     mean_reachable=[int(np.mean([r[variant]['reachable'] for r in days]))])
        for arm in arms:
            u = [float(row['utility']) for row in per_day[arm]]
            entry[arm] = dict(mean_utility=float(np.mean(u)),
                              mean_gap=float(np.mean(ub) - np.mean(u)),
                              min_gap=float(np.min(np.asarray(ub) - np.asarray(u))))
        bracket[variant] = entry

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    result = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                  failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                  stage='pickup_a1_budget_reachability_bound',
                  dataset_sha256=data_sha, dataset_meta_sha256=meta_sha, contract_sha256=contract.contract_hash,
                  tier=REG['objective']['tier'], budget_B=B, num_vehicles=vehicles, horizon=horizon,
                  days=days, bracket=bracket, violations=violations,
                  source_actual_files=len(start_closure['files']), source_actual_maps=len(start_closure['modules']),
                  elapsed_seconds=time.perf_counter() - started,
                  formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                  scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                  stable_online_gain=False)
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'days', 'violations')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
