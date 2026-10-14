"""Independent audit of the pickup A-v1 headroom upper bound.

Re-derives the frozen dataset identity, the per-order relaxations, the knapsack
bounds and the recorded-arm falsification checks with its own implementation, and
cross-checks the recorded control rows byte-for-byte against the sealed file.
"""
import hashlib
import json
import os
import sys
import time
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
R = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))
RUN_EXIT = json.loads((HERE / 'RUN_EXIT.json').read_text(encoding='utf-8'))


def save(name, value):
    seal.save(HERE / name, value)


def own_frac_knapsack(values, costs, cap):
    """Independent fractional-knapsack upper bound by explicit ratio sorting."""
    items = [(values[i], costs[i]) for i in range(len(values)) if values[i] > 0]
    items.sort(key=lambda vc: -(vc[0] / vc[1]) if vc[1] > 0 else float('-inf'))
    total, left = 0.0, float(cap)
    for v, c in items:
        if c <= 0:
            total += v
        elif c <= left:
            total += v
            left -= c
        else:
            total += v * (left / c)
            break
    return total


def make_contract():
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
    import run_exp_reserve as RR
    from run_identity import dataset_identity, dataset_meta_sha256
    from coldchain_evaluator_a1 import add_v2_initial_quality

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    for path, want in REG['sources'].items():
        ck('registered_source:' + path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == want)

    contract = make_contract()
    ds = add_v2_initial_quality(RR.generate_dataset(REG['frozen']['gate_instances'],
                                                    REG['frozen']['n_orders'],
                                                    REG['frozen']['gate_seed']), contract)
    ck('audit_dataset_identity_matches_recorded', dataset_identity(ds) == REG['frozen']['expected_data_sha256'])
    ck('audit_dataset_meta_matches_recorded', dataset_meta_sha256(ds) == REG['frozen']['expected_meta_sha256'])
    ck('audit_contract_identity_matches_recorded', contract.contract_hash == REG['frozen']['expected_contract_sha256'])
    ck('recorded_result_identity_matches',
       R['dataset_sha256'] == REG['frozen']['expected_data_sha256']
       and R['contract_sha256'] == REG['frozen']['expected_contract_sha256'])

    controls = json.loads((ROOT / REG['frozen']['recorded_controls']).read_text(encoding='utf-8'))
    per_day = controls['gate'][REG['objective']['tier']]['per_day']
    ck('recorded_rows_unmodified',
       hashlib.sha256((ROOT / REG['frozen']['recorded_controls']).read_bytes()).hexdigest()
       == REG['frozen']['recorded_controls_sha256'])
    ck('recorded_days', all(len(v) == REG['frozen']['gate_instances'] for v in per_day.values()))

    th = REG['objective']
    depot = np.asarray(th['depot_coord'], np.float64)
    kmu, spd = float(REG['frozen']['km_per_unit']), float(REG['frozen']['speed_kmh'])
    horizon, vehicles = float(REG['frozen']['booking_horizon']), int(REG['frozen']['num_vehicles'])
    svc, B = float(REG['frozen']['service_time_h']), float(REG['frozen']['budget_B'])
    rev = {int(k): float(v) for k, v in th['revenue_by_class'].items()}

    ub = {v: [] for v in ('tw_only', 'tw_and_horizon')}
    reach = {v: [] for v in ub}
    rev_reach = {v: [] for v in ub}
    time_total = {v: [] for v in ub}
    for i in range(REG['frozen']['gate_instances']):
        coords = np.asarray(ds['coords'][i], np.float64)
        q = np.asarray(ds['demands'][i], np.float64)
        cls = np.asarray(ds['temp_class'][i], np.int64)
        reveal = np.asarray(ds['reveal_time'][i], np.float64)
        tw_end = np.asarray(ds['tw_end'][i], np.float64)
        universe = [o for o in range(1, len(q)) if q[o] > 0]
        for variant, use_h in (('tw_only', False), ('tw_and_horizon', True)):
            values, energy, tau = [], [], []
            for o in universe:
                d = kmu * float(np.sqrt(((coords[o] - depot) ** 2).sum()))
                t = d / spd
                start = max(t, float(reveal[o]))
                ok = start + svc <= float(tw_end[o])
                if use_h:
                    ok = ok and start + svc + t <= horizon
                if not ok:
                    continue
                dt = max(0.0, th['ambient_c'] - th['target_c'][int(cls[o])])
                values.append(rev[int(cls[o])])
                energy.append(float(q[o]) * th['precool_energy_per_unit_per_c'] * dt
                              + th['door_heat_kwh'][int(cls[o])])
                tau.append(2.0 * t + svc)
            be = own_frac_knapsack(values, energy, B)
            bt = own_frac_knapsack(values, tau, vehicles * horizon)
            ub[variant].append(min(be, bt))
            reach[variant].append(len(values))
            rev_reach[variant].append(float(np.sum(values)))
            time_total[variant].append(float(np.sum(tau)))

    variants = ('tw_only', 'tw_and_horizon')
    for variant in variants:
        ck('audit_matches_recorded_bound:' + variant,
           np.allclose(np.asarray(ub[variant]), np.asarray([d[variant]['lp_min_bound'] for d in R['days']]),
                       rtol=1e-9, atol=1e-6))
        ck('audit_matches_recorded_reachable:' + variant,
           reach[variant] == [d[variant]['reachable'] for d in R['days']])
        for arm in per_day:
            rows = per_day[arm]
            ck('audit_no_utility_above_bound:%s:%s' % (arm, variant),
               all(float(r['utility']) <= ub[variant][k] + 1e-9 for k, r in enumerate(rows)))
            ck('audit_no_served_above_reachable:%s:%s' % (arm, variant),
               all(int(r['n_served']) <= reach[variant][k] for k, r in enumerate(rows)))
            ck('audit_no_revenue_above_reachable:%s:%s' % (arm, variant),
               all(float(r['revenue']) <= rev_reach[variant][k] + 1e-9 for k, r in enumerate(rows)))

    bracket = {}
    for variant in variants:
        entry = dict(mean_upper_bound=float(np.mean(ub[variant])),
                     min_upper_bound=float(np.min(ub[variant])), max_upper_bound=float(np.max(ub[variant])))
        for arm in per_day:
            u = np.asarray([float(r['utility']) for r in per_day[arm]])
            entry[arm] = dict(mean_utility=float(u.mean()),
                              mean_gap=float(np.mean(ub[variant]) - u.mean()),
                              min_gap=float(np.min(np.asarray(ub[variant]) - u)))
        bracket[variant] = entry
    ck('audit_bracket_matches_recorded',
       all(abs(bracket[v]['mean_upper_bound'] - R['bracket'][v]['mean_upper_bound']) < 1e-6 for v in variants))
    ck('audit_bracket_recomputed_reports_same_arm_gaps',
       all(abs(bracket[v][arm]['mean_gap'] - R['bracket'][v][arm]['mean_gap']) < 1e-6
           for v in variants for arm in per_day))

    ck('exit_codes_consistent', RUN_EXIT['exit_code'] == int(R['failed'] > 0)
       and len(R['failure_names']) == R['failed']
       and len(R['checks']) == R['checks_count'])
    ck('no_model_or_policy_was_loaded',
       not any('MASKCO_code' in p or p.lower().endswith('.ckpt')
               for p in list(start_closure['files']) + list(start_closure['modules'].values())))
    for flag in ('formal_strong_dynamic_qualified', 'full_request_budget_qualified', 'scientific_data_generated',
                 'new_science_family', 'scientific_family3_started', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, R[flag] is False)

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 dataset_sha256=dataset_identity(ds), bracket=bracket,
                 recorded_bracket=R['bracket'],
                 actual_source_files=len(start_closure['files']), actual_source_maps=len(start_closure['modules']),
                 audit_seconds=time.perf_counter() - started,
                 formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                 scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                 stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json', audit)
    save('AUDIT_RUN_EXIT.json', dict(exit_code=int(audit['failed'] > 0)))
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}))
    return int(audit['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
