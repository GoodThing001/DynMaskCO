"""Independent audit of the pickup budget-reserving admission panel.

Re-derives the frozen identities, the calibration table, the shadow-price decision
arithmetic and the paired differences with its own implementation, and checks that
the pre-declared decision rule was applied to the panel result as registered.
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


def own_calibrate(train_ds, obj, lambda_grid, clock_grid):
    """Independent calibration over the registered pre-cooling-energy cost."""
    from reserving_admission import e_lower_of
    dem = np.asarray(train_ds['demands'], dtype=np.float64)
    rev_t = np.asarray(train_ds['reveal_time'], dtype=np.float64)
    cls_t = np.asarray(train_ds['temp_class'], dtype=np.int64)
    days = []
    for d in range(dem.shape[0]):
        rows = []
        for o in range(1, dem.shape[1]):
            if dem[d, o] <= 0:
                continue
            c = int(cls_t[d, o])
            rows.append((float(rev_t[d, o]), float(e_lower_of(c, float(dem[d, o]), obj)),
                         {0: 10.0, 1: 20.0, 2: 30.0}[c]))
        days.append(sorted(rows, key=lambda r: r[0]))
    table = []
    for lam in lambda_grid:
        row = []
        for t in clock_grid:
            acc = 0.0
            for day in days:
                acc += sum(e for (rev, e, rv) in day if rev >= t and (rv - lam * e) > 0.0)
            row.append(acc / max(len(days), 1))
        table.append(row)
    return table


def make_contract():
    from dataclasses import replace
    from coldchain_contract import ColdChainContractV2, UnitScale
    c = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=float(REG['frozen']['km_per_unit']),
                      hours_per_time_unit=1.0, speed_kmph=float(REG['frozen']['speed_kmh']))
    prov = list(c.parameter_provenance)
    for i, p in enumerate(prov):
        if p.parameter_path == 'units.distance_km_per_unit':
            prov[i] = replace(p, sensitivity_high=50.0)
    c = replace(c, units=units, parameter_provenance=tuple(prov))
    c.validate()
    return c


def main():
    seal.initialize()
    import run_exp_reserve as RR
    from run_identity import dataset_identity
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import build_history_times_classes

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    for path, want in REG['sources'].items():
        ck('registered_source:' + path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == want)

    contract = make_contract()
    fz = REG['frozen']
    ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']), contract)
    ck('audit_dataset_identity', dataset_identity(ds) == fz['expected_data_sha256'])
    ck('audit_contract_identity', contract.contract_hash == fz['expected_contract_sha256'])
    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], build_history_times_classes(train_ds),
                                                   k=fz['k_neighbors'])
    ck('audit_budget_matches', abs(float(B) - fz['expected_budget_B']) < 1e-9
       and abs(float(B) - R['budget_B']) < 1e-9)
    ck('audit_cooling_share_matches', abs(float(cooling_share) - fz['expected_cooling_share']) < 1e-12)

    table = own_calibrate(train_ds, REG['objective'], R['lambda_grid'], R['clock_grid'])
    diff = float(np.max(np.abs(np.asarray(table) - np.asarray(R['g_table']))))
    ck('audit_calibration_table_reproduced', diff <= 1e-9)
    ck('audit_calibration_monotone_in_lambda',
       all(all(table[li][ti] >= table[li + 1][ti] - 1e-9 for ti in range(len(R['clock_grid'])))
           for li in range(len(R['lambda_grid']) - 1)))

    # decision arithmetic re-derived independently
    lam = list(R['lambda_grid'])
    g = R['g_table']

    def lam_for(clock, b_rem):
        ti = int(min(max(np.searchsorted(R['clock_grid'], float(clock), side='right') - 1, 0),
                     len(R['clock_grid']) - 1))
        for li, v in enumerate(lam):
            if g[li][ti] <= b_rem:
                return v
        return lam[-1]

    from reserving_admission import e_lower_of
    e_best = e_lower_of(0, 2.0, REG['objective'])
    e_worst = e_lower_of(2, 2.0, REG['objective'])
    ck('audit_cost_model_is_precooling_not_cooling_share', abs(e_best - 2.1) < 1e-9 and abs(e_worst - 12.9) < 1e-9)
    lam_large = lam_for(8.0, 1e9)
    lam_tight = lam_for(8.0, 200.0)
    ck('audit_shadow_price_lowest_at_large_budget', lam_large == lam[0])
    ck('audit_shadow_price_rises_when_budget_tight', lam_tight >= lam_large)
    ck('audit_tight_budget_prefers_best_class_over_worst',
       ({0: 10.0}[0] - lam_tight * e_best) > 0.0 and ({2: 30.0}[2] - lam_tight * e_worst) <= 0.0)

    # paired differences recomputed from the recorded arm rows
    ctl = R['arms']['control_frozen_saa']
    cand = R['arms']['candidate_reserving']
    ck('audit_panel_length', len(ctl) == REG['panel']['days'] and len(cand) == REG['panel']['days'])
    own = []
    for k in range(REG['panel']['days']):
        ck('audit_control_hard_feasible:%d' % k, ctl[k]['hard_feasible'] is True)
        ck('audit_candidate_hard_feasible:%d' % k, cand[k]['hard_feasible'] is True)
        ck('audit_control_budget_ok:%d' % k, ctl[k]['budget_violated'] is False)
        ck('audit_candidate_budget_ok:%d' % k, cand[k]['budget_violated'] is False)
        own.append(None if (ctl[k]['utility'] is None or cand[k]['utility'] is None)
                   else float(cand[k]['utility'] - ctl[k]['utility']))
    ck('audit_paired_deltas_reproduced',
       all((a is None and b['delta'] is None) or (a is not None and abs(a - b['delta']) < 1e-9)
           for a, b in zip(own, R['paired'])))
    vals = [v for v in own if v is not None]
    mean_delta = float(np.mean(vals)) if vals else None
    ck('audit_mean_delta_reproduced',
       (mean_delta is None and R['mean_delta'] is None)
       or abs(mean_delta - R['mean_delta']) < 1e-9)
    ck('audit_decision_rule_applied_as_registered',
       R['lead_positive'] == bool(mean_delta is not None and mean_delta > 0.0)
       and R['decision'] == ('lead_positive_extend_to_40_days_in_a_new_registered_stage'
                             if R['lead_positive']
                             else 'candidate_dominated_record_and_pivot_to_joint_coupling_relaxation'))
    ck('audit_recorded_reference_rows_unmodified',
       hashlib.sha256((ROOT / fz['recorded_controls']).read_bytes()).hexdigest() == fz['recorded_controls_sha256'])

    ck('exit_codes_consistent', RUN_EXIT['exit_code'] == int(R['failed'] > 0)
       and len(R['failure_names']) == R['failed'] and len(R['checks']) == R['checks_count'])
    ck('no_model_loaded', not any('MASKCO_code' in p or p.lower().endswith('.ckpt')
                                 for p in list(start_closure['files']) + list(start_closure['modules'].values())))
    for flag in ('formal_strong_dynamic_qualified', 'full_request_budget_qualified', 'scientific_data_generated',
                 'new_science_family', 'scientific_family3_started', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, R[flag] is False)

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 calibration_max_abs_diff=diff, paired_deltas=own, mean_delta=mean_delta,
                 lead_positive=R['lead_positive'], decision=R['decision'],
                 panel_summary=dict(
                     control=[dict(day=k, utility=ctl[k]['utility'], served=ctl[k]['n_served'],
                                   revenue=ctl[k]['revenue'], energy=ctl[k]['energy_kwh'])
                              for k in range(REG['panel']['days'])],
                     candidate=[dict(day=k, utility=cand[k]['utility'], served=cand[k]['n_served'],
                                     revenue=cand[k]['revenue'], energy=cand[k]['energy_kwh'])
                                for k in range(REG['panel']['days'])]),
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
