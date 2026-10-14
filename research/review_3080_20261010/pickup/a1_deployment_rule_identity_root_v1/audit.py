"""Independent audit: re-derive every logged decision from the registered rule text.

The audit does NOT import the runner's deployment_rule module.  It re-implements the
registered formula from REGISTRATION.json, recomputes the calibration table with its own
loop, and then checks the execution log decision by decision.
"""
import bisect
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
REV = {0: 10.0, 1: 20.0, 2: 30.0}


def save(name, value):
    seal.save(HERE / name, value)


def own_e_lower(class_id, quantity, objective):
    delta = max(0.0, float(objective['ambient_c']) - float(objective['target_c'][int(class_id)]))
    return float(quantity) * float(objective['precool_energy_per_unit_per_c']) * delta


def own_calibration(train_ds, objective, lambda_grid, clock_grid):
    """Independent calibration: explicit per-day, per-order loop with no vectorisation."""
    dem = np.asarray(train_ds['demands'], dtype=np.float64)
    rev_t = np.asarray(train_ds['reveal_time'], dtype=np.float64)
    cls_t = np.asarray(train_ds['temp_class'], dtype=np.int64)
    n_days = dem.shape[0]
    table = []
    for lam in lambda_grid:
        row = []
        for t in clock_grid:
            acc = 0.0
            for d in range(n_days):
                for o in range(1, dem.shape[1]):
                    if dem[d, o] <= 0:
                        continue
                    if float(rev_t[d, o]) < t:
                        continue
                    e = own_e_lower(int(cls_t[d, o]), float(dem[d, o]), objective)
                    if REV[int(cls_t[d, o])] - float(lam) * e > 0.0:
                        acc += e
            row.append(acc / max(n_days, 1))
        table.append(row)
    return table


def own_bucket(clock, clock_grid):
    return min(max(bisect.bisect_right(clock_grid, float(clock)) - 1, 0), len(clock_grid) - 1)


def own_decide(e, revenue, clock, b_rem, lambda_grid, clock_grid, g_table):
    if e > b_rem:
        return False, 'over_remaining_budget', None, None
    ti = own_bucket(clock, clock_grid)
    lam = lambda_grid[-1]
    for li, v in enumerate(lambda_grid):
        if g_table[li][ti] <= b_rem:
            lam = v
            break
    if (float(revenue) - lam * float(e)) > 0.0:
        return True, 'admit', lam, ti
    return False, 'below_shadow_price', lam, ti


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

    from dataclasses import replace
    from coldchain_contract import ColdChainContractV2, UnitScale
    c = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=float(REG['frozen']['km_per_unit']),
                      hours_per_time_unit=1.0, speed_kmph=float(REG['frozen']['speed_kmh']))
    prov = list(c.parameter_provenance)
    for i, p in enumerate(prov):
        if p.parameter_path == 'units.distance_km_per_unit':
            prov[i] = replace(p, sensitivity_high=50.0)
    contract = replace(c, units=units, parameter_provenance=tuple(prov))
    contract.validate()
    fz = REG['frozen']
    ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']), contract)
    ck('audit_gate_dataset_identity', dataset_identity(ds) == fz['expected_data_sha256'])
    ck('audit_contract_identity', contract.contract_hash == fz['expected_contract_sha256'])
    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], build_history_times_classes(train_ds),
                                                   k=fz['k_neighbors'])
    ck('audit_budget_matches_registered', abs(float(B) - float(REG['budget_B'])) < 1e-12
       and abs(float(B) - float(R['budget_B'])) < 1e-12)

    objective = REG['rule_spec']['objective']
    lam_grid = R['lambda_grid']
    clock_grid = R['clock_grid']
    table = own_calibration(train_ds, objective, lam_grid, clock_grid)
    diff = float(np.max(np.abs(np.asarray(table) - np.asarray(R['g_table']))))
    ck('audit_calibration_matches_runner', diff <= 1e-9)

    log = [json.loads(l) for l in (HERE / 'DECISION_LOG.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    ck('audit_log_non_empty', len(log) == R['decisions'])

    spent = {}
    mismatches = 0
    per_day_admits = {}
    for k, e in enumerate(log):
        d = e['inst']
        own_e = own_e_lower(e['class_id'], e['quantity'], objective)
        own_spent = spent.get(d, 0.0)
        own_brem = float(B) - own_spent
        admit, reason, lam, ti = own_decide(own_e, REV[e['class_id']], e['clock'], own_brem,
                                            lam_grid, clock_grid, table)
        ok = (abs(own_e - e['e_lower']) < 1e-12
              and abs(own_spent - e['spent_before']) < 1e-9
              and abs(own_brem - e['b_rem']) < 1e-9
              and admit == (e['rule_decision'] == 'admit')
              and reason == e['reason']
              and (lam is None) == (e['lam'] is None)
              and (lam is None or abs(lam - e['lam']) < 1e-12)
              and (ti is None) == (e['bucket'] is None)
              and (ti is None or int(ti) == int(e['bucket'])))
        if not ok:
            mismatches += 1
            if mismatches <= 5:
                ck('audit_decision_mismatch:%d' % k, False)
        if admit:
            spent[d] = own_spent + own_e
            per_day_admits[d] = per_day_admits.get(d, 0) + 1
    ck('audit_all_decisions_reproduced', mismatches == 0)
    ck('audit_admit_count_matches', sum(per_day_admits.values()) == R['admits'])
    ck('audit_reason_counts_match',
       {r: sum(1 for e in log if e['reason'] == r) for r in set(e['reason'] for e in log)} == R['reason_counts'])

    v4 = json.loads((ROOT / REG['v4_artifacts']['result']).read_text(encoding='utf-8'))
    v4rows = {r['i']: r for r in v4['arms'][REG['v4_artifacts']['arm']]}
    for r in R['per_day']:
        ref = v4rows[r['day']]
        ck('audit_day_utility_matches_v4:%d' % r['day'], r['utility'] == ref['utility'])
        ck('audit_recorded_day_row_consistent:%d' % r['day'],
           r['n_served'] == ref['n_served'] and r['energy_kwh'] == ref['energy_kwh'])
        ck('audit_day_hard_and_budget:%d' % r['day'], r['hard_feasible'] is True and not r['budget_violated'])

    ck('exit_codes_consistent', RUN_EXIT['exit_code'] == int(R['failed'] > 0)
       and len(R['failure_names']) == R['failed'] and len(R['checks']) == R['checks_count'])
    ck('no_model_loaded', not any('MASKCO_code' in p or p.lower().endswith('.ckpt')
                                  for p in list(start_closure['files']) + list(start_closure['modules'].values())))
    for flag in ('new_utility_day', 'scientific_family_started', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, R[flag] is False)

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 calibration_max_abs_diff=diff, decisions_checked=len(log), decision_mismatches=mismatches,
                 admits=sum(per_day_admits.values()), reason_counts=R['reason_counts'],
                 per_day_outcomes=[dict(day=r['day'], utility=r['utility'], served=r['n_served'],
                                        energy_kwh=r['energy_kwh']) for r in R['per_day']],
                 rule_identity_established=(mismatches == 0 and diff <= 1e-9),
                 resource=dict(audit_seconds=time.perf_counter() - started),
                 actual_source_files=len(start_closure['files']), actual_source_maps=len(start_closure['modules']),
                 unpassed_gates=['fresh-day utility comparison (not attempted; engineering stage)',
                                 'independent strong-dynamic accept/reject control',
                                 'same-initialisation trained CE and same-task trained random controls',
                                 'multi-round mask-reconstruct-repair under an equal-time request'],
                 new_utility_day=False, scientific_family_started=False, stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json', audit)
    save('AUDIT_RUN_EXIT.json', dict(exit_code=int(audit['failed'] > 0)))
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}))
    return int(audit['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
