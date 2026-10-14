"""Independent audit of the current-source pipeline deployment qualification."""
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

import numpy as np  # noqa: E402

import engineering_stage_closure_v1 as seal  # noqa: E402

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))
R = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))
RUN_EXIT = json.loads((HERE / 'RUN_EXIT.json').read_text(encoding='utf-8'))
REV = {0: 10.0, 1: 20.0, 2: 30.0}


def save(name, value):
    seal.save(HERE / name, value)


def own_e_lower(class_id, quantity, obj):
    delta = max(0.0, float(obj['ambient_c']) - float(obj['target_c'][int(class_id)]))
    return float(quantity) * float(obj['precool_energy_per_unit_per_c']) * delta


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
    ck('audit_budget_matches', abs(float(B) - float(REG['budget_B'])) < 1e-12)

    v3 = ROOT / REG['rule_stage']['directory']
    ck('audit_rule_stage_registration_hash',
       hashlib.sha256((v3 / 'REGISTRATION.json').read_bytes()).hexdigest() == REG['rule_stage']['registration_sha256'])
    ck('audit_rule_stage_result_hash',
       hashlib.sha256((v3 / 'RESULT.json').read_bytes()).hexdigest() == REG['rule_stage']['result_sha256'])
    rule = json.loads((v3 / 'RESULT.json').read_text(encoding='utf-8'))
    v4 = json.loads((ROOT / REG['v4_artifacts']['result']).read_text(encoding='utf-8'))
    v4rows = {a: {r['i']: r for r in v4['arms'][a]} for a in ('candidate_reserving', 'control_frozen_saa')}

    # day rows re-checked independently against the v4 records
    for k, d in enumerate(R['days']):
        ck('audit_rule_row_matches_v4:%d' % d,
           R['rule_utilities'][k] == v4rows['candidate_reserving'][d]['utility'])
        ck('audit_control_row_matches_v4:%d' % d,
           R['control_utilities'][k] == v4rows['control_frozen_saa'][d]['utility'])
        ck('audit_paired_delta_matches_v4:%d' % d,
           abs(R['paired_deltas'][k] - (v4rows['candidate_reserving'][d]['utility']
                                        - v4rows['control_frozen_saa'][d]['utility'])) < 1e-9)
    ck('audit_mean_delta', abs(float(np.mean(R['paired_deltas'])) - R['mean_paired_delta']) < 1e-12)

    # the rule arm log re-derived from the registered formula with its own implementation
    log = [json.loads(l) for l in (HERE / 'ONLINE_ARM_LOG.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    ck('audit_online_log_has_rule_entries', len(log) > 0 and all(e['arm'] == 'rule' for e in log))
    lam_grid, clock_grid, table = rule['lambda_grid'], rule['clock_grid'], rule['g_table']
    spent, mismatches = {}, 0
    admits = accepted = 0
    for k, e in enumerate(log):
        d = e['inst']
        own_e = own_e_lower(e['class_id'], e['quantity'], REG['online']['objective'])
        own_spent = spent.get(d, 0.0)
        own_brem = float(B) - own_spent
        if own_e > own_brem:
            own = (False, 'over_remaining_budget', None, None)
        else:
            ti = min(max(bisect.bisect_right(clock_grid, float(e['clock'])) - 1, 0), len(clock_grid) - 1)
            lam = lam_grid[-1]
            for li, v in enumerate(lam_grid):
                if table[li][ti] <= own_brem:
                    lam = v
                    break
            ok = (REV[e['class_id']] - lam * own_e) > 0.0
            own = (ok, 'admit' if ok else 'below_shadow_price', lam, ti)
        good = (abs(own_e - e['e_lower']) < 1e-12 and abs(own_spent - e['spent_before']) < 1e-9
                and abs(own_brem - e['b_rem']) < 1e-9
                and own[0] == (e['rule_decision'] == 'admit') and own[1] == e['reason']
                and (own[2] is None) == (e['lam'] is None)
                and (own[2] is None or abs(own[2] - e['lam']) < 1e-12))
        if not good:
            mismatches += 1
        if own[0]:
            admits += 1
        if e['base_decision'] == 'accepted':
            spent[d] = own_spent + own_e
            accepted += 1
    ck('audit_online_rule_decisions_reproduced', mismatches == 0)
    ck('audit_online_admits_match', admits == R['rule_decisions']['admit'])
    ck('audit_online_base_accepts_match', accepted == R['rule_decisions']['base_accepted'])
    ck('audit_control_not_rule_filtered',
       R['control_decisions']['total'] == 0 and R['control_utilities'] != R['rule_utilities'])

    ck('exit_codes_consistent', RUN_EXIT['exit_code'] == int(R['failed'] > 0)
       and len(R['failure_names']) == R['failed'] and len(R['checks']) == R['checks_count'])
    ck('no_model_loaded', not any('MASKCO_code' in p or p.lower().endswith('.ckpt')
                                  for p in list(start_closure['files']) + list(start_closure['modules'].values())))
    for flag in ('fresh_utility_day', 'new_utility_day', 'scientific_family_started', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, R[flag] is False)
    ck('no_timeouts_recorded', all(t == 0 for arm in R['timeouts'] for t in R['timeouts'][arm]))
    ck('day_wall_times_stable_against_v4',
       all(R['request_seconds'][arm][k] <= 3.0 * float(REG['v4_day_elapsed'][arm][str(d)]) + 30.0
           for arm in R['request_seconds'] for k, d in enumerate(R['days'])))

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 rule_utilities=R['rule_utilities'], control_utilities=R['control_utilities'],
                 paired_deltas=R['paired_deltas'], mean_paired_delta=R['mean_paired_delta'],
                 online_decisions_checked=len(log), online_decision_mismatches=mismatches,
                 online_admits=admits, online_base_accepts=accepted,
                 deployment_qualified=(R['failed'] == 0 and mismatches == 0),
                 resource=dict(audit_seconds=time.perf_counter() - started),
                 actual_source_files=len(start_closure['files']), actual_source_maps=len(start_closure['modules']),
                 fresh_utility_day=False, new_utility_day=False, scientific_family_started=False,
                 stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json', audit)
    save('AUDIT_RUN_EXIT.json', dict(exit_code=int(audit['failed'] > 0)))
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}))
    return int(audit['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
