"""Run the deployment-rule qualification through a current-source online pipeline."""
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


def save(name, value):
    seal.save(HERE / name, value)


def main():
    seal.initialize()
    import run_exp_reserve as RR
    from run_identity import dataset_identity
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import build_history_times_classes
    import pipeline

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    import hashlib
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
    gate_ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']),
                                     contract)
    ck('gate_dataset_identity', dataset_identity(gate_ds) == fz['expected_data_sha256'])
    ck('contract_identity', contract.contract_hash == fz['expected_contract_sha256'])
    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], build_history_times_classes(train_ds),
                                                   k=fz['k_neighbors'])
    ck('budget_matches_registered', abs(float(B) - float(REG['budget_B'])) < 1e-12)

    # rule artifacts come from the v3 stage, read-only
    v3 = ROOT / REG['rule_stage']['directory']
    rule = json.loads((v3 / 'RESULT.json').read_text(encoding='utf-8'))
    ck('rule_stage_registration_hash',
       hashlib.sha256((v3 / 'REGISTRATION.json').read_bytes()).hexdigest() == REG['rule_stage']['registration_sha256'])
    ck('rule_stage_result_hash',
       hashlib.sha256((v3 / 'RESULT.json').read_bytes()).hexdigest() == REG['rule_stage']['result_sha256'])

    days = REG['days']
    per_arm = {}
    arm_logs = {}
    for arm in ('rule', 'control'):
        t0 = time.perf_counter()
        rows, log, decisions = pipeline.run_arm(arm, gate_ds, days, contract, B, cooling_share, train_ds,
                                                REG['online'], rule)
        per_arm[arm] = dict(rows=rows, decisions=decisions, seconds=time.perf_counter() - t0)
        arm_logs[arm] = log
        for r in rows:
            ck('%s_day_hard_feasible:%d' % (arm, r['day']), r['hard_feasible'] is True)
            ck('%s_day_budget_respected:%d' % (arm, r['day']), not r['budget_violated'])
            ck('%s_day_within_time_limit:%d' % (arm, r['day']), not r['over_time_limit'])

    v4 = json.loads((ROOT / REG['v4_artifacts']['result']).read_text(encoding='utf-8'))
    v4rows = {a: {r['i']: r for r in v4['arms'][a]} for a in ('candidate_reserving', 'control_frozen_saa')}
    for r in per_arm['rule']['rows']:
        ck('rule_row_matches_v4_candidate:%d' % r['day'],
           r['utility'] == v4rows['candidate_reserving'][r['day']]['utility'])
    for r in per_arm['control']['rows']:
        ck('control_row_matches_v4_control:%d' % r['day'],
           r['utility'] == v4rows['control_frozen_saa'][r['day']]['utility'])

    rule_u = [r['utility'] for r in per_arm['rule']['rows']]
    ctl_u = [r['utility'] for r in per_arm['control']['rows']]
    deltas = [a - b for a, b in zip(rule_u, ctl_u)]
    ck('paired_delta_matches_v4', all(
        abs(d - (v4rows['candidate_reserving'][d0]['utility'] - v4rows['control_frozen_saa'][d0]['utility'])) < 1e-9
        for d, d0 in zip(deltas, days)))
    ck('control_is_executable_in_current_pipeline', all(r['utility'] is not None for r in per_arm['control']['rows']))

    with (HERE / 'ONLINE_ARM_LOG.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for arm in ('rule', 'control'):
            for e in arm_logs[arm]:
                f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + '\n')

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    result = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                  failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                  stage=REG['stage'], budget_B=float(B), days=days,
                  rule_utilities=rule_u, control_utilities=ctl_u, paired_deltas=deltas,
                  mean_paired_delta=float(np.mean(deltas)),
                  rule_decisions=per_arm['rule']['decisions'], control_decisions=per_arm['control']['decisions'],
                  control_decision_logging=('not instrumented: the control arm runs the unmodified SaaReplanner, so only '
                                            'its per-day rows (utility, served, energy, timeouts) are captured and '
                                            'compared; per-order logging is provided for the rule arm only'),
                  arm_seconds={a: per_arm[a]['seconds'] for a in per_arm},
                  request_seconds={a: [r['request_seconds'] for r in per_arm[a]['rows']] for a in per_arm},
                  timeouts={a: [r['timeouts'] for r in per_arm[a]['rows']] for a in per_arm},
                  online_log='ONLINE_ARM_LOG.jsonl',
                  source_actual_files=len(start_closure['files']),
                  source_actual_maps=len(start_closure['modules']),
                  elapsed_seconds=time.perf_counter() - started,
                  fresh_utility_day=False, new_utility_day=False, scientific_family_started=False,
                  stable_online_gain=False)
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('checks', 'request_seconds', 'timeouts', 'online_log')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
