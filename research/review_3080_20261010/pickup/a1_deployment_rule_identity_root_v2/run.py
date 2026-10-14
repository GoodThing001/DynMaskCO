"""Replay the registered deployment rule and emit an auditable per-order decision log.

Engineering only: the five replayed days were already consumed by the v4 panel, so this
stage can produce identity evidence but never new utility.
"""
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


def save(name, value):
    seal.save(HERE / name, value)


def make_contract(reg):
    from dataclasses import replace
    from coldchain_contract import ColdChainContractV2, UnitScale
    c = ColdChainContractV2()
    units = UnitScale(distance_km_per_unit=float(reg['frozen']['km_per_unit']),
                      hours_per_time_unit=1.0, speed_kmph=float(reg['frozen']['speed_kmh']))
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
    from run_identity import dataset_identity, dataset_meta_sha256
    from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import (build_history, build_history_times_classes, ExplicitFeatureSampler, SaaReplanner)
    from strict_online_env import StrictOnlineEnv
    from run_exp_reserve import SPEED_KMH, KM_PER_UNIT, BOOKING_HORIZON
    import deployment_rule as DR

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    for path, want in REG['sources'].items():
        import hashlib
        ck('registered_source:' + path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == want)

    contract = make_contract(REG)
    fz = REG['frozen']
    gate_ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']),
                                     contract)
    ck('gate_dataset_identity', dataset_identity(gate_ds) == fz['expected_data_sha256'])
    ck('gate_dataset_meta_identity', dataset_meta_sha256(gate_ds) == fz['expected_meta_sha256'])
    ck('contract_identity', contract.contract_hash == fz['expected_contract_sha256'])

    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], build_history_times_classes(train_ds),
                                                   k=fz['k_neighbors'])
    ck('budget_matches_registered_constant', abs(float(B) - float(REG['budget_B'])) < 1e-12)
    ck('cooling_share_matches_frozen', abs(float(cooling_share) - float(fz['expected_cooling_share'])) < 1e-12)

    lam_grid = DR.build_lambda_grid()
    clock_grid = DR.build_clock_grid(float(fz['booking_horizon']))
    g_table = DR.calibrate(train_ds, REG['rule_spec']['objective'], lam_grid, clock_grid)
    ck('lambda_grid_size', len(lam_grid) == 63)
    ck('clock_grid_size', len(clock_grid) == 33)
    ck('calibration_monotone_in_lambda',
       all(all(g_table[i][t] >= g_table[i + 1][t] - 1e-12 for t in range(len(clock_grid)))
           for i in range(len(lam_grid) - 1)))

    objective = REG['rule_spec']['objective']
    log = []

    class DeploymentRuleReplanner(SaaReplanner):
        def _rule_allows(self, env, inst_idx, clock, o):
            c = int(env.dataset['temp_class'][inst_idx, o])
            q = float(env.demands[inst_idx, o])
            e = DR.e_lower_of(c, q, objective)
            spent = 0.0
            for other in self._accepted:
                co = int(env.dataset['temp_class'][inst_idx, other])
                qo = float(env.demands[inst_idx, other])
                spent += DR.e_lower_of(co, qo, objective)
            b_rem = float(B) - spent
            admit, reason, lam, ti = DR.decide(e, DR.REV[c], clock, b_rem, lam_grid, clock_grid, g_table)
            entry = dict(inst=int(inst_idx), clock=float(clock), order=int(o), class_id=c, quantity=q,
                         e_lower=e, revenue=DR.REV[c], spent_before=spent, b_rem=b_rem,
                         bucket=(None if ti is None else int(ti)),
                         lam=(None if lam is None else float(lam)),
                         rule_decision=('admit' if admit else 'reject'), reason=reason)
            return admit, entry

        def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
            admit, entry = self._rule_allows(env, inst_idx, clock, o)
            if not admit:
                self._rejected.add(o)
                entry['base_decision'] = 'not_reached'
                log.append(entry)
                return
            super()._saa_decide(env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)
            entry['base_decision'] = 'accepted' if o in self._accepted else 'rejected'
            log.append(entry)

    p_c = {int(k): float(v) for k, v in REG['base_policy']['penalty_tiers'].items()}
    kw = dict(budget=B, capacity=fz['capacity'], booking_horizon=BOOKING_HORIZON, contract=contract,
              cooling_share=cooling_share, sampler=ExplicitFeatureSampler(build_history(train_ds)),
              reject_penalty=p_c, K=REG['base_policy']['K'], time_limit=REG['base_policy']['time_limit_s'],
              arm_seed=REG['base_policy']['arm_seed'], shadow_mode=REG['base_policy']['shadow_mode'],
              energy_pricing=REG['base_policy']['energy_pricing'],
              future_policy=REG['base_policy']['future_policy'], standby_orders=None,
              overage_price=None, incr_eval=False)
    days = REG['replay_days']['indices']
    panel = {k: v[:max(days) + 1] for k, v in gate_ds.items()}
    rp = DeploymentRuleReplanner(**kw)
    env = StrictOnlineEnv(panel, capacity=fz['capacity'], num_vehicles=fz['num_vehicles'],
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    rows = []
    for i in days:
        t0 = time.perf_counter()
        traces, _ = env.run(i)
        acc, rej = set(rp._accepted), set(rp._rejected)
        ev = evaluate_trace_a1(traces, {k: v[i] for k, v in panel.items()}, contract, acc, rej, B, p_c)
        rows.append(dict(day=int(i), utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
                         hard_feasible=bool(ev['hard_feasible']), failures=sorted(set(ev['failures'])),
                         n_served=int(ev['n_served']), revenue=float(ev['revenue']),
                         energy_kwh=float(ev['energy_kwh']), budget_violated=bool(ev['budget_violated']),
                         elapsed_s=time.perf_counter() - t0))
    elapsed = time.perf_counter() - started

    v4 = json.loads((ROOT / REG['v4_artifacts']['result']).read_text(encoding='utf-8'))
    v4rows = {r['i']: r for r in v4['arms'][REG['v4_artifacts']['arm']]}
    for r in rows:
        ck('day_hard_feasible:%d' % r['day'], r['hard_feasible'] is True)
        ck('day_budget_respected:%d' % r['day'], not r['budget_violated'])
        ref = v4rows[r['day']]
        ck('day_utility_matches_v4:%d' % r['day'], r['utility'] == ref['utility'])
        ck('day_served_matches_v4:%d' % r['day'], r['n_served'] == ref['n_served'])
        ck('day_energy_matches_v4:%d' % r['day'], r['energy_kwh'] == ref['energy_kwh'])

    with (HERE / 'DECISION_LOG.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for e in log:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + '\n')
    n_admit = sum(1 for e in log if e['rule_decision'] == 'admit')
    reasons = {}
    for e in log:
        reasons[e['reason']] = reasons.get(e['reason'], 0) + 1
    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))
    ck('decision_log_non_empty', len(log) > 0)

    # ---- source drift finding (recorded, not a gate) ---------------------------
    v4reg_path = ROOT / Path(REG['v4_artifacts']['result']).parent / 'REGISTRATION.json'
    v4reg = json.loads(v4reg_path.read_text(encoding='utf-8'))
    import hashlib as _h
    drift = {}
    for path, old in v4reg['sources'].items():
        p = Path(path)
        cur = _h.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        if cur != old:
            drift[path] = dict(v4_registered=old, current=cur)
    run_arm_broken = None
    try:
        import importlib
        importlib.import_module('run_a1_step2_gate')
        run_arm_broken = False
    except Exception as exc:
        run_arm_broken = repr(exc)
    ck('drift_finding_recorded', True)

    result = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                  failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                  stage=REG['stage'], rule=REG['chosen_rule'], budget_B=float(B),
                  lambda_grid=lam_grid, clock_grid=clock_grid,
                  g_table_sha256=DR_hash(g_table), g_table=g_table,
                  replay_days=days, decisions=len(log), admits=n_admit, reason_counts=reasons,
                  per_day=rows, v4_arm=REG['v4_artifacts']['arm'],
                  decision_log='DECISION_LOG.jsonl', elapsed_seconds=elapsed,
                  source_drift_since_v4_registration=drift,
                  v4_pipeline_importable=(run_arm_broken is False),
                  v4_pipeline_import_error=(None if run_arm_broken is False else run_arm_broken),
                  source_actual_files=len(start_closure['files']),
                  source_actual_maps=len(start_closure['modules']),
                  new_utility_day=False, scientific_family_started=False, stable_online_gain=False)
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'g_table', 'lambda_grid')}))
    return int(result['failed'] > 0)


def DR_hash(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    raise SystemExit(main())
