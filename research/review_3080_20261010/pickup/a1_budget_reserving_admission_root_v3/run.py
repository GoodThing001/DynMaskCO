"""Pickup: pre-registered budget-reserving admission panel (5 frozen days, controlled A/B).

Both arms use the identical sampler, environment, days, K, time limit, contract and
arm seed; only the replanner class differs.  The control is the unmodified frozen
SaaReplanner, so the paired difference isolates the reservation pre-filter.
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


def make_reserving_class():
    from scenario_saa import SaaReplanner
    from reserving_admission import BudgetReservingAdmission

    class ReservingSaaReplanner(SaaReplanner, BudgetReservingAdmission):
        def __init__(self, *args, lambda_grid=None, clock_grid=None, g_table=None, objective=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._reserving_init(lambda_grid, clock_grid, g_table, objective)

        def _saa_decide(self, env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0):
            token = {}
            if not self._reserving_allows(env, inst_idx, clock, o, token):
                self._rejected.add(o)
                if len(self.reserve_log) < 400:
                    self.reserve_log.append(dict(inst=int(inst_idx), clock=float(clock), order=int(o),
                                                 reason=token.get('reason'), lam=token.get('lambda'),
                                                 est=token.get('est'), revenue=token.get('revenue')))
                return
            super()._saa_decide(env, inst_idx, clock, vehicles, served_mask, o, scenarios, t0)

    return ReservingSaaReplanner


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
    from run_identity import dataset_identity, dataset_meta_sha256
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from scenario_saa import (build_history, build_history_times_classes, ExplicitFeatureSampler,
                              SaaReplanner)
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from run_a1_step2_gate import run_arm
    from reserving_admission import (build_lambda_grid, build_clock_grid, calibrate)

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    contract = make_contract()
    fz = REG['frozen']
    gate_ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']),
                                     contract)
    ck('frozen_dataset_identity', dataset_identity(gate_ds) == fz['expected_data_sha256'])
    ck('frozen_dataset_meta_identity', dataset_meta_sha256(gate_ds) == fz['expected_meta_sha256'])
    ck('frozen_contract_identity', contract.contract_hash == fz['expected_contract_sha256'])

    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], hist_tc, k=fz['k_neighbors'])
    ck('budget_recomputed_matches_recorded', abs(float(B) - fz['expected_budget_B']) < 1e-9)
    ck('cooling_share_recomputed_matches_recorded',
       abs(float(cooling_share) - fz['expected_cooling_share']) < 1e-12)

    lam_grid = build_lambda_grid()
    clock_grid = build_clock_grid(fz['booking_horizon'])
    g_table = calibrate(train_ds, REG['objective'], fz['booking_horizon'], lam_grid, clock_grid)
    ck('calibration_is_monotone_in_lambda',
       all(all(g_table[li][ti] >= g_table[li + 1][ti] - 1e-9 for ti in range(len(clock_grid)))
           for li in range(len(lam_grid) - 1)))
    ck('calibration_spans_budget',
       g_table[0][0] > B > g_table[-1][0])

    panel = {k: v[:REG['panel']['days']] for k, v in gate_ds.items()}
    p_c = {int(k): float(v) for k, v in REG['penalties'].items()}
    sampler_ctl = ExplicitFeatureSampler(hist)
    sampler_cand = ExplicitFeatureSampler(hist)

    arms = {}
    Reserving = make_reserving_class()
    t_cand = time.perf_counter()
    arms['candidate_reserving'] = run_arm(panel, contract, B, fz['capacity'], fz['num_vehicles'], cooling_share,
                                          sampler_cand, p_c, fz['K'], fz['time_limit'],
                                          REG['arm_seed'], shadow_mode=REG['shadow_mode'],
                                          energy_pricing=REG['energy_pricing'],
                                          future_policy=REG['future_policy'],
                                          replanner_cls=Reserving,
                                          extra_kwargs=dict(lambda_grid=lam_grid, clock_grid=clock_grid,
                                                            g_table=g_table, objective=REG['objective']),
                                          progress_file=str(HERE / 'progress_candidate.txt'))
    t_cand = time.perf_counter() - t_cand


    t_ctl = time.perf_counter()
    arms['control_frozen_saa'] = run_arm(panel, contract, B, fz['capacity'], fz['num_vehicles'], cooling_share,
                                         sampler_ctl, p_c, fz['K'], fz['time_limit'],
                                         REG['arm_seed'], shadow_mode=REG['shadow_mode'],
                                         energy_pricing=REG['energy_pricing'],
                                         future_policy=REG['future_policy'],
                                         replanner_cls=SaaReplanner,
                                         progress_file=str(HERE / 'progress_control.txt'))
    t_ctl = time.perf_counter() - t_ctl

    control = json.loads((ROOT / fz['recorded_controls']).read_text(encoding='utf-8'))['gate']['p_c=0']['per_day']

    paired = []
    for k in range(REG['panel']['days']):
        a, b = arms['control_frozen_saa'][k], arms['candidate_reserving'][k]
        ck('day_hard_feasible:control:%d' % k, a['hard_feasible'] is True)
        ck('day_hard_feasible:candidate:%d' % k, b['hard_feasible'] is True)
        ck('day_budget_respected:control:%d' % k, not a['budget_violated'])
        ck('day_budget_respected:candidate:%d' % k, not b['budget_violated'])
        paired.append(dict(day=k,
                           control_utility=a['utility'], candidate_utility=b['utility'],
                           delta=(None if (a['utility'] is None or b['utility'] is None)
                                  else float(b['utility'] - a['utility'])),
                           control=dict(served=a['n_served'], revenue=a['revenue'], energy=a['energy_kwh'],
                                        service_rate=a['service_rate'], distance_km=a['distance_km']),
                           candidate=dict(served=b['n_served'], revenue=b['revenue'], energy=b['energy_kwh'],
                                          service_rate=b['service_rate'], distance_km=b['distance_km']),
                           recorded_rows={arm: dict(utility=r[k]['utility'], served=r[k]['n_served'],
                                                    revenue=r[k]['revenue'], energy=r[k]['energy_kwh'])
                                          for arm, r in control.items()}))
    deltas = [p['delta'] for p in paired if p['delta'] is not None]
    mean_delta = float(np.mean(deltas)) if deltas else None
    ck('all_panel_days_produced_utilities', len(deltas) == REG['panel']['days'])
    lead_positive = bool(mean_delta is not None and mean_delta > 0.0)

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    result = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                  failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                  stage='pickup_a1_budget_reserving_admission',
                  panel_days=REG['panel']['days'], arm_seed=REG['arm_seed'],
                  budget_B=float(B), cooling_share=float(cooling_share),
                  dataset_sha256=dataset_identity(gate_ds), contract_sha256=contract.contract_hash,
                  control_seconds=t_ctl, candidate_seconds=t_cand,
                  lambda_grid=lam_grid, clock_grid=clock_grid, g_table=g_table,
                  arms={k: v for k, v in arms.items()},
                  paired=paired, mean_delta=mean_delta, lead_positive=lead_positive,
                  decision=('lead_positive_extend_to_40_days_in_a_new_registered_stage' if lead_positive
                            else 'candidate_dominated_record_and_pivot_to_joint_coupling_relaxation'),
                  source_actual_files=len(start_closure['files']), source_actual_maps=len(start_closure['modules']),
                  elapsed_seconds=time.perf_counter() - started,
                  formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                  scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                  stable_online_gain=False)
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'arms')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
