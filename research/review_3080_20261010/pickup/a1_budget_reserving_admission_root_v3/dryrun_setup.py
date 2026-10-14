"""Fast dry-run of the exact v3 entry-point setup path (no day simulation).

Validates the frozen identities, budget recomputation, calibration table, the
contract/environment/replanner construction path used by run_arm, and the
reservation-rule arithmetic.  Writes nothing into the registered stage outputs.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
sys.path.insert(0, str(ROOT / 'tools'))
for sub in ('scripts', 'scripts/evaluation', 'scripts/coldchain', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / sub))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))


def main():
    import numpy as np
    import run_exp_reserve as RR
    from run_identity import dataset_identity
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import build_history, build_history_times_classes, ExplicitFeatureSampler, SaaReplanner
    from run_a1_step2_gate import run_arm
    from reserving_admission import build_lambda_grid, build_clock_grid, calibrate

    sys.path.insert(0, str(HERE))
    from run import make_contract, make_reserving_class

    contract = make_contract()
    print('contract ok', contract.contract_hash[:16],
          contract.contract_hash == REG['frozen']['expected_contract_sha256'])
    fz = REG['frozen']
    gate_ds = add_v2_initial_quality(RR.generate_dataset(fz['gate_instances'], fz['n_orders'], fz['gate_seed']),
                                     contract)
    print('dataset ok', dataset_identity(gate_ds) == fz['expected_data_sha256'])

    train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
    hist = build_history(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(train_ds, contract, fz['num_vehicles'], fz['train_seed'],
                                                   fz['rho'], build_history_times_classes(train_ds),
                                                   k=fz['k_neighbors'])
    print('budget ok', abs(float(B) - fz['expected_budget_B']) < 1e-9,
          'cooling_share ok', abs(float(cooling_share) - fz['expected_cooling_share']) < 1e-12)

    lam_grid = build_lambda_grid()
    clock_grid = build_clock_grid(fz['booking_horizon'])
    g_table = calibrate(train_ds, REG['objective'], fz['booking_horizon'], lam_grid, clock_grid)
    print('calibration ok', g_table[0][0] > B > g_table[-1][0])

    Reserving = make_reserving_class()
    panel0 = {k: v[:1] for k, v in gate_ds.items()}
    sampler = ExplicitFeatureSampler(hist)
    p_c = {int(k): float(v) for k, v in REG['penalties'].items()}

    rows = run_arm(panel0, contract, B, fz['capacity'], fz['num_vehicles'], cooling_share, sampler, p_c,
                   fz['K'], fz['time_limit'], REG['arm_seed'], shadow_mode=REG['shadow_mode'],
                   energy_pricing=REG['energy_pricing'], future_policy=REG['future_policy'],
                   replanner_cls=Reserving,
                   extra_kwargs=dict(lambda_grid=lam_grid, clock_grid=clock_grid, g_table=g_table,
                                     objective=REG['objective']))
    print('replanner construction and one-day run ok, rows', len(rows))
    if rows:
        r = rows[0]
        print(' day0 utility', r['utility'], 'served', r['n_served'], 'energy', r['energy_kwh'],
              'hard', r['hard_feasible'], 'failures', r['failures'])
        print(' elapsed_s', round(r['elapsed_s'], 1))
        print(' decision counters', r.get('n_decisions'))
    print('DRYRUN OK')


if __name__ == '__main__':
    raise SystemExit(main())
