"""Pre-execution registration of the A-v1 deployment rule and its identity audit."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
V4 = ROOT / 'research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4'
PREREG = ROOT / 'research/review_3080_20261010/pickup/A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    ev = EXT / 'scripts/evaluation'
    sources = [HERE / n for n in ('deployment_rule.py', 'run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources += [ROOT / 'tools' / 'engineering_stage_closure_v1.py',
                V4 / 'reserving_admission.py', V4 / 'run.py', V4 / 'REGISTRATION.json', V4 / 'RESULT.json',
                PREREG,
                ev / 'run_exp_reserve.py', ev / 'run_identity.py', ev / 'run_exp_encoder_v3.py',
                ev / 'run_exp_energy_c0.py', ev / 'coldchain_evaluator_a1.py', ev / 'scenario_saa.py',
                ev / 'run_a1_step2_gate.py',
                EXT / 'scripts/coldchain/coldchain_contract.py',
                EXT / 'scripts/coldchain/coldchain_state.py',
                EXT / 'scripts/simulation/strict_online_env.py']
    for p in sources:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        if not p.exists():
            raise SystemExit('missing source %s' % p)
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='pickup_a1_deployment_rule_identity',
        purpose=('Close the registration-versus-action identity gap found by round 21: the first pre-registration '
                 'froze a reservation condition that is NOT equivalent to the condition the five-day panel actually '
                 'executed. This stage fixes ONE deployment rule in writing and then proves, by independent per-order '
                 're-derivation from the frozen execution log, that the executed decisions are exactly the registered '
                 'formula. No new utility day, no new seed, no training.'),
        chosen_rule='clock_shadow_price_budget_reserving_admission',
        rule_spec=dict(
            per_order_cost=('e_lower(i) = quantity_i * precool_energy_per_unit_per_c * '
                            'max(0, ambient_c - target_c[class_i])'),
            objective=dict(ambient_c=25.0, target_c=[18.0, 4.0, -18.0], precool_energy_per_unit_per_c=0.15,
                           revenue_by_class={'0': 10.0, '1': 20.0, '2': 30.0}),
            spent=('spent_before = sum of e_lower over the orders currently in the base replanner accepted set; '
                   'base-SAA acceptances only, so an order the rule admitted but the base rejected does not count'),
            budget_remaining='b_rem = B - spent_before',
            hard_precheck="if e_lower(i) > b_rem: reject with reason 'over_remaining_budget'",
            clock_bucket=('ti = min(max(bisect_right(clock_grid, clock) - 1, 0), len(clock_grid) - 1); '
                          'clock_grid = 0.0 .. 16.0 step 0.5 (33 values)'),
            price_selection=('lambda = the smallest value in the ascending lambda grid with G[lambda][ti] <= b_rem; '
                             'if no grid value satisfies it, lambda = max(grid); '
                             'lambda grid = 0.25 .. 8.0 step 0.125 (63 values, rounded to 6 dp)'),
            admit_rule='admit iff REV[class_i] - lambda * e_lower(i) > 0 (strictly greater)',
            reject_reasons=['over_remaining_budget', 'below_shadow_price'],
            post_admit='the frozen base SAA still decides; the rule can only add rejections, never admit what the base refuses',
            calibration=('G[lambda][ti] = mean over the 200 frozen TRAIN days (seed 20260925) of the sum of e_lower '
                         'over orders with reveal_time >= clock_grid[ti] and REV - lambda*e_lower > 0; '
                         'TRAIN only, no gate or fresh day is touched by calibration'),
        ),
        budget_B=710.6548328611384,
        budget_B_note=('the recomputed budget from the current source tree; the recorded batch budget 710.6548152249378 '
                       'is NOT reproducible because scenario_saa.py changed after that batch (delta 1.76e-05)'),
        frozen=dict(
            train_seed=20260925, train_instances=200, rho=0.6, k_neighbors=10,
            gate_seed=20260926, gate_instances=40, n_orders=200, num_vehicles=15, capacity=50.0,
            booking_horizon=16.0, speed_kmh=30.0, km_per_unit=20.0, service_time_h=0.05,
            expected_cooling_share=3.7208946347769882, expected_budget_B=710.6548152249378,
            expected_data_sha256='8767a03d787c0e0fcad112b4888edf46db44c55e9b50b9705bfc5830f4fb378d',
            expected_meta_sha256='72a805fd5bc0b2b7a94b0daa712cb11112bcc3e4a5daf7817ca8ca53e28322ba',
            expected_contract_sha256='e25a12104b20c3ea99cb4033f6d3bf32adecf2974de6b5168dbfa0d5a058a41b'),
        v4_artifacts=dict(
            result='research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/RESULT.json',
            registration='research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/REGISTRATION.json',
            arm='candidate_reserving',
            role='the execution this replay must reproduce per day, so that the registered rule is the executed rule'),
        base_policy=dict(replanner='frozen SaaReplanner', sampler='ExplicitFeatureSampler', arm_seed=7003,
                         K=10, time_limit_s=10.0, shadow_mode='greedy', energy_pricing='marginal',
                         future_policy='reveal', penalty_tier='p_c=0',
                         penalty_tiers={'0': 0.0, '1': 0.0, '2': 0.0}),
        replay_days=dict(seed=20260926, indices=[0, 1, 2, 3, 4],
                         note=('these five days were already consumed by the v4 panel, so the replay can only create '
                               'engineering evidence, never new utility; the registered plan for fresh days is a later, '
                               'separately registered stage')),
        primary_metrics=['per-order decision agreement between the registered formula and the execution log',
                         'per-day utility identity against the v4 recorded candidate rows',
                         'counts of decisions, admissions and each reject reason'],
        failure_criteria=['any per-order decision or reason mismatch',
                          'any per-day utility mismatch against v4',
                          'any budget violation or hard-infeasible day',
                          'any registered source or data hash mismatch',
                          'lambda-table digest mismatch against the independent calibration'],
        stop_rule='on any failure criterion the stage stops at the engineering phase and no utility comparison is proposed',
        scientific_family_started=False, new_utility_day=False, final_seeds_used=False,
        stable_online_gain=False, delivery_science_family3_started=False,
        preserved_prior_attempt=dict(
            directory='research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v1',
            outcome='producer raised TypeError on the replanner constructor and exited 1 before any day ran',
            detail=('the runner passed anytime_vote, which the current scenario_saa.SaaReplanner no longer accepts; '
                    'scenario_saa.py was modified after the v4 registration and the recorded pipeline run_arm '
                    'can no longer be imported because SoftKNNHistoricalSampler was removed'),
            artifacts_written=['SOURCE_START.json', 'run.stdout.txt', 'run.stderr.txt',
                               'RUN_SUPERVISION_RECEIPT.json']),
        sources={str(p): sha(p) for p in sources},
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
