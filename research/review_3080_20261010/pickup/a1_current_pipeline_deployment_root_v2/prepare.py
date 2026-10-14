"""Pre-execution registration: deployment qualification of the registered rule in a current-source pipeline."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
V3 = ROOT / 'research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v3'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    ev = EXT / 'scripts/evaluation'
    sources = [HERE / n for n in ('pipeline.py', 'run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources += [ROOT / 'tools' / 'engineering_stage_closure_v1.py',
                V3 / 'deployment_rule.py', V3 / 'REGISTRATION.json', V3 / 'RESULT.json',
                V3 / 'INDEPENDENT_AUDIT.json', V3 / 'DECISION_LOG.jsonl',
                ev / 'run_exp_reserve.py', ev / 'run_identity.py', ev / 'run_exp_encoder_v3.py',
                ev / 'coldchain_evaluator_a1.py', ev / 'scenario_saa.py',
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
        stage='pickup_current_pipeline_deployment_qualification',
        blocking_item=('round 21 showed the registered rule identity was only proven on a sealed log; the recorded online '
                       'pipeline run_a1_step2_gate can no longer be imported because scenario_saa.py dropped '
                       'SoftKNNHistoricalSampler and the anytime_vote path, so the rule had not been shown deployable '
                       'in a current-source online pipeline with a contemporaneous control'),
        purpose=('Qualify the registered deployment rule in a frozen current-source online pipeline together with a '
                 'contemporaneous control arm under identical sampler, seed, K, time limit and contract, on the five '
                 'already-consumed days. This is engineering: it produces no fresh utility day and starts no science.'),
        arms=dict(
            rule='registered deployment rule in front of the unmodified SaaReplanner',
            control='the unmodified SaaReplanner alone (identical sampler/seed/K/time limit)'),
        pipeline_semantics=dict(
            entry='pipeline.run_arm in this directory (current sources only; the old pipeline is never edited)',
            online_loop='StrictOnlineEnv.run per day, per-reveal decision, K=10 SAA scenarios, per-request time limit',
            timing='per-day request seconds are recorded and compared with the registered 10 s time limit',
            control_logging='the control arm is not instrumented per order; its behaviour is compared at day-row level'),
        identity_chain=('rule artifacts are read read-only from the v3 stage and both of its hashes are verified before '
                        'use, so this stage cannot silently fork the rule'),
        gates=dict(
            rule_reproduces_v4_candidate='per-day utility equality against the v4 candidate rows',
            control_reproduces_v4_control=('per-day utility equality against the v4 control rows; if the scenario_saa '
                                           'drift changed control behaviour this must fail and the five-day comparison '
                                           'is invalidated under current sources'),
            paired_delta_matches_v4='the paired difference must equal the v4 paired difference',
            hard_feasible_and_within_budget='every arm-day must be hard feasible and inside the budget',
            within_time_limit='every arm-day request must be inside the registered time limit'),
        days=[0, 1, 2, 3, 4],
        v4_day_elapsed=dict(rule={'0': 101.8, '1': 147.5, '2': 91.0, '3': 126.3, '4': 109.7},
                            control={'0': 112.3, '1': 114.8, '2': 98.6, '3': 134.5, '4': 141.9}),
        corrected_checks=('v1 recorded 11 failures from two mis-scoped checks of this harness: (a) the per-day wall time '
                          'was compared with the registered 10 s, which is a PER-DECISION deadline inside _saa_decide, '
                          'not a per-day budget, so every day failed by construction; (b) the loaded-closure start was '
                          'taken before pipeline.run_arm lazily imported the rule module, so the module map gained '
                          'deployment_rule. v2 warms the rule module before the closure and checks per-decision timeouts '
                          'plus day wall-time stability against the recorded v4 day instead. The substantive measurements '
                          'of v1 are preserved and unchanged.'),
        frozen=dict(
            train_seed=20260925, train_instances=200, rho=0.6, k_neighbors=10,
            gate_seed=20260926, gate_instances=40, n_orders=200, num_vehicles=15, capacity=50.0,
            booking_horizon=16.0, speed_kmh=30.0, km_per_unit=20.0, service_time_h=0.05,
            expected_cooling_share=3.7208946347769882, expected_budget_B=710.6548152249378,
            expected_data_sha256='8767a03d787c0e0fcad112b4888edf46db44c55e9b50b9705bfc5830f4fb378d',
            expected_meta_sha256='72a805fd5bc0b2b7a94b0daa712cb11112bcc3e4a5daf7817ca8ca53e28322ba',
            expected_contract_sha256='e25a12104b20c3ea99cb4033f6d3bf32adecf2974de6b5168dbfa0d5a058a41b'),
        budget_B=710.6548328611384,
        online=dict(objective=dict(ambient_c=25.0, target_c=[18.0, 4.0, -18.0],
                                   precool_energy_per_unit_per_c=0.15,
                                   revenue_by_class={'0': 10.0, '1': 20.0, '2': 30.0}),
                    penalty_tiers={'0': 0.0, '1': 0.0, '2': 0.0},
                    capacity=50.0, num_vehicles=15, K=10, time_limit_s=10.0, arm_seed=7003,
                    shadow_mode='greedy', energy_pricing='marginal', future_policy='reveal'),
        v4_artifacts=dict(result='research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4/RESULT.json'),
        rule_stage=dict(directory='research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v3',
                        registration_sha256=sha(V3 / 'REGISTRATION.json'), result_sha256=sha(V3 / 'RESULT.json')),
        failure_criteria=['rule or control row differs from the v4 rows',
                          'paired difference differs from the v4 paired difference',
                          'any arm-day not hard feasible, over budget, or over the time limit',
                          'any registered source or identity hash mismatch'],
        stop_rule=('on any failure the stage stops at engineering and reports that the five-day comparison is not '
                   'deployable under current sources'),
        preserved_prior_attempt=dict(
            directory='research/review_3080_20261010/pickup/a1_current_pipeline_deployment_root_v1',
            outcome=('producer 67 checks with 11 failures; the substantive rows still reproduced both v4 arms exactly'),
            detail='two mis-scoped harness checks as described in corrected_checks',
            measurements_produced=True, reused=False),
        preserves=['the v3 result and the recorded source-drift evidence, untouched',
                   'all old FAIL artifacts; nothing is re-run to chase a PASS'],
        fresh_utility_day=False, new_utility_day=False, scientific_family_started=False,
        delivery_science_family3_started=False, final_seeds_used=False, stable_online_gain=False,
        sources={str(p): sha(p) for p in sources},
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
