"""Prospective registration of the pickup budget-reserving admission panel."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
PREREG = ROOT / 'research/review_3080_20261010/pickup/A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    ev = EXT / 'scripts/evaluation'
    sources = [HERE / n for n in ('reserving_admission.py', 'run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources += [ROOT / 'tools' / 'engineering_stage_closure_v1.py', PREREG,
                ev / 'run_exp_reserve.py', ev / 'run_identity.py', ev / 'run_exp_encoder_v3.py',
                ev / 'run_exp_energy_c0.py', ev / 'coldchain_evaluator_a1.py',
                ev / 'scenario_saa.py', ev / 'run_a1_step2_gate.py',
                EXT / 'scripts/coldchain/coldchain_contract.py',
                EXT / 'scripts/coldchain/coldchain_state.py',
                EXT / 'scripts/simulation/strict_online_env.py']
    for p in sources:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        if not p.exists():
            raise SystemExit('missing registered source %s' % p)
    controls = EXT / 'results/a1_c1rr2_reveal_40/gate.json'
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p): sha(p) for p in sources},
        stage='pickup_a1_budget_reserving_admission',
        preregistration=str(PREREG.relative_to(ROOT)),
        preregistration_sha256=sha(PREREG),
        goal=('Test the pre-registered clock-dependent shadow-price admission rule against the unmodified frozen SAA on '
              'the same five frozen days with the same sampler, seed, K, time limit and contract, so the paired '
              'difference isolates the reservation filter.'),
        mechanism=('with p_c=0 the base SAA has no reason to refuse an order, so the served set is decided by arrival '
                   'order against the hard energy budget; the rule admits an order only when revenue exceeds '
                   'lambda(clock)*est, with lambda the smallest grid price whose expected accepted energy over the '
                   'remaining horizon still fits the remaining budget, calibrated on the frozen TRAIN split only'),
        controlled_ab=('both arms use ExplicitFeatureSampler, arm_seed 7003, K=10, time_limit 10 s, the same 5 days and '
                       'the same contract; only replanner_cls differs (frozen SaaReplanner versus the reserving subclass). '
                       'The rule can only add rejections, never admit what the base would refuse.'),
        panel=dict(days=5, fixed_before_execution=True, day_indices=[0, 1, 2, 3, 4],
                   no_day_selection_by_outcome=True),
        decision_rule=('primary = paired per-day difference candidate - control on utility over the 5 panel days; a '
                       'positive mean is a lead only and requires a separately registered 40-day stage; otherwise the '
                       'rule is recorded as dominated and the next step is the joint time-window/fleet coupling '
                       'relaxation named in the bound addendum'),
        arm_seed=7003, penalties={'0': 0.0, '1': 0.0, '2': 0.0},
        shadow_mode='greedy', energy_pricing='marginal', future_policy='reveal',
        frozen=dict(
            train_seed=20260925, train_instances=200, gate_seed=20260926, gate_instances=40, n_orders=200,
            num_vehicles=15, capacity=50.0, booking_horizon=16.0, speed_kmh=30.0, km_per_unit=20.0,
            service_time_h=0.05, K=10, time_limit=10.0, rho=0.6, k_neighbors=10,
            expected_budget_B=710.6548152249378, expected_cooling_share=3.7208946347769882,
            expected_data_sha256='8767a03d787c0e0fcad112b4888edf46db44c55e9b50b9705bfc5830f4fb378d',
            expected_meta_sha256='72a805fd5bc0b2b7a94b0daa712cb11112bcc3e4a5daf7817ca8ca53e28322ba',
            expected_contract_sha256='e25a12104b20c3ea99cb4033f6d3bf32adecf2974de6b5168dbfa0d5a058a41b',
            recorded_controls='C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1rr2_reveal_40/gate.json',
            recorded_controls_sha256=sha(controls)),
        known_confound=('the recorded control rows were produced by a batch whose scenario_saa.py and '
                        'run_a1_step2_gate.py differ from the current files, so the recorded rows are reported for '
                        'reference only; the binding comparison is the contemporaneous controlled A/B run in this stage'),
        supervision_cap_seconds=7200,
        resource=dict(python='3.12.7', numpy='1.26.4', cpu_affinity=[7], no_model_min_available_ram_gib=0.5),
        formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        stable_online_gain=False, final_seeds_used=False,
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
