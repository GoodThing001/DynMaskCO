"""Prospective registration of the pickup A-v1 headroom upper-bound stage."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    eval_dir = EXT / 'scripts/evaluation'
    sources = [HERE / n for n in ('run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources += [ROOT / 'tools' / 'engineering_stage_closure_v1.py',
                eval_dir / 'run_exp_reserve.py', eval_dir / 'run_identity.py',
                eval_dir / 'coldchain_evaluator_a1.py',
                EXT / 'scripts/coldchain/coldchain_contract.py',
                EXT / 'scripts/coldchain/coldchain_state.py']
    for p in sources:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        if not p.exists():
            raise SystemExit('missing registered source %s' % p)
    controls = EXT / 'results/a1_c1rr2_reveal_40/gate.json'
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p): sha(p) for p in sources},
        stage='pickup_a1_budget_reachability_bound',
        goal=('Bound the A-v1 daily utility from above with one-sided relaxations and pair the bound with the recorded '
              'control arms, to decide how much headroom exists above the strongest admissible non-learning control.'),
        why=('the only reference used so far ("clairvoyant") is a greedy value/energy heuristic that the strongest rule '
             'beats (uncond_dynamic 1161.7 > clairvoyant 1151.6 in the earlier panel), so it is not an upper bound and '
             'the headroom question is still open'),
        scope_boundary=('Analysis only: no arm is run, no policy is trained, no model is loaded, no new data is '
                        'generated. Every relaxation is one-sided so the number produced is an upper bound, and the '
                        'recorded control arms are used as a falsification oracle for each relaxation.'),
        frozen=dict(
            gate_seed=20260926, gate_instances=40, n_orders=200, num_vehicles=15,
            booking_horizon=16.0, speed_kmh=30.0, km_per_unit=20.0, service_time_h=0.05,
            budget_B=710.6548152249378,
            expected_data_sha256='8767a03d787c0e0fcad112b4888edf46db44c55e9b50b9705bfc5830f4fb378d',
            expected_meta_sha256='72a805fd5bc0b2b7a94b0daa712cb11112bcc3e4a5daf7817ca8ca53e28322ba',
            expected_contract_sha256='e25a12104b20c3ea99cb4033f6d3bf32adecf2974de6b5168dbfa0d5a058a41b',
            recorded_controls='C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1rr2_reveal_40/gate.json',
            recorded_controls_sha256=sha(controls)),
        objective=dict(
            tier='p_c=0', revenue_by_class={'0': 10.0, '1': 20.0, '2': 30.0},
            ambient_c=25.0, target_c=[18.0, 4.0, -18.0],
            precool_energy_per_unit_per_c=0.15, door_heat_kwh=[0.010, 0.035, 0.060],
            depot_coord=[0.5, 0.5],
            utility='revenue - fuel - reject_loss, with fuel = 0.1 * distance_km >= 0 and reject_loss = 0 at p_c=0',
            bound_chain='utility <= revenue; revenue = sum of per-class constants; so a revenue upper bound bounds utility'),
        relaxations=dict(
            R1_servability=('earliest possible service start is max(straight-line travel time, reveal_time); straight-line '
                            'distance is a lower bound on any real travel distance, so an order whose earliest service '
                            'cannot finish inside its time window is unservable by any policy'),
            R2_energy=('serving an order necessarily charges at least its pre-cooling energy q*0.15*(25-target_c) plus a '
                       'non-negative door term, so the served set obeys sum(e_lower) <= B'),
            R3_vehicle_time=('serving an order consumes at least one out-and-back trip plus the loading service time, so '
                             'the served set obeys sum(tau) <= num_vehicles * horizon'),
            aggregation='fractional knapsack on R2, on R3, and their minimum; each is an upper bound on the 0/1 knapsack',
            variants=dict(tw_only='time window only', tw_and_horizon='time window plus return inside the booking horizon'),
            variant_caveat=('tw_and_horizon is tighter and therefore riskier; the recorded arms decide whether it is '
                            'admissible, because an arm-day that serves an order outside that set falsifies it')),
        falsification_oracle=['utility <= bound on every arm-day', 'n_served <= reachable on every arm-day',
                              'revenue <= reachable revenue on every arm-day',
                              'accumulated vehicle time within num_vehicles*horizon on every arm-day'],
        known_boundary=('the recorded control rows come from a batch whose run_a1_step2_gate.py and scenario_saa.py '
                        'differ from the current files; this stage deliberately depends only on unchanged modules and '
                        'rebuilds the contract directly, and it verifies both the contract hash and the dataset hash '
                        'against the recorded identity before using the rows'),
        supervision_cap_seconds=600,
        resource=dict(python='3.12.7', numpy='1.26.4', cpu_affinity=[7], no_model_min_available_ram_gib=0.5),
        formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        stable_online_gain=False,
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
