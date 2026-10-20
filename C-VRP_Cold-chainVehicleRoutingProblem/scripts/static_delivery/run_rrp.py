"""Static public RRP entry point; every candidate uses independent evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from rrp_data import (EXTENSION_ROOT, SOURCE, load_instance, load_reference,
                      public_instances, source_protocol, PRIMARY_PROFILE, TIME_DOMAINS,
                      reference_wait_mapping)
from rrp_evaluator import evaluate, PROFILES, OBJECTIVES


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def audit(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in public_instances():
        instance = load_instance(path)
        ref = load_reference(SOURCE / 'Solutions' / path.parent.name / ('sol-' + path.stem + '.txt'))
        diagnostic_waits, diagnostic_origin = reference_wait_mapping(ref.route, ref.raw_waits_s, ref.saved)
        if ref.label_origin == 'derived_unique_zero_from_author_clock_equalities':
            diagnostic_waits, diagnostic_origin = ref.waits_s, ref.label_origin
        entry = {'instance': instance.name, 'source_sha256': instance.source_sha256,
                 'reference_sha256': ref.source_sha256, 'reference_objective': ref.objective,
                 'reference_costs': ref.reported_costs, 'reference_saved': ref.saved,
                 'reference_raw_waits_s': ref.raw_waits_s, 'mapped_waits_s': ref.waits_s,
                 'reference_wait_convention': ref.wait_convention,
                 'reference_label_origin': ref.label_origin,
                 'saved_clock_convention_diagnostic_only': ref.saved_clock_convention,
                 'archive_diagnostic_waits_s': diagnostic_waits,
                 'archive_diagnostic_wait_origin': diagnostic_origin,
                 'paper_repository_wait_conflict': instance.name in source_protocol()['paper_repository_wait_conflicts'],
                 'profiles': {}}
        for profile in PROFILES:
            waits = diagnostic_waits if profile == 'archive_compatibility' else ref.waits_s
            report = evaluate(instance, ref.route, waits, ref.objective, profile=profile)
            calculated = {}
            if report['costs'] is not None:
                c = report['costs']
                calculated = {k: c[k + '_eur_year'] for k in ('traction', 'transmission', 'infiltration', 'driver', 'total')}
                calculated['refrigeration'] = calculated['transmission'] + calculated['infiltration']
            deltas = {key: calculated[key] - value for key, value in ref.reported_costs.items() if key in calculated}
            clock_delta = None
            if report['complete']:
                arrivals = [leg['arrival_s'] for leg in report['legs']] + [report['return_arrival_s']]
                clock_delta = [a - b for a, b in zip(arrivals, ref.saved['Arrival times'])]
            entry['profiles'][profile] = {'evaluation': report, 'cost_residual_eur_year': deltas,
                                          'arrival_residual_s': clock_delta}
        entry['explicit_operational_9h_check'] = evaluate(instance, ref.route, ref.waits_s, ref.objective,
                                                       profile=PRIMARY_PROFILE, time_domain='operational_tour_9h')
        rows.append(entry)
    report = {'source_protocol': source_protocol(), 'instances': len(rows), 'rows': rows,
              'status': 'PAPER_STANDARD_AUDITED_REFERENCE_DIVERGENCES_DISCLOSED',
              'author_algorithm_reproduced': False, 'all_author_costs_reproduced': False,
              'definition': 'Clocks, loads and costs recomputed from route/waits, never copied from saved author clocks',
              'no_posthoc_tolerance_gate': True}
    dump(out / 'reference_audit.json', report)
    print(json.dumps({'instances': len(rows), 'status': report['status'], 'output': str(out)}))


def selected_case(name):
    paths = [p for p in public_instances() if p.stem == name]
    if len(paths) != 1:
        raise ValueError('Choose one exact public instance name')
    return load_instance(paths[0])


def identity():
    from rrp_data import COMMIT, PROTOCOL_PATH
    return {'author_commit': COMMIT, 'python': platform.python_version(),
            'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            'native_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')}}


def teachers(names, objective, profile, time_domain='source_day_domain'):
    accepted, rejected = [], []
    for name in names:
        instance = selected_case(name)
        ref = load_reference(SOURCE / 'Solutions' / name[:2] / ('sol-' + name + '.txt'))
        checked = evaluate(instance, ref.route, ref.waits_s, objective, profile=profile, time_domain=time_domain)
        row = {'instance': name, 'source_sha256': instance.source_sha256,
               'reference_sha256': ref.source_sha256, 'original_objective': ref.objective,
               'label_origin': ref.label_origin, 'raw_wait_count': len(ref.raw_waits_s),
               'evaluation': checked}
        if not checked['feasible'] or ref.objective != objective or ref.wait_convention == 'incomplete_or_unresolved':
            row['rejection'] = 'teacher_infeasible_under_profile_or_objective_mismatch'
            rejected.append(row)
        else:
            accepted.append((instance, ref, row))
    return accepted, rejected


def train_cases(*, names, out, steps, model_profile, seed, objective, profile,
                learning_rate, weight_decay, split=None, smoke=False, time_domain='source_day_domain'):
    from rrp_learning import (Trainer, make_batch, predict, parameter_digest,
                              save_checkpoint, load_checkpoint, validate_split)
    from rrp_model import make_model
    if type(steps) is not int or steps < 1:
        raise ValueError('Explicit positive steps required')
    if split is not None:
        validate_split(split)
    elif not smoke:
        raise ValueError('Formal training requires an explicit grouped split')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    accepted, rejected = teachers(names, objective, profile, time_domain)
    dump(out / 'teacher_validation.json', {'accepted': [a[2] for a in accepted], 'rejected': rejected})
    if rejected or not accepted:
        raise ValueError('Teacher validation failed; see teacher_validation.json. No silent filtering.')
    started = time.perf_counter()
    model = make_model(model_profile, seed)
    before = parameter_digest(model)
    trainer = Trainer(model, learning_rate, weight_decay)
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(steps):
        instance, ref, _ = accepted[step % len(accepted)]
        edges = list(zip(ref.route, ref.route[1:] + ref.route[:1]))
        count = int(np.floor(rng.uniform() * len(edges)))
        kept = {edges[int(i)] for i in rng.choice(len(edges), count, replace=False)}
        batch = make_batch(instance, ref.route, ref.waits_s, kept, objective=objective, profile=profile, time_domain=time_domain)
        metrics = trainer.step(batch)
        rows.append(dict(step=step + 1, instance=instance.name, retained_edges=count, **metrics))
        print(json.dumps(rows[-1]), flush=True)
    after = parameter_digest(model)
    if before == after or not any(r['updated_parameter_leaves'] > 0 for r in rows):
        raise RuntimeError('No actual parameter update')
    metadata = {'identity': identity(), 'training_cases': [a[2]['instance'] for a in accepted],
                'source_sha256': {a[0].name: a[0].source_sha256 for a in accepted},
                'objective': objective, 'profile': profile, 'time_domain': time_domain, 'split': split,
                'scope': 'functional_smoke_only' if smoke else 'public_reference_supervision',
                'optimizer': 'AdamW adaptation; not original MaskCO Muon',
                'loss': 'masked_successor_CE_plus_log1p_wait_MSE_equal_weight_design_choice'}
    save_checkpoint(out / 'checkpoint', trainer, metadata)
    expected = predict(model, batch)
    loaded, _ = load_checkpoint(out / 'checkpoint')
    actual = predict(loaded.model, batch)
    reload_exact = all(np.array_equal(a, b) for a, b in zip(expected, actual))
    if not reload_exact:
        raise RuntimeError('Checkpoint predictions changed on reload')
    report = {'metadata': metadata, 'updates': rows, 'weight_before_sha256': before,
              'weight_after_sha256': after, 'checkpoint_reload_predictions_exact': reload_exact,
              'wall_s': time.perf_counter() - started,
              'jax_backend': __import__('jax').default_backend(),
              'generalization_established': False, 'method_gain_established': False}
    dump(out / 'training.json', report)
    return report


def solve_case(*, case, out, methods, iterations, seconds, objective, profile, seed,
               checkpoint, model_profile, keeping_rate, time_domain='source_day_domain', max_vehicles=None):
    from rrp_search import nearest_route, solve
    started = time.perf_counter()
    instance = selected_case(case)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    route, waits = nearest_route(instance), [0.0] * instance.nodes
    model = None
    model_identity = None
    models = {}
    if 'maskco' in methods:
        if checkpoint is None:
            raise ValueError('Trained maskco method requires --checkpoint')
        from rrp_learning import load_checkpoint, parameter_digest
        trainer, metadata = load_checkpoint(checkpoint)
        if (metadata.get('objective'), metadata.get('profile'), metadata.get('time_domain')) != (objective, profile, time_domain):
            raise ValueError('Checkpoint objective/profile mismatch')
        model = trainer.model
        model_profile = model.config['profile']
        model_identity = {'parameter_sha256': parameter_digest(model), 'config': model.config, 'metadata': metadata}
        models['maskco'] = model_identity
    initial = None
    if 'maskco_initial' in methods:
        from rrp_model import make_model
        # Same architecture/initial seed as training, no label/reference lookup.
        initial = make_model(model_profile, model.config['seed'] if model is not None else seed)
        from rrp_learning import parameter_digest
        models['maskco_initial'] = {'config': initial.config, 'parameter_sha256': parameter_digest(initial),
                                    'scope': 'untrained_initialization_control'}
    prep_s = time.perf_counter() - started
    results = []
    options = {}
    if max_vehicles is not None:
        from rrp_fleet_search import solve_fleet
        solve = solve_fleet
        options['max_vehicles'] = max_vehicles
    for method in methods:
        results.append(solve(instance, route, waits, method=method, iterations=iterations,
                             seconds=seconds, objective=objective, profile=profile,
                             model=initial if method == 'maskco_initial' else model,
                             keeping_rate=keeping_rate, seed=seed, time_domain=time_domain, **options))
        results[-1]['model_identity'] = models.get(method)
    result = {'case': case, 'identity': identity(), 'objective': objective, 'profile': profile, 'time_domain': time_domain,
              'model': model_identity, 'models': models, 'same_initial_route': route, 'same_initial_waits_s': waits,
              'preparation_s': prep_s, 'total_wall_s': time.perf_counter() - started,
              'methods': results, 'scope': 'functional_common_objective_comparison',
              'reference_solution_used_for_initialization': False,
              'evaluation_protocol': 'same start/seed/seconds/iterations; forwards and JIT count in search time',
              'algorithm_reproduction': 'independent evaluator and adapted MaskCO, not author LAHC/MIP/CP'}
    if max_vehicles is not None:
        result['fleet_extension'] = {'max_vehicles': max_vehicles, 'customer_data': 'unchanged_public_RRP',
                                     'capacity_pallets': source_protocol()['pallet_capacity'],
                                     'fleet_size_origin': 'explicit_experiment_bound',
                                     'split_reference': ['10.1016/S0305-0548(03)00158-8', '10.1016/j.cor.2015.11.012'],
                                     'learned_assignment_head': False}
    dump(out / 'result.json', result)
    print(json.dumps({'case': case, 'methods': len(results), 'output': str(out)}), flush=True)
    return result


def fleet_scan(out, max_vehicles, seconds=15):
    """Bounded inference on every original case; no reference orders/labels."""
    from rrp_search import nearest_route
    from rrp_fleet_search import solve_fleet
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in public_instances():
        instance = load_instance(path)
        for domain in TIME_DOMAINS:
            result = solve_fleet(instance, nearest_route(instance), [0.] * instance.nodes,
                                 max_vehicles=max_vehicles, method='local_search', iterations=0,
                                 seconds=seconds, objective='with_wage', profile=PRIMARY_PROFILE,
                                 time_domain=domain)
            rows.append({'case': instance.name, 'time_domain': domain, 'result': result})
    report = {'identity': identity(), 'rows': rows, 'max_vehicles': max_vehicles,
              'scope': 'functional_fixed_nearest_order_Split_scan_no_labels_no_generalization_claim',
              'objective': 'with_wage', 'seconds_per_case_and_domain': seconds}
    dump(out / 'scan.json', report)
    print(json.dumps({'fleet_scan_cases': len(rows), 'feasible': sum(r['result']['report']['feasible'] for r in rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='Independently recompute all 30 public reference records')
    a.add_argument('--out', type=Path, required=True)
    for command in ('smoke', 'train', 'solve', 'fleet-smoke', 'fleet-solve'):
        a = sub.add_parser(command)
        a.add_argument('--out', type=Path, required=True)
        a.add_argument('--case', default='D1-n4-np33')
        a.add_argument('--profile', choices=PROFILES, default=PRIMARY_PROFILE)
        a.add_argument('--time-domain', choices=TIME_DOMAINS, default='source_day_domain')
        a.add_argument('--objective', choices=OBJECTIVES, default='with_wage')
        a.add_argument('--model-profile', choices=('maskco_256', 'smoke_cpu'), default='maskco_256')
        a.add_argument('--seed', type=int, default=888)
        a.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
        if command not in ('solve', 'fleet-solve'):
            a.add_argument('--steps', type=int, required=True)
            a.add_argument('--learning-rate', type=float, default=1e-3)
            a.add_argument('--weight-decay', type=float, default=1e-2)
        if command == 'train':
            a.add_argument('--split', type=Path, required=True)
        if command in ('smoke', 'solve', 'fleet-smoke', 'fleet-solve'):
            a.add_argument('--iterations', type=int, default=3)
            a.add_argument('--seconds', type=float, default=15)
            a.add_argument('--keeping-rate', type=float, default=.2)
        if command in ('solve', 'fleet-solve'):
            a.add_argument('--checkpoint', type=Path)
            a.add_argument('--methods', nargs='+', choices=('local_search', 'maskco', 'maskco_initial'),
                           default=['local_search'])
        if command.startswith('fleet-'):
            a.add_argument('--max-vehicles', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args.out)
        return
    # Set the backend before any lazy import of JAX/Flax or an upstream model.
    os.environ['JAX_PLATFORMS'] = args.device
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    smoke = args.command in ('smoke', 'fleet-smoke')
    if args.command in ('train', 'smoke', 'fleet-smoke'):
        split = json.loads(args.split.read_text(encoding='utf-8')) if args.command == 'train' else None
        train_out = args.out / 'train' if smoke else args.out
        train_cases(names=split['train'] if split is not None else [args.case], out=train_out,
                    steps=args.steps, model_profile=args.model_profile, seed=args.seed,
                    objective=args.objective, profile=args.profile, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, split=split, smoke=smoke, time_domain=args.time_domain)
    if args.command in ('smoke', 'solve', 'fleet-smoke', 'fleet-solve'):
        common = dict(case=args.case,
                   methods=['local_search', 'maskco_initial', 'maskco'] if smoke else args.methods,
                   checkpoint=args.out / 'train/checkpoint' if smoke else args.checkpoint,
                   model_profile=args.model_profile, iterations=args.iterations, seconds=args.seconds,
                   objective=args.objective, profile=args.profile, seed=args.seed, keeping_rate=args.keeping_rate, time_domain=args.time_domain)
        if args.command == 'fleet-smoke':
            audit(args.out / 'audit')
            solve_case(out=args.out / 'single', **common)
            solve_case(out=args.out / 'fleet', max_vehicles=args.max_vehicles, **common)
            fleet_scan(args.out / 'fleet_scan', args.max_vehicles)
        else:
            solve_case(out=args.out / 'solve' if smoke else args.out,
                       max_vehicles=args.max_vehicles if args.command == 'fleet-solve' else None, **common)


if __name__ == '__main__':
    main()
