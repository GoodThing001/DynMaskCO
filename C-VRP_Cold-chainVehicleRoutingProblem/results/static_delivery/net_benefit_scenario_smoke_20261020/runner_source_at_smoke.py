"""Bounded, explicit-hypothesis net-benefit pilot on intact published cases.

OR-Tools constructs a common cost warm start. Pinned upstream ALNS then runs
paired cost-only and net-benefit arms with the SAME time budget and random seed.
These are objective ablations, not a MaskCO comparison or a paper replication.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

from net_benefit_model import CONFIG_PATH, EXTENSION_ROOT, RouteModel, config, load_case, structural_audit
from net_benefit_audit import audit

COMPARISONS = EXTENSION_ROOT/'comparisons/static_coldchain'
DEFAULT_CASES = ['c101', 'c105', 'c201', 'c205', 'r101', 'r201', 'rc101', 'rc201']


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def cost_warm_start(ins, c, fleet, seconds):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ts, cs = 1000, 10000000
    nodes, distances = ins['nodes'], ins['distance']
    manager = pywrapcp.RoutingIndexManager(len(nodes), fleet, 0)
    model = pywrapcp.RoutingModel(manager)
    cost_matrix = [[round(d*c['routing']['distance_EUR_per_km']*cs) for d in row] for row in distances]
    times = [[math.ceil((d*60/c['routing']['speed_km_per_hour']+nodes[i][6])*ts) for d in row] for i, row in enumerate(distances)]
    cost = model.RegisterTransitMatrix(cost_matrix)
    model.SetArcCostEvaluatorOfAllVehicles(cost)
    dem = model.RegisterUnaryTransitVector([int(row[3]) for row in nodes])
    model.AddDimensionWithVehicleCapacity(dem, 0, [c['routing']['capacity_kg']]*fleet, True, 'Load')
    travel = model.RegisterTransitMatrix(times)
    horizon = round(nodes[0][5]*ts)
    model.AddDimension(travel, horizon, horizon, False, 'Time')
    dim = model.GetDimensionOrDie('Time')
    for i in range(1, len(nodes)):
        dim.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(nodes[i][4]*ts), math.floor(nodes[i][5]*ts))
    for k in range(fleet):
        dim.CumulVar(model.Start(k)).SetValue(round(nodes[0][4]*ts))
        dim.CumulVar(model.End(k)).SetRange(round(nodes[0][4]*ts), horizon)
    rate = c['refrigeration_cost_scenario']['cost_EUR_per_vehicle_min_derived']
    span_coefficient = round(rate*cs/ts)
    assert math.isclose(span_coefficient*ts/cs, rate)
    dim.SetSpanCostCoefficientForAllVehicles(span_coefficient)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(round(seconds*1000))
    assignment = model.SolveWithParameters(params)
    if assignment is None:
        return None, {'status': model.status(), 'reason': 'no feasible warm start within budget; not an infeasibility proof'}
    routes = []
    for k in range(fleet):
        idx, route = model.Start(k), []
        while not model.IsEnd(idx):
            i = manager.IndexToNode(idx)
            if i:
                route.append(i)
            idx = assignment.Value(model.NextVar(idx))
        if route:
            routes.append(tuple(route))
    return tuple(routes), {'status': model.status(), 'time_integer_scale': ts, 'cost_integer_scale': cs, 'travel_rounding': 'ceil; independent raw double audit', 'objective': 'distance plus vehicle refrigeration-duration proxy; no lexicographic min-vehicle term', 'random_seed_control': 'not applied to OR-Tools Routing'}


class State:
    def __init__(self, model, routes, pending=(), fallback=None):
        self.model, self.routes, self.pending, self.fallback = model, tuple(routes), tuple(pending), fallback
        self.value = math.inf if pending else model.objective(self.routes)

    def objective(self):
        return self.value


def destroy(state, rng, related):
    ids = [i for r in state.routes for i in r]
    count = int(rng.integers(4, 13))
    if related:
        center = int(rng.choice(ids))
        removed = sorted(ids, key=lambda i: (state.model.ins['distance'][center][i], i))[:count]
    else:
        removed = [int(i) for i in rng.choice(ids, size=count, replace=False)]
    removed_set = set(removed)
    routes = tuple(tuple(i for i in r if i not in removed_set) for r in state.routes)
    return State(state.model, tuple(r for r in routes if r), removed, state)


def random_removal(state, rng):
    return destroy(state, rng, False)


def related_removal(state, rng):
    return destroy(state, rng, True)


def repair(state, rng, earliest_due):
    model, routes = state.model, list(state.routes)
    pending = list(state.pending)
    if earliest_due:
        pending.sort(key=lambda i: (model.ins['nodes'][i][5], i))
    else:
        rng.shuffle(pending)
    for i in pending:
        best = (math.inf, None, None)
        for ridx, route in enumerate(routes):
            base = model.route(route)
            for pos in range(len(route)+1):
                trial = route[:pos]+(i,)+route[pos:]
                delta = model.route(trial)-base
                if delta < best[0]:
                    best = (delta, ridx, pos)
        if len(routes) < model.max_vehicles:
            extra = model.route((i,))
            if extra < best[0]:
                best = (extra, len(routes), 0)
        if best[1] is None:
            return state.fallback
        _, ridx, pos = best
        if ridx == len(routes):
            routes.append((i,))
        else:
            routes[ridx] = routes[ridx][:pos]+(i,)+routes[ridx][pos:]
    return State(model, tuple(routes))


def random_order_greedy(state, rng):
    return repair(state, rng, False)


def earliest_due_greedy(state, rng):
    return repair(state, rng, True)


def alns_solve(model, initial, seconds, seed):
    import numpy as np
    from alns import ALNS
    from alns.accept import HillClimbing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime
    algorithm = ALNS(np.random.default_rng(seed))
    algorithm.add_destroy_operator(random_removal)
    algorithm.add_destroy_operator(related_removal)
    algorithm.add_repair_operator(random_order_greedy)
    algorithm.add_repair_operator(earliest_due_greedy)
    selector = RouletteWheel([5, 2, 1, 0.1], 0.8, 2, 2)
    start = time.perf_counter()
    result = algorithm.iterate(State(model, initial), selector, HillClimbing(), MaxRuntime(seconds))
    elapsed = time.perf_counter()-start
    return result.best_state.routes, {'seed': seed, 'budget_s': seconds, 'elapsed_s': elapsed, 'iterations': len(result.statistics.objectives)-1, 'stop_boundary': 'checked between complete iterations; elapsed can exceed budget by one iteration', 'destroy_counts': dict(result.statistics.destroy_operator_counts), 'repair_counts': dict(result.statistics.repair_operator_counts), 'destroy_weights': selector.destroy_weights.tolist(), 'repair_weights': selector.repair_weights.tolist(), 'route_cache': str(model.route.cache_info())}


def record(ins, c, routes, fleet, method, rho_optimized, seed, meta):
    center = c['tomato']['rho_center']
    main_report = audit(ins['nodes'], routes, c, center, fleet)
    target_report = audit(ins['nodes'], routes, c, rho_optimized, fleet)
    independent = -RouteModel(ins, c, rho_optimized, fleet).objective(routes)
    if not main_report['feasible_routing'] or not math.isclose(independent, target_report['net_benefit_scenario_EUR'], abs_tol=1e-6):
        raise ValueError('independent routing/accounting audit failed')
    sensitivity = []
    for setting in c['tomato']['rho_sensitivity']:
        r = audit(ins['nodes'], routes, c, setting['rho'], fleet)
        sensitivity.append({'name': setting['name'], **{k: v for k, v in r.items() if k != 'tours'}})
    return {'case': ins['name'], 'method': method, 'optimization_rho': rho_optimized, 'seed': seed, 'source_sha256': ins['sha256'], 'source_file': ins['path'], 'routes': routes, 'solver': meta, 'report_at_common_center_rho': main_report, 'fixed_routes_sensitivity_not_reoptimization': sensitivity, 'optimization_target_net_benefit_EUR_independently_checked': target_report['net_benefit_scenario_EUR']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-vehicles', type=int, required=True)
    parser.add_argument('--warm-seconds', type=float, default=5)
    parser.add_argument('--search-seconds', type=float, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261020, 20261021, 20261022])
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    args = parser.parse_args()
    if not 1 <= args.max_vehicles <= 100 or min(args.warm_seconds, args.search_seconds) <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive budgets, unique seeds and explicit fleet bound1..100 required')
    if args.out.exists():
        raise FileExistsError('use a NEW output directory')
    args.out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str((args.out/'mpl_cache').resolve())
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(COMPARISONS/'.runtime/packages'))
    sys.path.insert(0, str(COMPARISONS/'upstream/ALNS'))
    c = config()
    (args.out/'scenario_config.json').write_bytes(CONFIG_PATH.read_bytes())
    source_audit = []
    for path in sorted((EXTENSION_ROOT/c['dataset']['root']).glob('*.txt')):
        source_audit.append(structural_audit(load_case(path.stem, c), c))
    write_json(args.out/'all56_structural_audit.json', source_audit)
    records, skipped, failures = [], [], []
    for name in args.cases:
        ins = load_case(name, c)
        necessary = structural_audit(ins, c)
        if not necessary['passes_necessary_bounds']:
            skipped.append(necessary)
            print(json.dumps({'case': name, 'status': 'STRUCTURALLY_INFEASIBLE_UNDER_README_RETURN_HORIZON', 'customers': [v['customer'] for v in necessary['impossible_customers']]}), flush=True)
            continue
        started = time.perf_counter()
        initial, meta = cost_warm_start(ins, c, args.max_vehicles, args.warm_seconds)
        meta['build_and_search_elapsed_s'] = time.perf_counter()-started
        meta['requested_search_budget_s'] = args.warm_seconds
        if initial is None:
            failures.append({'case': name, **meta})
            print(json.dumps({'case': name, 'status': 'NO_WARM_START', **meta}), flush=True)
            continue
        case_rows = [record(ins, c, initial, args.max_vehicles, 'OR-Tools_cost_warm_start', 1., None, meta)]
        for seed in args.seeds:
            for label, rho in [('ALNS_cost_only_control', 1.), ('ALNS_net_benefit_hypothesis', c['tomato']['rho_center'])]:
                model = RouteModel(ins, c, rho, args.max_vehicles)
                routes, details = alns_solve(model, initial, args.search_seconds, seed)
                row = record(ins, c, routes, args.max_vehicles, label, rho, seed, details)
                case_rows.append(row)
                r = row['report_at_common_center_rho']
                print(json.dumps({'case': name, 'method': label, 'seed': seed, 'vehicles': r['vehicles_used'], 'net_benefit_scenario_EUR': r['net_benefit_scenario_EUR'], 'distance_km': r['distance_km'], 'iterations': details['iterations'], 'elapsed_s': details['elapsed_s']}), flush=True)
        records.extend(case_rows)
        write_json(args.out/f'{name}.json', case_rows)
    hashes = {str(p.relative_to(EXTENSION_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG_PATH, Path(__file__), Path(__file__).with_name('net_benefit_model.py'), Path(__file__).with_name('net_benefit_audit.py')]}
    pins = json.loads((COMPARISONS/'SOURCE_PINS.json').read_text(encoding='utf-8'))
    result = {'scope': c['status'], 'objective': c['objective'], 'max_vehicles': args.max_vehicles, 'fixed_departure_policy': c['routing']['departure_policy'], 'config_and_code_sha256': hashes, 'source_commit': c['dataset']['author_commit'], 'alns_pin': next(p for p in pins if p['name'] == 'ALNS'), 'runtime_versions': {n: importlib.metadata.version(n) for n in ['ortools', 'numpy', 'matplotlib']}, 'alns_imported_from': str(COMPARISONS/'upstream/ALNS/alns'), 'seeds_predeclared': args.seeds, 'paired_search_budget_s': args.search_seconds, 'shared_warm_start_budget_s': args.warm_seconds, 'comparison_role': 'paired objective ablation with common warm start; no MaskCO or claim of cross-solver fair-budget superiority', 'sensitivity_role': 're-evaluation of fixed routes, not reoptimized scenarios and not a confidence interval', 'all56_necessary_bound_failures': sum(not x['passes_necessary_bounds'] for x in source_audit), 'skipped_structural': skipped, 'warm_start_failures': failures, 'rows': records, 'new_customers_generated': False, 'tomato_TA_to_price_empirically_calibrated': False, 'tomato_marketability_threshold_configured': False, 'mullet_profit_computed': False, 'new_neural_training': False}
    write_json(args.out/'results.json', result)
    print(json.dumps({'completed_rows': len(records), 'skipped_cases': len(skipped), 'warm_start_failures': len(failures), 'all56_structural_conflicts': result['all56_necessary_bound_failures']}), flush=True)


if __name__ == '__main__':
    main()
