"""Same-incumbent OR-Tools GLS and cold-chain-certified destroy/regret repair."""
from copy import deepcopy
import math
import time
import numpy as np
from delivery_evaluator import evaluate, validate_instance


def matrices(instance):
    x = np.asarray(instance['coords'], float)
    return np.sqrt(np.sum((x[:, None] - x[None, :]) ** 2, axis=-1))


def route_ok(route, instance, contract, dist, wave):
    if sum(float(instance['demands'][i]) for i in route) > contract.capacity + 1e-9:
        return False
    t, prev = wave + contract.loading_service_h, 0
    for i in route:
        if max(instance['ready_time'][i], instance['reveal_time'][i]) > wave + 1e-9:
            return False
        t = max(t + dist[prev, i] / contract.speed_kmph, instance['tw_start'][i])
        if t > instance['tw_end'][i] + 1e-9:
            return False
        t += instance['service_time'][i]
        prev = i
    return t + dist[prev, 0] / contract.speed_kmph <= contract.horizon_h + 1e-9


def _initial(instance, routes, contract, wave, deadline):
    report = evaluate(instance, routes, contract, wave, deadline)
    if not report['hard']:
        raise ValueError('Shared initial delivery route is not certified: ' + str(report['errors']))
    return report


def improve_ortools(instance, initial_routes, contract, time_limit_s=1.0, wave_h=0.0):
    start = time.perf_counter()
    deadline = start + time_limit_s
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError('Invalid total solver deadline')
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    initial = _initial(instance, initial_routes, contract, wave_h, deadline)
    d = matrices(instance)
    n, v = len(d), contract.vehicles
    if n == 1:
        return {'routes': [], 'report': initial, 'wall_s': time.perf_counter()-start,
            'timeout': False, 'method': 'ortools_gls', 'status': 'empty_wave',
            'same_initial_assignment': True, 'online_gain_established': False}
    scale = 1000000
    manager = pywrapcp.RoutingIndexManager(n, v, 0)
    routing = pywrapcp.RoutingModel(manager)
    distance = routing.RegisterTransitCallback(lambda i, j: int(round(d[manager.IndexToNode(i), manager.IndexToNode(j)] * scale)))
    routing.SetArcCostEvaluatorOfAllVehicles(distance)
    transit = routing.RegisterTransitCallback(lambda i, j: int(math.ceil((
        d[manager.IndexToNode(i), manager.IndexToNode(j)] / contract.speed_kmph +
        float(instance['service_time'][manager.IndexToNode(i)])) * scale)))
    horizon = int(math.floor(contract.horizon_h * scale))
    routing.AddDimension(transit, horizon, horizon, False, 'Time')
    td = routing.GetDimensionOrDie('Time')
    for i in range(1, n):
        td.CumulVar(manager.NodeToIndex(i)).SetRange(int(math.ceil(instance['tw_start'][i] * scale)),
                                                  int(math.floor(instance['tw_end'][i] * scale)))
    for k in range(v):
        td.CumulVar(routing.Start(k)).SetValue(int(math.ceil((wave_h + contract.loading_service_h) * scale)))
    consumption = routing.RegisterTransitCallback(lambda i, j: int(math.ceil(
        float(instance['demands'][manager.IndexToNode(j)]) * 1000)))
    routing.AddDimensionWithVehicleCapacity(consumption, 0,
        [int(math.floor(contract.capacity * 1000))] * v, True, 'Capacity')
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    reserve = max(0.02, 3*(time.perf_counter()-start))
    remaining = deadline - time.perf_counter() - reserve
    if remaining <= 0:
        return {'routes': deepcopy(initial_routes), 'report': initial,
            'wall_s': time.perf_counter()-start, 'timeout': True, 'method': 'ortools_gls', 'status': 'prep_budget_exhausted',
            'same_initial_assignment': True, 'online_gain_established': False}
    params.time_limit.FromMilliseconds(max(1, int(remaining * 1000)))
    routing.CloseModelWithParameters(params)
    seed_routes = deepcopy(initial_routes) + [[] for _ in range(v - len(initial_routes))]
    assignment = routing.ReadAssignmentFromRoutes(seed_routes, True)
    if assignment is None:
        raise RuntimeError('OR-Tools rejected the common certified incumbent; preserve rounding failure')
    # Recompute remaining time after constructing and importing the incumbent.
    params.time_limit.FromMilliseconds(max(1, int(max(0, deadline-time.perf_counter()-reserve)*1000)))
    solution = routing.SolveFromAssignmentWithParameters(assignment, params)
    best, report = deepcopy(initial_routes), initial
    status = 'no_candidate'
    if solution is not None and time.perf_counter() < deadline:
        candidate = []
        for k in range(v):
            idx, route = routing.Start(k), []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node:
                    route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            if route:
                candidate.append(route)
        checked = evaluate(instance, candidate, contract, wave_h, deadline)
        status = 'candidate_certified' if checked['hard'] else 'candidate_rejected'
        if checked['hard'] and checked['cost'] < report['cost']:
            best, report = candidate, checked
    late = time.perf_counter() >= deadline
    if late:
        best, report = deepcopy(initial_routes), initial
    return {'routes': best, 'report': report, 'wall_s': time.perf_counter()-start,
        'timeout': late, 'method': 'ortools_gls', 'status': status,
        'proposal_objective': 'travel_distance', 'selection_objective': 'delivery_contract_cost',
        'same_initial_assignment': True, 'online_gain_established': False}


def improve_regret_lns(instance, initial_routes, contract, time_limit_s=1.0, seed=41, wave_h=0.0):
    start = time.perf_counter()
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError('Invalid total LNS deadline')
    deadline = start + time_limit_s
    report = _initial(instance, initial_routes, contract, wave_h, deadline)
    initial_report = report
    dist = matrices(instance)
    best = deepcopy(initial_routes) + [[] for _ in range(contract.vehicles-len(initial_routes))]
    current, current_cost = deepcopy(best), report['cost']
    rng = np.random.default_rng(seed)
    customers = list(range(1, len(dist)))
    tried, feasible = 0, 0
    if not customers:
        return {'routes': [], 'report': report, 'wall_s': time.perf_counter()-start,
            'timeout': False, 'method': 'destroy_regret_lns', 'tried': 0, 'feasible': 0,
            'same_initial_assignment': True, 'online_gain_established': False}
    reserve = max(0.01, 2*(time.perf_counter()-start))
    while time.perf_counter() < deadline-reserve:
        tried += 1
        removed = set(map(int, rng.choice(customers, size=min(len(customers), max(2, len(customers)//5)), replace=False)))
        candidate = [[i for i in route if i not in removed] for route in current]
        remaining = sorted(removed)
        while remaining and time.perf_counter() < deadline-reserve:
            choices = []
            for i in remaining:
                if time.perf_counter() >= deadline-reserve:
                    break
                options = []
                for k, route in enumerate(candidate):
                    if time.perf_counter() >= deadline-reserve:
                        break
                    for at in range(len(route)+1):
                        if time.perf_counter() >= deadline-reserve:
                            break
                        new = route[:at] + [i] + route[at:]
                        if route_ok(new, instance, contract, dist, wave_h):
                            prev, nxt = (route[at-1] if at else 0), (route[at] if at<len(route) else 0)
                            delta = dist[prev, i]+dist[i, nxt]-dist[prev, nxt]
                            options.append((delta, k, at))
                options.sort()
                if options:
                    regret = (options[1][0]-options[0][0]) if len(options)>1 else float('inf')
                    choices.append((regret, i, options[0]))
            if not choices:
                break
            if time.perf_counter() >= deadline-reserve:
                break
            _, i, (_, k, at) = max(choices, key=lambda x: (x[0], -x[1]))
            candidate[k].insert(at, i)
            remaining.remove(i)
        if remaining or time.perf_counter() >= deadline-reserve:
            continue
        checked = evaluate(instance, candidate, contract, wave_h, deadline)
        if not checked['hard']:
            continue
        feasible += 1
        if checked['cost'] < report['cost']:
            best, report = deepcopy(candidate), checked
        temp = max(1e-6, 0.02*report['cost']*(deadline-time.perf_counter())/time_limit_s)
        if checked['cost'] <= current_cost or rng.random() < math.exp(max(-700, (current_cost-checked['cost'])/temp)):
            current, current_cost = candidate, checked['cost']
    late = time.perf_counter() >= deadline
    if late:
        best, report = deepcopy(initial_routes), initial_report
    return {'routes': [r for r in best if r], 'report': report,
        'wall_s': time.perf_counter()-start, 'timeout': late, 'method': 'destroy_regret_lns',
        'tried': tried, 'feasible': feasible, 'same_initial_assignment': True,
        'proposal_objective': 'distance_regret2', 'selection_objective': 'delivery_contract_cost',
        'online_gain_established': False}
