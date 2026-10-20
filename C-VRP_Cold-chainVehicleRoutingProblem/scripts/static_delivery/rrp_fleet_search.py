"""Masked giant-order reconstruction and common independent fleet Split."""
from __future__ import annotations

import math
import time
import numpy as np
from rrp_fleet import split_giant_tour
from rrp_search import _cost


def _proposals(route, waits, report):
    by_node = dict(zip(route, waits))
    for a in range(1, len(route) - 1):
        for b in range(a + 1, len(route)):
            candidate = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
            yield candidate, [by_node[v] for v in candidate], 'directed_2opt'
    for a in range(1, len(route)):
        for b in range(1, len(route)):
            if a != b:
                candidate = route.copy()
                candidate.insert(b, candidate.pop(a))
                yield candidate, [by_node[v] for v in candidate], 'relocate'
    for tour in report.get('tours', []):
        for leg in tour['legs'][1:]:
            position = route.index(leg['from'])
            arrival, service = leg['physical_arrival_s'], leg['service_s']
            values = {0.0}
            for hour in range(int(arrival // 3600) + 1, 24):
                values.update((hour * 3600 - arrival, hour * 3600 - arrival - service))
            for value in sorted(v for v in values if 0 <= v < tour['duration_limit_s'] and v != waits[position]):
                trial = waits.copy()
                trial[position] = value
                yield route.copy(), trial, 'hour_boundary_wait'


def solve_fleet(instance, initial_route, initial_waits, *, max_vehicles, method,
                iterations, seconds, objective, profile, model=None,
                keeping_rate=.2, seed=888, time_domain='source_day_domain'):
    if method not in ('local_search', 'maskco', 'maskco_initial'):
        raise ValueError('Unknown method')
    if type(iterations) is not int or iterations < 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid search limit')
    if not math.isfinite(keeping_rate) or not 0 <= keeping_rate <= 1:
        raise ValueError('Invalid keeping rate')
    if method != 'local_search' and model is None:
        raise ValueError('Actual MaskCO model is required')
    started = time.perf_counter()
    deadline = started + seconds
    route, waits = list(initial_route), list(initial_waits)
    incumbent = split_giant_tour(instance, route, waits, max_vehicles, objective,
                                profile=profile, time_domain=time_domain, deadline=deadline)
    calls, evaluations, forwards = 1, incumbent['actual_tour_evaluations'], 0
    rng, records, failures = np.random.default_rng(seed), [], []
    if not incumbent['report']['feasible']:
        failures.append({'stage': 'initial_split', 'route': route, 'waits_s': waits,
                         'errors': incumbent['report']['errors']})
    for iteration in range(iterations):
        if time.perf_counter() >= deadline:
            break
        if method == 'local_search':
            proposals = _proposals(route, waits, incumbent['report'])
        else:
            from rrp_reconstruct import reconstruct
            edges = list(zip(route, route[1:] + route[:1]))
            indices = rng.choice(len(edges), int(math.floor(keeping_rate * len(edges))), replace=False)
            kept = {edges[int(i)] for i in indices}
            candidate = reconstruct(model, instance, route, kept, waits, objective=objective,
                                    profile=profile, time_domain=time_domain)
            forwards += 1
            proposals = [(candidate['route'], candidate['waits_s'], 'masked_giant_order')]
        accepted, attempts = False, 0
        for order, candidate_waits, operator in proposals:
            if time.perf_counter() >= deadline:
                break
            checked = split_giant_tour(instance, order, candidate_waits, max_vehicles, objective,
                                       profile=profile, time_domain=time_domain, deadline=deadline)
            calls += 1
            evaluations += checked['actual_tour_evaluations']
            attempts += 1
            if not checked['report']['feasible']:
                failures.append({'stage': operator, 'iteration': iteration,
                                 'route': order, 'waits_s': candidate_waits,
                                 'errors': checked['report']['errors']})
            elif time.perf_counter() < deadline and (_cost(incumbent['report']) is None
                    or _cost(checked['report']) < _cost(incumbent['report'])):
                route, waits, incumbent = list(order), list(candidate_waits), checked
                accepted = True
                if method == 'local_search':
                    break
        records.append({'iteration': iteration, 'attempts': attempts, 'accepted': accepted,
                        'best_cost_eur_year': _cost(incumbent['report'])})
    return {'method': method, 'giant_route': route, 'giant_waits_s': waits,
            'routes': incumbent['routes'], 'waits_s_by_route': incumbent['waits_s_by_route'],
            'report': incumbent['report'], 'optimality_scope': incumbent['optimality_scope'],
            'iterations': records, 'failure_log': failures,
            'actual_split_calls': calls, 'actual_tour_evaluations': evaluations,
            'actual_model_forwards': forwards, 'wall_s': time.perf_counter() - started,
            'requested_search_seconds': seconds,
            'budget_scope': 'initial_and_candidate_Split_model_preparation_and_independent_acceptance',
            'keeping_rate': keeping_rate, 'seed': seed,
            'learned_fleet_assignment_head': False,
            'retained_edges_scope': 'giant_order_adjacency_Split_may_cut_vehicle_arcs',
            'scientific_gain_established': False}
