"""Static common-objective search with independent full-route acceptance."""
from __future__ import annotations

import math
import time
import numpy as np
from rrp_evaluator import evaluate


def nearest_route(instance):
    route, remaining = [0], set(range(1, instance.nodes))
    while remaining:
        node = min(remaining, key=lambda v: (instance.distances[route[-1], v], v))
        route.append(int(node))
        remaining.remove(node)
    return route


def _cost(report):
    return report['costs']['total_eur_year'] if report['feasible'] else None


def _rank(report):
    return (0, _cost(report)) if report['feasible'] else (1, report['duration_s'] or math.inf)


def _classical_candidates(instance, route, waits, report):
    by_node = dict(zip(route, waits))
    for a in range(1, len(route) - 1):
        for b in range(a + 1, len(route)):
            candidate = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
            yield candidate, [by_node[node] for node in candidate], 'directed_2opt'
    for a in range(1, len(route)):
        for b in range(1, len(route)):
            if a == b:
                continue
            candidate = route.copy()
            node = candidate.pop(a)
            candidate.insert(b, node)
            yield candidate, [by_node[node] for node in candidate], 'relocate'
    # Continuous values derived from actual source slot boundaries, not a
    # fabricated wait-time grid. This is a heuristic, not the author's LAHC.
    if not report['complete']:
        return
    speeds = sorted({row[4] for row in instance.raw['speed_ranges']})
    for position in range(1, len(route)):
        leg = report['legs'][position]
        arrival, service = leg['physical_arrival_s'], leg['service_s']
        candidates = {0.0}
        for hour in range(int(arrival // 3600) + 1, 24):
            boundary = hour * 3600
            candidates.update((boundary - arrival, boundary - arrival - service))
            for speed in speeds:
                candidates.add(boundary - arrival - service - leg['distance_km'] * 3600 / speed)
        for value in sorted(v for v in candidates if 0 <= v < report['duration_limit_s'] and v != waits[position]):
            trial = waits.copy()
            trial[position] = float(value)
            yield route.copy(), trial, 'hour_boundary_wait'


def solve(instance, initial_route, initial_waits, *, method, iterations, seconds,
          objective, profile, model=None, keeping_rate=0.2, seed=888, time_domain='source_day_domain'):
    if method not in ('local_search', 'maskco', 'maskco_initial'):
        raise ValueError('Unknown search method')
    if type(iterations) is not int or iterations < 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid search limit')
    if not 0 <= keeping_rate <= 1:
        raise ValueError('Invalid keeping rate')
    if method != 'local_search' and model is None:
        raise ValueError('Actual MaskCO model is required')
    started = time.perf_counter()
    deadline = started + seconds
    route, waits = list(initial_route), list(initial_waits)
    report = evaluate(instance, route, waits, objective, profile=profile, time_domain=time_domain)
    rng = np.random.default_rng(seed)
    records, evaluations, forwards = [], 1, 0
    failures = []
    if not report['feasible']:
        failures.append({'stage': 'initial_evaluation', 'route': route.copy(),
                         'waits_s': waits.copy(), 'errors': list(report['errors'])})
    for iteration in range(iterations):
        if time.perf_counter() >= deadline:
            break
        accepted = False
        if method == 'local_search':
            proposals = _classical_candidates(instance, route, waits, report)
        else:
            from rrp_reconstruct import reconstruct
            edges = list(zip(route, route[1:] + route[:1]))
            indices = rng.choice(len(edges), int(math.floor(keeping_rate * len(edges))), replace=False)
            kept = {edges[int(i)] for i in indices}
            candidate = reconstruct(model, instance, route, kept, waits, objective=objective, profile=profile, time_domain=time_domain)
            forwards += 1
            proposals = [(candidate['route'], candidate['waits_s'], 'masked_reconstruction')]
        attempts = 0
        for candidate_route, candidate_waits, operator in proposals:
            if time.perf_counter() >= deadline:
                break
            checked = evaluate(instance, candidate_route, candidate_waits, objective, profile=profile, time_domain=time_domain)
            evaluations += 1
            attempts += 1
            if not checked['feasible']:
                failures.append({'stage': operator, 'iteration': iteration,
                                 'route': list(candidate_route), 'waits_s': list(candidate_waits),
                                 'errors': list(checked['errors'])})
            if time.perf_counter() < deadline and _rank(checked) < _rank(report):
                route, waits, report = list(candidate_route), list(candidate_waits), checked
                accepted = True
                if method == 'local_search':
                    break  # First improvement, restart from the new incumbent.
        records.append({'iteration': iteration, 'attempts': attempts, 'accepted': accepted,
                        'best_cost_eur_year': _cost(report)})
    return {'method': method, 'route': route, 'waits_s': waits, 'report': report,
            'iterations': records, 'wall_s': time.perf_counter() - started,
            'failure_log': failures,
            'requested_search_seconds': seconds, 'actual_evaluations': evaluations,
            'actual_model_forwards': forwards, 'keeping_rate': keeping_rate,
            'seed': seed, 'initialization': 'caller_supplied_no_reference_lookup',
            'budget_scope': 'search_including_candidate_preparation_model_and_independent_acceptance',
            'author_LAHC_reproduced': False, 'scientific_gain_established': False}
