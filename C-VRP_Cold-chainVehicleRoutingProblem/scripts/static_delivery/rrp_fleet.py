"""Homogeneous RRP fleet and Bellman Split for a fixed order and node waits.

Prins (2004), doi:10.1016/S0305-0548(03)00158-8; Vidal (2016),
doi:10.1016/j.cor.2015.11.012. We use the general DAG dynamic program,
not Vidal's linear-time optimization: these segment costs are time dependent.
"""
from __future__ import annotations

import math
import time
from types import MappingProxyType
import numpy as np
from rrp_data import RRPInstance, PRIMARY_PROFILE, source_protocol
from rrp_evaluator import evaluate


def _fleet_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError('max_vehicles must be a positive integer')


def _coverage(instance, route):
    return (len(route) == instance.nodes and list(route[:1]) == [0]
            and all(type(v) in (int, np.int32, np.int64) for v in route)
            and set(route) == set(range(instance.nodes)))


def _tour(instance, route, waits, objective, profile, time_domain):
    # A view of assigned ORIGINAL orders, not a new customer dataset. Each
    # vehicle starts with only its cargo and resets the pallet-position clock.
    ids = np.asarray(route, dtype=int)
    demands = instance.demands[ids].copy()
    distances = instance.distances[np.ix_(ids, ids)].copy()
    demands.setflags(write=False)
    distances.setflags(write=False)
    raw = dict(instance.raw)
    raw.update(clients=len(route), demands=demands.tolist(), distances=distances.tolist())
    view = RRPInstance(instance.name, MappingProxyType(raw), demands, distances,
                       instance.source_sha256, instance.source_commit)
    report = evaluate(view, list(range(len(route))), waits, objective,
                      profile=profile, time_domain=time_domain)
    report['route'] = list(route)
    report['assigned_original_nodes'] = list(route[1:])
    for leg in report['legs']:
        leg['from'], leg['to'] = int(ids[leg['from']]), int(ids[leg['to']])
    if sum(demands) > source_protocol()['pallet_capacity']:
        report['errors'].append('vehicle_capacity')
        report['feasible'] = False
    return report


def evaluate_fleet(instance, routes, waits_by_route, objective, *, max_vehicles,
                   profile=PRIMARY_PROFILE, time_domain='source_day_domain'):
    _fleet_limit(max_vehicles)
    report = {'instance': instance.name, 'source_sha256': instance.source_sha256,
              'objective': objective, 'profile': profile, 'time_domain': time_domain,
              'max_vehicles': max_vehicles, 'vehicles_used': len(routes),
              'feasible': False, 'complete': False, 'errors': [], 'tours': [], 'costs': None,
              'fleet_size_origin': 'explicit_experiment_bound_not_an_author_dataset_field',
              'fixed_vehicle_cost_eur': None,
              'qualification': 'new_homogeneous_fleet_extension_of_original_single_vehicle_RRP'}
    if not routes or len(routes) > max_vehicles:
        report['errors'].append('fleet_limit')
    structurally_valid = all(len(r) >= 2 and r[0] == 0
                             and all(type(v) in (int, np.int32, np.int64) and 0 < v < instance.nodes for v in r[1:])
                             for r in routes)
    nodes = [v for r in routes for v in r[1:]]
    if not structurally_valid or sorted(nodes) != list(range(1, instance.nodes)):
        report['errors'].append('customer_coverage')
    if len(waits_by_route) != len(routes):
        report['errors'].append('invalid_waits')
    if report['errors']:
        return report
    for route, waits in zip(routes, waits_by_route):
        checked = _tour(instance, route, waits, objective, profile, time_domain)
        report['tours'].append(checked)
    report['complete'] = all(r['complete'] for r in report['tours'])
    if not all(r['feasible'] for r in report['tours']):
        report['errors'].append('vehicle_infeasible')
        return report
    report['costs'] = {k: sum(r['costs'][k] for r in report['tours']) for k in report['tours'][0]['costs']}
    for field in ('distance_km', 'driving_s', 'service_total_s', 'waiting_total_s'):
        report[field] = sum(r[field] for r in report['tours'])
    report['makespan_s'] = max(r['duration_s'] for r in report['tours'])
    report['feasible'] = True
    return report


def split_giant_tour(instance, route, waits_s, max_vehicles, objective, *,
                     profile=PRIMARY_PROFILE, time_domain='source_day_domain', deadline=None):
    """Minimize over all contiguous cuts, at most K trucks. O(K*n²) DP.

    Each cached segment independently starts at the published depot start.
    This optimizes neither the giant order nor continuous node waits globally.
    Segment evaluation is additional work, beyond the stated DP complexity.
    """
    _fleet_limit(max_vehicles)
    result = {'routes': [], 'waits_s_by_route': [], 'actual_tour_evaluations': 0,
              'optimality_scope': 'cuts_for_fixed_order_and_waits_only',
              'report': {'feasible': False, 'complete': False, 'errors': [], 'costs': None}}
    if not _coverage(instance, route):
        result['report']['errors'] = ['customer_coverage']
        return result
    if (len(waits_s) != len(route) or any(not isinstance(v, (int, float, np.number))
            or not math.isfinite(v) or v < 0 for v in waits_s) or waits_s[0] != 0):
        result['report']['errors'] = ['invalid_waits']
        return result
    def expired():
        return deadline is not None and time.perf_counter() >= deadline
    customers = list(route[1:])
    node_wait = dict(zip(route, waits_s))
    n, kmax = len(customers), min(max_vehicles, len(customers))
    segments = {}
    for i in range(n):
        for j in range(i + 1, n + 1):
            if expired():
                result['report']['errors'] = ['search_budget_exhausted']
                return result
            segment = [0, *customers[i:j]]
            checked = _tour(instance, segment, [node_wait[v] for v in segment], objective, profile, time_domain)
            result['actual_tour_evaluations'] += 1
            if checked['feasible']:
                segments[i, j] = checked['costs']['total_eur_year']
    dp = {(0, 0): 0.0}
    previous = {}
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            if expired():
                result['report']['errors'] = ['search_budget_exhausted']
                return result
            candidates = [(dp[k - 1, i] + segments[i, j], i) for i in range(k - 1, j)
                          if (k - 1, i) in dp and (i, j) in segments]
            if candidates:
                dp[k, j], previous[k, j] = min(candidates)
    complete = [(dp[k, n], k) for k in range(1, kmax + 1) if (k, n) in dp]
    if not complete:
        result['report']['errors'] = ['no_feasible_partition']
        return result
    _, k = min(complete)
    j, routes = n, []
    while k:
        i = previous[k, j]
        routes.append([0, *customers[i:j]])
        j, k = i, k - 1
    routes.reverse()
    waits = [[node_wait[v] for v in r] for r in routes]
    if expired():
        result['report']['errors'] = ['search_budget_exhausted']
        return result
    checked = evaluate_fleet(instance, routes, waits, objective, max_vehicles=max_vehicles,
                             profile=profile, time_domain=time_domain)
    result['actual_tour_evaluations'] += len(routes)
    if expired():
        result['report']['errors'] = ['search_budget_exhausted']
        return result
    result.update(routes=routes, waits_s_by_route=waits, report=checked)
    return result
