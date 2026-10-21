"""Frozen literature composite. Original instances only; analytic route scheduling."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = next(p/'scripts' for p in Path(__file__).resolve().parents
               if (p/'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT

CONFIG = EXTENSION_ROOT/'configs/static_delivery/publication_model_v1.json'
EPS = 1e-7


def load_config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def load_original(name, config):
    base = EXTENSION_ROOT/config['dataset']['root']
    paths = [p for p in base.glob('*.txt') if p.stem.lower() == name.lower()]
    if len(paths) != 1:
        raise ValueError('one existing original instance required')
    path = paths[0]
    nodes = [list(map(float, s.split())) for s in path.read_text().splitlines() if s.strip()]
    assert len(nodes) == 101 and all(len(r) == 7 for r in nodes)
    assert [r[0] for r in nodes] == list(range(101))
    assert all(math.isfinite(x) for r in nodes for x in r)
    assert all(r[3] >= 0 and r[4] <= r[5] and r[6] >= 0 for r in nodes)
    distances = [[math.dist(a[1:3], b[1:3]) for b in nodes] for a in nodes]
    geometry = hashlib.sha256(json.dumps([r[1:3] for r in nodes], separators=(',', ':')).encode()).hexdigest()
    return dict(name=path.stem.lower(), path=str(path.relative_to(EXTENSION_ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), nodes=nodes,
                distance=distances, geometry_sha256=geometry)


@dataclass
class RouteResult:
    route: tuple
    feasible: bool
    reason: str
    departure: float = 0.0
    returned: float = 0.0
    distance: float = 0.0
    revenue: float = 0.0
    refrigeration_cost: float = 0.0
    quality_loss_value: float = 0.0
    net_benefit: float = 0.0
    min_quality: float = 1.0
    stops: tuple = ()


class PublicationModel:
    def __init__(self, instance, config, profile=None, minimum_quality=None):
        self.ins, self.c = instance, config
        q = config['quality']
        self.profile = profile or q['main_profile']
        p = q['profiles'][self.profile]
        self.rate = 60*p['k_ref_per_s']*math.exp(-p['Ea_J_per_mol']/q['R_J_per_mol_K']
                                               *(1/q['T_cabin_K']-1/q['T_ref_K']))
        self.minimum = q['minimum_retention'] if minimum_quality is None else minimum_quality
        self.price = config['money']['reference_value_U_per_kg']
        self.cold = config['money']['refrigeration']['cost_U_per_vehicle_min']
        self.max_age = (1-self.minimum)/self.rate

    def route(self, route, enforce_quality=True, departure=None):
        route = tuple(route)
        def fail(reason):
            return RouteResult(route, False, reason)
        if not route or len(set(route)) != len(route) or any(type(i) is not int or i < 1 or i > 100 for i in route):
            return fail('invalid_or_repeated_customer')
        rows, d = self.ins['nodes'], self.ins['distance']
        if sum(rows[i][3] for i in route) > self.c['routing']['capacity_kg']+EPS:
            return fail('capacity')
        # With no waiting, prefix is elapsed time to each service start.
        # Every upper window bounds departure by b_i-prefix_i.
        prefix, previous, latest = 0.0, 0, rows[0][5]
        for i in route:
            prefix += d[previous][i]/self.c['routing']['speed_km_per_min']
            latest = min(latest, rows[i][5]-prefix)
            prefix += rows[i][6]
            previous = i
        latest = min(latest, rows[0][5]-prefix-d[previous][0]/self.c['routing']['speed_km_per_min'])
        departure = latest if departure is None else departure
        if departure < rows[0][4]-EPS or departure > latest+EPS:
            return fail('departure_bounds')
        time, previous, distance, revenue, stops = departure, 0, 0.0, 0.0, []
        min_quality = 1.0
        for i in route:
            arc = d[previous][i]
            distance += arc
            arrival = time+arc/self.c['routing']['speed_km_per_min']
            start = max(arrival, rows[i][4])
            if start > rows[i][5]+EPS:
                return fail('customer_window')
            time = start+rows[i][6]
            quality = 1-self.rate*(time-departure)
            if enforce_quality and quality < self.minimum-EPS:
                return fail('quality')
            min_quality = min(min_quality, quality)
            revenue += self.price*rows[i][3]*quality
            stops.append(dict(customer=i, arrival=arrival, start=start, completion=time,
                              quality=quality, wait=start-arrival))
            previous = i
        distance += d[previous][0]
        time += d[previous][0]/self.c['routing']['speed_km_per_min']
        if time > rows[0][5]+EPS:
            return fail('depot_return')
        cold = self.cold*(time-departure)
        nominal = self.price*sum(rows[i][3] for i in route)
        return RouteResult(route, True, 'ok', departure, time, distance, revenue, cold,
                           nominal-revenue, revenue-distance*self.c['money']['distance_U_per_km']-cold,
                           min_quality, tuple(stops))

    def solution(self, routes, enforce_quality=True):
        if not routes or len(routes) > 100:
            raise ValueError('nonempty routes with K <= n required')
        coverage = [i for r in routes for i in r]
        if sorted(coverage) != list(range(1, 101)):
            raise ValueError('each original customer must appear exactly once')
        results = [self.route(r, enforce_quality=enforce_quality) for r in routes]
        if not all(r.feasible for r in results):
            raise ValueError('infeasible route')
        return results
