"""Scenario evaluator for ORIGINAL author instances. No customer generation.

The new TA-to-price hypothesis and ideal refrigeration boundary are explicit in
the config. This is neither field profit nor reproduction of the VRPTWQ model.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

SCRIPTS = next(p/'scripts' for p in Path(__file__).resolve().parents if (p/'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT

CONFIG_PATH = EXTENSION_ROOT/'configs/static_delivery/net_benefit_scenario_v1.json'


def config():
    c = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    h = c['refrigeration_cost_scenario']
    power = h['conductivity_W_per_m_C'] * h['surface_m2'] / h['wall_m'] * (h['ambient_C']-c['temperature']['product_and_cabin_C'])
    assert math.isclose(power, h['thermal_heat_leak_W_derived'])
    assert math.isclose(power*h['coefficient_EUR_per_W_min'], h['cost_EUR_per_vehicle_min_derived'])
    assert 0 <= c['tomato']['rho_center'] <= 1
    return c


def load_case(name, c):
    base = EXTENSION_ROOT/c['dataset']['root']
    matches = [p for p in base.glob('*.txt') if p.stem.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError('case must name one existing original file')
    path = matches[0]
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 101 or any(len(r) != 7 or not all(math.isfinite(x) for x in r) for r in rows):
        raise ValueError('not original100-customer format')
    if [r[0] for r in rows] != list(range(101)) or any(r[3] < 0 or r[4] > r[5] or r[6] < 0 for r in rows):
        raise ValueError('invalid original node data')
    distances = [[math.dist(a[1:3], b[1:3]) for b in rows] for a in rows]
    return {'name': path.stem.lower(), 'path': str(path.relative_to(EXTENSION_ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'nodes': rows, 'distance': distances}


def structural_audit(ins, c):
    """Necessary singleton bounds; independent of K and any search heuristic."""
    rows, d = ins['nodes'], ins['distance']
    factor = 60/c['routing']['speed_km_per_hour']
    impossible = []
    for i, row in enumerate(rows[1:], 1):
        earliest = max(rows[0][4]+d[0][i]*factor, row[4])
        returned = earliest+row[6]+d[i][0]*factor
        reasons = []
        if earliest > row[5]+1e-9:
            reasons.append('earliest_singleton_service_after_latest_start')
        if returned > rows[0][5]+1e-9:
            reasons.append('earliest_singleton_return_after_depot_close')
        if row[3] > c['routing']['capacity_kg']:
            reasons.append('single_customer_exceeds_capacity')
        if reasons:
            impossible.append({'customer': i, 'reasons': reasons, 'earliest_service_min': earliest, 'earliest_return_min': returned, 'depot_close_min': rows[0][5], 'proof': 'Euclidean direct travel is a lower bound on any path; waiting/services elsewhere cannot make it earlier.'})
    return {'case': ins['name'], 'source_sha256': ins['sha256'], 'passes_necessary_bounds': not impossible, 'is_sufficient_feasibility_proof': False, 'impossible_customers': impossible}


class RouteModel:
    def __init__(self, ins, c, rho, max_vehicles):
        if not 0 <= rho <= 1 or not isinstance(max_vehicles, int) or max_vehicles < 1:
            raise ValueError('rho in[0,1], explicit positive integer fleet bound required')
        self.ins, self.c, self.rho, self.max_vehicles = ins, c, rho, max_vehicles
        self.n = len(ins['nodes'])-1
        temp = c['temperature']['product_and_cabin_C']
        t = c['tomato']
        self.k_min = t['A_per_day'] * math.exp(-t['Ea_J_per_mol']/(t['R_J_per_mol_K']*(temp+273.15))) / 1440
        self.discount_coeff = t['fresh_price_EUR_per_kg']*(1-rho)
        self.refrigeration_rate = c['refrigeration_cost_scenario']['cost_EUR_per_vehicle_min_derived']
        self.constant_margin = sum(r[3] for r in ins['nodes'])*(t['fresh_price_EUR_per_kg']-t['purchase_EUR_per_kg'])
        self.route = lru_cache(maxsize=60000)(self._route)

    def _route(self, route):
        if not route:
            return 0.
        nodes, d = self.ins['nodes'], self.ins['distance']
        if sum(nodes[i][3] for i in route) > self.c['routing']['capacity_kg']+1e-9:
            return math.inf
        departure = nodes[0][4]
        time, previous, distance, discount = departure, 0, 0., 0.
        travel_factor = 60/self.c['routing']['speed_km_per_hour']
        for i in route:
            distance += d[previous][i]
            time = max(time+d[previous][i]*travel_factor, nodes[i][4])
            if time > nodes[i][5]+1e-8:
                return math.inf
            time += nodes[i][6]
            discount += nodes[i][3]*self.discount_coeff*(-math.expm1(-self.k_min*(time-departure)))
            previous = i
        distance += d[previous][0]
        time += d[previous][0]*travel_factor
        if time > nodes[0][5]+1e-8:
            return math.inf
        return distance*self.c['routing']['distance_EUR_per_km']+self.refrigeration_rate*(time-departure)+discount

    def objective(self, routes):
        """Negative net benefit because upstream ALNS minimizes; no sign trick."""
        if len(routes) > self.max_vehicles or any(not r for r in routes):
            return math.inf
        seen = [i for r in routes for i in r]
        if sorted(seen) != list(range(1, self.n+1)):
            return math.inf
        return math.fsum(self.route(tuple(r)) for r in routes)-self.constant_margin
