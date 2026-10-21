"""Multi-route frozen-profit task, split gates, and exact feasibility oracle."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import EXTENSION_ROOT, configure_imports
configure_imports(('static_profit',))
sys.path.insert(0, str(EXTENSION_ROOT/'comparisons/static_coldchain/adapters'))
sys.path.insert(0, str(EXTENSION_ROOT/'tools'))
from publication_model_v1_strict import CONFIG, PublicationModel, load_config, load_original
from preflight_publication_model_v1 import independent_audit, nearest_neighbour_witness

FREEZE = EXTENSION_ROOT/'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json'
FOLDS = EXTENSION_ROOT/'results/static_delivery/publication_protocol_v1_preflight_20261021/geometry_folds.json'
SCHEMA = 'static-profit-maskco-features-v0-16channels-partial-only'
VARIANTS = {
    'maskco_base': (False, False),
    'quality_only': (True, False),
    'directed_only': (False, True),
    'quality_directed': (True, True),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_research_freeze():
    lock = json.loads(FREEZE.read_text(encoding='utf-8'))
    expected = FREEZE.with_suffix('.sha256').read_text(encoding='ascii').split()[0]
    if digest(FREEZE) != expected:
        raise ValueError('Scientific lock hash mismatch')
    for row in lock['files']:
        path = (EXTENSION_ROOT/row['path']).resolve()
        if not path.is_relative_to(EXTENSION_ROOT.resolve()) or digest(path) != row['sha256']:
            raise ValueError('Frozen research artifact changed: '+row['path'])
    return dict(freeze_id=lock['freeze_id'], sha256=expected, files=len(lock['files']))


def fold_definition(fold):
    folds = json.loads(FOLDS.read_text(encoding='utf-8'))['folds']
    if type(fold) is not int or not 0 <= fold < len(folds):
        raise ValueError('explicit outer fold 0..3 required')
    return folds[fold]


class ProfitTask:
    def __init__(self, case, *, fold, role, profile='main_Product1', minimum_quality=0.8, value_multiplier=1):
        if role not in ('train', 'validation'):
            raise ValueError('Formal outer-test execution disabled until method-manifest gate is met')
        split = fold_definition(fold)
        if case.lower() not in split[role]:
            raise ValueError('Instance is not in the declared fold/role')
        if role == 'train' and value_multiplier != 1:
            raise ValueError('Frozen protocol trains at nominal value only')
        if value_multiplier not in (0.5, 1, 2) or minimum_quality not in (0.8, 0.85):
            raise ValueError('undeclared economic/threshold condition')
        self.c = load_config()
        if profile not in self.c['quality']['profiles']:
            raise ValueError('undeclared kinetic profile')
        self.c['money']['reference_value_U_per_kg'] *= value_multiplier
        self.model = PublicationModel(load_original(case, self.c), self.c, profile, minimum_quality)
        self.ins, self.fold, self.role = self.model.ins, fold, role
        self.condition = dict(profile=profile, minimum_quality=minimum_quality, value_multiplier=value_multiplier)
        self.route = lru_cache(maxsize=30000)(self.model.route)

    def evaluate(self, routes, *, audit=False):
        routes = tuple(tuple(r) for r in routes)
        if not routes or len(routes) > 100 or any(not r for r in routes):
            raise ValueError('invalid route count/empty route')
        coverage = [i for r in routes for i in r]
        if any(type(i) is not int for i in coverage) or sorted(coverage) != list(range(1,101)):
            raise ValueError('mandatory source customers must occur exactly once')
        scores = [self.route(r) for r in routes]
        if not all(r.feasible for r in scores):
            raise ValueError('candidate violates frozen hard constraints')
        benefit = math.fsum(r.net_benefit for r in scores)
        if audit:
            independently = independent_audit(self.model, scores, True)
            if not math.isclose(benefit, independently, abs_tol=1e-7):
                raise ValueError('independent net-benefit audit mismatch')
        nominal = self.model.price*sum(row[3] for row in self.ins['nodes'])
        return dict(net_benefit_U=benefit, equivalent_loss_cost_U=nominal-benefit,
                    quality_loss_value_U=math.fsum(r.quality_loss_value for r in scores),
                    distance_km=math.fsum(r.distance for r in scores),
                    refrigeration_cost_U=math.fsum(r.refrigeration_cost for r in scores),
                    used_vehicles=len(routes), minimum_quality=min(r.min_quality for r in scores),
                    independently_audited=audit)

    def initial(self):
        scores = nearest_neighbour_witness(self.model, True)
        routes = tuple(r.route for r in scores)
        self.evaluate(routes, audit=True)
        return routes

    def identity(self):
        return dict(case=self.ins['name'], source_sha256=self.ins['sha256'],
                    geometry_sha256=self.ins['geometry_sha256'], fold=self.fold, role=self.role,
                    condition=self.condition, model_sha256=digest(CONFIG), feature_schema=SCHEMA)


def route_edges(routes):
    return [(a,b) for r in routes for a,b in zip((0,)+tuple(r), tuple(r)+(0,))]


def partial_batch(task, routes, removed, variant):
    """Derived features use ONLY the visible partial solution or singleton bounds.

    Teacher service times, full-solution ages and removed-node teacher positions
    are never used as inputs. The full routes are used only for supervision.
    """
    quality_context, directed = VARIANTS[variant]
    removed = set(removed)
    if not removed or not removed <= set(range(1,101)):
        raise ValueError('a nonempty subset of original customers must be masked')
    task.evaluate(routes)
    rows, model = task.ins['nodes'], task.model
    n, horizon = len(rows), 1440.0
    features = np.zeros((n,16), np.float32)
    for i,row in enumerate(rows):
        features[i,:11] = [(row[1]-rows[0][1])/horizon, (row[2]-rows[0][2])/horizon,
            row[3]/20000, row[4]/horizon, row[5]/horizon, row[6]/horizon,
            model.rate*horizon, model.minimum,
            math.log1p(model.price*20000/horizon), model.cold,
            float(i == 0 or i not in removed)]
    partial = tuple(tuple(i for i in r if i not in removed) for r in routes)
    partial = tuple(r for r in partial if r)
    if quality_context:
        # Removed-node reference uses its own singleton, not its hidden teacher age.
        states = {i: task.route((i,)) for i in removed}
        for r in partial:
            result = task.route(r)
            if not result.feasible:
                raise ValueError('Deleting customers produced an infeasible partial route')
            states.update({i: result for i in r})
        for i,result in states.items():
            stop = next(s for s in result.stops if s['customer']==i)
            utilization = sum(rows[j][3] for j in result.route)/20000
            features[i,11:] = [(stop['completion']-result.departure)/horizon,
                (stop['quality']-model.minimum)/(1-model.minimum),
                math.log1p(model.price*rows[i][3]*model.rate/(1+model.cold)),
                (rows[i][5]-stop['start'])/horizon, utilization]
    targets, adjacency = np.zeros((n,n), np.float32), np.zeros((n,n), np.float32)
    for a,b in route_edges(routes):
        targets[a,b] += 1
        kept = a not in removed and b not in removed
        if kept:
            adjacency[a,b] += 1
        if not directed:
            targets[b,a] += 1
            if kept:
                adjacency[b,a] += 1
    denominator = targets.sum(axis=1, keepdims=True)
    targets = targets/np.maximum(denominator,1)
    row_weights = np.ones(n,np.float32)
    row_weights[0] = 0  # Same customer-row training convention as upstream CVRP.
    timestep = adjacency[1:].sum()/(100*(1 if directed else 2))
    return dict(features=features[None], adjacency=adjacency[None],
                timestep=np.array([timestep],np.float32), targets=targets[None],
                row_weights=row_weights[None]), partial
