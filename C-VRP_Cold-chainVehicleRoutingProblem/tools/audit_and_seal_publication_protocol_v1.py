"""Independent LP, accounting, source and scoring-boundary audit before seal.

Only original routes/customers are used. No formal solver comparison or model
training is run. The original preflight files and numerical configuration stay
unchanged; the seal covers an additional reviewed evaluation/analysis contract.
"""
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import scipy
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from project_paths import EXTENSION_ROOT
sys.path.insert(0, str(EXTENSION_ROOT/'comparisons/static_coldchain/adapters'))
from publication_model_v1_strict import CONFIG, PublicationModel, load_config, load_original

ROOT = EXTENSION_ROOT
PRIOR = ROOT/'results/static_delivery/publication_protocol_v1_preflight_20261021'
OUT = ROOT/'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021'
PLAN = ROOT/'configs/static_delivery/publication_experiment_freeze_v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def independent_schedule_lp(ins, c, route, enforce_quality=True):
    """Minimize money loss over ALL dispatch/service/return times of fixed order.

    Variables: departure, each service-start, return. No canonical scheduling
    code or evaluator's cached times are used to construct this linear program.
    """
    rows, n = ins['nodes'], len(route)
    if sum(rows[i][3] for i in route) > c['routing']['capacity_kg']:
        return dict(status=2, message='capacity infeasible before scheduling LP')
    q, price = c['quality'], c['money']['reference_value_U_per_kg']
    p = q['profiles'][q['main_profile']]
    rate = 60*p['k_ref_per_s']*math.exp(-p['Ea_J_per_mol']/q['R_J_per_mol_K']*(1/q['T_cabin_K']-1/q['T_ref_K']))
    cold = c['money']['refrigeration']['cost_U_per_vehicle_min']
    weights = np.array([rows[i][3]*price*rate for i in route])
    objective = np.r_[-weights.sum()-cold, weights, cold]
    A, b = [], []
    previous, distance = 0, 0.0
    for pos, i in enumerate(route, 1):
        arc = math.hypot(rows[previous][1]-rows[i][1], rows[previous][2]-rows[i][2])
        distance += arc
        row = np.zeros(n+2)
        row[pos-1], row[pos] = 1, -1
        A.append(row)
        b.append(-arc/c['routing']['speed_km_per_min']-(rows[previous][6] if previous else 0))
        if enforce_quality:
            row = np.zeros(n+2)
            row[pos], row[0] = 1, -1
            A.append(row)
            b.append((1-q['minimum_retention'])/rate-rows[i][6])
        previous = i
    arc = math.hypot(rows[previous][1]-rows[0][1], rows[previous][2]-rows[0][2])
    distance += arc
    row = np.zeros(n+2)
    row[n], row[n+1] = 1, -1
    A.append(row)
    b.append(-rows[previous][6]-arc/c['routing']['speed_km_per_min'])
    bounds = [(rows[0][4], rows[0][5])] + [(rows[i][4], rows[i][5]) for i in route] + [(rows[0][4], rows[0][5])]
    result = linprog(objective, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='highs')
    if result.status != 0:
        return dict(status=int(result.status), message=result.message)
    constant = distance*c['money']['distance_U_per_km']+sum(weights[j]*rows[i][6] for j, i in enumerate(route))
    nominal = price*sum(rows[i][3] for i in route)
    return dict(status=0, net_benefit=float(nominal-result.fun-constant),
                lp_dispatch=float(result.x[0]), lp_return=float(result.x[-1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing_lock = OUT/'FREEZE_LOCK.json'
    if existing_lock.exists():
        lock = json.loads(existing_lock.read_text(encoding='utf-8'))
        assert (OUT/'FREEZE_LOCK.sha256').read_text(encoding='ascii').split()[0] == sha(existing_lock)
        for entry in lock['files']:
            path = (ROOT/entry['path']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            assert sha(path) == entry['sha256'], 'Frozen artifact changed; create a new version, do not reseal.'
        print(json.dumps(dict(status='EXISTING_SCIENTIFIC_SEAL_VALID',
                              sealed_files=len(lock['files']), lock_sha256=sha(existing_lock))))
        return
    plan, config = json.loads(PLAN.read_text(encoding='utf-8')), load_config()
    assert sha(CONFIG) == plan['base_model_sha256']
    source_paths = [
        (ROOT/'docs/static_delivery_research_20261020/sources/quality_driven_2021.pdf', config['sources']['quality']['pdf_sha256']),
        (ROOT/'docs/static_delivery_research_20261021/sources/joint_distribution_RCR_2020.pdf', config['sources']['relative_value']['pdf_sha256']),
        (ROOT/'data/classic_static_sources_20261020/VRPTWQ_Solomon/README.md', config['sources']['dataset']['README_sha256'])]
    assert all(sha(path) == expected for path, expected in source_paths)
    old_pins = json.loads((PRIOR/'source_code_sha256.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/path) == expected for path, expected in old_pins.items())
    manifest = json.loads((PRIOR/'dataset_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/r['path']) == r['sha256'] for r in manifest['files'])
    witnesses = json.loads((PRIOR/'feasibility_witnesses.json').read_text(encoding='utf-8'))
    folds = json.loads((PRIOR/'geometry_folds.json').read_text(encoding='utf-8'))['folds']
    for f in folds:
        role_groups = [set(next(r['geometry_sha256'] for r in manifest['files'] if r['name']==case)
                           for case in f[role]) for role in ['train','validation','test']]
        assert all(not role_groups[i] & role_groups[j] for i in range(3) for j in range(i))

    lp_rows, gaps, economic_rows, accounting_checks = [], [], [], 0
    for witness in witnesses:
        ins = load_original(witness['case'], config)
        model = PublicationModel(ins, config)
        scores = model.solution([r['route'] for r in witness['feasible_solution']])
        assert all(math.isclose(r.net_benefit, old['net_benefit'], abs_tol=1e-8)
                   for r, old in zip(scores, witness['feasible_solution']))
        # Exact algebraic collapse under v=1, fixed service and isothermal costs.
        D = sum(r.distance for r in scores)
        W = sum(stop['wait'] for r in scores for stop in r.stops)
        S = sum(ins['nodes'][i][6] for r in scores for i in r.route)
        age_weight = sum(ins['nodes'][stop['customer']][3]*(stop['completion']-r.departure)
                         for r in scores for stop in r.stops)
        nominal = model.price*sum(ins['nodes'][i][3] for i in range(1,101))
        loss_cost = nominal-sum(r.net_benefit for r in scores)
        reduced = (1+model.cold)*D+model.cold*W+model.cold*S+model.price*model.rate*age_weight
        assert math.isclose(S, 9000) and math.isclose(loss_cost, reduced, abs_tol=1e-6)
        accounting_checks += 1
        longest = max(scores, key=lambda r: (len(r.route), r.route))
        candidates = [longest.route, tuple(witness['best_pair']['route'])]
        for route in candidates:
            actual = model.route(route)
            independent = independent_schedule_lp(ins, config, route)
            assert independent['status'] == 0, (ins['name'], route, independent)
            gap = abs(actual.net_benefit-independent['net_benefit'])
            assert gap <= 1e-6, (ins['name'], gap)
            gaps.append(gap)
            lp_rows.append(dict(case=ins['name'], route=list(route), kind='feasible_schedule_optimum',
                                net_gap_U=gap, **independent))
        if witness['explicit_constraint_witness']:
            route = witness['explicit_constraint_witness']['route']
            assert not model.route(route).feasible
            independent = independent_schedule_lp(ins, config, route)
            assert independent['status'] == 2, (ins['name'], independent)
            lp_rows.append(dict(case=ins['name'], route=route, kind='quality_infeasibility', **independent))
        for multiplier in plan['conditions']['economic_weight_sensitivity']['reference_value_multipliers']:
            adjusted = json.loads(json.dumps(config))
            adjusted['money']['reference_value_U_per_kg'] *= multiplier
            other = PublicationModel(ins, adjusted)
            reference = independent_schedule_lp(ins, adjusted, longest.route)
            actual = other.route(longest.route)
            assert reference['status'] == 0 and abs(reference['net_benefit']-actual.net_benefit) <= 1e-6
            economic_rows.append(dict(case=ins['name'], multiplier=multiplier,
                P_U_per_kg=other.price, fixed_route_schedule_check_only=True,
                LP_gap_U=abs(reference['net_benefit']-actual.net_benefit)))
    for excluded in manifest['exclusions']:
        ins = load_original(excluded['case'], config)
        route = [excluded['customers'][0]['customer']]
        reference = independent_schedule_lp(ins, config, route, enforce_quality=False)
        assert reference['status'] == 2
        lp_rows.append(dict(case=ins['name'], route=route, kind='source_routing_infeasibility', **reference))

    ins = load_original('c105', config)
    strict = PublicationModel(ins, config)
    invalid_inputs = [float('nan'), float('inf'), -float('inf'), True, 'nan']
    assert all(not strict.route([1], departure=x).feasible for x in invalid_inputs)
    all_singletons = [[i] for i in range(1,101)]
    for routes in [all_singletons[:-1], all_singletons+[[1]]]:
        try:
            strict.solution(routes)
        except ValueError:
            pass
        else:
            raise AssertionError('missing/duplicate customer accepted')
    try:
        strict.solution(all_singletons, enforce_quality=False)
    except ValueError:
        pass
    else:
        raise AssertionError('formal solution accepted disabled quality checks')
    assert not strict.route(list(range(1,101))).feasible
    assert 2/2**4 == plan['inference']['minimum_two_sided_exact_sign_p_at_four_groups']
    assert len(plan['budget_and_repetitions']['independent_repetition_seeds']) == 10

    write(OUT/'independent_lp_checks.json', lp_rows)
    write(OUT/'economic_weight_schedule_checks.json', economic_rows)
    summary = dict(status='PASS_FOR_MODEL_DATA_AND_ANALYSIS_FREEZE', date='2026-10-21',
        model_numerical_parameters_changed=False, original_preflight_pins_unchanged=True,
        source_files_rehashed=56, included_cases=47, geometry_folds_disjoint=4,
        source_papers_rehashed=2, author_README_rehashed=True,
        independently_optimized_feasible_route_LPs=len(gaps), maximum_LP_gap_U=max(gaps),
        quality_infeasibility_LPs=sum(r['kind']=='quality_infeasibility' for r in lp_rows),
        original_routing_infeasibility_LPs=sum(r['kind']=='source_routing_infeasibility' for r in lp_rows),
        economic_weight_fixed_route_LPs=len(economic_rows), reduced_objective_identity_cases=accounting_checks,
        rejected_nonfinite_or_invalid_departures=len(invalid_inputs),
        missing_duplicate_quality_disabled_capacity_checks_passed=True,
        economic_sensitivity_full_solver_comparisons_run=False, method_training_or_gain_claimed=False,
        implementation_gate_still_required=plan['execution_gate']['required_before_first_formal_test'],
        runtime=dict(python=platform.python_version(), scipy=scipy.__version__, numpy=np.__version__))
    write(OUT/'audit_summary.json', summary)
    paths = [CONFIG, PLAN, Path(__file__),
        ROOT/'comparisons/static_coldchain/adapters/publication_model_v1.py',
        ROOT/'comparisons/static_coldchain/adapters/publication_model_v1_strict.py',
        ROOT/'tools/preflight_publication_model_v1.py',
        PRIOR/'dataset_manifest.json', PRIOR/'geometry_folds.json', PRIOR/'source_code_sha256.json',
        ROOT/'docs/static_delivery_research_20261021/科研冻结审查_v1.md',
        OUT/'audit_summary.json', OUT/'independent_lp_checks.json', OUT/'economic_weight_schedule_checks.json']
    paths += [ROOT/r['path'] for r in manifest['files']]
    paths += [path for path, expected in source_paths]
    lock = dict(freeze_id=plan['freeze_id'], date='2026-10-21',
        scope='model/data/split/evaluator/review/analysis contract; not yet trained method manifests',
        files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in paths])
    write(OUT/'FREEZE_LOCK.json', lock)
    (OUT/'FREEZE_LOCK.sha256').write_text(sha(OUT/'FREEZE_LOCK.json')+'  FREEZE_LOCK.json\n', encoding='ascii')
    for target in re.findall(r'\]\(([^)]+)\)', paths[9].read_text(encoding='utf-8')):
        if '://' not in target:
            assert (paths[9].parent/target).exists(), target
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
