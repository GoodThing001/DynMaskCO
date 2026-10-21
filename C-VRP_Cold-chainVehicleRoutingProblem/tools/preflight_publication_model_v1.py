"""Freeze dataset identity/folds and check the numerical protocol on original cases.

The nearest-neighbour routes are feasibility/constraint witnesses, not method
comparison results. No generated nodes, formal training, or solver win claims.
"""
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from project_paths import EXTENSION_ROOT
sys.path.insert(0, str(EXTENSION_ROOT/'comparisons/static_coldchain/adapters'))
from publication_model_v1 import CONFIG, PublicationModel, load_config, load_original

OUT = EXTENSION_ROOT/'results/static_delivery/publication_protocol_v1_preflight_20261021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def independent_audit(model, results, enforce_quality):
    """Rebuild directed timelines and latest departures by backward scheduling."""
    c, rows = model.c, model.ins['nodes']
    all_seen, benefit = [], 0.0
    q = c['quality']
    p = q['profiles'][model.profile]
    rate = p['k_ref_per_s']*60*math.exp(p['Ea_J_per_mol']/q['R_J_per_mol_K']
                                      *(1/q['T_ref_K']-1/q['T_cabin_K']))
    for r in results:
        assert r.feasible
        all_seen.extend(r.route)
        assert sum(rows[i][3] for i in r.route) <= c['routing']['capacity_kg']+1e-7
        # Backward time-window recursion, separate from evaluator prefix bounds.
        bound, successor = rows[0][5], 0
        for i in reversed(r.route):
            travel = math.hypot(rows[i][1]-rows[successor][1],
                                rows[i][2]-rows[successor][2])/c['routing']['speed_km_per_min']
            bound = min(rows[i][5], bound-travel-rows[i][6])
            assert bound >= rows[i][4]-1e-7
            successor = i
        departure = bound-math.hypot(rows[0][1]-rows[successor][1],
                                     rows[0][2]-rows[successor][2])/c['routing']['speed_km_per_min']
        assert math.isclose(departure, r.departure, abs_tol=1e-7)
        assert departure >= rows[0][4]-1e-7
        clock, previous, distance, value, minimum = departure, 0, 0.0, 0.0, 1.0
        for index, i in enumerate(r.route):
            arc = math.hypot(rows[i][1]-rows[previous][1], rows[i][2]-rows[previous][2])
            distance += arc
            clock = max(clock+arc/c['routing']['speed_km_per_min'], rows[i][4])
            assert clock <= rows[i][5]+1e-7
            clock += rows[i][6]
            retained = 1-rate*(clock-departure)
            if enforce_quality:
                assert retained >= model.minimum-1e-7
            minimum = min(minimum, retained)
            value += rows[i][3]*c['money']['reference_value_U_per_kg']*retained
            assert math.isclose(clock, r.stops[index]['completion'], abs_tol=1e-7)
            assert math.isclose(retained, r.stops[index]['quality'], abs_tol=1e-10)
            previous = i
        arc = math.hypot(rows[previous][1]-rows[0][1], rows[previous][2]-rows[0][2])
        distance += arc
        clock += arc/c['routing']['speed_km_per_min']
        assert clock <= rows[0][5]+1e-7
        wall = c['money']['refrigeration']
        heat = wall['conductivity_W_per_m_K']*wall['surface_m2']/wall['wall_m']*(q['T_ambient_K']-q['T_cabin_K'])
        cold = heat*wall['coefficient_U_per_W_min']*(clock-departure)
        net = value-distance*c['money']['distance_U_per_km']-cold
        assert math.isclose(net, r.net_benefit, abs_tol=1e-7)
        assert math.isclose(minimum, r.min_quality, abs_tol=1e-10)
        benefit += net
    assert sorted(all_seen) == list(range(1, 101))
    return benefit


def nearest_neighbour_witness(model, enforce_quality):
    remaining, routes = set(range(1, 101)), []
    while remaining:
        route = []
        while True:
            previous = route[-1] if route else 0
            choices = sorted(remaining, key=lambda j: (model.ins['distance'][previous][j], j))
            candidate = next((i for i in choices if model.route(route+[i], enforce_quality).feasible), None)
            if candidate is None:
                break
            route.append(candidate)
            remaining.remove(candidate)
        assert route, 'singleton certificate should guarantee constructive progress'
        routes.append(route)
    return model.solution(routes, enforce_quality)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load_config()
    assert math.isclose(c['money']['reference_value_U_per_kg']*3, 5000/1000)
    wall, q = c['money']['refrigeration'], c['quality']
    heat = wall['conductivity_W_per_m_K']*wall['surface_m2']/wall['wall_m']*(q['T_ambient_K']-q['T_cabin_K'])
    assert math.isclose(heat, wall['heat_load_W'])
    assert math.isclose(heat*wall['coefficient_U_per_W_min'], wall['cost_U_per_vehicle_min'])
    assert q['profiles'][q['main_profile']]['k_ref_per_s'] == sorted(p['k_ref_per_s'] for p in q['profiles'].values())[1]
    base = EXTENSION_ROOT/c['dataset']['root']
    source_root = base.parents[1]
    assert sha(source_root/'README.md') == c['sources']['dataset']['README_sha256']
    archive = source_root/'vrptwq_instances.zip'
    assert sha(archive) == 'fc80a921974333f7f0e66519d0c84fcf43af5fedc258dabd818fccd59761cccc'
    with zipfile.ZipFile(archive) as z:
        originals = {Path(p).stem.lower(): hashlib.sha256(z.read(p)).hexdigest()
                     for p in z.namelist() if p.lower().endswith('.txt')}
    paths = sorted(base.glob('*.txt'))
    assert len(paths) == len(originals) == 56
    manifest, excluded, included, regimes, witnesses = [], [], [], [], []
    geometry = {}
    audited_route_sets = 0
    departures_checked = 0
    for path in paths:
        ins = load_original(path.stem, c)
        assert ins['sha256'] == originals[ins['name']]
        geometry.setdefault(ins['geometry_sha256'], []).append(ins['name'])
        model = PublicationModel(ins, c)
        certificates = [model.route([i]) for i in range(1, 101)]
        bad = [dict(customer=r.route[0], reason=r.reason) for r in certificates if not r.feasible]
        manifest.append({k: ins[k] for k in ['name', 'path', 'sha256', 'geometry_sha256']} | dict(eligible=not bad))
        # Same mandatory customer cohort for every declared kinetic/threshold cell.
        for profile in c['quality']['kinetic_sensitivity']:
            for threshold in c['quality']['same_model_threshold_sensitivity']:
                other = PublicationModel(ins, c, profile, threshold)
                other_bad = [i for i in range(1, 101) if not other.route([i]).feasible]
                assert bool(other_bad) == bool(bad), 'scenario-dependent cohort requires explicit revision'
                regimes.append(dict(case=ins['name'], profile=profile, minimum_quality=threshold,
                                    singleton_feasible_customers=100-len(other_bad)))
        if bad:
            excluded.append(dict(case=ins['name'], customers=bad,
                proof='If a mandatory customer cannot be served by its direct singleton route at the best departure, extra stops cannot reduce its age/travel or relax original windows. K=n is otherwise sufficient.'))
            continue
        included.append(ins['name'])
        independent_audit(model, certificates, True)
        audited_route_sets += 1
        # An exhaustive pair witness can disprove the trivial all-singletons optimum.
        # It is a finite local certificate, not a claim that the constructed route is optimal.
        best_pair, best_gain, cost_value_tradeoff = None, 0.0, False
        for i in range(1, 101):
            for j in range(1, 101):
                if i == j:
                    continue
                pair = model.route([i, j])
                if not pair.feasible:
                    continue
                a, b = certificates[i-1], certificates[j-1]
                gain = pair.net_benefit-a.net_benefit-b.net_benefit
                operating_saving = (a.distance+b.distance-pair.distance)*c['money']['distance_U_per_km']+a.refrigeration_cost+b.refrigeration_cost-pair.refrigeration_cost
                if operating_saving > 1e-7 and gain < -1e-7:
                    cost_value_tradeoff = True
                if gain > best_gain:
                    best_pair, best_gain = pair, gain
        assert best_pair is not None and best_gain > 1e-7
        pair_solution = [best_pair]+[r for r in certificates if r.route[0] not in best_pair.route]
        independent_audit(model, pair_solution, True)
        audited_route_sets += 1
        relaxed = nearest_neighbour_witness(model, False)
        constrained = nearest_neighbour_witness(model, True)
        independent_audit(model, relaxed, False)
        independent_audit(model, constrained, True)
        audited_route_sets += 2
        violations = [r for r in relaxed if r.min_quality < model.minimum-1e-7]
        for r in constrained:
            # Fixed route: latest dispatch must dominate any earlier feasible one.
            early = model.route(r.route, False, departure=ins['nodes'][0][4])
            if early.feasible:
                assert r.net_benefit >= early.net_benefit-1e-7
                assert r.min_quality >= early.min_quality-1e-7
                departures_checked += 1
        witnesses.append(dict(case=ins['name'], geometry_sha256=ins['geometry_sha256'],
            diagnostic_only_not_algorithm_comparison=True,
            relaxed_route_count=len(relaxed), quality_feasible_route_count=len(constrained),
            relaxed_routes_below_quality_threshold=len(violations),
            relaxed_min_quality=min(r.min_quality for r in relaxed),
            constrained_min_quality=min(r.min_quality for r in constrained),
            all_singletons_net_benefit_U=sum(r.net_benefit for r in certificates),
            best_pair=asdict(best_pair), best_pair_net_gain_U=best_gain,
            all_singletons_provably_not_optimal=True,
            operating_cost_saving_can_reduce_net_benefit=cost_value_tradeoff,
            explicit_constraint_witness=asdict(violations[0]) if violations else None,
            feasible_solution=[asdict(r) for r in constrained]))
        print(json.dumps(dict(case=ins['name'], status='FEASIBILITY_CERTIFIED',
                              relaxed_quality_violating_routes=len(violations))), flush=True)

    assert len(geometry) == 4
    groups = sorted(geometry)
    folds = []
    for i, test in enumerate(groups):
        validation = groups[(i+1) % len(groups)]
        train = set(groups)-{test, validation}
        split = dict(fold=i, test_geometry=test, validation_geometry=validation, train_geometries=sorted(train),
            train=[m['name'] for m in manifest if m['eligible'] and m['geometry_sha256'] in train],
            validation=[x for x in geometry[validation] if x in included],
            test=[x for x in geometry[test] if x in included])
        assert split['train'] and split['validation'] and split['test']
        assert not set(split['train']) & set(split['test'])
        folds.append(split)
    assert sorted(x for f in folds for x in f['test']) == sorted(included)
    assert len(included) == 47 and len(excluded) == 9
    assert sorted(x['case'] for x in excluded) == ['c101','c102','c103','c104','c201','c202','c203','c204','rc205']
    active = [r['case'] for r in witnesses if r['relaxed_routes_below_quality_threshold']]
    assert active, 'do not freeze a claimed active quality constraint without a real witness'
    source_pdfs = [
        ('docs/static_delivery_research_20261020/sources/quality_driven_2021.pdf', c['sources']['quality']['pdf_sha256']),
        ('docs/static_delivery_research_20261021/sources/joint_distribution_RCR_2020.pdf', c['sources']['relative_value']['pdf_sha256']),
    ]
    for name, expected in source_pdfs:
        assert sha(EXTENSION_ROOT/name) == expected
    preserved = []
    for name, expected in [
        ('configs/static_delivery/product_evidence_v1.json', '3dd8c10768781145fe3a084dbce2c2e80cb7e13351de89d17ef6cab646b53947'),
        ('configs/static_delivery/net_benefit_scenario_v1.json', 'b74c9adb0d813ac1b8f35e10c4e31181459e23b8f73cc8f1ee56954938b9e38e'),
        ('results/static_delivery/net_benefit_scenario_pilot_20261020/results.json', '02f3b2ee2f67547501afed1ef3f8514cfe567992062294cd676d94830d39391b'),
    ]:
        assert sha(EXTENSION_ROOT/name) == expected
        preserved.append(dict(path=name, sha256=expected))
    save('dataset_manifest.json', dict(protocol=c['protocol_id'], config_sha256=sha(CONFIG),
        source_archive_sha256=sha(archive), files=manifest, exclusions=excluded,
        geometry_groups=geometry, all_original_bytes_match_source_archive=True))
    save('geometry_folds.json', dict(rule=c['dataset']['split'], folds=folds,
        inferential_limit='Only four spatial groups. Window variants and solver seeds are dependent; do not present 47 variants as 47 independent geometries.'))
    save('feasibility_witnesses.json', witnesses)
    save('sensitivity_preflight.json', regimes)
    summary = dict(status='PASS', protocol=c['protocol_id'], config_sha256=sha(CONFIG),
        original_cases=56, eligible_cases=len(included), proven_infeasible_cases=len(excluded),
        distinct_geometries=len(geometry), folds=len(folds), audited_route_sets=audited_route_sets,
        kinetic_threshold_case_cells=len(regimes), main_quality_rate_per_min=60*3.08e-6,
        maximum_delivery_age_min=(1-.8)/(60*3.08e-6),
        refrigeration_U_per_min=c['money']['refrigeration']['cost_U_per_vehicle_min'],
        value_U_per_kg=c['money']['reference_value_U_per_kg'],
        cases_with_quality_violation_witness=len(active), violation_witness_cases=active,
        latest_dispatch_dominance_checks=departures_checked, preserved_files=preserved,
        cases_with_certified_improving_pair=sum(w['all_singletons_provably_not_optimal'] for w in witnesses),
        cases_with_cost_value_tradeoff_witness=sum(w['operating_cost_saving_can_reduce_net_benefit'] for w in witnesses),
        no_new_customers=True, no_training=True, no_algorithm_superiority_claim=True)
    save('summary.json', summary)
    save('source_code_sha256.json', {str(p.relative_to(EXTENSION_ROOT)): sha(p)
         for p in [Path(__file__), EXTENSION_ROOT/'comparisons/static_coldchain/adapters/publication_model_v1.py', CONFIG]})
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
