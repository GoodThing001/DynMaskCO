import importlib
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parents[2]
MODULES = EXT / 'scripts/static_delivery'
sys.path.insert(0, str(MODULES))
from rrp_data import SOURCE, load_instance
from rrp_evaluator import evaluate


def api():
    assert (MODULES / 'rrp_search.py').is_file(), 'Static search module missing'
    return importlib.import_module('rrp_search')


def test_nearest_initial_solution_and_monotone_acceptance():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')
    search = api()
    route = search.nearest_route(x)
    assert route[0] == 0 and set(route) == set(range(x.nodes))
    initial = evaluate(x, route, [0] * x.nodes, 'with_wage')
    result = search.solve(x, route, [0] * x.nodes, method='local_search',
                          iterations=2, seconds=10, objective='with_wage', profile='paper')
    assert result['report']['feasible']
    assert result['report']['costs']['total_eur_year'] <= initial['costs']['total_eur_year']
    costs = [r['best_cost_eur_year'] for r in result['iterations']]
    assert costs == sorted(costs, reverse=True)
    assert result['initialization'] == 'caller_supplied_no_reference_lookup'


def test_timed_search_can_preserve_incumbent_without_attempt():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')
    search = api()
    route = search.nearest_route(x)
    result = search.solve(x, route, [0] * x.nodes, method='local_search',
                          iterations=0, seconds=1, objective='without_wage', profile='paper')
    assert result['route'] == route
    assert result['iterations'] == []
    assert result['report']['costs']['driver_eur_year'] == 0


def test_failed_single_candidates_retain_routes_waits_and_errors():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n34-np33.json')
    search = api()
    route = search.nearest_route(x)
    result = search.solve(x, route, [0] * x.nodes, method='local_search',
                          iterations=1, seconds=.03, objective='with_wage', profile='paper',
                          time_domain='operational_tour_9h')
    assert not result['report']['feasible']
    assert result['failure_log'][0]['stage'] == 'initial_evaluation'
    assert len(result['failure_log']) == result['actual_evaluations']
    for row in result['failure_log']:
        assert len(row['route']) == len(row['waits_s']) == x.nodes
        assert 'tour_horizon' in row['errors']
