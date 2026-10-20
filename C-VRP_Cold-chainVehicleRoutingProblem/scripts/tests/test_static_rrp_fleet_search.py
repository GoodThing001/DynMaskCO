from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'static_delivery'))
from rrp_data import SOURCE, load_instance, PRIMARY_PROFILE
from rrp_search import nearest_route
import rrp_fleet_search as search


def test_common_initial_split_and_monotone_fleet_acceptance():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')
    result = search.solve_fleet(x, nearest_route(x), [0] * x.nodes, max_vehicles=2,
                               method='local_search', iterations=2, seconds=3,
                               objective='with_wage', profile=PRIMARY_PROFILE)
    assert result['report']['feasible'] and result['actual_split_calls'] >= 1
    assert result['actual_tour_evaluations'] >= 6
    costs = [r['best_cost_eur_year'] for r in result['iterations']]
    assert costs == sorted(costs, reverse=True)
    assert result['learned_fleet_assignment_head'] is False


def test_model_is_required_and_tiny_budget_has_no_partial_success():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')
    with pytest.raises(ValueError, match='model'):
        search.solve_fleet(x, nearest_route(x), [0] * x.nodes, max_vehicles=2,
                           method='maskco', iterations=1, seconds=1,
                           objective='with_wage', profile=PRIMARY_PROFILE)
    result = search.solve_fleet(x, nearest_route(x), [0] * x.nodes, max_vehicles=2,
                               method='local_search', iterations=1, seconds=1e-12,
                               objective='with_wage', profile=PRIMARY_PROFILE)
    assert not result['report']['feasible']
    assert result['failure_log'][0]['errors'] == ['search_budget_exhausted']
