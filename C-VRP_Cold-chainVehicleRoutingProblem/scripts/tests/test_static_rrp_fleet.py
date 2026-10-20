"""All schedules use existing public orders; no generated customer instances."""
import itertools
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'static_delivery'))
from rrp_data import SOURCE, load_instance, load_reference, PRIMARY_PROFILE
from rrp_evaluator import evaluate, service_seconds
import rrp_fleet as fleet


def small():
    return load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')


def test_two_trucks_only_load_their_orders_and_reset_pallet_positions():
    x = small()
    report = fleet.evaluate_fleet(x, [[0, 1, 2], [0, 3]], [[0, 0, 0], [0, 0]],
                                  'with_wage', max_vehicles=2)
    assert report['feasible'] and report['vehicles_used'] == 2
    a, b = report['tours']
    assert a['initial_load_kg'] == 22 * 600
    assert b['initial_load_kg'] == 11 * 600
    assert a['terminal_load_kg'] == b['terminal_load_kg'] == 0
    assert b['legs'][1]['service_s'] == service_seconds(x, 11, 0)
    assert b['legs'][1]['from'] == 3
    assert report['costs']['total_eur_year'] == pytest.approx(sum(r['costs']['total_eur_year'] for r in (a, b)))
    assert x.raw['clients'] == 4 and x.demands.tolist() == [0, 11, 11, 11]


@pytest.mark.parametrize('routes,error', [([[0, 1], [0, 1, 2, 3]], 'customer_coverage'),
                                         ([[0, 1], [0, 2]], 'customer_coverage'),
                                         ([[0, 1], [0, 2], [0, 3]], 'fleet_limit')])
def test_global_coverage_and_fleet_limit_are_independent_constraints(routes, error):
    checked = fleet.evaluate_fleet(small(), routes, [[0] * len(r) for r in routes],
                                   'with_wage', max_vehicles=2)
    assert not checked['feasible'] and error in checked['errors']


def test_one_vehicle_split_equals_single_tour():
    x = small()
    route, waits = [0, 1, 2, 3], [0, 0, 0, 0]
    expected = evaluate(x, route, waits, 'with_wage')
    result = fleet.split_giant_tour(x, route, waits, 1, 'with_wage')
    assert result['report']['feasible'] and result['routes'] == [route]
    assert result['report']['costs']['total_eur_year'] == pytest.approx(expected['costs']['total_eur_year'])


def test_split_matches_exhaustive_partitions_for_fixed_public_order():
    x = small()
    customers, values = [1, 2, 3], []
    for count in (0, 1):
        for cut in itertools.combinations((1, 2), count):
            bounds = (0, *cut, 3)
            routes = [[0, *customers[a:b]] for a, b in zip(bounds, bounds[1:])]
            r = fleet.evaluate_fleet(x, routes, [[0] * len(t) for t in routes], 'with_wage', max_vehicles=2)
            if r['feasible']:
                values.append(r['costs']['total_eur_year'])
    actual = fleet.split_giant_tour(x, [0, *customers], [0] * 4, 2, 'with_wage')
    assert actual['report']['costs']['total_eur_year'] == pytest.approx(min(values))
    assert actual['optimality_scope'] == 'cuts_for_fixed_order_and_waits_only'


def test_n34_can_be_split_under_explicit_nine_hour_operational_limit():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n34-np33.json')
    r = load_reference(SOURCE / 'Solutions/D1/sol-D1-n34-np33.txt')
    single = fleet.split_giant_tour(x, r.route, r.waits_s, 1, 'with_wage', time_domain='operational_tour_9h')
    multiple = fleet.split_giant_tour(x, r.route, r.waits_s, 2, 'with_wage', time_domain='operational_tour_9h')
    assert not single['report']['feasible']
    assert multiple['report']['feasible'] and multiple['report']['vehicles_used'] == 2


def test_expired_deadline_never_claims_partial_partition_is_feasible():
    result = fleet.split_giant_tour(small(), [0, 1, 2, 3], [0] * 4, 2, 'with_wage', deadline=0)
    assert not result['report']['feasible']
    assert 'search_budget_exhausted' in result['report']['errors']


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_fleet_size_is_rejected(value):
    with pytest.raises(ValueError, match='vehicles'):
        fleet.split_giant_tour(small(), [0, 1, 2, 3], [0] * 4, value, 'with_wage')
