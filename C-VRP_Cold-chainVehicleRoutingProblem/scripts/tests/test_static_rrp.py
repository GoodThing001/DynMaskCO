"""Tests use fixed public RRP cases; invalid records are test mutations only."""
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EXT = Path(__file__).resolve().parents[2]
MODULES = EXT / 'scripts/static_delivery'
SOURCE = EXT / 'data/public_sources_20261020/refrigerated_routing_rrp'
sys.path.insert(0, str(MODULES))


def api(name):
    assert (MODULES / (name + '.py')).is_file(), 'Static RRP implementation is missing'
    return importlib.import_module(name)


def small():
    return api('rrp_data').load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')


def test_original_manifest_and_matrix_without_fake_coordinates():
    instance = small()
    assert instance.nodes == 4
    assert instance.demands.tolist() == [0, 11, 11, 11]
    assert instance.distances.shape == (4, 4)
    assert not instance.distances.flags.writeable
    assert instance.raw['Tint'] == -20
    assert instance.source_commit == 'f5c5317a5ecd8fe64df779e79bd0397904bc0f9e'


def test_source_hash_drift_rejected(tmp_path):
    source = SOURCE / 'Datasets/D1/D1-n4-np33.json'
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'):
        api('rrp_data').load_instance(changed)


def test_unused_author_diagonal_is_preserved_not_silently_rewritten():
    instance = api('rrp_data').load_instance(SOURCE / 'Datasets/D1/D1-n34-np33.json')
    assert instance.distances[33, 33] == 48


@pytest.mark.parametrize('field', ['U', 'mass', 'A', 'driver_wage', 'temperature_ranges'])
def test_missing_physical_field_has_no_pilot_fallback(field):
    raw = json.loads((SOURCE / 'Datasets/D1/D1-n4-np33.json').read_text())
    del raw[field]
    with pytest.raises(ValueError, match=field):
        api('rrp_data').parse_instance(raw, 'invalid_test_record')


def test_nonfinite_matrix_and_temperature_rejected():
    raw = json.loads((SOURCE / 'Datasets/D1/D1-n4-np33.json').read_text())
    raw['distances'][0][1] = float('nan')
    with pytest.raises(ValueError, match='distances'):
        api('rrp_data').parse_instance(raw, 'invalid_test_record')
    raw['distances'][0][1] = 1
    raw['temperature_ranges'][0][8] = 0
    with pytest.raises(ValueError, match='COP'):
        api('rrp_data').parse_instance(raw, 'invalid_test_record')


def test_reference_waits_follow_paper_index_and_raw_fields_are_preserved():
    record = api('rrp_data').load_reference(SOURCE / 'Solutions/D2/sol-D2-n11-np30.txt')
    assert record.route[:3] == (0, 2, 6)
    assert record.waits_s[:3] == (0, 0, 327)
    assert record.raw_waits_s[:3] == (0, 327, 837)
    assert record.reported_costs['total'] == 42147.968
    assert record.objective == 'without_wage'


def test_reference_cp_arc_waits_are_shifted_to_customer_positions():
    record = api('rrp_data').load_reference(SOURCE / 'Solutions/D2/sol-D2-n4-np33.txt')
    assert record.raw_waits_s == (1514, 1348, 0, 0)
    assert record.waits_s == (0, 1514, 1348, 0)
    assert record.wait_convention == 'paper_pre_service_from_outgoing_arc'
    instance = api('rrp_data').load_instance(SOURCE / 'Datasets/D2/D2-n4-np33.json')
    checked = api('rrp_evaluator').evaluate(instance, record.route, record.waits_s, 'without_wage')
    assert checked['feasible']
    assert checked['legs'][1]['arrival_s'] == record.saved['Arrival times'][1]
    archive = api('rrp_evaluator').evaluate(instance, record.route, record.waits_s,
                                           'without_wage', profile='archive_pre_service')
    assert archive['feasible']
    assert abs(archive['costs']['total_eur_year'] - record.reported_costs['total']) < .1


def test_archive_profile_preserves_source_depot_delay_but_paper_fixes_start():
    instance = api('rrp_data').load_instance(SOURCE / 'Datasets/D2/D2-n12-np33.json')
    record = api('rrp_data').load_reference(SOURCE / 'Solutions/D2/sol-D2-n12-np33.txt')
    assert record.saved_clock_convention == 'departure_position_wait'
    assert record.raw_waits_s[0] == 288 and record.waits_s[0] == 0
    assert not api('rrp_evaluator').evaluate(instance, record.route, record.raw_waits_s, 'without_wage')['feasible']
    assert api('rrp_evaluator').evaluate(instance, record.route, record.raw_waits_s,
                                        'without_wage', profile='archive_compatibility')['feasible']


def test_incomplete_raw_wait_vector_is_preserved_alongside_unique_derivation():
    record = api('rrp_data').load_reference(SOURCE / 'Solutions/D1/sol-D1-n9-np32.txt')
    assert len(record.raw_waits_s) == 8 and len(record.route) == 9
    assert record.wait_convention == 'derived_unique_zero_from_author_clock_equalities'
    assert record.waits_s == (0.,) * 9


def test_paper_service_roundtrip_factor_is_not_silently_halved():
    evaluator = api('rrp_evaluator')
    instance = small()
    assert evaluator.service_seconds(instance, quantity=3, delivered_pallets=0, profile='paper') == 432
    assert evaluator.service_seconds(instance, quantity=3, delivered_pallets=3, profile='paper') == 450
    assert evaluator.service_seconds(instance, quantity=3, delivered_pallets=3, profile='archive_compatibility') == 441


def test_hour_boundary_uses_next_half_open_speed_slot():
    evaluator = api('rrp_evaluator')
    instance = small()
    assert evaluator.speed_at(instance, 8 * 3600 - 1e-5) == float(instance.raw['speed_ranges'][7][4])
    assert evaluator.speed_at(instance, 8 * 3600) == float(instance.raw['speed_ranges'][8][4])


def test_paper_pre_service_wait_is_accounted_in_incoming_model_clock():
    result = api('rrp_evaluator').evaluate(small(), [0, 1, 2, 3], [0, 3600, 0, 0], 'with_wage')
    incoming, customer = result['legs'][:2]
    assert customer['arrival_s'] == pytest.approx(incoming['next_physical_arrival_s'] + 3600)
    assert incoming['next_arrival_s'] == customer['arrival_s']
    assert sum(leg['next_arrival_s'] - leg['arrival_s'] for leg in result['legs']) == pytest.approx(result['duration_s'])


def test_recomputes_clocks_load_and_cost_instead_of_copying_reference():
    instance = small()
    record = api('rrp_data').load_reference(SOURCE / 'Solutions/D1/sol-D1-n4-np33.txt')
    evaluator = api('rrp_evaluator')
    result = evaluator.evaluate(instance, record.route, record.waits_s, 'with_wage')
    assert result['feasible']
    assert result['initial_load_kg'] == 33 * 600
    assert result['terminal_load_kg'] == 0
    assert result['complete']
    assert len(result['legs']) == 4
    assert result['legs'][0]['load_kg'] == 19800
    assert result['legs'][-1]['load_kg'] == 0
    assert result['costs']['total_eur_year'] == pytest.approx(sum(result['costs'][k] for k in (
        'traction_eur_year', 'transmission_eur_year', 'infiltration_eur_year', 'driver_eur_year')))
    no_wage = evaluator.evaluate(instance, record.route, record.waits_s, 'without_wage')
    assert no_wage['costs']['driver_eur_year'] == 0
    assert result['costs']['total_eur_year'] - no_wage['costs']['total_eur_year'] == pytest.approx(
        instance.raw['driver_wage'] * instance.raw['days'] * result['duration_s'])


@pytest.mark.parametrize('route,waits,error', [
    ([0, 1, 1, 3], [0, 0, 0, 0], 'route_coverage'),
    ([0, 1, 2], [0, 0, 0], 'route_coverage'),
    ([0, 1, 2, 3], [0, -1, 0, 0], 'invalid_waits'),
    ([0, 1, 2, 3], [0, 80000, 0, 0], 'clock_outside_day'),
])
def test_invalid_candidate_is_retained_as_infeasible(route, waits, error):
    result = api('rrp_evaluator').evaluate(small(), route, waits, 'with_wage')
    assert not result['feasible']
    assert error in result['errors']


def test_unknown_profile_and_unknown_objective_are_rejected():
    evaluator = api('rrp_evaluator')
    with pytest.raises(ValueError, match='profile'):
        evaluator.evaluate(small(), [0, 1, 2, 3], [0] * 4, 'with_wage', profile='guess')
    with pytest.raises(ValueError, match='objective'):
        evaluator.evaluate(small(), [0, 1, 2, 3], [0] * 4, 'guess')


def test_return_after_midnight_is_outside_source_clock_domain():
    raw = json.loads((SOURCE / 'Datasets/D1/D1-n4-np33.json').read_text())
    raw['start_time'] = 20.75  # Invalid candidate fixture only, not method data.
    instance = api('rrp_data').parse_instance(raw, 'late_start_test')
    report = api('rrp_evaluator').evaluate(instance, [0, 1, 2, 3], [0] * 4, 'with_wage')
    assert not report['feasible']
    assert 'clock_outside_day' in report['errors']
