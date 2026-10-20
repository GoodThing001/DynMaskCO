"""Paper-priority contract, unique reference recovery, and explicit time domains."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'static_delivery'))
from rrp_data import SOURCE, load_instance, load_reference, standard_reference_wait_mapping
from rrp_evaluator import evaluate


def small():
    return load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')


def test_missing_zero_wait_is_uniquely_derived_with_raw_provenance():
    ref = load_reference(SOURCE / 'Solutions/D1/sol-D1-n9-np32.txt')
    assert len(ref.raw_waits_s) == 8
    assert ref.waits_s == (0.,) * 9
    assert ref.label_origin == 'derived_unique_zero_from_author_clock_equalities'
    assert ref.source_sha256 == 'd9d5892773d4901f25b0862393e427b6f3d459042072bfb356d5a770bbc47bb6'


def test_conflicting_clock_record_never_receives_a_guessed_wait():
    ref = load_reference(SOURCE / 'Solutions/D1/sol-D1-n9-np32.txt')
    saved = {k: list(v) for k, v in ref.saved.items()}
    saved['Departure times'][2] += 1
    waits, origin = standard_reference_wait_mapping(ref.route, ref.raw_waits_s, saved)
    assert len(waits) == 8 and origin == 'incomplete_or_unresolved'


def test_paper_index_wins_over_inconsistent_saved_clock_interpretation():
    ref = load_reference(SOURCE / 'Solutions/D2/sol-D2-n11-np30.txt')
    assert ref.waits_s[:4] == (0, 0, 327, 837)
    assert ref.saved_clock_convention == 'departure_position_wait'


def test_one_day_and_operational_nine_hours_are_distinct_domains():
    x = small()
    route, waits = [0, 1, 2, 3], [0., 9 * 3600., 0., 0.]
    day = evaluate(x, route, waits, 'with_wage')
    limited = evaluate(x, route, waits, 'with_wage', time_domain='operational_tour_9h')
    assert day['profile'] == 'rrp_mip_dimensional_v1'
    assert day['feasible'] and day['exceeds_operational_9h']
    assert day['duration_s'] > 9 * 3600
    assert not limited['feasible'] and 'tour_horizon' in limited['errors']
    assert day['driving_s'] + day['service_total_s'] + day['waiting_total_s'] == pytest.approx(day['duration_s'])


def test_primary_mip_clock_covers_every_wait_once():
    r = evaluate(small(), [0, 1, 2, 3], [0, 100, 200, 300], 'with_wage')
    assert r['feasible']
    assert sum(leg['next_arrival_s'] - leg['arrival_s'] for leg in r['legs']) == pytest.approx(r['duration_s'])
    assert r['costs']['driver_eur_year'] == pytest.approx(.0022 * 329 * r['duration_s'])


def test_unknown_time_domain_is_rejected():
    with pytest.raises(ValueError, match='time domain'):
        evaluate(small(), [0, 1, 2, 3], [0] * 4, 'with_wage', time_domain='unexplained')
