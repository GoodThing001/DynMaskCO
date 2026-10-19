"""GAP-01 synthetic-only tests: no NPZ, model loading, rollout or optimizer."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('experimental', 'evaluation', 'training', 'simulation', 'coldchain'))
from p2_pos01c_gap import (CapacityConditionedProposal, PublicPrefix, mask_diagnostic,
                          prefix_from_view, ref, ScenarioOrder, make_c0_contract_v2)
from p2_pos01_teacher import EventRNG
from scenario_saa import CondHistoricalSampler


def order(oid, reveal, x=.25, typ=0):
    return ScenarioOrder(oid, reveal, x, .25, 1., typ, reveal, 15., .01)


def prefix():
    return PublicPrefix(1., 16., 3, (order(1, 0.), order(2, .5)))


def history():
    return [tuple([order(j+1, .1, typ=i % 3) for j in range(i % 4)] +
                  [order(20+j, 2.+j) for j in range(1+i % 3)]) for i in range(12)]


class PositionRNG:
    def __init__(self, position):
        self.position = position

    def integers(self, low, high):
        assert low == 0 and self.position < high
        return self.position


def test_original_knn_membership_and_order_exactly_match_actual_sampler():
    hist = history()
    base = CondHistoricalSampler(hist, n_neighbors=10)
    audit = CapacityConditionedProposal(hist).support(prefix())
    for pos, idx in enumerate(audit['nearest']):
        actual = base.sample(prefix(), PositionRNG(pos), 1)[0]
        expected = [o for o in hist[idx] if o.reveal > 1.+1e-6]
        assert [replace(o, oid=0) for o in actual] == [replace(o, oid=0) for o in expected]
    assert audit['free_slots'] == 1
    assert audit['eligible'] == [i for i in audit['nearest']
                                 if audit['suffix_counts'][i] <= 1]
    assert audit['retained_mass'] == len(audit['eligible'])/10


def test_full_suffix_preserved_without_clipping_or_redraw():
    sampler = CapacityConditionedProposal(history())
    for k in range(20):
        future, receipt = sampler.draw(prefix(), EventRNG('SYNTHETIC', 3),
                                       ('EXOGENOUS_FUTURE', float(1).hex(), k), synthetic=True)
        original = [o for o in sampler.history[receipt['selected_history_index']]
                    if o.reveal > 1.+1e-6]
        assert len(future) == len(original) <= receipt['free_slots']
        assert [replace(o, oid=0) for o in future] == [replace(o, oid=0) for o in original]


def test_no_support_never_refills_neighbors_or_retries():
    hist = [(order(1, 2.), order(2, 3.))]*2 + [(order(1, 2.),)]*4
    sampler = CapacityConditionedProposal(hist, n_neighbors=2)
    audit = sampler.support(prefix())
    assert audit['nearest'] == [0, 1] and audit['all_history_compatible'] == 4
    assert audit['status'] == 'NO_SUPPORT'
    class MustNotDraw:
        def generator(self, key):
            raise AssertionError('must reject before any draw')
    with pytest.raises(ValueError, match='NO_SUPPORT'):
        sampler.draw(prefix(), MustNotDraw(), 'event', synthetic=True)


def test_single_root_is_low_support_and_real_draw_remains_locked():
    sampler = CapacityConditionedProposal([(order(1, 2.),),
                                          (order(1, 2.), order(2, 3.))])
    assert sampler.support(prefix())['status'] == 'LOW_SUPPORT'
    with pytest.raises(ValueError, match='LOW_SUPPORT'):
        sampler.draw(prefix(), EventRNG('SYNTHETIC', 4), 'key', synthetic=True)
    with pytest.raises(PermissionError, match='NOT_AUTHORISED'):
        sampler.draw(prefix(), EventRNG('SYNTHETIC', 4), 'key')


def test_future_public_prefix_and_invalid_horizon_rejected():
    sampler = CapacityConditionedProposal(history())
    with pytest.raises(ValueError, match='INVALID_PUBLIC_PREFIX'):
        sampler.support(replace(prefix(), orders=prefix().orders+(order(3, 2.),)))
    bad_history = [(order(1, 16.),)]*2
    with pytest.raises(ValueError, match='HISTORY_OUTSIDE_PUBLIC_HORIZON'):
        CapacityConditionedProposal(bad_history).support(prefix())


def test_pairing_independent_of_branch_order_and_other_rng_consumption():
    sampler = CapacityConditionedProposal(history())
    events = EventRNG('SYNTHETIC_FROZEN_NAMESPACE', 913)
    expected = {}
    for k in (0, 1):
        key = ('EXOGENOUS_FUTURE', float(1).hex(), k)
        expected[k] = sampler.draw(prefix(), events, key, synthetic=True)
    for branch in ('MASK_B', 'KEEP', 'MASK_A', 'DIRECT'):
        events.normal(('UNRELATED', branch), 123)
        for k in (1, 0):
            key = ('EXOGENOUS_FUTURE', float(1).hex(), k)
            assert sampler.draw(prefix(), events, key, synthetic=True) == expected[k]


def synthetic_view(n=6, future_x=.7, future_reveal=9.):
    coords = [(0., 0.), (.01, .03), (.02, .01), (.04, .025),
              (.07, .01), (.08, .04), (.055, .075)]
    rows = [dict(oid=i, x=x, y=y, demand=float(i != 0), tw_start=0.,
                 tw_end=22. if i == 0 else 15., service=.01 if i else 0.,
                 temp_class=0, initial_quality=1., reveal=0.)
            for i, (x, y) in enumerate(coords[:n+1])]
    rows.append(dict(rows[1], oid=49, x=future_x, reveal=future_reveal))
    vehicles = ((ref.Vehicle(0, mutable_suffix=(3, 1, 2)),
                 ref.Vehicle(1, mutable_suffix=(6, 4, 5))) if n == 6 else
                (ref.Vehicle(0, mutable_suffix=(1, 2)),))
    return ref.project_public(rows, vehicles, accepted=tuple(range(1, n+1)), rejected=(),
        served=(), replan_ids=tuple(v.vehicle_id for v in vehicles), clock=1.,
        booking_horizon=16., capacity=50., tw_speed=1.5, energy_budget=186.06639500089943,
        contract=make_c0_contract_v2())


def test_hidden_source_perturbation_does_not_change_public_masks_support_or_wait():
    a, b = synthetic_view(), synthetic_view(future_x=99999., future_reveal=15.9)
    assert ref.identity(a) == ref.identity(b)
    assert mask_diagnostic(a) == mask_diagnostic(b)
    assert prefix_from_view(a) == prefix_from_view(b)
    sampler = CapacityConditionedProposal(history())
    assert sampler.support(prefix_from_view(a)) == sampler.support(prefix_from_view(b))
    for x, y in zip(a.vehicles, b.vehicles):
        assert ref.wait_or_return(a, x) == ref.wait_or_return(b, y)


def test_mask_count_requires_actual_distinct_proposals_and_keeps_protections():
    single = mask_diagnostic(synthetic_view(n=2))
    assert single['duplicate_raw_proposals'] == 3
    assert single['distinct_masks'] == 1 and single['distinct_mask_pairs'] == 0
    assert not single['multi_mask_eligible']
    view = synthetic_view()
    multi = mask_diagnostic(view)
    assert multi['distinct_masks'] >= 2 and multi['multi_mask_eligible']
    assert multi['conservative_base_routes_pass']
    protected = mask_diagnostic(replace(view, protected=(0, 1)))
    assert protected['mutable_customers'] == [] and protected['distinct_masks'] == 0
    assert not protected['multi_mask_eligible']
    over_capacity = replace(view, vehicles=(replace(view.vehicles[0], current_load=50.),
                                            view.vehicles[1]))
    with pytest.raises(ValueError, match='capacity'):
        mask_diagnostic(over_capacity)


def test_empty_suffix_is_legal_and_all_public_ids_consume_slots():
    sampler = CapacityConditionedProposal([(order(1, 0.),)]*2)
    full = replace(prefix(), declared_customer_slots=2)
    assert full.free_slots() == 0  # no reuse of served/rejected public identities
    future, receipt = sampler.draw(full, EventRNG('SYNTHETIC', 1), 'key', synthetic=True)
    assert future == () and receipt['selected_suffix_count'] == 0
