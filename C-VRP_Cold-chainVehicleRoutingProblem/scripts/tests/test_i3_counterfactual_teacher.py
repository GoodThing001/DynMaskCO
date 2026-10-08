"""Engineering tests only: tiny training-seed fixture, never development/test days."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_counterfactual_teacher import (CaptureTeacherReplanner, ReplaySafeSaaReplanner,
    TRAIN_SEED, record_identity, replay_action, scenario_margin, teacher_preference)
from scenario_saa import CondHistoricalSampler, build_history, make_c0_contract_v2
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from run_identity import dataset_identity, dataset_meta_sha256
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from recourse_snapshot import snapshot_state_hash
from strict_online_env import StrictOnlineEnv


def check_raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError('Expected ' + exc.__name__)


def fixture(infeasible=False):
    contract = make_c0_contract_v2()
    ds = generate_dataset(1, 4, TRAIN_SEED)
    # Controlled fixture: same-time decisions, non-index reveal order and an
    # in-flight vehicle at the subsequent event. Not a scientific dataset.
    ds['coords'][0] = [[.5, .5], [.53, .5], [.54, .5], [.52, .5], [.55, .5]]
    ds['demands'][0] = [0, 2, 2, 2, 2]
    if infeasible:
        ds['demands'][0, 3] = 60
    ds['temp_class'][0] = 0
    ds['reveal_time'][0] = [1e6, 1, 1.02, 1, 2]
    ds['tw_start'][0] = 0
    ds['tw_end'][0] = 22
    ds = add_v2_initial_quality(ds, contract)
    history = build_history(generate_dataset(3, 4, TRAIN_SEED))

    def factory(pc=5, capture=False):
        cls = CaptureTeacherReplanner if capture else ReplaySafeSaaReplanner
        opts = {'capture_ordinals': set(range(4))} if capture else {}
        return cls(budget=1e9, capacity=50, booking_horizon=BOOKING_HORIZON,
            contract=contract, cooling_share=1, sampler=CondHistoricalSampler(history),
            reject_penalty={0: pc, 1: 10, 2: 15}, K=2, time_limit=float('inf'),
            arm_seed=7001, shadow_mode='greedy', energy_pricing='marginal',
            future_policy='density', **opts)

    policy = factory(capture=True)
    records = []
    env = StrictOnlineEnv(ds, capacity=50, tw_speed=SPEED_KMH / KM_PER_UNIT,
        num_vehicles=2, replanner=policy, coldchain_contract=contract,
        booking_horizon=BOOKING_HORIZON)
    env.snapshot_hook = policy.capture_hook(records, train_seed=TRAIN_SEED,
        dataset_sha256=dataset_identity(ds), dataset_meta_sha256=dataset_meta_sha256(ds))
    traces, _ = env.run(0)
    original = evaluate_trace_a1(traces, {k: v[0] for k, v in ds.items()}, contract,
        policy._accepted, policy._rejected, policy.budget, policy.reject_penalty)
    return ds, contract, factory, policy, records, original


def replay(record, ds, contract, factory, accept):
    return replay_action(record, ds, factory, contract, BOOKING_HORIZON,
                         SPEED_KMH / KM_PER_UNIT, accept)


def test_predecision_replay():
    ds, contract, factory, policy, records, original = fixture()
    assert len(records) == 4, len(records)
    assert original['hard_feasible'], original
    assert records[0].pending == (3,), records[0].pending
    assert records[0].visible == records[1].visible
    assert records[0].visible.accepted == frozenset()
    assert 1 in records[1].snapshot['replanner_state']['accepted']
    assert records[0].snapshot['replanner_state']['i3_replay']['id_map'] == {1: 1, 3: 2}
    assert records[-1].snapshot['replanner_state']['i3_replay']['id_map'][2] == 3
    data_hash = dataset_identity(ds)
    for record in records:
        state = record.snapshot['replanner_state']
        assert record.candidate not in state['accepted']
        assert record.candidate not in state['rejected']
        assert all(o.reveal <= record.visible.clock + 1e-6 for o in record.visible.orders)
        snapshot_hash = snapshot_state_hash(record.snapshot)
        chosen = replay(record, ds, contract, factory, record.candidate in policy._accepted)
        assert chosen['hard_feasible'], chosen
        assert chosen['accepted'] == sorted(policy._accepted), (record.candidate, chosen)
        assert chosen['rejected'] == sorted(policy._rejected)
        for metric in ('utility', 'revenue', 'fuel_cost', 'reject_loss', 'energy_kwh', 'distance_km'):
            np.testing.assert_allclose(chosen[metric], original[metric], rtol=0, atol=1e-9,
                                       err_msg=str((record.candidate, metric)))
        for accept in (False, True):
            branch = replay(record, ds, contract, factory, accept)
            repeat = replay(record, ds, contract, factory, accept)
            assert branch['action_feasible'] and branch['hard_feasible'], branch
            assert branch['utility'] == repeat['utility']
            assert branch['accepted'] == repeat['accepted']
            assert branch['rejected'] == repeat['rejected']
            assert (record.candidate in branch['accepted']) == accept
            assert 3 in branch['accepted'] or 3 in branch['rejected']
        assert snapshot_state_hash(record.snapshot) == snapshot_hash
    assert dataset_identity(ds) == data_hash
    print('PASS: pre-decision state, same-event pending decisions, reveal ID map, full-trajectory parity and isolated repeat replay')


def test_rejection_guards_and_margin():
    ds, contract, factory, policy, records, _ = fixture()
    record = records[0]
    changed = deepcopy(ds)
    changed['coords'][0, -1, 0] += .1
    check_raises(lambda: replay(record, changed, contract, factory, True))
    shape_changed = deepcopy(ds)
    shape_changed['tw_start'] = shape_changed['tw_start'].T.copy()
    assert dataset_identity(shape_changed) == dataset_identity(ds)
    check_raises(lambda: replay(record, shape_changed, contract, factory, True))
    corrupt = deepcopy(record.snapshot)
    corrupt['clock'] += .1
    check_raises(lambda: replay(replace(record, snapshot=corrupt), ds, contract, factory, True))
    check_raises(lambda: replay(replace(record, train_seed=20260926), ds, contract, factory, True))
    check_raises(lambda: policy.capture_hook([], train_seed=20260926,
                 dataset_sha256='x', dataset_meta_sha256='x'))
    check_raises(lambda: replay(replace(record, pending=()), ds, contract, factory, True))
    check_raises(lambda: replay(record, ds, contract, lambda: factory(pc=0), True))
    margin = scenario_margin(record, ds, factory, contract, BOOKING_HORIZON,
                             SPEED_KMH / KM_PER_UNIT, [[], []])
    assert margin['action_feasible'] and np.isfinite(margin['margin'])
    margin0 = scenario_margin(record, ds, lambda: factory(pc=0), contract,
                              BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, [[], []])
    np.testing.assert_allclose(margin['margin'] - margin0['margin'], 5, rtol=0, atol=1e-9)
    check_raises(lambda: scenario_margin(record, ds, factory, contract,
                                        BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, []))
    check_raises(lambda: scenario_margin(record, ds, factory, contract,
        BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, [[record.visible.orders[0]]]))
    future_changed = deepcopy(ds)
    future_changed['coords'][0, [2, 4]] = [[.1, .9], [.9, .1]]
    future_changed['demands'][0, [2, 4]] = 40
    future_changed['temp_class'][0, [2, 4]] = 2
    # New synthetic private record to test causality, not bypassing the identity
    # guard on an archived teacher record. Model-visible input stays identical.
    altered = replace(record, dataset_sha256=dataset_identity(future_changed),
                      dataset_meta_sha256=dataset_meta_sha256(future_changed), record_hash='')
    altered = replace(altered, record_hash=record_identity(altered))
    changed_margin = scenario_margin(altered, future_changed, factory, contract,
                                    BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, [[], []])
    assert changed_margin['margin'] == margin['margin']
    ds, contract, factory, _, records, original = fixture(infeasible=True)
    record = next(r for r in records if r.candidate == 3)
    assert original['hard_feasible']
    bad_accept = replay(record, ds, contract, factory, True)
    assert not bad_accept['action_feasible'] and bad_accept['utility'] is None
    assert replay(record, ds, contract, factory, False)['hard_feasible']
    pref = teacher_preference(bad_accept, replay(record, ds, contract, factory, False),
                              minimum_gap=5)
    assert not pref['teacher_valid'] and not pref['decision_eligible']
    assert pref['teacher_delta'] is None and pref['loss_teacher_margin'] == 0
    good = {'action_feasible': True, 'hard_feasible': True, 'utility': 10}
    equal = teacher_preference(good, good, minimum_gap=0)
    assert not equal['decision_eligible'] and equal['teacher_valid']
    bad = dict(good, hard_feasible=False, utility=float('-inf'))
    assert teacher_preference(good, bad, minimum_gap=5)['loss_teacher_margin'] == 0
    assert not scenario_margin(record, ds, factory, contract, BOOKING_HORIZON,
                               SPEED_KMH / KM_PER_UNIT, [[], []])['action_feasible']
    print('PASS: data/shape/snapshot/record/config/seed guards, shadow future causality, finite per-K margin, penalty once, invalid/tie teachers retained with zero decision weight')


if __name__ == '__main__':
    test_predecision_replay()
    test_rejection_guards_and_margin()
    print('ALL PASS: I3 counterfactual teacher engineering tests (no formal training or gain evidence)')
