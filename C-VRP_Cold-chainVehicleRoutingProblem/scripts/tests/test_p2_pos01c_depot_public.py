"""P2-C-FIX-01: depot sentinel regression; real NPZs are projected only at t=0.

Never calls the smoke entry, a real source-day engine, real continuations or an
optimizer. Original A/B regression is run separately with its gradient excluded.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import EXTENSION_ROOT, configure_imports
configure_imports(('tests', 'experimental', 'models', 'training', 'simulation', 'evaluation', 'coldchain'))
import p2_pos01_reference as ref
import p2_pos01_teacher as teacher
from strict_online_env import StrictOnlineEnv, VehicleState
from scenario_saa import make_c0_contract_v2, build_history
from test_p2_pos01_reference import (
    synthetic_case, synthetic_view, saa_synthetic_dataset, backbone, model)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_projection(data, spec, clock, admit=False):
    """Static engine containers; optional admission is for HANDCRAFTED data only."""
    engine = StrictOnlineEnv(data, spec.capacity, spec.tw_speed, num_vehicles=2,
        coldchain_contract=spec.contract, booking_horizon=spec.booking_horizon)
    policy = teacher.PublicSaaReplanner(spec)
    vehicles = [VehicleState(vehicle_id=i) for i in range(2)]
    served = np.zeros(51, bool)
    served[0] = True
    proxy, published = ref.saa_public_env(engine, 0, clock)
    if admit:
        policy.on_reveal(engine, 0, clock, vehicles, served, published)
    view = ref.view_from_saa(proxy, policy, clock, vehicles, served)
    return view, proxy, published, policy


@pytest.mark.parametrize('relative_time', ('zero', 'before', 'at', 'after'))
def test_n50_depot_sentinel_and_first_reveal_saa_boundary(relative_time):
    # N50 fixture with the exact original-generator sentinel and no t=0 order.
    data, spec = saa_synthetic_dataset()
    data['reveal_time'][0, 0] = 1e6
    before = teacher.dataset_digest(data)
    first = float(data['reveal_time'][0, 1:].min())
    assert first > 2*ref.EVENT_TOL
    clock = {'zero': 0., 'before': first-2*ref.EVENT_TOL,
             'at': first, 'after': first+2*ref.EVENT_TOL}[relative_time]
    view, proxy, published, policy = static_projection(data, spec, clock, admit=True)
    expected = {0} | {c for c in range(1, 51) if data['demands'][0, c] > 0 and
                     data['reveal_time'][0, c] <= clock+ref.EVENT_TOL}
    assert {o.oid for o in view.orders} == set(published) == expected
    assert proxy.reveal_time[0, 0] == proxy.dataset['reveal_time'][0, 0] == view.orders[0].reveal == 0.
    assert set(view.accepted) == policy._accepted
    assert set(view.rejected) == policy._rejected
    assert set(view.accepted) | set(view.rejected) == expected-{0}
    if relative_time in ('zero', 'before'):
        assert expected == {0} and policy.admission_count == 0
    else:
        assert expected == set(range(7)) and policy.admission_count == 6
    assert data['reveal_time'][0, 0] == 1e6 and teacher.dataset_digest(data) == before


def test_depot_sentinel_future_invariance_in_features_choice_and_wait(model):
    rows, vehicles, args = synthetic_case()
    rows[0]['reveal'] = 1e6
    before = copy.deepcopy(rows)
    class HiddenCustomer(dict):
        def __getitem__(self, key):
            if key not in ('oid', 'reveal'):
                raise AssertionError('hidden customer field was read: '+key)
            return super().__getitem__(key)
    one = ref.project_public(rows+[HiddenCustomer(oid=7, reveal=1., x=999.)], vehicles, **args)
    two = ref.project_public(rows+[HiddenCustomer(oid=7, reveal=99., x=-999.),
                                   HiddenCustomer(oid=8, reveal=100., demand=999.)], vehicles, **args)
    assert one == two == synthetic_view()
    base = ref.plans_from_view(one)
    masks = ref.mask_pool(one, base)
    assert masks and masks == ref.mask_pool(two, base)
    np.testing.assert_array_equal(model.features(one, base, masks), model.features(two, base, masks))
    np.testing.assert_array_equal(model.score_masks(one, base, masks), model.score_masks(two, base, masks))
    selected1, log1 = ref.decide(one, model, time.monotonic()+10.)
    selected2, log2 = ref.decide(two, model, time.monotonic()+10.)
    assert selected1 == selected2 and log1 == log2 and log1['status'] == 'COMMIT'
    ready = replace(one.vehicles[0], status='ready', current_node=1)
    for clock, expected in ((0., ()), (1., (0,))):
        assert ref.wait_or_return(replace(one, clock=clock, booking_horizon=1.), ready) == expected
        assert ref.wait_or_return(replace(two, clock=clock, booking_horizon=1.), ready) == expected
    assert rows == before  # normalization must not mutate the source record


def test_future_customer_fields_are_not_copied_by_saa_projection():
    source, spec = saa_synthetic_dataset()
    source['reveal_time'][0, 0] = 1e6
    clock = float(source['reveal_time'][0, 1])
    changed = copy.deepcopy(source)
    hidden = changed['reveal_time'][0] > clock+ref.EVENT_TOL
    hidden[0] = False
    changed['coords'][0, hidden] = 999.
    changed['demands'][0, hidden] = 19.
    changed['temp_class'][0, hidden] = 2
    changed['initial_quality'][0, hidden] = 123.
    changed['reveal_time'][0, hidden] = 100.
    one = static_projection(source, spec, clock, admit=True)
    two = static_projection(changed, spec, clock, admit=True)
    assert one[0] == two[0]
    assert one[3].export_state() == two[3].export_state()
    for field in source:
        np.testing.assert_array_equal(one[1].dataset[field], two[1].dataset[field])


def test_malformed_depot_and_future_customer_still_rejected():
    rows, vehicles, args = synthetic_case()
    rows[0]['reveal'] = 1e6
    with pytest.raises(ValueError, match='duplicate/missing public identity'):
        ref.project_public(rows+[dict(rows[0])], vehicles, **args)
    with pytest.raises(ValueError, match='duplicate/missing public identity'):
        ref.project_public(rows[1:], vehicles, **args)
    valid = ref.project_public(rows, vehicles, **args)
    future = replace(valid.orders[1], oid=7, reveal=1.)
    with pytest.raises(ValueError, match='nonpublic/invalid order'):
        ref.validate_view(replace(valid, orders=valid.orders+(future,)))
    with pytest.raises(ValueError, match='public acceptance ledger mismatch'):
        ref.project_public(rows+[dict(rows[1], oid=7, reveal=1.)], vehicles,
                           **dict(args, accepted=args['accepted']+(7,)))
    with pytest.raises(ValueError, match='invalid clock/depot'):
        ref.validate_view(replace(valid, orders=(replace(valid.orders[0], reveal=-1.),)+valid.orders[1:]))


def test_registered_two_npz_t0_projection_only(monkeypatch):
    """Real files are byte-read only; engine/control execution is forbidden here."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError('real-source simulation/admission/continuation is not authorised')
    monkeypatch.setattr(StrictOnlineEnv, 'run', forbidden)
    monkeypatch.setattr(StrictOnlineEnv, 'run_resumed', forbidden)
    monkeypatch.setattr(teacher.PublicSaaReplanner, 'on_reveal', forbidden)
    monkeypatch.setattr(teacher.PublicSaaReplanner, 'plan', forbidden)
    manifest = json.loads((EXTENSION_ROOT/'configs/p2_pos01c_r0_20261019_v1/data_identity.json').read_text(encoding='utf-8'))
    hist = []
    for day in manifest['days']:
        if day['role'] == 'HIST_TRAIN':
            path = EXTENSION_ROOT/day['relative_path']
            assert file_hash(path) == day['file_sha256']
            with np.load(path, allow_pickle=False) as z:
                hist.extend(build_history({k: z[k] for k in z.files}))
    spec = teacher.SaaSpec(contract=make_c0_contract_v2(), history=tuple(tuple(d) for d in hist),
        booking_horizon=16., budget=186.06639500089943, capacity=50., tw_speed=1.5,
        K=4, n_neighbors=10, time_limit=6., cooling_share=4.452, reject_penalty=(5., 10., 15.),
        stream_id='p2_pos01c_smoke_20261019_v1', stream_seed=2026101903,
        shadow_mode='ls', energy_pricing='marginal', standby_orders=8.)
    records = []
    for day in manifest['days']:
        if day['role'] != 'SOURCE_TRAIN':
            continue
        path = EXTENSION_ROOT/day['relative_path']
        assert file_hash(path) == day['file_sha256']
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        assert data['reveal_time'][0, 0] == 1e6
        assert (data['reveal_time'][0, 1:] > 0).all()
        engine = StrictOnlineEnv(data, 50., 1.5, num_vehicles=15,
            coldchain_contract=spec.contract, booking_horizon=16.)
        policy = teacher.PublicSaaReplanner(spec)
        vehicles = [VehicleState(vehicle_id=i) for i in range(15)]
        served = np.zeros(51, bool)
        served[0] = True
        proxy, published = ref.saa_public_env(engine, 0, 0.)
        view = ref.view_from_saa(proxy, policy, 0., vehicles, served)
        assert published == (0,) and tuple(o.oid for o in view.orders) == (0,)
        assert view.orders[0].reveal == proxy.reveal_time[0, 0] == 0.
        assert all(not x for x in (view.accepted, view.rejected, view.served))
        assert policy.admission_count == 0 and len(view.vehicles) == 15
        assert data['reveal_time'][0, 0] == 1e6 and file_hash(path) == day['file_sha256']
        records.append({'day_id': day['day_id'], 'sha256': day['file_sha256'],
                        'public_ids': [0], 'public_depot_reveal': 0., 'raw_depot_reveal': 1e6,
                        'source_simulations': 0, 'continuations': 0})
    assert len(records) == 2
    print('\nREAD_ONLY_TWO_NPZ_T0', json.dumps(records, sort_keys=True))
