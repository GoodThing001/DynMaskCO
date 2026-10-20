"""Structural and real small-model checks on fixed public instances."""
import importlib
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import pytest

EXT = Path(__file__).resolve().parents[2]
MODULES = EXT / 'scripts/static_delivery'
sys.path.insert(0, str(MODULES))
from rrp_data import SOURCE, load_instance, load_reference


def api(name):
    assert (MODULES / (name + '.py')).is_file(), 'Learning module missing'
    return importlib.import_module(name)


def case():
    x = load_instance(SOURCE / 'Datasets/D1/D1-n4-np33.json')
    r = load_reference(SOURCE / 'Solutions/D1/sol-D1-n4-np33.txt')
    return x, r


def test_directed_cycle_retains_edges_without_subtours():
    scores = np.array([[0, 1, 2, 9], [2, 0, 9, 1], [9, 1, 0, 2], [1, 9, 2, 0]], float)
    route = api('rrp_reconstruct').build_cycle(scores, {(0, 1), (2, 3)})
    edges = set(zip(route, route[1:] + route[:1]))
    assert {(0, 1), (2, 3)} <= edges
    assert route[0] == 0 and set(route) == set(range(4))


def test_invalid_kept_subtour_rejected():
    with pytest.raises(ValueError, match='subtour'):
        api('rrp_reconstruct').build_cycle(np.zeros((4, 4)), {(1, 2), (2, 1)})


def test_wait_mask_has_no_held_out_wait_input_leak():
    x, r = case()
    learning = api('rrp_learning')
    first = learning.make_batch(x, r.route, [0, 123, 456, 789], kept_edges={(0, r.route[1])})
    second = learning.make_batch(x, r.route, [0, 999, 888, 777], kept_edges={(0, r.route[1])})
    np.testing.assert_array_equal(first['features'], second['features'])
    assert first['distance_bias'].shape == (1, 4, 4)
    assert first['masked_rows'].sum() == 3


def test_geometry_groups_keep_d1_d2_variants_together():
    learning = api('rrp_learning')
    groups = learning.geometry_groups()
    assert sum(len(v) for v in groups.values()) == 30
    containing = [v for v in groups.values() if 'D1-n4-np33' in v][0]
    assert 'D2-n4-np33' in containing
    with pytest.raises(ValueError, match='geometry'):
        learning.validate_split({'train': ['D1-n4-np33'], 'test': ['D2-n4-np33']})


def test_model_inputs_include_speed_climate_and_objective_conditions():
    x, r = case()
    other = load_instance(SOURCE / 'Datasets/D2/D2-n4-np33.json')
    learning = api('rrp_learning')
    a = learning.make_batch(x, r.route, r.waits_s, set(), objective='with_wage', profile='paper')
    b = learning.make_batch(other, r.route, r.waits_s, set(), objective='with_wage', profile='paper')
    c = learning.make_batch(x, r.route, r.waits_s, set(), objective='without_wage', profile='paper')
    assert not np.array_equal(a['features'], b['features'])
    assert not np.array_equal(a['features'], c['features'])


def test_real_gradient_update_direction_head_and_checkpoint(tmp_path):
    x, r = case()
    learning = api('rrp_learning')
    model_api = api('rrp_model')
    model = model_api.make_model('smoke_cpu', seed=888)
    batch = learning.make_batch(x, r.route, r.waits_s, kept_edges=set())
    scores, waits = learning.predict(model, batch)
    assert scores.shape == (4, 4) and waits.shape == (4,)
    assert np.max(np.abs(scores - scores.T)) > 1e-6
    before = learning.parameter_digest(model)
    trainer = learning.Trainer(model, learning_rate=1e-3, weight_decay=1e-2)
    metrics = trainer.step(batch)
    assert np.isfinite(metrics['loss'])
    assert metrics['updated_parameter_leaves'] > 0
    assert learning.parameter_digest(model) != before
    expected = learning.predict(model, batch)
    learning.save_checkpoint(tmp_path / 'checkpoint', trainer, {'scope': 'functional_smoke'})
    loaded, metadata = learning.load_checkpoint(tmp_path / 'checkpoint')
    actual = learning.predict(loaded.model, batch)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert metadata['scope'] == 'functional_smoke'
    candidate = api('rrp_reconstruct').reconstruct(loaded.model, x, list(r.route), {(0, r.route[1])}, list(r.waits_s))
    edges = set(zip(candidate['route'], candidate['route'][1:] + candidate['route'][:1]))
    assert (0, r.route[1]) in edges


@pytest.mark.parametrize('field,value', [('learning_rate', float('nan')), ('weight_decay', float('inf'))])
def test_optimizer_nonfinite_configuration_is_rejected(field, value):
    settings = dict(learning_rate=1e-3, weight_decay=1e-2)
    settings[field] = value
    with pytest.raises(ValueError, match='optimizer'):
        api('rrp_learning').Trainer(api('rrp_model').make_model('smoke_cpu', 888), **settings)


@pytest.mark.parametrize('filename', ['weights.npz', 'optimizer.npz'])
def test_hash_valid_nonfinite_checkpoint_leaf_is_rejected(tmp_path, filename):
    import hashlib
    import json
    learning = api('rrp_learning')
    trainer = learning.Trainer(api('rrp_model').make_model('smoke_cpu', 888), 1e-3, 1e-2)
    destination = tmp_path / 'ckpt'
    learning.save_checkpoint(destination, trainer, {})
    with np.load(destination / filename) as source:
        arrays = {key: source[key] for key in source.files}
    key = next(k for k, a in arrays.items() if np.issubdtype(a.dtype, np.floating) and a.size)
    arrays[key].flat[0] = np.nan
    np.savez(destination / filename, **arrays)
    metadata = json.loads((destination / 'checkpoint.json').read_text())
    metadata['files'][filename] = hashlib.sha256((destination / filename).read_bytes()).hexdigest()
    (destination / 'checkpoint.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Nonfinite'):
        learning.load_checkpoint(destination)
