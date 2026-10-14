"""Fixed-schedule training tests on a controlled 4-order physical fixture."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
for folder in ('tests', 'training', 'models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from test_i3_counterfactual_teacher import fixture, replay, check_raises
from i3_counterfactual_teacher import teacher_preference
from i3_training_engine import TrainingSettings, train_fixed_schedule, reconstruction_example
from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools,
                            save_model, load_model)
from i3_optimizer import parameter_hash
from scenario_saa import build_history
from run_exp_reserve import BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT


def actor():
    model = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(343))
    model.head_count.kernel.value = jnp.zeros_like(model.head_count.kernel.value)
    model.head_count.bias.value = jnp.full_like(model.head_count.bias.value, -10.).at[2].set(10.)
    return model


def test_engine():
    ds, contract, factory, _, records, _ = fixture(infeasible=True, terminal=True)
    assert len(records) == 4 and records[-1].visible.clock == 16.
    entries = []
    for ordinal, record in enumerate(records):
        branches = [replay(record, ds, contract, factory, accept) for accept in (True, False)]
        entries.append({'day_id': 0, 'ordinal': ordinal, 'record': record,
                        'branches': branches,
                        'preference': teacher_preference(*branches, minimum_gap=5)})
    history = {0: build_history(ds)[0]}
    settings = TrainingSettings(steps=4, K=2, independent_sets=2, learning_rate=1e-5,
        weight_decay=0., max_grad_norm=1., reconstruction_weight=.1, tau=20.,
        weight_cap=100., baseline_decay=.9, schedule_seed=23, actor_seed=507,
        label_seed=71, objective='decision_reconstruction')
    model = actor()
    sampler = MaskCOScenarioSampler(model, HistoryPools(factory().sampler.history))
    initial = parameter_hash(model, frozen=False)
    result = train_fixed_schedule(sampler, entries, ds, history, lambda day: factory,
        contract, BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT, settings,
        minimum_teacher_gap=5)
    assert result['steps_completed'] == 4
    assert set(r['ordinal'] for r in result['logs']) == set(range(4))
    assert result['initial_trainable_hash'] == initial != result['final_trainable_hash']
    for row in result['logs']:
        assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        if not row['decision_eligible'] or not row['future_support']:
            assert not row['actor_update'] and row['estimated_margins'] is None
            assert row['prior_baseline'] == row['next_baseline']
            assert row['trainable_before'] != row['trainable_after']
    assert sum(not r['teacher_valid'] for r in result['logs']) == 1
    assert sum(not r['future_support'] for r in result['logs']) == 1
    assert any(r['actor_update'] for r in result['logs'])
    with tempfile.TemporaryDirectory() as temp:
        path = str(Path(temp) / 'trained.bin')
        save_model(model, path)
        restored = actor()
        load_model(restored, path)
        assert parameter_hash(restored, frozen=False) == result['final_trainable_hash']
    # Separate reconstruction-only control shares init/schedule/step count.
    control = MaskCOScenarioSampler(actor(), HistoryPools(factory().sampler.history))
    c = train_fixed_schedule(control, entries, ds, history, lambda day: factory,
        contract, BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT,
        replace(settings, objective='reconstruction_control'), minimum_teacher_gap=5)
    assert c['initial_trainable_hash'] == initial
    assert [r['ordinal'] for r in c['logs']] == [r['ordinal'] for r in result['logs']]
    assert all(not r['actor_update'] for r in c['logs'])
    assert all(r['next_baseline'] == 0 for r in c['logs'])
    corrupt = deepcopy(entries)
    corrupt[0]['preference']['loss_teacher_margin'] += 1
    check_raises(lambda: train_fixed_schedule(control, corrupt, ds, history,
        lambda day: factory, contract, BOOKING_HORIZON, SPEED_KMH / KM_PER_UNIT,
        settings, minimum_teacher_gap=5))
    # Future data changes target labels/count, never model conditioning.
    a = reconstruction_example(control, records[0].visible, history[0], np.random.default_rng(71))
    b = reconstruction_example(control, records[0].visible, history[0][:-1], np.random.default_rng(71))
    np.testing.assert_array_equal(a['recon_x'], b['recon_x'])
    assert a['recon_n'][0] == b['recon_n'][0] + 1
    print('ALL PASS: all-state fixed training schedule, real teacher actor updates, invalid/terminal reconstruction-only, lagged EMA, matched-init reconstruction control, checkpoint reload, label guard and causal reconstruction input')


if __name__ == '__main__':
    test_engine()
