"""Tiny teacher -> fresh actor -> real shadow loss -> update -> saved model.

Controlled four-order fixture, random scenario actor and explicit TEST settings.
Not a formal training runner, selected model or positive online experiment.
"""
from pathlib import Path
import sys
import tempfile

import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
for folder in ('tests', 'training', 'models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from test_i3_counterfactual_teacher import fixture, replay
from i3_counterfactual_teacher import scenario_margin, teacher_preference
from i3_actor_draws import draw_single_step_sets, validated_actor_arrays
from i3_decision_objective import decision_loss, validate_decision_inputs
from i3_optimizer import create_optimizer, make_gradient_step, apply_gradient_step, parameter_hash
from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools,
                            MASK_TOKEN, order_to_token, save_model, load_model)
from scenario_saa import build_history
from run_exp_reserve import BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT


def test_real_teacher_step():
    ds, contract, factory, _, records, _ = fixture()
    record = records[0]  # Pre-fixed ordinal, not selected by observed teacher gain.
    branches = [replay(record, ds, contract, factory, accept) for accept in (True, False)]
    teacher = teacher_preference(*branches, minimum_gap=5)
    assert teacher['decision_eligible']
    model = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(343))
    # Test-only small-count head prevents a tiny fixture from simulating hundreds
    # of decoded future orders. Sampling is still from the CURRENT full 201 head;
    # neither draw tracing nor objective clamps counts or edits logits.
    model.head_count.kernel.value = jnp.zeros_like(model.head_count.kernel.value)
    model.head_count.bias.value = jnp.full_like(model.head_count.bias.value, -10.).at[2].set(10.)
    pools = HistoryPools(factory().sampler.history)
    sampler = MaskCOScenarioSampler(model, pools)
    tx, opt_state, trainable = create_optimizer(model, learning_rate=1e-5,
                                               weight_decay=0., max_grad_norm=1.)
    # First update has no reconstruction term: nonzero gradient must come from
    # real teacher/shadow losses through newly sampled scenario likelihood.
    compute = make_gradient_step(tx, reconstruction_weight=0., use_jit=True)
    packet = draw_single_step_sets(sampler, record.visible, np.random.default_rng(507),
                                   independent_sets=2, K=2)
    margins = np.array([[scenario_margin(record, ds, factory, contract, BOOKING_HORIZON,
                         SPEED_KMH / KM_PER_UNIT, scenes)['margin']
                         for scenes in packet['scenarios']]], dtype=np.float32)
    delta = np.array([teacher['loss_teacher_margin']], dtype=np.float32)
    eligible = np.array([teacher['decision_eligible']], dtype=bool)
    prior = np.array([0.], dtype=np.float32)  # Before observing this step's draws.
    validate_decision_inputs(delta, margins, eligible, prior)
    losses = np.asarray(decision_loss(delta, margins, eligible, tau=20., weight_cap=100.),
                        dtype=np.float32)
    arrays = validated_actor_arrays(sampler, packet)
    # Private true future is used ONLY for detached reconstruction labels. It is
    # never passed to draw_single_step_sets/actor conditioning.
    future = [order_to_token(o) for o in build_history(ds)[0]
              if o.reveal > record.visible.clock + 1e-6]
    target = np.zeros_like(arrays['actor_x'])
    positions = arrays['mask_positions'][0]
    target[0, positions[:len(future)]] = future
    batch = dict(arrays, losses=losses, prior_baseline=prior,
                 decision_eligible=eligible,
                 recon_x=arrays['actor_x'].copy(), recon_y=target,
                 recon_n=np.array([len(future)], dtype=np.int32),
                 recon_cb=arrays['actor_cb'].copy())
    initial = packet['sampled_trainable_hash']
    stats, _, _ = apply_gradient_step(model, compute, opt_state, trainable, batch,
        sampled_trainable_hash=initial, sampled_frozen_hash=packet['sampled_frozen_hash'])
    assert stats['gradient_norm'] > 0 and np.isfinite(stats['loss']), stats
    updated = parameter_hash(model, frozen=False)
    assert updated != initial
    # A lagged baseline is computed only AFTER the completed update.
    next_prior = .1 * float(losses.mean())
    assert np.isfinite(next_prior)
    before = model(jnp.asarray(arrays['actor_x']), clock_bin=jnp.asarray(arrays['actor_cb']))
    with tempfile.TemporaryDirectory() as temp:
        path = str(Path(temp) / 'updated.bin')
        save_model(model, path)
        restored = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(99))
        load_model(restored, path)
        assert parameter_hash(restored, frozen=False) == updated
        for a, b in zip(before, restored(jnp.asarray(arrays['actor_x']),
                                         clock_bin=jnp.asarray(arrays['actor_cb']))):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    assert np.all(batch['recon_x'] == packet['actor_x'])
    assert np.all(batch['recon_x'][0, positions] == MASK_TOKEN)
    print('ALL PASS: real tiny-fixture teacher/shadow losses reach nonzero single-step actor gradient (recon weight=0), committed parameters and saved/reloaded outputs; no formal training or online-gain evidence')
    print('TEST ONLY: teacher_delta=%s shadow_margins=%s gradient_norm=%s' %
          (float(delta[0]), margins.tolist(), stats['gradient_norm']))


if __name__ == '__main__':
    test_real_teacher_step()
