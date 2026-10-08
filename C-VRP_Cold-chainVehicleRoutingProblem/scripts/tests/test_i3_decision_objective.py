"""Exact tiny-distribution proof of I3 score-function gradient, no experiment data."""
import itertools
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from i3_decision_objective import (scenario_set_log_prob, validate_draws,
                                   score_function_loss, decision_loss,
                                   validate_decision_inputs)


def test_exact_gradient():
    # Count 0,1,2; each real token is either 1 or 2. Enumerate the entire support.
    samples, ns, rewards = [], [], []
    for n in range(3):
        for seq in itertools.product((1, 2), repeat=n):
            samples.append(list(seq) + [0] * (2 - n))
            ns.append(n)
            rewards.append(float((n - 1) ** 2 + seq.count(2)))
    tokens = jnp.asarray(samples, dtype=jnp.int32)[None, :, None, :]
    counts = jnp.asarray(ns, dtype=jnp.int32)[None, :, None]
    support = jnp.asarray([[True, True, True, False]])
    validate_draws(tokens, counts, support, 2, 4)
    rewards = jnp.asarray(rewards)[None, :]

    def lp(theta):
        tl = jnp.stack([theta[:4], theta[:4]], axis=0)[None]
        return scenario_set_log_prob(tl, theta[4:][None], tokens, counts, support)

    def exact_risk(theta):
        return jnp.sum(jnp.exp(lp(theta)) * rewards)

    theta = jnp.asarray([7.0, 0.2, -0.4, 99.0, 0.1, 0.6, -0.2])
    probabilities = jax.lax.stop_gradient(jnp.exp(lp(theta)))

    def surrogate(theta):
        # Exact weighted expectation substitutes Monte Carlo mean only in this proof.
        advantage = jax.lax.stop_gradient(rewards - 0.37)
        return jnp.sum(probabilities * advantage * lp(theta))

    grad_exact = np.asarray(jax.grad(exact_risk)(theta))
    grad_sf = np.asarray(jax.grad(surrogate)(theta))
    np.testing.assert_allclose(grad_sf, grad_exact, atol=2e-6, rtol=2e-5)
    assert np.isfinite(grad_sf).all() and np.linalg.norm(grad_sf) > 1e-3
    assert grad_sf[0] == 0 and grad_sf[3] == 0  # NULL/unsupported token logits.
    assert float(exact_risk(theta - 0.02 * grad_sf)) < float(exact_risk(theta))
    assert abs(float(jnp.exp(lp(theta)).sum()) - 1.0) < 1e-6

    # The K-scenario joint likelihood must sum scores across ALL K draws.
    duplicate_tokens = jnp.repeat(tokens, 2, axis=2)
    duplicate_counts = jnp.repeat(counts, 2, axis=2)
    tl = jnp.stack([theta[:4], theta[:4]], axis=0)[None]
    double = scenario_set_log_prob(tl, theta[4:][None], duplicate_tokens,
                                  duplicate_counts, support)
    np.testing.assert_allclose(double, 2 * lp(theta), atol=1e-6)
    print('I3 exact expected-risk gradient, legal support and K-set score PASS')


def test_baseline_and_bad_draws():
    # Baseline must not receive gradients; observed simulator losses are detached.
    logp = jnp.asarray([[-2.0, -4.0]])
    loss = jnp.asarray([[1.0, 3.0]])
    baseline = jnp.asarray([0.5])
    grad_l, grad_b = jax.grad(score_function_loss, argnums=(1, 2))(logp, loss, baseline)
    np.testing.assert_array_equal(grad_l, 0)
    np.testing.assert_array_equal(grad_b, 0)
    sf = jax.grad(score_function_loss)(logp, loss, baseline)
    np.testing.assert_allclose(sf, [[0.25, 1.25]])
    # Document the one-set/current-loss baseline zero-gradient failure mode.
    self_baseline = jax.grad(score_function_loss)(logp[:, :1], loss[:, :1], loss[:, 0])
    np.testing.assert_array_equal(self_baseline, 0)
    support = np.array([[True, True, False]])
    for tokens, counts in (([[[[0, 0]]]], [[[1]]]),
                           ([[[[2, 0]]]], [[[1]]]),
                           ([[[[1, 0]]]], [[[3]]])):
        try:
            validate_draws(np.array(tokens), np.array(counts), support, 2, 3)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid draw accepted')
    margins = decision_loss(jnp.asarray([20.0, -10.0]),
                           jnp.asarray([[30.0, -30.0], [30.0, -30.0]]),
                           jnp.asarray([True, False]), tau=10.0, weight_cap=100.0)
    assert float(margins[0, 0]) < float(margins[0, 1])
    np.testing.assert_array_equal(margins[1], 0)
    for bad in (np.nan, -np.inf):
        try:
            validate_decision_inputs([bad], [[1.0]], [False], [0.0])
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite ineligible teacher can cause inf*zero NaN')
    print('I3 detached baseline, forbidden draws and teacher eligibility PASS')


def test_model_graph_and_checkpoint():
    """Synthetic input through the real model class, no pretraining/data claims."""
    import tempfile
    import optax
    from flax import nnx
    scripts = Path(__file__).resolve().parents[1]
    for rel in ('models', 'evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(scripts / rel))
    from maskco_scenario import (MaskCOScenarioModel, MASK_TOKEN, PAD_TOKEN,
                                 VOCAB_SIZE, clock_support_mask, save_model, load_model)
    m = 200
    model = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(11))
    x = np.full((1, 2 * m), PAD_TOKEN, dtype=np.int32)
    x[0, :2] = [1, 37]
    x[0, 2:2 + m] = MASK_TOKEN
    x, cb = jnp.asarray(x), jnp.asarray([6], dtype=jnp.int32)
    tl, cl = model(x, clock_bin=cb)
    support = clock_support_mask(3.0)[None]
    allowed = np.flatnonzero(support[0] & (np.arange(VOCAB_SIZE) != 0))
    prob_t = np.asarray(jax.nn.softmax(tl[0, 2:2 + m][:, allowed], axis=-1), float)
    prob_n = np.asarray(jax.nn.softmax(cl[0]), float)
    # Normalize host floats as the deployed sampler does; use fixed synthetic RNG.
    prob_t /= prob_t.sum(-1, keepdims=True)
    prob_n /= prob_n.sum()
    rng = np.random.default_rng(19)
    draws = np.zeros((1, 2, 2, m), dtype=np.int32)
    counts = np.zeros((1, 2, 2), dtype=np.int32)
    for g in range(2):
        for k in range(2):
            counts[0, g, k] = rng.choice(m + 1, p=prob_n)
            for j in range(m):
                draws[0, g, k, j] = rng.choice(allowed, p=prob_t[j])
    validate_draws(draws, counts, support, m, VOCAB_SIZE)
    draws, counts, support = map(jnp.asarray, (draws, counts, support))

    def objective(model):
        t, c = model(x, clock_bin=cb)
        logp = scenario_set_log_prob(t[:, 2:2 + m], c, draws, counts, support)
        return score_function_loss(logp, jnp.asarray([[0.3, 1.2]]), jnp.asarray([0.2]))

    value, grads = nnx.value_and_grad(objective)(model)
    leaves = jax.tree_util.tree_leaves(grads.to_pure_dict())
    norm = float(np.sqrt(sum(float(jnp.sum(z * z)) for z in leaves)))
    assert np.isfinite(float(value)) and np.isfinite(norm) and norm > 0
    # Smoke update outside a donated/jitted model, matching the corrected save path.
    state = nnx.state(model, nnx.Param)
    tx = optax.sgd(1e-5)
    updates, _ = tx.update(grads, tx.init(state), state)
    nnx.update(model, optax.apply_updates(state, updates))
    after, after_c = model(x, clock_bin=cb)
    assert not np.array_equal(np.asarray(cl), np.asarray(after_c))
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'model.bin')
        save_model(model, path)
        restored = MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(11))
        load_model(restored, path)
        restored_t, restored_c = restored(x, clock_bin=cb)
        np.testing.assert_allclose(restored_t, after, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(restored_c, after_c, atol=2e-6, rtol=2e-6)
    print('I3 real-model-class synthetic gradient/update/checkpoint roundtrip PASS')


if __name__ == '__main__':
    test_exact_gradient()
    test_baseline_and_bad_draws()
    test_model_graph_and_checkpoint()
