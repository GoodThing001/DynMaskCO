"""Deployment sampling parity and preservation of pre-filter latent counts."""
from copy import deepcopy
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
for folder in ('training', 'models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_actor_draws import draw_single_step_sets, validated_actor_arrays
from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler,
    HistoryPools, VOCAB_SIZE, M_MAX)
from scenario_saa import VisibleSnapshot, ScenarioOrder


class FixedModel(nnx.Module):
    def __call__(self, tokens, clock_bin=None):
        b, length = tokens.shape
        # Time bin 6: [3.0,3.5), class/spot/demand bins all zero.
        token = 1 + 6 * 3 * 3 * 2
        t = jnp.full((b, length, VOCAB_SIZE), -1000.)
        t = t.at[:, :, token].set(0.)
        c = jnp.full((b, M_MAX + 1), -1000.)
        c = c.at[:, 3].set(0.)
        return t, c


def visible(clock):
    order = ScenarioOrder(1, 1., .25, .25, 1.5, 0, 0., 4., .05)
    return VisibleSnapshot(clock, (order,), frozenset(), frozenset(), (),
                           0., 710., 16., 50.)


def test_parity(model, clock):
    snap = visible(clock)
    pools = HistoryPools([[snap.orders[0]]])
    sampler = MaskCOScenarioSampler(model, pools)
    # G sequential independently drawn K-sets, same RNG sequence as deployment.
    a, b = np.random.default_rng(503), np.random.default_rng(503)
    traced = draw_single_step_sets(sampler, snap, a, independent_sets=2, K=3)
    expected = [sampler.sample(snap, b, 3) for _ in range(2)]
    assert traced['scenarios'] == expected
    assert a.integers(1 << 30) == b.integers(1 << 30)  # no extra random draw consumed
    assert traced['counts'].shape == (1, 2, 3)
    arrays = validated_actor_arrays(sampler, traced)
    assert not np.shares_memory(arrays['counts'], traced['counts'])
    changed = deepcopy(traced)
    changed['actor_x'][0, 0] += 1
    try:
        validated_actor_arrays(sampler, changed)
    except ValueError:
        pass
    else:
        raise AssertionError('Raw draws must remain bound to their original conditioning input')
    return traced


if __name__ == '__main__':
    test_parity(MaskCOScenarioModel(dim=16, arm='random', rngs=nnx.Rngs(33)), 3.)
    edge = test_parity(FixedModel(), 3.4999999)
    np.testing.assert_array_equal(edge['counts'], 3)
    np.testing.assert_array_equal(edge['decoded_counts'], 0)
    # Rebuilding N from decoded scenario length would attach the wrong likelihood.
    assert np.any(edge['counts'] != edge['decoded_counts'])
    sampler = MaskCOScenarioSampler(FixedModel(), HistoryPools([list(visible(1).orders)]),
                                    iterative=True)
    try:
        draw_single_step_sets(sampler, visible(3), np.random.default_rng(0),
                              independent_sets=1, K=1)
    except ValueError:
        pass
    else:
        raise AssertionError('Iteration needs its own full-path likelihood')
    print('ALL PASS: exact deployed scenario/RNG parity, pre-filter latent count/token preservation, input/draw binding and copied batch, remasking rejected')
