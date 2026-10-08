"""Update/save/freeze tests, synthetic losses only; no I3 method-gain evidence."""
import argparse
from pathlib import Path
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
for folder in ('', 'models', 'training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_optimizer import (parameter_hash, create_optimizer, make_gradient_step,
                         apply_gradient_step)
from maskco_scenario import (MaskCOScenarioModel, MASK_TOKEN, PAD_TOKEN,
                            VOCAB_SIZE, clock_support_mask, save_model, load_model)


class TinyCVRP(nnx.Module):
    """Stub only for the real PretrainedEncoder branch's freeze/update interface."""
    def __init__(self):
        self.proj = nnx.Linear(3, 512, rngs=nnx.Rngs(23))

    def encode(self, x, attn_options=None):
        return self.proj(x)


def raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected invalid batch to be rejected')


def build_batch(model):
    m = 200
    x = np.full((1, 2 * m), PAD_TOKEN, dtype=np.int32)
    x[0, :2] = [1, 37]
    x[0, 2:2 + m] = MASK_TOKEN
    cb = np.array([6], dtype=np.int32)
    tl, cl = model(jnp.asarray(x), clock_bin=jnp.asarray(cb))
    support = clock_support_mask(3.0)[None]
    allowed = np.flatnonzero(support[0] & (np.arange(VOCAB_SIZE) != 0))
    pt = np.asarray(jax.nn.softmax(tl[0, 2:2 + m][:, allowed], axis=-1), dtype=float)
    pn = np.asarray(jax.nn.softmax(cl[0]), dtype=float)
    pt /= pt.sum(-1, keepdims=True)
    pn /= pn.sum()
    rng = np.random.default_rng(919)
    tokens = np.zeros((1, 2, 2, m), dtype=np.int32)
    counts = np.zeros((1, 2, 2), dtype=np.int32)
    for g in range(2):
        for k in range(2):
            counts[0, g, k] = rng.choice(m + 1, p=pn)
            for slot in range(m):
                tokens[0, g, k, slot] = rng.choice(allowed, p=pt[slot])
    y = np.zeros_like(x)
    y[0, 2:5] = allowed[:3]
    return {'actor_x': x, 'actor_cb': cb,
            'mask_positions': np.arange(2, 2 + m, dtype=np.int32)[None],
            'tokens': tokens, 'counts': counts, 'support': support,
            'losses': np.array([[.3, 1.2]], dtype=np.float32),
            'prior_baseline': np.array([.2], dtype=np.float32),
            'recon_x': x.copy(), 'recon_y': y,
            'recon_n': np.array([3], dtype=np.int32), 'recon_cb': cb.copy()}


def test_updates(cvrp_factory, use_jit):
    model = MaskCOScenarioModel(dim=16, arm='pretrained', cvrp_model=cvrp_factory(),
                               rngs=nnx.Rngs(11))
    frozen = parameter_hash(model, frozen=True)
    initial = parameter_hash(model, frozen=False)
    batch = build_batch(model)
    tx, opt_state, trainable = create_optimizer(model, learning_rate=1e-5,
                                               weight_decay=.3, max_grad_norm=1)
    compute = make_gradient_step(tx, reconstruction_weight=.1, use_jit=use_jit)
    stale_trainable = trainable
    report, opt_state, trainable = apply_gradient_step(model, compute, opt_state, trainable,
        batch, sampled_trainable_hash=initial, sampled_frozen_hash=frozen)
    assert report['gradient_norm'] > 0 and np.isfinite(report['loss']), report
    assert parameter_hash(model, frozen=True) == frozen
    updated = parameter_hash(model, frozen=False)
    assert updated != initial
    # Saved old draws cannot be reused after changing their actor distribution.
    raises(lambda: apply_gradient_step(model, compute, opt_state, trainable, batch,
                       sampled_trainable_hash=initial, sampled_frozen_hash=frozen))
    fresh = build_batch(model)
    raises(lambda: apply_gradient_step(model, compute, opt_state, stale_trainable, fresh,
                       sampled_trainable_hash=updated, sampled_frozen_hash=frozen))
    bad = dict(fresh, losses=np.array([[float('nan'), 1]], dtype=np.float32))
    raises(lambda: apply_gradient_step(model, compute, opt_state, trainable, bad,
                       sampled_trainable_hash=updated, sampled_frozen_hash=frozen))
    duplicate = fresh['mask_positions'].copy()
    duplicate[0, 1] = duplicate[0, 0]
    raises(lambda: apply_gradient_step(model, compute, opt_state, trainable,
             dict(fresh, mask_positions=duplicate), sampled_trainable_hash=updated,
             sampled_frozen_hash=frozen))
    # The exact same-actor second step uses newly sampled scenes and prior-step baseline.
    _, opt_state, trainable = apply_gradient_step(model, compute, opt_state, trainable,
        fresh, sampled_trainable_hash=updated, sampled_frozen_hash=frozen)
    before_save = model(jnp.asarray(fresh['actor_x']), clock_bin=jnp.asarray(fresh['actor_cb']))
    final_hash = parameter_hash(model, frozen=False)
    with tempfile.TemporaryDirectory() as temp:
        path = str(Path(temp) / 'updated.bin')
        save_model(model, path)
        # Reload into a new scenario model. The frozen encoder is unchanged and
        # may be shared; the learnable adapter/heads are independently initialized.
        restored = MaskCOScenarioModel(dim=16, arm='pretrained', cvrp_model=model.encoder._cvrp,
                                      rngs=nnx.Rngs(99))
        load_model(restored, path)
        assert parameter_hash(restored, frozen=False) == final_hash
        assert parameter_hash(restored, frozen=True) == frozen
        after_load = restored(jnp.asarray(fresh['actor_x']), clock_bin=jnp.asarray(fresh['actor_cb']))
        for a, b in zip(before_save, after_load):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    print('PASS: frozen encoder/AdamW isolation, fresh on-policy draws, stale optimizer guard, finite/position guards, outside-jit commit and checkpoint roundtrip; jit=' + str(use_jit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvrp-ckpt')
    args = parser.parse_args()
    if args.cvrp_ckpt:
        from mpre import load_cvrp_model
        cvrp, _, _ = load_cvrp_model(args.cvrp_ckpt)
        test_updates(lambda: cvrp, use_jit=True)
        print('ALL PASS: actual frozen CVRP checkpoint branch, synthetic I3 losses; no training-history or online-gain claim')
    else:
        test_updates(TinyCVRP, use_jit=False)
        test_updates(TinyCVRP, use_jit=True)
        print('ALL PASS: stub CVRP encoder only; actual checkpoint branch not verified in this invocation')
