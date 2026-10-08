"""I3 parameter-update primitives, not a training/data-selection runner.

Use fresh single-step draws from the exact current actor. Teacher/shadow values
are detached. Frozen _cvrp leaves are excluded from BOTH optimizer gradients
and weight decay. The model update happens outside jit before checkpointing.
All optimizer/loss hyperparameters are supplied by a registered caller.
"""
import hashlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models'))
from maskco_scenario import MASK_TOKEN, VOCAB_SIZE
from i3_decision_objective import (scenario_set_log_prob, combined_loss,
                                   validate_draws)


def is_frozen(path):
    return any(str(part).startswith('_cvrp') for part in path)


def parameter_hash(model, *, frozen):
    """Hash array values/shape/dtype, never VariableState object bytes."""
    h = hashlib.sha256()
    fs = nnx.state(model, nnx.Param).flat_state()
    for path, variable in sorted(zip(fs.paths, fs.leaves), key=lambda kv: str(kv[0])):
        if is_frozen(path) != frozen:
            continue
        value = np.asarray(variable.value)
        h.update(str(path).encode())
        h.update(value.dtype.str.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def trainable_state(model):
    fs = nnx.state(model, nnx.Param).flat_state()
    leaves = {path: variable for path, variable in zip(fs.paths, fs.leaves)
              if not is_frozen(path)}
    if not leaves:
        raise ValueError('No trainable actor parameters')
    return nnx.State.from_flat_path(leaves)


def validate_actor_batch(model, batch, *, sampled_trainable_hash, sampled_frozen_hash):
    """Reject stale/off-policy batches; no unregistered importance weighting.

    This validates shapes/support/finite detached inputs, not teacher provenance.
    A formal runner must verify teacher record/data identity separately.
    """
    if (parameter_hash(model, frozen=False) != sampled_trainable_hash or
            parameter_hash(model, frozen=True) != sampled_frozen_hash):
        raise ValueError('Draws came from a different actor; fresh on-policy sampling required')
    x, cb, positions = map(np.asarray,
                          (batch['actor_x'], batch['actor_cb'], batch['mask_positions']))
    if x.ndim != 2 or positions.ndim != 2 or len(positions) != len(x) or cb.shape != (len(x),):
        raise ValueError('Invalid actor input/clock/mask positions')
    b, m = positions.shape
    if (positions.dtype.kind not in 'iu' or np.any(positions < 0) or
            np.any(positions >= x.shape[1]) or
            any(len(set(row.tolist())) != m for row in positions) or
            np.any(np.take_along_axis(x, positions, axis=1) != MASK_TOKEN) or
            np.any((x == MASK_TOKEN).sum(axis=1) != m)):
        raise ValueError('Use all and only distinct single-step MASK positions')
    validate_draws(batch['tokens'], batch['counts'], batch['support'], m, VOCAB_SIZE)
    if np.asarray(batch['tokens']).shape[0] != b:
        raise ValueError('Draw and actor batch size mismatch')
    losses, baseline = map(np.asarray, (batch['losses'], batch['prior_baseline']))
    if losses.shape != np.asarray(batch['tokens']).shape[:2] or baseline.shape != (b,):
        raise ValueError('Detached decision losses/baseline shape mismatch')
    if not np.isfinite(losses).all() or not np.isfinite(baseline).all():
        raise ValueError('Nonfinite detached decision loss/baseline')
    rx, ry, rn, rcb = map(np.asarray, (batch['recon_x'], batch['recon_y'],
                                     batch['recon_n'], batch['recon_cb']))
    if (rx.ndim != 2 or ry.shape != rx.shape or rn.shape != (len(rx),) or
            rcb.shape != rn.shape or np.any((rx == MASK_TOKEN).sum(axis=1) == 0)):
        raise ValueError('Reconstruction inputs require masked targets and matching labels')
    if (np.any(ry < 0) or np.any(ry >= VOCAB_SIZE) or np.any(rn < 0) or np.any(rn > m)):
        raise ValueError('Invalid reconstruction target/count labels')


def reconstruction_loss(model, batch):
    t, c = model(batch['recon_x'], clock_bin=batch['recon_cb'])
    mask = batch['recon_x'] == MASK_TOKEN
    ce = optax.softmax_cross_entropy_with_integer_labels(t, batch['recon_y'])
    token = jnp.where(mask, ce, 0).sum() / jnp.maximum(mask.sum(), 1)
    count = optax.softmax_cross_entropy_with_integer_labels(c, batch['recon_n']).mean()
    return token + count


def actor_objective(model, batch, reconstruction_weight):
    t, c = model(batch['actor_x'], clock_bin=batch['actor_cb'])
    p = batch['mask_positions']
    t = jnp.take_along_axis(t, p[..., None], axis=1)
    if c.shape[-1] != p.shape[-1] + 1:
        raise ValueError('No clamping: count head must match the generated slot capacity')
    lp = scenario_set_log_prob(t, c, batch['tokens'], batch['counts'], batch['support'])
    recon = reconstruction_loss(model, batch)
    return combined_loss(lp, batch['losses'], batch['prior_baseline'],
                         recon, reconstruction_weight)


def create_optimizer(model, *, learning_rate, weight_decay, max_grad_norm):
    if (not all(np.isfinite(v) for v in (learning_rate, weight_decay, max_grad_norm)) or
            learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0):
        raise ValueError('Optimizer settings must be explicit finite valid values')
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                     optax.adamw(learning_rate, weight_decay=weight_decay))
    state = trainable_state(model)
    return tx, tx.init(state), state


def make_gradient_step(tx, *, reconstruction_weight, use_jit=True):
    if not np.isfinite(reconstruction_weight) or reconstruction_weight < 0:
        raise ValueError('Invalid declared reconstruction weight')

    def compute(model, opt_state, trainable, batch):
        value, grads = nnx.value_and_grad(actor_objective)(model, batch, reconstruction_weight)
        fs = grads.flat_state()
        selected = set(trainable.flat_state().paths)
        grads = nnx.State.from_flat_path({p: v for p, v in zip(fs.paths, fs.leaves)
                                          if p in selected})
        norm = optax.global_norm(grads)
        updates, next_opt = tx.update(grads, opt_state, trainable)
        next_params = optax.apply_updates(trainable, updates)
        return value, norm, next_opt, next_params

    return nnx.jit(compute, donate_argnums=()) if use_jit else compute


def apply_gradient_step(model, compute, opt_state, trainable, batch, *,
                        sampled_trainable_hash, sampled_frozen_hash):
    """Validate -> compute -> reject nonfinite -> commit model OUTSIDE jit.

    Caller's baseline EMA updates only AFTER this step, using detached losses.
    Save the committed model, not a captured init state or an unreturned jit copy.
    """
    validate_actor_batch(model, batch, sampled_trainable_hash=sampled_trainable_hash,
                         sampled_frozen_hash=sampled_frozen_hash)
    current = trainable_state(model).flat_state()
    given = trainable.flat_state()
    if (list(current.paths) != list(given.paths) or
            any(not np.array_equal(np.asarray(a.value), np.asarray(b.value))
                for a, b in zip(current.leaves, given.leaves))):
        raise ValueError('Optimizer parameters are stale relative to the current actor')
    value, norm, next_opt, next_params = compute(model, opt_state, trainable,
                        {k: jnp.asarray(v) for k, v in batch.items()})
    if (not np.isfinite(float(value)) or not np.isfinite(float(norm)) or
            not all(np.isfinite(np.asarray(v.value)).all()
                    for v in next_params.flat_state().leaves) or
            not all(np.isfinite(np.asarray(v)).all()
                    for v in jax.tree_util.tree_leaves(next_opt))):
        raise ValueError('Nonfinite optimizer output; actor was not updated')
    nnx.update(model, next_params)
    if parameter_hash(model, frozen=True) != sampled_frozen_hash:
        raise RuntimeError('Frozen encoder changed; update is not valid')
    return {'loss': float(value), 'gradient_norm': float(norm)}, next_opt, next_params
