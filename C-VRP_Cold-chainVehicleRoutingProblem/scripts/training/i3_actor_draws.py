"""Single-step deployment-matched draws with raw latent likelihood bookkeeping.

Existing sampler returns decoded/filtered scenarios only. REINFORCE must retain
the ORIGINAL sampled count/tokens before any decoder filtering. History-pool
draws are theta-independent; latent likelihood remains valid after filtering.
No remasking, teacher/private environment input or model update is performed.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from maskco_scenario import (MASK_TOKEN, PAD_TOKEN, NULL_TOKEN, VOCAB_SIZE, M_MAX,
                            clock_support_mask, clock_bin_of, _norm_probs)
from i3_optimizer import parameter_hash
from i3_decision_objective import validate_draws

ACTOR_FIELDS = ('actor_x', 'actor_cb', 'mask_positions', 'tokens', 'counts', 'support')


def draw_identity(packet):
    """Bind conditioning input and raw draws together, not just actor weights."""
    h = hashlib.sha256()
    for key in ACTOR_FIELDS:
        value = np.asarray(packet[key])
        h.update(key.encode())
        h.update(value.dtype.str.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def validated_actor_arrays(sampler, packet):
    """Check the sealed packet after teacher scoring, before an optimizer step.

    Returned copies may be combined with detached losses/reconstruction labels.
    A formal runner must call this check, not rebuild N from decoded lengths or
    attach old draws to a newly edited conditioning state.
    """
    if (draw_identity(packet) != packet.get('draw_hash') or
            parameter_hash(sampler.model, frozen=False) != packet['sampled_trainable_hash'] or
            parameter_hash(sampler.model, frozen=True) != packet['sampled_frozen_hash']):
        raise ValueError('Sampling packet input/draws/actor changed')
    return {key: np.array(packet[key], copy=True) for key in ACTOR_FIELDS}


def draw_single_step_sets(sampler, visible, rng, *, independent_sets, K):
    """Return one-state actor batch fragment and G sets of decoded K scenarios.

    For the same visible input/rng, each set exactly follows existing
    sample_from_logits. Count logits are NOT silently clamped. Caller seals
    pool/training provenance and records this actor's fresh weight hashes.
    """
    if sampler.iterative or sampler.m_max != M_MAX:
        raise ValueError('Only registered full-capacity single-step likelihood is implemented')
    if (not isinstance(independent_sets, int) or independent_sets <= 0 or
            not isinstance(K, int) or K <= 0):
        raise ValueError('G/K must be explicit positive integers')
    train_hash = parameter_hash(sampler.model, frozen=False)
    frozen_hash = parameter_hash(sampler.model, frozen=True)
    prefix = sampler._visible_tokens(visible)
    clock, m = float(visible.clock), sampler.m_max
    cb = clock_bin_of(clock)
    support = clock_support_mask(clock)
    if not np.any(support & (np.arange(VOCAB_SIZE) != NULL_TOKEN)):
        raise ValueError('Terminal state has no future support; do not create a decision draw')
    tl, cl = sampler._forward_masks(prefix, clock_bin=cb)
    if tl.shape != (m, VOCAB_SIZE) or cl.shape != (m + 1,):
        raise ValueError('Count/token output shape mismatch; no clamping or truncation')
    tl = np.where(support[None], np.asarray(tl), -1e9)
    tl = np.where((np.arange(VOCAB_SIZE) != NULL_TOKEN)[None], tl, -1e9)
    pt = _norm_probs(jax.nn.softmax(tl, axis=-1))
    pn = _norm_probs(jax.nn.softmax(np.asarray(cl)[None], axis=-1))[0]
    tokens = np.zeros((1, independent_sets, K, m), dtype=np.int32)
    counts = np.zeros((1, independent_sets, K), dtype=np.int32)
    sets = []
    for g in range(independent_sets):
        scenarios = []
        for k in range(K):
            n = int(np.argmax(rng.multinomial(1, pn)))
            indices = rng.multinomial(1, pt).argmax(-1)
            counts[0, g, k], tokens[0, g, k] = n, indices
            scenario = []
            for slot in range(n):
                order = sampler.pools.decode_token(int(indices[slot]), rng, clock=clock)
                order = replace(order, oid=-(1_000_000 + slot + 1))
                if order.reveal > clock + 1e-6:
                    scenario.append(order)
            scenarios.append(scenario)
        sets.append(scenarios)
    validate_draws(tokens, counts, support[None], m, VOCAB_SIZE)
    if (parameter_hash(sampler.model, frozen=False) != train_hash or
            parameter_hash(sampler.model, frozen=True) != frozen_hash):
        raise ValueError('Actor changed during sampling')
    x = np.full((1, 2 * m), PAD_TOKEN, dtype=np.int32)
    x[0, :len(prefix)] = prefix
    x[0, len(prefix):len(prefix) + m] = MASK_TOKEN
    packet = {'actor_x': x, 'actor_cb': np.array([cb], dtype=np.int32),
            'mask_positions': np.arange(len(prefix), len(prefix) + m, dtype=np.int32)[None],
            'tokens': tokens, 'counts': counts, 'support': support[None],
            'sampled_trainable_hash': train_hash, 'sampled_frozen_hash': frozen_hash,
            'scenarios': sets,
            'decoded_counts': np.array([[list(map(len, s)) for s in sets]], dtype=np.int32)}
    packet['draw_hash'] = draw_identity(packet)
    return packet
