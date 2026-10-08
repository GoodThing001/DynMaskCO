"""I3 score-function objective kernel, not an implemented training runner.

Matches the single-step deployed count + non-NULL, clock-supported token
distribution. Draws are [state, independent set, scenario, slot]. An ensemble
decision loss is assigned to the entire K-scenario set. History-pool decoding
is independent of theta, so its random draws need no score-function term.

This module does not build teachers, read evaluation data, update parameters
or select any protocol hyperparameters. In particular it does not implement
the likelihood of multi-round remasking; iterative I3 training must not use it.
"""
import jax
import jax.numpy as jnp


def scenario_set_log_prob(token_logits, count_logits, tokens, counts, support):
    """Return [B,G] log probability of G independently sampled K-scenario sets.

    token_logits: [B,M,V], including NULL at index 0.
    count_logits: [B,M+1], counts 0..M; no clamp of a larger count head allowed.
    tokens: [B,G,K,M]; only slots j<count contribute, unused draws marginalized.
    counts: [B,G,K]. support: [B,V] boolean deployment time support.
    The caller must validate active tokens/counts before entering a jit step.
    """
    if token_logits.ndim != 3 or tokens.ndim != 4 or counts.ndim != 3:
        raise ValueError('Expected logits [B,M,V], tokens [B,G,K,M], counts [B,G,K]')
    b, m, v = token_logits.shape
    if (tokens.shape[0] != b or tokens.shape[-1] != m or
            counts.shape != tokens.shape[:-1] or
            count_logits.shape != (b, m + 1) or support.shape != (b, v)):
        raise ValueError('Count head/support/draw shape mismatch; no silent clamping')
    allowed = jnp.asarray(support, dtype=bool).at[:, 0].set(False)
    # True -inf produces exactly zero mass for impossible tokens. Every row
    # must have non-NULL support (validated by validate_draws before jit).
    tl = jnp.where(allowed[:, None, :], token_logits, -jnp.inf)
    log_tok = jax.nn.log_softmax(tl, axis=-1)
    log_cnt = jax.nn.log_softmax(count_logits, axis=-1)
    g, k = tokens.shape[1:3]
    expanded = jnp.broadcast_to(log_tok[:, None, None, :, :], (b, g, k, m, v))
    picked = jnp.take_along_axis(expanded, tokens[..., None], axis=-1)[..., 0]
    active = jnp.arange(m)[None, None, None, :] < counts[..., None]
    token_lp = jnp.where(active, picked, 0.0).sum(axis=-1)
    cnt_expanded = jnp.broadcast_to(log_cnt[:, None, None, :], (b, g, k, m + 1))
    count_lp = jnp.take_along_axis(cnt_expanded, counts[..., None], axis=-1)[..., 0]
    return (count_lp + token_lp).sum(axis=-1)  # SUM over K, not mean over K.


def validate_draws(tokens, counts, support, m, vocab_size):
    """Host-side fail-closed checks; NULL padding is legal only in inactive slots."""
    import numpy as np
    tokens, counts, support = map(np.asarray, (tokens, counts, support))
    if not np.issubdtype(tokens.dtype, np.integer) or \
            not np.issubdtype(counts.dtype, np.integer):
        raise ValueError('Tokens and counts must be integers')
    if tokens.ndim != 4 or counts.shape != tokens.shape[:-1] or tokens.shape[-1] != m:
        raise ValueError('Malformed draw shapes')
    if support.shape != (tokens.shape[0], vocab_size) or support.dtype != np.bool_:
        raise ValueError('Support must be boolean [B,V]')
    if np.any(counts < 0) or np.any(counts > m):
        raise ValueError('Counts outside the deployed count head')
    if np.any(tokens < 0) or np.any(tokens >= vocab_size):
        raise ValueError('Token index outside vocabulary')
    allowed = support.copy()
    allowed[:, 0] = False
    if not np.all(allowed.any(axis=-1)):
        raise ValueError('No legal future token support; define terminal behavior separately')
    active = np.arange(m)[None, None, None, :] < counts[..., None]
    permitted = np.take_along_axis(
        np.broadcast_to(allowed[:, None, None, None, :], tokens.shape + (vocab_size,)),
        tokens[..., None], axis=-1)[..., 0]
    if np.any(active & ~permitted):
        raise ValueError('Active token violates non-NULL/time support')


def decision_loss(teacher_margin, estimated_margin, eligible, tau, weight_cap):
    """Weighted preference loss [B,G]; fixed hyperparameters supplied by caller.

    eligible [B] encodes teacher hard-feasibility and the predeclared |delta|
    threshold. Ineligible states remain in reconstruction batches, weight zero.
    Teacher quantities and simulator margins are never differentiated. Run
    validate_decision_inputs on the host before jit; ineligible hard-infeasible
    teacher states use a finite placeholder margin, never None/NaN/-infinity.
    """
    if tau <= 0 or weight_cap <= 0:
        raise ValueError('tau and weight_cap must be strictly positive')
    teacher = jax.lax.stop_gradient(jnp.asarray(teacher_margin))
    estimated = jax.lax.stop_gradient(jnp.asarray(estimated_margin))
    if estimated.ndim != 2 or teacher.shape != estimated.shape[:1] or \
            eligible.shape != teacher.shape:
        raise ValueError('Margins must be teacher [B], estimated [B,G], eligible [B]')
    weight = jnp.where(eligible, jnp.minimum(jnp.abs(teacher), weight_cap), 0.0)
    return weight[:, None] * jax.nn.softplus(
        -jnp.sign(teacher)[:, None] * estimated / tau)


def validate_decision_inputs(teacher_margin, estimated_margin, eligible, prior_baseline):
    import numpy as np
    teacher, estimated, eligible, baseline = map(
        np.asarray, (teacher_margin, estimated_margin, eligible, prior_baseline))
    if teacher.ndim != 1 or estimated.ndim != 2 or \
            estimated.shape[0] != len(teacher) or eligible.shape != teacher.shape or \
            eligible.dtype != np.bool_ or baseline.shape != teacher.shape:
        raise ValueError('Invalid margin/eligibility/baseline shapes or types')
    if not all(np.isfinite(x).all() for x in (teacher, estimated, baseline)):
        raise ValueError('Nonfinite teacher/simulator/baseline input; no inf*zero masking')


def score_function_loss(log_prob_sets, losses, prior_baseline):
    """Unbiased score-function surrogate with baseline independent of current draw.

    prior_baseline [B] must come from PREVIOUS training steps (e.g. lagged EMA),
    or leave-one-out independent sets. A mean containing the current set is not
    an independent baseline; with one set it makes the gradient identically zero.
    Stop-gradient alone does not remove that statistical dependence.
    """
    if losses.shape != log_prob_sets.shape or \
            prior_baseline.shape != log_prob_sets.shape[:1]:
        raise ValueError('Expected log-prob/loss [B,G], prior baseline [B]')
    advantage = jax.lax.stop_gradient(losses - prior_baseline[:, None])
    return jnp.mean(advantage * log_prob_sets)


def combined_loss(log_prob_sets, losses, prior_baseline,
                  reconstruction_loss, reconstruction_weight):
    if reconstruction_weight < 0:
        raise ValueError('Reconstruction weight must be nonnegative')
    return score_function_loss(log_prob_sets, losses, prior_baseline) + \
        reconstruction_weight * reconstruction_loss
