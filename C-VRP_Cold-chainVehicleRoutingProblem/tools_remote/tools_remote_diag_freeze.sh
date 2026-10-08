#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python - <<'EOF' 2>&1 | tail -n 20
import sys, os, hashlib
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/evaluation')
sys.path.insert(0, 'scripts/models'); sys.path.insert(0, 'scripts/coldchain')
sys.path.insert(0, 'scripts/simulation')
import numpy as np
import jax, jax.numpy as jnp
import optax
from flax import nnx
from mpre import load_cvrp_model
from maskco_scenario import MaskCOScenarioModel, MASK_TOKEN, NULL_TOKEN, PAD_TOKEN
from run_exp_reserve import generate_dataset
from scenario_saa import build_history
from maskco_scenario import make_training_slices

cvrp, cfg, step0 = load_cvrp_model('../MASKCO_code/ckpts/cvrp100.ckpt')
model = MaskCOScenarioModel(dim=128, arm='pretrained', cvrp_model=cvrp, rngs=nnx.Rngs(42))

def frozen_vals(m):
    fs = nnx.state(m, nnx.Param).flat_state()
    return {str(p): np.asarray(v.value) for p, v in zip(fs.paths, fs.leaves)
            if any(str(x).startswith('_cvrp') for x in p)}

before = frozen_vals(model)
hist = build_history(generate_dataset(20, 200, 20260925))
slices = make_training_slices(hist, cuts_per_day=4, rng=np.random.default_rng(42))

def make_batch_local(slices, idxs, m_max=200):
    L = 2 * m_max
    X = np.full((len(idxs), L), PAD_TOKEN, dtype=np.int32)
    Y = np.full((len(idxs), L), NULL_TOKEN, dtype=np.int32)
    N, CB = [], []
    for b, i in enumerate(idxs):
        vis, tgt, n, cb = slices[i]
        vis = vis[:m_max]; tgt = tgt[:m_max]
        X[b, :len(vis)] = vis
        X[b, len(vis):len(vis) + m_max] = MASK_TOKEN
        Y[b, len(vis):len(vis) + len(tgt)] = tgt
        N.append(min(int(n), m_max)); CB.append(int(cb))
    return X, Y, np.array(N, np.int32), np.array(CB, np.int32)

def _is_frozen(path):
    return any(str(x).startswith('_cvrp') for x in path)

def flat(m):
    fs = nnx.state(m, nnx.Param).flat_state()
    return {k: v for k, v in zip(fs.paths, fs.leaves)}

trainable = nnx.State.from_flat_path({k: v for k, v in flat(model).items() if not _is_frozen(k)})
tx = optax.adamw(1e-3)
opt_state = tx.init(trainable)

def loss_fn(model, X, Y, N, CB):
    tl, cl = model(X, clock_bin=CB)
    mask = X == MASK_TOKEN
    nll = optax.softmax_cross_entropy_with_integer_labels(tl, Y)
    n = jnp.maximum(mask.sum(), 1.0)
    lt = jnp.sum(jnp.where(mask, nll, 0.0)) / n
    lc = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(cl, N.astype(jnp.int32)))
    return lt + lc

@nnx.jit
def step(model, opt_state, trainable, X, Y, N, CB):
    loss, grads = nnx.value_and_grad(loss_fn)(model, X, Y, N, CB)
    gfs = grads.flat_state()
    gd = {k: v for k, v in zip(gfs.paths, gfs.leaves)}
    tk = {k for k in trainable.flat_state().paths}
    grads = nnx.State.from_flat_path({k: v for k, v in gd.items() if k in tk})
    updates, opt_state = tx.update(grads, opt_state, trainable)
    trainable = optax.apply_updates(trainable, updates)
    nnx.update(model, trainable)
    return loss, opt_state, trainable

rng = np.random.default_rng(42)
for it in range(5):
    idxs = rng.integers(0, len(slices), 4)
    X, Y, N, CB = make_batch_local(slices, list(idxs))
    loss, opt_state, trainable = step(model, opt_state, trainable,
                                      jnp.asarray(X), jnp.asarray(Y),
                                      jnp.asarray(N), jnp.asarray(CB))
after = frozen_vals(model)
maxd = 0.0
keys_d = []
for k in before:
    d = float(np.abs(after[k] - before[k]).max())
    if d > maxd:
        maxd = d
    if d > 0:
        keys_d.append((k, d))
print('max |delta| over all frozen leaves:', maxd)
print('n leaves with nonzero delta:', len(keys_d))
for k, d in sorted(keys_d, key=lambda x: -x[1])[:5]:
    print('  ', k, d)
print('sample value: before=%.6f after=%.6f' % (
    float(before[list(before)[0]].ravel()[0]), float(after[list(before)[0]].ravel()[0])))
EOF
