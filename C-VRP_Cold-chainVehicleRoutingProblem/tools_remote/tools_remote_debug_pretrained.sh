#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python - <<'EOF'
import sys, os
sys.path.insert(0, 'scripts/evaluation'); sys.path.insert(0, 'scripts/models')
sys.path.insert(0, 'scripts/coldchain'); sys.path.insert(0, 'scripts/simulation')
import numpy as np
import jax, jax.numpy as jnp
from scenario_saa import SPOT_CENTERS
from maskco_scenario import (MASK_TOKEN, PAD_TOKEN, NULL_TOKEN, N_TOKENS, VOCAB_SIZE,
                             N_DEMAND_BINS, N_SPOTS, N_CLASSES)

B, L = 8, 383
rng = np.random.default_rng(0)
tokens = np.full((B, L), MASK_TOKEN, dtype=np.int32)
tokens[:, :183] = rng.integers(1, VOCAB_SIZE, size=(B, 183))
tokens = jnp.asarray(tokens)

idx = jnp.arange(N_TOKENS)
v = jnp.maximum(idx - 1, 0)
d = v % N_DEMAND_BINS
v = v // N_DEMAND_BINS
b = v % N_SPOTS
v = v // N_SPOTS
c = v % N_CLASSES
t = v // N_CLASSES
valid0 = idx >= 1
spots = jnp.asarray(SPOT_CENTERS)
bx = jnp.where(valid0, spots[b, 0], 0.0)
by = jnp.where(valid0, spots[b, 1], 0.0)
bd = jnp.where(valid0, jnp.array([1.5, 2.5])[d], 0.0)
table = jnp.stack([bx, by, bd], axis=-1)
raw = table[tokens]
coords = raw[..., :2]
valid = ((tokens != MASK_TOKEN) & (tokens != PAD_TOKEN)
         & (tokens != NULL_TOKEN)).astype(jnp.float32)
prod = coords * valid[..., None]
print('prod', prod.shape)
s = prod.sum(axis=1, keepdims=True)
print('sum', s.shape)
n_valid = jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
print('n_valid', n_valid.shape)
m2 = s / n_valid
print('m2', m2.shape)
EOF
