"""Matrix-aware directed RRP adapter around the read-only MaskCO backbone."""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports()

import jax
import jax.numpy as jnp
from flax import nnx
from models.TSPModel import TSPModel
from rrp_features import FEATURE_DIM

PROFILES = {
    'maskco_256': {'num_layers': (16, 6), 'embed_dim': 256, 'num_heads': 8},
    'smoke_cpu': {'num_layers': (1, 1), 'embed_dim': 32, 'num_heads': 4},
}


def softcap_attention(q=None, k=None, v=None, qkv=None, bias=None,
                      causal=False, sm_scale=None, softcap=None, **unused):
    """Portable CPU implementation; softcap is applied before adjacency bias.

    Installed upstream CPU attention drops softcap. Override only our instances,
    never a global function or source file in MASKCO_code.
    """
    if qkv is not None:
        q, k, v = (qkv[:, :, i] for i in range(3))
    dtype = q.dtype
    q, k, v = (a.astype(jnp.float32) for a in (q, k, v))
    logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) * (sm_scale if sm_scale is not None else q.shape[-1] ** -0.5)
    if softcap is not None:
        logits = softcap * jnp.tanh(logits / softcap)
    if bias is not None:
        logits = logits + (bias[:, None] if bias.ndim == 3 else bias)
    if causal:
        logits = jnp.where(jnp.tril(jnp.ones(logits.shape[-2:], bool)), logits, -jnp.inf)
    return jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, axis=-1), v).astype(dtype)


class RRPModel(nnx.Module):
    def __init__(self, profile, seed):
        if profile not in PROFILES:
            raise ValueError('Unknown model profile')
        dims = PROFILES[profile]
        self.config = {'profile': profile, 'seed': int(seed)}
        rngs = nnx.Rngs(seed)
        dim = dims['embed_dim']
        self.backbone = TSPModel(num_layers=dims['num_layers'], embed_dim=(dim, dim),
                                 num_heads=(dims['num_heads'], dims['num_heads']),
                                 encoder_input_dim=FEATURE_DIM, dtype=jnp.float32, softcap=30.0,
                                 final_norm=True, final_norm_use_scale=True,
                                 final_softcap=None, final_logit_scale=16 / dim, rngs=rngs)
        for layer in self.backbone.encoder.layers + self.backbone.decoder.layers:
            layer.attn.attention_fn = softcap_attention
        self.query = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.wait_head = nnx.Linear(dim, 1, rngs=rngs)
        self.scale = math.sqrt(dim)

    def __call__(self, features, distance_bias, adjacency, timestep):
        h = self.backbone.encode(features, attn_options={'bias': distance_bias})
        h = self.backbone.decode(h, timestep, adjacency, attn_options={}, target='embed')
        h = self.backbone.final_norm(h)
        logits = jnp.matmul(self.query(h), self.key(h).transpose(0, 2, 1)) / self.scale
        n = logits.shape[-1]
        logits = jnp.where(jnp.eye(n, dtype=bool)[None], -1e9, logits)
        wait_log = jax.nn.softplus(self.wait_head(h)[..., 0])
        return logits, wait_log


def make_model(profile='maskco_256', seed=888):
    return RRPModel(profile, seed)
