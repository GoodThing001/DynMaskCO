"""紧凑 DynMaskCO-CC 模型（A/B/C/D 消融用）。

相比 dynmaskco_cc.py 的 2100 维端点拼接，这里把 H/Z 先做共享低维投影（d_compact），再按
customer/anchor/predecessor/successor 四端点 gather，避免输入维度与 context 数失衡。

关键修复：
  - 节点有效性 mask：decoder 只收邻接 bias，补零节点会经线性层 + timestep 参与注意力；
    这里用 node_valid 把补零节点的注意力 bias 置 -1e9 并清零其输出。
  - 候选可区分：显式特征含候选局部几何（coord/demand/tw/temp + 插入增量），去编号后端点
    仍有几何区分度；A 组纯显式特征即可区分候选。

结构：
  H = frozen_encoder(visible_state)             # [B, N, 256]
  h = compact_proj(H)                           # [B, N, d_compact]（B/C/D）
  z = compact_decoder(h, ts, A_in, node_valid)  # [B, N, d_compact]（C/D）
  feat = [explicit(70) | + gather_endpoints(h) | + gather_endpoints(z)] → ScoringMLP → 分数
"""
import numpy as np
from flax import nnx
import jax
import jax.numpy as jnp

from modules import TransformerEncoder, TimestepEmbedder, MLPBlock


def gather_endpoints(X, endpoints, valid):
    """候选有序端点 gather（低维）。endpoints [B, M, 4] int32；valid [B, M, 4] bool。

    返回 [B, M, 4*d]（缺失端点置零）。
    """
    B, M, _ = endpoints.shape
    d = X.shape[-1]
    g = X[jnp.arange(B)[:, None, None], endpoints]       # [B, M, 4, d]
    v = valid[..., None].astype(g.dtype)
    return (g * v).reshape(B, M, -1)


class CompactDecoder(nnx.Module):
    """低维 masked decoder，带节点有效性 mask（补零节点不参与注意力）。"""

    def __init__(self, d=32, num_layers=6, num_heads=4, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.mid_proj = nnx.Linear(d, d, use_bias=True, rngs=rngs)
        self.timestep_embedder = TimestepEmbedder(
            d, frequency_embed_dim=256, timestep_scale=1000., dtype=jnp.float32, rngs=rngs)
        self.transformer = TransformerEncoder(
            num_layers, d, num_heads, None, 'silu', False,
            remat_layernorm=False, qkv_packed=True, qk_norm=False, dtype=jnp.float32, rngs=rngs)
        self.final_proj = MLPBlock(d, d, d, use_bias=True, activation='silu', dtype=jnp.float32,
                                   rngs=rngs)

    def __call__(self, H_compact, timestep, adjmat, node_valid):
        x = self.mid_proj(H_compact)
        x = x + self.timestep_embedder(timestep).astype(x.dtype)[:, None]
        v = node_valid[..., None].astype(x.dtype)          # [B, Nv, 1]
        vv = v * jnp.transpose(v, (0, 2, 1))               # [B, Nv, Nv]，两者都有效才 1
        bias = jnp.where(vv > 0.5, adjmat, -1e9)
        x = self.transformer(x, attn_options={'bias': bias})
        x = self.final_proj(x)
        return x * v                                        # 清零补零节点输出


class CompactModel(nnx.Module):
    """紧凑三组模型。encoder 冻结；compact_proj / decoder / head 可训练。

    use_H=True 加入 compact H 端点；use_Z=True 加入 compact Z 端点（需 decoder）。
    """

    def __init__(self, encoder, explicit_dim, d_compact=32, num_layers=6, num_heads=4,
                 hidden=(128, 64), use_H=True, use_Z=True, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.encoder = encoder              # frozen
        self.d_compact = d_compact
        self.use_H = use_H
        self.use_Z = use_Z
        self.compact_proj = nnx.Linear(256, d_compact, use_bias=False, rngs=rngs)
        if use_Z:
            self.decoder = CompactDecoder(d_compact, num_layers, num_heads, rngs=rngs)
        in_dim = explicit_dim + (4 * d_compact if use_H else 0) + (4 * d_compact if use_Z else 0)
        self.head = ScoringMLP(in_dim, hidden, rngs=rngs)

    def encode_compact(self, H_full, local_to_node):
        Hv = H_full[:, np.asarray(local_to_node, dtype=np.int32)]   # [B, Nv, 256]
        h = self.compact_proj(Hv)                                   # [B, Nv, d]
        return h

    def score(self, explicit, h, z, endpoints, valid):
        parts = [explicit]
        if self.use_H:
            parts.append(gather_endpoints(h, endpoints, valid))
        if self.use_Z:
            parts.append(gather_endpoints(z, endpoints, valid))
        x = jnp.concatenate(parts, axis=-1)
        return self.head(x)                                        # [B, M]


class ScoringMLP(nnx.Module):
    def __init__(self, in_dim, hidden=(128, 64), rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.l1 = nnx.Linear(in_dim, hidden[0], rngs=rngs)
        self.l2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
        self.l3 = nnx.Linear(hidden[1], 1, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.gelu(self.l1(x))
        x = jax.nn.gelu(self.l2(x))
        return self.l3(x)[..., 0]
