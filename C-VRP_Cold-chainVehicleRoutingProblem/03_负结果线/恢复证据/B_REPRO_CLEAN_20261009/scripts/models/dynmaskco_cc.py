"""DynMaskCO-CC 模型（M1-edge-v1）：冻结 encoder + 可训练 masked decoder + utility head。

计算链（首版）：
    H = frozen_encoder(visible_state)
    Z = masked_decoder(H_visible, timestep, A_in)        # target='embed'
    candidate_features = ordered_gather(H_visible, Z) + explicit_features
    utility_scores = utility_head(candidate_features)

utility loss 经 Z 反传到 decoder（不是旁路 MLP）。encoder 冻结（不参与训练），decoder/head
可训练。邻接是 MaskCO 的计划条件（对称软 attention bias 0/1），不是执行真值；执行由有向
FleetPlan + action certificate 决定。

候选表征顺序（有序端点拼接，缺失零向量 + 有效位）：
    [H_customer, H_anchor, H_predecessor, H_successor,
     Z_customer, Z_anchor, Z_predecessor, Z_successor, endpoint_valid[4]]
"""
import numpy as np
from flax import nnx
import jax
import jax.numpy as jnp

from modules import TransformerEncoder, TimestepEmbedder, MLPBlock


class MaskCODecoder(nnx.Module):
    """可训练 masked decoder（镜像上游 TSPModel.decode 的 target='embed' 路径）。

    输入 H_visible [B, Nv, d_enc]，timestep [B]，adjmat [B, Nv, Nv]（0/1 软注意力 bias）。
    输出 Z [B, Nv, d_dec]。
    """

    def __init__(self, d_enc=256, d_dec=256, num_layers=6, num_heads=8, hidden_dim=None,
                 activation='silu', use_swiglu=True, dtype=jnp.bfloat16, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.mid_proj = nnx.Linear(d_enc, d_dec, use_bias=True, rngs=rngs)
        self.timestep_embedder = TimestepEmbedder(
            d_dec, frequency_embed_dim=256, timestep_scale=1000., dtype=dtype, rngs=rngs)
        self.transformer = TransformerEncoder(
            num_layers, d_dec, num_heads, hidden_dim, activation, use_swiglu,
            remat_layernorm=False, qkv_packed=True, qk_norm=False, dtype=dtype, rngs=rngs)
        self.final_proj = MLPBlock(d_dec, d_dec, d_dec, use_bias=True, activation='silu',
                                   dtype=dtype, rngs=rngs)
        self.dtype = dtype

    def __call__(self, H_visible, timestep, adjmat):
        x = self.mid_proj(H_visible)
        x = x + self.timestep_embedder(timestep).astype(x.dtype)[:, None]
        x = self.transformer(x, attn_options={'bias': adjmat})
        x = self.final_proj(x)
        return x


def gather_endpoints(Hv, Z, endpoints, valid):
    """候选有序端点 gather → [B, M, 4*d_H + 4*d_Z + 4]。

    endpoints [B, M, 4] int32（局部可见索引）；valid [B, M, 4] bool。缺失端点置零。
    """
    B, M, _ = endpoints.shape
    h_g = Hv[jnp.arange(B)[:, None, None], endpoints]          # [B, M, 4, d_H]
    z_g = Z[jnp.arange(B)[:, None, None], endpoints]           # [B, M, 4, d_Z]
    v = valid[..., None].astype(h_g.dtype)
    h_g = (h_g * v).reshape(B, M, -1)
    z_g = (z_g * v).reshape(B, M, -1)
    return jnp.concatenate([h_g, z_g, valid.astype(h_g.dtype)], axis=-1)


class ScoringMLP(nnx.Module):
    """候选评分 MLP：[..., F] → [...]（标量效用，越大越有利）。"""

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


class DynMaskCOModel(nnx.Module):
    """M1 主模型。encoder 冻结（held 引用，不进训练 state），decoder/head 可训练。"""

    def __init__(self, encoder, decoder, utility_head, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.encoder = encoder          # frozen DynamicColdChainModel（不做 nnx.split 的 trainable 引用）
        self.decoder = decoder          # MaskCODecoder（可训练）
        self.utility_head = utility_head  # ScoringMLP（可训练）

    def encode_visible(self, raw_features, visible_mask, local_to_node):
        """冻结 encoder → H [B, N_total, d]；压缩可见节点 → Hv [B, Nv, d]。"""
        H = self.encoder.encode(raw_features, visible_mask=visible_mask)
        Hv = H[:, np.asarray(local_to_node, dtype=np.int32)]
        return H, Hv

    def decode(self, Hv, timestep, A_in):
        return self.decoder(Hv, timestep, A_in)                # Z [B, Nv, d]

    def score(self, Hv, Z, endpoints, valid, explicit_feats):
        """候选评分：ordered_gather(Hv,Z) + explicit_feats → utility [B, M]。"""
        struct = gather_endpoints(Hv, Z, endpoints, valid)     # [B, M, 4d_H+4d_Z+4]
        x = jnp.concatenate([struct, explicit_feats], axis=-1)
        return self.utility_head(x)                            # [B, M]


class M1Scorer:
    """持久 JIT 评分器：decoder → endpoint gather → utility head → scores，创建一次。

    参数与输入全部作为函数参数传入（timestep / 邻接内容不设为静态参数）；编译缓存按形状
    复用。encoder 保持冻结（encode 单独调用，每事件一次）。Hv/Z 尽量留在设备上，只在需要
    在线选择时取回最终 scores。
    """

    def __init__(self, model):
        self.encoder = model.encoder
        self.d_graphdef, self.d_params = nnx.split(model.decoder)
        self.h_graphdef, self.h_params = nnx.split(model.utility_head)
        d_gd, h_gd = self.d_graphdef, self.h_graphdef

        @jax.jit
        def _score(d_params, h_params, Hv, ts, A_in, endpoints, valid, explicit):
            dec = nnx.merge(d_gd, d_params)
            hd = nnx.merge(h_gd, h_params)
            Z = dec(Hv, ts, A_in)
            struct = gather_endpoints(Hv, Z, endpoints, valid)
            x = jnp.concatenate([struct, explicit], axis=-1)
            return hd(x)                                    # [B, M]

        self._score = _score

    def encode(self, raw, visible_mask):
        return self.encoder.encode(raw, visible_mask=visible_mask)

    def score(self, Hv, ts, A_in, endpoints, valid, explicit):
        return self._score(self.d_params, self.h_params, Hv, ts, A_in, endpoints, valid,
                           explicit)


def load_m1_model(ckpt_path, encoder, num_layers=6, num_heads=8, hidden=(128, 64)):
    """从训练 checkpoint 重建 decoder + head，与冻结 encoder 组装 DynMaskCOModel。"""
    import pickle
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    d_model = data['d_model']
    explicit_dim = data['explicit_dim']
    decoder = MaskCODecoder(d_enc=d_model, d_dec=d_model, num_layers=num_layers,
                            num_heads=num_heads, rngs=0)
    head = ScoringMLP(4 * d_model * 2 + 4 + explicit_dim, hidden=hidden, rngs=0)
    d_gd, _ = nnx.split(decoder)
    h_gd, _ = nnx.split(head)
    decoder = nnx.merge(d_gd, data['decoder'])
    head = nnx.merge(h_gd, data['head'])
    return DynMaskCOModel(encoder, decoder, head, rngs=0)
