"""
DynamicColdChainModel — 7D 动态冷链模型。
方向二修复: spoilage_bias(有害-6pp) → Learnable Temperature Embedding
"""

import sys, os
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT
sys.path.insert(0, str(MASKCO_ROOT))

import jax, jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from ColdChainModel import ColdChainModel, ColdChainModelConfig

K_TEMP = jnp.array([0.01, 0.002, 0.0002], dtype=jnp.float32)


@dataclass
class DynamicColdChainModelConfig(ColdChainModelConfig):
    encoder_input_dim: int = 7

    @staticmethod
    def get_config(which=''):
        match which:
            case '' | 'default':
                return DynamicColdChainModelConfig()
            case 'softcap_fn':
                return DynamicColdChainModelConfig(
                    embed_dim=256, num_heads=8,
                    qk_norm=False, softcap=30.,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                )
            case _:
                raise ValueError(f"Unknown config: {which}")

    def construct_model(self):
        return DynamicColdChainModel(**vars(self))


class DynamicColdChainModel(ColdChainModel):
    """动态冷链模型 — Temp Embedding 替代 spoilage_bias。"""

    def __init__(self, encoder_input_dim=7, **kwargs):
        if 'rngs' in kwargs and isinstance(kwargs['rngs'], int):
            kwargs['rngs'] = nnx.Rngs(kwargs['rngs'])
        # 移除 kwargs 中的 spoilage_lambda（旧参数兼容）
        kwargs.pop('spoilage_lambda', None)
        super().__init__(encoder_input_dim=encoder_input_dim, **kwargs)

        embed_dim = kwargs.get('embed_dim', (256, 256))
        encoder_embed_dim = embed_dim[0] if isinstance(embed_dim, tuple) else embed_dim

        # 节点类型专属 Embedding（typed embedding，借鉴 ESWA 2025 的节点专属投影 W0/Wp/Wd/Wr）:
        #   0=depot, 1=常温, 2=冷藏, 3=冷冻, 4=未来订单（reveal 前不可见）。
        # 替代旧 temp_embed(3)，并修复旧版「temp_class/2.0 被 astype(int32) 截断、
        # 冷藏(0.5→0)被误归为常温」的 bug。
        self.type_embed = nnx.Embed(
            num_embeddings=5, features=encoder_embed_dim,
            embedding_init=nnx.initializers.normal(1e-6),
            rngs=kwargs['rngs'],
        )
        # 显式边特征权重（P0-6）：energy_mat / 边品质代价 → 编码器注意力偏置的可学习缩放。
        # init=0 使边特征默认不起作用（安全 no-op），训练时逐步学习。
        self.edge_weight = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def encode(self, raw_features, attn_options=None, visible_mask=None, edge_feat=None):
        """编码 7D/8D 特征。

        visible_mask: (B, N), 1=可见, 0=未来订单。将不可见节点注意力门控为零。
        edge_feat:    (B, N, N) 可选，显式边特征（如 energy_mat），作为注意力偏置注入
                     编码器（借鉴 GAT 边特征 embedding）。
        """
        if attn_options is None:
            attn_options = {}
        attn_options.update(softcap=self.softcap, sm_scale=self.sm_scale)
        depot_bias = self.depot_bias.value.astype(self.dtype)
        tw_bias = self.tw_bias.value.astype(self.dtype)

        features = self.init_proj(raw_features)
        features = features.at[:, 0].add(depot_bias[None])
        features = features + tw_bias[None, None, :]

        # === 节点类型专属 Embedding ===
        # raw_features[...,5] = temp_class/2.0 ∈ {0, 0.5, 1.0}（dataloader 与 decoder 均归一化）。
        # ×2+0.5 再取整以正确恢复 0/1/2（修复旧 temp_embed 的截断 bug）。
        temp_class = jnp.clip((raw_features[..., 5] * 2.0 + 0.5).astype(jnp.int32), 0, 2)
        node_type = temp_class + 1              # 1=常温, 2=冷藏, 3=冷冻
        node_type = node_type.at[:, 0].set(0)   # 0=depot
        if visible_mask is not None:
            node_type = jnp.where(visible_mask > 0.5, node_type, 4)  # 4=未来订单
        type_emb = self.type_embed(node_type).astype(self.dtype)
        features = features + type_emb

        # === 注意力偏置：可见性门控 (P0-2) + 显式边特征 (P0-6) ===
        combined_bias = None
        if visible_mask is not None:
            B, N = visible_mask.shape
            # vis_bias[q,k] = 0 if node k is visible, else -inf
            vis_bias = jnp.where(visible_mask[:, None, :], 0.0, -1e9)
            # depot (col 0) always visible regardless of mask
            vis_bias = vis_bias.at[:, :, 0].set(0.0)
            combined_bias = vis_bias
        if edge_feat is not None:
            # 修复 2026-08-26：edge_feat 乘 visible 掩码，屏蔽未来节点的边信息泄漏。
            # 原实现 edge_bias = edge_weight * edge_feat，future 节点的边（energy_mat）仍为真值，
            # 导致未来节点 embedding 通过这些真实边信息变得不同，重新打开泄漏通道。
            if visible_mask is not None:
                edge_mask = visible_mask[:, None, :] * visible_mask[:, :, None]
                edge_feat = edge_feat * edge_mask
            edge_bias = self.edge_weight.value * edge_feat
            combined_bias = edge_bias if combined_bias is None else combined_bias + edge_bias
        if combined_bias is not None:
            existing = attn_options.get('bias', None)
            attn_options['bias'] = combined_bias if existing is None else existing + combined_bias

        features = self.encoder(features, attn_options=attn_options)
        return features

    def compute_legacy_delivery_risk_matrix(self, raw_features, speed=1.0):
        """历史 delivery-style 边风险代理；不得作为 C0 权威指标。

        ``raw_features[..., 5]`` 存储的是 ``temp_class / 2``。该代理只可
        用作旧 checkpoint 的表示输入；pickup-to-depot 品质真值必须由
        ``coldchain_state`` 基于 cargo manifest 和执行轨迹递推。
        """
        coords = raw_features[..., :2]
        temp_class = jnp.clip(
            (raw_features[..., 5] * 2.0 + 0.5).astype(jnp.int32), 0, 2)

        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = jnp.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
        travel_time = dist / speed

        k = K_TEMP[temp_class]
        k_expanded = k[:, :, None]

        spoilage = 1.0 - jnp.exp(-k_expanded * travel_time)
        spoilage = spoilage.at[:, 0, :].set(0)
        spoilage = spoilage.at[:, :, 0].set(0)
        return spoilage

    def compute_spoilage_matrix(self, raw_features, speed=1.0):
        """兼容旧调用；返回非权威的 delivery-style 表示代理。"""
        return self.compute_legacy_delivery_risk_matrix(raw_features, speed=speed)
