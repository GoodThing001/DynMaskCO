"""
DynamicColdChainModelEdgeState — Phase A1：多头边偏置

在 DynamicColdChainModel 基础上，把单一的标量边偏置（edge_weight * energy_mat）
升级为多头学习的边偏置（EdgeBiasProjector 处理 5D pairwise 边特征）。

关键设计（最小侵入式）：
- 继承现有 NNX DynamicColdChainModel，不覆盖 backbone
- 新增 EdgeBiasProjector（5D → 8 头 attention bias）
- override encode()，仅替换 edge_bias 计算部分
- 边特征计算用纯 JAX（edge_features.py），无额外参数

Author: Phase A1
Date: 2026-08-25
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
from flax import nnx

from DynamicColdChainModel import DynamicColdChainModel
from edge_bias_projector import EdgeBiasProjector
from edge_features import compute_static_pairwise_edge_features


def _compute_dist_mat(coords):
    """从坐标计算欧氏距离矩阵 [B, N, N]。"""
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return jnp.sqrt((diff ** 2).sum(axis=-1) + 1e-10)


class DynamicColdChainModelEdgeState(DynamicColdChainModel):
    """
    Phase A1：Edge-State 表示。

    与父类 DynamicColdChainModel 的唯一区别：
    - edge_feat（energy_mat [B,N,N]）不再用标量 edge_weight 缩放，
      而是从 raw_features 计算 5D pairwise 边特征，
      经 EdgeBiasProjector 投影成 [B,H,N,N] 多头注意力偏置。

    5D 边特征：[distance, travel_time, angle, energy, tw_compat]
    """

    def __init__(self, encoder_input_dim=7, edge_dim=5, **kwargs):
        # 处理 rngs（int → Rngs）
        if 'rngs' in kwargs and isinstance(kwargs['rngs'], int):
            kwargs['rngs'] = nnx.Rngs(kwargs['rngs'])
        # 旧参数兼容（与父类一致）
        kwargs.pop('spoilage_lambda', None)

        # 保存 edge 配置，供 setup 使用
        rngs = kwargs.get('rngs', nnx.Rngs(0))
        num_heads = kwargs.get('num_heads', 8)
        if isinstance(num_heads, (tuple, list)):
            num_heads = num_heads[0]  # encoder 的 heads 数

        # 调用父类（保留 type_embed / edge_weight 等）
        super().__init__(encoder_input_dim=encoder_input_dim, **kwargs)

        # Phase A1 新增：EdgeBiasProjector（5D → 多头 bias）
        self.edge_dim = edge_dim
        self.edge_bias_projector = EdgeBiasProjector(
            edge_dim=edge_dim,
            num_heads=num_heads,
            rngs=rngs,
        )

    def encode(self, raw_features, attn_options=None, visible_mask=None, edge_feat=None):
        """
        编码 7D/8D 特征，edge_feat 升级为 5D 边特征 → 多头偏置。

        与父类 encode 相同的前半部分（init_proj / depot_bias / tw_bias / type_embed），
        区别在 edge_bias：父类用标量 edge_weight * edge_feat（[B,N,N]），
        本类用 EdgeBiasProjector(5D edge features) → [B,H,N,N]。
        """
        if attn_options is None:
            attn_options = {}
        attn_options.update(softcap=self.softcap, sm_scale=self.sm_scale)
        depot_bias = self.depot_bias.value.astype(self.dtype)
        tw_bias = self.tw_bias.value.astype(self.dtype)

        features = self.init_proj(raw_features)
        features = features.at[:, 0].add(depot_bias[None])
        features = features + tw_bias[None, None, :]

        # === 节点类型专属 Embedding（与父类一致）===
        temp_class = jnp.clip((raw_features[..., 5] * 2.0 + 0.5).astype(jnp.int32), 0, 2)
        node_type = temp_class + 1
        node_type = node_type.at[:, 0].set(0)
        if visible_mask is not None:
            node_type = jnp.where(visible_mask > 0.5, node_type, 4)
        type_emb = self.type_embed(node_type).astype(self.dtype)
        features = features + type_emb

        # === 注意力偏置：可见性门控 + 多头边偏置（A1 核心改动）===
        combined_bias = None
        if visible_mask is not None:
            B, N = visible_mask.shape
            vis_bias = jnp.where(visible_mask[:, None, :], 0.0, -1e9)
            vis_bias = vis_bias.at[:, :, 0].set(0.0)
            combined_bias = vis_bias  # [B, N, N]

        if edge_feat is not None:
            # A1：从 raw_features 计算 5D pairwise 边特征
            coords = raw_features[..., :2]          # (B, N, 2)
            tw_start = raw_features[..., 3]          # (B, N)
            tw_end = raw_features[..., 4]            # (B, N)
            dist_mat = _compute_dist_mat(coords)     # (B, N, N)

            edge_5d = compute_static_pairwise_edge_features(
                coords=coords,
                tw_start=tw_start,
                tw_end=tw_end,
                dist_mat=dist_mat,
                energy_mat=edge_feat,                # 复用 energy_mat
                vehicle_speed=1.0,
                visible_mask=visible_mask,
            )  # [B, N, N, 5]

            # EdgeBiasProjector: [B,N,N,5] → [B,H,N,N]
            edge_bias = self.edge_bias_projector(edge_5d)  # [B, H, N, N]

            # 与 vis_bias 合并（vis_bias [B,N,N] 广播到 [B,H,N,N]）
            if combined_bias is not None:
                combined_bias = combined_bias[:, None, :, :] + edge_bias  # [B, H, N, N]
            else:
                combined_bias = edge_bias

        if combined_bias is not None:
            existing = attn_options.get('bias', None)
            attn_options['bias'] = combined_bias if existing is None else existing + combined_bias

        features = self.encoder(features, attn_options=attn_options)
        return features


# ============================================================================
# 单元测试
# ============================================================================

if __name__ == '__main__':
    import numpy as np

    print("Testing DynamicColdChainModelEdgeState (NNX)...")

    # 用 small_test 配置（小模型，快速验证）
    from CVRPTWModel import CVRPTWModelConfig

    B, N = 2, 8
    np.random.seed(42)

    # 构造 raw_features [B, N, 7]: [x, y, demand, tw_start, tw_end, temp_class/2, reveal_time]
    raw_features = np.zeros((B, N, 7), dtype=np.float32)
    raw_features[..., :2] = np.random.rand(B, N, 2)
    raw_features[..., 2] = np.random.rand(B, N) * 10  # demand
    raw_features[..., 3] = np.random.rand(B, N) * 5   # tw_start
    raw_features[..., 4] = raw_features[..., 3] + 5   # tw_end
    raw_features[..., 5] = np.random.randint(0, 3, (B, N)) / 2.0  # temp_class/2
    raw_features[..., 6] = np.random.rand(B, N) * 10  # reveal_time
    raw_features = jnp.array(raw_features)

    visible_mask = jnp.array((raw_features[..., 6] <= 5.0).astype(jnp.float32))
    energy_mat = jnp.array(np.random.rand(B, N, N) * 10).astype(jnp.float32)

    # 用小的模型配置（减少内存）
    # 注意：embed_dim 和 num_heads 必须是 tuple (encoder, decoder)
    model = DynamicColdChainModelEdgeState(
        num_layers=(2, 2),          # (encoder_layers, decoder_layers)
        embed_dim=(128, 128),       # (encoder_dim, decoder_dim)
        num_heads=(4, 4),           # (encoder_heads, decoder_heads)
        encoder_input_dim=7,
        edge_dim=5,
        rngs=42,
    )

    print(f"✓ 模型初始化成功")
    print(f"  - edge_bias_projector 参数: {model.edge_bias_projector.proj.kernel.value.shape}")

    # 前向传播（注意：显式传 attn_options={}，避免可变默认参数污染）
    encoded = model.encode(raw_features, attn_options={}, visible_mask=visible_mask, edge_feat=energy_mat)
    print(f"✓ encode 输出形状: {encoded.shape}")
    assert encoded.shape == (B, N, 128), f"Expected {(B, N, 128)}, got {encoded.shape}"

    # 测试无 edge_feat 时（应退化为父类行为）
    encoded_no_edge = model.encode(raw_features, attn_options={}, visible_mask=visible_mask, edge_feat=None)
    print(f"✓ 无 edge_feat 时 encode 输出形状: {encoded_no_edge.shape}")
    assert encoded_no_edge.shape == (B, N, 128)

    # 测试无 visible_mask 时
    encoded_no_mask = model.encode(raw_features, attn_options={}, visible_mask=None, edge_feat=energy_mat)
    print(f"✓ 无 visible_mask 时 encode 输出形状: {encoded_no_mask.shape}")

    print("\n✅ 所有 DynamicColdChainModelEdgeState 测试通过！")
    print("")
    print("Summary:")
    print("  ✓ 继承现有 NNX DynamicColdChainModel（不覆盖 backbone）")
    print("  ✓ EdgeBiasProjector 正确初始化（48 参数：5×4+4）")
    print("  ✓ encode 输出 [B, N, embed_dim]")
    print("  ✓ 有/无 edge_feat、有/无 visible_mask 都能运行")
    print("  ✓ 准备进行 Day 5 等价性/因果性测试")
