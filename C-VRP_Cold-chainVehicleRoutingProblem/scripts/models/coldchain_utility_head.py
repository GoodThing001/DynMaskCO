"""M0 效用打分 head（feature-only 与后续 Real/Shuffle 共用同一接口）。

`UtilityHead` 是唯一可训练的打分头：context 向量 + 候选动作特征（值 + 有效位）→ 标量效用
（越大越有利）。context 向量由三种来源之一产生，训练器只看到「context 向量 + 动作特征」，
不感知来源，因此 feature-only / encoder-Real / encoder-Shuffle 共用同一训练流程：

  - feature_only：掩码聚合当前 context 的可见订单/车队特征（`feature_only_context`）；
  - encoder Real：冻结 MaskCO encoder 的节点表征按可见掩码聚合（`encoder_context`）；
  - encoder Shuffle：同一表征先按节点打乱再聚合（`shuffle_node_embeddings`，破坏节点身份，
    作为控制组验证 head 是否真用了 encoder 的节点结构）。

JAX / Flax NNX（与 MaskCO 同栈）；NumPy 只用于读取与预处理。
"""
import numpy as np
from flax import nnx
import jax
import jax.numpy as jnp

ORDER_FEAT_DIM = 9
FLEET_FEAT_DIM = 16
ACTION_FEAT_DIM = 8
FEATURE_CONTEXT_DIM = ORDER_FEAT_DIM + FLEET_FEAT_DIM   # 25


def masked_mean(x, mask):
    """对倒数第二维（节点/车辆）做 mask 平均，结果去掉该维。

    x: [..., N, D]，mask: [..., N]（bool/float）→ [..., D]。
    隐藏节点（mask=False）不参与求和与归一化，因此追加/扰动隐藏节点不改变结果。
    """
    m = mask.astype(x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    denom = jnp.maximum(m.sum(axis=-2), 1.0)
    return (x * m).sum(axis=-2) / denom


def feature_only_context(order_feats, node_visible, fleet_feats, vehicle_valid):
    """feature-only context 向量：可见订单/车队特征的掩码平均拼接。→ [..., 25]。"""
    o = masked_mean(order_feats, node_visible)
    f = masked_mean(fleet_feats, vehicle_valid)
    return jnp.concatenate([o, f], axis=-1)


def encoder_context(H, node_visible):
    """encoder context 向量：节点表征 H 按可见掩码聚合。→ [..., D]。"""
    return masked_mean(H, node_visible)


def shuffle_node_embeddings(H, rng=None, keep_depot=True):
    """表征 Shuffle 控制：按 batch 逐条打乱非 depot 节点的表征行（depot 行保留）。

    破坏「节点身份 ↔ 表征」的对应，用于验证 head 是否真的依赖 encoder 的节点结构。
    纯 NumPy，供 encoder 模式的 context 预计算使用。输入/输出均为 [B, N, D]（2D 输入升为
    [1, N, D]）。
    """
    H = np.asarray(H, dtype=np.float32)
    if H.ndim == 2:
        H = H[None]
    if rng is None:
        rng = np.random.default_rng(0)
    out = H.copy()
    B, N, D = H.shape
    start = 1 if keep_depot else 0
    for b in range(B):
        idx = np.arange(start, N)
        rng.shuffle(idx)
        out[b, start:] = H[b, idx]
    return out


class UtilityHead(nnx.Module):
    """共享打分头：context 向量 + 候选动作（值 + 有效位）→ 标量效用。

    动作特征按 [vals, valid] 拼接（缺失字段用有效位区分，不与 depot=0 混同）。
    """

    def __init__(self, context_dim, action_dim=ACTION_FEAT_DIM, hidden=(128, 64), rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        in_dim = context_dim + 2 * action_dim
        self.l1 = nnx.Linear(in_dim, hidden[0], rngs=rngs)
        self.l2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
        self.l3 = nnx.Linear(hidden[1], 1, rngs=rngs)

    def __call__(self, context, action_vals, action_valid):
        a = jnp.concatenate([action_vals, action_valid.astype(action_vals.dtype)], axis=-1)
        # context 是每个 context 共享的向量（比动作少一个候选维），广播到候选维与动作对齐。
        context = jnp.broadcast_to(jnp.expand_dims(context, axis=-2),
                                   a.shape[:-1] + (context.shape[-1],))
        x = jnp.concatenate([context, a], axis=-1)
        x = jax.nn.gelu(self.l1(x))
        x = jax.nn.gelu(self.l2(x))
        return self.l3(x)[..., 0]


class FeatureOnlyProbe(nnx.Module):
    """feature-only 打分器：feature_only_context → UtilityHead。"""

    def __init__(self, order_dim=ORDER_FEAT_DIM, fleet_dim=FLEET_FEAT_DIM,
                 action_dim=ACTION_FEAT_DIM, hidden=(128, 64), rngs=None):
        self.head = UtilityHead(order_dim + fleet_dim, action_dim, hidden, rngs=rngs)

    def context(self, order_feats, node_visible, fleet_feats, vehicle_valid):
        return feature_only_context(order_feats, node_visible, fleet_feats, vehicle_valid)

    def __call__(self, order_feats, node_visible, fleet_feats, vehicle_valid,
                 action_vals, action_valid):
        c = self.context(order_feats, node_visible, fleet_feats, vehicle_valid)
        return self.head(c, action_vals, action_valid)
