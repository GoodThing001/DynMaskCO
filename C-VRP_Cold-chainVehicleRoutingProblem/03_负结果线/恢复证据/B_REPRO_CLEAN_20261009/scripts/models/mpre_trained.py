"""M-trained 模型：冻结 CVRP512 encoder + 微调 decoder + 冷链适配层 + 有向插入残差头。

结构：
    MpreTrainedModel
    ├── backbone       完整 CVRP512 预训练模型（cvrp100.ckpt）
    ├── c_adapter      冷链状态 → 每节点 512 维残差（最后一层零初始化）
    └── delta_head     有序插入动作 → 标量残差（最后一层零初始化）

前向（全部 JAX 可微）：
    H0 = backbone.encode(raw_3d, attn_options=encoder_bias)
    H0 = stop_gradient(H0)
    H  = H0 + c_adapter(node_feats)
    Z  = backbone.decode(H, timestep, decoder_bias, attn_options={}, target='embed')
    L  = backbone.feature2logit(Z)                        # 对称边 logits
    s_edge = edge_insertion_scores(L, cust, pred, succ)   # 增删边评分
    scores = s_edge + delta_head(directed_feats(Z, actions), explicit)

冻结靠参数分区（不是梯度清零）：init_proj/encoder/depot_bias 冻结；mid_proj/timestep_embedder/
decoder/final_proj/final_norm + c_adapter + delta_head 训练。零初始化只作用于两个残差出口的
最后一层，故训练前 s_trained == s_pre。
"""
import os
import sys
import types

import jax
import jax.numpy as jnp
from flax import nnx

from mpre import load_cvrp_model, _ensure_tensorboardx_stub, _setup_maskco_paths


def _zeros_init():
    return nnx.initializers.zeros_init()


class MLP(nnx.Module):
    """简单 MLP。zero_last=True 时最后一层 kernel/bias 全零。"""

    def __init__(self, in_dim, hidden, out_dim, zero_last=False, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        dims = [in_dim] + list(hidden) + [out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            last = (i == len(dims) - 2)
            if last and zero_last:
                self.layers.append(nnx.Linear(
                    dims[i], dims[i + 1], rngs=rngs,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init()))
            else:
                self.layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        return x


class ColdChainAdapter(nnx.Module):
    """冷链状态 → 每节点 512 维残差（最后一层零初始化）。"""

    def __init__(self, node_feat_dim, hidden=(512, 512), out_dim=512, rngs=None):
        self.mlp = MLP(node_feat_dim, hidden, out_dim, zero_last=True, rngs=rngs)

    def __call__(self, node_feats):
        return self.mlp(node_feats)                      # [B, N, 512]


class DirectedInsertionHead(nnx.Module):
    """有序插入动作 → 标量残差（最后一层零初始化）。"""

    def __init__(self, action_feat_dim, hidden=(128, 64), rngs=None):
        self.mlp = MLP(action_feat_dim, hidden, 1, zero_last=True, rngs=rngs)

    def __call__(self, action_feats):
        return self.mlp(action_feats)[..., 0]            # [B, M]


def edge_insertion_scores(L, cust, pred, succ):
    """增删边评分（JAX 可微）：增 L[pred,c]+L[c,succ]；仅 pred≠succ 时删 L[pred,succ]。

    空车开路线 pred==succ 无自环可删，对角哨兵不进算术。返回 [B, M]。
    """
    B, M = cust.shape
    b = jnp.arange(B)[:, None]                          # [B, 1]
    add = L[b, pred, cust] + L[b, cust, succ]
    delete = jnp.where(pred != succ, L[b, pred, succ], 0.0)
    return add - delete


def gather_directed_features(Z, cust, pred, succ):
    """按 (customer, predecessor, successor) 有序拼接 Z 端点 → [B, M, 3d]（有方向）。"""
    B, M = cust.shape
    b = jnp.arange(B)[:, None]
    return jnp.concatenate([Z[b, cust], Z[b, pred], Z[b, succ]], axis=-1)


class MpreTrainedModel(nnx.Module):
    """M-trained：backbone(CVRP512) + c_adapter + delta_head。"""

    def __init__(self, backbone, node_feat_dim, action_feat_dim, rngs=None, d_model=512):
        if rngs is None or isinstance(rngs, int):
            rngs = nnx.Rngs(rngs if rngs is not None else 0)
        self.backbone = backbone
        self.c_adapter = ColdChainAdapter(node_feat_dim, rngs=rngs)
        self.delta_head = DirectedInsertionHead(3 * d_model + action_feat_dim, rngs=rngs)

    def encode_residual(self, raw_3d, node_valid, node_feats):
        """encoder 前向 + 冷链残差。返回 (H0, H)。node_valid 用于 encoder padding 屏蔽。"""
        pair_valid = node_valid[..., None] * node_valid[..., None, :]
        encoder_bias = jnp.where(pair_valid, 0.0, -1e9)
        H0 = self.backbone.encode(raw_3d, attn_options={'bias': encoder_bias})
        H0 = jax.lax.stop_gradient(H0)
        H = H0 + self.c_adapter(node_feats)
        return H0, H

    def decode_logits(self, H, timestep, adjmat, node_valid):
        """decoder 前向：组合计划邻接 + padding 屏蔽 → 对称边 logits。"""
        pair_valid = node_valid[..., None] * node_valid[..., None, :]
        decoder_bias = jnp.where(pair_valid, adjmat, -1e9)
        Z = self.backbone.decode(H, timestep, decoder_bias, attn_options={}, target='embed')
        L = self.backbone.feature2logit(Z)
        return Z, L

    def score_actions(self, raw_3d, node_valid, node_feats, timestep, adjmat,
                      cust, pred, succ, action_feats):
        """完整前向 → 每个动作的标量分数（edge + directed delta）。[B, M]。"""
        H0, H = self.encode_residual(raw_3d, node_valid, node_feats)
        Z, L = self.decode_logits(H, timestep, adjmat, node_valid)
        s_edge = edge_insertion_scores(L, cust, pred, succ)
        directed = gather_directed_features(Z, cust, pred, succ)
        delta_in = jnp.concatenate([directed, action_feats], axis=-1)
        delta = self.delta_head(delta_in)
        return s_edge + delta, (H0, Z, L, s_edge, delta)

    def score_from_H(self, H, timestep, adjmat, node_valid, cust, pred, succ, action_feats):
        """从已编码的 H 解码 + 评分（跳过 encoder，供 JIT 调用）。返回 [B, M]。"""
        Z, L = self.decode_logits(H, timestep, adjmat, node_valid)
        s_edge = edge_insertion_scores(L, cust, pred, succ)
        directed = gather_directed_features(Z, cust, pred, succ)
        delta_in = jnp.concatenate([directed, action_feats], axis=-1)
        delta = self.delta_head(delta_in)
        return s_edge + delta

    def encode_H0(self, raw_3d, node_valid):
        """冻结 encoder 前向（H0），供同一事件内缓存复用。"""
        pair_valid = node_valid[..., None] * node_valid[..., None, :]
        encoder_bias = jnp.where(pair_valid, 0.0, -1e9)
        return jax.lax.stop_gradient(self.backbone.encode(raw_3d, attn_options={'bias': encoder_bias}))

    def score_from_H0(self, H0, node_feats, timestep, adjmat, node_valid, cust, pred, succ,
                      action_feats):
        """从缓存的 H0 计算 C 残差 + decoder + δ 评分。返回 [B, M]。"""
        H = H0 + self.c_adapter(node_feats)
        return self.score_from_H(H, timestep, adjmat, node_valid, cust, pred, succ, action_feats)


# --------------------------------------------------------------------------- #
# 参数分区
# --------------------------------------------------------------------------- #

FROZEN_ROOTS = {'init_proj', 'encoder', 'depot_bias'}
TRAINABLE_ROOTS = {'mid_proj', 'timestep_embedder', 'decoder', 'final_proj', 'final_norm'}


def _is_trainable(path, value):
    return (path[0] in {'c_adapter', 'delta_head'}
            or (len(path) > 1 and path[0] == 'backbone' and path[1] in TRAINABLE_ROOTS))


def _is_frozen(path, value):
    return (len(path) > 1 and path[0] == 'backbone' and path[1] in FROZEN_ROOTS)


def partition(model):
    """→ (graphdef, train_state, frozen_state, other_state)。other_state 应为空（无 Param 漏网）。"""
    return nnx.split(model,
                     nnx.All(nnx.Param, _is_trainable),
                     nnx.All(nnx.Param, _is_frozen),
                     nnx.Param)


def param_manifest(model):
    """逐参数审计清单：[(path, shape, dtype, trainable)]。"""
    gd, train, frozen, other = partition(model)
    out = []
    def _collect(state, trainable):
        for p, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            path = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in p)
            out.append((path, tuple(leaf.shape), str(leaf.dtype), trainable))
    _collect(train, True)
    _collect(frozen, False)
    _collect(other, None)   # leak
    return out


def load_mpre_trained(cvrp_ckpt, node_feat_dim, action_feat_dim, seed=0):
    """加载 cvrp100.ckpt → MpreTrainedModel。"""
    _ensure_tensorboardx_stub()
    _setup_maskco_paths()
    backbone, cfg, step = load_cvrp_model(cvrp_ckpt)
    d_model = cfg.embed_dim[1]
    model = MpreTrainedModel(backbone, node_feat_dim, action_feat_dim, rngs=seed,
                             d_model=d_model)
    return model, cfg, step
