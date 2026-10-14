"""
CVRPTW utility functions — visible-aware coordinate normalization.

Fixes the normalization leakage: parent MaskCO's coord_normalize computes
mean/norm over ALL nodes (axis=-2), so future nodes' coordinates affect
visible node normalization statistics. This creates a non-anticipatory leak.

coord_normalize_visible computes statistics from visible nodes only,
then applies the same transformation to ALL nodes.
"""

import jax.numpy as jnp


def coord_normalize_visible(inputs, visible_mask=None):
    """Normalize coordinates using statistics from visible nodes only.

    When visible_mask is None, falls back to standard coord_normalize
    (statistics over all nodes). When visible_mask is provided, computes
    mean and L2-norm statistics from visible nodes only, then applies
    the same centering and scaling to ALL nodes (including future ones).

    This ensures visible nodes' normalized coordinates are invariant to
    future node properties — satisfying non-anticipativity.
    """
    if visible_mask is None:
        # Fallback: use all nodes (same as parent coord_normalize)
        inputs = inputs - inputs.mean(axis=-2, keepdims=True)
        inputs_norm = jnp.linalg.norm(inputs, axis=-1, keepdims=True, ord=2)
        inputs_norm_mean = inputs_norm.mean(axis=-2, keepdims=True)
        inputs = inputs / inputs_norm_mean
        return inputs

    # Expand mask to coordinate dimension: (B, N) -> (B, N, 1) or (N,) -> (N, 1)
    vis = visible_mask.astype(inputs.dtype)
    while vis.ndim < inputs.ndim:
        vis = vis[..., None]

    # === Centering: mean over visible nodes only ===
    masked = inputs * vis
    vis_count = vis.sum(axis=-2, keepdims=True).clip(min=1)
    vis_mean = masked.sum(axis=-2, keepdims=True) / vis_count
    inputs = inputs - vis_mean

    # === Scaling: mean L2-norm over visible nodes only ===
    # Recompute masked after centering
    masked = inputs * vis
    norms = jnp.linalg.norm(masked, axis=-1, keepdims=True, ord=2)
    norms_mean = norms.sum(axis=-2, keepdims=True) / vis_count.squeeze(-1)[..., None]
    inputs = inputs / norms_mean.clip(min=1e-8)

    return inputs


def _project_route(route, visible_mask):
    """projection 共享实现，返回 (projected, order, kept)。"""
    B, L = route.shape
    node = route
    vis = visible_mask.astype(jnp.float32)
    # 每个 route 节点的可见性（depot 恒保留）
    vis_node = jnp.take_along_axis(vis, node, axis=1)
    keep = (node == 0) | (vis_node > 0.5)
    # stable sort：keep 节点按原顺序排前，~keep 排后 → 删除 future 且保持可见顺序
    order = jnp.argsort((~keep).astype(jnp.int32), axis=1, stable=True)
    projected = jnp.take_along_axis(node, order, axis=1)
    kept = jnp.take_along_axis(keep, order, axis=1)
    projected = jnp.where(kept, projected, 0)
    return projected.astype(node.dtype), order, kept


def project_route_to_visible(route, visible_mask):
    """删除 route 中 future 客户，保持 depot separator 与可见节点顺序，pad 到原长度。

    导师 D2/B2：训练 target 需先 visible route projection（删 future 重连可见），
    再 sol2adj。三条规则：① future 客户删除；② 同 segment 内可见节点自然重连；
    ③ depot separator 保留（不跨 depot 重连）。

    route: (B, L) int，depot=0 分隔、0 padding
    visible_mask: (B, N+1) float/bool，1=可见（depot 恒可见）
    Returns: (B, L) int（projected route，future 已删，顺序保持，pad 补 0）
    """
    projected, _, _ = _project_route(route, visible_mask)
    return projected


def build_causal_target_adj(target, visible_mask, num_nodes):
    """构建 causal 训练的 loss target adjacency（B2 正式口径）。

    流程：full target → visible route projection → sol2adj → vis×vis 兜底。
    production（train_dynamic_cc.py）与 test（C1）共用，避免测试/生产分叉。

    target: (B, L) int route（depot=0 分隔）
    visible_mask: (B, N+1) float/bool，1=可见
    num_nodes: int（客户数，adj 维度 = num_nodes+1）
    Returns: (B, num_nodes+1, num_nodes+1) float32
    """
    from helpers import sol2adj  # lazy import（父项目 helpers，避免模块级循环依赖）
    if visible_mask is None:
        return sol2adj(target, dtype=jnp.float32, is_cvrp=True, num_nodes=num_nodes + 1)
    projected = project_route_to_visible(target, visible_mask)
    edge_vis = visible_mask[:, None, :] * visible_mask[:, :, None]
    tgt = sol2adj(projected, dtype=jnp.float32, is_cvrp=True, num_nodes=num_nodes + 1)
    return tgt * edge_vis.astype(tgt.dtype)


def build_causal_training_adj(target, route_mask, visible_mask, num_nodes):
    """构建 causal 训练的 decoder 输入 adjacency（B2 正式口径）。

    流程：full target → visible route projection → sol2adj_with_mask → vis×vis 兜底。

    target: (B, L) int route（depot=0 分隔）
    route_mask: (B, L) bool，MaskCO solution mask（保留哪些边）
    visible_mask: (B, N+1) float/bool，1=可见
    num_nodes: int（客户数，adj 维度 = num_nodes+1）
    Returns: (B, num_nodes+1, num_nodes+1) float16
    """
    from helpers import sol2adj_with_mask  # lazy import
    if visible_mask is None:
        return sol2adj_with_mask(target, mask=route_mask, dtype=jnp.float16,
                                 is_cvrp=True, num_nodes=num_nodes + 1)
    projected, order, kept = _project_route(target, visible_mask)
    # mask 随 projection 重排（对齐 projected route，避免 mask 与投影后边错位）
    mask_r = jnp.take_along_axis(route_mask.astype(jnp.int32), order, axis=1)
    mask_r = (mask_r & kept.astype(jnp.int32)).astype(bool)
    edge_vis = visible_mask[:, None, :] * visible_mask[:, :, None]
    cur = sol2adj_with_mask(projected, mask=mask_r, dtype=jnp.float16,
                            is_cvrp=True, num_nodes=num_nodes + 1)
    return cur * edge_vis.astype(cur.dtype)

