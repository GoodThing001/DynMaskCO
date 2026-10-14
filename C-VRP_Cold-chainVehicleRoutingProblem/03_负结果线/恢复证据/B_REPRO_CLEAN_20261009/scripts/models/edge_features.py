"""
Edge Feature Computation for Phase A1 (Pure JAX)

Computes 5D static pairwise edge features:
1. Distance: Euclidean distance
2. Travel time: distance / vehicle_speed
3. Spatial angle: direction from i to j
4. Energy: existing energy matrix (temperature-aware)
5. TW compatibility: temporal feasibility slack

NO trainable parameters - pure JAX tensor operations.

Author: Phase A1 implementation
Date: 2026-08-25 (corrected)
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def compute_static_pairwise_edge_features(
    coords: jnp.ndarray,           # [B, N, 2] (x, y)
    tw_start: jnp.ndarray,         # [B, N]
    tw_end: jnp.ndarray,           # [B, N]
    dist_mat: jnp.ndarray,         # [B, N, N]
    energy_mat: Optional[jnp.ndarray] = None,  # [B, N, N] (existing)
    vehicle_speed: float = 1.0,
    visible_mask: Optional[jnp.ndarray] = None  # [B, N]
) -> jnp.ndarray:
    """
    Compute 5D static pairwise edge features.

    Args:
        coords: Node coordinates [B, N, 2]
        tw_start: Time window start [B, N]
        tw_end: Time window end [B, N]
        dist_mat: Precomputed distance matrix [B, N, N]
        energy_mat: Existing energy matrix [B, N, N] (from DynMaskCO)
        vehicle_speed: Vehicle speed (default 1.0)
        visible_mask: Causality mask [B, N] (1=visible, 0=future)

    Returns:
        edge_features: [B, N, N, 5] static pairwise edge features
            dim 0: distance
            dim 1: travel_time
            dim 2: angle (normalized to [0, 1])
            dim 3: energy (existing energy_mat)
            dim 4: tw_compatibility
    """
    B, N, _ = coords.shape

    # 1. Distance (already computed)
    distance_feat = dist_mat  # [B, N, N]

    # 2. Travel time = distance / speed
    travel_time = dist_mat / vehicle_speed  # [B, N, N]

    # 3. Spatial angle: arctan2(dy, dx) from node i to j
    x_coords = coords[..., 0]  # [B, N]
    y_coords = coords[..., 1]  # [B, N]

    dx = x_coords[:, None, :] - x_coords[:, :, None]  # [B, N, N]
    dy = y_coords[:, None, :] - y_coords[:, :, None]  # [B, N, N]
    angle = jnp.arctan2(dy, dx)  # [B, N, N] in [-π, π]

    # Normalize angle to [0, 1] for better numerical stability
    angle_normalized = (angle + jnp.pi) / (2 * jnp.pi)  # [B, N, N]

    # 4. Energy (use existing energy_mat if provided, else zeros)
    if energy_mat is None:
        energy_feat = jnp.zeros_like(dist_mat)  # [B, N, N]
    else:
        energy_feat = energy_mat  # [B, N, N]

    # 5. TW compatibility: slack time for edge (i→j)
    # tw_compat[i,j] = max(0, tw_end[i] - tw_start[j] - travel_time[i,j])
    tw_end_i = tw_end[:, :, None]  # [B, N, 1]
    tw_start_j = tw_start[:, None, :]  # [B, 1, N]
    tw_slack = tw_end_i - tw_start_j - travel_time  # [B, N, N]
    tw_compat = jnp.maximum(0.0, tw_slack)  # [B, N, N]

    # Stack all features: [B, N, N, 5]，归一化到相似尺度（避免大尺度特征主导梯度）
    # distance/travel_time: / sqrt(2) → 0-1（[0,1]^2 坐标最大距离）
    # angle: 已 0-1
    # energy/tw_compat: per-batch max 归一化 → 0-1
    max_energy = energy_feat.max(axis=(-1, -2), keepdims=True) + 1e-6
    max_tw = tw_compat.max(axis=(-1, -2), keepdims=True) + 1e-6

    edge_features = jnp.stack([
        distance_feat / (2.0 ** 0.5),
        travel_time / (2.0 ** 0.5),
        angle_normalized,
        energy_feat / max_energy,
        tw_compat / max_tw,
    ], axis=-1)  # [B, N, N, 5]

    # Apply causality mask if provided
    if visible_mask is not None:
        # Only keep edges where both nodes are visible
        mask_i = visible_mask[:, :, None]  # [B, N, 1]
        mask_j = visible_mask[:, None, :]  # [B, 1, N]
        edge_mask = mask_i * mask_j  # [B, N, N]
        edge_features = edge_features * edge_mask[..., None]  # [B, N, N, 5]

    return edge_features


def compute_travel_time_matrix(
    dist_mat: jnp.ndarray,
    vehicle_speed: float = 1.0
) -> jnp.ndarray:
    """
    Compute travel time matrix from distance matrix.

    Args:
        dist_mat: [B, N, N] distance matrix
        vehicle_speed: vehicle speed

    Returns:
        time_mat: [B, N, N] travel time matrix
    """
    return dist_mat / vehicle_speed


def compute_tw_compatibility_matrix(
    tw_start: jnp.ndarray,      # [B, N]
    tw_end: jnp.ndarray,        # [B, N]
    travel_time: jnp.ndarray    # [B, N, N]
) -> jnp.ndarray:
    """
    Compute TW compatibility matrix.

    TW_compat[i,j] = max(0, tw_end[i] - tw_start[j] - travel_time[i,j])

    Positive values indicate temporal slack; negative means infeasible direct edge.

    Args:
        tw_start: [B, N] time window start
        tw_end: [B, N] time window end
        travel_time: [B, N, N] travel time matrix

    Returns:
        tw_compat: [B, N, N] TW compatibility matrix
    """
    tw_end_i = tw_end[:, :, None]  # [B, N, 1]
    tw_start_j = tw_start[:, None, :]  # [B, 1, N]
    tw_slack = tw_end_i - tw_start_j - travel_time
    return jnp.maximum(0.0, tw_slack)


# ============================================================================
# Unit tests
# ============================================================================

if __name__ == '__main__':
    import numpy as np

    print("Testing edge feature computation (pure JAX)...")

    # Create dummy data
    B, N = 2, 8
    np.random.seed(42)

    coords = jnp.array(np.random.rand(B, N, 2))
    tw_start = jnp.array(np.random.rand(B, N) * 10)
    tw_end = tw_start + jnp.array(np.random.rand(B, N) * 5 + 5)

    # Distance matrix (Euclidean)
    diff = coords[:, :, None, :] - coords[:, None, :, :]  # [B, N, N, 2]
    dist_mat = jnp.sqrt((diff ** 2).sum(axis=-1))  # [B, N, N]

    # Energy matrix (dummy)
    energy_mat = jnp.array(np.random.rand(B, N, N) * 10)

    # Visible mask (causality)
    visible_mask = jnp.array(np.random.rand(B, N) > 0.3).astype(jnp.float32)

    # Compute static edge features
    edge_features = compute_static_pairwise_edge_features(
        coords, tw_start, tw_end, dist_mat, energy_mat,
        vehicle_speed=1.0, visible_mask=visible_mask
    )

    print(f"✓ Edge features shape: {edge_features.shape}")  # Should be [B, N, N, 5]
    assert edge_features.shape == (B, N, N, 5), f"Expected {(B, N, N, 5)}, got {edge_features.shape}"

    # Check individual features
    print(f"✓ Feature 0 (distance) range: [{edge_features[..., 0].min():.3f}, {edge_features[..., 0].max():.3f}]")
    print(f"✓ Feature 1 (travel_time) range: [{edge_features[..., 1].min():.3f}, {edge_features[..., 1].max():.3f}]")
    print(f"✓ Feature 2 (angle) range: [{edge_features[..., 2].min():.3f}, {edge_features[..., 2].max():.3f}]")
    print(f"✓ Feature 3 (energy) range: [{edge_features[..., 3].min():.3f}, {edge_features[..., 3].max():.3f}]")
    print(f"✓ Feature 4 (tw_compat) range: [{edge_features[..., 4].min():.3f}, {edge_features[..., 4].max():.3f}]")

    # Check angle is in [0, 1]
    assert edge_features[..., 2].min() >= 0.0, "Angle should be >= 0"
    assert edge_features[..., 2].max() <= 1.0, "Angle should be <= 1"

    # Check causality: future nodes should have zero edge features
    future_mask = (1 - visible_mask).astype(bool)
    if future_mask.any():
        # Edges involving future nodes should be masked
        for b in range(B):
            for i in range(N):
                if future_mask[b, i]:
                    # Edges to/from this future node should be zero
                    assert jnp.allclose(edge_features[b, i, :, :], 0.0), "Future node edges not masked (row)"
                    assert jnp.allclose(edge_features[b, :, i, :], 0.0), "Future node edges not masked (col)"
        print(f"✓ Causality mask applied correctly")

    # Test helper functions
    travel_time = compute_travel_time_matrix(dist_mat, vehicle_speed=1.0)
    assert travel_time.shape == (B, N, N)
    print(f"✓ Travel time matrix shape: {travel_time.shape}")

    tw_compat = compute_tw_compatibility_matrix(tw_start, tw_end, travel_time)
    assert tw_compat.shape == (B, N, N)
    print(f"✓ TW compatibility matrix shape: {tw_compat.shape}")

    print("\n✅ All edge feature tests passed!")
    print("")
    print("Summary:")
    print("  ✓ 5D static pairwise edge features")
    print("  ✓ Causality mask respected")
    print("  ✓ Pure JAX (no trainable parameters)")
    print("  ✓ Ready for EdgeBiasProjector integration")
