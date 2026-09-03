"""
Edge Bias Projector for Phase A1 (Flax NNX)

Projects 5D static pairwise edge features to multi-head attention bias.

Architecture:
    edge_features [B, N, N, 5]
        → Linear(5, num_heads)
        → transpose to [B, H, N, N]
        → added to attention logits

Author: Phase A1 implementation
Date: 2026-08-25 (corrected for NNX)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


class EdgeBiasProjector(nnx.Module):
    """
    Projects edge features to multi-head attention bias.

    This is the ONLY trainable component in Phase A1.
    All other edge computation is pure JAX (edge_features.py).

    Usage:
        projector = EdgeBiasProjector(edge_dim=5, num_heads=8, rngs=nnx.Rngs(42))
        edge_bias = projector(edge_features)  # [B,N,N,5] → [B,H,N,N]
    """

    def __init__(
        self,
        edge_dim: int = 5,
        num_heads: int = 8,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        """
        Initialize EdgeBiasProjector.

        Args:
            edge_dim: Dimension of input edge features (default 5)
            num_heads: Number of attention heads (default 8)
            use_bias: Whether to use bias in linear projection
            rngs: NNX random number generator
        """
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        # Linear projection: [B, N, N, edge_dim] → [B, N, N, num_heads]
        # 关键：kernel 初始化为接近 0（安全 no-op），与父类 edge_weight 的 init=0 一致。
        # 否则训练初期边 bias 就非零，干扰训练（Day 7 screening 暴露的问题）。
        self.proj = nnx.Linear(
            in_features=edge_dim,
            out_features=num_heads,
            use_bias=use_bias,
            kernel_init=nnx.initializers.normal(1e-6),
            rngs=rngs
        )

    def __call__(self, edge_features: jnp.ndarray) -> jnp.ndarray:
        """
        Project edge features to multi-head attention bias.

        Args:
            edge_features: [B, N, N, edge_dim] edge features

        Returns:
            edge_bias: [B, H, N, N] multi-head attention bias
        """
        B, N, N_check, edge_dim = edge_features.shape
        assert N == N_check, f"Edge features must be square: {N} != {N_check}"
        assert edge_dim == self.edge_dim, f"Edge dim mismatch: {edge_dim} != {self.edge_dim}"

        # Project: [B, N, N, edge_dim] → [B, N, N, num_heads]
        edge_bias = self.proj(edge_features)  # [B, N, N, H]

        # Transpose to [B, H, N, N] for multi-head attention
        edge_bias = jnp.transpose(edge_bias, (0, 3, 1, 2))  # [B, H, N, N]

        return edge_bias


# ============================================================================
# Unit tests
# ============================================================================

if __name__ == '__main__':
    import numpy as np

    print("Testing EdgeBiasProjector (Flax NNX)...")

    # Test parameters
    B, N, edge_dim, num_heads = 2, 8, 5, 8

    # Create dummy edge features
    np.random.seed(42)
    edge_features = jnp.array(np.random.randn(B, N, N, edge_dim).astype(np.float32))

    # Initialize projector
    rngs = nnx.Rngs(42)
    projector = EdgeBiasProjector(
        edge_dim=edge_dim,
        num_heads=num_heads,
        rngs=rngs
    )

    print(f"✓ EdgeBiasProjector initialized")
    print(f"  - edge_dim: {projector.edge_dim}")
    print(f"  - num_heads: {projector.num_heads}")
    print(f"  - proj params: {projector.proj.kernel.value.shape}")

    # Forward pass
    edge_bias = projector(edge_features)

    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {edge_features.shape}")
    print(f"  - Output shape: {edge_bias.shape}")
    assert edge_bias.shape == (B, num_heads, N, N), f"Expected {(B, num_heads, N, N)}, got {edge_bias.shape}"

    # Test JIT compilation
    @jax.jit
    def jit_forward(edge_features):
        return projector(edge_features)

    edge_bias_jit = jit_forward(edge_features)
    assert jnp.allclose(edge_bias, edge_bias_jit), "JIT result differs from eager"
    print(f"✓ JIT compilation successful")

    # Test equivalence: if we zero out 4 dimensions, only energy dimension remains
    edge_features_energy_only = jnp.zeros((B, N, N, edge_dim))
    edge_features_energy_only = edge_features_energy_only.at[..., 3].set(edge_features[..., 3])  # Keep energy only

    edge_bias_energy = projector(edge_features_energy_only)
    print(f"✓ Energy-only bias computed")

    # Test with different batch sizes
    edge_features_small = jnp.array(np.random.randn(1, 4, 4, edge_dim).astype(np.float32))
    edge_bias_small = projector(edge_features_small)
    assert edge_bias_small.shape == (1, num_heads, 4, 4)
    print(f"✓ Different batch size works")

    # Test parameter count
    kernel_params = projector.proj.kernel.value.size
    bias_params = projector.proj.bias.value.size if projector.proj.use_bias else 0
    total_params = kernel_params + bias_params
    print(f"✓ Total parameters: {total_params} ({edge_dim}×{num_heads} + {num_heads} = {kernel_params} + {bias_params})")

    print("\n✅ All EdgeBiasProjector tests passed!")
    print("")
    print("Summary:")
    print(f"  ✓ NNX Module with {total_params} parameters")
    print(f"  ✓ Projects [B,N,N,{edge_dim}] → [B,{num_heads},N,N]")
    print(f"  ✓ JIT compatible")
    print(f"  ✓ Ready for DynamicColdChainModel integration")
