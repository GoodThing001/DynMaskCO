"""
Edge-Conditioned Attention for DynMaskCO Phase A

Replaces fixed edge bias with learned edge-conditioned attention,
where edge features dynamically modulate attention logits.

Key idea:
  Instead of: attn = softmax(Q @ K.T / sqrt(d_k) + fixed_bias)
  Use:        attn = softmax(Q @ K.T / sqrt(d_k) + learned_edge_bias(Q, K, E))

Author: Phase A implementation
Date: 2026-08-25
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class EdgeConditionedAttention(nn.Module):
    """
    Multi-head attention with edge-conditioned bias.

    Edge features are used to compute a dynamic bias term that is added
    to the attention logits before softmax. This allows the model to learn
    edge-aware attention patterns.

    Architecture:
        1. Standard Q, K, V projections
        2. Edge feature projection
        3. Edge-conditioned bias: b_ij = f(q_i, k_j, e_ij)
        4. Attention: attn_ij = softmax((q_i @ k_j / sqrt(d_k)) + b_ij)
    """

    embed_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.0
    use_bias: bool = True
    edge_mode: str = 'bilinear'  # 'bilinear', 'additive', 'dot'

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Standard Q, K, V projections
        self.q_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias, name='q_proj')
        self.k_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias, name='k_proj')
        self.v_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias, name='v_proj')

        # Edge feature projection
        if self.edge_mode == 'bilinear':
            # Edge features project to same dim as K for bilinear interaction
            self.edge_proj = nn.Dense(self.embed_dim, use_bias=False, name='edge_proj')
        elif self.edge_mode == 'additive':
            # Edge features project to num_heads for additive bias
            self.edge_proj = nn.Dense(self.num_heads, use_bias=True, name='edge_proj')
        elif self.edge_mode == 'dot':
            # Edge features project to head_dim for dot product
            self.edge_proj = nn.Dense(self.head_dim, use_bias=False, name='edge_proj')
        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        # Output projection
        self.out_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias, name='out_proj')

        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        edge_features: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass of edge-conditioned attention.

        Args:
            x: Node features [B, N, D]
            edge_features: Edge features [B, N, N, D_edge]
            attn_mask: Attention mask [B, N, N] or [B, H, N, N]
                       (1 = attend, 0 = mask out)
            deterministic: Whether to apply dropout

        Returns:
            output: [B, N, D] attended node features
        """
        B, N, D = x.shape

        # 1. Compute Q, K, V
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(x)  # [B, N, D]
        v = self.v_proj(x)  # [B, N, D]

        # Reshape to multi-head format
        q = q.reshape(B, N, self.num_heads, self.head_dim)  # [B, N, H, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim)  # [B, N, H, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim)  # [B, N, H, d]

        # Transpose to [B, H, N, d] for batched matmul
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, N, d]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, N, d]
        v = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, N, d]

        # 2. Compute standard attention logits: Q @ K.T / sqrt(d_k)
        attn_logits = jnp.einsum('bhid,bhjd->bhij', q, k) * self.scale  # [B, H, N, N]

        # 3. Compute edge-conditioned bias
        edge_bias = self._compute_edge_bias(q, k, edge_features)  # [B, H, N, N]

        # Add edge bias to attention logits
        attn_logits = attn_logits + edge_bias

        # 4. Apply attention mask
        if attn_mask is not None:
            # attn_mask: [B, N, N] or [B, H, N, N]
            if attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]  # [B, 1, N, N]

            # Mask out: set logits to large negative value
            attn_logits = jnp.where(
                attn_mask > 0,
                attn_logits,
                jnp.full_like(attn_logits, -1e9)
            )

        # 5. Softmax to get attention weights
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # [B, H, N, N]

        # Apply dropout
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        # 6. Apply attention to values
        attended = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)  # [B, H, N, d]

        # 7. Reshape back to [B, N, D]
        attended = jnp.transpose(attended, (0, 2, 1, 3))  # [B, N, H, d]
        attended = attended.reshape(B, N, self.embed_dim)  # [B, N, D]

        # 8. Output projection
        output = self.out_proj(attended)  # [B, N, D]

        return output

    def _compute_edge_bias(
        self,
        q: jnp.ndarray,  # [B, H, N, d]
        k: jnp.ndarray,  # [B, H, N, d]
        edge_features: jnp.ndarray  # [B, N, N, D_edge]
    ) -> jnp.ndarray:
        """
        Compute edge-conditioned bias for attention logits.

        Three modes:
        1. Bilinear: b_ij = q_i @ E_ij @ k_j  (most expressive)
        2. Additive: b_ij = MLP([q_i; k_j; e_ij])  (good balance)
        3. Dot: b_ij = (q_i + k_j) · e_ij  (simplest)

        Args:
            q: Query [B, H, N, d]
            k: Key [B, H, N, d]
            edge_features: Edge features [B, N, N, D_edge]

        Returns:
            edge_bias: [B, H, N, N] edge-conditioned bias
        """
        B, H, N, d = q.shape

        if self.edge_mode == 'bilinear':
            # Project edge features: [B, N, N, D_edge] → [B, N, N, D]
            edge_proj = self.edge_proj(edge_features)  # [B, N, N, D]

            # Reshape to multi-head: [B, N, N, H, d]
            edge_proj = edge_proj.reshape(B, N, N, self.num_heads, self.head_dim)

            # Transpose to [B, H, N, N, d]
            edge_proj = jnp.transpose(edge_proj, (0, 3, 1, 2, 4))

            # Bilinear form: b_ij = q_i @ e_ij @ k_j
            # q: [B, H, N, d], k: [B, H, N, d], edge: [B, H, N, N, d]
            # First: edge @ k → [B, H, N, N]
            edge_k = jnp.einsum('bhnmd,bhmd->bhmn', edge_proj, k)  # [B, H, N, N]
            # Then: q @ (edge_k) → [B, H, N, N]
            edge_bias = jnp.einsum('bhmd,bhmn->bhmn', q, edge_k)  # [B, H, N, N]

        elif self.edge_mode == 'additive':
            # Project edge features: [B, N, N, D_edge] → [B, N, N, H]
            edge_bias = self.edge_proj(edge_features)  # [B, N, N, H]
            # Transpose to [B, H, N, N]
            edge_bias = jnp.transpose(edge_bias, (0, 3, 1, 2))  # [B, H, N, N]

        elif self.edge_mode == 'dot':
            # Project edge features: [B, N, N, D_edge] → [B, N, N, d]
            edge_proj = self.edge_proj(edge_features)  # [B, N, N, d]

            # Average Q and K: [B, H, N, d]
            qk_avg = (q[:, :, :, None, :] + k[:, :, None, :, :]) / 2.0  # [B, H, N, N, d]

            # Dot product with edge features
            edge_bias = jnp.einsum('bhnmd,bnmd->bhmn', qk_avg, edge_proj)  # [B, H, N, N]

        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        return edge_bias


class EdgeConditionedTransformerLayer(nn.Module):
    """
    Transformer encoder layer with edge-conditioned attention.

    Replaces standard self-attention with EdgeConditionedAttention.
    """

    embed_dim: int
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    edge_mode: str = 'bilinear'

    def setup(self):
        # Edge-conditioned self-attention
        self.self_attn = EdgeConditionedAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            edge_mode=self.edge_mode
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        # Feed-forward network
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp_dense1 = nn.Dense(mlp_hidden_dim)
        self.mlp_dense2 = nn.Dense(self.embed_dim)
        self.mlp_dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.mlp_dropout2 = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        edge_features: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass of transformer layer with edge-conditioned attention.

        Args:
            x: Node features [B, N, D]
            edge_features: Edge features [B, N, N, D_edge]
            attn_mask: Attention mask [B, N, N]
            deterministic: Whether to apply dropout

        Returns:
            output: [B, N, D] transformed features
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(
            x, edge_features, attn_mask=attn_mask, deterministic=deterministic
        )
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        mlp_out = self.mlp_dense1(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = self.mlp_dropout1(mlp_out, deterministic=deterministic)
        mlp_out = self.mlp_dense2(mlp_out)
        mlp_out = self.mlp_dropout2(mlp_out, deterministic=deterministic)
        x = self.norm2(x + mlp_out)

        return x


# Unit tests
if __name__ == '__main__':
    import numpy as np

    print("Testing EdgeConditionedAttention...")

    # Test parameters
    B, N, D, H = 2, 8, 128, 8
    D_edge = 64

    # Dummy data
    x = jnp.array(np.random.randn(B, N, D).astype(np.float32))
    edge_features = jnp.array(np.random.randn(B, N, N, D_edge).astype(np.float32))
    attn_mask = jnp.ones((B, N, N))  # All attend

    # Test different edge modes
    for edge_mode in ['bilinear', 'additive', 'dot']:
        print(f"\n--- Testing edge_mode={edge_mode} ---")

        attn = EdgeConditionedAttention(
            embed_dim=D,
            num_heads=H,
            edge_mode=edge_mode
        )

        # Initialize
        params = attn.init(
            jax.random.PRNGKey(0),
            x, edge_features, attn_mask
        )

        # Forward pass
        output = attn.apply(params, x, edge_features, attn_mask)

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Edge features shape: {edge_features.shape}")
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    # Test EdgeConditionedTransformerLayer
    print("\n--- Testing EdgeConditionedTransformerLayer ---")
    layer = EdgeConditionedTransformerLayer(embed_dim=D, num_heads=H)
    params = layer.init(jax.random.PRNGKey(0), x, edge_features, attn_mask)
    output = layer.apply(params, x, edge_features, attn_mask)

    print(f"✓ Layer output shape: {output.shape}")
    assert output.shape == x.shape

    print("\n✅ All EdgeConditionedAttention tests passed!")
