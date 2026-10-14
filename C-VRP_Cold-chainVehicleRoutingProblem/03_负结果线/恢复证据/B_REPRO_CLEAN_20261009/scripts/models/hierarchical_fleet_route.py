"""
Hierarchical Fleet–Route Masked Reconstruction（HFR-M0）—— 模型（导师 v3 文档 §9-15）。

把 MaskCO 从单层 route adjacency reconstruction 升级为 coarse fleet grouping + fine route
reconstruction 的层次重构。关键：A 必须显式依赖 G（A_joint = A_base + ΔA + β·f(G)），不能 G∥A。

wrapper-first：不动原 MaskCO checkpoint 参数树。Causal MaskCO Encoder + 原 Route Head 都 frozen，
只训 GroupingHead + RouteResidualAdapter。

结构：
  H = frozen encoder(raw_features)
  G_logits = GroupingHead(H)          （对称，pairwise same-route）
  A_base   = frozen route head(H)     （原 MaskCO edge logits）
  ΔA       = RouteResidualAdapter(H, G_logits)
  A_joint  = A_base + ΔA + β·log_sigmoid(G_logits)
"""
import jax
import jax.numpy as jnp
from flax import nnx


class GroupingHead(nnx.Module):
    """G_ij = 1[i,j same route] 的 pairwise 预测。对称 pair feature。"""

    def __init__(self, d_model=256, hidden=256, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.fc1 = nnx.Linear(3 * d_model, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden // 2, rngs=rngs)
        self.out = nnx.Linear(hidden // 2, 1, rngs=rngs)

    def __call__(self, H):
        hi = H[:, :, None, :]          # [B,N,1,D]
        hj = H[:, None, :, :]          # [B,1,N,D]
        phi = jnp.concatenate([hi + hj, jnp.abs(hi - hj), hi * hj], axis=-1)  # [B,N,N,3D]
        x = jax.nn.silu(self.fc1(phi))
        x = jax.nn.silu(self.fc2(x))
        logits = self.out(x)[..., 0]   # [B,N,N]
        # 强制数值对称（G_ij = G_ji）
        logits = 0.5 * (logits + jnp.swapaxes(logits, 1, 2))
        return logits


class RouteResidualAdapter(nnx.Module):
    """ΔA：让 route reconstruction 显式依赖 G。有向（route 是有方向的，不需对称）。"""

    def __init__(self, d_model=256, hidden=128, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.fc1 = nnx.Linear(2 * d_model + 1, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, H, group_logits):
        hi = H[:, :, None, :]          # [B,N,1,D]
        hj = H[:, None, :, :]          # [B,1,N,D]
        p_same = jax.nn.sigmoid(group_logits)[..., None]   # [B,N,N,1]
        hi_full = jnp.broadcast_to(hi, (*group_logits.shape, H.shape[-1]))
        hj_full = jnp.broadcast_to(hj, (*group_logits.shape, H.shape[-1]))
        phi = jnp.concatenate([hi_full, hj_full, p_same], axis=-1)  # [B,N,N,2D+1]
        x = jax.nn.silu(self.fc1(phi))
        return self.out(x)[..., 0]     # [B,N,N]


class HierarchicalFleetRouteModel(nnx.Module):
    """HFR-M0 wrapper：frozen base MaskCO + trainable GroupingHead + RouteResidualAdapter。"""

    def __init__(self, base_maskco, beta=1.0, d_model=256, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.base = base_maskco
        self.group_head = GroupingHead(d_model=d_model, rngs=rngs)
        self.route_adapter = RouteResidualAdapter(d_model=d_model, rngs=rngs)
        self.beta = beta

    def __call__(self, raw_features, visible_mask, partial_adj, timestep):
        H = self.base.encode(raw_features, visible_mask=visible_mask)
        H = jax.lax.stop_gradient(H)

        G_logits = self.group_head(H)

        A_base = self.base.decode(H, timestep, partial_adj)
        A_base = jax.lax.stop_gradient(A_base)

        delta_A = self.route_adapter(H, G_logits)
        group_bias = jax.nn.log_sigmoid(G_logits)

        A_joint = A_base + delta_A + self.beta * group_bias

        return {
            'group_logits': G_logits,          # [B,N,N]
            'route_logits_base': A_base,       # [B,N,N]
            'route_logits_joint': A_joint,     # [B,N,N]
        }
