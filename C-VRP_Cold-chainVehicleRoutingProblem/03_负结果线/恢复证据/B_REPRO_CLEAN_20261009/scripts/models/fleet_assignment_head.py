"""
FleetAssignmentHead + JF2Runtime — JF2 的 fleet-level assignment residual scorer。

wrapper-first（主控文档 Step 6）：不动旧 MaskCO ckpt，把 FleetAssignmentHead 作为**独立可训练
module**，与 base MaskCO encoder 组合成 JF2Runtime。旧 ckpt 加载行为完全不变。

输入（canonical: N=51 nodes, K=25 vehicles, D=256 embed, Fv=车辆动态特征, Fp=pair 特征）：
  H              : [B, N, D]      节点 embedding（来自 base.encode）
  anchor_ids     : [B, K]         每辆车 anchor 节点 id（committed=committed_next，否则 current_node）
  veh_feat       : [B, K, Fv]     车辆动态特征（anchor_time/T, load/Q, remaining_cap/Q, status）
  pair_feat      : [B, K, N, Fp]  pair 特征（travel/ETA/TW-slack/scarcity/regret/...）
  base_score     : [B, K, N]      JF1-H min-travel base score（S_H）
  candidate_mask : [B, K, N]      sound hard mask（True=可行）

输出：
  score = base_score + alpha * residual,  candidate_mask 外置 -1e9
  residual : [B, K, N]

alpha 初始化为 0（主控文档 Step 8）：alpha=0 时 score == base_score，JF2 精确退化为 JF1-H。
"""
import jax
import jax.numpy as jnp
from flax import nnx


class FleetAssignmentHead(nnx.Module):
    def __init__(self, embed_dim=256, veh_dim=5, pair_dim=12, hidden_dim=256, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.vehicle_proj = nnx.Linear(embed_dim + veh_dim, hidden_dim, use_bias=True, rngs=rngs)
        self.customer_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=True, rngs=rngs)
        self.pair_mlp1 = nnx.Linear(hidden_dim * 2 + pair_dim, hidden_dim, use_bias=True, rngs=rngs)
        self.pair_mlp2 = nnx.Linear(hidden_dim, 1, use_bias=True, rngs=rngs)
        self.alpha = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

    def __call__(self, H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask):
        B, N, D = H.shape
        K = anchor_ids.shape[1]
        # anchor embedding: H[b, anchor_ids[b,k], :] -> [B, K, D]
        h_anchor = H[jnp.arange(B)[:, None], anchor_ids]
        v = jax.nn.silu(self.vehicle_proj(jnp.concatenate([h_anchor, veh_feat], axis=-1)))  # [B,K,H]
        c = jax.nn.silu(self.customer_proj(H))                                              # [B,N,H]

        v_expand = jnp.broadcast_to(v[:, :, None, :], (B, K, N, v.shape[-1]))
        c_expand = jnp.broadcast_to(c[:, None, :, :], (B, K, N, c.shape[-1]))

        z = jnp.concatenate([v_expand, c_expand, pair_feat], axis=-1)   # [B,K,N,2H+Fp]
        z = jax.nn.silu(self.pair_mlp1(z))
        residual = self.pair_mlp2(z)[..., 0]                            # [B,K,N]

        alpha = self.alpha.value.astype(residual.dtype)
        score = base_score + alpha * residual
        score = jnp.where(candidate_mask, score, -1e9)
        return score, residual


class JF2Runtime:
    """wrapper-first：base MaskCO encoder + 独立 FleetAssignmentHead 组合（不动旧 ckpt）。"""

    def __init__(self, base_maskco_model, fleet_head):
        self.base = base_maskco_model
        self.fleet_head = fleet_head

    def encode(self, *args, **kwargs):
        """委托 base MaskCO encoder，返回节点 embedding H [B, N, D]。"""
        return self.base.encode(*args, **kwargs)

    def score(self, H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask):
        """返回 (score [B,K,N], residual [B,K,N])。"""
        return self.fleet_head(H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask)
