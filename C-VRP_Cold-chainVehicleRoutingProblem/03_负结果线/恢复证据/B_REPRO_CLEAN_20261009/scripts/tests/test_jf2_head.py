"""
JF2 Step 6（FleetAssignmentHead wrapper-first）shape 测试。

验证 FleetAssignmentHead 的输入/输出 shape 与 alpha=0 退化行为。需 JAX/Flax，服务器跑：

    python scripts/tests/test_jf2_head.py
"""
import sys, os
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_BASE), 'models'))

import jax
import jax.numpy as jnp
from fleet_assignment_head import FleetAssignmentHead, JF2Runtime


def test_head_shapes():
    B, N, D, K = 2, 8, 16, 4
    Fv, Fp, hidden = 5, 12, 32

    head = FleetAssignmentHead(embed_dim=D, veh_dim=Fv, pair_dim=Fp, hidden_dim=hidden, rngs=0)

    H = jax.random.normal(jax.random.PRNGKey(0), (B, N, D))
    anchor_ids = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]])  # [B,K]
    veh_feat = jax.random.normal(jax.random.PRNGKey(1), (B, K, Fv))
    pair_feat = jax.random.normal(jax.random.PRNGKey(2), (B, K, N, Fp))
    base_score = -jax.random.normal(jax.random.PRNGKey(3), (B, K, N))
    candidate_mask = jnp.ones((B, K, N), dtype=bool)

    score, residual = head(H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask)

    assert score.shape == (B, K, N), f"score shape {score.shape} != {(B, K, N)}"
    assert residual.shape == (B, K, N), f"residual shape {residual.shape} != {(B, K, N)}"


def test_alpha_zero_degrades_to_base():
    """alpha=0 时 score 必须精确等于 base_score（candidate_mask 内）。"""
    B, N, D, K = 2, 8, 16, 4
    Fv, Fp, hidden = 5, 12, 32

    head = FleetAssignmentHead(embed_dim=D, veh_dim=Fv, pair_dim=Fp, hidden_dim=hidden, rngs=0)

    H = jax.random.normal(jax.random.PRNGKey(0), (B, N, D))
    anchor_ids = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    veh_feat = jax.random.normal(jax.random.PRNGKey(1), (B, K, Fv))
    pair_feat = jax.random.normal(jax.random.PRNGKey(2), (B, K, N, Fp))
    base_score = jax.random.normal(jax.random.PRNGKey(3), (B, K, N))
    candidate_mask = jnp.ones((B, K, N), dtype=bool)

    score, _ = head(H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask)
    assert jnp.allclose(score, base_score), "alpha=0 时 score 应 == base_score"


def test_candidate_mask_masks():
    """candidate_mask=False 处 score 应被置 -1e9。"""
    B, N, D, K = 2, 8, 16, 4
    Fv, Fp, hidden = 5, 12, 32

    head = FleetAssignmentHead(embed_dim=D, veh_dim=Fv, pair_dim=Fp, hidden_dim=hidden, rngs=0)
    H = jax.random.normal(jax.random.PRNGKey(0), (B, N, D))
    anchor_ids = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    veh_feat = jax.random.normal(jax.random.PRNGKey(1), (B, K, Fv))
    pair_feat = jax.random.normal(jax.random.PRNGKey(2), (B, K, N, Fp))
    base_score = jax.random.normal(jax.random.PRNGKey(3), (B, K, N))
    candidate_mask = jnp.zeros((B, K, N), dtype=bool)

    score, _ = head(H, anchor_ids, veh_feat, pair_feat, base_score, candidate_mask)
    assert jnp.all(score == -1e9), "candidate_mask=False 处 score 应 == -1e9"


def test_runtime_encapsulates_base():
    """JF2Runtime 应正确封装 base 与 head，encode 委托 base。"""
    head = FleetAssignmentHead(embed_dim=16, veh_dim=5, pair_dim=12, hidden_dim=32, rngs=0)

    class _FakeBase:
        def encode(self, x):
            return x * 2.0

    rt = JF2Runtime(_FakeBase(), head)
    x = jnp.ones((1, 8, 16))
    assert jnp.all(rt.encode(x) == x * 2.0), "encode 应委托 base"


if __name__ == '__main__':
    test_head_shapes()
    test_alpha_zero_degrades_to_base()
    test_candidate_mask_masks()
    test_runtime_encapsulates_base()
    print("PASS: test_jf2_head — FleetAssignmentHead shape + alpha=0 退化")
