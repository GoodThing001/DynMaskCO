"""
CVRPTW Model — 带时间窗的 Capacitated Vehicle Routing Problem.

基于 MaskCO 的 CVRPModel 扩展：
- 输入特征从 3D (x, y, demand) 扩展为 5D (x, y, demand, tw_start, tw_end)
- 增加 time_window_bias 参数，类似 depot_bias 的机制
- 其余架构（encoder-decoder transformer）与 CVRPModel 完全一致

设计原则：不修改原始 MaskCO 代码，通过子类化实现扩展。
"""

import sys
import os
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT
sys.path.insert(0, str(MASKCO_ROOT))

import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
from typing import Any

# 从原始 MaskCO 导入基类
from models.TSPModel import TSPModel, TSPModelConfig
from models.CVRPModel import CVRPModel, CVRPModelConfig


@dataclass
class CVRPTWModelConfig(CVRPModelConfig):
    """CVRPTW 模型配置 — 继承 CVRPModelConfig，仅修改 encoder_input_dim."""

    encoder_input_dim: int = 5  # x, y, demand, tw_start, tw_end

    @staticmethod
    def get_config(which: str = ''):
        """预设配置。"""
        match which:
            case '' | 'default':
                return CVRPTWModelConfig()
            case 'softcap_fn':  # 256-dim，适合 50-node
                return CVRPTWModelConfig(
                    embed_dim=256, num_heads=8,
                    qk_norm=False, softcap=30.,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                )
            case 'softcap_fn_512':  # 512-dim，适合 100-node
                return CVRPTWModelConfig(
                    embed_dim=512, num_heads=8,
                    qk_norm=False, softcap=30.,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 512,
                )
            case 'small_test':  # 快速测试小模型
                return CVRPTWModelConfig(
                    num_layers=(4, 2), embed_dim=128, num_heads=4,
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 128,
                )
            case _:
                raise ValueError(
                    f"Unknown CVRPTWModel config '{which}'. "
                    f"Available: default, softcap_fn, softcap_fn_512, small_test"
                )

    def construct_model(self):
        return CVRPTWModel(**vars(self))


class CVRPTWModel(CVRPModel):
    """
    CVRPTW 模型 — 继承 CVRPModel，扩展为 5D 输入。

    新增：
    - tw_bias: 时间窗特征的可学习偏置，加在 depot 偏置之后
    - 输入特征：x, y, demand, tw_start_normalized, tw_end_normalized

    其余与 CVRPModel 完全一致：encoder → mid_proj → decoder → feature2logit
    """

    def __init__(
        self,
        num_layers: tuple[int, int] = (16, 6),
        embed_dim: tuple[int, int] = 256,
        hidden_dim: int | None = None,
        num_heads: tuple[int, int] = 8,
        activation: str = 'silu',
        use_swiglu: bool = True,
        *,
        dtype: jax.typing.DTypeLike = jnp.bfloat16,
        encoder_input_dim: int = 5,
        softcap: float | None = 30.,
        final_norm: bool = True,
        qk_norm: bool = False,
        sm_scale: float | None = None,
        final_norm_use_scale: bool = False,
        final_softcap: float | None = 50.,
        final_logit_scale: float = 1.,
        rngs: nnx.Rngs | int,
    ):
        # 先调用 CVRPModel.__init__，它会调用 TSPModel.__init__
        # 这里 encoder_input_dim=5 会自动设置 init_proj = Linear(5, embed_dim)
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            activation=activation,
            use_swiglu=use_swiglu,
            dtype=dtype,
            encoder_input_dim=encoder_input_dim,
            softcap=softcap,
            qk_norm=qk_norm,
            sm_scale=sm_scale,
            final_norm=final_norm,
            final_norm_use_scale=final_norm_use_scale,
            final_softcap=final_softcap,
            final_logit_scale=final_logit_scale,
            rngs=rngs,
        )

        # 时间窗偏置：类似 depot_bias，为时间窗特征提供可学习的偏置
        encoder_embed_dim, _ = embed_dim
        self.tw_bias = nnx.Param(
            nnx.initializers.normal(1e-6)(rngs.params(), [encoder_embed_dim], jnp.float32)
        )

    def encode(self, raw_features: jax.Array, attn_options: dict[str, Any] | None = None):
        """
        编码 5D 输入特征 [x, y, demand, tw_start, tw_end]。

        相比 CVRPModel.encode():
        - 在 depot_bias 之后，额外对时间窗维度 (索引 3, 4) 的投影加 tw_bias
        """
        if attn_options is None:
            attn_options = {}
        attn_options.update(softcap=self.softcap, sm_scale=self.sm_scale)
        depot_bias = self.depot_bias.value.astype(self.dtype)
        tw_bias = self.tw_bias.value.astype(self.dtype)

        # Linear 投影: (batch, nodes, 5) → (batch, nodes, embed_dim)
        features = self.init_proj(raw_features)

        # depot bias: 加到 depot 节点 (index 0)
        features = features.at[:, 0].add(depot_bias[None])

        # tw bias: 全局时间窗偏置
        features = features + tw_bias[None, None, :]

        # Transformer encoder
        features = self.encoder(features, attn_options=attn_options)
        return features


# ============================================================
# 时间窗工具函数（纯 NumPy/JAX，用于数据预处理和解码约束）
# ============================================================

def compute_tw_feasibility_matrix(
    coords: jnp.ndarray,           # (batch, nodes, 2) 或 (nodes, 2)
    tw_start: jnp.ndarray,         # (batch, nodes) 或 (nodes,)
    tw_end: jnp.ndarray,           # (batch, nodes) 或 (nodes,)
    service_time: jnp.ndarray,     # (batch, nodes) 或 (nodes,)
    speed: float = 1.0,            # 行驶速度（坐标单位/时间单位）
) -> jnp.ndarray:
    """
    计算时间窗可行性矩阵。

    对于每对节点 (i, j)，检查：
        tw_start[i] + service_time[i] + dist(i,j)/speed <= tw_end[j]

    Returns:
        feasibility: (batch, nodes, nodes) bool 矩阵
        feasibility[i, j] = True 表示从 i 出发能在 j 的时间窗关闭前到达
    """
    if coords.ndim == 2:
        coords = coords[None, ...]
        tw_start = tw_start[None, ...]
        tw_end = tw_end[None, ...]
        service_time = service_time[None, ...]

    # 欧氏距离矩阵: (batch, nodes, nodes)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = jnp.sqrt((diff ** 2).sum(axis=-1) + 1e-10)
    travel_time = dist / speed

    # 从 i 出发时间 + 服务 i + 行驶(i,j) <= 下一节点 j 的时间窗关闭
    # 修复 P0-9：原用 tw_start[:, None, :]（=tw_start[j]）方向反了，应为 tw_start[i]。
    ready_at_j = tw_start[:, :, None] + service_time[:, :, None] + travel_time
    feasible = ready_at_j <= tw_end[:, None, :]

    return feasible  # (batch, nodes_from, nodes_to)


def filter_edges_by_tw(
    candidate_edges: jnp.ndarray,   # (batch, num_edges, 2)
    coords: jnp.ndarray,            # (batch, nodes, 2)
    tw_start: jnp.ndarray,          # (batch, nodes)
    tw_end: jnp.ndarray,            # (batch, nodes)
    service_time: jnp.ndarray,      # (batch, nodes)
    current_arrival: jnp.ndarray,   # (batch, nodes) 当前已知到达时间
    speed: float = 1.0,
) -> jnp.ndarray:
    """
    过滤时间窗不可行的候选边。

    对于每条候选边 (i, j)：
    - 计算从 i 出发到达 j 的时间
    - 如果到达时间 > tw_end[j]，标记为不可行
    - 返回过滤后的边和可行掩码

    Returns:
        feasible_mask: (batch, num_edges) bool
    """
    batch_size = candidate_edges.shape[0]
    nodes = coords.shape[1]

    i_idx = candidate_edges[..., 0]  # (batch, num_edges)
    j_idx = candidate_edges[..., 1]  # (batch, num_edges)

    # 向量化取坐标
    coords_i = coords[jnp.arange(batch_size)[:, None], i_idx]  # (batch, num_edges, 2)
    coords_j = coords[jnp.arange(batch_size)[:, None], j_idx]  # (batch, num_edges, 2)

    # 距离和行驶时间
    dist = jnp.sqrt(((coords_i - coords_j) ** 2).sum(axis=-1) + 1e-10)
    travel_time = dist / speed

    # 当前到达时间（已知边）
    arrival_i = current_arrival[jnp.arange(batch_size)[:, None], i_idx]

    # 预计到达 j 的时间
    est_arrival_j = arrival_i + service_time[jnp.arange(batch_size)[:, None], i_idx] + travel_time

    # 必须在 tw_end[j] 之前到达
    feasible = est_arrival_j <= tw_end[jnp.arange(batch_size)[:, None], j_idx]

    # depot 总是可行的
    feasible = feasible | (j_idx == 0)

    return feasible


def compute_service_times(
    demands: jnp.ndarray,
    service_time_per_unit: float = 0.1,
    fixed_service_time: float = 0.0,
) -> jnp.ndarray:
    """
    根据需求量估算服务时间。

    Args:
        demands: (batch, nodes) 需求量（已归一化到 [0, 1]）
        service_time_per_unit: 每单位需求的服务时间
        fixed_service_time: 固定服务时间

    Returns:
        service_time: (batch, nodes) 服务时间
    """
    return fixed_service_time + demands * service_time_per_unit
