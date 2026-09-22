"""M-pre：真实预训练 MaskCO CVRP 重构（复用完整预训练 encoder+decoder）。

加载 `MASKCO_code/ckpts/cvrp100.ckpt`（`models.CVRPModel`，512 维，3D 输入 coord+demand），
用 `decode(target='logit')` 出**对称**边 logits（features@features.T，经 final_norm +
final_logit_scale=16/512 + softcap，对角线 mask），经固定「增删边评分」映射转插入偏好。

这是「未微调 MaskCO」对照：不训练、不接随机头。见 MaskCO直接目标训练工作包.md 第二步。
输入归一化沿用原始 CVRP：coords 用 `coord_normalize`（全节点 mean/L2，静态），
demand/capacity。capacity=50（cvrp100 训练口径）。
"""
import os
import sys
import types

import numpy as np


def _ensure_tensorboardx_stub():
    if 'tensorboardX' in sys.modules:
        return
    try:
        import tensorboardX  # noqa: F401
    except ImportError:
        tbx = types.ModuleType('tensorboardX')

        class _SW:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return lambda *a, **k: None
        tbx.SummaryWriter = _SW
        sys.modules['tensorboardX'] = tbx


def _setup_maskco_paths():
    _here = os.path.dirname(os.path.abspath(__file__))
    _cv = os.path.dirname(os.path.dirname(_here))
    from project_paths import MASKCO_ROOT
    for p in (str(MASKCO_ROOT), os.path.join(str(MASKCO_ROOT), 'models')):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_cvrp_model(ckpt_path):
    """加载 cvrp100.ckpt → (merged CVRPModel, config, step)。dtype 转 float32。"""
    _ensure_tensorboardx_stub()
    _setup_maskco_paths()
    from training import load_ckpt
    from flax import nnx
    params, _, _, cfg, _, step = load_ckpt(ckpt_path)
    cfg.dtype = 'float32'
    model = cfg.construct_model()
    return nnx.merge(nnx.graphdef(model), params), cfg, step


def encode_cvrp(model, coords, demands, capacity):
    """coords [B,N,2], demands [B,N] → H [B,N,512]（原始 CVRP 3D 输入 + 归一化）。"""
    import jax.numpy as jnp
    from cvrptw_utils import coord_normalize_visible
    coords = jnp.asarray(coords, dtype=jnp.float32)
    demands = jnp.asarray(demands, dtype=jnp.float32)
    coords_n = coord_normalize_visible(coords, None)          # 静态全节点 mean/L2
    dem_n = (demands / capacity)[..., None]
    raw = jnp.concatenate([coords_n, dem_n], axis=-1)         # [B,N,3]
    return np.asarray(model.encode(raw))


def decode_edge_logits(model, H, timestep, adjmat):
    """H [B,N,512] → 对称边 logits [B,N,N]。timestep=[B]，adjmat=[B,N,N]（0/1 偏置）。"""
    import jax.numpy as jnp
    H = jnp.asarray(H, dtype=jnp.float32)
    ts = jnp.asarray(timestep, dtype=jnp.float32)
    A = jnp.asarray(adjmat, dtype=jnp.float32)
    L = model.decode(H, ts, A, target='logit')
    return np.asarray(L)
