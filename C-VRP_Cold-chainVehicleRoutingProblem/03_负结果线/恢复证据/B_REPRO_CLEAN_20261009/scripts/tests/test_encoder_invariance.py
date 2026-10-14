"""Encoder 评分不变性检查（服务器，GPU）——验证冻结 encoder context 对隐藏节点扰动/追加不变。

与 feature-only 的防泄漏测试对应：隐藏节点不进 pooling/归一化/注意力，因此
  1) 扰动未揭示订单坐标 → encoder context 不变；
  2) 追加未揭示订单 → encoder context 不变。

用法（服务器 MASKCO_env，CUDA_VISIBLE_DEVICES=1）：
    python scripts/tests/test_encoder_invariance.py
"""
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)
for p in ('models', 'data', 'training', 'simulation', 'evaluation', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax.numpy as jnp
from cvrptw_utils import coord_normalize_visible
from coldchain_utility_head import encoder_context
from train_fleet_head import load_base_model

CKPT = os.path.join(_CVRPTW, 'ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt')
N, D = 51, 256
CAPACITY, TW_MAX = 50.0, 24.0


def _build_raw(coords, demands, tw_start, tw_end, temp_class, reveal, visible):
    raw = np.concatenate([
        coords, (demands / CAPACITY)[..., None], (tw_start / TW_MAX)[..., None],
        (tw_end / TW_MAX)[..., None], (temp_class / 2.0)[..., None],
        (reveal / TW_MAX)[..., None],
    ], axis=-1).astype(np.float32)
    vis = visible[..., None]
    raw[..., 2:] = raw[..., 2:] * vis
    raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5
    raw_j = jnp.array(raw)
    raw_j = raw_j.at[..., :2].set(coord_normalize_visible(raw_j[..., :2], jnp.array(visible)))
    return raw_j


def _ctx(encoder, raw, visible):
    H = encoder.encode(raw, visible_mask=jnp.array(visible))
    return np.asarray(encoder_context(H, jnp.array(visible)))


def main():
    encoder = load_base_model(CKPT)
    rng = np.random.default_rng(0)

    coords = rng.uniform(0, 1, (1, N, 2)).astype(np.float32)
    coords[:, 0] = [0.5, 0.5]
    demands = rng.integers(1, 11, (1, N)).astype(np.float32)
    demands[:, 0] = 0
    tw_start = np.zeros((1, N), np.float32)
    tw_end = np.full((1, N), TW_MAX, np.float32)
    temp_class = rng.integers(0, 3, (1, N)).astype(np.float32)
    temp_class[:, 0] = 0
    reveal = np.zeros((1, N), np.float32)
    reveal[0, -1] = 10.0                          # 节点 N-1 未揭示
    visible = np.ones((1, N), bool)
    visible[0, -1] = False

    base = _ctx(encoder, _build_raw(coords, demands, tw_start, tw_end, temp_class, reveal, visible),
                visible)

    # (1) 扰动未揭示节点坐标 → context 不变
    coords2 = coords.copy()
    coords2[0, -1] = [999., 999.]
    pert = _ctx(encoder, _build_raw(coords2, demands, tw_start, tw_end, temp_class, reveal, visible),
                visible)
    d1 = float(np.abs(base - pert).max())

    # (2) 追加一个未揭示节点 → context 不变
    N2 = N + 1
    coords3 = np.concatenate([coords, rng.uniform(0, 1, (1, 1, 2)).astype(np.float32)], axis=1)
    demands3 = np.concatenate([demands, np.array([[1.0]], np.float32)], axis=1)
    tw_start3 = np.concatenate([tw_start, np.zeros((1, 1), np.float32)], axis=1)
    tw_end3 = np.concatenate([tw_end, np.full((1, 1), TW_MAX, np.float32)], axis=1)
    temp_class3 = np.concatenate([temp_class, np.array([[1]], np.float32)], axis=1)
    reveal3 = np.concatenate([reveal, np.array([[100.0]], np.float32)], axis=1)
    visible3 = np.concatenate([visible, np.zeros((1, 1), bool)], axis=1)
    app = _ctx(encoder, _build_raw(coords3, demands3, tw_start3, tw_end3, temp_class3, reveal3, visible3),
               visible3)
    d2 = float(np.abs(base - app).max())

    # perturb 必须精确 0（隐藏坐标被掩码）；append 是恒定 float32 数值底噪（append 1/2/5/10
    # 均为 ~2.9e-3，不随 cardinality 增长，见 test_encoder_append_scale.py），阈值按此区分。
    ok = d1 < 1e-5 and d2 < 1e-2
    print(f"  encoder_invariance: perturb_hidden={d1:.2e} append_hidden={d2:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
