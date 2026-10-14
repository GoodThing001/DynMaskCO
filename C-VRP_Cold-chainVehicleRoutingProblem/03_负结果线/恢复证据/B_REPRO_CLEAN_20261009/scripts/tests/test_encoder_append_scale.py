"""Encoder 追加隐藏节点残差的规模定位（服务器 GPU）——区分数值精度 vs 真实 cardinality 泄漏。"""
import os, sys
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
N, CAP, TW = 51, 50.0, 24.0


def build(coords, demands, ts, te, temp, reveal, vis):
    raw = np.concatenate([coords, (demands / CAP)[..., None], (ts / TW)[..., None],
                          (te / TW)[..., None], (temp / 2.0)[..., None],
                          (reveal / TW)[..., None]], axis=-1).astype(np.float32)
    v = vis[..., None]
    raw[..., 2:] *= v
    raw[..., :2] = raw[..., :2] * v + (1.0 - v) * 0.5
    rj = jnp.array(raw)
    rj = rj.at[..., :2].set(coord_normalize_visible(rj[..., :2], jnp.array(vis)))
    return rj


def ctx(enc, raw, vis):
    H = enc.encode(raw, visible_mask=jnp.array(vis))
    return np.asarray(encoder_context(H, jnp.array(vis)))


def main():
    enc = load_base_model(CKPT)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (1, N, 2)).astype(np.float32); coords[:, 0] = [0.5, 0.5]
    demands = rng.integers(1, 11, (1, N)).astype(np.float32); demands[:, 0] = 0
    ts = np.zeros((1, N), np.float32); te = np.full((1, N), TW, np.float32)
    temp = rng.integers(0, 3, (1, N)).astype(np.float32); temp[:, 0] = 0
    reveal = np.zeros((1, N), np.float32); reveal[0, -1] = 10.0
    vis = np.ones((1, N), bool); vis[0, -1] = False

    base = ctx(enc, build(coords, demands, ts, te, temp, reveal, vis), vis)
    print(f"context |mean|={float(np.abs(base).mean()):.4f} max|. |={float(np.abs(base).max()):.4f}")
    for k in (1, 2, 5, 10):
        c2 = np.concatenate([coords, rng.uniform(0, 1, (1, k, 2)).astype(np.float32)], axis=1)
        d2 = np.concatenate([demands, np.ones((1, k), np.float32)], axis=1)
        ts2 = np.concatenate([ts, np.zeros((1, k), np.float32)], axis=1)
        te2 = np.concatenate([te, np.full((1, k), TW, np.float32)], axis=1)
        t2 = np.concatenate([temp, np.ones((1, k), np.float32)], axis=1)
        r2 = np.concatenate([reveal, np.full((1, k), 100.0, np.float32)], axis=1)
        v2 = np.concatenate([vis, np.zeros((1, k), bool)], axis=1)
        app = ctx(enc, build(c2, d2, ts2, te2, t2, r2, v2), v2)
        print(f"append {k:2d}: max|diff|={float(np.abs(base - app).max()):.3e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
