"""真实 checkpoint 的 JIT 等价性验收（服务器 GPU）。

比较非 JIT 路径（model.decode + model.score）与持久 JIT 路径（M1Scorer.score）在相同
模型/输入/候选顺序/精度下：分数最大误差 + argmax 是否一致。覆盖不同 Nv、M。

用法（服务器 MASKCO_env）：
    python scripts/tests/test_m1_jit_equiv.py --ckpt <m1.ckpt> --encoder-ckpt <encoder.ckpt>
"""
import argparse
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
for p in ('models', 'data', 'training', 'simulation', 'evaluation', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax.numpy as jnp
from train_fleet_head import load_base_model
from dynmaskco_cc import load_m1_model, M1Scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='训练好的 M1 checkpoint')
    ap.add_argument('--encoder-ckpt', required=True)
    args = ap.parse_args()

    encoder = load_base_model(args.encoder_ckpt)
    model = load_m1_model(args.ckpt, encoder)
    scorer = M1Scorer(model)

    rng = np.random.default_rng(0)
    D = 256
    cases = [(12, 16), (30, 32), (45, 48)]   # (Nv, M)
    all_ok = True
    for Nv, M in cases:
        Hv = rng.standard_normal((1, Nv, D)).astype(np.float32)
        ts = np.array([0.5], np.float32)
        A_in = (rng.random((1, Nv, Nv)) > 0.7).astype(np.float32)
        endpoints = rng.integers(0, Nv, (1, M, 4)).astype(np.int32)
        valid = rng.random((1, M, 4)) > 0.2
        explicit = rng.standard_normal((1, M, 48)).astype(np.float32)

        # 非 JIT
        Z = model.decode(jnp.asarray(Hv), jnp.asarray(ts), jnp.asarray(A_in))
        s_nonjit = np.asarray(model.score(jnp.asarray(Hv), Z,
                                          jnp.asarray(endpoints), jnp.asarray(valid),
                                          jnp.asarray(explicit)))
        # JIT
        s_jit = np.asarray(scorer.score(jnp.asarray(Hv), jnp.asarray(ts), jnp.asarray(A_in),
                                        jnp.asarray(endpoints), jnp.asarray(valid),
                                        jnp.asarray(explicit)))
        err = np.abs(s_nonjit - s_jit)
        denom = np.maximum(np.abs(s_nonjit), 1e-8)
        rel = (err / denom).max()
        argmax_ok = np.array_equal(s_nonjit.argmax(-1), s_jit.argmax(-1))
        all_ok = all_ok and np.isfinite(s_jit).all() and argmax_ok
        print(f"  Nv={Nv:2d} M={M:2d}  max_abs={err.max():.2e}  max_rel={rel:.2e}  "
              f"argmax={'SAME' if argmax_ok else 'DIFF'}")
    print(f"  {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
