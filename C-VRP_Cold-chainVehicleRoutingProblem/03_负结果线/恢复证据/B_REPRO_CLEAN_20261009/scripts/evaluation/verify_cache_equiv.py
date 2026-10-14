"""验证缓存 score_fn（缓存 H0/H）与未缓存 score_fn 分数 + argmax 等价。

对合成状态分别用未缓存、缓存（首次 miss、二次 hit）评分，比较 max|Δ| 与 argmax 一致性。
"""
import argparse
import os
import sys

import numpy as np
import jax

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from mpre import load_cvrp_model
from mtrained_replanner import (load_trained_model, mpre_score_fn, mtrained_score_fn,
                                mpre_score_fn_cached, mtrained_score_fn_cached)
from repair_state import F_NODE, F_ACTION_EXPLICIT


def make_state(Nv, M, seed):
    rng = np.random.default_rng(seed)
    return {
        'raw_3d': rng.standard_normal((1, Nv, 3)).astype(np.float32),
        'node_valid': np.ones((1, Nv), bool),
        'node_feats': rng.standard_normal((1, Nv, F_NODE)).astype(np.float32),
        'adjmat': rng.standard_normal((1, Nv, Nv)).astype(np.float32),
        'cust': rng.integers(0, Nv, size=(1, M)).astype(np.int32),
        'pred': rng.integers(0, Nv, size=(1, M)).astype(np.int32),
        'succ': rng.integers(0, Nv, size=(1, M)).astype(np.int32),
        'action_feats': rng.standard_normal((1, M, F_ACTION_EXPLICIT)).astype(np.float32),
        'timestep': 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--model-ckpt', required=True)
    args = ap.parse_args()

    backbone = load_cvrp_model(args.cvrp_ckpt)[0]
    model, _ = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)

    pairs = [
        ('Mpre', mpre_score_fn(backbone, 51, 2048), mpre_score_fn_cached(backbone, 51, 2048)),
        ('Mtrained', mtrained_score_fn(model, 51, 2048),
         mtrained_score_fn_cached(model, 51, 2048)),
    ]

    worst = 0.0
    for name, unc, cac in pairs:
        for seed in range(4):
            for Nv, M in [(5, 10), (17, 40), (51, 100), (51, 2048)]:
                st = make_state(Nv, M, seed)
                a = np.asarray(unc(st))
                b = np.asarray(cac(st))       # cache miss → compute
                c = np.asarray(cac(st))       # cache hit → reuse
                d1 = float(np.max(np.abs(a - b)))
                d2 = float(np.max(np.abs(a - c)))
                arg1 = int(np.argmax(a)) == int(np.argmax(b))
                arg2 = int(np.argmax(a)) == int(np.argmax(c))
                worst = max(worst, d1, d2)
                status = 'OK' if (arg1 and arg2) else 'MISMATCH'
                print(f"  {name} seed={seed} Nv={Nv} M={M}: miss|Δ|={d1:.2e} hit|Δ|={d2:.2e} "
                      f"argmax(miss/hit)={arg1}/{arg2} {status}")
    print(f"\nworst |Δ| = {worst:.3e}")
    print('EQUIVALENT' if worst < 1e-6 else 'WARNING: non-zero delta')


if __name__ == '__main__':
    main()
