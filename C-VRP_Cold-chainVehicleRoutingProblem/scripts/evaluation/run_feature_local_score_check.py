"""feature-local 70 维：checkpoint 身份 + 离线/在线分数对账（④检查 1 + 4）。

1. 加载 model.ckpt，校验 in_dim=70、s>0、use_state=False（F 组）；
2. 离线：build_explicit_and_labels 的 70 维特征 + ScoringMLP 分数；
3. 在线：feature_local_replanner.load_feature_local_scorer 对同一特征打分；
4. 比较二者分数（应逐位一致，同一模型同一输入）。

用法（服务器）：
    python scripts/evaluation/run_feature_local_score_check.py \
        --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
        --teacher-dir results/m0_scale/dev_check \
        --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
        --max-contexts 4 --deindex --out results/m0_scale/iface_score
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax.numpy as jnp
from flax import nnx

from coldchain_teacher_dataset import load_teacher_dataset
from train_m1_compact import build_explicit_and_labels
from dynmaskco_cc_compact import ScoringMLP
from feature_local_replanner import load_feature_local_scorer, IN_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-contexts', type=int, default=4)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 检查 1：checkpoint 身份 ----
    with open(args.ckpt, 'rb') as f:
        ck = pickle.load(f)
    identity = {
        'in_dim': ck.get('in_dim'), 's': ck.get('s'), 'use_state': ck.get('use_state'),
        'use_rank': ck.get('use_rank'),
    }
    id_ok = (ck.get('in_dim') == IN_DIM and ck.get('s') and ck.get('s') > 0
             and not ck.get('use_state'))
    print(f"  [identity] {identity} in_dim_ok={ck.get('in_dim')==IN_DIM} id_ok={id_ok}",
          flush=True)

    # ---- 检查 4：离线 vs 在线分数 ----
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    npz = dict(np.load(args.data))
    explicit, t = build_explicit_and_labels(ds, npz, args.capacity, args.deindex)

    # 离线打分（与 train_state_2x2 同一 ScoringMLP + params）
    graphdef, _ = nnx.split(ScoringMLP(IN_DIM, (128, 64), rngs=0))
    off_scorer = nnx.merge(graphdef, ck['params'])

    # 在线打分器（adapter 的 load_feature_local_scorer）
    on_scorer = load_feature_local_scorer(args.ckpt)

    max_diff = 0.0
    n_rows = 0
    for i in range(min(args.max_contexts, explicit.shape[0])):
        x = jnp.asarray(explicit[i:i+1])
        off = np.asarray(off_scorer(x))[0]
        on = np.asarray(on_scorer.score(x))[0]
        d = float(np.abs(off - on).max())
        max_diff = max(max_diff, d)
        n_rows += off.shape[0]
        print(f"  ctx{i}: candidates={off.shape[0]} score_max_diff={d:.3e}", flush=True)

    out = {'identity': identity, 'identity_ok': bool(id_ok),
           'score_max_diff': max_diff, 'n_scored': n_rows,
           'score_consistent': max_diff < 1e-5}
    with open(os.path.join(args.out, 'score_check.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  score_max_diff={max_diff:.3e} consistent={max_diff < 1e-5}")
    print(f"saved: {args.out}/score_check.json")


if __name__ == '__main__':
    main()
