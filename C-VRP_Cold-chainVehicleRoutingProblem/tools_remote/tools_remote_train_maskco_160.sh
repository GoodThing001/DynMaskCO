#!/bin/bash
# A3 步骤3 训练：pretrained 臂 + 同架构 random 对照（160 天 → 留出 160..199 做 I1 诚实校准）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT_BASE=results/a1_step3
mkdir -p $OUT_BASE

echo "=== stream prefix check: generate_dataset(160) == generate_dataset(200)[:160] ==="
$PY - <<'EOF'
import sys, os
sys.path.insert(0, 'scripts/evaluation')
sys.path.insert(0, 'scripts/coldchain')
sys.path.insert(0, 'scripts/simulation')
from run_exp_reserve import generate_dataset
import numpy as np
a = generate_dataset(160, 200, 20260925)
b = generate_dataset(200, 200, 20260925)
same = True
for k in a:
    if k == 'quality' or 'quality' in k:
        continue
    aa, bb = np.asarray(a[k]), np.asarray(b[k])[:160]
    if aa.shape != bb.shape or not np.allclose(aa, bb, equal_nan=True):
        same = False
        print('MISMATCH', k, aa.shape, bb.shape)
print('prefix_same=', same)
EOF

echo "=== train pretrained arm (160d, dim128, mmax200, 2000 steps) ==="
$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_pretrained_s42 > $OUT_BASE/train160_pretrained.log 2>&1
echo "PRETRAINED_EXIT=$?"

echo "=== train random arm (同架构对照, 同参数预算) ==="
$PY scripts/training/train_maskco_scenario.py \
  --arm random --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_random_s42 > $OUT_BASE/train160_random.log 2>&1
echo "RANDOM_EXIT=$?"
echo "TRAIN_DONE"
