#!/bin/bash
# A3 步骤3 训练（修复版）：pretrained + random_cvrp 同架构随机对照（GPU0，顺序执行）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT_BASE=results/a1_step3
mkdir -p $OUT_BASE

$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_pretrained_s42 > $OUT_BASE/train160_pretrained.log 2>&1
echo "PRETRAINED_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_random_cvrp_s42 > $OUT_BASE/train160_random_cvrp.log 2>&1
echo "RANDOM_CVRP_EXIT=$?"
echo "TRAIN2_DONE"
