#!/bin/bash
# I2 世代训练（预声明 2026-09-30 20:30：部分掩码训练，同 160 天/dim128/mmax200/2000步/batch8/seed42 谱系）
# 运行条件：maskco_i1（I1 时钟世代）完成且 GPU 空闲；顺序：pretrained → random_cvrp。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
OUT_BASE=results/a1_step3
mkdir -p $OUT_BASE

$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --partial-mask --p-max 0.8 \
  --out $OUT_BASE/maskco_train160_partial_pretrained_s42 \
  > $OUT_BASE/train160_partial_pretrained.log 2>&1
echo "PARTIAL_PRETRAINED_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --partial-mask --p-max 0.8 \
  --out $OUT_BASE/maskco_train160_partial_random_cvrp_s42 \
  > $OUT_BASE/train160_partial_random_cvrp.log 2>&1
echo "PARTIAL_RANDOM_CVRP_EXIT=$?"
echo "TRAIN4_DONE"
