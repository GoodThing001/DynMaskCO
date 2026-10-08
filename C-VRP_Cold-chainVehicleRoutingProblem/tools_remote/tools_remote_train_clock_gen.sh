#!/bin/bash
# I1 世代训练（时钟条件 + 全部 P0 修复；GPU1，顺序：pretrained → random_cvrp）
# 2026-09-30 修复：autotune_level=0（pretrained 臂编译风暴治理；固定批长已修）+
# 固定批长 L=2*m_max（trainer 内，仅编译一次）。
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
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_clock_pretrained_s42 \
  > $OUT_BASE/train160_clock_pretrained.log 2>&1
echo "CLOCK_PRETRAINED_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $OUT_BASE/maskco_train160_clock_random_cvrp_s42 \
  > $OUT_BASE/train160_clock_random_cvrp.log 2>&1
echo "CLOCK_RANDOM_CVRP_EXIT=$?"
echo "TRAIN3_DONE"
