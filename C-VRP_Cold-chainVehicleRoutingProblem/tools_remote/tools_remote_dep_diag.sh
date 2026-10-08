#!/bin/bash
# I2 部署采样诊断：四臂单步 + partial-pretrained 迭代（K=10）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_step3
COMMON="--history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 --K 10 --allow-code-mismatch"

$PY scripts/evaluation/diag_i2_deployment.py \
  --model-bin $B/maskco_train160_clock_pretrained_s42/model.bin --arm pretrained \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt $COMMON \
  --out $B/dep_clock_pretrained > $B/dep_clock_pretrained.log 2>&1
echo "DEP_CLOCK_PRE_EXIT=$?"

$PY scripts/evaluation/diag_i2_deployment.py \
  --model-bin $B/maskco_train160_clock_random_cvrp_s42/model.bin --arm random_cvrp \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt $COMMON \
  --out $B/dep_clock_random_cvrp > $B/dep_clock_random_cvrp.log 2>&1
echo "DEP_CLOCK_RCVRP_EXIT=$?"

$PY scripts/evaluation/diag_i2_deployment.py \
  --model-bin $B/maskco_train160_partial_pretrained_s42/model.bin --arm pretrained \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt $COMMON \
  --out $B/dep_partial_pretrained > $B/dep_partial_pretrained.log 2>&1
echo "DEP_PARTIAL_PRE_EXIT=$?"

$PY scripts/evaluation/diag_i2_deployment.py \
  --model-bin $B/maskco_train160_partial_pretrained_s42/model.bin --arm pretrained \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt $COMMON --iterative --remask-frac 0.5 --rounds 1 \
  --out $B/dep_partial_pretrained_iter > $B/dep_partial_pretrained_iter.log 2>&1
echo "DEP_PARTIAL_PRE_ITER_EXIT=$?"

$PY scripts/evaluation/diag_i2_deployment.py \
  --model-bin $B/maskco_train160_partial_random_cvrp_s42/model.bin --arm random_cvrp \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt $COMMON \
  --out $B/dep_partial_random_cvrp > $B/dep_partial_random_cvrp.log 2>&1
echo "DEP_PARTIAL_RCVRP_EXIT=$?"
echo "DEP_DIAG_DONE"
