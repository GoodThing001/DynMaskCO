#!/bin/bash
# I2 世代留出校准（partial 模型 [160,200)，与 I1 单步同口径；判据 = 不劣于单步全掩码）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
OUT_BASE=results/a1_step3
mkdir -p $OUT_BASE

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_partial_pretrained_s42/model.bin \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_partial_pretrained > $OUT_BASE/calib_partial_pretrained.log 2>&1
echo "CALIB_PARTIAL_PRE_EXIT=$?"

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_partial_random_cvrp_s42/model.bin \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_partial_random_cvrp > $OUT_BASE/calib_partial_random_cvrp.log 2>&1
echo "CALIB_PARTIAL_RCVRP_EXIT=$?"
echo "CALIB_PARTIAL_DONE"
