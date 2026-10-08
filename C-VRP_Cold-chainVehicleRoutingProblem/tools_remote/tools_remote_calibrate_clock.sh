#!/bin/bash
# I1 世代留出校准（[160,200) 训练历史日，模型未见；诚实口径）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
OUT_BASE=results/a1_step3
mkdir -p $OUT_BASE

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_clock_pretrained_s42/model.bin \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_clock_pretrained > $OUT_BASE/calib_clock_pretrained.log 2>&1
echo "CALIB_PRE_EXIT=$?"

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_clock_random_cvrp_s42/model.bin \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_clock_random_cvrp > $OUT_BASE/calib_clock_random_cvrp.log 2>&1
echo "CALIB_RCVRP_EXIT=$?"
echo "CALIB_DONE"
