#!/bin/bash
# I1 重校准（用户复核修订版：按天聚类 CI + 平滑/时钟条件基线）——覆盖旧 calibration.json
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
OUT_BASE=results/a1_step3

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_clock_pretrained_s42/model.bin \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_clock_pretrained > $OUT_BASE/calib_clock_pretrained.log 2>&1
echo "RECALIB_PRE_EXIT=$?"

$PY scripts/evaluation/diag_i1_calibration.py \
  --model-bin $OUT_BASE/maskco_train160_clock_random_cvrp_s42/model.bin \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT_BASE/calib_clock_random_cvrp > $OUT_BASE/calib_clock_random_cvrp.log 2>&1
echo "RECALIB_RCVRP_EXIT=$?"
echo "RECALIB_DONE"
