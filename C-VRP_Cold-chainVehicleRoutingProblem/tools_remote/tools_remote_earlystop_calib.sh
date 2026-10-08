#!/bin/bash
# 早停验证：对 gencurve 的 step 200/400 checkpoint 跑按天聚类三基线校准（pretrained 臂）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_step3

for STEP in 200 400; do
  $PY scripts/evaluation/diag_i1_calibration.py \
    --model-bin $B/gencurve_pretrained/ckpt_step_$STEP.bin \
    --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --history-days 200 --day-start 160 --day-end 200 --cuts-per-day 4 \
    --allow-code-mismatch \
    --out $B/calib_earlystop_$STEP > $B/calib_earlystop_$STEP.log 2>&1
  echo "CALIB_STEP_${STEP}_EXIT=$?"
done
echo "EARLYSTOP_CALIB_DONE"
