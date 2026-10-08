#!/bin/bash
# 泛化曲线诊断（2026-10-01 用户指派）：pretrained 臂 160 天 2000 步，每 200 步存 checkpoint
# 随后对全部 checkpoint 计算训练日 vs 留出日的同口径计数 NLL 曲线。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_step3
OUT=$B/gencurve_pretrained

rm -rf $OUT
$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --save-every 200 \
  --out $OUT > $OUT.train.log 2>&1
echo "TRAIN_EXIT=$?"

$PY scripts/evaluation/diag_i1_generalization.py \
  --ckpt-dir $OUT --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-days 160 --history-days 200 --held-start 160 --cuts-per-day 4 \
  --allow-code-mismatch \
  --out $OUT > $OUT.curve.log 2>&1
echo "CURVE_EXIT=$?"
tail -n 20 $OUT.curve.log
echo "GENCURVE_DONE"
