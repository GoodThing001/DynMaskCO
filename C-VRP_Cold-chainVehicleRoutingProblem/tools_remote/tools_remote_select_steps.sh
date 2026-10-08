#!/bin/bash
# A3 正式步数选择（预声明规则）：train=0..128、valid=128..160，每 50 步存 checkpoint，
# valid 计数 NLL 首个极小点 → s*（只登记一次，不回调）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_step3
OUT=$B/stepsel_pretrained

rm -rf $OUT
$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --train-days 128 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --save-every 50 \
  --out $OUT > $OUT.train.log 2>&1
echo "CV_TRAIN_EXIT=$?"

$PY scripts/evaluation/diag_select_steps.py \
  --ckpt-dir $OUT --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --valid-start 128 --valid-end 160 --cuts-per-day 4 \
  --min-window 200 --allow-code-mismatch \
  --out $OUT > $OUT.select.log 2>&1
echo "SELECT_EXIT=$?"
grep -E 's_star|nll_at' $OUT/step_selection.json
echo "STEPSEL_DONE"
