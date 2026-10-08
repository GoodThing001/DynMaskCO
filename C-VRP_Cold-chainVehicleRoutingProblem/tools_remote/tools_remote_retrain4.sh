#!/bin/bash
# 2026-10-01 权重保存 bug 修复后：归档全部 init 权重 checkpoint，重训四臂
# （clock v2 + partial，pretrained + random_cvrp），顺序执行。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
B=results/a1_step3
mkdir -p $B

for d in maskco_train160_clock_pretrained_s42 maskco_train160_clock_random_cvrp_s42 \
         maskco_train160_partial_pretrained_s42 maskco_train160_partial_random_cvrp_s42; do
  if [ -d $B/$d ]; then
    mv $B/$d $B/${d}_init_weights_bug
    echo "archived $d -> ${d}_init_weights_bug"
  fi
done

$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $B/maskco_train160_clock_pretrained_s42 > $B/train160_clock_pretrained.log 2>&1
echo "CLOCK_PRE_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 \
  --out $B/maskco_train160_clock_random_cvrp_s42 > $B/train160_clock_random_cvrp.log 2>&1
echo "CLOCK_RCVRP_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --partial-mask --p-max 0.8 \
  --out $B/maskco_train160_partial_pretrained_s42 > $B/train160_partial_pretrained.log 2>&1
echo "PARTIAL_PRE_EXIT=$?"

$PY scripts/training/train_maskco_scenario.py \
  --arm random_cvrp --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --train-instances 160 --n-orders 200 --dim 128 --m-max 200 \
  --steps 2000 --batch-size 8 --seed 42 --partial-mask --p-max 0.8 \
  --out $B/maskco_train160_partial_random_cvrp_s42 > $B/train160_partial_random_cvrp.log 2>&1
echo "PARTIAL_RCVRP_EXIT=$?"
echo "RETRAIN4_DONE"
