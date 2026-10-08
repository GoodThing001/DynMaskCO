#!/bin/bash
# 步骤 3 驱动烟测（I1 时钟世代；1 天 × 4 臂 × 主档，time_limit=2s 仅验接线/身份/世代校验；数值非证据）
# 注意：驱动自动从 model.bin 的 config.json 读 train_instances/dim/m_max/arm（谱系一致），无需显式传。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
OUT=results/a1_step3/_smoke_driver_1d_clock
$PY scripts/evaluation/run_a1_step3_compare.py \
  --model-bin results/a1_step3/maskco_train160_clock_pretrained_s42/model.bin \
  --arm pretrained \
  --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --dev-instances 1 --n-orders 200 --main-only \
  --time-limit 2 --K 10 --workers 4 \
  --allow-code-mismatch \
  --out $OUT > $OUT.log 2>&1
echo "SMOKE_EXIT=$?"
tail -n 25 $OUT.log
echo "SMOKE_DONE"
