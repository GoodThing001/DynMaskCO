#!/bin/bash
# step5 闭环 gate：在留出实例上比较 B / R / M-pre / M-trained（冻结预算 4.0）。
# 用法：bash _run_step5.sh <seed> <gpu_id>
set -e
SEED="$1"
GPU="$2"
source /home/hzeng/miniconda3/etc/profile.d/conda.sh
conda activate MASKCO_env
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/evaluation/run_step5_gate.py \
    --data data/heldout/dcc_50_r1_edod05_heldout.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --model-ckpt "results/m0_scale/mpre_reinforce_s${SEED}_clean/model.ckpt" \
    --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --budget 4.0 --max-instances 16 \
    --out "results/m0_scale/step5_heldout_s${SEED}" \
    > "results/m0_scale/_step5_heldout_s${SEED}.log" 2>&1
