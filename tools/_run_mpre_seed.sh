#!/bin/bash
# M-pre/M-trained REINFORCE 直接目标训练（种子稳健性）。用法：bash _run_mpre_seed.sh <seed> <gpu_id>
set -e
SEED="$1"
GPU="$2"
source /home/hzeng/miniconda3/etc/profile.d/conda.sh
conda activate MASKCO_env
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/training/train_mpre_reinforce.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --num-steps 1000 --batch-states 8 --K 4 --seed "$SEED" \
    --out "results/m0_scale/mpre_reinforce_s${SEED}_clean" \
    > "results/m0_scale/_mpre_s${SEED}_clean.log" 2>&1
