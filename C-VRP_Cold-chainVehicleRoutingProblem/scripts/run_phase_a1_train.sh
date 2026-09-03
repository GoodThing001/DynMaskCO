#!/bin/bash
# Phase A1 训练（R1-0.5 单域，50000 steps，用于 Day 7 screening）
#
# 注意：这是 screening 用的单域训练（快速判断 A1 是否有改进）。
# 正式 Phase A 用 9-domain balanced（见优化方案.md §A.4）。

set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node/train/dcc_50_r1_edod05_train.npz"
SAVEDIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_train/seed42"
LOGDIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/phase_a1_train"

echo "=== Phase A1 训练（R1-0.5, 50000 steps, seed 42）==="
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u \
  "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py" \
  --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
  --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
  --num_steps 50000 --save_interval 5000 --data "$DATA" \
  --masking_mode spatio_temporal --online_seq_training --online_seq_steps 5 \
  --use_edge_feat --use_edge_state \
  --logdir "$LOGDIR" --savedir "$SAVEDIR" \
  --optimizer_type adamw --weight_decay 1e-2 --seed 42

echo ""
echo "=== 训练完成 ==="
echo "  checkpoint: $SAVEDIR/step50000.ckpt"
