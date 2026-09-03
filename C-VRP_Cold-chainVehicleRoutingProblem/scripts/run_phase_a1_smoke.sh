#!/bin/bash
# Phase A1 smoke test — 1-seed 2000 steps 验证 EdgeState 模型能训练
#
# 目的：验证 DynamicColdChainModelEdgeState 能正常训练（loss 下降、无 NaN）。
# 用 R1 EDoD=0.5 单域快速验证（正式训练用 9-domain balanced）。
#
# 用法:
#   bash scripts/run_phase_a1_smoke.sh

set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node/train/dcc_50_r1_edod05_train.npz"
SAVEDIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_smoke/seed42"
LOGDIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/phase_a1_smoke"

echo "=== Phase A1 smoke test ==="
echo "  模型: DynamicColdChainModelEdgeState (5D 边特征 + EdgeBiasProjector)"
echo "  数据: R1 EDoD=0.5 (单域，快速验证)"
echo "  步数: 2000 (smoke)"
echo "  seed: 42"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u \
  "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py" \
  --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
  --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
  --num_steps 2000 --save_interval 1000 --data "$DATA" \
  --masking_mode spatio_temporal --online_seq_training --online_seq_steps 5 \
  --use_edge_feat --use_edge_state \
  --logdir "$LOGDIR" --savedir "$SAVEDIR" \
  --optimizer_type adamw --weight_decay 1e-2 --seed 42

echo ""
echo "=== smoke test 完成 ==="
echo "  checkpoint: $SAVEDIR/step2000.ckpt"
echo ""
echo "验收标准（Day 6）："
echo "  ✓ 训练 loss 正常下降（无 NaN / 不爆炸）"
echo "  ✓ checkpoint 正常保存"
echo "  ✓ 边特征参数（edge_bias_projector）参与训练"
