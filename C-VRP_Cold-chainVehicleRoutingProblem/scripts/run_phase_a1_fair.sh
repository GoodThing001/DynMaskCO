#!/bin/bash
# Phase A1 公平对比 — A0（基线）vs A1（EdgeState），相同数据（R1-0.5 单域）
#
# 关键：两者都用 R1-0.5 单域训练，唯一差异是 --use_edge_state。
# 消除「混合数据 vs 单域」的干扰，准确定位 EdgeBiasProjector 是否有效。

set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
DATA_TRAIN="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node/train/dcc_50_r1_edod05_train.npz"
DATA_VAL="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node/val/dcc_50_r1_edod05_val.npz"

COMMON_TRAIN_FLAGS="--gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
  --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
  --num_steps 50000 --save_interval 5000 \
  --masking_mode spatio_temporal --online_seq_training --online_seq_steps 5 \
  --optimizer_type adamw --weight_decay 1e-2 --seed 42"

COMMON_DECODE_FLAGS="--capacity 50 --penalty 3. \
  --keep_rate 0.9 --two_opt_steps 100 --batch_size 128 --runs 128 --cycles 1 \
  --sampling_steps 5 --augment_level 0 --gumbel_scale_factor 5.0 --seed 42 \
  --threads_over_batches 1 \
  --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
  --enable_resource_decoder --beam_width 16"

echo "=== Phase A1 公平对比（R1-0.5 单域，A0 vs A1）==="
echo ""

# --- Step 1: 训练 A0 基线（DynamicColdChainModel + edge_feat，无 edge_state）---
echo "━━━ Step 1: 训练 A0 基线（单域）━━━"
A0_CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_fair/a0/seed42/step50000.ckpt"
if [ ! -f "$A0_CKPT" ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$SCRIPT" \
        --data "$DATA_TRAIN" --use_edge_feat \
        --logdir "C-VRP_Cold-chainVehicleRoutingProblem/logs/phase_a1_fair/a0" \
        --savedir "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_fair/a0" \
        $COMMON_TRAIN_FLAGS
else
    echo "  [SKIP] A0 已训练: $A0_CKPT"
fi
echo ""

# --- Step 2: 训练 A1（EdgeState，单域）---
echo "━━━ Step 2: 训练 A1（EdgeState，单域）━━━"
A1_CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_fair/a1/seed42/step50000.ckpt"
if [ ! -f "$A1_CKPT" ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$SCRIPT" \
        --data "$DATA_TRAIN" --use_edge_feat --use_edge_state \
        --logdir "C-VRP_Cold-chainVehicleRoutingProblem/logs/phase_a1_fair/a1" \
        --savedir "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_fair/a1" \
        $COMMON_TRAIN_FLAGS
else
    echo "  [SKIP] A1 已训练: $A1_CKPT"
fi
echo ""

# --- Step 3: 评估对比 ---
echo "━━━ Step 3: 评估对比（R1-0.5 val）━━━"
echo ""
echo "【A0 基线（单域）】"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
    --ckpt "$A0_CKPT" --data "$DATA_VAL" $COMMON_DECODE_FLAGS 2>&1 | \
    grep -E "mean cost|TW feas rate" | tail -4

echo ""
echo "【A1 EdgeState（单域）】"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
    --ckpt "$A1_CKPT" --data "$DATA_VAL" --use_edge_state $COMMON_DECODE_FLAGS 2>&1 | \
    grep -E "mean cost|TW feas rate" | tail -4

echo ""
echo "=== 公平对比完成 ==="
echo ""
echo "判断标准（公平对比后）："
echo "  A1 < A0 → EdgeBiasProjector 有效，之前 +13% 是混合数据优势"
echo "  A1 ≈ A0 → EdgeBiasProjector 中性，5D 边特征无额外信息"
echo "  A1 > A0 → EdgeBiasProjector 引入噪声，方向需调整"
