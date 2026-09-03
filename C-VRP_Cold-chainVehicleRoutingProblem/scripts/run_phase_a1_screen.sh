#!/bin/bash
# Phase A1 screening — 评估 A0（typed_edge）和 A1（EdgeState）在 R1-0.5 val 上对比
#
# 注意：需先运行 run_phase_a1_train.sh 训练 A1（50000 steps）。

set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node/val/dcc_50_r1_edod05_val.npz"

A0_CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1_edge/phase3c/seed42/step50000.ckpt"
A1_CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phase_a1_train/seed42/step50000.ckpt"

COMMON_FLAGS="--capacity 50 --penalty 3. \
  --keep_rate 0.9 --two_opt_steps 100 --batch_size 128 --runs 128 --cycles 1 \
  --sampling_steps 5 --augment_level 0 --gumbel_scale_factor 5.0 --seed 42 \
  --threads_over_batches 1 \
  --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
  --enable_resource_decoder --beam_width 16"

echo "=== Phase A1 screening: A0 vs A1 (R1-0.5 val) ==="
echo ""

# --- 评估 A0（typed_edge 基线）---
if [ -f "$A0_CKPT" ]; then
    echo "━━━ A0 (typed_edge) ━━━"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$SCRIPT" \
        --ckpt "$A0_CKPT" --data "$DATA" $COMMON_FLAGS 2>&1 | \
        grep -E "mean cost|TW feas rate" | tail -4
    echo ""
else
    echo "⚠️  A0 checkpoint 不存在: $A0_CKPT"
fi

# --- 评估 A1（EdgeState）---
if [ -f "$A1_CKPT" ]; then
    echo "━━━ A1 (EdgeState) ━━━"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$SCRIPT" \
        --ckpt "$A1_CKPT" --data "$DATA" --use_edge_state $COMMON_FLAGS 2>&1 | \
        grep -E "mean cost|TW feas rate" | tail -4
    echo ""
else
    echo "⚠️  A1 checkpoint 不存在: $A1_CKPT（先运行 run_phase_a1_train.sh）"
fi

echo "=== screening 完成 ==="
echo ""
echo "验收标准（Day 7）："
echo "  A1 cost 比 A0 低 ≥2% → 进入 9-domain 正式训练（W3）"
echo "  A1 cost 与 A0 相当或更差 → 回退，检查 EdgeBiasProjector 设计"
