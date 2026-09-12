#!/bin/bash
# ============================================================
# P0-1: 新增训练种子（5 → 10 seeds）
# ============================================================
# 背景：5-seed paired Wilcoxon 的最小 p 值只有 2/2^5=0.0625，
#       无法达到 p<0.05。新增 5 个 seed，把核心比较补到 10 seed。
#
# 新增 seeds: 100, 200, 300, 400, 500（避开现有 42/123/999/2025/2026）
#
# 用法:
#   bash run_add_seeds.sh           # 训练新增 5 个 seed 的 Phase 3c
#   bash run_add_seeds.sh smoke     # 冒烟（1 个 seed, 2K 步）
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

MODE="${1:-full}"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_mixed_edod_train.npz"
SAVEDIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/frozen_v1/phase3c"
TRAIN="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"

if [ "$MODE" = "smoke" ]; then
    NEW_SEEDS=(100)
    STEPS=2000
    SAVE_INT=2000
else
    NEW_SEEDS=(100 200 300 400 500)
    STEPS=50000
    SAVE_INT=5000
fi

echo "============================================================"
echo "P0-1: 新增训练种子 (${#NEW_SEEDS[@]} seeds)"
echo "============================================================"

for SEED in "${NEW_SEEDS[@]}"; do
    CKPT="${SAVEDIR}/seed${SEED}/step${STEPS}.ckpt"
    if [ -f "$CKPT" ]; then
        echo "  [SKIP] seed=${SEED} 已存在: $CKPT"
        continue
    fi
    echo "  [TRAIN] Phase3c seed=${SEED} (${STEPS} steps)..."
    mkdir -p "${SAVEDIR}/seed${SEED}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN" \
        --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
        --num_steps $STEPS --save_interval $SAVE_INT \
        --data "$DATA" --masking_mode spatio_temporal \
        --online_seq_training --online_seq_steps 5 \
        --logdir "C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/frozen_v1/phase3c_seed${SEED}" \
        --savedir "${SAVEDIR}/seed${SEED}" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None \
        --seed $SEED
done

echo ""
echo "=== 新增 seed 训练完成 ==="
ls -lh "${SAVEDIR}"/seed{100,200,300,400,500}/step${STEPS}.ckpt 2>/dev/null || true
echo ""
echo "下一步: 用 10 seed 重跑统计评估（run_method_freeze.sh 需先更新 TRAIN_SEEDS）"
