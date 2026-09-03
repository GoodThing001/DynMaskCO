#!/bin/bash
# ============================================================
# run_typed_retrain.sh — R1 strict-online baseline 重训（train-only）
# ============================================================
# 目的：在修复后的数据（D1 release-feasible）+ 修复后的训练（vis_k 逐步揭示 +
#       visible-only loss + reveal_time 掩码 + edge_feat 掩码）上重训 5-seed。
#
# 用法:
#   bash run_typed_retrain.sh smoke          # 1 seed 2000 steps 冒烟 (~10min)
#   bash run_typed_retrain.sh single         # 1 seed 50K steps（R1.5 seed42 单 seed 正式）
#   bash run_typed_retrain.sh full           # 5 seed 50K steps 全量 (~5h)
#   USE_EDGE=1 bash run_typed_retrain.sh full   # 边特征变体（正式 baseline）
#   USE_EDGE=0 bash run_typed_retrain.sh full   # 无边特征（A0 消融）
#
# ⚠️ 本脚本只训练，不做 offline cvrptw.py 评估。原因：offline 单发 decode 只
#    服务 t=0 可见客户（~51%），complete=0%，不是动态问题合法口径。
#    正式评估用 scripts/decoding/run_r1_eval.py（strict-online 滚动时域）。
# ============================================================
set -e

MODE="${1:-smoke}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

# 边特征变体开关（默认 1 = typed_v1_edge，正式 baseline）
USE_EDGE="${USE_EDGE:-1}"
# P1-F：checkpoint 目录版本化。旧 timestep 训练的 ckpt 在 r1_baseline/（legacy），
# 修复后的干净重训写到 r1_5_baseline/，避免 SKIP 逻辑把「重训」误判为「已存在」。
RUN_TAG="${RUN_TAG:-r1_5_baseline}"

# ═══════════════════════════════════════════════════════════
# Frozen Parameters
# ═══════════════════════════════════════════════════════════
MODEL_CONFIG="softcap_fn"
ENCODER_INPUT_DIM=7
NUM_NODES=50
CAPACITY=50
BATCH_SIZE=64
PEAK_LR="1e-3"
OPTIMIZER="adamw"
WEIGHT_DECAY="1e-2"
MASKING_MODE="spatio_temporal"
ONLINE_SEQ_STEPS=5

case "$MODE" in
    smoke)
        NUM_STEPS=2000; SAVE_INTERVAL=2000
        TRAIN_SEEDS=(42)
        ;;
    single)
        NUM_STEPS=50000; SAVE_INTERVAL=5000
        TRAIN_SEEDS=(42)
        ;;
    full)
        NUM_STEPS=50000; SAVE_INTERVAL=5000
        TRAIN_SEEDS=(42 123 999 2025 2026)
        ;;
    *)
        echo "Unknown mode: $MODE (use smoke|single|full)"; exit 1
        ;;
esac

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/baseline/50_node"
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"

if [ "$USE_EDGE" = "1" ]; then
    CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/${RUN_TAG}/typed_v1_edge"
    LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/${RUN_TAG}/typed_v1_edge"
    EDGE_FLAG="--use_edge_feat"
    TAG="typed_v1_edge"
else
    CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/${RUN_TAG}/typed_v1"
    LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/${RUN_TAG}/typed_v1"
    EDGE_FLAG=""
    TAG="typed_v1"
fi
MERGED_TRAIN="${DATA_DIR}/dcc_50_mixed_edod_train.npz"

echo "============================================================"
echo "R1 baseline 重训（train-only）| tag=$TAG | mode=$MODE | edge=$USE_EDGE"
echo "============================================================"
echo "Seeds: ${TRAIN_SEEDS[*]} | Steps: $NUM_STEPS | Edge: $EDGE_FLAG"
echo ""

mkdir -p "$CKPT_DIR/phase3c" "$LOG_DIR"

# ═══════════════════════════════════════════════════════════
# 合并训练数据（9 域 balanced，每次无条件重建，P1-4 修复复用旧文件风险）
# ═══════════════════════════════════════════════════════════
rm -f "$MERGED_TRAIN"
echo "--- Merging training data (fresh) ---"
python3 -c "
import numpy as np
files = []
for t in ['r1','c1','rc1']:
    for e in ['02','05','08']:
        f = '${DATA_DIR}/train/dcc_50_' + t + '_edod' + e + '_train.npz'
        files.append(dict(np.load(f)))
merged = {}
for key in files[0].keys():
    merged[key] = np.concatenate([f[key] for f in files], axis=0)
np.savez('${MERGED_TRAIN}', **merged)
print(f'Merged {len(files)} files ({merged[\"coords\"].shape[0]} instances) -> ${MERGED_TRAIN}')
"

# ═══════════════════════════════════════════════════════════
# 训练（Phase 3c K=5 online seq）
# ═══════════════════════════════════════════════════════════
for SEED in "${TRAIN_SEEDS[@]}"; do
    SEED_CKPT="${CKPT_DIR}/phase3c/seed${SEED}"
    CKPT_FILE="${SEED_CKPT}/step${NUM_STEPS}.ckpt"
    mkdir -p "$SEED_CKPT"
    if [ -f "$CKPT_FILE" ]; then
        echo "  [SKIP] seed=${SEED} 已存在: $CKPT_FILE"
        continue
    fi
    echo "  [TRAIN] Phase3c K=5 seed=${SEED} (${NUM_STEPS} steps, edge=$USE_EDGE)..."
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
        --gpu_id 0 --num_nodes $NUM_NODES --capacity $CAPACITY \
        --model_config $MODEL_CONFIG --encoder_input_dim $ENCODER_INPUT_DIM \
        --peak_lr $PEAK_LR --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS --save_interval $SAVE_INTERVAL \
        --data "$MERGED_TRAIN" --masking_mode $MASKING_MODE \
        --online_seq_training --online_seq_steps $ONLINE_SEQ_STEPS \
        $EDGE_FLAG \
        --logdir "${LOG_DIR}/phase3c_seed${SEED}" --savedir "$SEED_CKPT" \
        --optimizer_type $OPTIMIZER --weight_decay $WEIGHT_DECAY \
        --target_disruption None --seed $SEED
done

echo ""
echo "============================================================"
echo "训练完成"
echo "============================================================"
echo "CKPT: $CKPT_DIR/phase3c/seed*/step${NUM_STEPS}.ckpt"
echo ""
echo "正式评估（strict-online，滚动时域）："
echo "  cd C-VRP_Cold-chainVehicleRoutingProblem"
echo "  python scripts/decoding/run_r1_eval.py --mode full"
