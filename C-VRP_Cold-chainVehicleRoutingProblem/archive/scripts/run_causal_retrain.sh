#!/bin/bash
# ============================================================
# P0-3a Causal Retraining — coord_normalize_visible fixed
# ============================================================
# 训练使用修复后的 coord_normalize_visible（仅可见节点统计量），
# 消除旧 normalizer 的未来坐标统计泄漏。
#
# 产出：
#   ckpts/p0_fix/causal_v1/step50000.ckpt  — ST-mask baseline
#   ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt — K=5 online seq (最强)
#
# 用法:
#   bash run_causal_retrain.sh smoke     # 2K steps 冒烟验证 (~5min)
#   bash run_causal_retrain.sh full      # 50K steps 完整训练 (~2h)
#   bash run_causal_retrain.sh phase3c   # 50K online seq 最强 (~4h)
# ============================================================
set -e

MODE="${1:-smoke}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

# ── Paths ──
DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_BASE="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/causal_v1"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/causal_v1"
MERGED="${DATA_DIR}/dcc_50_mixed_edod_train.npz"
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"

mkdir -p "$CKPT_BASE" "$LOG_DIR"

# ── Step 1: Verify merged data exists ──
if [ ! -f "$MERGED" ]; then
    echo "========== STEP 0: 合并训练数据 =========="
    python3 -c "
import numpy as np
files = []
for t in ['r1','c1','rc1']:
    for e in ['02','05','08']:
        f = '${DATA_DIR}/dcc_50_' + t + '_edod' + e + '_train.npz'
        files.append(dict(np.load(f)))
merged = {}
for key in files[0].keys():
    merged[key] = np.concatenate([f[key] for f in files], axis=0)
np.savez('${MERGED}', **merged)
print(f'Merged {len(files)} files -> ${MERGED} ({merged[\"coords\"].shape[0]} instances)')
vm = merged['visible_mask']
print(f'visible: {vm.mean():.1%} | future: {(1-vm).mean():.1%}')
"
fi

# ── Training params ──
case "$MODE" in
    smoke)
        STEPS=2000; SAVE_INT=2000; BATCH=64
        echo "=== MODE: SMOKE (${STEPS} steps) ==="
        ;;
    full|phase3c)
        STEPS=50000; SAVE_INT=5000; BATCH=64
        echo "=== MODE: FULL (${STEPS} steps) ==="
        ;;
esac

# ============================================================
# Training 1: ST-mask baseline (no online seq)
# ============================================================
CKPT_ST="${CKPT_BASE}/step${STEPS}.ckpt"

if [ "$MODE" != "phase3c" ]; then
    echo ""
    echo "============================================================"
    echo "Training 1: Causal ST-mask (coord_normalize_visible)"
    echo "============================================================"
    if [ -f "$CKPT_ST" ]; then
        echo "  SKIP (ckpt exists: $CKPT_ST)"
    else
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
            --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
            --encoder_input_dim 7 --peak_lr 1e-3 --batch_size $BATCH \
            --num_steps $STEPS --save_interval $SAVE_INT \
            --data "$MERGED" --masking_mode spatio_temporal \
            --logdir "${LOG_DIR}/st_mask" --savedir "$CKPT_BASE" \
            --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
    fi
fi

# ============================================================
# Training 2: Phase 3c — K=5 online seq (strongest)
# ============================================================
CKPT_3C="${CKPT_BASE}/phase3c/step${STEPS}.ckpt"

if [ "$MODE" = "phase3c" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "============================================================"
    echo "Training 2: Phase 3c K=5 Online Seq (strongest)"
    echo "============================================================"
    mkdir -p "${CKPT_BASE}/phase3c"
    if [ -f "$CKPT_3C" ]; then
        echo "  SKIP (ckpt exists: $CKPT_3C)"
    else
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
            --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
            --encoder_input_dim 7 --peak_lr 1e-3 --batch_size $BATCH \
            --num_steps $STEPS --save_interval $SAVE_INT \
            --data "$MERGED" --masking_mode spatio_temporal \
            --online_seq_training --online_seq_steps 5 \
            --logdir "${LOG_DIR}/phase3c" --savedir "${CKPT_BASE}/phase3c" \
            --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
    fi
fi

# ============================================================
# Evaluation: quick smoke on all 9 test sets
# ============================================================
echo ""
echo "============================================================"
echo "Quick Evaluation (1 seed, beam decode)"
echo "============================================================"

CKPT_EVAL="$CKPT_3C"
if [ ! -f "$CKPT_EVAL" ]; then
    CKPT_EVAL="$CKPT_ST"
fi

if [ ! -f "$CKPT_EVAL" ]; then
    echo "  SKIP (no checkpoint available)"
    exit 0
fi

RESULT_FILE="${LOG_DIR}/eval_quick.csv"
echo "type,edod,seed,cost,tw_feas" > "$RESULT_FILE"

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"
    if [ ! -f "$DATA" ]; then continue; fi

    SEED=42
    echo -n "  ${TYPE} EDoD=${EDOD} ... "
    OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u "$DECODE_SCRIPT" \
        --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT_EVAL" \
        --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 \
        --runs 8 --cycles 40 --sampling_steps 2 \
        --augment_level 0 --gumbel_scale_factor 0. --seed $SEED \
        --threads_over_batches 1 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1)
    COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
    echo "${TYPE},${EDOD},${SEED},${COST},${FEAS}" >> "$RESULT_FILE"
    echo "feas=${FEAS} cost=${COST}"
  done
done

# Quick summary
python3 -c "
import pandas as pd
df = pd.read_csv('${RESULT_FILE}')
print()
print('=== Causal v1 Quick Eval (${MODE}, seed=42) ===')
feas_vals = df['tw_feas'].str.rstrip('%').astype(float)
print('Overall TW Feas: %.1f%%' % feas_vals.mean())
print('Avg Cost: %.2f' % df['cost'].mean())
print()
print(df.to_string(index=False))
"

echo ""
echo "=== DONE ==="
echo "Checkpoints: ${CKPT_BASE}/"
echo "  ST-mask:    $CKPT_ST"
echo "  Phase3c:    $CKPT_3C"
echo "Results:    $RESULT_FILE"
