#!/bin/bash
# ============================================================
# P0-5: Method Freeze — DynMaskCO v1 One-Click Reproduction
# ============================================================
# FROZEN CONFIG (2026-08-12):
#   Model:   DynamicColdChainModel (7D), softcap_fn
#   Norm:    coord_normalize_visible (P0-3a fix)
#   Data:    R1+C1+RC1 × EDoD 0.2+0.5+0.8 (mixed, ~10K instances)
#   Train:   50K steps, AdamW lr=1e-3, batch=64, ST-mask
#   Phase3c: K=5 online seq, keep=0.15→0.85 schedule
#   Decode:  K=16 beam, TW 2-opt 4 steps, 8 runs × 40 cycles
#   Seeds:   5 train (42/123/999/2025/2026) × 3 eval (42/123/999)
#   Baseline:OR-Tools clairvoyant J*=9.1 (R1 EDoD=0.5, 25 veh, 60s)
#
# Usage:
#   bash run_method_freeze.sh smoke         # Quick verify (1 seed, 2K steps, ~10min)
#   bash run_method_freeze.sh phase3c       # Train 5-seed Phase 3c only (~5h)
#   bash run_method_freeze.sh full          # Full 5-seed ST+Phase3c+eval (~24h)
#   bash run_method_freeze.sh eval          # Evaluate existing checkpoints
#   bash run_method_freeze.sh env           # Print frozen environment
# ============================================================
set -e

MODE="${1:-smoke}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

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
BEAM_WIDTH=16
KEEP_RATE=0.3
TWO_OPT_STEPS=4
GUMBEL_SCALE=0.0
AUGMENT_LEVEL=0
SAMPLING_STEPS=2

case "$MODE" in
    smoke)
        NUM_STEPS=2000; SAVE_INTERVAL=2000
        TRAIN_SEEDS=(42)
        DECODE_RUNS=4; DECODE_CYCLES=20
        DO_PHASE3C=true
        ;;
    phase3c)
        NUM_STEPS=50000; SAVE_INTERVAL=5000
        TRAIN_SEEDS=(42 123 999 2025 2026 100 200 300 400 500)
        DECODE_RUNS=8; DECODE_CYCLES=40
        DO_PHASE3C=true; SKIP_ST=true
        ;;
    full)
        NUM_STEPS=50000; SAVE_INTERVAL=5000
        TRAIN_SEEDS=(42 123 999 2025 2026 100 200 300 400 500)
        DECODE_RUNS=8; DECODE_CYCLES=40
        DO_PHASE3C=true; SKIP_ST=false
        ;;
    eval|env)
        NUM_STEPS=50000; SAVE_INTERVAL=5000
        TRAIN_SEEDS=(42 123 999 2025 2026 100 200 300 400 500)
        DECODE_RUNS=8; DECODE_CYCLES=40
        DO_PHASE3C=false; SKIP_ST=true
        ;;
esac

EVAL_SEED=42   # fixed decode seed; randomness across train seeds is what matters

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/frozen_v1"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/frozen_v1"
MERGED_TRAIN="${DATA_DIR}/dcc_50_mixed_edod_train.npz"
RESULT_CSV="${LOG_DIR}/results/full_matrix.csv"
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"

TEST_TYPES=(R1 C1 RC1)
TEST_EDODS=(0.2 0.5 0.8)

# ═══════════════════════════════════════════════════════════
# env — Print frozen environment and exit
# ═══════════════════════════════════════════════════════════

if [ "$MODE" = "env" ]; then
    echo "============================================================"
    echo "FROZEN ENVIRONMENT — DynMaskCO v1"
    echo "============================================================"
    echo "Date:     $(date -I)"
    echo "Host:     $(hostname)"
    echo "Python:   $(python3 --version)"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
    echo ""
    echo "--- Critical packages ---"
    pip freeze 2>/dev/null | grep -E "^(jax|flax|numpy|optax|triton|ortools|pyvrp|torch)="
    echo ""
    echo "--- Frozen Config ---"
    echo "  model=$MODEL_CONFIG  dim=$ENCODER_INPUT_DIM  nodes=$NUM_NODES  cap=$CAPACITY"
    echo "  steps=$NUM_STEPS  batch=$BATCH_SIZE  lr=$PEAK_LR  opt=$OPTIMIZER wd=$WEIGHT_DECAY"
    echo "  mask=$MASKING_MODE  online_seq=$ONLINE_SEQ_STEPS"
    echo "  beam=K${BEAM_WIDTH}  keep=$KEEP_RATE  2opt=${TWO_OPT_STEPS}steps"
    echo "  runs=${DECODE_RUNS}  cycles=${DECODE_CYCLES}  aug=$AUGMENT_LEVEL"
    echo "  train_seeds=${TRAIN_SEEDS[*]}  eval_seed=${EVAL_SEED} (fixed)"
    exit 0
fi

mkdir -p "$CKPT_DIR" "$CKPT_DIR/phase3c" "$LOG_DIR" "$(dirname "$RESULT_CSV")"

# ═══════════════════════════════════════════════════════════
# Header + env validation
# ═══════════════════════════════════════════════════════════

echo "============================================================"
echo "DynMaskCO v1 — Method Freeze"
echo "============================================================"
echo "Mode: $MODE | Train seeds: ${TRAIN_SEEDS[*]} | Steps: $NUM_STEPS"
echo "Phase3c: ${DO_PHASE3C} | ST-mask: $([ "$SKIP_ST" = true ] && echo SKIP || echo YES)"
echo ""

python3 -c "
import jax, flax, numpy as np
print(f'JAX {jax.__version__} | Flax {flax.__version__} | NumPy {np.__version__}')
print(f'GPU: {jax.devices()}')
from ortools.constraint_solver import pywrapcp
print('OR-Tools: OK')
" || { echo "FATAL: environment broken"; exit 1; }

# ═══════════════════════════════════════════════════════════
# Merge training data if missing
# ═══════════════════════════════════════════════════════════

if [ ! -f "$MERGED_TRAIN" ]; then
    echo ""
    echo "--- Merging training data ---"
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
np.savez('${MERGED_TRAIN}', **merged)
print(f'Merged {len(files)} files ({merged[\"coords\"].shape[0]} instances) -> ${MERGED_TRAIN}')
"
fi

# ═══════════════════════════════════════════════════════════
# Normalizer verification
# ═══════════════════════════════════════════════════════════

echo ""
echo "--- Normalizer Verification ---"
python3 -c "
import sys, os
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models')
from cvrptw_utils import coord_normalize_visible
import jax.numpy as jnp
coords = jnp.array([[[0.2,0.3],[0.5,0.5],[0.8,0.9]]])
vis = jnp.array([[1.0, 0.0, 1.0]])
out = coord_normalize_visible(coords, vis)
print('coord_normalize_visible: OK')
" || { echo "FATAL: coord_normalize_visible missing"; exit 1; }

# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

for SEED in "${TRAIN_SEEDS[@]}"; do
    SEED_CKPT="${CKPT_DIR}/seed${SEED}"
    SEED_3C="${CKPT_DIR}/phase3c/seed${SEED}"
    mkdir -p "$SEED_CKPT" "$SEED_3C"

    echo ""
    echo "============================================================"
    echo "Seed=$SEED"
    echo "============================================================"

    # ST-mask baseline
    if [ "$SKIP_ST" != true ]; then
        CKPT_FILE="${SEED_CKPT}/step${NUM_STEPS}.ckpt"
        if [ -f "$CKPT_FILE" ]; then
            echo "  [SKIP] ST-mask: $CKPT_FILE"
        else
            echo "  [TRAIN] ST-mask seed=$SEED (${NUM_STEPS} steps)..."
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
                --gpu_id 0 --num_nodes $NUM_NODES --capacity $CAPACITY \
                --model_config $MODEL_CONFIG --encoder_input_dim $ENCODER_INPUT_DIM \
                --peak_lr $PEAK_LR --batch_size $BATCH_SIZE \
                --num_steps $NUM_STEPS --save_interval $SAVE_INTERVAL \
                --data "$MERGED_TRAIN" --masking_mode $MASKING_MODE \
                --logdir "${LOG_DIR}/st_seed${SEED}" --savedir "$SEED_CKPT" \
                --optimizer_type $OPTIMIZER --weight_decay $WEIGHT_DECAY \
                --target_disruption None --seed $SEED
        fi
    fi

    # Phase 3c K=5 online seq
    if [ "$DO_PHASE3C" = true ]; then
        CKPT_3C="${SEED_3C}/step${NUM_STEPS}.ckpt"
        if [ -f "$CKPT_3C" ]; then
            echo "  [SKIP] Phase3c: $CKPT_3C"
        else
            echo "  [TRAIN] Phase3c K=5 seed=$SEED (${NUM_STEPS} steps)..."
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
                --gpu_id 0 --num_nodes $NUM_NODES --capacity $CAPACITY \
                --model_config $MODEL_CONFIG --encoder_input_dim $ENCODER_INPUT_DIM \
                --peak_lr $PEAK_LR --batch_size $BATCH_SIZE \
                --num_steps $NUM_STEPS --save_interval $SAVE_INTERVAL \
                --data "$MERGED_TRAIN" --masking_mode $MASKING_MODE \
                --online_seq_training --online_seq_steps $ONLINE_SEQ_STEPS \
                --logdir "${LOG_DIR}/phase3c_seed${SEED}" --savedir "$SEED_3C" \
                --optimizer_type $OPTIMIZER --weight_decay $WEIGHT_DECAY \
                --target_disruption None --seed $SEED
        fi
    fi
done

# ═══════════════════════════════════════════════════════════
# Evaluation — all Phase 3c train seeds × full type/EDoD matrix
# ═══════════════════════════════════════════════════════════
# Statistical unit = train seed (independent training runs).
# Each train seed evaluated with ONE fixed decode seed (EVAL_SEED).
# ═══════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "Evaluation: ${#TEST_TYPES[@]} types × ${#TEST_EDODS[@]} EDoDs × ${#TRAIN_SEEDS[@]} train seeds"
echo "============================================================"

BEAM_FLAGS="--enable_resource_decoder --beam_width $BEAM_WIDTH"
echo "type,edod,train_seed,cost,tw_feas,tw_viol" > "$RESULT_CSV"

N_EVAL=0
N_SKIP_TRAIN=0

for TSEED in "${TRAIN_SEEDS[@]}"; do
    CK="${CKPT_DIR}/phase3c/seed${TSEED}/step${NUM_STEPS}.ckpt"
    if [ ! -f "$CK" ]; then
        echo "  [SKIP] train seed=$TSEED (no phase3c checkpoint)"
        N_SKIP_TRAIN=$((N_SKIP_TRAIN+1))
        continue
    fi

    echo ""
    echo "--- Train seed=$TSEED ---"

    for TYPE in "${TEST_TYPES[@]}"; do
      for EDOD in "${TEST_EDODS[@]}"; do
        ESTR=$(echo "$EDOD" | sed 's/\.//')
        DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"
        [ ! -f "$DATA" ] && continue

        echo -n "  ${TYPE} EDoD=${EDOD} ... "
        OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -u "$DECODE_SCRIPT" \
            --capacity $CAPACITY --penalty 3. --data "$DATA" --ckpt "$CK" \
            --keep_rate $KEEP_RATE --two_opt_steps $TWO_OPT_STEPS \
            --batch_size 8 --runs $DECODE_RUNS --cycles $DECODE_CYCLES \
            --sampling_steps $SAMPLING_STEPS --augment_level $AUGMENT_LEVEL \
            --gumbel_scale_factor $GUMBEL_SCALE --seed $EVAL_SEED \
            --threads_over_batches 1 \
            --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
            $BEAM_FLAGS 2>&1)
        COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
        FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
        VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
        echo "${TYPE},${EDOD},${TSEED},${COST},${FEAS},${VIOL}" >> "$RESULT_CSV"
        echo "feas=${FEAS} cost=${COST}"
        N_EVAL=$((N_EVAL+1))
      done
    done
done

echo ""
echo "  Total evals: $N_EVAL (skipped $N_SKIP_TRAIN train seeds without checkpoint)"

# ═══════════════════════════════════════════════════════════
# Statistical Report — aggregate across train seeds
# ═══════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "Statistical Report (5 train seeds)"
echo "============================================================"

python3 << PYEOF
import pandas as pd, numpy as np

df = pd.read_csv('${RESULT_CSV}')
df['feas_num'] = df['tw_feas'].str.rstrip('%').astype(float)

n_train = df['train_seed'].nunique()

print()
print("=== DynMaskCO v1 — Frozen Results ({} train seeds) ===".format(n_train))
print(f"  Evaluations: {df.shape[0]} ({df.shape[0]//n_train} per train seed)")
print(f"  Config:      Phase3c K=5 + K={${BEAM_WIDTH}} beam + TW 2opt")
print()

print(f"{'Type':<6} {'EDoD':<6} {'Feas':>14} {'Cost':>14}")
print('-' * 44)
for t in ['R1', 'C1', 'RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type'] == t) & (df['edod'] == e)]
        if len(sub) == 0: continue
        fm, fs = sub['feas_num'].mean(), sub['feas_num'].std()
        cm, cs = sub['cost'].mean(), sub['cost'].std()
        print(f"{t:<6} {e:<6} {fm:>5.1f}% ±{fs:>4.1f}%  {cm:>6.2f} ±{cs:>4.2f}")

print('-' * 44)
ov_feas = df['feas_num'].mean()
ov_cost = df['cost'].mean()
print(f"{'ALL':<6} {'':<6} {ov_feas:>5.1f}%         {ov_cost:>6.2f}")

# Per-type avg (aggregating EDoDs first, then train seeds)
print()
print("--- Per-type summary (mean over EDoDs, ± over train seeds) ---")
for t in ['R1', 'C1', 'RC1']:
    # per train seed: mean over its 3 EDoDs
    sub = df[df['type'] == t].groupby('train_seed').agg(
        feas=('feas_num', 'mean'), cost=('cost', 'mean'))
    print(f"  {t:<4} feas={sub['feas'].mean():.1f}%±{sub['feas'].std():.1f}%  "
          f"cost={sub['cost'].mean():.2f}±{sub['cost'].std():.2f}")

print()
print("--- Reference Comparison ---")
print(f"  OR-Tools Clairvoyant J* = 9.1  (R1 EDoD=0.5, static full-info, 25 veh, 60s)")
r1_05 = df[(df['type'] == 'R1') & (df['edod'] == 0.5)]
if len(r1_05) > 0:
    r1_cost = r1_05['cost'].mean()
    r1_std = r1_05['cost'].std()
    gap = r1_cost - 9.1
    print(f"  DynMaskCO R1 EDoD=0.5     = {r1_cost:.2f} ± {r1_std:.2f}")
    print(f"  Clairvoyance Gap           = {gap:.2f} ({gap/9.1*100:.0f}%)")
    print(f"  TW Feas                    = {r1_05['feas_num'].mean():.1f}%")

print()
print("--- Frozen Config ---")
print(f"  Seeds (train):  ${TRAIN_SEEDS[*]}")
print(f"  Seeds (eval):   ${EVAL_SEED} (fixed)")
print(f"  Norm:           coord_normalize_visible (P0-3a)")
print(f"  Masking:        ${MASKING_MODE}")
print(f"  Beam:           K=${BEAM_WIDTH}, 2opt=${TWO_OPT_STEPS} steps")
print(f"  Runs×Cycles:    ${DECODE_RUNS}×${DECODE_CYCLES}")
PYEOF

echo ""
echo "============================================================"
echo "METHOD FREEZE COMPLETE"
echo "============================================================"
echo "Results:    $RESULT_CSV"
echo "CKPTs:      $CKPT_DIR/phase3c/"
echo "Reproduce:  bash $0 full"
