#!/bin/bash
# ============================================================
# Phase 3c EDoD 全矩阵: 3c模型 + K=16 beam decoder
# R1/C1/RC1 × EDoD 0.2/0.5/0.8 + 3 seeds + anytime 4 budget points
# 用法: bash run_phase3c_full_matrix.sh
# ============================================================
set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/phase3c/step50000.ckpt"
RESULT_FILE="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/results_phase3c_full.csv"
ANYTIME_FILE="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/results_phase3c_anytime.csv"
DECODER="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"

echo "============================================================"
echo "Phase 3c EDoD Full Matrix + 方向1(热约束) + 方向2(beam+2opt)"
echo "  Model: phase3c (K=5 online seq training)"
echo "  Decoder: K=16 beam resource-state"
echo "  Extra: --enable_thermal_tracking + --enable_tw_aware_2opt_py"
echo "============================================================"
echo ""

# ── Part 1: EDoD 全矩阵 9 组 × 3 seeds ──
echo "type,edod,seed,cost,tw_feas,tw_viol" > "$RESULT_FILE"

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"

    for SEED in 42 123 999; do
        echo -n "  ${TYPE} EDoD=${EDOD} seed=${SEED} ... "
        OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -u "$DECODER" \
            --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
            --keep_rate 0.3 --batch_size 8 --runs 1 --cycles 1 --sampling_steps 1 \
            --two_opt_steps 0 --seed $SEED --gumbel_scale_factor 0. \
            --threads_over_batches 1 \
            --enable_resource_decoder --beam_width 16 \
            --enable_tw_aware_2opt_py --two_opt_steps 4 2>&1)
        COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
        FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
        VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
        echo "${TYPE},${EDOD},${SEED},${COST},${FEAS},${VIOL}" >> "$RESULT_FILE"
        echo "feas=${FEAS} cost=${COST}"
    done
  done
done

# ── Part 2: Anytime budget sweep (R1 EDoD=0.5, seed=42, 4 budget points) ──
echo ""
echo "--- Anytime Budget Sweep (R1 EDoD=0.5) ---"
echo "budget_ms,cost,tw_feas,tw_viol" > "$ANYTIME_FILE"
DATA="${DATA_DIR}/dcc_50_r1_edod05_test.npz"

for BUDGET in 50 100 200 500; do
    echo -n "  budget=${BUDGET}ms ... "
    OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u "$DECODER" \
        --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
        --keep_rate 0.3 --batch_size 8 --runs 1 --cycles 1 --sampling_steps 1 \
        --two_opt_steps 4 --seed 42 --gumbel_scale_factor 0. \
        --threads_over_batches 1 \
        --enable_resource_decoder --beam_width 16 \
        --enable_tw_aware_2opt_py \
        --enable_anytime --time_budget_ms $BUDGET 2>&1)
    COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
    VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
    echo "${BUDGET},${COST},${FEAS},${VIOL}" >> "$ANYTIME_FILE"
    echo "feas=${FEAS} cost=${COST}"
done

# ── Part 3: Summary ──
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"

python3 -c "
import pandas as pd, numpy as np

df = pd.read_csv('${RESULT_FILE}')
print()
print('=== Phase 3c EDoD Full Matrix (K=16 beam, 3 seeds) ===')
print()
print('%-6s %-6s %-16s %-16s %-10s' % ('Type','EDoD','Feas%','Cost','Viol'))
print('-' * 54)
for t in ['R1','C1','RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type']==t) & (df['edod']==e)]
        if len(sub)==0: continue
        f = sub['tw_feas'].str.rstrip('%').astype(float)
        c = sub['cost']
        v = sub['tw_viol']
        print('%-6s %-6s %4.1f%% ± %4.1f%%   %7.2f ± %5.2f   %4.1f' % (t, e, f.mean(), f.std(), c.mean(), c.std(), v.mean()))
    print()

print('Avg by Type:')
for t in ['R1','C1','RC1']:
    f = df[df['type']==t]['tw_feas'].str.rstrip('%').astype(float)
    c = df[df['type']==t]['cost']
    print('  %s: feas=%4.1f%%  cost=%6.2f' % (t, f.mean(), c.mean()))
print('Overall: feas=%4.1f%%  cost=%6.2f' % (
    df['tw_feas'].str.rstrip('%%').astype(float).mean(),
    df['cost'].mean()))

print()
print('--- Anytime Budget Sweep ---')
df_a = pd.read_csv('${ANYTIME_FILE}')
for _, row in df_a.iterrows():
    print('  %4dms: feas=%s  cost=%.2f' % (row['budget_ms'], row['tw_feas'], row['cost']))

print()
print('=== Comparison with old results ===')
print('  Oracle (leaked):     76.6%% feas, mixed_edod ckpt')
print('  8D Quality-Aware:    77.9%% feas (C++ insertion + EDD+TW2opt)')
print('  P0-3 Causal beam:   ~21.5 cost, 100%% feas, 0 viol (mixed_edod ckpt)')
print('  Phase 3c beam:       above (K=5 online seq ckpt, −3.8%% cost vs P0-3)')
"

echo ""
echo "=== DONE ==="
echo "Results: ${RESULT_FILE}"
echo "Anytime: ${ANYTIME_FILE}"
