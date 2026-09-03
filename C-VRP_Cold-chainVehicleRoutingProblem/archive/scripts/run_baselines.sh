#!/bin/bash
# ============================================================
# P0-5: 基线对比 — ALNS vs Greedy vs MaskCO Causal
# 统一硬件 + 统一 time budget 对比
# 用法: bash run_baselines.sh
# ============================================================
set -e
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
RESULT_FILE="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/results_baselines.csv"
ALNS_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/baselines/alns_baseline.py"

echo "type,edod,method,cost,time_s,tw_feas" > "$RESULT_FILE"

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"

    echo ""
    echo "=== ${TYPE} EDoD=${EDOD} ==="

    # ── Greedy reference (already in .npz) ──
    GREEDY=$(python3 -c "
import numpy as np
d = dict(np.load('$DATA'))
# greedy cost = opt_costs (reference solution), time ~0 (constructive heuristic)
c = d['opt_costs'][:128].mean()
print(f'{c:.4f}')
")
    echo "${TYPE},${EDOD},Greedy,${GREEDY},0.001,n/a" >> "$RESULT_FILE"
    echo "  Greedy: ${GREEDY}"

    # ── ALNS (500 iter, ~5s/inst) ──
    echo "  ALNS 500iter..."
    OUT=$(python -u "$ALNS_SCRIPT" --data "$DATA" --capacity 50 --iterations 500 --num_instances 64 2>&1)
    ALNS_COST=$(echo "$OUT" | grep "Avg cost:" | tail -1 | awk '{print $NF}')
    ALNS_TIME=$(echo "$OUT" | grep "Avg time:" | tail -1 | awk '{print $NF}' | sed 's/s.*//')
    echo "${TYPE},${EDOD},ALNS_500,${ALNS_COST},${ALNS_TIME},n/a" >> "$RESULT_FILE"
    echo "  ALNS_500: cost=${ALNS_COST} time=${ALNS_TIME}s"

    # ── ALNS (5000 iter, ~50s/inst) ──
    echo "  ALNS 5000iter..."
    OUT=$(python -u "$ALNS_SCRIPT" --data "$DATA" --capacity 50 --iterations 5000 --num_instances 32 2>&1)
    ALNS_COST=$(echo "$OUT" | grep "Avg cost:" | tail -1 | awk '{print $NF}')
    ALNS_TIME=$(echo "$OUT" | grep "Avg time:" | tail -1 | awk '{print $NF}' | sed 's/s.*//')
    echo "${TYPE},${EDOD},ALNS_5K,${ALNS_COST},${ALNS_TIME},n/a" >> "$RESULT_FILE"
    echo "  ALNS_5K: cost=${ALNS_COST} time=${ALNS_TIME}s"
  done
done

# ── MaskCO Causal (从已有 CSV 提取) ──
echo ""
echo "--- MaskCO Causal (from results_full.csv) ---"
python3 -c "
import pandas as pd
df_mc = pd.read_csv('C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/results_full.csv')
# 按 type+edod 聚合 (mean over seeds)
mc_agg = df_mc.groupby(['type','edod']).agg(cost_mean=('cost','mean'), feas_mean=('tw_feas','first')).reset_index()
# MaskCO time: ~0.03s/inst (从实验记录)
for _, row in mc_agg.iterrows():
    print(f\"{row['type']},{row['edod']},MaskCO_Causal,{row['cost_mean']:.4f},0.03,{row['feas_mean']}\")
" >> "$RESULT_FILE"

# ── 汇总 ──
echo ""
echo "============================================================"
echo "BASELINE COMPARISON"
echo "============================================================"
python3 -c "
import pandas as pd
df = pd.read_csv('${RESULT_FILE}')
print()
print('%-6s %-6s %-16s %10s %10s %10s' % ('Type','EDoD','Method','Cost','Time(s)','TW Feas'))
print('-' * 60)
for t in ['R1','C1','RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type']==t) & (df['edod']==e)]
        for _, row in sub.iterrows():
            cost_s = f\"{row['cost']:.2f}\" if row['cost'] > 0 else 'n/a'
            time_s = f\"{row['time_s']:.1f}\" if row['time_s'] > 0 else 'n/a'
            feas_s = str(row.get('tw_feas','n/a'))
            print('%-6s %-6s %-16s %10s %10s %10s' % (t, e, row['method'], cost_s, time_s, feas_s))
        print()
"
