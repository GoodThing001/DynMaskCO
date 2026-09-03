#!/bin/bash
# ============================================================
# P0-2: OR-Tools-RH 时间预算扫描 — cost-latency Pareto
# ============================================================
# 公平对比的核心：把 OR-Tools-RH 的每事件预算扫到与 DynMaskCO
# 相当（DynMaskCO beam ≈ 1.5s/实例），看同预算下谁在 Pareto front。
#
# 用法:
#   bash run_ortools_rh_budget_sweep.sh          # 完整 sweep (5 budgets)
#   bash run_ortools_rh_budget_sweep.sh smoke    # 快速 (3 budgets, 8 inst)
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate

MODE="${1:-full}"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz"
SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/baselines/ortools_rolling_horizon.py"
RESULT_CSV="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/ortools_rh_budget_sweep.csv"

if [ "$MODE" = "smoke" ]; then
    BUDGETS=(100 250 500)
    NUM_INST=8
else
    BUDGETS=(50 100 250 500 1000)
    NUM_INST=16
fi

mkdir -p "$(dirname "$RESULT_CSV")"

echo "============================================================"
echo "OR-Tools-RH 时间预算扫描"
echo "============================================================"
echo "Data: $DATA | instances=$NUM_INST"
echo ""

echo "budget_ms,avg_cost,feas_rate,avg_latency_ms" > "$RESULT_CSV"

for B in "${BUDGETS[@]}"; do
    echo -n "  budget=${B}ms ... "
    OUT=$(python -u "$SCRIPT" \
        --data "$DATA" --capacity 50 --num_vehicles 25 \
        --time_limit_ms "$B" --num_instances "$NUM_INST" 2>&1)

    COST=$(echo "$OUT" | grep "Avg cost:" | tail -1 | awk '{print $NF}' | sed 's/[^0-9.]//g')
    FEAS=$(echo "$OUT" | grep "Feasible:" | tail -1 | awk '{print $2}')
    LATENCY=$(echo "$OUT" | grep "Avg latency:" | tail -1 | awk '{print $NF}' | sed 's/[^0-9.]//g')

    echo "${B},${COST},${FEAS},${LATENCY}" >> "$RESULT_CSV"
    echo "cost=${COST} feas=${FEAS} latency=${LATENCY}ms"
done

# ============================================================
# Pareto 汇总表
# ============================================================
echo ""
echo "============================================================"
echo "Pareto 汇总（vs DynMaskCO beam）"
echo "============================================================"

python3 << PYEOF
import pandas as pd

df = pd.read_csv('${RESULT_CSV}')
df['budget_ms'] = df['budget_ms'].astype(int)
df['avg_cost'] = df['avg_cost'].astype(float)
df['avg_latency_ms'] = df['avg_latency_ms'].astype(float)

# 解析 feas_rate "16/16" → 比例
df['feas_ratio'] = df['feas_rate'].apply(lambda s: eval(s) if isinstance(s, str) and '/' in s else 1.0)

print()
print(f"{'budget/event':<14} {'avg_cost':>10} {'feas':>8} {'total_ms':>10}  {'vs DynMaskCO 14.77':>18}")
print('-' * 66)
for _, r in df.iterrows():
    delta = (r['avg_cost'] - 14.77) / 14.77 * 100
    print(f"{str(r['budget_ms'])+'ms':<14} {r['avg_cost']:>10.2f} {r['feas_ratio']*100:>7.0f}% {r['avg_latency_ms']:>10.0f}  {delta:>+17.1f}%")

print()
print("DynMaskCO beam 参考: cost=14.77, feas=100%, ~1.5s/实例 (约 1500ms total)")
print()
print("--- 结论 ---")
# 找与 DynMaskCO 时间相当（total ≈ 1500ms）的 budget
closest = df.iloc[(df['avg_latency_ms'] - 1500).abs().argsort()[0]]
print(f"  与 DynMaskCO 时间相当的 budget={closest['budget_ms']}ms/event "
      f"(total={closest['avg_latency_ms']:.0f}ms): cost={closest['avg_cost']:.2f}")
if closest['avg_cost'] < 14.77:
    print(f"  → OR-Tools-RH 在同预算下更优 (低 {14.77 - closest['avg_cost']:.2f})")
    print(f"  ⚠️ 需诚实面对：重新定位为低延迟可行构造，或分析优势区间")
else:
    print(f"  → DynMaskCO 在同预算下更优 (低 {closest['avg_cost'] - 14.77:.2f})")
PYEOF

echo ""
echo "Results: $RESULT_CSV"
