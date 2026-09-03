#!/bin/bash
# ============================================================
# 方向 C (简单版): 自适应破坏-修复 — keep_rate × EDoD sweep
# ============================================================
# 目标：对 greedy mask-reconstruct 路径，按 EDoD 级别统计最优 keep_rate，
#       验证「按 EDoD 差异化 keep_rate 是否优于统一 0.3」。
#
# 背景：greedy 路径已被 Phase 3c beam 取代，但作为「模型 vs 解码器」
#       消融的对照，仍需报告 greedy 的最优配置。
#
# 用法:
#   bash run_keep_rate_sweep.sh smoke   # 快速 (每格 runs=2 cycles=20)
#   bash run_keep_rate_sweep.sh full    # 完整 (每格 runs=4 cycles=40)
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

MODE="${1:-smoke}"

CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/causal_v1/step50000.ckpt"
DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
RESULT_CSV="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/keep_rate_sweep.csv"

if [ "$MODE" = "full" ]; then
    KEEP_RATES=(0.1 0.2 0.3 0.4 0.5)
    RUNS=4; CYCLES=40
else
    KEEP_RATES=(0.15 0.3 0.5)
    RUNS=2; CYCLES=20
fi

TYPES=(R1 C1 RC1)
EDODS=(0.2 0.5 0.8)

mkdir -p "$(dirname "$RESULT_CSV")"

echo "============================================================"
echo "方向 C 简单版: keep_rate × EDoD sweep"
echo "============================================================"
echo "Mode: $MODE | keep_rates: ${KEEP_RATES[*]}"
echo "CKPT: $CKPT"
echo ""

echo "type,edod,keep_rate,feas,cost" > "$RESULT_CSV"

for TYPE in "${TYPES[@]}"; do
  for EDOD in "${EDODS[@]}"; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"
    [ ! -f "$DATA" ] && { echo "  [SKIP] missing: $DATA"; continue; }

    for KR in "${KEEP_RATES[@]}"; do
        echo -n "  ${TYPE} EDoD=${EDOD} keep_rate=${KR} ... "
        OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -u "$DECODE" \
            --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
            --keep_rate $KR --two_opt_steps 4 --batch_size 8 \
            --runs $RUNS --cycles $CYCLES --sampling_steps 2 \
            --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
            --threads_over_batches 1 \
            --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1)
        FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
        COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
        echo "${TYPE},${EDOD},${KR},${FEAS},${COST}" >> "$RESULT_CSV"
        echo "feas=${FEAS} cost=${COST}"
    done
  done
done

# ============================================================
# 分组统计: 每组 (type, EDoD) 的最优 keep_rate
# ============================================================
echo ""
echo "============================================================"
echo "最优 keep_rate 统计"
echo "============================================================"

python3 << PYEOF
import pandas as pd

df = pd.read_csv('${RESULT_CSV}')
df['feas_num'] = df['feas'].str.rstrip('%').astype(float)

print()
print("--- 每组 (type, EDoD) 的最优 keep_rate ---")
print(f"{'Type':<6} {'EDoD':<6} {'best_kr':<8} {'feas':>8} {'cost':>8}  {'统一0.3 feas':>12}")
print('-' * 60)

summary = []
for t in ['R1', 'C1', 'RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type'] == t) & (df['edod'] == e)]
        if len(sub) == 0:
            continue
        # 最优 keep_rate: 先最大化 feas, 再最小化 cost
        best = sub.sort_values(['feas_num', 'cost'], ascending=[False, True]).iloc[0]
        kr_default = sub[sub['keep_rate'] == 0.3]
        feas_default = kr_default['feas_num'].values[0] if len(kr_default) else float('nan')
        print(f"{t:<6} {e:<6} {best['keep_rate']:<8} {best['feas_num']:>6.1f}% {best['cost']:>8.2f}  {feas_default:>10.1f}%")
        summary.append({'type': t, 'edod': e, 'best_kr': best['keep_rate'],
                        'best_feas': best['feas_num'], 'feas_0.3': feas_default})

print()
print("--- 结论 ---")
s = pd.DataFrame(summary)
# 按 EDoD 聚合: 每组平均最优 keep_rate
print("按 EDoD 聚合的最优 keep_rate（均值）:")
for e in [0.2, 0.5, 0.8]:
    sub = s[s['edod'] == e]
    print(f"  EDoD={e}: best_kr 均值 = {sub['best_kr'].mean():.2f}, "
          f"best feas = {sub['best_feas'].mean():.1f}%, "
          f"统一0.3 feas = {sub['feas_0.3'].mean():.1f}%")

# 自适应 vs 统一 的收益
gain = (s['best_feas'] - s['feas_0.3']).mean()
print()
if gain > 1.0:
    print(f"  自适应 keep_rate 平均提升 feas +{gain:.1f}pp（vs 统一 0.3）")
elif gain > 0:
    print(f"  自适应 keep_rate 收益微弱 +{gain:.1f}pp — 统一 0.3 已接近最优")
else:
    print(f"  自适应无收益 ({gain:.1f}pp) — 统一 0.3 即最优，方向 C 可终止")
PYEOF

echo ""
echo "Results: $RESULT_CSV"
