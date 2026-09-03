#!/bin/bash
# Week 1 v6: Beam Search + Constraint Cutting
# 束搜索 + 约束切割实验脚本
#
# 实验设计：
# - Baseline: 当前 Resource Beam (无约束切割)
# - +Fast Heuristic: 快速可达性检查
# - 对比 cost / time / feasibility / violations
#
# 运行：bash scripts/run_week1_constraint_cutting.sh [smoke|full]

MODE=${1:-smoke}

# GPU 设置
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# 路径
BASE="C-VRP_Cold-chainVehicleRoutingProblem"
CKPT="${BASE}/ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt"
LOGDIR="${BASE}/logs/week1_constraint_cutting"
mkdir -p "$LOGDIR"

# 数据配置
if [ "$MODE" == "smoke" ]; then
    TYPES=("r1")
    EDODS=("0.5")
    RUNS=1
    CYCLES=5
    echo "=== Week 1 Smoke Test (1 type × 1 EDoD × 1 run × 5 cycles) ==="
else
    TYPES=("r1" "c1" "rc1")
    EDODS=("0.2" "0.5" "0.8")
    RUNS=8
    CYCLES=40
    echo "=== Week 1 Full Experiment (3 types × 3 EDoDs × 8 runs × 40 cycles) ==="
fi

# Experiment configurations
CONFIGS=(
    "baseline:--enable_resource_decoder --beam_width 16"
    "fast_heuristic:--enable_resource_decoder --beam_width 16 --enable_constraint_cutting --constraint_mode fast"
    "fast_top32:--enable_resource_decoder --beam_width 16 --enable_constraint_cutting --constraint_mode fast --constraint_check_top_k 32"
)

echo "Checkpoint: $CKPT"
echo "Configurations: ${#CONFIGS[@]}"
echo "Start time: $(date)"
echo ""

# Main loop
for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        # 数据文件名格式：edod05 而非 edod0.5（没有小数点）
        edod_filename=$(echo "$edod" | sed 's/\.//g')  # 0.5 -> 05
        DATA="${BASE}/data/p0_fix/dcc_50_${type}_edod${edod_filename}_test.npz"

        if [ ! -f "$DATA" ]; then
            echo "⚠️  Data not found: $DATA, skipping..."
            continue
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Type=${type^^} EDoD=${edod}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            FLAGS="${config#*:}"

            OUTLOG="${LOGDIR}/${type}_edod${edod}_${NAME}.log"

            echo ""
            echo "▶ Running: $NAME"
            echo "  Flags: $FLAGS"

            # Run decoding
            python -u "${BASE}/scripts/decoding/cvrptw.py" \
                --capacity 50 --penalty 3.0 \
                --data "$DATA" --ckpt "$CKPT" \
                --keep_rate 0.3 --batch_size 8 \
                --runs $RUNS --cycles $CYCLES --sampling_steps 1 \
                --two_opt_steps 4 --seed 42 \
                --gumbel_scale_factor 0.0 \
                --threads_over_batches 1 \
                --enable_tw_aware_2opt_py \
                $FLAGS \
                > "$OUTLOG" 2>&1

            # Extract key metrics
            if [ -f "$OUTLOG" ]; then
                COST=$(grep -oP 'Final.*cost.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'violations.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                TIME=$(grep -oP 'Total time.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                echo "  ✓ Cost: ${COST:-N/A}"
                echo "  ✓ Feas: ${FEAS:-N/A}"
                echo "  ✓ Viol: ${VIOL:-N/A}"
                echo "  ✓ Time: ${TIME:-N/A}s"

                # Constraint checker statistics (if available)
                if grep -q "Constraint checker stats" "$OUTLOG"; then
                    echo "  📊 Constraint Checker Statistics:"
                    grep -A 10 "Constraint checker stats" "$OUTLOG" | head -8 | sed 's/^/    /'
                fi
            else
                echo "  ❌ Log file not found"
            fi
        done
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Week 1 Experiment Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Logs saved to: $LOGDIR"
echo "End time: $(date)"
echo ""

# Generate summary table
SUMMARY="${LOGDIR}/结果汇总表.md"
echo "# Week 1 Constraint Cutting Summary" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Generated: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| Type | EDoD | Config | Cost | Feas | Viol | Time (s) |" >> "$SUMMARY"
echo "|------|------|--------|------|------|------|----------|" >> "$SUMMARY"

for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            OUTLOG="${LOGDIR}/${type}_edod${edod}_${NAME}.log"

            if [ -f "$OUTLOG" ]; then
                COST=$(grep -oP 'Final.*cost.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'violations.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                TIME=$(grep -oP 'Total time.*\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                echo "| ${type^^} | $edod | $NAME | ${COST:-N/A} | ${FEAS:-N/A} | ${VIOL:-N/A} | ${TIME:-N/A} |" >> "$SUMMARY"
            fi
        done
    done
done

echo "" >> "$SUMMARY"
echo "## Notes" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "- **baseline**: Current Resource Beam (no constraint cutting)" >> "$SUMMARY"
echo "- **fast_heuristic**: Fast heuristic extendability check (all candidates)" >> "$SUMMARY"
echo "- **fast_top32**: Fast heuristic check (top-32 candidates only)" >> "$SUMMARY"

echo "📄 Summary table: $SUMMARY"
cat "$SUMMARY"
