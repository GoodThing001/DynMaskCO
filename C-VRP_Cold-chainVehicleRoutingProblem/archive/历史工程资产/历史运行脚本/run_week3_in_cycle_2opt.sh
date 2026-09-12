#!/bin/bash
# Week 3 v6: In-Cycle Local Search 实验脚本
# 在 mask-reconstruct 过程中插入 2-opt，优化成本
#
# 运行：bash scripts/run_week3_in_cycle_2opt.sh [smoke|sweep|best]

MODE=${1:-smoke}

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

BASE="C-VRP_Cold-chainVehicleRoutingProblem"
CKPT="${BASE}/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"
LOGDIR="${BASE}/logs/week3_in_cycle_2opt"
mkdir -p "$LOGDIR"

# 检查是否在正确的环境中
if ! python -c "import jax" 2>/dev/null; then
    echo "❌ 错误：JAX 未安装或 conda 环境未激活"
    echo "请先运行：conda activate MASKCO_env"
    exit 1
fi

echo "============================================================"
echo "Week 3 v6: In-Cycle Local Search"
echo "Checkpoint: $CKPT"
echo "============================================================"
echo ""

# 基准配置（共享参数）
COMMON_FLAGS="--capacity 50 --penalty 3.0 \
    --keep_rate 0.3 --batch_size 8 \
    --sampling_steps 1 --two_opt_steps 4 --seed 42 \
    --gumbel_scale_factor 0.0 --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 16 \
    --enable_tw_repair_edd --enable_tw_aware_2opt_py"

if [ "$MODE" == "smoke" ]; then
    echo "=== Smoke Test (1 type × 1 EDoD × 2 configs) ==="
    TYPES=("r1")
    EDODS=("05")  # edod05
    RUNS=1
    CYCLES=5

    CONFIGS=(
        "baseline:$COMMON_FLAGS"
        "in_cycle_f5_s2_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 10"
    )

elif [ "$MODE" == "sweep" ]; then
    echo "=== Hyperparameter Sweep (1 type × 1 EDoD × 9 configs) ==="
    TYPES=("r1")
    EDODS=("05")
    RUNS=8
    CYCLES=40

    CONFIGS=(
        "baseline:$COMMON_FLAGS"
        # Frequency sweep (steps=2, start=10)
        "in_cycle_f3_s2_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 3 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 10"
        "in_cycle_f5_s2_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 10"
        "in_cycle_f10_s2_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 10 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 10"
        # Steps sweep (freq=5, start=10)
        "in_cycle_f5_s1_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 1 --in_cycle_2opt_start 10"
        "in_cycle_f5_s3_st10:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 3 --in_cycle_2opt_start 10"
        # Start sweep (freq=5, steps=2)
        "in_cycle_f5_s2_st0:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 0"
        "in_cycle_f5_s2_st5:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 5"
        "in_cycle_f5_s2_st15:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 15"
    )

elif [ "$MODE" == "best" ]; then
    echo "=== Best Config (3 types × 3 EDoDs × 2 configs) ==="
    TYPES=("r1" "c1" "rc1")
    EDODS=("02" "05" "08")
    RUNS=8
    CYCLES=40

    # 使用 sweep 阶段确定的最佳配置
    CONFIGS=(
        "baseline:$COMMON_FLAGS"
        "best_in_cycle:$COMMON_FLAGS --in_cycle_2opt --in_cycle_2opt_frequency 5 --in_cycle_2opt_steps 2 --in_cycle_2opt_start 10"
    )
else
    echo "❌ 未知模式：$MODE"
    echo "用法：bash run_week3_in_cycle_2opt.sh [smoke|sweep|best]"
    exit 1
fi

echo "Start time: $(date)"
echo ""

# Main loop
for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        DATA="${BASE}/data/p0_fix/dcc_50_${type}_edod${edod}_test.npz"

        if [ ! -f "$DATA" ]; then
            echo "⚠️  Data not found: $DATA, skipping..."
            continue
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Type=${type^^} EDoD=0.${edod}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            FLAGS="${config#*:}"

            OUTLOG="${LOGDIR}/${type}_edod${edod}_${NAME}.log"

            echo ""
            echo "▶ Running: $NAME"

            python -u "${BASE}/scripts/decoding/cvrptw.py" \
                --data "$DATA" --ckpt "$CKPT" \
                --runs $RUNS --cycles $CYCLES \
                $FLAGS \
                > "$OUTLOG" 2>&1

            # Extract key metrics
            if [ -f "$OUTLOG" ]; then
                COST=$(grep -oP 'mean cost:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas rate:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'Avg TW viol/inst:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                echo "  ✓ Cost: ${COST:-N/A}"
                echo "  ✓ Feas: ${FEAS:-N/A}%"
                echo "  ✓ Viol: ${VIOL:-N/A}"
            else
                echo "  ❌ Log file not found"
            fi
        done
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Week 3 Experiment Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Logs: $LOGDIR"
echo "End time: $(date)"

# Generate summary table
SUMMARY="${LOGDIR}/summary_${MODE}.md"
echo "# Week 3 In-Cycle 2-opt Summary ($MODE)" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Generated: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| Type | EDoD | Config | Cost | Δ vs Baseline | Feas (%) | Viol |" >> "$SUMMARY"
echo "|------|------|--------|------|--------------|----------|------|" >> "$SUMMARY"

# Compute baseline costs for comparison
declare -A BASELINE_COSTS

for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        BASELINE_LOG="${LOGDIR}/${type}_edod${edod}_baseline.log"
        if [ -f "$BASELINE_LOG" ]; then
            COST=$(grep -oP 'mean cost:.*?\K[0-9]+\.[0-9]+' "$BASELINE_LOG" | tail -1)
            BASELINE_COSTS["${type}_${edod}"]="$COST"
        fi
    done
done

for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            OUTLOG="${LOGDIR}/${type}_edod${edod}_${NAME}.log"

            if [ -f "$OUTLOG" ]; then
                COST=$(grep -oP 'mean cost:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas rate:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'Avg TW viol/inst:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                # Calculate delta
                BASELINE="${BASELINE_COSTS[${type}_${edod}]}"
                if [ -n "$COST" ] && [ -n "$BASELINE" ] && [ "$NAME" != "baseline" ]; then
                    DELTA=$(awk "BEGIN {printf \"%.1f\", ($COST - $BASELINE) / $BASELINE * 100}")
                    DELTA_STR="${DELTA}%"
                else
                    DELTA_STR="—"
                fi

                echo "| ${type^^} | 0.${edod} | $NAME | ${COST:-N/A} | $DELTA_STR | ${FEAS:-N/A} | ${VIOL:-N/A} |" >> "$SUMMARY"
            fi
        done
    done
done

echo "" >> "$SUMMARY"
echo "## Configuration Details" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "- **baseline**: Standard mask-reconstruct + final 2-opt (4 steps)" >> "$SUMMARY"
echo "- **in_cycle_fN_sM_stK**: In-cycle 2-opt every N cycles, M steps per cycle, starting from cycle K" >> "$SUMMARY"
echo "  - f=frequency, s=steps, st=start" >> "$SUMMARY"

echo ""
echo "📄 Summary: $SUMMARY"
cat "$SUMMARY"
