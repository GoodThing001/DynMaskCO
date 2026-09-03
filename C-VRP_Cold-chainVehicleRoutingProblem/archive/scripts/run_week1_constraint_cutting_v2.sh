#!/bin/bash
# Week 1 v6: 修复版实验脚本
# 根据服务器实际数据文件命名调整

MODE=${1:-smoke}

# 确保使用 conda 环境中的 Python
# 不要在脚本中 conda activate，而是要求用户在运行前激活
if ! python -c "import jax" 2>/dev/null; then
    echo "❌ 错误：JAX 未安装或 conda 环境未激活"
    echo "请先运行：conda activate MASKCO_env"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

BASE="C-VRP_Cold-chainVehicleRoutingProblem"
# 使用 typed_v1 checkpoint（之前验证过 100% feasible）
CKPT="${BASE}/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"
LOGDIR="${BASE}/logs/week1_constraint_cutting"
mkdir -p "$LOGDIR"

# 先检查数据文件实际格式
echo "=== 检查数据文件格式 ==="
SAMPLE_FILE=$(ls ${BASE}/data/p0_fix/dcc_50_r1_*_test.npz 2>/dev/null | head -1)
if [ -z "$SAMPLE_FILE" ]; then
    echo "❌ 错误：找不到数据文件"
    echo "请检查路径：${BASE}/data/p0_fix/"
    exit 1
fi
echo "找到样例文件：$SAMPLE_FILE"

# 从文件名推断格式（edod02, edod05, edod08）
if echo "$SAMPLE_FILE" | grep -qE "edod0[258]"; then
    EDOD_FORMAT="no_dot"  # 02, 05, 08 (两位数字)
    echo "数据格式：无小数点（edod02, edod05, edod08）"
elif echo "$SAMPLE_FILE" | grep -q "edod0\."; then
    EDOD_FORMAT="with_dot"  # 0.2, 0.5, 0.8
    echo "数据格式：有小数点（edod0.2, edod0.5）"
else
    echo "❌ 无法识别 EDoD 格式：$SAMPLE_FILE"
    exit 1
fi

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

# Experiment configurations (添加完整的可行性保证组件)
CONFIGS=(
    "baseline:--enable_resource_decoder --beam_width 16 --enable_tw_repair_edd --enable_tw_aware_2opt_py"
    "fast_heuristic:--enable_resource_decoder --beam_width 16 --enable_tw_repair_edd --enable_tw_aware_2opt_py --enable_constraint_cutting --constraint_mode fast"
    "fast_top32:--enable_resource_decoder --beam_width 16 --enable_tw_repair_edd --enable_tw_aware_2opt_py --enable_constraint_cutting --constraint_mode fast --constraint_check_top_k 32"
)

echo "Checkpoint: $CKPT"
echo "Start time: $(date)"
echo ""

# Main loop
for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        # 根据检测到的格式构建文件名
        if [ "$EDOD_FORMAT" == "no_dot" ]; then
            # 0.5 -> 05, 0.2 -> 02, 0.8 -> 08
            edod_file=$(echo "$edod" | sed 's/0\./0/g')
        else
            edod_file="$edod"  # 保持 0.5
        fi

        DATA="${BASE}/data/p0_fix/dcc_50_${type}_edod${edod_file}_test.npz"

        if [ ! -f "$DATA" ]; then
            echo "⚠️  Data not found: $DATA, skipping..."
            continue
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Type=${type^^} EDoD=${edod} (file: edod${edod_file})"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            FLAGS="${config#*:}"

            OUTLOG="${LOGDIR}/${type}_edod${edod_file}_${NAME}.log"

            echo ""
            echo "▶ Running: $NAME"

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
                COST=$(grep -oP 'mean cost:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas rate:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'Avg TW viol/inst:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                echo "  ✓ Cost: ${COST:-N/A}"
                echo "  ✓ Feas: ${FEAS:-N/A}%"
                echo "  ✓ Viol: ${VIOL:-N/A}"

                # Constraint checker statistics
                if grep -q "total_checks" "$OUTLOG"; then
                    echo "  📊 Constraint Checker:"
                    grep -E "(total_checks|success_rate|avg_time_ms)" "$OUTLOG" | sed 's/^/    /'
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
echo "📁 Logs: $LOGDIR"
echo "End time: $(date)"

# Generate summary
SUMMARY="${LOGDIR}/结果汇总表.md"
echo "# Week 1 Constraint Cutting Summary" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Generated: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| Type | EDoD | Config | Cost | Feas (%) | Viol | Time (s) |" >> "$SUMMARY"
echo "|------|------|--------|------|----------|------|----------|" >> "$SUMMARY"

for type in "${TYPES[@]}"; do
    for edod in "${EDODS[@]}"; do
        if [ "$EDOD_FORMAT" == "no_dot" ]; then
            edod_file=$(echo "$edod" | sed 's/\.//g')
        else
            edod_file="$edod"
        fi

        for config in "${CONFIGS[@]}"; do
            NAME="${config%%:*}"
            OUTLOG="${LOGDIR}/${type}_edod${edod_file}_${NAME}.log"

            if [ -f "$OUTLOG" ]; then
                COST=$(grep -oP 'mean cost:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                FEAS=$(grep -oP 'TW feas rate:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)
                VIOL=$(grep -oP 'Avg TW viol/inst:.*?\K[0-9]+\.[0-9]+' "$OUTLOG" | tail -1)

                echo "| ${type^^} | $edod | $NAME | ${COST:-N/A} | ${FEAS:-N/A} | ${VIOL:-N/A} | N/A |" >> "$SUMMARY"
            fi
        done
    done
done

echo "" >> "$SUMMARY"
echo "## Notes" >> "$SUMMARY"
echo "- **baseline**: Current Resource Beam (no constraint cutting)" >> "$SUMMARY"
echo "- **fast_heuristic**: Fast heuristic check (all candidates)" >> "$SUMMARY"
echo "- **fast_top32**: Fast heuristic check (top-32 only)" >> "$SUMMARY"

echo ""
echo "📄 Summary: $SUMMARY"
cat "$SUMMARY"
