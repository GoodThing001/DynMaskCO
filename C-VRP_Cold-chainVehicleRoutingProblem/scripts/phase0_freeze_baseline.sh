#!/bin/bash
# Phase 0: 冻结 typed_edge Baseline 评估脚本
# 日期：2026-08-23
# 目标：完整 9-cell × 5-seed 评估，生成 baseline_frozen.json

set -e

PROJECT_ROOT="/d/PyCharm_/MaskCO-main/C-VRP_Cold-chainVehicleRoutingProblem"
cd "$PROJECT_ROOT"

echo "=== Phase 0: typed_edge Baseline 冻结评估 ==="
echo ""
echo "【评估配置】"
echo "  模型: typed_v1_edge (边特征版本)"
echo "  数据: data/baseline/50_node/test/ (9-cell)"
echo "  种子: [42, 123, 999, 2025, 2026]"
echo "  Decode seed: 42"
echo "  Beam K: 16"
echo "  Online K: 5"
echo ""

# 输出目录
OUTPUT_DIR="results/typed_v1_edge_phase0"
mkdir -p "$OUTPUT_DIR"

# 检查 checkpoint
CKPT_DIR="ckpts/p0_fix/typed_v1_edge/phase3c"
if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Checkpoint 目录不存在: $CKPT_DIR"
    exit 1
fi

# 检查数据完整性
echo "【数据完整性检查】"
TEST_DIR="data/baseline/50_node/test"
EXPECTED_FILES=9

ACTUAL_FILES=$(find "$TEST_DIR" -name "*.npz" | wc -l)
if [ "$ACTUAL_FILES" -ne "$EXPECTED_FILES" ]; then
    echo "❌ 测试数据不完整: 预期 $EXPECTED_FILES 个文件，实际 $ACTUAL_FILES 个"
    exit 1
fi

echo "✓ 测试数据完整: $ACTUAL_FILES 个文件"

# 验证 checksum
if [ -f "data/baseline/DATA_MANIFEST.sha256" ]; then
    echo "✓ Checksum 文件存在，验证中..."
    cd data/baseline
    sha256sum -c DATA_MANIFEST.sha256 --quiet && echo "✓ Checksum 验证通过" || echo "⚠️  Checksum 验证失败"
    cd "$PROJECT_ROOT"
fi

echo ""
echo "【开始 9-cell × 5-seed 评估】"
echo ""

# Problem types
TYPES=("r1" "c1" "rc1")
EDODS=("02" "05" "08")
SEEDS=(42 123 999 2025 2026)

# 结果汇总文件
SUMMARY_CSV="$OUTPUT_DIR/matrix_summary.csv"
echo "type,edod,seed,cost,tw_feas,cap_feas,complete,num_vehicles,wall_time_ms" > "$SUMMARY_CSV"

# 计数器
TOTAL_RUNS=$((3 * 3 * 5))
CURRENT_RUN=0

# 遍历 9-cell × 5-seed
for TYPE in "${TYPES[@]}"; do
    for EDOD in "${EDODS[@]}"; do
        DATA_FILE="$TEST_DIR/dcc_50_${TYPE}_edod${EDOD}_test.npz"

        if [ ! -f "$DATA_FILE" ]; then
            echo "⚠️  数据文件不存在，跳过: $DATA_FILE"
            continue
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "【${TYPE^^} EDoD=0.${EDOD}】"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for SEED in "${SEEDS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] Seed $SEED"

            # Checkpoint 路径
            CKPT="$CKPT_DIR/seed${SEED}/step50000.ckpt"

            if [ ! -f "$CKPT" ]; then
                echo "⚠️  Checkpoint 不存在: $CKPT"
                # 尝试找最新的 checkpoint
                CKPT=$(find "$CKPT_DIR" -name "*.ckpt" | head -1)
                if [ -z "$CKPT" ]; then
                    echo "❌ 无可用 checkpoint，跳过"
                    continue
                fi
                echo "  使用备用 checkpoint: $CKPT"
            fi

            # 输出文件
            OUT_JSON="$OUTPUT_DIR/${TYPE}_edod${EDOD}_seed${SEED}.json"

            # 运行评估（假设使用 run_typed_retrain.sh 的 eval 模式）
            # 注意：这里需要根据实际项目的评估脚本调整
            # 示例命令（需替换为实际命令）:

            # 方案 1: 如果有统一的评估脚本
            # python scripts/evaluate.py \
            #     --ckpt "$CKPT" \
            #     --data "$DATA_FILE" \
            #     --seed 42 \
            #     --beam_k 16 \
            #     --online_k 5 \
            #     --output "$OUT_JSON"

            # 方案 2: 使用现有的 run_typed_retrain.sh
            # 由于路径已更新，需要临时设置环境变量
            export USE_EDGE=1
            export EVAL_SEED=$SEED
            export EVAL_DATA="$DATA_FILE"
            export EVAL_CKPT="$CKPT"
            export EVAL_OUTPUT="$OUT_JSON"

            # 实际评估命令（这里需要根据项目实际情况调整）
            echo "  运行评估..."
            # bash scripts/run_typed_retrain.sh eval 2>&1 | tee "$OUTPUT_DIR/${TYPE}_edod${EDOD}_seed${SEED}.log"

            # 临时：由于评估脚本可能需要调整，这里先生成占位符
            echo "  ⚠️  评估命令待完善（需根据实际项目调整）"
            echo "  数据: $DATA_FILE"
            echo "  Checkpoint: $CKPT"
            echo "  输出: $OUT_JSON"

            # 提取结果到 CSV（假设 JSON 输出格式）
            # if [ -f "$OUT_JSON" ]; then
            #     COST=$(jq -r '.cost' "$OUT_JSON")
            #     TW_FEAS=$(jq -r '.tw_feasibility' "$OUT_JSON")
            #     CAP_FEAS=$(jq -r '.capacity_feasibility' "$OUT_JSON")
            #     COMPLETE=$(jq -r '.completion_rate' "$OUT_JSON")
            #     NUM_VEH=$(jq -r '.num_vehicles' "$OUT_JSON")
            #     WALL_TIME=$(jq -r '.wall_time_ms' "$OUT_JSON")
            #
            #     echo "$TYPE,0.$EDOD,$SEED,$COST,$TW_FEAS,$CAP_FEAS,$COMPLETE,$NUM_VEH,$WALL_TIME" >> "$SUMMARY_CSV"
            # fi
        done
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【评估完成】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 生成 baseline_frozen.json 元数据
FROZEN_JSON="$OUTPUT_DIR/baseline_frozen.json"
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -Iseconds)

cat > "$FROZEN_JSON" <<EOF
{
  "phase": "Phase 0",
  "model": "typed_v1_edge",
  "description": "typed_edge baseline with edge features (energy_mat)",
  "freeze_date": "$TIMESTAMP",
  "git_commit": "$GIT_COMMIT",
  "dataset": {
    "version": "v1.1_20260823",
    "manifest": "data/baseline/DATA_MANIFEST.sha256",
    "test_root": "data/baseline/50_node/test",
    "files": 9,
    "instances_per_file": 128
  },
  "evaluation": {
    "train_seeds": [42, 123, 999, 2025, 2026],
    "decode_seed": 42,
    "beam_k": 16,
    "online_k": 5,
    "anytime_budgets_ms": [50, 100, 250, 500, 1000]
  },
  "checkpoint": {
    "base_dir": "ckpts/p0_fix/typed_v1_edge/phase3c",
    "format": "seed{SEED}/step50000.ckpt"
  },
  "results": {
    "summary_csv": "$SUMMARY_CSV",
    "individual_results": "$OUTPUT_DIR/*.json"
  }
}
EOF

echo "【输出文件】"
echo "  汇总表: $SUMMARY_CSV"
echo "  元数据: $FROZEN_JSON"
echo "  详细结果: $OUTPUT_DIR/*.json"
echo ""

# 计算统计量（如果 CSV 有数据）
if [ -f "$SUMMARY_CSV" ] && [ $(wc -l < "$SUMMARY_CSV") -gt 1 ]; then
    echo "【统计摘要】"
    python3 - <<PYEOF
import pandas as pd
import numpy as np

df = pd.read_csv("$SUMMARY_CSV")

if len(df) > 0:
    # 按 type 分组
    grouped = df.groupby('type')['cost'].agg(['mean', 'std', 'min', 'max'])
    print(grouped)

    # 全矩阵均值
    overall_mean = df['cost'].mean()
    overall_std = df['cost'].std()
    print(f"\n全矩阵均值: {overall_mean:.2f} ± {overall_std:.2f}")

    # 可行性
    tw_feas = df['tw_feas'].mean()
    cap_feas = df['cap_feas'].mean()
    complete = df['complete'].mean()
    print(f"TW可行率: {tw_feas*100:.1f}%")
    print(f"Cap可行率: {cap_feas*100:.1f}%")
    print(f"Complete率: {complete*100:.1f}%")
else:
    print("⚠️  无评估数据")
PYEOF
fi

echo ""
echo "=== Phase 0 评估完成 ==="
echo ""
echo "⚠️  注意：当前脚本中的评估命令为占位符，需要根据实际项目调整。"
echo "    建议：检查 scripts/run_typed_retrain.sh 的 eval 模式，确保支持新的数据路径。"
