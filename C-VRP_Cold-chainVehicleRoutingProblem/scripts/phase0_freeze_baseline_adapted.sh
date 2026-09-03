#!/bin/bash
# Phase 0: typed_edge Baseline 冻结评估（适配服务器版本）
# 基于 run_typed_retrain.sh 的实际评估逻辑
# 日期：2026-08-23

set -e

PROJECT_ROOT="$PWD"
echo "=== Phase 0: typed_edge Baseline 冻结评估 ==="
echo "工作目录: $PROJECT_ROOT"
echo ""

# ============================================================
# 配置（与 run_typed_retrain.sh 保持一致）
# ============================================================
NUM_NODES=50
CAPACITY=50
BEAM_WIDTH=16
KEEP_RATE=0.9
TWO_OPT_STEPS=100
DECODE_RUNS=128
DECODE_CYCLES=1
SAMPLING_STEPS=5
AUGMENT_LEVEL=0  # 修正：设置为 0（DynamicAugment 有 bug）
GUMBEL_SCALE=5.0
EVAL_SEED=42
BATCH_SIZE=128  # 优化：增大 batch size（从 8 → 128，单次评估从 36min → ~3min）

# 脚本路径
DECODE_SCRIPT="scripts/decoding/cvrptw.py"

# Checkpoint 和数据路径
CKPT_DIR="ckpts/p0_fix/typed_v1_edge"
DATA_DIR="data/baseline/50_node"
NUM_STEPS=50000

# 评估配置
TRAIN_SEEDS=(42 123 999 2025 2026)
TEST_TYPES=(R1 C1 RC1)
TEST_EDODS=(0.2 0.5 0.8)

# 输出目录
OUTPUT_DIR="results/phase0_baseline_freeze"
mkdir -p "$OUTPUT_DIR"

# 结果文件
RESULT_CSV="$OUTPUT_DIR/matrix_summary.csv"
FROZEN_JSON="$OUTPUT_DIR/baseline_frozen.json"

echo "【配置】"
echo "  模型: typed_v1_edge (边特征版本)"
echo "  Checkpoint: $CKPT_DIR/phase3c/seed*/step${NUM_STEPS}.ckpt"
echo "  测试数据: $DATA_DIR/test/"
echo "  Train seeds: ${TRAIN_SEEDS[@]}"
echo "  Beam K: $BEAM_WIDTH"
echo "  Eval seed: $EVAL_SEED"
echo ""

# ============================================================
# 预检
# ============================================================
echo "【预检】"

# 检查 decode 脚本
if [ ! -f "$DECODE_SCRIPT" ]; then
    echo "❌ Decode 脚本不存在: $DECODE_SCRIPT"
    exit 1
fi
echo "✓ Decode 脚本: $DECODE_SCRIPT"

# 检查 checkpoint
MISSING_CKPTS=0
for SEED in "${TRAIN_SEEDS[@]}"; do
    CK="$CKPT_DIR/phase3c/seed${SEED}/step${NUM_STEPS}.ckpt"
    if [ ! -f "$CK" ]; then
        echo "⚠  Checkpoint 不存在: $CK"
        MISSING_CKPTS=$((MISSING_CKPTS + 1))
    fi
done

if [ $MISSING_CKPTS -gt 0 ]; then
    echo "❌ 缺少 $MISSING_CKPTS 个 checkpoint"
    exit 1
fi
echo "✓ 所有 checkpoint 存在 (5 seeds)"

# 检查测试数据
TEST_FILES=$(find "$DATA_DIR/test" -name "*.npz" | wc -l)
if [ "$TEST_FILES" -ne 9 ]; then
    echo "❌ 测试数据不完整: 预期 9 个，实际 $TEST_FILES 个"
    exit 1
fi
echo "✓ 测试数据完整 (9 files)"

echo ""

# ============================================================
# 9-cell × 5-seed 评估
# ============================================================
echo "【开始评估】9-cell × 5-seed = 45 runs"
echo ""

# CSV 表头
echo "type,edod,train_seed,cost,tw_feas,tw_viol,num_vehicles,complete_rate" > "$RESULT_CSV"

BEAM_FLAGS="--enable_resource_decoder --beam_width $BEAM_WIDTH"
TOTAL_RUNS=$((${#TEST_TYPES[@]} * ${#TEST_EDODS[@]} * ${#TRAIN_SEEDS[@]}))
CURRENT_RUN=0

for TSEED in "${TRAIN_SEEDS[@]}"; do
    CK="$CKPT_DIR/phase3c/seed${TSEED}/step${NUM_STEPS}.ckpt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "【Train Seed $TSEED】"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for TYPE in "${TEST_TYPES[@]}"; do
      for EDOD in "${TEST_EDODS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # 构造数据文件名
        ESTR=$(echo "$EDOD" | sed 's/\.//')
        DATA="$DATA_DIR/test/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"

        if [ ! -f "$DATA" ]; then
            echo "  [$CURRENT_RUN/$TOTAL_RUNS] ${TYPE} EDoD=${EDOD} ... ⚠  数据文件不存在"
            continue
        fi

        echo -n "  [$CURRENT_RUN/$TOTAL_RUNS] ${TYPE} EDoD=${EDOD} ... "

        # 运行评估（使用 GPU 0，优化 batch_size）
        OUT=$(CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -u "$DECODE_SCRIPT" \
            --capacity $CAPACITY --penalty 3. --data "$DATA" --ckpt "$CK" \
            --keep_rate $KEEP_RATE --two_opt_steps $TWO_OPT_STEPS \
            --batch_size $BATCH_SIZE --runs $DECODE_RUNS --cycles $DECODE_CYCLES \
            --sampling_steps $SAMPLING_STEPS --augment_level $AUGMENT_LEVEL \
            --gumbel_scale_factor $GUMBEL_SCALE --seed $EVAL_SEED \
            --threads_over_batches 1 \
            --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
            $BEAM_FLAGS 2>&1)

        # 提取指标
        COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
        FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
        VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
        NUM_VEH=$(echo "$OUT" | grep "num vehicles:" | tail -1 | awk '{print $NF}' || echo "N/A")
        COMPLETE=$(echo "$OUT" | grep "complete rate:" | tail -1 | awk '{print $NF}' || echo "1.0")

        # 写入 CSV
        echo "${TYPE},${EDOD},${TSEED},${COST},${FEAS},${VIOL},${NUM_VEH},${COMPLETE}" >> "$RESULT_CSV"

        # 控制台输出
        echo "cost=${COST} feas=${FEAS}"
      done
    done
    echo ""
done

# ============================================================
# 统计摘要
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【评估完成】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 生成统计摘要
python3 - << 'PYEOF'
import pandas as pd
import numpy as np
import sys

try:
    df = pd.read_csv("$RESULT_CSV")
except:
    print("⚠  无法读取结果文件")
    sys.exit(0)

if len(df) > 0:
    print("【统计摘要】")
    print("")

    # 按 type 分组
    print("按问题类型分组:")
    grouped = df.groupby('type')['cost'].agg(['mean', 'std', 'count'])
    print(grouped)
    print("")

    # 全矩阵均值
    overall_mean = df['cost'].mean()
    overall_std = df['cost'].std()
    print(f"全矩阵均值: {overall_mean:.2f} ± {overall_std:.2f}")
    print("")

    # 可行性
    tw_feas = df['tw_feas'].mean()
    complete = df['complete_rate'].mean()
    print(f"TW 可行率: {tw_feas*100:.1f}%")
    print(f"Complete 率: {complete*100:.1f}%")
    print("")

    # R1/C1/RC1 × EDoD=0.5 的精确数字
    print("关键指标 (EDoD=0.5):")
    for ptype in ['R1', 'C1', 'RC1']:
        subset = df[(df['type'] == ptype) & (df['edod'] == 0.5)]
        if len(subset) > 0:
            mean_cost = subset['cost'].mean()
            std_cost = subset['cost'].std()
            print(f"  {ptype}: {mean_cost:.2f} ± {std_cost:.2f}")

    # 5-seed 稳定性检查
    print("")
    print("5-seed 稳定性检查:")
    seed_stability = df.groupby(['type', 'edod'])['cost'].std().max()
    print(f"  最大 std: {seed_stability:.3f}")
    if seed_stability <= 0.15:
        print("  ✓ 稳定性良好 (std ≤ 0.15)")
    else:
        print(f"  ⚠  稳定性不足 (std > 0.15)")
else:
    print("⚠  无评估数据")
PYEOF

# ============================================================
# 生成元数据
# ============================================================
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -Iseconds)

cat > "$FROZEN_JSON" <<JSONEOF
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
    "keep_rate": 0.9,
    "two_opt_steps": 100,
    "decode_runs": 128
  },
  "checkpoint": {
    "base_dir": "ckpts/p0_fix/typed_v1_edge/phase3c",
    "format": "seed{SEED}/step50000.ckpt",
    "num_steps": 50000
  },
  "results": {
    "summary_csv": "$RESULT_CSV",
    "output_dir": "$OUTPUT_DIR"
  }
}
JSONEOF

echo ""
echo "【输出文件】"
echo "  汇总表: $RESULT_CSV"
echo "  元数据: $FROZEN_JSON"
echo ""
echo "=== Phase 0 评估完成 ==="
