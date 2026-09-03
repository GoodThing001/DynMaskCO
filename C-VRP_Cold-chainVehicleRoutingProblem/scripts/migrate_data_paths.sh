#!/bin/bash
# 数据路径迁移脚本：p0_fix → baseline/experimental
# 日期：2026-08-23
# 用途：批量更新所有脚本中的旧数据路径

set -e

PROJECT_ROOT="/d/PyCharm_/MaskCO-main/C-VRP_Cold-chainVehicleRoutingProblem"
cd "$PROJECT_ROOT"

echo "=== Phase 0.0: 数据路径迁移 ==="
echo ""

# 备份列表
BACKUP_DIR="scripts/.backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 需要更新的文件列表
FILES_TO_UPDATE=(
    "scripts/analysis/constraint_ablation.py"
    "scripts/analysis/edge_influence.py"
    "scripts/run_baseline_audit.sh"
    "scripts/run_c2_sensitivity.sh"
    "scripts/run_coldchain_thermal_eval.sh"
    "scripts/run_constraint_ablation.sh"
    "scripts/run_core_ablation.sh"
    "scripts/run_cross_scale.sh"
    "scripts/run_lambda_q_sweep.sh"
    "scripts/run_method_freeze.sh"
    "scripts/run_p0_verification.sh"
    "scripts/run_transfer.sh"
    "scripts/run_typed_retrain.sh"
)

echo "【第一步】备份原始文件..."
for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  ✓ 备份: $file"
    fi
done
echo ""

echo "【第二步】执行路径替换..."
echo ""

# 替换规则映射
declare -A PATH_MAP=(
    # 训练数据: p0_fix → baseline/50_node/train
    ["data/p0_fix/dcc_50_r1_edod05_train.npz"]="data/baseline/50_node/train/dcc_50_r1_edod05_train.npz"
    ["data/p0_fix/dcc_50_mixed_edod_train.npz"]="data/experimental/transfer/dcc_50_mixed_edod_train.npz"

    # 测试数据: p0_fix → baseline/50_node/test
    ["data/p0_fix/dcc_50_r1_edod05_test.npz"]="data/baseline/50_node/test/dcc_50_r1_edod05_test.npz"
    ["data/p0_fix/dcc_50_c1_edod05_test.npz"]="data/baseline/50_node/test/dcc_50_c1_edod05_test.npz"
    ["data/p0_fix/dcc_50_rc1_edod05_test.npz"]="data/baseline/50_node/test/dcc_50_rc1_edod05_test.npz"

    # 100-node 数据
    ["data/p0_fix/dcc_100_r1_edod05_test.npz"]="data/baseline/100_node/test/dcc_100_r1_edod05_test.npz"

    # 消融实验数据: p0_fix → experimental/ablation
    ["data/p0_fix/dcc_50_r1_edod05_test_tempswap.npz"]="data/experimental/ablation/dcc_50_r1_edod05_test_tempswap.npz"
    ["data/p0_fix/dcc_50_r1_edod05_test_twrelax.npz"]="data/experimental/ablation/dcc_50_r1_edod05_test_twrelax.npz"

    # 数据目录
    ["data/p0_fix"]="data/baseline/50_node/test"

    # Checkpoint 路径（保持 p0_fix，因为这是历史 checkpoint 位置）
    # ckpts/p0_fix/ 不修改（历史checkpoint路径保持不变）

    # Log 路径（保持 p0_fix）
    # logs/p0_fix/ 不修改（历史log路径保持不变）
)

# 执行替换
for file in "${FILES_TO_UPDATE[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ⚠ 文件不存在，跳过: $file"
        continue
    fi

    echo "处理: $file"

    for old_path in "${!PATH_MAP[@]}"; do
        new_path="${PATH_MAP[$old_path]}"

        # 只替换 data/ 路径，不替换 ckpts/ 和 logs/
        if [[ "$old_path" == data/* ]]; then
            if grep -q "$old_path" "$file"; then
                sed -i "s|$old_path|$new_path|g" "$file"
                echo "  ✓ 替换: $old_path → $new_path"
            fi
        fi
    done

    echo ""
done

echo "【第三步】验证替换结果..."
echo ""

# 检查是否还有遗漏的 p0_fix data 路径
REMAINING=$(grep -RIn "data/p0_fix" scripts --include="*.py" --include="*.sh" | grep -v "ckpts/p0_fix" | grep -v "logs/p0_fix" | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ 所有 data/p0_fix 路径已更新完毕"
else
    echo "⚠️  仍有 $REMAINING 处 data/p0_fix 引用，请手动检查："
    grep -RIn "data/p0_fix" scripts --include="*.py" --include="*.sh" | grep -v "ckpts/p0_fix" | grep -v "logs/p0_fix"
fi

echo ""
echo "【备份位置】$BACKUP_DIR"
echo ""
echo "=== 路径迁移完成 ==="
