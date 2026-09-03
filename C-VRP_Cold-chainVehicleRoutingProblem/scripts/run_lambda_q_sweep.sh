#!/bin/bash
# ============================================================
# run_lambda_q_sweep.sh — 品质感知权重 lambda_q 扫描
# ============================================================
# 验证「品质感知」的核心证据：提高 lambda_q → 降低 Num unsalable
# （用 typed_v1 已有 checkpoint，不依赖 typed_edge 训练结果）
#
# 用法:
#   bash run_lambda_q_sweep.sh
#
# 输出: lambda_q sweep 表格（cost / TW feas / 最终路线 Num / beam Num）
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=1   # 卡1（卡0 在跑 typed_edge 训练）

DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"
RESULT_CSV="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/typed_v1/results/lambda_q_sweep.csv"

LAMBDA_QS=(0.0 0.1 0.5 1.0 2.0)
THRESHOLD=0.02

mkdir -p "$(dirname "$RESULT_CSV")"
echo "lambda_q,cost,tw_feas,final_num,beam_num" > "$RESULT_CSV"

echo "============================================================"
echo "lambda_q sweep（品质感知权重）"
echo "  data: $DATA"
echo "  ckpt: $CKPT（typed_v1 seed42，无边特征）"
echo "  threshold: $THRESHOLD"
echo "============================================================"

for LQ in "${LAMBDA_QS[@]}"; do
    echo ""
    echo "=== lambda_q=$LQ ==="
    OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
        --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
        --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
        --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
        --threads_over_batches 1 \
        --enable_resource_decoder --beam_width 16 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
        --enable_quality --lambda_q $LQ --quality_salable_threshold $THRESHOLD 2>&1)
    COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
    FINAL_NUM=$(echo "$OUT" | grep "最终路线" | tail -1 | sed -E 's/.*\): ([0-9.]+)\/inst.*/\1/')
    BEAM_NUM=$(echo "$OUT" | grep "beam 生成阶段" | tail -1 | sed -E 's/.*阶段\): ([0-9.]+)\/inst.*/\1/')
    [ -z "$FINAL_NUM" ] && FINAL_NUM="NA"
    [ -z "$BEAM_NUM" ] && BEAM_NUM="NA"
    echo "$LQ,$COST,$FEAS,$FINAL_NUM,$BEAM_NUM" >> "$RESULT_CSV"
    echo "  cost=$COST feas=$FEAS finalNum=$FINAL_NUM beamNum=$BEAM_NUM"
done

echo ""
echo "============================================================"
echo "lambda_q sweep 汇总"
echo "============================================================"
cat "$RESULT_CSV"
echo ""
echo "解读："
echo "  lambda_q 越大 → 品质感知越强 → beam 越倾向低损耗路线"
echo "  预期：finalNum 随 lambda_q 下降（品质优化），cost 可能轻微上升（距离-品质权衡）"
echo "  结果存: $RESULT_CSV"
