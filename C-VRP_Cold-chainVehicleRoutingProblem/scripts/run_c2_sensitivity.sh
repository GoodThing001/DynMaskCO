#!/bin/bash
# ============================================================
# run_c2_sensitivity.sh — 宽 TW sensitivity（品质项决策自由度）
# ============================================================
# 验证导师的机制性观点：「当硬 TW 约束放松后，品质目标拥有更大的决策自由度」。
#
# 用 R2（宽 TW，horizon=48，窗口 0.6）对比 R1（窄 TW，horizon=24，窗口 0.2），
# 看品质项（λ_q=0 vs 2.0）对常温服务顺序的影响是否在宽 TW 下更强。
#
# 用法:
#   bash run_c2_sensitivity.sh
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

GEN="C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
ANALYSIS="C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/mechanism_analysis.py"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"

# 数据：R2 宽 TW，temp_dist "0.5,0,0.5"（50% 常温 + 50% 冷冻，K 差异 50 倍，让品质项有区分度）
DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
R2_TEST="${DATA_DIR}/dcc_50_r2_mixed_temp_edod05_test.npz"

echo "============================================================"
echo "C2/R2 宽 TW sensitivity（品质项决策自由度）"
echo "============================================================"

# 1. 生成 R2 宽 TW 数据
if [ ! -f "$R2_TEST" ]; then
    echo "--- 生成 R2 宽 TW 数据（50% 常温 + 50% 冷冻）---"
    python -u "$GEN" \
        --problem_size 50 --num_instances 128 --type R2 --capacity 50 \
        --temp_dist "0.5,0,0.5" --edod 0.5 \
        --output "$R2_TEST"
else
    echo "--- R2 数据已存在 ---"
fi

# 2. 跑 λ_q=0 vs 2.0，保存路线
for LQ in 0.0 2.0; do
    echo "--- λ_q=$LQ ---"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python -u "$DECODE" \
        --capacity 50 --penalty 3. --data "$R2_TEST" --ckpt "$CKPT" \
        --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
        --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
        --threads_over_batches 1 \
        --enable_resource_decoder --beam_width 16 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
        --enable_quality --lambda_q $LQ --quality_salable_threshold 0.02 \
        --save_routes "/tmp/routes_r2_lq${LQ}.npz" 2>&1 | grep -E "mean cost|最终路线" | tail -2
done

# 3. 机制验证（对比 R2 宽 TW 下品质项效果）
echo ""
echo "--- 机制验证（R2 宽 TW，λ_q=0 vs 2.0）---"
python -u "$ANALYSIS" \
    --routes /tmp/routes_r2_lq0.0.npz \
    --data "$R2_TEST" \
    --routes_lq /tmp/routes_r2_lq2.0.npz

echo ""
echo "============================================================"
echo "解读："
echo "  R2 宽 TW 下，若常温 rank 从 λ_q=0 到 2.0 的下降 >> R1 的 0.18，"
echo "  则证明「硬 TW 约束放松后，品质目标拥有更大决策自由度」。"
echo "  若仍 ~0.2，则品质项在真实场景下确实弱，冷链降为 application。"
echo "============================================================"
