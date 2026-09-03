#!/bin/bash
# ============================================================
# run_core_ablation.sh — 核心演化消融（论文第一贡献的证据）
# ============================================================
# 证明「原始 MaskCO 不能直接解决动态冷链 VRP，经过改造后能解」。
# 用同一个 causal checkpoint（typed_v1 seed42），逐层开启 decode 改造：
#
#   A. Model-only         = 原始 MaskCO（静态范式直接用于动态，无约束处理）
#   B. +Resource Beam     = 资源状态可行性保证（Phase 3a）
#   C. +EDD repair        = 时间窗修复（Earliest Due Date）
#   D. +TW 2-opt（最终）  = 约束感知局部搜索
#
# 预期 feas 演化：A ~0% → B ~90% → C ~95% → D 100%
#
# 用法:
#   bash run_core_ablation.sh [r1|c1|rc1]
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

TYPE="${1:-r1}"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_${TYPE}_edod05_test.npz"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"

echo "============================================================"
echo "核心演化消融（${TYPE^^} EDoD=0.5）"
echo "  ckpt: $CKPT（typed_v1 seed42，因果模型）"
echo "============================================================"
echo ""

# 公共 decode 参数（不含 TW/beam 处理）
COMMON="--capacity 50 --penalty 3. --data $DATA --ckpt $CKPT \
    --keep_rate 0.3 --two_opt_steps 0 --batch_size 8 --runs 8 --cycles 40 \
    --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
    --threads_over_batches 1"

run_one() {
    local label="$1"; shift
    local extra="$1"; shift
    echo "--- $label ---"
    OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" $COMMON $extra 2>&1)
    COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
    VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
    echo "  cost=$COST feas=$FEAS viol=$VIOL"
    echo ""
}

# A. Model-only（原始 MaskCO：无 beam、无 TW 修复）
run_one "A. Model-only（原始 MaskCO）" ""

# B. +Resource Beam
run_one "B. +Resource Beam" "--enable_resource_decoder --beam_width 16"

# C. +EDD repair
run_one "C. +EDD repair" "--enable_resource_decoder --beam_width 16 --enable_tw_repair_edd"

# D. +TW 2-opt（最终）
run_one "D. +TW 2-opt（最终）" "--enable_resource_decoder --beam_width 16 --enable_tw_repair_edd --enable_tw_aware_2opt_py --two_opt_steps 4 --enable_tw_filter"

echo "============================================================"
echo "消融完成"
echo "解读：A feas 应接近 0（原始 MaskCO 无法处理 TW），"
echo "      D feas 应 100%（完整 DynMaskCO）。"
echo "      这证明「原始 MaskCO 不能直接解，改造后能解」。"
echo "============================================================"
