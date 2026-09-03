#!/bin/bash
# ============================================================
# run_cross_scale.sh — cross-scale 迁移（50-node 模型 zero-shot 100-node）
# ============================================================
# 验证「masked representation 未只记忆 50-node 规模」，支持 scale transfer。
#
# 用 50-node typed_v1 模型 zero-shot 跑 100-node R1 数据，
# 对比 50-node IID（14.79）看 scale 增长是否次线性。
#
# 用法:
#   bash run_cross_scale.sh
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

GEN="C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1/phase3c/seed42/step50000.ckpt"
DATA_100="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_100_r1_edod05_test.npz"

echo "============================================================"
echo "cross-scale 迁移：50-node 模型 zero-shot 100-node R1"
echo "  ckpt: $CKPT（50-node typed_v1）"
echo "============================================================"

# 1. 生成 100-node 数据（64 实例，100-node 较慢）
if [ ! -f "$DATA_100" ]; then
    echo "--- 生成 100-node R1 数据 ---"
    python -u "$GEN" \
        --problem_size 100 --num_instances 64 --type R1 --capacity 50 --edod 0.5 \
        --output "$DATA_100"
else
    echo "--- 100-node 数据已存在 ---"
fi

# 2. zero-shot 跑 100-node
echo "--- zero-shot 100-node decode ---"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
    --capacity 50 --penalty 3. --data "$DATA_100" --ckpt "$CKPT" \
    --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
    --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
    --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 16 \
    --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1 | tail -8

echo ""
echo "============================================================"
echo "对比基准："
echo "  50-node IID（typed_v1）: cost=14.79, feas=100%"
echo "  100-node zero-shot: 看上方 cost/feas"
echo "  若 cost 增长 < 2×（次线性）+ feas 仍高，则 scale transfer 成立"
echo "============================================================"
