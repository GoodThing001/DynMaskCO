#!/bin/bash
# ============================================================
# P0-3: 四层消融矩阵（用 P0-2 因果模型重跑）
# 用法: bash run_ablation.sh
# ============================================================
set -e
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"

CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/r1_edod05/step50000.ckpt"
DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz"
DECODER="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
BASE_FLAGS="--capacity 50 --penalty 3. --data $DATA --ckpt $CKPT --keep_rate 0.3 --batch_size 8 --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 --threads_over_batches 1"

echo "============================================================"
echo "P0-3: 四层消融矩阵 (R1 EDoD=0.5, P0-2 Causal Model)"
echo "============================================================"
echo ""

# L1: model_only (无任何TW后处理)
echo "--- L1: model_only ---"
CUDA_VISIBLE_DEVICES=0 python -u "$DECODER" $BASE_FLAGS --runs 1 --cycles 1 --two_opt_steps 0 2>&1 | grep -E "mean cost|TW feas|Avg TW viol"
echo ""

# L2: +TW_filter
echo "--- L2: +TW_filter ---"
CUDA_VISIBLE_DEVICES=0 python -u "$DECODER" $BASE_FLAGS --runs 1 --cycles 1 --two_opt_steps 0 --enable_tw_filter 2>&1 | grep -E "mean cost|TW feas|Avg TW viol"
echo ""

# L3: +EDD repair
echo "--- L3: +EDD repair ---"
CUDA_VISIBLE_DEVICES=0 python -u "$DECODER" $BASE_FLAGS --runs 1 --cycles 1 --two_opt_steps 0 --enable_tw_filter --enable_tw_repair_edd 2>&1 | grep -E "mean cost|TW feas|Avg TW viol"
echo ""

# L4: +TW 2opt (full pipeline)
echo "--- L4: +TW 2opt (full) ---"
CUDA_VISIBLE_DEVICES=0 python -u "$DECODER" $BASE_FLAGS --runs 8 --cycles 40 --two_opt_steps 4 --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1 | grep -E "mean cost|TW feas|Avg TW viol|Gap_ref"

echo ""
echo "============================================================"
echo "Done. Format:"
echo "  L1 model_only | L2 +filter | L3 +EDD | L4 +TW2opt"
echo "============================================================"
