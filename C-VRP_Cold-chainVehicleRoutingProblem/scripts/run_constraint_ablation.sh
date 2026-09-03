#!/bin/bash
# ============================================================
# run_constraint_ablation.sh — Week 4 Day 3-4 约束消融实验
# ============================================================
# 目的：验证「模型/管线决策因果依赖约束参数」，三个扰动实验：
#   1. temp_swap — 打乱非 depot 已知客户温度标签（模型层信号）
#   2. tw_relax  — 客户时间窗宽度 ×2（管线层优化空间）
#   3. capacity  — decode 改 --capacity {40,60,70}（管线层容量约束）
#
# 对比基准：已保存的 typed_v1_edge seed42 baseline 路线（8 runs × 40 cycles,
#           capacity=50，原始 TW/temp），见 BASELINE_ROUTES_*。
#
# 用法:
#   bash run_constraint_ablation.sh perturb   # 只生成扰动数据
#   bash run_constraint_ablation.sh decode    # 只解码扰动数据（含 capacity）
#   bash run_constraint_ablation.sh baseline  # 只重解 baseline（原始数据 seed42，本会话）
#   bash run_constraint_ablation.sh noise     # 只解码噪声地板（同数据不同 seed）
#   bash run_constraint_ablation.sh compare   # 只对比（含 effect_gap，若 baseline/noise 已解码）
#   bash run_constraint_ablation.sh full      # 一步到位（默认）
#
# 建议用 tmux 跑（>5min）：
#   tmux new-session -d -s ablation 'bash .../run_constraint_ablation.sh full > /tmp/ablation.log 2>&1'
# ============================================================
set -e

MODE="${1:-full}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
ABLATE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/analysis/constraint_ablation.py"
CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/typed_v1_edge/phase3c/seed42/step50000.ckpt"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/week4"

# 已保存的 baseline 路线（8 runs × 40 cycles，capacity=50，原始数据）
# ⚠️ 若你的文件命名不同，改这里
BASELINE_ROUTES_R1="${LOG_DIR}/routes_typed_edge_r1_edod05.npz"
BASELINE_ROUTES_C1="${LOG_DIR}/routes_typed_edge_c1_edod05.npz"
BASELINE_ROUTES_RC1="${LOG_DIR}/routes_typed_edge_rc1_edod05.npz"

TYPES=(R1 C1 RC1)

# 解码参数（与 baseline 严格一致）
CAPACITY=50
KEEP_RATE=0.3
TWO_OPT_STEPS=4
RUNS=8
CYCLES=40
SAMPLING_STEPS=2
SEED=42
SEED_NOISE=43   # 噪声地板：同数据不同 seed，量化解码随机性

mkdir -p "$LOG_DIR"

run_decode() {
    # $1 = data, $2 = ckpt, $3 = capacity, $4 = save_routes, $5 = seed（默认 $SEED）
    local dseed="${5:-$SEED}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE_SCRIPT" \
        --capacity "$3" --penalty 3. --data "$1" --ckpt "$2" \
        --keep_rate $KEEP_RATE --two_opt_steps $TWO_OPT_STEPS \
        --batch_size 8 --runs $RUNS --cycles $CYCLES \
        --sampling_steps $SAMPLING_STEPS --augment_level 0 \
        --gumbel_scale_factor 0. --seed $dseed \
        --threads_over_batches 1 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py \
        --enable_resource_decoder --beam_width 16 \
        --save_routes "$4"
}

# ═══════════════════════════════════════════════════════════
# 1. 生成扰动数据（CPU，秒级）
# ═══════════════════════════════════════════════════════════
gen_perturbed() {
    for TYPE in "${TYPES[@]}"; do
        T=$(echo "$TYPE" | tr 'A-Z' 'a-z')
        ORIG="${DATA_DIR}/dcc_50_${T}_edod05_test.npz"

        # temp_swap（始终重生成，避免复用前一 session 可能被 SSH 打断的坏文件）
        echo "--- [temp_swap] 扰动 $TYPE ---"
        python "$ABLATE" perturb --ptype temp_swap \
            --data "$ORIG" \
            --out "${DATA_DIR}/dcc_50_${T}_edod05_test_tempswap.npz"

        # tw_relax（同上）
        echo "--- [tw_relax] 扰动 $TYPE ---"
        python "$ABLATE" perturb --ptype tw_relax \
            --data "$ORIG" \
            --out "${DATA_DIR}/dcc_50_${T}_edod05_test_twrelax.npz"
    done
}

# ═══════════════════════════════════════════════════════════
# 2. 解码扰动数据（GPU）
# ═══════════════════════════════════════════════════════════
decode_perturbed() {
    for TYPE in "${TYPES[@]}"; do
        T=$(echo "$TYPE" | tr 'A-Z' 'a-z')

        # temp_swap
        echo "--- [decode] temp_swap $TYPE ---"
        run_decode "${DATA_DIR}/dcc_50_${T}_edod05_test_tempswap.npz" "$CKPT" $CAPACITY \
            "${LOG_DIR}/routes_tempswap_${T}.npz"

        # tw_relax
        echo "--- [decode] tw_relax $TYPE ---"
        run_decode "${DATA_DIR}/dcc_50_${T}_edod05_test_twrelax.npz" "$CKPT" $CAPACITY \
            "${LOG_DIR}/routes_twrelax_${T}.npz"
    done

    # capacity 扰动（数据不动，只改 --capacity，先跑 R1 最省钱，其余类型可按需开）
    for CAP in 40 60 70; do
        echo "--- [decode] capacity=$CAP R1 ---"
        run_decode "${DATA_DIR}/dcc_50_r1_edod05_test.npz" "$CKPT" $CAP \
            "${LOG_DIR}/routes_cap${CAP}_r1.npz"
    done
}

# ═══════════════════════════════════════════════════════════
# 2b. 重解 baseline（原始数据，seed 42，本会话）—— 消除跨会话 confound
# ═══════════════════════════════════════════════════════════
decode_baseline() {
    for TYPE in "${TYPES[@]}"; do
        T=$(echo "$TYPE" | tr 'A-Z' 'a-z')
        echo "--- [decode] baseline (seed=$SEED, 本会话) $TYPE ---"
        run_decode "${DATA_DIR}/dcc_50_${T}_edod05_test.npz" "$CKPT" $CAPACITY \
            "${LOG_DIR}/routes_baseline_seed${SEED}_${T}.npz" "$SEED"
    done
}

# ═══════════════════════════════════════════════════════════
# 2c. 解码噪声地板（同数据，不同 seed）—— 用于分离解码随机性
# ═══════════════════════════════════════════════════════════
decode_noise() {
    for TYPE in "${TYPES[@]}"; do
        T=$(echo "$TYPE" | tr 'A-Z' 'a-z')
        echo "--- [decode] baseline noise (seed=$SEED_NOISE) $TYPE ---"
        run_decode "${DATA_DIR}/dcc_50_${T}_edod05_test.npz" "$CKPT" $CAPACITY \
            "${LOG_DIR}/routes_baseline_seed${SEED_NOISE}_${T}.npz" "$SEED_NOISE"
    done
}

# ═══════════════════════════════════════════════════════════
# 3. 对比
# ═══════════════════════════════════════════════════════════
compare_all() {
    for TYPE in "${TYPES[@]}"; do
        T=$(echo "$TYPE" | tr 'A-Z' 'a-z')
        ORIG="${DATA_DIR}/dcc_50_${T}_edod05_test.npz"

        # baseline（本会话 seed42 优先，回退前一 session 已保存）+ 噪声地板 路线选择
        FRESH_BASE="${LOG_DIR}/routes_baseline_seed${SEED}_${T}.npz"
        if [ -f "$FRESH_BASE" ]; then
            BASE="$FRESH_BASE"
        else
            case "$TYPE" in
                R1)  BASE="$BASELINE_ROUTES_R1";;
                C1)  BASE="$BASELINE_ROUTES_C1";;
                RC1) BASE="$BASELINE_ROUTES_RC1";;
            esac
            echo "  [WARN] 用前一 session baseline（建议重解: bash $0 baseline）: $BASE"
        fi
        NOISE="${LOG_DIR}/routes_baseline_seed${SEED_NOISE}_${T}.npz"
        [ -f "$BASE" ]  || { echo "  [SKIP] baseline 路线缺失: $BASE"; continue; }
        [ -f "$NOISE" ] || { echo "  [WARN] 噪声地板缺失（无 effect_gap）: $NOISE"; NOISE=""; }

        echo ""
        echo "════════ temp_swap $TYPE ════════"
        python "$ABLATE" compare --data "$ORIG" \
            --routes_orig "$BASE" \
            --routes_pert "${LOG_DIR}/routes_tempswap_${T}.npz" \
            ${NOISE:+--routes_noise "$NOISE"} \
            --out "${LOG_DIR}/ablation_tempswap_${T}"

        echo ""
        echo "════════ tw_relax $TYPE ════════"
        python "$ABLATE" compare --data "$ORIG" \
            --routes_orig "$BASE" \
            --routes_pert "${LOG_DIR}/routes_twrelax_${T}.npz" \
            ${NOISE:+--routes_noise "$NOISE"} \
            --out "${LOG_DIR}/ablation_twrelax_${T}"
    done

    # capacity 对比（R1）
    FRESH_R1="${LOG_DIR}/routes_baseline_seed${SEED}_r1.npz"
    BASE_R1="$FRESH_R1"; [ -f "$FRESH_R1" ] || BASE_R1="$BASELINE_ROUTES_R1"
    NOISE_R1="${LOG_DIR}/routes_baseline_seed${SEED_NOISE}_r1.npz"
    [ -f "$NOISE_R1" ] || NOISE_R1=""
    for CAP in 40 60 70; do
        echo ""
        echo "════════ capacity=$CAP R1 ════════"
        python "$ABLATE" compare --data "${DATA_DIR}/dcc_50_r1_edod05_test.npz" \
            --routes_orig "$BASE_R1" \
            --routes_pert "${LOG_DIR}/routes_cap${CAP}_r1.npz" \
            ${NOISE_R1:+--routes_noise "$NOISE_R1"} \
            --out "${LOG_DIR}/ablation_cap${CAP}_r1"
    done
}

case "$MODE" in
    perturb)  gen_perturbed;;
    decode)   decode_perturbed;;
    baseline) decode_baseline;;
    noise)    decode_noise;;
    compare)  compare_all;;
    full)
        gen_perturbed
        decode_perturbed
        decode_baseline
        decode_noise
        compare_all
        ;;
    *)
        echo "用法: bash $0 {perturb|decode|baseline|noise|compare|full}"; exit 1;;
esac

echo ""
echo "============================================================"
echo "约束消融完成。对比明细见 ${LOG_DIR}/ablation_*_compare.csv"
echo "============================================================"
