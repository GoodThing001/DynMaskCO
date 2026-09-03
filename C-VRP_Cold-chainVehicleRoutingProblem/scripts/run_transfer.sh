#!/bin/bash
# ============================================================
# 迁移实验补齐：5-seed 迁移 + cross-scale 迁移
# ============================================================
# 方向五：验证「可迁移表示」的稳定性（跨 seed + 跨规模）
#   - 5seed:      训练 4 个纯 R1 seed（123/999/2025/2026）+ zero-shot C1/RC1
#   - cross_scale: 50 节点模型 zero-shot 100 节点（跨规模迁移）
#
# 用法:
#   bash run_transfer.sh 5seed          # 5-seed 迁移（~2.5h 训练 + 评估）
#   bash run_transfer.sh cross_scale    # cross-scale 迁移（快速，用现有 50 节点模型）
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

MODE="${1:-5seed}"
TRAIN="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"
DECODE="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
R1_TRAIN="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_train.npz"

echo "============================================================"
echo "迁移实验补齐: $MODE"
echo "============================================================"

if [ "$MODE" = "5seed" ]; then
    # 训练 4 个纯 R1 seed（seed 42 已有 transfer_r1/step50000.ckpt）
    for SEED in 123 999 2025 2026; do
        CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/transfer_r1/seed${SEED}/step50000.ckpt"
        if [ -f "$CKPT" ]; then
            echo "  [SKIP] seed=${SEED} 已存在"
            continue
        fi
        echo "  [TRAIN] 纯 R1 seed=${SEED} ..."
        mkdir -p "$(dirname "$CKPT")"
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN" \
            --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
            --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
            --num_steps 50000 --save_interval 5000 \
            --data "$R1_TRAIN" --masking_mode spatio_temporal \
            --online_seq_training --online_seq_steps 5 \
            --logdir "C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/transfer_r1_seed${SEED}" \
            --savedir "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/transfer_r1/seed${SEED}" \
            --optimizer_type adamw --weight_decay 1e-2 --target_disruption None --seed $SEED
    done

    # zero-shot 评估：5 个 seed × C1/RC1
    echo ""
    echo "=== zero-shot 评估（5 seed × C1/RC1）==="
    for SEED in 42 123 999 2025 2026; do
        CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/transfer_r1/step50000.ckpt"
        [ "$SEED" != "42" ] && CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/transfer_r1/seed${SEED}/step50000.ckpt"
        for TYPE in c1 rc1; do
            DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_${TYPE}_edod05_test.npz"
            echo -n "  seed=${SEED} ${TYPE^^} ... "
            OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
                --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
                --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
                --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
                --threads_over_batches 1 \
                --enable_resource_decoder --beam_width 16 \
                --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1)
            COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
            FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
            echo "cost=${COST} feas=${FEAS}"
        done
    done

elif [ "$MODE" = "cross_scale" ]; then
    # cross-scale 迁移：50 节点模型 zero-shot 100 节点
    echo "  cross-scale 迁移：50 节点模型 → 100 节点数据"
    CKPT="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/transfer_r1/step50000.ckpt"
    DATA="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_100_r1_edod05_test.npz"

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODE" \
        --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
        --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
        --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 \
        --threads_over_batches 1 \
        --enable_resource_decoder --beam_width 16 \
        --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py
fi
