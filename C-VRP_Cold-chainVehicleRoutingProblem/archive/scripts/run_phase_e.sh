#!/bin/bash
# ============================================================
# Phase E: P0-2(动态泄漏修复) + P0-4(冷链物理建模) 全流程
# 用法: bash run_phase_e.sh [smoke|full]
#   smoke — 小数据量快速验证 (1280 instances, 500 steps)
#   full  — 完整实验 (12800 instances, 50000 steps)
# ============================================================
set -e

MODE="${1:-smoke}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"

# conda
source /home/hzeng/envs/MASKCO_env/bin/activate

# ── GPU: 固定卡0 (卡1留给别人) ──
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 路径
DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix"
mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR"

# 量级参数
if [ "$MODE" = "smoke" ]; then
    N_TRAIN=1280; N_VAL=128; N_TEST=128
    STEPS=500; SAVE_INT=500; BATCH=16
    R_EVAL="--runs 2 --cycles 10"
    echo "=== MODE: SMOKE (${N_TRAIN} inst, ${STEPS} steps) ==="
else
    N_TRAIN=12800; N_VAL=1280; N_TEST=1280
    STEPS=50000; SAVE_INT=5000; BATCH=64
    R_EVAL="--runs 8 --cycles 40"
    echo "=== MODE: FULL (${N_TRAIN} inst, ${STEPS} steps) ==="
fi

# ============================================================
# 1. 生成数据 (R1/C1/RC1 × 0.2/0.5/0.8)
# ============================================================
echo ""
echo "========== STEP 1: 生成数据 =========="

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    PREFIX="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}"

    if [ -f "${PREFIX}_test.npz" ]; then
        echo "  SKIP ${TYPE} EDoD=${EDOD} (exists)"
        continue
    fi

    echo "  Generating ${TYPE} EDoD=${EDOD}..."

    python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
        --problem_size 50 --num_instances $N_TRAIN --type $TYPE --capacity 50 \
        --edod $EDOD --output "${PREFIX}_train.npz" 2>&1 | tail -1

    python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
        --problem_size 50 --num_instances $N_VAL --type $TYPE --capacity 50 \
        --edod $EDOD --output "${PREFIX}_val.npz" 2>&1 | tail -1

    python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
        --problem_size 50 --num_instances $N_TEST --type $TYPE --capacity 50 \
        --edod $EDOD --output "${PREFIX}_test.npz" 2>&1 | tail -1
  done
done

# 验证关键字段
echo ""
echo "--- 数据字段验证 ---"
python -c "
import numpy as np, os, glob
for f in sorted(glob.glob('${DATA_DIR}/*_test.npz'))[:1]:
    d = dict(np.load(f))
    name = os.path.basename(f)
    print(f'{name}: keys={sorted(d.keys())}')
    print(f'  quality_loss: [{d[\"quality_loss\"].min():.4f}, {d[\"quality_loss\"].max():.4f}]')
    print(f'  energy_mat:   {d[\"energy_mat\"].shape}')
    print(f'  visible_mask: {(d[\"visible_mask\"]==0).mean():.1%} future orders')
    print(f'  reveal_time:  {int((d[\"reveal_time\"]>0).sum())} dynamic nodes')
"

# ============================================================
# 2. 训练 (EDoD=0.5 为基础模型)
# ============================================================
echo ""
echo "========== STEP 2: 训练 DCC (P0-2 fixed) =========="

TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"

for TYPE in R1 C1 RC1; do
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod05_train.npz"
    SAVE="${CKPT_DIR}/${TYPE,,}_edod05"
    LOG="${LOG_DIR}/train_${TYPE,,}_edod05"

    if [ -f "${SAVE}/step${STEPS}.ckpt" ]; then
        echo "  SKIP training ${TYPE} EDoD=0.5 (ckpt exists)"
        continue
    fi

    echo "  Training ${TYPE} EDoD=0.5 ..."
    python -u "$TRAIN_SCRIPT" \
        --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --encoder_input_dim 7 --peak_lr 1e-3 --batch_size $BATCH \
        --num_steps $STEPS --save_interval $SAVE_INT \
        --data "$DATA" --masking_mode spatio_temporal \
        --logdir "$LOG" --savedir "$SAVE" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
done

# ============================================================
# 3. 评估 (9组 EDoD 矩阵 × 3 seeds)
# ============================================================
echo ""
echo "========== STEP 3: 评估 =========="

DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
SEEDS=(42 123 999)
RESULT_FILE="${LOG_DIR}/results_${MODE}.csv"
echo "type,edod,seed,cost,tw_feas,tw_viol" > "$RESULT_FILE"

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    CKPT="${CKPT_DIR}/${TYPE,,}_edod05/step${STEPS}.ckpt"
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"

    for SEED in ${SEEDS[@]}; do
        echo "  Eval ${TYPE} EDoD=${EDOD} seed=${SEED}"

        OUT=$(CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -u "$DECODE_SCRIPT" \
            --capacity 50 --penalty 3. --data "$DATA" --ckpt "$CKPT" \
            --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 $R_EVAL \
            --sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed $SEED \
            --threads_over_batches 1 \
            --enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py 2>&1)

        COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
        FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
        VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
        echo "${TYPE},${EDOD},${SEED},${COST},${FEAS},${VIOL}" >> "$RESULT_FILE"
    done
  done
done

# ============================================================
# 4. 汇总
# ============================================================
echo ""
echo "========== STEP 4: 汇总 =========="

python -c "
import pandas as pd, numpy as np

df = pd.read_csv('${RESULT_FILE}')
print('=== EDoD 矩阵 (mean ± std over seeds) ===')
print()
print('%-6s %-6s %-16s %-16s %-10s' % ('Type','EDoD','Feas%','Cost','Viol'))
print('-' * 54)
for t in ['R1','C1','RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type']==t) & (df['edod']==e)]
        if len(sub) == 0: continue
        f = sub['tw_feas'].str.rstrip('%').astype(float)
        c = sub['cost']
        v = sub['tw_viol']
        print('%-6s %-6s %4.1f%% ± %4.1f%%   %7.3f ± %7.3f   %4.1f' % (t, e, f.mean(), f.std(), c.mean(), c.std(), v.mean()))
print()
print('Avg by Type:')
for t in ['R1','C1','RC1']:
    f = df[df['type']==t]['tw_feas'].str.rstrip('%').astype(float)
    print('  %s: %.1f%%' % (t, f.mean()))
print('Overall: %.1f%%' % df['tw_feas'].str.rstrip('%%').astype(float).mean())
"

# 冷链物理指标报告
echo ""
echo "--- 冷链物理指标 (参考解 quality_loss + energy) ---"
python -c "
import numpy as np, os, glob

for f in sorted(glob.glob('${DATA_DIR}/*_test.npz')):
    d = dict(np.load(f))
    name = os.path.basename(f).replace('.npz','')
    ql_mean = d['quality_loss'].sum(axis=1).mean()
    en_mean = d['energy_mat'].sum(axis=(1,2)).mean() / d['energy_mat'].shape[1]
    print(f'{name}:  quality_loss={ql_mean:.4f}  energy/edge={en_mean:.4f}')
"

echo ""
echo "=== DONE ==="
echo "结果: ${RESULT_FILE}"
echo "模型: ${CKPT_DIR}"
