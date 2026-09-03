#!/bin/bash
# ============================================================
# 方向A: 混合 EDoD 训练 (R1+C1+RC1 × 0.2+0.5+0.8)
# 用法: bash run_mixed_edod.sh [smoke|full]
# ============================================================
set -e

MODE="${1:-smoke}"
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/mixed_edod"
LOG_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix"
MERGED="${DATA_DIR}/dcc_50_mixed_edod_train.npz"
RESULT_FILE="${LOG_DIR}/results_mixed_edod.csv"
mkdir -p "$CKPT_DIR"

# ── Step 1: 合并数据 ──
echo "========== STEP 1: 合并混合训练数据 =========="
python3 -c "
import numpy as np

files = []
for t in ['r1','c1','rc1']:
    for e in ['02','05','08']:
        f = '${DATA_DIR}/dcc_50_' + t + '_edod' + e + '_train.npz'
        files.append(dict(np.load(f)))

merged = {}
for key in files[0].keys():
    merged[key] = np.concatenate([f[key] for f in files], axis=0)

np.savez('${MERGED}', **merged)
print(f'Merged {len(files)} files -> ${MERGED}')
print(f'Instances: {merged[\"coords\"].shape[0]}')
# 验证：三种类型 + 三种EDoD 的可见比例
vm = merged['visible_mask']
print(f'visible_mask: {vm.mean():.1%} visible ({(1-vm).mean():.1%} future)')
"
echo ""

if [ "$MODE" = "smoke" ]; then
    STEPS=2000; SAVE_INT=2000; BATCH=64
    R_EVAL="--runs 4 --cycles 20"
    echo "=== MODE: SMOKE (${STEPS} steps) ==="
else
    STEPS=50000; SAVE_INT=5000; BATCH=64
    R_EVAL="--runs 8 --cycles 40"
    echo "=== MODE: FULL (${STEPS} steps) ==="
fi

# ── Step 2: 训练 ──
echo ""
echo "========== STEP 2: 混合 EDoD 训练 =========="
TRAIN_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py"

if [ -f "${CKPT_DIR}/step${STEPS}.ckpt" ]; then
    echo "  SKIP (ckpt exists)"
else
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u "$TRAIN_SCRIPT" \
        --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --encoder_input_dim 7 --peak_lr 1e-3 --batch_size $BATCH \
        --num_steps $STEPS --save_interval $SAVE_INT \
        --data "$MERGED" --masking_mode spatio_temporal \
        --logdir "${LOG_DIR}/mixed_edod" --savedir "$CKPT_DIR" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
fi

# ── Step 3: 评估 9 组 EDoD × 3 seeds ──
echo ""
echo "========== STEP 3: 评估 =========="
DECODE_SCRIPT="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
CKPT="${CKPT_DIR}/step${STEPS}.ckpt"

echo "type,edod,seed,cost,tw_feas,tw_viol" > "$RESULT_FILE"

for TYPE in R1 C1 RC1; do
  for EDOD in 0.2 0.5 0.8; do
    ESTR=$(echo "$EDOD" | sed 's/\.//')
    DATA="${DATA_DIR}/dcc_50_${TYPE,,}_edod${ESTR}_test.npz"

    for SEED in 42 123 999; do
        echo -n "  ${TYPE} EDoD=${EDOD} seed=${SEED} ... "
        OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
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
        echo "feas=${FEAS} cost=${COST}"
    done
  done
done

# ── Step 4: 汇总 + 对比 ──
echo ""
echo "========== STEP 4: 汇总 =========="
python3 -c "
import pandas as pd, numpy as np

df = pd.read_csv('${RESULT_FILE}')
print()
print('=== 混合 EDoD 训练结果 (Causal, mean ± std over 3 seeds) ===')

# 加载单 EDoD 基线对比
df_single = pd.read_csv('${LOG_DIR}/results_full.csv')
df_single_agg = df_single.groupby(['type','edod']).agg(
    feas_mean=('tw_feas', lambda x: x.str.rstrip('%').astype(float).mean()),
    feas_std=('tw_feas', lambda x: x.str.rstrip('%').astype(float).std()),
).reset_index()

print()
hdr = 'Type  EDoD  Mixed_Feas         Single_Feas        Δ'
print(hdr)
print('-' * 70)
for t in ['R1','C1','RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df[(df['type']==t) & (df['edod']==e)]
        if len(sub)==0: continue
        f = sub['tw_feas'].str.rstrip('%').astype(float)
        c = sub['cost']
        fm, fs = f.mean(), f.std()
        cm, cs = c.mean(), c.std()

        s = df_single_agg[(df_single_agg['type']==t) & (df_single_agg['edod']==e)]
        s_feas = f'{s[\"feas_mean\"].values[0]:.1f}%' if len(s)>0 else 'n/a'

        delta = fm - s['feas_mean'].values[0] if len(s)>0 else 0
        delta_s = f'+{delta:.1f}pp' if delta > 0 else f'{delta:.1f}pp'
        print('%-5s %-5s %4.1f%% ± %4.1f%%   %-18s %s' % (t, e, fm, fs, s_feas, delta_s))
    print()

print('Overall Mixed: %.1f%%' % df['tw_feas'].str.rstrip('%%').astype(float).mean())
print('Overall Single: %.1f%%' % df_single['tw_feas'].str.rstrip('%%').astype(float).mean())
print()

# Cost 对比
print('--- Cost 对比 ---')
for t in ['R1','C1','RC1']:
    sub = df[df['type']==t]
    sc = df_single[df_single['type']==t]['cost'].mean()
    mc = sub['cost'].mean()
    print('%s: Mixed=%.2f  Single=%.2f  Δ=%.2f' % (t, mc, sc, mc-sc))
"

echo ""
echo "=== DONE ==="
echo "结果: ${RESULT_FILE}"
echo "模型: ${CKPT_DIR}"
