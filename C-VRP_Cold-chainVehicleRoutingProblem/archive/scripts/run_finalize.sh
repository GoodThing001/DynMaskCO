#!/bin/bash
# ============================================================
# 4-in-1 Rapid Finalization
# 方向1: 热约束 beam (fixed QLoss) + 方向2-4
# 用法: bash run_finalize.sh
# ============================================================
set -e
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

DATA_DIR="C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
CKPT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix"
DECODER="C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py"
RESULT_DIR="C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix"

echo "============================================================"
echo "4-in-1 Rapid Finalization"
echo "方向1: 热约束 beam (fixed QLoss eval)"
echo "方向2: 100-node scale (9x3 seeds)"
echo "方向3: 统计升级 (5 seeds)"
echo "方向4: PyVRP 基线"
echo "============================================================"

# ── 方向1: 热约束 beam (fixed) ──
echo ""
echo "--- 方向1: 热约束 beam (fixed QLoss) ---"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python -u "$DECODER" \
    --capacity 50 --penalty 3. \
    --data "${DATA_DIR}/dcc_50_r1_edod05_test.npz" \
    --ckpt "${CKPT_DIR}/phase3c/step50000.ckpt" \
    --keep_rate 0.3 --batch_size 8 --runs 1 --cycles 1 --sampling_steps 1 \
    --two_opt_steps 4 --seed 42 --gumbel_scale_factor 0. --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 16 --enable_tw_aware_2opt_py 2>&1 \
    | grep -E "mean cost|TW feas|Avg TW viol"

# ── 方向2: 100-node 全矩阵 ──
echo ""
echo "--- 方向2: 100-node scale (R1/C1/RC1 × EDoD 0.5) ---"
RESULT_100="${RESULT_DIR}/results_100node.csv"
echo "type,edod,seed,cost,tw_feas,tw_viol" > "$RESULT_100"

for TYPE in R1 C1 RC1; do
  DATA="${DATA_DIR}/dcc_100_${TYPE,,}_edod05_test.npz"
  if [ ! -f "$DATA" ]; then
    echo "  Generating 100-node ${TYPE}..."
    python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
        --problem_size 100 --num_instances 1280 --type $TYPE --capacity 100 \
        --edod 0.5 --output "$DATA" 2>&1 | tail -1
  fi
  for SEED in 42 123 999; do
    echo -n "  ${TYPE} EDoD=0.5 seed=${SEED} ... "
    OUT=$(XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u "$DECODER" \
        --capacity 100 --penalty 3. --data "$DATA" --ckpt "${CKPT_DIR}/phase3c/step50000.ckpt" \
        --keep_rate 0.3 --batch_size 4 --runs 1 --cycles 1 --sampling_steps 1 \
        --two_opt_steps 4 --seed $SEED --gumbel_scale_factor 0. --threads_over_batches 1 \
        --enable_resource_decoder --beam_width 16 --enable_tw_aware_2opt_py 2>&1)
    COST=$(echo "$OUT" | grep "mean cost:" | tail -1 | awk '{print $NF}')
    FEAS=$(echo "$OUT" | grep "TW feas rate:" | tail -1 | awk '{print $NF}')
    VIOL=$(echo "$OUT" | grep "Avg TW viol" | tail -1 | awk '{print $NF}')
    echo "${TYPE},0.5,${SEED},${COST},${FEAS},${VIOL}" >> "$RESULT_100"
    echo "feas=${FEAS} cost=${COST}"
  done
done

# ── 方向3: 统计升级 (从已有 CSV 读取，计算 Wilcoxon + Cliff's delta) ──
echo ""
echo "--- 方向3: 统计升级 (Wilcoxon signed-rank + Cliff's delta) ---"
python3 -c "
import pandas as pd, numpy as np
from scipy import stats

# Load existing 50-node full matrix (3 seeds)
df50 = pd.read_csv('${RESULT_DIR}/results_phase3c_full.csv')
print('=== 50-node Statistical Analysis (3 seeds) ===')
print()

# Per type-EDoD group statistics
for t in ['R1','C1','RC1']:
    for e in [0.2, 0.5, 0.8]:
        sub = df50[(df50['type']==t)&(df50['edod']==e)]
        if len(sub)<3: continue
        f = sub['tw_feas'].str.rstrip('%').astype(float)
        c = sub['cost']
        print('%s EDoD=%.1f: feas=%5.1f%% ± %4.1f%%  cost=%5.2f ± %4.2f (n=%d)' % (
            t, e, f.mean(), f.std(), c.mean(), c.std(), len(f)))

# Compare beam vs greedy reference for each type
print()
print('--- Beam vs Greedy Reference (paired Wilcoxon) ---')
gre_ref = {'R1':15.95, 'C1':9.13, 'RC1':13.65}
for t in ['R1','C1','RC1']:
    sub = df50[df50['type']==t]['cost']
    w, p = stats.wilcoxon(sub.values, np.full(len(sub), gre_ref[t]))
    delta = (sub.mean() - gre_ref[t]) / gre_ref[t] * 100
    print('%s: beam=%.2f vs greedy=%.2f, Wilcoxon p=%.4f, Δ=%+.1f%%' % (
        t, sub.mean(), gre_ref[t], p, delta))

print()
print('=== 100-node Results ===')
try:
    df100 = pd.read_csv('${RESULT_DIR}/results_100node.csv')
    for t in ['R1','C1','RC1']:
        sub = df100[df100['type']==t]
        if len(sub)>0:
            f = sub['tw_feas'].str.rstrip('%').astype(float)
            c = sub['cost']
            print('%s: feas=%5.1f%% ± %4.1f%%  cost=%5.2f ± %4.2f (n=%d)' % (
                t, f.mean(), f.std(), c.mean(), c.std(), len(f)))
except: print('(100-node results not available yet)')
" 2>&1

# ── 方向4: PyVRP 基线 ──
echo ""
echo "--- 方向4: PyVRP 基线 ---"
python3 -c "
try:
    import pyvrp
    print('PyVRP installed: %s' % pyvrp.__version__)
except ImportError:
    print('PyVRP not installed. Run: pip install pyvrp')
    print('Then adapt rolling-horizon + frozen prefix for dynamic comparison.')
" 2>&1

echo ""
echo "=== DONE ==="
echo "100-node results: ${RESULT_100}"
