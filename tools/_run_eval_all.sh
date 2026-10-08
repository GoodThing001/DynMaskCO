#!/bin/bash
# seed 稳健性 + 留出实例的固定状态评估与配对分析（自动等训练完成后跑）。
# 等待两个训练 tmux 会话结束
while tmux ls 2>/dev/null | grep -qE "mpre_s4[34]"; do
  sleep 60
done
set -e
source /home/hzeng/miniconda3/etc/profile.d/conda.sh
conda activate MASKCO_env
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

PROFILE="results/o0cc/scale_v2/objective_profile.json"
CVRP="../MASKCO_code/ckpts/cvrp100.ckpt"

run_fsq() {
  local DATA="$1" CKPT="$2" SPLIT="$3" OUT="$4"
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python scripts/evaluation/run_fixed_state_quality.py \
      --data "$DATA" --cvrp-ckpt "$CVRP" --model-ckpt "$CKPT" \
      --objective-profile "$PROFILE" --split "$SPLIT" --out "$OUT" \
      > "${OUT}.log" 2>&1
  python scripts/evaluation/analyze_fixed_state.py \
      --per-state "$OUT/per_state.json" --out "$OUT/analysis.json" >> "${OUT}.log" 2>&1
}

for SEED in 43 44; do
  CKPT="results/m0_scale/mpre_reinforce_s${SEED}_clean/model.ckpt"
  run_fsq data/m0_scale/dcc_50_r1_edod05_train_teacher.npz "$CKPT" train "results/m0_scale/fsq_s${SEED}_train"
  run_fsq data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz "$CKPT" cal "results/m0_scale/fsq_s${SEED}_cal"
done

# 留出实例（clean seed42 checkpoint）
run_fsq data/heldout/dcc_50_r1_edod05_heldout.npz \
  results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
  cal results/m0_scale/fsq_heldout_clean

echo "ALL_EVAL_DONE"
