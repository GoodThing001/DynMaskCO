#!/bin/bash
# A-v1 外部排序臂 ×4 方法顺序跑（tmux xam）：AttentionModel / DeepACO / Omni-VRP / LIH。
# 每方法两条臂：p_c=0 与 p_c=(5,10,15)（40 天开发集 seed 20260926）。
# 启动（服务器，仓库根）：
#   tmux new -d -s xam 'bash CC_Compare/a1_accept_common/run_xam_arms.sh'
# 进度标记：/tmp/xam.done（DONE_<name>_0 / DONE_<name>_51015 / ALLDONE）
#
# 注意：本批次为「诊断批次」（时限修复前口径，2026-09-30 主线更新后统一正式重跑）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
ROOT=/home/hzeng/project/MASKCO-Main
MARK=/tmp/xam.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"

run_arm () {   # $1=provider 名  $2=ckpt 绝对路径  $3=penalty  $4=out 子目录名  $5=标记名
  $PY scripts/evaluation/run_a1_external_accept.py \
    --solver ordering --provider "$1" \
    --provider-ckpt "$2" --provider-device cpu \
    --baseline-from "$BASE" \
    --dev-instances 40 --workers 6 \
    --penalty "$3" --out "results/a1_ext_extra/$4"
  echo "DONE_$5 $?" >> "$MARK"
}

# ---- 1. AttentionModel (greedy, pretrained/cvrp_100/epoch-99.pt) ----
run_arm attention "$ROOT/CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt" \
  "p_c=0" attention_0 attention_0
run_arm attention "$ROOT/CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt" \
  "p_c=(5,10,15)" attention_51015 attention_51015

# ---- 2. DeepACO (n_ants=20, T=100, pretrained/cvrp/cvrp100.pt) ----
run_arm deepaco "$ROOT/CC_Compare/DeepACO/pretrained/cvrp/cvrp100.pt" \
  "p_c=0" deepaco_0 deepaco_0
run_arm deepaco "$ROOT/CC_Compare/DeepACO/pretrained/cvrp/cvrp100.pt" \
  "p_c=(5,10,15)" deepaco_51015 deepaco_51015

# ---- 3. Omni-VRP (POMO-CVRP instance-norm, aug×8) ----
run_arm omnivrp "$ROOT/CC_Compare/Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt" \
  "p_c=0" omnivrp_0 omnivrp_0
run_arm omnivrp "$ROOT/CC_Compare/Omni-VRP/pretrained/POMO-CVRP/uniform/checkpoint-30500-cvrp100-instance-norm.pt" \
  "p_c=(5,10,15)" omnivrp_51015 omnivrp_51015

# ---- 4. LIH (CVRP50, NeuRewriter steps=100) ----
run_arm lih "$ROOT/CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt" \
  "p_c=0" lih_0 lih_0
run_arm lih "$ROOT/CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt" \
  "p_c=(5,10,15)" lih_51015 lih_51015

echo ALLDONE >> "$MARK"
