#!/bin/bash
# POMO / Sym-NCO / SGBS 40-day accept arms（P3 批次，2026-09-29/30）。
# 每方法两条接单臂：p_c=0 与 p_c=(5,10,15)，CPU 推理（GPU0/GPU1 均被他人占用）。
# 启动（服务器，仓库根）：
#   tmux new -d -s xpomo 'bash CC_Compare/run_xpomo_arms.sh'
# 进度标记：/tmp/xpomo.done（DONE_<name>_0 / DONE_<name>_51015 / ALLDONE）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xpomo.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
POMO_CKPT=/home/hzeng/project/MASKCO-Main/CC_Compare/POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt
SYMNCO_CKPT=/home/hzeng/project/MASKCO-Main/CC_Compare/Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt
SGBS_CKPT=/home/hzeng/project/MASKCO-Main/CC_Compare/SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
: > "$MARK"

run_arm () {
  local name="$1" ckpt="$2" penalty="$3" out="$4" tag="$5"
  echo "=== START $tag $(date '+%F %T') ==="
  $PY scripts/evaluation/run_a1_external_accept.py \
    --solver ordering --provider "$name" \
    --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" \
    --dev-instances 40 --workers 6 \
    --penalty "$penalty" --out "results/a1_ext_extra/$out"
  echo "DONE_${tag} $?" >> "$MARK"
  echo "=== END $tag $(date '+%F %T') ==="
}

run_arm pomo   "$POMO_CKPT"   "p_c=0"          pomo_0     pomo_0
run_arm pomo   "$POMO_CKPT"   "p_c=(5,10,15)"  pomo_51015 pomo_51015
run_arm symnco "$SYMNCO_CKPT" "p_c=0"          symnco_0     symnco_0
run_arm symnco "$SYMNCO_CKPT" "p_c=(5,10,15)"  symnco_51015 symnco_51015
run_arm sgbs   "$SGBS_CKPT"   "p_c=0"          sgbs_0     sgbs_0
run_arm sgbs   "$SGBS_CKPT"   "p_c=(5,10,15)"  sgbs_51015 sgbs_51015

echo ALLDONE >> "$MARK"
