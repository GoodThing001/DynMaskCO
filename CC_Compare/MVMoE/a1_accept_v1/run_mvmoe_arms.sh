#!/bin/bash
# MVMoE 40-day accept arms: p_c=0 + p_c=(5,10,15)。CPU 推理、cc_compare env、tmux 内运行。
# 启动（服务器，仓库根）：tmux new -d -s xmv 'bash CC_Compare/MVMoE/a1_accept_v1/run_mvmoe_arms.sh'
# 进度标记：/tmp/xmv.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xmv.done
CKPT=/home/hzeng/project/MASKCO-Main/CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider mvmoe \
  --provider-ckpt "$CKPT" --provider-device cpu \
  --baseline-from "$BASE" \
  --dev-instances 40 --workers 6 \
  --penalty "p_c=0" --out results/a1_ext_extra/mvmoe_0
echo "DONE_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider mvmoe \
  --provider-ckpt "$CKPT" --provider-device cpu \
  --baseline-from "$BASE" \
  --dev-instances 40 --workers 6 \
  --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/mvmoe_51015
echo "DONE_51015 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
