#!/bin/bash
# RouteFinder 40-day accept arms: p_c=0 + p_c=(5,10,15)。GPU 1、cc_compare env、tmux 内运行。
# 启动（服务器，仓库根）：tmux new -d -s xrf 'bash CC_Compare/RouteFinder/a1_accept_v1/run_rf_arms.sh'
# 进度标记：/tmp/xrf.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xrf.done
CKPT=../CC_Compare/RouteFinder/checkpoints/100/rf-transformer.ckpt
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider routefinder \
  --provider-ckpt "$CKPT" --provider-device cuda \
  --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 \
  --penalty "p_c=0" --out results/a1_ext_extra/routefinder_0
echo "DONE_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider routefinder \
  --provider-ckpt "$CKPT" --provider-device cuda \
  --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 \
  --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/routefinder_51015
echo "DONE_51015 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
