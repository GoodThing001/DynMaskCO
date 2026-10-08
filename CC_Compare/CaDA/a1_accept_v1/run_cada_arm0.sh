#!/bin/bash
# CaDA 诊断批次 p_c=0 臂单独重跑（首跑被外部 SIGTERM 中断）。
# 启动（服务器，仓库根）：tmux new -d -s xcada0 'bash CC_Compare/CaDA/a1_accept_v1/run_cada_arm0.sh'
# 进度标记：/tmp/xcada0.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xcada0.done
CKPT=../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider cada \
  --provider-ckpt "$CKPT" --provider-device cpu \
  --baseline-from "$BASE" \
  --dev-instances 40 --workers 6 \
  --penalty "p_c=0" --out results/a1_ext_extra/cada_0
echo "DONE_0 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
