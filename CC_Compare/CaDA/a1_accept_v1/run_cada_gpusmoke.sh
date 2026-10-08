#!/bin/bash
# CaDA GPU 1 天烟测（主线正式批前置）：GPU1、1 实例 1 worker、p_c=0。
# 启动（服务器，仓库根）：tmux new -d -s xcgpusmoke 'bash CC_Compare/CaDA/a1_accept_v1/run_cada_gpusmoke.sh'
# 进度标记：/tmp/xcgpusmoke.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xcgpusmoke.done
CKPT=../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt
BASE=results/a1_step2_gate_c1_20260926/gate.json
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py \
  --solver ordering --provider cada \
  --provider-ckpt "$CKPT" --provider-device cuda \
  --baseline-from "$BASE" \
  --dev-instances 1 --workers 1 \
  --penalty "p_c=0" --out results/a1_ext_extra/cada_gpusmoke
echo "DONE_GPU_SMOKE $?" >> "$MARK"
echo ALLDONE >> "$MARK"
