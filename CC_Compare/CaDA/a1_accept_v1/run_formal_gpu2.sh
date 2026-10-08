#!/bin/bash
# 正式批次 GPU 续（2026-09-30 修复版代码）：CaDA ×2 → MVMoE ×2，GPU 1、40 天、seed 20260926。
# 前置：xformal_gpu 完成（/tmp/xformal_gpu.done 含 ALLDONE）后再启动，避免 GPU 竞争。
# 启动：tmux new -d -s xformal_gpu2 'bash CC_Compare/CaDA/a1_accept_v1/run_formal_gpu2.sh'
# 标记：/tmp/xformal_gpu2.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xformal_gpu2.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
CADA=../CC_Compare/CaDA/100/result/2024-1121-1355/checkpoint-300.pt
MVM=../CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider cada \
  --provider-ckpt "$CADA" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=0" --out results/a1_ext_extra/cada_0_f
echo "DONE_cada_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider cada \
  --provider-ckpt "$CADA" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/cada_51015_f
echo "DONE_cada_51015 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider mvmoe \
  --provider-ckpt "$MVM" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=0" --out results/a1_ext_extra/mvmoe_0_f
echo "DONE_mvmoe_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider mvmoe \
  --provider-ckpt "$MVM" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/mvmoe_51015_f
echo "DONE_mvmoe_51015 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
