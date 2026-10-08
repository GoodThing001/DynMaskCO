#!/bin/bash
# 正式批次 GPU 腿（2026-09-30 修复版 driver + rrnco_accept_replanner sha 2c4dee57）：
# 仅 RouteFinder p_c=0 → 主档，GPU 1、40 天、seed 20260926。
# RRNCO 正式腿由独立脚本 CC_Compare/RRNCO/a1_accept_v1/run_formal_rr.sh 写 rrnco_*_f，
# 本脚本不含 RRNCO 段（2026-09-30 复核第 4 点：避免与独立脚本双写同一目录）。
# 启动（服务器仓库根）：tmux new -d -s xformal_gpu 'bash CC_Compare/RouteFinder/a1_accept_v1/run_formal_gpu.sh'
# 进度标记：/tmp/xformal_gpu.done
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xformal_gpu.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
RFP=../CC_Compare/RouteFinder/checkpoints/100/rf-transformer.ckpt
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider routefinder \
  --provider-ckpt "$RFP" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=0" --out results/a1_ext_extra/routefinder_0_f
echo "DONE_rf_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider routefinder \
  --provider-ckpt "$RFP" --provider-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/routefinder_51015_f
echo "DONE_rf_51015 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
