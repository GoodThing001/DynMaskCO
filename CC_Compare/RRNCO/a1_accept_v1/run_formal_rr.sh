#!/bin/bash
# 正式批次 RRNCO 腿（修复版 driver，seal 路径已修）：p_c=0 → 主档，GPU 1。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
export CUDA_VISIBLE_DEVICES=1
PY=/home/hzeng/envs/cc_compare/bin/python
MARK=/tmp/xformal_rr.done
BASE=results/a1_step2_gate_c1_20260926/gate.json
RRC=../CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt
: > "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver rrnco \
  --rrnco-ckpt "$RRC" --rrnco-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=0" --out results/a1_ext_extra/rrnco_0_f
echo "DONE_rr_0 $?" >> "$MARK"
$PY scripts/evaluation/run_a1_external_accept.py --solver rrnco \
  --rrnco-ckpt "$RRC" --rrnco-device cuda --baseline-from "$BASE" \
  --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results/a1_ext_extra/rrnco_51015_f
echo "DONE_rr_51015 $?" >> "$MARK"
echo ALLDONE >> "$MARK"
