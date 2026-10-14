#!/bin/bash
# A-v1 正式批次 v2 · CPU 腿 B（lih 单跑，最慢：旧正式臂 5.3h/档）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
MARK=/tmp/xformal_v2/cpuB.done
mkdir -p "$OUT"; : > "$MARK"

LIH=../CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt
for tier in "p_c=0:0" "p_c=(5,10,15):51015"; do
  p="${tier%%:*}"; sfx="${tier##*:}"
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider lih --provider-ckpt "$LIH" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 \
    --penalty "$p" --out "$OUT/lih_${sfx}"
  echo "DONE_lih_${sfx} $?" >> "$MARK"
done
echo ALLDONE >> "$MARK"
