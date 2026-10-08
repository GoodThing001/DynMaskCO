#!/bin/bash
# A3 落盘后聚合裁决链（2026-10-03）。只读：run.exit × 9 + gate.json × 9 → diag_a3_adjudicate。
# 前置：9 批全部落盘（gate.json 存在且 run.exit 有 RUN_EXIT）。本脚本不启动任何批次。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=$(date +%m%d_%H%M%S)
LOCK=results/A2_LOCK.json
OUT=results/a1_a3/A3_ADJUDICATION_$TS.json

DIRS=""
for d in results/a1_a3_online_density_*_40_1003_040454; do
  [ -f "$d/gate.json" ] || { echo "MISSING gate.json: $d"; continue; }
  [ -f "$d/run.exit" ] || { echo "MISSING run.exit: $d"; continue; }
  echo "== $d: $(tr -d '\n' < $d/run.exit)"
  DIRS="$DIRS $d"
done
N=$(echo $DIRS | wc -w)
echo "batches with gate.json+run.exit: $N / 9"
[ "$N" -lt 9 ] && { echo "ABORT: 不是全部 9 批已落盘（拒绝部分聚合）"; exit 2; }

$PY scripts/evaluation/diag_a3_adjudicate.py $DIRS --lock $LOCK --out $OUT \
  | head -120
echo "A3 adjudication written: $OUT"
