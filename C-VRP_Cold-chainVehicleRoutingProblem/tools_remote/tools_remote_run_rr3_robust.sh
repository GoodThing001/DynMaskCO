#!/bin/bash
# rr3 通过后的稳健性复验启动器（20261001 预写，仅当 rr3 主门通过后启用）。
# 同一冻结源码树 results/rr3_src_frozen、同一驱动/参数、仅换 --gate-seed 20260930 / 20261001。
# 模式：默认 reveal 两种子（A2 主门口径）；--with-density 追加 density；--density-only 仅 density。
# 落盘后 tmux 链自动验收（diag_adjudicate → verify_end）。
# 用法：bash tools_remote/tools_remote_run_rr3_robust.sh [--with-density|--density-only]
#   --with-density = reveal + density 各两种子；--density-only = 仅 density 两种子
#   （2026-10-02：reveal 主门合格负结果后，用户决策只复验 density 杠杆 → --density-only）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
FROZEN=results/rr3_src_frozen
MODE=reveal
[ "${1:-}" = "--with-density" ] && MODE=both
[ "${1:-}" = "--density-only" ] && MODE=density
test -f $FROZEN/SOURCE_MANIFEST.sha256 || { echo "REJECT: 冻结树缺失"; exit 2; }

TS=$(date +%m%d_%H%M%S)
run_one() {
  SEED=$1; FUTURE=$2
  TAG=${SEED}_${FUTURE}
  OUT=results/a1_rr3_robust_${TAG}_40_$TS
  SESS=rr3_robust_${TAG}_$TS
  EXTRA=""
  [ "$FUTURE" = "density" ] && EXTRA="--future-policy density"
  mkdir $OUT || { echo "REJECT: out dir exists $OUT"; exit 2; }
  tmux has-session -t $SESS 2>/dev/null && { echo "REJECT: session $SESS exists"; exit 2; }
  (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_start.txt 2>&1 \
    || { echo "REJECT: 冻结树清单核验失败 ($TAG)"; exit 2; }
  {
    echo "launch: $(date '+%F %T')"
    echo "out: $OUT   frozen: $FROZEN   gate_seed: $SEED   future_policy: $FUTURE"
    uptime
    echo "nproc: $(nproc)  workers: 9  time_limit: 10s（驱动默认）"
  } > $OUT/resource.txt
  tmux new-session -d -s "$SESS" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
     RR3_FROZEN_SRC=$FROZEN RR3_EVIDENCE_OUT=$OUT/import_evidence.txt \
       $PY $FROZEN/rr3_run_frozen.py \
       --gate-instances 40 --gate-seed $SEED --energy-pricing marginal $EXTRA --workers 9 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
     $PY scripts/evaluation/diag_adjudicate.py $OUT/gate.json --expected 40 \
       --out $OUT/adjudication.json > $OUT/adjudication.log 2>&1; arc=\$?; \
     echo ADJUDICATE_EXIT=\$arc >> $OUT/run.exit; \
     (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_end.txt 2>&1; \
     echo \"$(date '+%F %T') $SESS done run=\$rc adj=\$arc\""
}
[ "$MODE" != "density" ] && run_one 20260930 reveal
[ "$MODE" != "density" ] && run_one 20261001 reveal
[ "$MODE" != "reveal" ] && run_one 20260930 density
[ "$MODE" != "reveal" ] && run_one 20261001 density
sleep 3
tmux ls
echo "launched: robust mode=$MODE (ts=$TS)"
