#!/bin/bash
# S3-3 连续超额价格（冻结链版，2026-10-01 预写）。⚠️ 条件启用：rr3 formal 未过门且
# 执行端按预声明顺序推进到 S3-3 之后。与旧 tools_remote_s3_3_smoke.sh 的差异 = 全部
# 从 results/rr3_src_frozen 冻结树运行（身份一致），逐次逐文件核验 + 空目录/会话防撞。
# 协议（路线图 09-30 第 143/144/160 行）：{5,10} 元/kWh 先各 2 天 p_c=0 烟测
# （正确性/价格尺度，非证据），再按协议各 ≤1 次 40 天三档（p_c 三档由驱动默认覆盖）。
# 用法：bash tools_remote/tools_remote_s3_3_frozen.sh smoke   # 烟测两档
#       bash tools_remote/tools_remote_s3_3_frozen.sh full    # 全量两档（烟测通过后）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
FROZEN=results/rr3_src_frozen
MODE=${1:-smoke}
test -f $FROZEN/SOURCE_MANIFEST.sha256 || { echo "REJECT: 冻结树缺失"; exit 2; }

TS=$(date +%m%d_%H%M%S)
run_one() {
  OP=$1; N=$2; PEN=$3
  TAG=op${OP}_n${N}_${MODE}
  OUT=results/a1_s3_3_${TAG}_$TS
  SESS=s3_3_${TAG}_$TS
  mkdir $OUT || { echo "REJECT: out dir exists $OUT"; exit 2; }
  tmux has-session -t $SESS 2>/dev/null && { echo "REJECT: session $SESS exists"; exit 2; }
  (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_start.txt 2>&1 \
    || { echo "REJECT: 冻结树清单核验失败 ($TAG)"; exit 2; }
  {
    echo "launch: $(date '+%F %T')  mode=$MODE  overage_price=$OP  days=$N  penalty=$PEN"
    uptime
    echo "nproc: $(nproc)  workers: 6  time_limit: 10s（驱动默认）"
  } > $OUT/resource.txt
  PENARG=""
  if [ "$PEN" != "None" ]; then PENARG="--penalty $PEN"; fi
  ADJ=""
  if [ "$N" = "40" ]; then
    ADJ="; $PY scripts/evaluation/diag_adjudicate.py $OUT/gate.json --expected 40 --out $OUT/adjudication.json > $OUT/adjudication.log 2>&1; arc=\$?; echo ADJUDICATE_EXIT=\$arc >> $OUT/run.exit"
  fi
  tmux new-session -d -s "$SESS" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
     RR3_FROZEN_SRC=$FROZEN RR3_EVIDENCE_OUT=$OUT/import_evidence.txt \
       $PY $FROZEN/rr3_run_frozen.py \
       --gate-instances $N $PENARG --energy-pricing marginal \
       --overage-price $OP --workers 6 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit$ADJ; \
     (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_end.txt 2>&1; \
     echo \"$(date '+%F %T') $SESS done run=\$rc\""
}

if [ "$MODE" = "smoke" ]; then
  run_one 5  2 p_c=0
  run_one 10 2 p_c=0
elif [ "$MODE" = "full" ]; then
  run_one 5  40 None
  run_one 10 40 None
else
  echo "usage: $0 smoke|full"; exit 2
fi
sleep 3
tmux ls
echo "launched S3-3 $MODE (ts=$TS)"
