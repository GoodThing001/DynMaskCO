#!/bin/bash
# rr3 正式重跑启动器（第二步：从冻结源码树运行；空输出目录；新会话名；资源记录；落盘自动验收）
# 2026-10-01 用户决策补齐：启动/结束逐文件清单核验（sha256sum -c）、冻结树置只读、
# import 路径证据存档。
# 使用前：tools_remote/tools_remote_freeze_rr3_src.sh 已创建 results/rr3_src_frozen（不可变）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
FROZEN=results/rr3_src_frozen
test -f $FROZEN/SOURCE_MANIFEST.sha256 || { echo "REJECT: 冻结树不存在，请先跑 tools_remote/tools_remote_freeze_rr3_src.sh"; exit 2; }

TS=$(date +%m%d_%H%M%S)
OUT_R=results/a1_rr3_reveal_40_$TS
OUT_D=results/a1_rr3_density_40_$TS
SESS_R=rr3_reveal_$TS
SESS_D=rr3_density_$TS

# 前置：输出目录必须为空（mkdir 失败即拒绝）、会话名不覆盖
mkdir $OUT_R || { echo "REJECT: out dir exists $OUT_R"; exit 2; }
mkdir $OUT_D || { echo "REJECT: out dir exists $OUT_D"; exit 2; }
tmux has-session -t $SESS_R 2>/dev/null && { echo "REJECT: session $SESS_R exists"; exit 2; }
tmux has-session -t $SESS_D 2>/dev/null && { echo "REJECT: session $SESS_D exists"; exit 2; }

# 启动逐文件清单核验（在冻结树内执行；任何 mismatch 即拒绝启动）
(cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT_R/verify_start.txt 2>&1 \
  || { cp $OUT_R/verify_start.txt $OUT_D/verify_start.txt; echo "REJECT: 冻结树清单核验失败"; exit 2; }
cp $OUT_R/verify_start.txt $OUT_D/verify_start.txt

# 冻结树置只读（运行期与验收期不再被写入）
chmod -R a-w $FROZEN
echo "frozen tree read-only: $(ls -ld $FROZEN)" | tee -a $OUT_R/verify_start.txt >> $OUT_D/verify_start.txt

# 资源口径记录（两批同 workers/time-limit；启动时负载/核数存档）
{
  echo "launch: $(date '+%F %T')"
  echo "out: $OUT_R / $OUT_D   frozen: $FROZEN"
  uptime
  echo "nproc: $(nproc)  workers: 9  time_limit: 10s（驱动默认）"
  sha256sum $FROZEN/SOURCE_MANIFEST.sha256 | cut -d' ' -f1
} | tee $OUT_R/resource.txt > $OUT_D/resource.txt

run_one() {
  SESS=$1; OUT=$2; EXTRA=$3
  tmux new-session -d -s "$SESS" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
     RR3_FROZEN_SRC=$FROZEN RR3_EVIDENCE_OUT=$OUT/import_evidence.txt \
       $PY $FROZEN/rr3_run_frozen.py \
       --gate-instances 40 --energy-pricing marginal $EXTRA --workers 9 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
     $PY scripts/evaluation/diag_adjudicate.py $OUT/gate.json --expected 40 \
       --out $OUT/adjudication.json > $OUT/adjudication.log 2>&1; arc=\$?; \
     echo ADJUDICATE_EXIT=\$arc >> $OUT/run.exit; \
     (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_end.txt 2>&1; \
     echo \"$(date '+%F %T') $SESS done run=\$rc adj=\$arc\""
}

run_one $SESS_R $OUT_R ""
run_one $SESS_D $OUT_D "--future-policy density"
sleep 3
tmux ls
echo "launched: $OUT_R / $OUT_D (sessions $SESS_R / $SESS_D)"
