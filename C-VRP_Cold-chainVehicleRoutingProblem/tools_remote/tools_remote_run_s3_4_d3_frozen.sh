#!/bin/bash
# S3-4-D3 正式 40 天启动器（烟测通过后；从冻结源码树运行；空输出目录；新会话名；资源记录；
# 落盘自动验收）。⚠️ 前置：① 本地 T1–T33 ALL PASS；② D3 烟测（ON/OFF 2 天）通过并已记录；
# ③ A3 全部批次已落盘（工作目录源码在 A3 结束后才可替换为冻结口径的同一修订版）；
# ④ tools_remote/tools_remote_freeze_s3_4_d3_src.sh 已创建 results/s3_4_d3_src_frozen（不可变）。
# 主对照口径（注册件 S3-4-D3 §2）：ON/OFF 同修订版按天配对（唯一差别=超时语义），各 40 天三档。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
FROZEN=results/s3_4_d3_src_frozen
test -f $FROZEN/SOURCE_MANIFEST.sha256 || { echo "REJECT: 冻结树不存在，请先跑 tools_remote/tools_remote_freeze_s3_4_d3_src.sh"; exit 2; }

TS=$(date +%m%d_%H%M%S)
OUT_ON=results/a1_s3_4_d3_on_40_$TS
OUT_OFF=results/a1_s3_4_d3_off_40_$TS
SESS_ON=s3_4_d3_on_$TS
SESS_OFF=s3_4_d3_off_$TS

mkdir $OUT_ON || { echo "REJECT: out dir exists $OUT_ON"; exit 2; }
mkdir $OUT_OFF || { echo "REJECT: out dir exists $OUT_OFF"; exit 2; }
tmux has-session -t $SESS_ON 2>/dev/null && { echo "REJECT: session $SESS_ON exists"; exit 2; }
tmux has-session -t $SESS_OFF 2>/dev/null && { echo "REJECT: session $SESS_OFF exists"; exit 2; }

(cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT_ON/verify_start.txt 2>&1 \
  || { cp $OUT_ON/verify_start.txt $OUT_OFF/verify_start.txt; echo "REJECT: 冻结树清单核验失败"; exit 2; }
cp $OUT_ON/verify_start.txt $OUT_OFF/verify_start.txt

chmod -R a-w $FROZEN
echo "frozen tree read-only: $(ls -ld $FROZEN)" | tee -a $OUT_ON/verify_start.txt >> $OUT_OFF/verify_start.txt

{
  echo "launch: $(date '+%F %T')  S3-4-D3 formal 40d x 3 tiers"
  echo "out: $OUT_ON / $OUT_OFF   frozen: $FROZEN"
  echo "config: shadow=ls incr-eval marginal reveal 10s；ON=--anytime-vote；OFF=旧语义同修订版"
  uptime
  echo "nproc: $(nproc)  workers: 6  time_limit: 10s"
  sha256sum $FROZEN/SOURCE_MANIFEST.sha256 | cut -d' ' -f1
} | tee $OUT_ON/resource.txt > $OUT_OFF/resource.txt

run_one() {
  SESS=$1; OUT=$2; EXTRA=$3
  tmux new-session -d -s "$SESS" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
     S34D3_FROZEN_SRC=$FROZEN S34D3_EVIDENCE_OUT=$OUT/import_evidence.txt \
       $PY $FROZEN/s3_4_d3_run_frozen.py \
       --gate-instances 40 --energy-pricing marginal --shadow-mode ls --incr-eval \
       $EXTRA --workers 6 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
     $PY scripts/evaluation/diag_adjudicate.py $OUT/gate.json --expected 40 \
       --out $OUT/adjudication.json > $OUT/adjudication.log 2>&1; arc=\$?; \
     echo ADJUDICATE_EXIT=\$arc >> $OUT/run.exit; \
     (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_end.txt 2>&1; \
     echo \"$(date '+%F %T') $SESS done run=\$rc adj=\$arc\""
}

# OFF 先起（旧语义基线），ON 后起；两批并行但错峰 60s（空载窗口要求由启动者确认）
run_one $SESS_OFF $OUT_OFF ""
sleep 60
run_one $SESS_ON $OUT_ON "--anytime-vote"
sleep 3
tmux ls
echo "launched: $OUT_ON / $OUT_OFF (sessions $SESS_ON / $SESS_OFF)"
