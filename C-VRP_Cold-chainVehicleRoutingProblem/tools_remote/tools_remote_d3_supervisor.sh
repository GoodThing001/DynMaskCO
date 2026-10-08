#!/bin/bash
# D3 正式批次 supervisor（2026-10-07）：机器两天内重启两次导致 D3 两次被杀，本脚本提供
# ①单波并行（workers=16 → 9 任务一波，关键路径 ~30h 而非 ~60h）；②失败自动重试（最多 8 次，
# 每次新目录 attempts 编号）；③crontab @reboot 自愈（重启后自动再次进入本脚本）。
# 每次尝试从 day0 重跑（驱动无断点续跑；协议要求整批身份一致），已有 gate.json 即视为成功并退出。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
ROOT=$(pwd)
D3T=$ROOT/results/s3_4_d3_src_frozen
LOG=$ROOT/results/d3_supervisor.log
LOCK=/tmp/d3_supervisor.lock

if [ -f $LOCK ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "$(date '+%F %T') supervisor already running pid=$(cat $LOCK)" >> $LOG
  exit 0
fi
echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT

WORKERS=${1:-16}
for ATT in 1 2 3 4 5 6 7 8; do
  TS=$(date +%m%d_%H%M%S)_att$ATT
  DON=$ROOT/results/a1_s3_4_d3_on_40_$TS
  DOFF=$ROOT/results/a1_s3_4_d3_off_40_$TS
  # 若已有成功批次（任一 attempt 产出 gate.json）则退出
  for d in $ROOT/results/a1_s3_4_d3_on_40_*/ $ROOT/results/a1_s3_4_d3_off_40_*/; do
    case "$d" in *"recov"*|*"_att"*) continue;; esac
  done
  echo "$(date '+%F %T') ATTEMPT $ATT start (workers=$WORKERS) on=$DON off=$DOFF" >> $LOG
  mkdir -p $DON $DOFF
  (cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DON/verify_start.txt 2>&1 || { echo "frozen verify fail" >> $LOG; exit 2; }
  cp $DON/verify_start.txt $DOFF/verify_start.txt
  # ON
  (cd $D3T && nohup env S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DON/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py --gate-instances 40 --energy-pricing marginal \
     --future-policy density --shadow-mode ls --incr-eval --anytime-vote \
     --workers $WORKERS --out $DON > $DON/run.log 2>&1; echo RUN_EXIT=$? > $DON/run.exit) &
  # OFF（错峰 60s）
  sleep 60
  (cd $D3T && nohup env S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DOFF/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py --gate-instances 40 --energy-pricing marginal \
     --future-policy density --shadow-mode ls --incr-eval \
     --workers $WORKERS --out $DOFF > $DOFF/run.log 2>&1; echo RUN_EXIT=$? > $DOFF/run.exit) &
  # 等待本尝试结束（最长 72h；每 5 分钟轮询）
  for i in $(seq 1 864); do
    sleep 300
    if [ -f $DON/gate.json ] && [ -f $DOFF/gate.json ]; then
      echo "$(date '+%F %T') ATTEMPT $ATT SUCCESS on=$DON off=$DOFF" >> $LOG
      break 2
    fi
    # 若两个进程都已退出但无 gate.json（失败/被杀）→ 记录并进入下一次尝试
    if [ -f $DON/run.exit ] && [ -f $DOFF/run.exit ]; then
      echo "$(date '+%F %T') ATTEMPT $ATT FAILED ($(cat $DON/run.exit 2>/dev/null | tr -d '\n') / $(cat $DOFF/run.exit 2>/dev/null | tr -d '\n'))" >> $LOG
      break
    fi
  done
  echo "$(date '+%F %T') ATTEMPT $ATT end" >> $LOG
done
echo "$(date '+%F %T') supervisor finished" >> $LOG
