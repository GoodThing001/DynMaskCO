#!/bin/bash
# A1 正式 40 天三档（烟测通过后；从冻结源码树运行；空输出目录；落盘自动验收）。
# ⚠️ 前置：① A3 全部落盘；② 服务器 T 全套（含 T34–T36）绿；③ A1 烟测通过；④ V 正式训练完成；
# ⑤ tools_remote/tools_remote_freeze_a1_src.sh 已建 results/a1_src_frozen（不可变）。
# 五臂同批同日：uncond_hist/cond_hist/explicit_feat/a1_consensus/a1_rollout，40 天 × 三档。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
FROZEN=results/a1_src_frozen
test -f $FROZEN/SOURCE_MANIFEST.sha256 || { echo "REJECT: 冻结树不存在，请先跑 tools_remote/tools_remote_freeze_a1_src.sh"; exit 2; }

TS=$(date +%m%d_%H%M%S)
OUT=results/a1_a1_formal_40_$TS
SESS=a1_formal_$TS
mkdir $OUT || { echo "REJECT: out dir exists $OUT"; exit 2; }
tmux has-session -t $SESS 2>/dev/null && { echo "REJECT: session $SESS exists"; exit 2; }

(cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_start.txt 2>&1 \
  || { echo "REJECT: 冻结树清单核验失败"; exit 2; }
chmod -R a-w $FROZEN
echo "frozen tree read-only: $(ls -ld $FROZEN)" >> $OUT/verify_start.txt

{
  echo "launch: $(date '+%F %T')  A1 formal 40d x 3 tiers x 5 arms"
  echo "config: density marginal greedy 10s K=10 v_ckpt=$FROZEN/models/a1_v_model.npz"
  uptime
  echo "nproc: $(nproc)  workers: 10"
  sha256sum $FROZEN/SOURCE_MANIFEST.sha256 | cut -d' ' -f1
} > $OUT/resource.txt

tmux new-session -d -s "$SESS" \
  "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
   A1_FROZEN_SRC=$FROZEN A1_EVIDENCE_OUT=$OUT/import_evidence.txt \
     $PY $FROZEN/a1_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode greedy --arms uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout \
     --a1-v-ckpt $FROZEN/models/a1_v_model.npz --a1-h 2.0 \
     --workers 10 --out $OUT \
     > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
   $PY scripts/evaluation/diag_adjudicate.py $OUT/gate.json --expected 40 \
     --out $OUT/adjudication.json > $OUT/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $OUT/run.exit; \
   (cd $FROZEN && sha256sum -c SOURCE_MANIFEST.sha256) > $OUT/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') $SESS done run=\$rc adj=\$arc\""
sleep 3
tmux ls
echo "launched: $OUT (session $SESS)"
