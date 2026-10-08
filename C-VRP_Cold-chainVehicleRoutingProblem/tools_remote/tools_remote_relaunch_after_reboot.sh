#!/bin/bash
# 服务器重启后恢复启动器（2026-10-04 ~17:20）：重启 A3 C×3 + D3 正式 ON/OFF + A1 正式。
# 资产已验证完整：模型 bin、A2_LOCK、双冻结树（清单 12/12、13/13）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
ROOT=$(pwd)
TS=$(date +%m%d_%H%M%S)
LOG=$ROOT/results/relaunch_$TS.log
exec > >(tee -a $LOG) 2>&1
echo "relaunch after reboot: $(date '+%F %T')"

LOCK=$ROOT/results/A2_LOCK.json
D3T=$ROOT/results/s3_4_d3_src_frozen
A1T=$ROOT/results/a1_src_frozen

get_base() {
  $PY -c "import json;d=json.load(open('$LOCK',encoding='utf-8'));print(d['gate_json_by_seed'].get('$1',''))"
}

# ---- 1) A3 C×3（与原始启动器同配置：density 基线、迭代 remask 0.5 1 轮、GPU1、workers 2）----
for SEED in 20260926 20260930 20261001; do
  BASE=$(get_base $SEED)
  [ -n "$BASE" ] || { echo "REJECT: 缺 seed $SEED 基线路径"; exit 2; }
  OUT=$ROOT/results/a1_a3_online_density_C_${SEED}_40_$TS
  mkdir -p $OUT
  {
    echo "relaunch C $SEED: $(date '+%F %T')  baseline_from=$BASE"
    uptime
  } > $OUT/resource.txt
  tmux new-session -d -s a3c_${SEED}_$TS \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_FLAGS=--xla_gpu_autotune_level=0 \
     $PY scripts/evaluation/run_a1_step3_compare.py \
       --model-bin $ROOT/results/a1_a3/a3_train200_partial_pretrained_s42/model.bin \
       --arm pretrained --only maskco --baseline-from $BASE \
       --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
       --gate-seed $SEED --energy-pricing marginal --future-policy density \
       --iterative --remask-frac 0.5 --rounds 1 --workers 2 --out $OUT \
       > $OUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $OUT/run.exit; \
     echo \"$(date '+%F %T') a3c_$SEED done run=\$rc\""
  sleep 30
done

# ---- 2) D3 正式 ON/OFF（冻结树，清单已核验）----
DON=$ROOT/results/a1_s3_4_d3_on_40_$TS
DOFF=$ROOT/results/a1_s3_4_d3_off_40_$TS
mkdir -p $DON $DOFF
(cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DON/verify_start.txt 2>&1 \
  || { echo "REJECT: D3 冻结树核验失败"; exit 2; }
cp $DON/verify_start.txt $DOFF/verify_start.txt
tmux new-session -d -s d3off_$TS \
  "cd $D3T && S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DOFF/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode ls --incr-eval --workers 6 --out $DOFF \
     > $DOFF/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $DOFF/run.exit; \
   $PY $ROOT/scripts/evaluation/diag_adjudicate.py $DOFF/gate.json --expected 40 \
     --out $DOFF/adjudication.json > $DOFF/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $DOFF/run.exit; \
   (cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DOFF/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') d3off done run=\$rc adj=\$arc\""
sleep 30
tmux new-session -d -s d3on_$TS \
  "cd $D3T && S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DON/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode ls --incr-eval --anytime-vote --workers 6 --out $DON \
     > $DON/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $DON/run.exit; \
   $PY $ROOT/scripts/evaluation/diag_adjudicate.py $DON/gate.json --expected 40 \
     --out $DON/adjudication.json > $DON/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $DON/run.exit; \
   (cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DON/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') d3on done run=\$rc adj=\$arc\""
sleep 30

# ---- 3) A1 正式五臂（冻结树 + V npz）----
AOUT=$ROOT/results/a1_a1_formal_40_$TS
mkdir -p $AOUT
(cd $A1T && sha256sum -c SOURCE_MANIFEST.sha256) > $AOUT/verify_start.txt 2>&1 \
  || { echo "REJECT: A1 冻结树核验失败"; exit 2; }
tmux new-session -d -s a1_$TS \
  "cd $A1T && A1_FROZEN_SRC=$A1T A1_EVIDENCE_OUT=$AOUT/import_evidence.txt \
     $PY $A1T/a1_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode greedy --arms uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout \
     --a1-v-ckpt $A1T/models/a1_v_model.npz --a1-h 2.0 \
     --workers 10 --out $AOUT \
     > $AOUT/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $AOUT/run.exit; \
   $PY $ROOT/scripts/evaluation/diag_adjudicate.py $AOUT/gate.json --expected 40 \
     --out $AOUT/adjudication.json > $AOUT/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $AOUT/run.exit; \
   (cd $A1T && sha256sum -c SOURCE_MANIFEST.sha256) > $AOUT/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') a1 done run=\$rc adj=\$arc\""

sleep 3
tmux ls
echo "relaunch complete: C×3 + d3on/d3off + a1 (ts=$TS)"
