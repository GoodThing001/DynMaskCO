#!/bin/bash
# 正式 40 天批次启动器（2026-10-04，烟测通过后）：D3 ON/OFF + A1 五臂，全部从独立冻结树启动。
# 冻结树从 results/_newrev_staging（完整新修订源码集）构建，只读；不触碰工作目录。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
ROOT=$(pwd)
S=$ROOT/results/_newrev_staging
TS=$(date +%m%d_%H%M%S)
LOG=$ROOT/results/campaign_formals_$TS.log
exec > >(tee -a $LOG) 2>&1
echo "formals launch: $(date '+%F %T')"

VNPZ=$ROOT/results/a1_optim_campaign/a1_v_train_1004_011629/v_model.npz
test -f $VNPZ || { echo "REJECT: 缺 V 模型 $VNPZ"; exit 2; }

# ---- D3 冻结树（10 密封文件 + a1_strong_controls + 运行器；驱动顶层 import a1 模块）----
D3T=$ROOT/results/s3_4_d3_src_frozen
if [ ! -e $D3T ]; then
  mkdir -p $D3T/scripts/evaluation $D3T/scripts/simulation $D3T/scripts/coldchain
  cp $S/scripts/evaluation/run_a1_step2_gate.py $S/scripts/evaluation/run_identity.py \
     $S/scripts/evaluation/scenario_saa.py $S/scripts/evaluation/run_exp_energy_c0.py \
     $S/scripts/evaluation/coldchain_evaluator_a1.py $S/scripts/evaluation/run_exp_reserve.py \
     $S/scripts/evaluation/run_exp_encoder_v3.py $D3T/scripts/evaluation/
  cp $S/scripts/simulation/strict_online_env.py $S/scripts/simulation/a1_strong_controls.py $D3T/scripts/simulation/
  cp $S/scripts/coldchain/coldchain_state.py $S/scripts/coldchain/coldchain_contract.py $D3T/scripts/coldchain/
  cp $S/s3_4_d3_run_frozen.py $D3T/s3_4_d3_run_frozen.py
  {
    echo "# s3_4_d3 frozen $(date '+%F %T')"
    for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
             scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
             scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
             scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
             scripts/simulation/a1_strong_controls.py \
             scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
      echo "$(sha256sum $D3T/$f | cut -d' ' -f1)  $f"
    done
    echo "$(sha256sum $D3T/s3_4_d3_run_frozen.py | cut -d' ' -f1)  s3_4_d3_run_frozen.py"
  } > $D3T/SOURCE_MANIFEST.sha256
  chmod -R a-w $D3T
  echo "D3 frozen tree built"
else
  echo "D3 frozen tree exists（拒绝重建）"
fi

# ---- A1 冻结树（同上 + V npz + a1 运行器）----
A1T=$ROOT/results/a1_src_frozen
if [ ! -e $A1T ]; then
  mkdir -p $A1T/scripts/evaluation $A1T/scripts/simulation $A1T/scripts/coldchain $A1T/models
  cp $D3T/scripts/evaluation/*.py $A1T/scripts/evaluation/
  cp $D3T/scripts/simulation/*.py $A1T/scripts/simulation/
  cp $D3T/scripts/coldchain/*.py $A1T/scripts/coldchain/
  cp $VNPZ $A1T/models/a1_v_model.npz
  cp $S/a1_run_frozen.py $A1T/a1_run_frozen.py
  {
    echo "# a1 frozen $(date '+%F %T')"
    for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
             scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
             scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
             scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
             scripts/simulation/a1_strong_controls.py \
             scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
      echo "$(sha256sum $A1T/$f | cut -d' ' -f1)  $f"
    done
    echo "$(sha256sum $A1T/models/a1_v_model.npz | cut -d' ' -f1)  models/a1_v_model.npz"
    echo "$(sha256sum $A1T/a1_run_frozen.py | cut -d' ' -f1)  a1_run_frozen.py"
  } > $A1T/SOURCE_MANIFEST.sha256
  chmod -R a-w $A1T
  echo "A1 frozen tree built"
else
  echo "A1 frozen tree exists（拒绝重建）"
fi

# ---- 启动：D3 OFF（旧语义基线，先起）、D3 ON、A1 五臂 ----
DON=$ROOT/results/a1_s3_4_d3_on_40_$TS
DOFF=$ROOT/results/a1_s3_4_d3_off_40_$TS
AOUT=$ROOT/results/a1_a1_formal_40_$TS
mkdir -p $DON $DOFF $AOUT

(cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DON/verify_start.txt 2>&1 \
  || { echo "REJECT: D3 冻结树核验失败"; exit 2; }
cp $DON/verify_start.txt $DOFF/verify_start.txt
(cd $A1T && sha256sum -c SOURCE_MANIFEST.sha256) > $AOUT/verify_start.txt 2>&1 \
  || { echo "REJECT: A1 冻结树核验失败"; exit 2; }

tmux new-session -d -s d3_off_40_$TS \
  "cd $D3T && S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DOFF/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode ls --incr-eval --workers 6 --out $DOFF \
     > $DOFF/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $DOFF/run.exit; \
   $PY $ROOT/scripts/evaluation/diag_adjudicate.py $DOFF/gate.json --expected 40 \
     --out $DOFF/adjudication.json > $DOFF/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $DOFF/run.exit; \
   (cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DOFF/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') d3_off done run=\$rc adj=\$arc\""
sleep 30
tmux new-session -d -s d3_on_40_$TS \
  "cd $D3T && S34D3_FROZEN_SRC=$D3T S34D3_EVIDENCE_OUT=$DON/import_evidence.txt \
     $PY $D3T/s3_4_d3_run_frozen.py \
     --gate-instances 40 --energy-pricing marginal --future-policy density \
     --shadow-mode ls --incr-eval --anytime-vote --workers 6 --out $DON \
     > $DON/run.log 2>&1; rc=\$?; echo RUN_EXIT=\$rc > $DON/run.exit; \
   $PY $ROOT/scripts/evaluation/diag_adjudicate.py $DON/gate.json --expected 40 \
     --out $DON/adjudication.json > $DON/adjudication.log 2>&1; arc=\$?; \
   echo ADJUDICATE_EXIT=\$arc >> $DON/run.exit; \
   (cd $D3T && sha256sum -c SOURCE_MANIFEST.sha256) > $DON/verify_end.txt 2>&1; \
   echo \"$(date '+%F %T') d3_on done run=\$rc adj=\$arc\""
sleep 30
tmux new-session -d -s a1_40_$TS \
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
   echo \"$(date '+%F %T') a1_40 done run=\$rc adj=\$arc\""

sleep 3
tmux ls
echo "formals launched: d3_off=$DOFF d3_on=$DON a1=$AOUT (ts=$TS)"
