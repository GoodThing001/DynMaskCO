#!/bin/bash
# 全实验启动器（2026-10-04 用户指示：现在跑全部实验 + 定点检查 + 总汇报）。
# 隔离源码区方案：绝不修改工作目录 scenario_saa.py/run_a1_step2_gate.py
# （A3 C 批结束时对工作目录做 end-seal 复查，中途替换会破坏其 source_stable）。
# 所有新实验从 results/_newrev_staging（完整新修订源码集 + 运行器）以 staging 为 cwd 运行，
# 相对源码 hash 全部解析到 staging（identity 如实记录新修订），输出目录用绝对路径。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
ROOT=$(pwd)
S=$ROOT/results/_newrev_staging
TS=$(date +%m%d_%H%M%S)
LOG=$ROOT/results/campaign_launch_$TS.log

exec > >(tee -a $LOG) 2>&1
echo "launch: $(date '+%F %T')  campaign-all from staging=$S"

# 0) 组装 staging 完整源码集：8 个未变文件从工作目录复制（只读，不改工作目录）
test -f $S/scripts/evaluation/scenario_saa.py || { echo "REJECT: staging 缺 scenario_saa.py（先上传新文件）"; exit 2; }
mkdir -p $S/scripts/evaluation $S/scripts/simulation $S/scripts/coldchain
cp -n scripts/evaluation/run_identity.py scripts/evaluation/run_exp_energy_c0.py \
      scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
      scripts/evaluation/run_exp_encoder_v3.py $S/scripts/evaluation/
cp -n scripts/simulation/strict_online_env.py $S/scripts/simulation/
cp -n scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py $S/scripts/coldchain/
echo "staging 源码集: $(ls $S/scripts/evaluation | wc -l) eval + $(ls $S/scripts/simulation | wc -l) sim + $(ls $S/scripts/coldchain | wc -l) cc"
sha256sum $S/scripts/evaluation/scenario_saa.py $S/scripts/evaluation/run_a1_step2_gate.py \
           $S/scripts/simulation/a1_strong_controls.py

# 1) 服务器 T 套件（staging cwd，CPU-only，~5-10 min）
tmux new-session -d -s camp_tests \
  "cd $S && $PY scripts/tests/test_a1_saa_correctness.py > $ROOT/results/camp_tests.log 2>&1; \
   $PY scripts/tests/test_a1_strong_controls.py >> $ROOT/results/camp_tests.log 2>&1; \
   $PY scripts/tests/test_s3_6a_softknn.py >> $ROOT/results/camp_tests.log 2>&1; \
   echo T_SUITE_EXIT=\$? > $ROOT/results/camp_tests.exit"
echo "T1: camp_tests launched"

# 2) V 正式训练（60 训练日 × 200 单，8 worker，~10-20 min）→ 完成后链式启动 A1 烟测
VOUT=$ROOT/results/a1_optim_campaign/a1_v_train_$TS
mkdir -p $VOUT
tmux new-session -d -s camp_vtrain \
  "cd $S && $PY scripts/training/train_a1_rollout_v.py \
     --max-days 60 --max-events-per-day 200 --K 10 --h 2.0 \
     --n-orders 200 --n-held-days 4 --workers 8 --seed 42 --out $VOUT \
     > $VOUT/train.log 2>&1; rc=\$?; echo VTRAIN_EXIT=\$rc > $VOUT/run.exit; \
   if [ \$rc -eq 0 ] && [ -f $VOUT/v_model.npz ]; then \
     AOUT=$ROOT/results/a1_a1_smoke_$TS; mkdir -p \$AOUT; \
     tmux new-session -d -s camp_a1smoke \"cd $S && \
       A1_FROZEN_SRC=$S A1_EVIDENCE_OUT=\$AOUT/import_evidence.txt \
       $PY $S/a1_run_frozen.py \
       --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
       --future-policy density --shadow-mode greedy \
       --arms uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout \
       --a1-v-ckpt $VOUT/v_model.npz --a1-h 2.0 --workers 5 --out \$AOUT \
       > \$AOUT/run.log 2>&1; echo A1_SMOKE_EXIT=\\\$? > \$AOUT/run.exit\"; \
     echo \"A1 smoke launched (v_ckpt=$VOUT/v_model.npz)\"; \
   else echo \"REJECT: V 训练失败，A1 烟测不启动\"; fi"
echo "T2: vtrain (+chained A1 smoke) launched"

# 3) S3-6a 烟测（4 臂 2 天 p_c=0，greedy density，~40 min）
SOUT=$ROOT/results/a1_s3_6a_smoke_$TS
mkdir -p $SOUT
tmux new-session -d -s camp_s36a \
  "cd $S && S34D3_FROZEN_SRC=$S S34D3_EVIDENCE_OUT=$SOUT/import_evidence.txt \
     $PY $S/s3_4_d3_run_frozen.py \
     --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
     --future-policy density --shadow-mode greedy \
     --arms uncond_hist,cond_hist,explicit_feat,soft_knn \
     --softknn-mult 1.0 --workers 4 --out $SOUT \
     > $SOUT/run.log 2>&1; echo S36A_SMOKE_EXIT=\$? > $SOUT/run.exit"
echo "T3: s3_6a smoke launched"

# 4) D3 烟测（ON=anytime / OFF=旧语义同修订版，ls+incr，2 天 p_c=0，~3-4h）
DON=$ROOT/results/a1_s3_4_d3_smoke_on_$TS
DOFF=$ROOT/results/a1_s3_4_d3_smoke_off_$TS
mkdir -p $DON $DOFF
tmux new-session -d -s camp_d3 \
  "cd $S && S34D3_FROZEN_SRC=$S S34D3_EVIDENCE_OUT=$DON/import_evidence.txt \
     $PY $S/s3_4_d3_run_frozen.py \
     --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
     --shadow-mode ls --incr-eval --anytime-vote --workers 3 --out $DON \
     > $DON/run.log 2>&1; echo D3_ON_EXIT=\$? > $DON/run.exit; \
   S34D3_FROZEN_SRC=$S S34D3_EVIDENCE_OUT=$DOFF/import_evidence.txt \
     $PY $S/s3_4_d3_run_frozen.py \
     --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
     --shadow-mode ls --incr-eval --workers 3 --out $DOFF \
     > $DOFF/run.log 2>&1; echo D3_OFF_EXIT=\$? > $DOFF/run.exit"
echo "T4: d3 smoke (ON+OFF) launched"

sleep 3
tmux ls
echo "campaign launch complete: tests/vtrain(+A1)/s36a/d3 共 5 会话"
echo "outputs: V=$VOUT  s36a=$SOUT  d3=$DON/$DOFF  A1 smoke (chained)=$ROOT/results/a1_a1_smoke_$TS"
