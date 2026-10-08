#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=1003_040454
BASE=results/a1_rr3_robust_20261001_density_40_1002_145359/gate.json
MB=results/a1_a3/a3_train200_random_cvrp_s42/model.bin
OUT=results/a1_a3_online_density_D_20261001_40_$TS
SESS=a3_density_D_20261001_$TS
test -f $MB || exit 2
mkdir $OUT || exit 2
tmux new-session -d -s "$SESS" \
  "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && \
   CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
   XLA_FLAGS=--xla_gpu_autotune_level=0 \
   $PY scripts/evaluation/run_a1_step3_compare.py \
     --model-bin $MB --arm random_cvrp --only maskco --baseline-from $BASE \
     --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
     --gate-seed 20261001 --energy-pricing marginal --future-policy density \
     --workers 2 --out $OUT \
   > $OUT/run.log 2>&1; echo RUN_EXIT=\$? > $OUT/run.exit"
echo LAUNCHED_D_20261001
