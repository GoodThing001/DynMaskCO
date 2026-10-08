#!/bin/bash
# A1 臂 2 V 模型正式训练（服务器；A3 落盘后执行）。训练日因果回放 ≥60 日 × 200 单。
# 前置：新代码已上传（train_a1_rollout_v.py / a1_strong_controls.py / scenario_saa.py）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
DAYS="${1:-60}"
TS=$(date +%m%d_%H%M%S)
OUT=results/a1_optim_campaign/a1_v_train_$TS
mkdir -p $OUT || exit 2
{
  echo "launch: $(date '+%F %T')  A1 V 正式训练 days=$DAYS"
  sha256sum scripts/training/train_a1_rollout_v.py scripts/simulation/a1_strong_controls.py \
            scripts/evaluation/scenario_saa.py scripts/simulation/strict_online_env.py
  uptime
} > $OUT/resource.txt
$PY scripts/training/train_a1_rollout_v.py \
  --max-days $DAYS --max-events-per-day 200 --K 10 --h 2.0 \
  --n-orders 200 --n-held-days 4 --workers 8 --seed 42 --out $OUT \
  > $OUT/train.log 2>&1
echo "A1_V_TRAIN_EXIT=$?" > $OUT/run.exit
tail -60 $OUT/train.log
echo "A1 V train done: $OUT"
