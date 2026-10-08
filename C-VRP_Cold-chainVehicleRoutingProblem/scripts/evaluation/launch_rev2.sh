#!/bin/bash
# 修订版同版配对重跑（低负载）：reveal + density，各自独立 tmux 会话。
# 成功标记：仅当 Python 退出码 0 且 gate.json 非空可读才写 DONE；否则写 FAIL。
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

run_one() {
  NAME=$1
  EXTRA=$2
  OUT=$3
  tmux new-session -d -s "$NAME" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/miniconda3/envs/MASKCO_env/bin/python scripts/evaluation/run_a1_step2_gate.py --gate-instances 40 --energy-pricing marginal $EXTRA --workers 9 --out $OUT > $OUT.log 2>&1; rc=\$?; if [ \$rc -eq 0 ] && [ -s $OUT/gate.json ]; then echo DONE rc=\$rc > $OUT.done; else echo FAIL rc=\$rc > $OUT.failed; fi"
}

run_one a1c1rr2 "" results/a1_c1rr2_reveal_40
run_one a1s3r2b "--future-policy density" results/a1_s3_2rr2_density_40
sleep 2
tmux ls
