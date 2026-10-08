#!/bin/bash
# rr3 就绪度复验：10 封存文件 hash + 最近 24h 有无变动
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
echo "now: $(date '+%F %T')"
for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
         scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
         scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
         scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
         scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
  echo "$f $(sha256sum $f | cut -d' ' -f1) $(stat -c %y $f | cut -d'.' -f1)"
done
