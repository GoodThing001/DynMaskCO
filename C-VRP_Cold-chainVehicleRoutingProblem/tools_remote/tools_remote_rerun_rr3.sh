#!/bin/bash
# 09:09 批次被裁定为版本污染诊断（run_a1_step2_gate.py 启动/结束 hash 不一致）。
# 本脚本 = 不可变隔离目录重跑规格（rr3）：启动前先对 10 个封存文件做 SHA 快照存档，
# 再以同预声明配置（reveal + density）重跑。使用前必须满足：
#   1) density 旧批（a1_s3_2rr2_density_40）已落盘；
#   2) 源码树冻结：两个新批次运行期间禁止任何脚本同步/修改（否则再次污染）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9

SNAP=results/a1_rr3_seal_snapshot.json
{
  echo '{'
  echo '  "snapshot_time": "'$(date +%Y-%m-%dT%H:%M:%S)'",'
  echo '  "files": {'
  FIRST=1
  for f in scripts/evaluation/run_a1_step2_gate.py scripts/evaluation/run_identity.py \
           scripts/evaluation/scenario_saa.py scripts/evaluation/run_exp_energy_c0.py \
           scripts/evaluation/coldchain_evaluator_a1.py scripts/evaluation/run_exp_reserve.py \
           scripts/evaluation/run_exp_encoder_v3.py scripts/simulation/strict_online_env.py \
           scripts/coldchain/coldchain_state.py scripts/coldchain/coldchain_contract.py; do
    h=$(sha256sum "$f" | cut -d' ' -f1)
    if [ $FIRST -eq 1 ]; then FIRST=0; else echo ','; fi
    printf '    "%s": "%s"' "$f" "$h"
  done
  echo ''
  echo '  }'
  echo '}'
} > $SNAP
echo "seal snapshot -> $SNAP"
cat $SNAP

run_one() {
  NAME=$1
  EXTRA=$2
  OUT=$3
  tmux kill-session -t "$NAME" 2>/dev/null
  tmux new-session -d -s "$NAME" \
    "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/miniconda3/envs/MASKCO_env/bin/python scripts/evaluation/run_a1_step2_gate.py --gate-instances 40 --energy-pricing marginal $EXTRA --workers 9 --out $OUT > $OUT.log 2>&1; rc=\$?; if [ \$rc -eq 0 ] && [ -s $OUT/gate.json ]; then echo DONE rc=\$rc > $OUT.done; else echo FAIL rc=\$rc > $OUT.failed; fi"
}

run_one a1c1rr3 "" results/a1_c1rr3_reveal_40
run_one a1s3r3b "--future-policy density" results/a1_s3_2rr3_density_40
sleep 2
tmux ls
