#!/bin/bash
# S3-3 烟测（预声明：{5,10} 元/kWh 各 2 天 p_c=0，仅正确性/价格尺度；非证据）
# 运行前提（路线图）：rr3 formal 未过门且执行端已按序推进到 S3-3；源码树冻结。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

for OP in 5 10; do
  OUT=results/a1_s3_3_smoke_op${OP}
  $PY scripts/evaluation/run_a1_step2_gate.py \
    --gate-instances 2 --penalty p_c=0 --energy-pricing marginal \
    --overage-price $OP --workers 6 --out $OUT > $OUT.log 2>&1
  echo "S3_3_SMOKE_OP${OP}_EXIT=$?"
done
echo "S3_3_SMOKE_DONE"
