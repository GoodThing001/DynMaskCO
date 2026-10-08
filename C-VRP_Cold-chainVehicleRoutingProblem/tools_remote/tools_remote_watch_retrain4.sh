#!/bin/bash
# 等待四臂重训完成（四个新 checkpoint config.json 落盘且无训练进程）→ 四臂校准
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
for i in $(seq 1 60); do
  B=results/a1_step3
  ALL=1
  for d in maskco_train160_clock_pretrained_s42 maskco_train160_clock_random_cvrp_s42 \
           maskco_train160_partial_pretrained_s42 maskco_train160_partial_random_cvrp_s42; do
    test -f $B/$d/config.json || ALL=0
  done
  RUNNING=$(ps aux | grep train_maskco_scenario | grep -v grep | wc -l)
  if [ "$ALL" -eq 1 ] && [ "$RUNNING" -eq 0 ]; then
    echo "retrain4 done at $(date); launching calibration chain"
    bash tools_remote/tools_remote_calibrate_clock.sh
    bash tools_remote/tools_remote_calibrate_i2.sh
    echo "CALIB4_FINISHED"
    exit 0
  fi
  sleep 300
done
echo "WATCH4_TIMEOUT"
