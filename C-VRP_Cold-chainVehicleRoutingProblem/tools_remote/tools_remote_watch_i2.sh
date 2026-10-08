#!/bin/bash
# 等待 I2 partial 训练完成（两臂 config.json 落盘且无训练进程）→ 自动跑 I2 留出校准
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
for i in $(seq 1 60); do
  RC=results/a1_step3/maskco_train160_partial_random_cvrp_s42/config.json
  PRE=results/a1_step3/maskco_train160_partial_pretrained_s42/config.json
  RUNNING=$(ps aux | grep train_maskco_scenario | grep -v grep | wc -l)
  if [ -f "$RC" ] && [ -f "$PRE" ] && [ "$RUNNING" -eq 0 ]; then
    echo "I2 training complete at $(date); launching I2 calibration"
    bash tools_remote/tools_remote_calibrate_i2.sh
    echo "POST_I2_FINISHED"
    exit 0
  fi
  sleep 300
done
echo "WATCH2_TIMEOUT"
