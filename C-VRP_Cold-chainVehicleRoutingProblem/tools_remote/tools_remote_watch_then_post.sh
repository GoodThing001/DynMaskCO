#!/bin/bash
# 等待 I1 训练链完成（两臂 checkpoint config.json 落盘且 5 分钟内无新训练进程）后自动跑后处理链
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
for i in $(seq 1 60); do
  RC=results/a1_step3/maskco_train160_clock_random_cvrp_s42/config.json
  PRE=results/a1_step3/maskco_train160_clock_pretrained_s42/config.json
  RUNNING=$(ps aux | grep train_maskco_scenario | grep -v grep | wc -l)
  if [ -f "$RC" ] && [ -f "$PRE" ] && [ "$RUNNING" -eq 0 ]; then
    echo "I1 chain complete at $(date); launching post-I1"
    bash tools_remote/tools_remote_post_i1.sh
    echo "POST_I1_FINISHED"
    exit 0
  fi
  sleep 300
done
echo "WATCH_TIMEOUT"
