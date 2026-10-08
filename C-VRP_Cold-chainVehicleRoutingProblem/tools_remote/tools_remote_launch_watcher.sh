#!/bin/bash
# 重启 watcher（训练链已基本完成 → 应尽快触发后处理链）
tmux kill-session -t a1post 2>/dev/null
tmux new-session -d -s a1post bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_watch_then_post.sh
sleep 3
tmux ls
echo ===RCVRP_CONFIG===
cat /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/maskco_train160_clock_random_cvrp_s42/config.json 2>/dev/null | grep -E 'final_loss|frozen_unchanged'
echo ===PS===
ps aux | grep train_maskco_scenario | grep -v grep | wc -l
