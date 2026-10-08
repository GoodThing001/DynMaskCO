#!/bin/bash
# 启动 maskco160 tmux 训练会话（幂等：先杀旧会话）
tmux kill-session -t maskco160 2>/dev/null
tmux new-session -d -s maskco160 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_train_maskco_160.sh
sleep 8
tmux ls
sleep 2
tail -n 8 /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/train160_pretrained.log 2>/dev/null || echo no_log_yet
