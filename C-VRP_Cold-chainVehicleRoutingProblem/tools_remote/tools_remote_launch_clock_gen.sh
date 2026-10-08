#!/bin/bash
# 启动 I1 世代训练会话（GPU1；杀旧会话）
tmux kill-session -t maskco_i1 2>/dev/null
tmux new-session -d -s maskco_i1 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_train_clock_gen.sh
sleep 6
tmux ls
