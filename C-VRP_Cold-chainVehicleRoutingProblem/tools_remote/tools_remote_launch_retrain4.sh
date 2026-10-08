#!/bin/bash
# 启动四臂重训（权重保存 bug 修复后）+ 完成后自动重跑四臂校准
tmux kill-session -t retrain4 2>/dev/null
tmux new-session -d -s retrain4 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_retrain4.sh
sleep 5
tmux ls
