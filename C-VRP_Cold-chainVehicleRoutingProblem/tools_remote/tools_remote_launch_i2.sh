#!/bin/bash
# 启动 I2 partial 训练（预声明 2026-09-30 20:30 配置；运行前提 = I1 校准落盘且已复核）
tmux kill-session -t maskco_i2 2>/dev/null
tmux new-session -d -s maskco_i2 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_train_i2_partial.sh
sleep 5
tmux ls
