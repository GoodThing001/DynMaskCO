#!/bin/bash
# 杀掉旧链（编译风暴中的校准 + watcher），重启后处理链（固定形状修复版）
tmux kill-session -t a1post 2>/dev/null
pkill -f diag_i1_calibration.py 2>/dev/null
sleep 3
tmux new-session -d -s a1post bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_watch_then_post.sh
tmux ls
