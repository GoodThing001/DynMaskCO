#!/bin/bash
tmux kill-session -t a1post4 2>/dev/null
tmux new-session -d -s a1post4 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_watch_retrain4.sh
sleep 3
tmux ls
