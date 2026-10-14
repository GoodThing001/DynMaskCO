#!/bin/bash
set -u
tmux kill-session -t xf2_cp 2>/dev/null
tmux new -d -s xf2_cp 'bash /home/hzeng/project/MASKCO-Main/CC_Compare/tools/_probe_coenrich_pchgs.sh'
sleep 3
tmux ls
