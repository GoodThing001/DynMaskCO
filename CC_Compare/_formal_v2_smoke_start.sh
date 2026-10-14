#!/bin/bash
set -u
mkdir -p /tmp/xformal_v2
cd /home/hzeng/project/MASKCO-Main || exit 1
tmux kill-session -t xf2smoke 2>/dev/null
tmux new -d -s xf2smoke 'bash CC_Compare/_formal_v2_smoke.sh > /tmp/xformal_v2/smoke.log 2>&1'
sleep 3
tmux ls
