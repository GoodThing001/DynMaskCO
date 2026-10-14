#!/bin/bash
# 挂起「F1 完成后收口」编排（幂等；可重复执行）
set -u
mkdir -p /tmp/xformal_v2
cd /home/hzeng/project/MASKCO-Main || exit 1
tmux kill-session -t xf2_after 2>/dev/null
tmux new -d -s xf2_after 'bash CC_Compare/_formal_v2_after.sh > /tmp/xformal_v2/after.log 2>&1'
sleep 3
tmux ls
echo "---- after.log ----"
tail -5 /tmp/xformal_v2/after.log 2>/dev/null
