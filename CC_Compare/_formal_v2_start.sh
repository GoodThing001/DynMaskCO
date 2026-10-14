#!/bin/bash
# 启动 A-v1 正式批次 v2（把基线 + 四条腿放进 tmux，SSH 断开不影响）
set -u
mkdir -p /tmp/xformal_v2
cd /home/hzeng/project/MASKCO-Main || exit 1
tmux kill-session -t xf2_launch 2>/dev/null
tmux new -d -s xf2_launch 'bash CC_Compare/_formal_v2_launch.sh > /tmp/xformal_v2/launch.log 2>&1'
sleep 3
echo "== tmux sessions =="
tmux ls
echo "== launch.log =="
cat /tmp/xformal_v2/launch.log 2>/dev/null | head -40
