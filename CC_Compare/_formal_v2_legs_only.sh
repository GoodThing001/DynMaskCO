#!/bin/bash
# ============================================================================
# 只启动 A-v1 v2 的 4 条腿（**不重跑基线**）。
#
# 用途：基线已产出、但腿因 `_formal_v2_launch.sh` 的相对路径 bug 秒退之后的补救
# （2026-10-13：`bash CC_Compare/_formal_v2_*.sh` 在扩展根下不存在 → 4 个 tmux 会话
#  立刻退出、无日志；基线与 base.done 正常，launch.log 还打印了「已启动 tmux」）。
#
# 幂等：已存在 /tmp/xformal_v2/<leg>.done 的腿跳过。
# 用法（服务器任意目录）：bash /home/hzeng/project/MASKCO-Main/CC_Compare/_formal_v2_legs_only.sh
# ============================================================================
set -u
LEGS=/home/hzeng/project/MASKCO-Main/CC_Compare
LOGD=/tmp/xformal_v2
mkdir -p "$LOGD"

for leg in gpu cpuA cpuB cpuC; do
  if [ -f "$LOGD/$leg.done" ]; then
    echo "SKIP $leg（已有 $LOGD/$leg.done）"
    continue
  fi
  tmux kill-session -t "xf2_$leg" 2>/dev/null
  tmux new -d -s "xf2_$leg" "bash $LEGS/_formal_v2_$leg.sh > $LOGD/$leg.log 2>&1"
  echo "STARTED xf2_$leg -> $LOGD/$leg.log"
done

sleep 8
echo "---- 存活的会话 ----"
tmux ls
echo "---- 各腿日志前 3 行 ----"
for leg in gpu cpuA cpuB cpuC; do
  echo "== $leg =="
  head -3 "$LOGD/$leg.log" 2>/dev/null || echo "(无日志)"
done
