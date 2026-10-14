#!/bin/bash
# xf2 流水线守护（幂等、只追加、不删任何结果）
#
# 为什么需要它（2026-10-13 实测缺陷）：
#   `_formal_v2_after.sh` 的三段 `wait_for` 用「MARK 里出现 ALLDONE」当完成信号，
#   而子腿脚本开头就 `: > "$MARK"`（截断建文件）、**只有跑到最后一行才追加 ALLDONE**。
#   ⇒ 子腿只要在任何一条 FATAL 早退（缺基线 / 缺权重 / provider 未接线 / 缺 PCHGS 可执行），
#     MARK 就永远没有 ALLDONE，而 `wait_for` 会**白等满配额**：
#     CO-enriched 2400×60s = **40 h**、MAPT 600×60s = **10 h**。
#     更糟的是全程日志看起来像「还在跑」，交付物却永远不落地。
#
# 本守护只做两件事，且都要求**确凿证据**：
#   ① 子腿「tmux 会话已消失」**且**「无该 provider 的 python 进程」**且**「MARK 无 ALLDONE」
#      ⇒ 判定为早退/崩溃，往 MARK 补 `ALLDONE`，让 after.sh 继续走到采集与收口
#      （宁可产出「缺该腿」的表并在日志里大声记录，也不要静默等 40 h）。
#   ② `xf2_after` 会话消失而 after.log 里没有 `收口完成` ⇒ 重新拉起 after.sh（追加写同一日志）。
#      after.sh 本身幂等：已完成的臂有 gate.json 会 SKIP。
#
# 启动（服务器）：
#   tmux new -d -s xf2_guard 'bash /home/hzeng/project/MASKCO-Main/CC_Compare/tools/_xf2_guard.sh >> /tmp/xformal_v2/guard.log 2>&1'
set -u
ROOT=/home/hzeng/project/MASKCO-Main
RUN=/tmp/xformal_v2
AFTER_LOG=$RUN/after.log
mkdir -p "$RUN"

log() { echo "[$(date '+%F %T')] $*" >> "$RUN/guard.log"; }

# 只把「真正想看的那条腿」算作存活：用 [p] 技巧避免匹配到 pgrep 自身
alive_proc() { pgrep -f "[p]rovider $1" >/dev/null 2>&1; }
alive_sess() { tmux has-session -t "$1" 2>/dev/null; }

guard_leg() {  # mark session provider label
  local mark=$1 sess=$2 prov=$3 label=$4
  [ -f "$mark" ] || return 0
  grep -q ALLDONE "$mark" 2>/dev/null && return 0
  alive_sess "$sess" && return 0
  alive_proc "$prov" && return 0
  log "GUARD: $label 会话 $sess 已消失、无 [$prov] 进程、MARK 无 ALLDONE ⇒ 判定早退，补 ALLDONE 让 after 继续"
  {
    echo "GUARD_FORWARDED_NO_ALLDONE $(date '+%F %T')"
    echo "GUARD_REASON: session=$sess gone, no '--provider $prov' process, mark had no ALLDONE"
    echo "ALLDONE"
  } >> "$mark"
}

i=0
while true; do
  i=$((i + 1))
  if [ -f "$AFTER_LOG" ] && grep -q '收口完成' "$AFTER_LOG" 2>/dev/null; then
    log "after.log 已出现「收口完成」⇒ 守护退出（本轮检查 $i 次）"
    exit 0
  fi
  guard_leg "$RUN/coenrich.done" xf2_coe coenrich CO-enriched
  guard_leg "$RUN/mapt.done" xf2_mapt mapt MAPT
  if ! alive_sess xf2_after; then
    log "WARN: xf2_after 会话消失且未收口 ⇒ 重新拉起 after.sh（幂等，追加同一日志）"
    tmux new -d -s xf2_after "bash $ROOT/CC_Compare/_formal_v2_after.sh >> $RUN/after.log 2>&1" \
      || log "ERROR: 重启 after.sh 失败"
  fi
  sleep 300
done
