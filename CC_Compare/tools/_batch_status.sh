#!/bin/bash
# 正式批次 v2 状态巡检（只读）
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
echo "== time =="; date -u
echo "== tmux =="; tmux ls 2>&1
echo "== done markers (/tmp/xformal_v2/*.done) =="
for f in /tmp/xformal_v2/*.done; do
  [ -f "$f" ] && echo "--- $f" && cat "$f"
done
echo "== launch.log tail =="; tail -4 /tmp/xformal_v2/launch.log 2>/dev/null
echo "== smoke.log tail =="; tail -30 /tmp/xformal_v2/smoke.log 2>/dev/null
echo "== running python =="
ps -eo pid,etime,pcpu,args | grep -E 'run_a1_step2_gate|run_a1_external_accept|run_a1_downstream' | grep -v grep | cut -c1-150
echo "== baseline per-combo last progress =="
cd "$EXT" || exit 1
for f in results/a1_c1fresh_reveal_40/progress_*.txt; do
  [ -f "$f" ] && echo -n "$(basename "$f"): " && tail -1 "$f"
done
echo "== smoke dir =="
ls -la results/a1_c1smoke2_reveal 2>&1 | tail -12
echo "== formal2 arms so far =="
ls results/a1_ext_formal2 2>/dev/null
# ---- 门禁文件完整性（2026-10-13 新增）----------------------------------------- #
# 动机：本批发生过一次真实事故——错误的上传把 1003 上 8 个 GPU gate 里的 7 个
# **截断成 0 字节**（见 README §13.3 缺陷 I 的后续）。当时是等 check_gpu_device.py
# 读到空 JSON **抛异常**才暴露的。这段把它变成**一眼可见**：列出 8 个 GPU gate 的字节数，
# 并直接把 0 字节/过小（<8 KB）的文件标成 **BAD**。巡检时先看这一段。
echo "== GPU gate 完整性（8 个臂；BAD = 0 字节/过小 ⇒ 必须重传）=="
GPU8="mvmoe_0 mvmoe_51015 routefinder_0 routefinder_51015 cada_0 cada_51015 rrnco_0 rrnco_51015"
bad=0
for a in $GPU8; do
  f="results/a1_ext_formal2/$a/gate.json"
  if [ ! -f "$f" ]; then
    printf '  %-18s MISSING\n' "$a"; bad=$((bad+1)); continue
  fi
  sz=$(stat -c %s "$f")
  tag="OK"
  if [ "$sz" -lt 8192 ]; then tag="**BAD**"; bad=$((bad+1)); fi
  printf '  %-18s %8s B  %s\n' "$a" "$sz" "$tag"
done
echo "  --> GPU gate 坏/缺：$bad 个（应为 0）"
echo "== 全部 formal2 gate 中的 0 字节文件（应为空）=="
find results/a1_ext_formal2 -name gate.json -size -8192c 2>/dev/null
echo "== uptime / gpu =="; uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
