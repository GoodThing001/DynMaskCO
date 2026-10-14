#!/bin/bash
# 只读：检查 F1 新基线逐日进度里有没有异常标记（超时/失败），判断刚才的负载尖峰是否污染配对基线
set -u
EXT=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
cd "$EXT" || exit 1
echo "== 每个 combo 的进度文件：行数 / 最后一行 / 非 ok 行 =="
for f in results/a1_c1fresh_reveal_40/progress_*.txt; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  last=$(tail -1 "$f")
  bad=$(grep -vc ' ok$' "$f" || true)
  printf '%-46s lines=%-4s last="%s" non_ok=%s\n' "$(basename "$f")" "$n" "$last" "$bad"
done
echo "== 任何非 ok 行原文（应为空）=="
grep -h -v ' ok$' results/a1_c1fresh_reveal_40/progress_*.txt 2>/dev/null | head -20
echo "== 当前时间 / load =="
date -u; uptime
