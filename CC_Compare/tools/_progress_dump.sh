#!/bin/bash
# 打印基线各 combo 的逐日耗时。
# 注意：progress_*.txt 每行是「<day> <当日秒数> ok」——第 2 列是**当日墙钟**，不是累计值
# （累计值可从 c1rr2 gate 的 elapsed_total_s 反推：22735s / 120 个 (p_c×arm×day) ≈ 189s/日）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
D=results/a1_c1fresh_reveal_40
for f in "$D"/progress_*.txt; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  avg=$(awk '{s+=$2} END {if (NR>0) printf "%.0f", s/NR; else print "0"}' "$f")
  mx=$(awk 'BEGIN{m=0} {if ($2>m) m=$2} END{printf "%.0f", m}' "$f")
  mn=$(awk 'BEGIN{m=1e18} {if ($2<m) m=$2} END{printf "%.0f", m}' "$f")
  echo "$(basename "$f"): 天数=$n  当日耗时 min/avg/max = ${mn}/${avg}/${mx} 秒"
done
echo
echo "参考：c1rr2 的 cond_hist elapsed_total_s=22735s（40 天 × 3 档）≈ 189 秒/日"
