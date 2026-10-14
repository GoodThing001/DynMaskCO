#!/bin/bash
# 紧急：停掉参与度对照脚本进程（不要碰 F1 的 run_a1_step2_gate）
set -u
pkill -f verify_provider_participation 2>/dev/null
sleep 3
echo "== 自己的参与度脚本进程（应为空）=="
ps -eo pid,pcpu,etime,args | grep verify_provider | grep -v grep | cut -c1-140
echo "== F1 基线进程（必须still在）=="
ps -eo pid,pcpu,etime,args | grep run_a1_step2_gate | grep -v grep | wc -l
echo "== load =="
uptime
echo "== 已完成的 participation 产物 =="
ls -la /home/hzeng/project/MASKCO-Main/CC_Compare/results/a1_compare_table/participation/ 2>/dev/null
