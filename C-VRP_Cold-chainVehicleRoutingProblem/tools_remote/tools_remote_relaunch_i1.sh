#!/bin/bash
# 重训 I1 时钟世代（最终 trainer：A-07 hash 修复 + 固定批长 + autotune_level=0）
# 旧 artifact-hash 版本改名归档（不删除）
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
B=results/a1_step3
for d in maskco_train160_clock_pretrained_s42 maskco_train160_clock_random_cvrp_s42; do
  if [ -d $B/$d ]; then
    mv $B/$d $B/${d}_artifact_hash_v1
    echo "archived $d -> ${d}_artifact_hash_v1"
  fi
done
tmux kill-session -t maskco_i1 2>/dev/null
tmux new-session -d -s maskco_i1 bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_train_clock_gen.sh
sleep 5
tmux ls
