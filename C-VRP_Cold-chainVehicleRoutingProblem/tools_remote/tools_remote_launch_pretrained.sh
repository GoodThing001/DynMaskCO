#!/bin/bash
# 重启 pretrained+random_cvrp 训练会话（GPU0；杀旧会话）
tmux kill-session -t maskco_pre 2>/dev/null
tmux new-session -d -s maskco_pre bash /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/tools_remote/tools_remote_train_pretrained_only.sh
sleep 6
tmux ls
echo ====RANDOM_MLP====
grep -c 'step ' /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/train160_random.log 2>/dev/null
tail -n 2 /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/train160_random.log 2>/dev/null
echo ====BATCH====
ls /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1rr2_reveal_40/gate.json /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_s3_2rr2_density_40/gate.json 2>/dev/null
for f in /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1rr2_reveal_40/progress_p_c=0_cond_hist.txt /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_s3_2rr2_density_40/progress_p_c=0_cond_hist.txt; do test -f $f && echo $f $(wc -l < $f); done
uptime
