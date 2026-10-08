#!/bin/bash
tmux ls
echo ===PS===
ps aux | grep run_a1_step2_gate | grep -v grep | awk '{print $2, $3, $4}'
echo ===PROGRESS===
for f in /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_s3_2rr2_density_40/progress_*.txt; do echo $f $(wc -l < $f); done
echo ===LOG===
tail -n 4 /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_s3_2rr2_density_40.log
echo ===PRETRAIN===
grep -E 'step [0-9]+:|slices:' /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/train160_pretrained.log | tail -n 3
