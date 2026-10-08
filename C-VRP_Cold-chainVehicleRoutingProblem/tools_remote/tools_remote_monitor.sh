#!/bin/bash
echo ===STEPSEL===
ls -la /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/stepsel_pretrained/step_selection.json 2>/dev/null
grep -E 'CV_TRAIN_EXIT|SELECT_EXIT|STEPSEL_DONE' /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3/stepsel_pretrained.select.log 2>/dev/null
ps aux | grep -E 'train_maskco_scenario|diag_select_steps' | grep -v grep | awk '{print $2, $3, $13}'
echo ===RR3===
ls -d /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_rr3_* 2>/dev/null
tmux ls 2>/dev/null
uptime
