#!/bin/bash
# 校准链预检：4 天留出切片验证 v2 checkpoint 加载 + 世代校验 + 时钟条件前向
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python scripts/evaluation/diag_i1_calibration.py \
  --model-bin results/a1_step3/maskco_train160_clock_pretrained_s42/model.bin \
  --arm pretrained --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
  --history-days 200 --day-start 160 --day-end 164 --cuts-per-day 2 \
  --out results/a1_step3/_calib_precheck > /tmp/calib_precheck.log 2>&1
echo EXIT=$?
grep -E 'generation|slices|marginal baseline|n_eff|bias_mean|nll_gain|pit_mean' /tmp/calib_precheck.log | head -n 10
tail -n 5 /tmp/calib_precheck.log
