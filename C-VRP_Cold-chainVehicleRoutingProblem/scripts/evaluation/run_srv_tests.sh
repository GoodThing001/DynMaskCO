#!/bin/bash
# 服务器端 T1–T20 + 因果测试（写入日志，无引号嵌套）
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python scripts/tests/test_a1_saa_correctness.py > results/t20test.log 2>&1
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python scripts/tests/test_a1_step2_causal.py >> results/t20test.log 2>&1
echo DONE > results/t20test.done
