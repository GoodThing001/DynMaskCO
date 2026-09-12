#!/bin/bash
# Week 1 v6: 快速修复版 - 检查数据文件并运行实验

cd /home/hzeng/project/MASKCO-Main
conda activate MASKCO_env

echo "=== 检查数据文件 ==="
ls -lh C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_*_test.npz | head -10

echo ""
echo "=== 检查 checkpoint ==="
ls -lh C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/causal_v1/phase3c/step50000.ckpt

echo ""
echo "请确认上述文件存在，然后运行："
echo "bash C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_week1_constraint_cutting_v2.sh smoke"
