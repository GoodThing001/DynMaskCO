#!/bin/bash
# Q-04 步骤3 身份封存：服务器侧 T24 + 全量回归（只读，不触碰 run_identity.py）
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
$PY -c "import sys; sys.path.insert(0,'scripts/tests'); import test_a1_saa_correctness as t; sys.exit(0 if t.test_t24_step3_identity_seal() else 1)"
echo "T24_EXIT=$?"
$PY scripts/tests/test_a1_saa_correctness.py
echo "SUITE_EXIT=$?"
