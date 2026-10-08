#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
$PY -c "import sys; sys.path.insert(0,'scripts/tests'); import test_a1_saa_correctness as t; sys.exit(0 if t.test_t31_saved_weights_are_trained() else 1)" 2>&1 | tail -n 12
echo "T31_EXIT=$?"
