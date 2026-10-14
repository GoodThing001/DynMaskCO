#!/bin/bash
# 独立复核 L2D / PIP 的资产与判死依据（重跑子代理的 probe 脚本）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
echo "########## L2D probe ##########"
$PY "$PWD/../CC_Compare/Learning-to-Delegate/a1_accept_v1/probe_l2d_weights.py" 2>&1 | tail -30
echo "L2D_EXIT=$?"
echo "########## PIP probe ##########"
$PY "$PWD/../CC_Compare/PIP-constraint/a1_accept_v1/probe_pip_assets.py" 2>&1 | tail -30
echo "PIP_EXIT=$?"
