#!/bin/bash
# 独立复核 CO-enriched-ML 适配（重跑其烟测 + 检查权重文件身份）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
COE=../CC_Compare/CO-enriched-ML/a1_accept_v1
PY=/home/hzeng/envs/cc_compare/bin/python
echo "== artifacts =="
ls -la "$COE/artifacts" "$COE/artifacts_a1" 2>&1
echo "== sha256 =="
sha256sum "$COE/artifacts_a1/a1_weights.npz" "$COE/artifacts/official_weights.npz" 2>&1
echo "== PCHGS binary =="
ls -la "$COE/build_pchgs/PCHGS" 2>&1
echo "== smoke（独立重跑）=="
$PY "$COE/smoke_coenrich.py" --artifacts "$COE/artifacts_a1" \
    --pchgs-exe "$PWD/$COE/build_pchgs/PCHGS" --pchgs-seconds 2 2>&1 | tail -45
echo "SMOKE_EXIT=$?"
