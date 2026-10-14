#!/bin/bash
# 核验 MAPT provider 的可行档默认值 + checkpoint 身份（独立于子代理自述）
set -u
cd /home/hzeng/project/MASKCO-Main || exit 1
P=CC_Compare/MAPT_code/a1_accept_v1/mapt_provider.py
echo "== provider 默认档相关行 =="
grep -n "MAPT_PROVIDER_CONFIG\|n_frame_eval\|n_phantom\|cpu_threads\|set_num_threads\|'fast'\|\"fast\"" "$P" | head -30
echo
echo "== checkpoint 身份 =="
sha256sum CC_Compare/MAPT_code/a1_accept_v1/mapt_a1.pt 2>&1
ls -la CC_Compare/MAPT_code/a1_accept_v1/mapt_a1.pt CC_Compare/MAPT_code/a1_accept_v1/mapt_a1_final.pt 2>&1
echo
echo "== marker 内容 =="
head -20 CC_Compare/MAPT_code/a1_accept_v1/RUN_ARM_APPROVED
echo
echo "== provider 注册表 =="
cd C-VRP_Cold-chainVehicleRoutingProblem || exit 1
/home/hzeng/envs/cc_compare/bin/python -c "
import sys; sys.path.insert(0,'scripts/evaluation')
import ordering_providers as op
print(sorted(op._REGISTRY))
"
