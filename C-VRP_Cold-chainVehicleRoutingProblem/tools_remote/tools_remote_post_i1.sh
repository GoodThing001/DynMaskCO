#!/bin/bash
# I1 世代后处理链（在 maskco_i1 完成后启动）：留出校准（两臂）→ 步骤 3 驱动烟测
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
bash tools_remote/tools_remote_calibrate_clock.sh
echo "CALIB_CHAIN_EXIT=$?"
bash tools_remote/tools_remote_step3_smoke.sh
echo "POST_I1_DONE"
