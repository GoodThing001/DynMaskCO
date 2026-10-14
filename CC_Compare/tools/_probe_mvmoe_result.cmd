@echo off
REM Verify the first completed GPU arm (mvmoe, both tiers) on the 3080.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===GATES===
dir /b C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_ext_formal2_3080\mvmoe_0\gate.json C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_ext_formal2_3080\mvmoe_51015\gate.json 2>nul
echo ===CHECK_mvmoe_0===
D:\anaconda\envs\cc_compare\python.exe CC_Compare\tools\check_arm_gate.py C-VRP_Cold-chainVehicleRoutingProblem\results\a1_ext_formal2_3080\mvmoe_0\gate.json
echo ===CHECK_mvmoe_51015===
D:\anaconda\envs\cc_compare\python.exe CC_Compare\tools\check_arm_gate.py C-VRP_Cold-chainVehicleRoutingProblem\results\a1_ext_formal2_3080\mvmoe_51015\gate.json
echo ===END===
