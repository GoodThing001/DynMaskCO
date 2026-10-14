@echo off
REM 3080 GPU leg: cada (cc_compare env), two penalty tiers sequentially (device=cuda)
set PYTHONIOENCODING=utf-8
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == cada START %DATE% %TIME% == > C:\Users\zh\gpu3080_logs\cada.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider cada --provider-ckpt ..\CC_Compare\CaDA\100\result\2024-1121-1355\checkpoint-300.pt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=0" --out results\a1_ext_formal2_3080\cada_0 >> C:\Users\zh\gpu3080_logs\cada.log 2>&1
echo -- cada 0 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\cada.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider cada --provider-ckpt ..\CC_Compare\CaDA\100\result\2024-1121-1355\checkpoint-300.pt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results\a1_ext_formal2_3080\cada_51015 >> C:\Users\zh\gpu3080_logs\cada.log 2>&1
echo -- cada 51015 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\cada.log
echo DONE_cada %DATE% %TIME% >> C:\Users\zh\gpu3080_logs\cada.done
echo == cada END %DATE% %TIME% == >> C:\Users\zh\gpu3080_logs\cada.log
