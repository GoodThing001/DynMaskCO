@echo off
REM 3080 GPU leg: mvmoe (pytorch env), two penalty tiers sequentially (device=cuda)
set PYTHONIOENCODING=utf-8
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == mvmoe START %DATE% %TIME% == > C:\Users\zh\gpu3080_logs\mvmoe.log
conda run -n pytorch python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider mvmoe --provider-ckpt ..\CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=0" --out results\a1_ext_formal2_3080\mvmoe_0 >> C:\Users\zh\gpu3080_logs\mvmoe.log 2>&1
echo -- mvmoe 0 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\mvmoe.log
conda run -n pytorch python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider mvmoe --provider-ckpt ..\CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results\a1_ext_formal2_3080\mvmoe_51015 >> C:\Users\zh\gpu3080_logs\mvmoe.log 2>&1
echo -- mvmoe 51015 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\mvmoe.log
echo DONE_mvmoe %DATE% %TIME% >> C:\Users\zh\gpu3080_logs\mvmoe.done
echo == mvmoe END %DATE% %TIME% == >> C:\Users\zh\gpu3080_logs\mvmoe.log
