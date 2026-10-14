@echo off
REM 3080 GPU leg: rrnco (cc_compare env), two penalty tiers sequentially (device=cuda)
set PYTHONIOENCODING=utf-8
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == rrnco START %DATE% %TIME% == > C:\Users\zh\gpu3080_logs\rrnco.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver rrnco --rrnco-ckpt ..\CC_Compare\RRNCO\checkpoints\rcvrptw\epoch_199.ckpt --rrnco-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=0" --out results\a1_ext_formal2_3080\rrnco_0 >> C:\Users\zh\gpu3080_logs\rrnco.log 2>&1
echo -- rrnco 0 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\rrnco.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver rrnco --rrnco-ckpt ..\CC_Compare\RRNCO\checkpoints\rcvrptw\epoch_199.ckpt --rrnco-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results\a1_ext_formal2_3080\rrnco_51015 >> C:\Users\zh\gpu3080_logs\rrnco.log 2>&1
echo -- rrnco 51015 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\rrnco.log
echo DONE_rrnco %DATE% %TIME% >> C:\Users\zh\gpu3080_logs\rrnco.done
echo == rrnco END %DATE% %TIME% == >> C:\Users\zh\gpu3080_logs\rrnco.log
