@echo off
REM 3080 GPU leg: routefinder (cc_compare env), two penalty tiers sequentially (device=cuda)
set PYTHONIOENCODING=utf-8
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == routefinder START %DATE% %TIME% == > C:\Users\zh\gpu3080_logs\routefinder.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider routefinder --provider-ckpt ..\CC_Compare\RouteFinder\checkpoints\100\rf-transformer.ckpt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=0" --out results\a1_ext_formal2_3080\routefinder_0 >> C:\Users\zh\gpu3080_logs\routefinder.log 2>&1
echo -- routefinder 0 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\routefinder.log
conda run -n cc_compare python scripts\evaluation\run_a1_external_accept.py --solver ordering --provider routefinder --provider-ckpt ..\CC_Compare\RouteFinder\checkpoints\100\rf-transformer.ckpt --provider-device cuda --baseline-from results\a1_c1fresh_reveal_40\gate.json --dev-instances 40 --workers 4 --penalty "p_c=(5,10,15)" --out results\a1_ext_formal2_3080\routefinder_51015 >> C:\Users\zh\gpu3080_logs\routefinder.log 2>&1
echo -- routefinder 51015 rc=%ERRORLEVEL% -- >> C:\Users\zh\gpu3080_logs\routefinder.log
echo DONE_routefinder %DATE% %TIME% >> C:\Users\zh\gpu3080_logs\routefinder.done
echo == routefinder END %DATE% %TIME% == >> C:\Users\zh\gpu3080_logs\routefinder.log
