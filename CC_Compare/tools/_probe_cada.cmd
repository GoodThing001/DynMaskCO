@echo off
REM Diagnose the cada arm failure.
echo ===GPU===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo ===PYPROCS===
powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"
echo ===CADA_LOG===
powershell -NoProfile -Command "Get-Content C:\Users\zh\gpu3080_logs\cada.log -Tail 40"
echo ===DONE_MARKERS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.done -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
echo ===OUTDIRS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_ext_formal2_3080 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
echo ===END===
