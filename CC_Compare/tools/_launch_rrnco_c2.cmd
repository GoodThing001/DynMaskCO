@echo off
REM Launch the 3080 side RRNCO-C2 watcher detached (WMI Create).
set ROOT=C:\Users\zh\project\MASKCO-Main
echo ===LAUNCH_RRNCO_C2===
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_gpu3080_rrnco_c2.ps1'} | Select-Object -ExpandProperty ProcessId"
echo ===MARKERS===
dir /b C:\Users\zh\gpu3080_logs\rrnco_c2.* 2>nul
echo ===END===
