@echo off
REM Clean the 3080 leg completely, then launch the SEQUENTIAL driver detached.
set EXT=C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
set ROOT=C:\Users\zh\project\MASKCO-Main
echo ===CLEAR_FAILED_ATTEMPT===
if exist "%EXT%\results\a1_ext_formal2_3080" rmdir /s /q "%EXT%\results\a1_ext_formal2_3080"
del /q C:\Users\zh\gpu3080_logs\mvmoe.done 2>nul
del /q C:\Users\zh\gpu3080_logs\routefinder.done 2>nul
del /q C:\Users\zh\gpu3080_logs\cada.done 2>nul
del /q C:\Users\zh\gpu3080_logs\rrnco.done 2>nul
del /q C:\Users\zh\gpu3080_logs\sequential.done 2>nul
echo ===LAUNCH_SEQUENTIAL===
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c %ROOT%\CC_Compare\tools\_gpu3080_sequential.cmd'} | Select-Object -ExpandProperty ProcessId"
echo ===DONE_MARKER_CHECK===
dir /b C:\Users\zh\gpu3080_logs\*.done 2>nul
echo ===END===
