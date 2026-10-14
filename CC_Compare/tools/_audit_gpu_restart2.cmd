@echo off
REM Lean, crash-free audit of the 3080 GPU leg restart.
set EXT=C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo ===PYPROC_COUNT===
powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"
echo ===LOGS_BY_TIME===
dir /b /o-d C:\Users\zh\gpu3080_logs\*.log
echo ===LOG_SIZES===
for %%F in (C:\Users\zh\gpu3080_logs\mvmoe.log C:\Users\zh\gpu3080_logs\routefinder.log C:\Users\zh\gpu3080_logs\cada.log C:\Users\zh\gpu3080_logs\rrnco.log) do @if exist %%F (for %%A in (%%F) do @echo %%~nxA %%~zA %%~tA) else (echo %%~nxF MISSING)
echo ===DONEMARKERS===
dir /b C:\Users\zh\gpu3080_logs\*.done 2>nul
echo ===OUTDIRS===
dir /b "%EXT%\results\a1_ext_formal2_3080" 2>nul
echo ===GPU===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo ===END===
