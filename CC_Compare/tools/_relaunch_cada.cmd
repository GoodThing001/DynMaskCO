@echo off
REM Clear the stale cada.done (from the entmax-missing failure) and re-run the cada arm detached.
del /q C:\Users\zh\gpu3080_logs\cada.done >nul 2>&1
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_gpu3080\_gpu3080_cada.cmd'} | Select-Object -ExpandProperty ProcessId"
