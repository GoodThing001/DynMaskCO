@echo off
REM Kill the stalled cc_compare setup tree, then relaunch the verbose resume detached.
taskkill /F /T /PID 19932 >nul 2>&1
timeout /t 3 /nobreak >nul
del /q C:\Users\zh\gpu3080_logs\cc_compare_setup.done >nul 2>&1
echo --- remaining python/conda ---
powershell -NoProfile -Command "Get-Process python,conda -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | Format-Table -HideTableHeaders"
echo --- launching resume ---
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_setup_cc_compare_resume.cmd'} | Select-Object -ExpandProperty ProcessId"
