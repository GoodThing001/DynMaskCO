@echo off
REM Kill the current resume tree and launch setup3 (live-streaming) detached.
taskkill /F /T /PID 3116 >nul 2>&1
timeout /t 3 /nobreak >nul
echo --- remaining python/conda ---
powershell -NoProfile -Command "Get-Process python,conda -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | Format-Table -HideTableHeaders"
echo --- launching setup3 ---
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_setup_cc_compare3.cmd'} | Select-Object -ExpandProperty ProcessId"
