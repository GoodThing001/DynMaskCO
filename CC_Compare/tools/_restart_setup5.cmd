@echo off
REM Kill setup4 tree and launch setup5 (curl from download.pytorch.org) detached.
taskkill /F /T /PID 13156 >nul 2>&1
timeout /t 3 /nobreak >nul
echo --- remaining python/conda/curl ---
powershell -NoProfile -Command "Get-Process python,conda,curl -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | Format-Table -HideTableHeaders"
echo --- launching setup5 ---
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_setup_cc_compare5.cmd'} | Select-Object -ExpandProperty ProcessId"
