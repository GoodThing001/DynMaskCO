@echo off
REM Per-process IO counters: sample twice ~45s apart to see if pip is actually transferring.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===SAMPLE1===
powershell -NoProfile -Command "Get-CimInstance Win32_PerfRawData_PerfProc_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'python|conda' } | ForEach-Object { '{0} {1} read={2} write={3}' -f $_.IDProcess, $_.Name, $_.IOReadBytes, $_.IOWriteBytes }"
timeout /t 45 /nobreak > nul
echo ===SAMPLE2===
powershell -NoProfile -Command "Get-CimInstance Win32_PerfRawData_PerfProc_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'python|conda' } | ForEach-Object { '{0} {1} read={2} write={3}' -f $_.IDProcess, $_.Name, $_.IOReadBytes, $_.IOWriteBytes }"
echo ===END===
