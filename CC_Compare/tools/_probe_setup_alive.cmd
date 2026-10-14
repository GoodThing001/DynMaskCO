@echo off
REM Is the cc_compare install still alive and making progress?
cd /d C:\Users\zh\project\MASKCO-Main
echo ===LOG===
for %%A in (C:\Users\zh\gpu3080_logs\cc_compare_setup.log) do echo size=%%~zA modified=%%~tA
echo ===PROCS===
powershell -NoProfile -Command "Get-Process python,conda,pip -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WS | Format-Table -AutoSize"
echo ===PIPTEMP===
powershell -NoProfile -Command "Get-ChildItem $env:TEMP -Filter 'pip-*' -Directory -ErrorAction SilentlyContinue | ForEach-Object { '{0} {1:N1} MB' -f $_.Name, ((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1MB) }"
echo ===ENVSIZE===
powershell -NoProfile -Command "'{0:N2} GB' -f ((Get-ChildItem D:\anaconda\envs\cc_compare -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)"
echo ===END===
