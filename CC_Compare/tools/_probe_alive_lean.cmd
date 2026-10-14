@echo off
REM Lean liveness probe for the cc_compare torch download.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===NOW===
date /t
time /t
echo ===PIPCACHE===
powershell -NoProfile -Command "$c=Join-Path $env:LOCALAPPDATA 'pip\Cache'; if (Test-Path $c) { '{0:N2} GB' -f ((Get-ChildItem $c -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB) } else { 'no cache dir' }"
echo ===ENVSIZE===
powershell -NoProfile -Command "'{0:N2} GB' -f ((Get-ChildItem D:\anaconda\envs\cc_compare -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB)"
echo ===CPU===
powershell -NoProfile -Command "Get-Process python,conda -ErrorAction SilentlyContinue | Select-Object Id,CPU | Format-Table -HideTableHeaders"
echo ===END===
