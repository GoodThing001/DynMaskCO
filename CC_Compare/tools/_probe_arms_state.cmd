@echo off
REM State of the 4 GPU-leg arms on the 3080.
echo ===GPU===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ===PYPROCS===
powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"
echo ===ARMLOGS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.log | Where-Object { $_.Name -notlike 'cc_compare*' } | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize"
echo ===DONEMARKERS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.done -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
echo ===ERRSCAN===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.log | Where-Object { $_.Name -notlike 'cc_compare*' } | ForEach-Object { $n=$_.Name; $c=(Select-String -Path $_.FullName -Pattern 'Traceback|Error|error:' -ErrorAction SilentlyContinue | Measure-Object).Count; '{0}: errmatches={1}' -f $n, $c }"
echo ===END===
