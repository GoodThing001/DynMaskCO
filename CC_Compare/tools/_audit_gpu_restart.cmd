@echo off
REM Post-restart audit of the 3080 GPU leg.
set ROOT=C:\Users\zh\project\MASKCO-Main
set EXT=%ROOT%\C-VRP_Cold-chainVehicleRoutingProblem
echo ===SCENARIO_SAA_HASH_expect_36e08e06===
powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 '%EXT%\scripts\evaluation\scenario_saa.py').Hash"
echo ===ARM_PROCS===
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_a1_external_accept*' } | ForEach-Object { '  ' + $_.ProcessId + ' ' + $_.Name + ' :: ' + ($_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length))) }"
echo ===ARM_PROC_COUNT===
powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_a1_external_accept*' } | Measure-Object).Count"
echo ===LOGS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.log | Where-Object { $_.Name -notlike 'cc_compare*' } | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize"
echo ===DONEMARKERS===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\*.done -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
echo ===OUTDIRS===
powershell -NoProfile -Command "Get-ChildItem '%EXT%\results\a1_ext_formal2_3080' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
echo ===GPU===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo ===END===
