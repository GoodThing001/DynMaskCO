@echo off
REM ============================================================================
REM Restart the 3080 GPU leg after fixing scenario_saa.py.
REM
REM Why: the arm gates reported `identity_verified=False` with
REM `problems=['shared_source_sha256_mismatch']` -> `formal_adjudication=False`.
REM Cause: only `scripts/evaluation/scenario_saa.py` differed between 1003
REM (36e08e06..., post-09-30 fix) and the 3080 (4e47b99f..., pre-fix) -- because the
REM original 3080 source push came from the LOCAL workspace, which was itself behind
REM 1003 for that file. The 6 other SHARED_SOURCE_FILES already matched.
REM The 20 other differing files are not used by these four arms
REM (run_a1_external_accept.py / the replanners / all other shared files match).
REM
REM So: the arms that already ran (or are running) sealed the WRONG shared hash and
REM must be re-run from scratch after the file is corrected.
REM ============================================================================
set ROOT=C:\Users\zh\project\MASKCO-Main
set EXT=%ROOT%\C-VRP_Cold-chainVehicleRoutingProblem
echo ===SCENARIO_SAA_HASH_expect_36e08e06===
powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 '%EXT%\scripts\evaluation\scenario_saa.py').Hash"
echo ===KILL_ARMS===
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'cmd.exe') -and ($_.CommandLine -like '*_gpu3080_*' -or $_.CommandLine -like '*run_a1_external_accept*') } | ForEach-Object { Write-Output ('  kill ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 6 /nobreak >nul
echo ===CLEAR_OUTPUTS_AND_MARKERS===
if exist "%EXT%\results\a1_ext_formal2_3080" rmdir /s /q "%EXT%\results\a1_ext_formal2_3080"
del /q C:\Users\zh\gpu3080_logs\mvmoe.done C:\Users\zh\gpu3080_logs\routefinder.done C:\Users\zh\gpu3080_logs\cada.done C:\Users\zh\gpu3080_logs\rrnco.done 2>nul
echo ===RELAUNCH_4_ARMS===
for %%T in (mvmoe routefinder cada rrnco) do powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c %ROOT%\CC_Compare\tools\_gpu3080\_gpu3080_%%T.cmd'} | Select-Object -ExpandProperty ProcessId"
echo ===PYPROCS_NOW===
powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"
echo ===END===
