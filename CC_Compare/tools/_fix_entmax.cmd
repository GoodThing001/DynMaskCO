@echo off
REM Install entmax (CaDA's missing dep) and report arm liveness.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===INSTALL_ENTMAX===
D:\anaconda\Scripts\conda.exe run --no-capture-output -n cc_compare python -m pip install --timeout 60 --retries 5 entmax
echo ---- entmax rc=%ERRORLEVEL% ----
D:\anaconda\Scripts\conda.exe run --no-capture-output -n cc_compare python -c "import entmax;print('entmax OK', entmax.__file__)"
echo ===GPU===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ===PYPROCS===
powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count"
echo ===ARMLOG_TIMES===
powershell -NoProfile -Command "Get-ChildItem C:\Users\zh\gpu3080_logs\mvmoe.log,C:\Users\zh\gpu3080_logs\routefinder.log,C:\Users\zh\gpu3080_logs\rrnco.log -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize"
echo ===END===
