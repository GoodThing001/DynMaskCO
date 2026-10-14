@echo off
REM Measure real throughput from the 3080 to download.pytorch.org (bounded 60 MB range request).
set URL=https://download.pytorch.org/whl/cu128/torch-2.11.0%%2Bcu128-cp311-cp311-win_amd64.whl
echo ===CURL_RANGE_60MB===
curl.exe -s -o NUL -r 0-62914559 -w "http=%%{http_code} size=%%{size_download} time=%%{time_total}s speed=%%{speed_download}B/s" "%URL%"
echo.
echo ===TMP_NEWEST===
powershell -NoProfile -Command "Get-ChildItem $env:TEMP -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 4 | ForEach-Object { '{0:N1} MB  {1}  {2}' -f ($_.Length/1MB), $_.LastWriteTime.ToString('HH:mm:ss'), $_.Name }"
echo ===SETUP3_TAIL===
powershell -NoProfile -Command "Get-Content C:\Users\zh\gpu3080_logs\cc_compare_setup3.log -Tail 3"
echo ===END===
