@echo off
REM Decisive: has the big torch wheel landed, and has install started?
echo ===SITEPKG_TORCH===
if exist D:\anaconda\envs\cc_compare\Lib\site-packages\torch (echo TORCH_DIR_EXISTS) else (echo TORCH_DIR_ABSENT)
echo ===BIGGEST_CACHED_WHEELS===
powershell -NoProfile -Command "Get-ChildItem (Join-Path $env:LOCALAPPDATA 'pip\Cache') -Recurse -File -Filter '*.whl' -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 4 | ForEach-Object { '{0:N0} MB  {1}' -f ($_.Length/1MB), $_.Name }"
echo ===NEWEST_CACHE_FILES===
powershell -NoProfile -Command "Get-ChildItem (Join-Path $env:LOCALAPPDATA 'pip\Cache') -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | ForEach-Object { '{0}  {1:N1} MB' -f $_.LastWriteTime.ToString('HH:mm:ss'), ($_.Length/1MB) }"
echo ===END===
