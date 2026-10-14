@echo off
REM Where is the 2.70 GB, and is anything still growing?
echo ===TOP5_LARGEST_IN_CACHE===
powershell -NoProfile -Command "Get-ChildItem (Join-Path $env:LOCALAPPDATA 'pip\Cache') -Recurse -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 5 | ForEach-Object { '{0:N0} MB  {1}  {2}' -f ($_.Length/1MB), $_.LastWriteTime.ToString('HH:mm:ss'), $_.Name }"
echo ===UNPACK_DIRS===
powershell -NoProfile -Command "Get-ChildItem $env:TEMP -Filter 'pip-*' -Directory -ErrorAction SilentlyContinue | ForEach-Object { '{0}  {1:N1} MB  {2}' -f $_.LastWriteTime.ToString('HH:mm:ss'), ((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1MB), $_.Name }"
echo ===TEMP_TOP3===
powershell -NoProfile -Command "Get-ChildItem $env:TEMP -Recurse -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 3 | ForEach-Object { '{0:N0} MB  {1}  {2}' -f ($_.Length/1MB), $_.LastWriteTime.ToString('HH:mm:ss'), $_.Name }"
echo ===END===
