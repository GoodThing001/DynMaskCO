@echo off
REM Is a Chinese PyTorch-wheel mirror faster/reachable for the same wheel?
set F=torch-2.11.0%%2Bcu128-cp311-cp311-win_amd64.whl
set TUNA=https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128/%F%
set ALI=https://mirrors.aliyun.com/pytorch-wheels/cu128/%F%
echo ===TUNA_HEAD===
curl.exe -s -I -m 30 "%TUNA%" | findstr /I "HTTP/ content-length"
echo ===TUNA_RANGE_60MB===
curl.exe -s -o NUL -r 0-62914559 -m 120 -w "http=%%{http_code} size=%%{size_download} time=%%{time_total}s speed=%%{speed_download}B/s" "%TUNA%"
echo.
echo ===ALI_HEAD===
curl.exe -s -I -m 30 "%ALI%" | findstr /I "HTTP/ content-length"
echo ===END===
