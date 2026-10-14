@echo off
REM Force the exact CUDA local version so pip cannot match the installed 2.14.1+cpu.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===cu129===
call conda run -n cc_gpu python -m pip install --dry-run torch==2.14.1+cu129 --index-url https://download.pytorch.org/whl/cu129 > _pi129.txt 2>&1
findstr /I "Would install ERROR No matching Downloading" _pi129.txt
echo ===cu128===
call conda run -n cc_gpu python -m pip install --dry-run torch==2.14.1+cu128 --index-url https://download.pytorch.org/whl/cu128 > _pi128.txt 2>&1
findstr /I "Would install ERROR No matching Downloading" _pi128.txt
echo ===END===
