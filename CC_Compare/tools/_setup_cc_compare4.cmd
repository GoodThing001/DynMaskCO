@echo off
REM ============================================================================
REM Definitive cc_compare install, bypassing the stalled single-stream download.
REM
REM Why: pip's own download of torch-2.11.0+cu128 (2753.1 MB) sat at 0 bytes for
REM minutes even though a bounded range request to download.pytorch.org measured
REM 5.5 MB/s. mirrors.aliyun.com/pytorch-wheels carries the identical wheel
REM (Content-Length 2753148611 == 2753.1 MB), so we fetch it with curl using
REM -C - (resume) inside a retry loop, then install from the local file.
REM ============================================================================
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\gpu3080_logs\cc_compare_setup4.log
set CONDA=D:\anaconda\Scripts\conda.exe
set ROOT=C:\Users\zh\project\MASKCO-Main
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
set WHEELDIR=D:\anaconda\envs\cc_compare\_wheels
if not exist "%WHEELDIR%" mkdir "%WHEELDIR%"
set WHL=%WHEELDIR%\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl
set TURL=https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%%2Bcu128-cp311-cp311-win_amd64.whl
set WANT=2753148611
echo == SETUP4 START %DATE% %TIME% == > %LOG%

set /a TRY=0
:dl
set /a TRY+=1
echo ---- curl attempt !TRY! %TIME% ---- >> %LOG%
curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors --speed-time 60 --speed-limit 20000 -o "%WHL%" "%TURL%" >> %LOG% 2>&1
set GOT=0
for %%A in ("%WHL%") do set GOT=%%~zA
echo ---- attempt !TRY! got=!GOT! want=%WANT% ---- >> %LOG%
if "!GOT!"=="%WANT%" goto dlok
if !TRY! GEQ 12 goto dlfail
timeout /t 10 /nobreak >nul
goto dl

:dlok
echo ---- download complete, installing ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install --no-cache-dir "%WHL%" >> %LOG% 2>&1
echo ---- torch rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install --timeout 60 --retries 10 torchvision --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
echo ---- torchvision rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install --timeout 60 --retries 10 rl4co==0.6.0 >> %LOG% 2>&1
echo ---- rl4co rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install --timeout 60 --retries 10 einops pyyaml tqdm "huggingface_hub[cli]" >> %LOG% 2>&1
echo ---- extras rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python %ROOT%\CC_Compare\tools\_probe_cuda.py >> %LOG% 2>&1
%CONDA% run --no-capture-output -n cc_compare python -c "import torch,rl4co,torchrl,tensordict;print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'| rl4co',rl4co.__version__,'| torchrl',torchrl.__version__,'| tensordict',tensordict.__version__)" >> %LOG% 2>&1
echo ---- verify rc=%ERRORLEVEL% ---- >> %LOG%
echo == SETUP4 DONE %DATE% %TIME% == >> %LOG%
echo SETUP4_DONE %DATE% %TIME% > C:\Users\zh\gpu3080_logs\cc_compare_setup4.done
goto :eof

:dlfail
echo DOWNLOAD_FAILED after !TRY! attempts >> %LOG%
echo == SETUP4 FAILED %DATE% %TIME% == >> %LOG%
echo SETUP4_FAILED > C:\Users\zh\gpu3080_logs\cc_compare_setup4.failed
