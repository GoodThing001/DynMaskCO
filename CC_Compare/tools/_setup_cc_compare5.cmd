@echo off
REM ============================================================================
REM cc_compare install, take 5: curl (not pip) from download.pytorch.org.
REM
REM Measurements that drove this:
REM   - download.pytorch.org range request (60 MB):  5.46 MB/s  -> ~9 min for 2.75 GB
REM   - mirrors.aliyun.com same wheel:              ~0.29 MB/s -> ~2.5 h   (rejected)
REM   - pip's own download:                         stalled at 0 bytes
REM So: fetch with resumable curl (-C -) in a retry loop, verify the exact byte
REM count, then install from the local file.
REM ============================================================================
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\gpu3080_logs\cc_compare_setup5.log
set CONDA=D:\anaconda\Scripts\conda.exe
set ROOT=C:\Users\zh\project\MASKCO-Main
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
set WHEELDIR=D:\anaconda\envs\cc_compare\_wheels
if not exist "%WHEELDIR%" mkdir "%WHEELDIR%"
set WHL=%WHEELDIR%\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl
set TURL=https://download.pytorch.org/whl/cu128/torch-2.11.0%%2Bcu128-cp311-cp311-win_amd64.whl
set WANT=2753148611
echo == SETUP5 START %DATE% %TIME% == > %LOG%

REM Start clean: the existing partial came from the (slow) Aliyun mirror.
del /q "%WHL%" >nul 2>&1

set /a TRY=0
:dl
set /a TRY+=1
echo ---- curl attempt !TRY! %TIME% ---- >> %LOG%
curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors --speed-time 60 --speed-limit 300000 -o "%WHL%" "%TURL%" >> %LOG% 2>&1
set GOT=0
for %%A in ("%WHL%") do set GOT=%%~zA
echo ---- attempt !TRY! got=!GOT! want=%WANT% ---- >> %LOG%
if "!GOT!"=="%WANT%" goto dlok
if !TRY! GEQ 15 goto dlfail
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
echo == SETUP5 DONE %DATE% %TIME% == >> %LOG%
echo SETUP5_DONE %DATE% %TIME% > C:\Users\zh\gpu3080_logs\cc_compare_setup5.done
goto :eof

:dlfail
echo DOWNLOAD_FAILED after !TRY! attempts >> %LOG%
echo == SETUP5 FAILED %DATE% %TIME% == >> %LOG%
echo SETUP5_FAILED > C:\Users\zh\gpu3080_logs\cc_compare_setup5.failed
