@echo off
REM 3080: build the canonical cc_compare env, mirroring 1003's /home/hzeng/envs/cc_compare
REM (torch 2.11.0+cu128 + rl4co 0.6.0) and CC_Compare/setup_envs.sh.
REM
REM Root cause this fixes: the earlier _setup_cc_gpu_3080.cmd pinned torch==2.4.1+cu121,
REM but rl4co==0.6.0 requires a newer torch, so pip silently REPLACED it with PyPI's
REM CPU-only build (2.14.1+cpu) -> cc_gpu has no CUDA. Hence a fresh env is needed.
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\gpu3080_logs\cc_compare_setup.log
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
set CONDA=D:\anaconda\Scripts\conda.exe
set ROOT=C:\Users\zh\project\MASKCO-Main
echo == SETUP START %DATE% %TIME% == > %LOG%
%CONDA% create -n cc_compare python=3.11 -y >> %LOG% 2>&1
echo ---- create rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install --upgrade pip >> %LOG% 2>&1
echo ---- pip rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
echo ---- torch rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install rl4co==0.6.0 >> %LOG% 2>&1
echo ---- rl4co rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install einops pyyaml tqdm "huggingface_hub[cli]" >> %LOG% 2>&1
echo ---- extras rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python %ROOT%\CC_Compare\tools\_probe_cuda.py >> %LOG% 2>&1
%CONDA% run -n cc_compare python -c "import torch,rl4co,torchrl,tensordict;print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'| rl4co',rl4co.__version__,'| torchrl',torchrl.__version__,'| tensordict',tensordict.__version__)" >> %LOG% 2>&1
echo ---- verify rc=%ERRORLEVEL% ---- >> %LOG%
echo == SETUP DONE %DATE% %TIME% == >> %LOG%
echo SETUP_DONE_CC_COMPARE %DATE% %TIME% > C:\Users\zh\gpu3080_logs\cc_compare_setup.done
