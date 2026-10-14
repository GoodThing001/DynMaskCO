@echo off
REM 3080：为 GPU 腿（RouteFinder/RRNCO/CaDA）建隔离环境 cc_gpu（py3.10 + torch cu121 + rl4co）
REM 不动 base / pytorch 两个共享环境。日志：C:\Users\zh\cc_gpu_setup.log
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\cc_gpu_setup.log
set CONDA=D:\anaconda\Scripts\conda.exe
echo == SETUP START %DATE% %TIME% == > %LOG%
%CONDA% create -n cc_gpu python=3.10 -y >> %LOG% 2>&1
echo ---- create rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_gpu python -m pip install --upgrade pip >> %LOG% 2>&1
%CONDA% run -n cc_gpu python -m pip install "numpy<2" >> %LOG% 2>&1
%CONDA% run -n cc_gpu python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 >> %LOG% 2>&1
echo ---- torch rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_gpu python -m pip install rl4co==0.6.0 >> %LOG% 2>&1
echo ---- rl4co rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_gpu python -c "import torch, rl4co, tensordict; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); import rl4co as r; print('rl4co', r.__version__)" >> %LOG% 2>&1
echo ---- verify rc=%ERRORLEVEL% ---- >> %LOG%
echo == SETUP DONE %DATE% %TIME% == >> %LOG%
