@echo off
REM Resume the cc_compare install with VISIBLE progress (-v) and explicit retries.
REM Rationale: the first attempt went silent for 10+ min with no cache/log/site-packages
REM change, so it was unobservable. -v makes each step land in the log; --timeout/--retries
REM make a flaky read self-heal instead of hanging.
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\gpu3080_logs\cc_compare_setup2.log
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
set CONDA=D:\anaconda\Scripts\conda.exe
set ROOT=C:\Users\zh\project\MASKCO-Main
set PIPARGS=--timeout 60 --retries 10 -v
echo == RESUME START %DATE% %TIME% == > %LOG%
%CONDA% run -n cc_compare python -m pip install %PIPARGS% torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
echo ---- torch rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install %PIPARGS% rl4co==0.6.0 >> %LOG% 2>&1
echo ---- rl4co rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python -m pip install %PIPARGS% einops pyyaml tqdm "huggingface_hub[cli]" >> %LOG% 2>&1
echo ---- extras rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run -n cc_compare python %ROOT%\CC_Compare\tools\_probe_cuda.py >> %LOG% 2>&1
%CONDA% run -n cc_compare python -c "import torch,rl4co,torchrl,tensordict;print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'| rl4co',rl4co.__version__,'| torchrl',torchrl.__version__,'| tensordict',tensordict.__version__)" >> %LOG% 2>&1
echo ---- verify rc=%ERRORLEVEL% ---- >> %LOG%
echo == RESUME DONE %DATE% %TIME% == >> %LOG%
echo RESUME_DONE %DATE% %TIME% > C:\Users\zh\gpu3080_logs\cc_compare_setup2.done
