@echo off
REM Definitive cc_compare install: `conda run --no-capture-output` so the log streams in REAL TIME.
REM (Root cause of the earlier "silence": conda run captures the child's stdout/stderr and only
REM  re-emits it at exit, so a 5-10 min download looked identical to a hang.)
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\zh\gpu3080_logs\cc_compare_setup3.log
if not exist C:\Users\zh\gpu3080_logs mkdir C:\Users\zh\gpu3080_logs
set CONDA=D:\anaconda\Scripts\conda.exe
set ROOT=C:\Users\zh\project\MASKCO-Main
echo == SETUP3 START %DATE% %TIME% == > %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install -v --timeout 60 --retries 10 torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
echo ---- torch rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install -v --timeout 60 --retries 10 rl4co==0.6.0 >> %LOG% 2>&1
echo ---- rl4co rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python -m pip install --timeout 60 --retries 10 einops pyyaml tqdm "huggingface_hub[cli]" >> %LOG% 2>&1
echo ---- extras rc=%ERRORLEVEL% ---- >> %LOG%
%CONDA% run --no-capture-output -n cc_compare python %ROOT%\CC_Compare\tools\_probe_cuda.py >> %LOG% 2>&1
%CONDA% run --no-capture-output -n cc_compare python -c "import torch,rl4co,torchrl,tensordict;print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'| rl4co',rl4co.__version__,'| torchrl',torchrl.__version__,'| tensordict',tensordict.__version__)" >> %LOG% 2>&1
echo ---- verify rc=%ERRORLEVEL% ---- >> %LOG%
echo == SETUP3 DONE %DATE% %TIME% == >> %LOG%
echo SETUP3_DONE %DATE% %TIME% > C:\Users\zh\gpu3080_logs\cc_compare_setup3.done
