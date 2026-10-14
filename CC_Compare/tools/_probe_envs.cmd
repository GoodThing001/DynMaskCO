@echo off
REM Probe which 3080 conda env can host the rl4co-based providers on CUDA.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===ENVS===
call conda env list
echo ===PYTORCH===
call conda run -n pytorch python CC_Compare\tools\_probe_cuda.py
call conda run -n pytorch python -c "import rl4co;print('rl4co',rl4co.__version__)"
echo ===CCGPU===
call conda run -n cc_gpu python CC_Compare\tools\_probe_cuda.py
call conda run -n cc_gpu python -c "import rl4co;print('rl4co',rl4co.__version__)"
echo ===END===
