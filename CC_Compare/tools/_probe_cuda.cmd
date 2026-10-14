@echo off
REM Probe torch/CUDA in both 3080 conda envs + GPU memory, in one shot.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===CC_GPU===
call conda run -n cc_gpu python CC_Compare\tools\_probe_cuda.py
echo ===PYTORCH===
call conda run -n pytorch python CC_Compare\tools\_probe_cuda.py
echo ===SMI===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo ===END===
