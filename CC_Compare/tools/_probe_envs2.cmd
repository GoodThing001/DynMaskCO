@echo off
REM Full version/driver inventory of the 3080 box for the GPU leg decision.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===SMI===
nvidia-smi
echo ===ENVS===
call conda env list
echo ===CCGPU_TORCH===
call conda run -n cc_gpu python -c "import sys,torch;print(sys.version.split()[0]);print('torch',torch.__version__);print('cuda_build',torch.version.cuda)"
echo ===CCGPU_PIP===
call conda run -n cc_gpu python -m pip list 2>nul | findstr /I "torch rl4co lightning tensordict torchrl"
echo ===PYTORCH_PIP===
call conda run -n pytorch python -m pip list 2>nul | findstr /I "torch rl4co lightning tensordict torchrl"
echo ===END===
