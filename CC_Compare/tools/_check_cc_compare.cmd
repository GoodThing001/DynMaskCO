@echo off
REM Functional readiness check for the 3080 cc_compare env (used as an orchestrator gate).
REM Prints ENVOK only if torch+CUDA+rl4co+torchrl+tensordict all import in cc_compare.
D:\anaconda\Scripts\conda.exe run -n cc_compare python -c "import torch,rl4co,torchrl,tensordict;print('ENVOK' if torch.cuda.is_available() else 'ENVNOTREADY')" 2>nul
