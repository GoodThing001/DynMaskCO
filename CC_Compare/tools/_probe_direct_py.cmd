@echo off
REM Diagnose: does the cc_compare env's python exist and import the RRNCO stack?
set PY=D:\anaconda\envs\cc_compare\python.exe
if not exist "%PY%" (
  echo NO_DIRECT_PY %PY%
  exit /b 2
)
"%PY%" -c "import sys,torch;print('DIRECT_PY_OK',sys.version.split()[0],torch.__version__)"
echo RC=%ERRORLEVEL%
"%PY%" -c "import rl4co,tensordict,torchrl;print('RL4CO_OK')"
echo RC2=%ERRORLEVEL%
