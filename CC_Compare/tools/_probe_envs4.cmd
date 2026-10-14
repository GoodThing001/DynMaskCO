@echo off
REM Look for an existing 3080 env that already has CUDA torch (+ rl4co), and check disk.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===DISK===
wmic logicaldisk get caption,freespace,size
for %%E in (MASKCO_env MASKCO_py310 pt310 cc_gpu pytorch) do call :probe %%E
echo ===END===
goto :eof

:probe
echo ===ENV %1===
call conda run -n %1 python CC_Compare\tools\_probe_cuda.py 2>nul
call conda run -n %1 python -c "import rl4co;print('rl4co',rl4co.__version__)" 2>nul
goto :eof
