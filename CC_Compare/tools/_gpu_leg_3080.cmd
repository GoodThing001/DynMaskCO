@echo off
REM ============================================================================
REM 3080 GPU 腿（8 臂 = 4 方法 × 2 档），替代 1003 上被他人占 89% 的 GPU 工作。
REM 方法并行（每方法一个进程，各自顺序跑 p_c=0 → 主档），日志按方法分开。
REM 环境：mvmoe 用现有 pytorch(env, torch2.0.1)；routefinder/cada/rrnco 用 cc_gpu(torch2.4.1+rl4co)
REM 前置：基线 gate 必须已放到 results\a1_c1fresh_reveal_40\gate.json
REM ============================================================================
set PYTHONIOENCODING=utf-8
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
set BASE=results\a1_c1fresh_reveal_40\gate.json
set OUT=results\a1_ext_formal2_3080
set L=C:\Users\zh\gpu3080_logs
if not exist %L% mkdir %L%

call :run mvmoe          pytorch mvmoe         ..\CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt
call :run routefinder    cc_gpu  routefinder   ..\CC_Compare\RouteFinder\checkpoints\100\rf-transformer.ckpt
call :run cada           cc_gpu  cada          ..\CC_Compare\CaDA\100\result\2024-1121-1355\checkpoint-300.pt
call :run rrnco          cc_gpu  rrnco         ..\CC_Compare\RRNCO\checkpoints\rcvrptw\epoch_199.ckpt
echo ALLDONE_GPU3080 > %L%\ALLDONE
goto :eof

:run
REM %1=tag %2=env %3=provider %4=ckpt    （在后台起一个进程，内部顺序跑两档）
start "gpu3080_%1" /b cmd /c "cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem && ^
  ( echo == %3 p_c=0 == & conda run -n %2 python scripts\evaluation\run_a1_external_accept.py --solver ordering ^
      --provider %3 --provider-ckpt %4 --provider-device cpu --baseline-from %BASE% ^
      --dev-instances 40 --workers 4 --penalty \"p_c=0\" --out %OUT%\%1_0 ) > %L%\%1.log 2>&1 && ^
  ( echo == %3 main == & conda run -n %2 python scripts\evaluation\run_a1_external_accept.py --solver ordering ^
      --provider %3 --provider-ckpt %4 --provider-device cpu --baseline-from %BASE% ^
      --dev-instances 40 --workers 4 --penalty \"p_c=(5,10,15)\" --out %OUT%\%1_51015 ) >> %L%\%1.log 2>&1 && ^
  echo DONE_%1 >> %L%\%1.done"
goto :eof
